"""KDViT — KD-tree Vision Transformer for airfoil CFD surrogate (E3-D2).

Round 3 refactor — ablation-friendly, batched, DDP/compile-ready.

Key invariants:
  * ``forward`` now takes explicit tensor arguments (no PyG ``Data`` access),
    so ``torch.compile`` does not insert graph breaks on attribute lookup.
  * All activations carry a leading batch dim ``B``. ``B=1`` is just
    ``[1, ...]`` — there is no separate unbatched code path.
  * RoPE tables are precomputed per-case OUTSIDE the model (Part 2) and
    passed in. The pooled-KV branch is dead (``pool_kv_factor`` forced to 0),
    so ``k_rope == q_rope`` everywhere.

Caller responsibility: stack per-case tensors into ``[B, ...]`` before
calling the model. The model never inserts a synthetic batch dim.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class PointNetEncoder(nn.Module):
    """Masked max-pooled PointNet.

    Input shape ``[B, L, k, feat]``. We pool over the neighbor axis ``k``
    (= ``dim=-2``), not over leaves and not over the batch.
    """

    def __init__(self, input_dim=7, hidden_dim=32, out_dim=128, num_layers=2):
        super().__init__()
        assert num_layers >= 1
        layers = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, pn_input, valid_mask):
        # pn_input: [B, L, k, F]; valid_mask: [B, L, k]
        safe_input = torch.where(
            valid_mask[..., None], pn_input, torch.zeros_like(pn_input))
        feats = self.mlp(safe_input)                       # [B, L, k, out]
        neg_inf = torch.full_like(feats, float('-inf'))
        masked = torch.where(valid_mask[..., None], feats, neg_inf)
        pooled = masked.max(dim=-2).values                 # [B, L, out]
        no_valid = ~valid_mask.any(dim=-1)                 # [B, L]
        pooled = torch.where(
            no_valid[..., None], torch.zeros_like(pooled), pooled)
        return pooled


# ---------------------------------------------------------------------------
# Patchify (only used when no_patchify=False; gated to B=1 in forward)
# ---------------------------------------------------------------------------

class TreePatchify(nn.Module):
    def __init__(self, patch_groups, latent_dim=256):
        super().__init__()
        self.register_buffer('patch_groups', patch_groups)
        children = patch_groups.shape[1]
        self.proj = nn.Linear(children * latent_dim, latent_dim)

    def forward(self, x):
        # x: [L, D] (only invoked on unbatched [L, D]; B=1 paths squeeze first)
        gathered = x[self.patch_groups]
        flat = gathered.reshape(gathered.shape[0], -1)
        return self.proj(flat)


class TreeUnpatchify(nn.Module):
    def __init__(self, patch_groups, n_leaves, latent_dim=256):
        super().__init__()
        self.register_buffer('patch_groups', patch_groups)
        children = patch_groups.shape[1]
        self.proj = nn.Linear(latent_dim, children * latent_dim)
        self.n_leaves = n_leaves
        self.latent_dim = latent_dim
        self.num_children = children

    def forward(self, x):
        expanded = self.proj(x)
        expanded = expanded.reshape(-1, self.num_children, self.latent_dim)
        out = torch.zeros(self.n_leaves, self.latent_dim,
                          device=x.device, dtype=x.dtype)
        out[self.patch_groups.flatten()] = expanded.reshape(
            -1, self.latent_dim)
        return out


# ---------------------------------------------------------------------------
# Transformer pieces
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = (x.float().pow(2).mean(dim=-1, keepdim=True)
                .add(self.eps).rsqrt())
        return (x.float() * norm).type_as(x) * self.weight


def build_rope_table(head_dim, token_coords, domain_x, domain_y,
                     device, dtype, rope_scale=32.0, rope_base=100.0,
                     num_registers=0):
    """Build (cos_x, sin_x, cos_y, sin_y) for one case.

    Each output: ``[T + num_registers, pair_d]`` where ``pair_d = head_dim/4``.
    Register rows are identity (cos=1, sin=0). Used by Part 2's per-case
    precompute.
    """
    half_dim = head_dim // 2
    inv_freqs = 1.0 / (
        rope_base ** (torch.arange(0, half_dim, 2,
                                    dtype=dtype, device=device) / half_dim))
    x = token_coords[:, 0].to(dtype=dtype, device=device)
    y = token_coords[:, 1].to(dtype=dtype, device=device)
    dx_lo = torch.as_tensor(domain_x[0], dtype=dtype, device=device)
    dx_hi = torch.as_tensor(domain_x[1], dtype=dtype, device=device)
    dy_lo = torch.as_tensor(domain_y[0], dtype=dtype, device=device)
    dy_hi = torch.as_tensor(domain_y[1], dtype=dtype, device=device)
    x_scaled = (x - dx_lo) / (dx_hi - dx_lo + 1e-8) * rope_scale
    y_scaled = (y - dy_lo) / (dy_hi - dy_lo + 1e-8) * rope_scale
    angles_x = x_scaled[:, None] * inv_freqs[None, :]
    angles_y = y_scaled[:, None] * inv_freqs[None, :]
    cos_x = torch.cos(angles_x); sin_x = torch.sin(angles_x)
    cos_y = torch.cos(angles_y); sin_y = torch.sin(angles_y)
    if num_registers > 0:
        pair_d = cos_x.shape[1]
        ones = torch.ones(num_registers, pair_d, dtype=dtype, device=device)
        zeros = torch.zeros(num_registers, pair_d, dtype=dtype, device=device)
        cos_x = torch.cat([cos_x, ones], dim=0)
        sin_x = torch.cat([sin_x, zeros], dim=0)
        cos_y = torch.cat([cos_y, ones], dim=0)
        sin_y = torch.cat([sin_y, zeros], dim=0)
    return cos_x, sin_x, cos_y, sin_y


def apply_rope_2d(x, cos_x, sin_x, cos_y, sin_y):
    """Apply 2-D RoPE to ``x: [B, H, T, head_dim]``.

    Tables ``cos_x`` etc. have shape ``[B, T, pair_d]`` (batched) and are
    reshaped to ``[B, 1, T, pair_d]`` — broadcast over heads only, **not**
    over the batch.
    """
    B, H, T, D = x.shape
    half_d = D // 2
    pair_d = half_d // 2
    x_xhalf = x[..., :half_d]
    x_yhalf = x[..., half_d:]
    x_x_pairs = x_xhalf.reshape(B, H, T, pair_d, 2)
    x_y_pairs = x_yhalf.reshape(B, H, T, pair_d, 2)
    cos_xb = cos_x.reshape(B, 1, T, pair_d)
    sin_xb = sin_x.reshape(B, 1, T, pair_d)
    cos_yb = cos_y.reshape(B, 1, T, pair_d)
    sin_yb = sin_y.reshape(B, 1, T, pair_d)
    x0, x1 = x_x_pairs[..., 0], x_x_pairs[..., 1]
    rot_x = torch.stack([x0 * cos_xb - x1 * sin_xb,
                         x0 * sin_xb + x1 * cos_xb],
                        dim=-1).reshape(B, H, T, half_d)
    y0, y1 = x_y_pairs[..., 0], x_y_pairs[..., 1]
    rot_y = torch.stack([y0 * cos_yb - y1 * sin_yb,
                         y0 * sin_yb + y1 * cos_yb],
                        dim=-1).reshape(B, H, T, half_d)
    return torch.cat([rot_x, rot_y], dim=-1)


class MultiHeadAttention(nn.Module):
    """MHA with QK-RMSNorm + RoPE. Pooled-KV is removed; GQA stays dormant."""

    def __init__(self, dim, num_heads=8, attn_dropout=0.0, kv_heads=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 4 == 0, \
            f'head_dim ({self.head_dim}) must be divisible by 4 for 2D RoPE'
        if kv_heads is None or kv_heads <= 0 or kv_heads == num_heads:
            self.kv_heads = num_heads
        else:
            assert num_heads % kv_heads == 0
            self.kv_heads = int(kv_heads)
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * self.kv_heads * self.head_dim,
                                 bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = attn_dropout
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope):
        # x: [B, T, C]; rope = (cos_x, sin_x, cos_y, sin_y) each [B, T, pair_d]
        B, T, C = x.shape
        q = self.q_proj(x).reshape(
            B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(x).reshape(
            B, T, 2, self.kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope_2d(q, *rope)
        k = apply_rope_2d(k, *rope)
        if self.kv_heads != self.num_heads:
            r = self.num_heads // self.kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.out(F.silu(self.gate(x)) * self.value(x)))


class TransformerBlock(nn.Module):
    """Pre-norm attn + FFN with adaLN-Zero modulation from ``uinf``.

    ``uinf`` is ``[B, uinf_dim]``; mod outputs are ``[B, dim]`` and are
    unsqueezed to ``[B, 1, dim]`` so they broadcast over tokens.
    """

    def __init__(self, dim=256, num_heads=8, ffn_hidden=1024,
                 dropout=0.0, drop_path_rate=0.0, attn_dropout=0.0,
                 uinf_dim=2, kv_heads=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads,
                                       attn_dropout=attn_dropout,
                                       kv_heads=kv_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden, dropout=dropout)
        self.drop_path_rate = drop_path_rate
        self.modulation = nn.Linear(uinf_dim, 6 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, rope, uinf):
        if (self.training and self.drop_path_rate > 0
                and torch.rand(1).item() < self.drop_path_rate):
            return x
        mod = self.modulation(uinf)                           # [B, 6*dim]
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
        s1 = shift1.unsqueeze(1); c1 = scale1.unsqueeze(1)
        g1 = gate1.unsqueeze(1)
        s2 = shift2.unsqueeze(1); c2 = scale2.unsqueeze(1)
        g2 = gate2.unsqueeze(1)
        h = self.norm1(x) * (1.0 + c1) + s1
        x = x + g1 * self.attn(h, rope)
        h = self.norm2(x) * (1.0 + c2) + s2
        x = x + g2 * self.ffn(h)
        return x


class ViTProcessor(nn.Module):
    """Stack of TransformerBlocks with optional U-Net skip pairs.

    RoPE tables are supplied from outside (precomputed per-case).
    """

    def __init__(self, dim=256, num_layers=5, num_heads=8, ffn_hidden=1024,
                 dropout=0.1, drop_path_rate=0.0, attn_dropout=0.0,
                 layerwise_scaling=False, use_unet_skip=True, kv_heads=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_unet_skip = bool(use_unet_skip)

        dp_rates = [drop_path_rate * i / max(num_layers - 1, 1)
                    for i in range(num_layers)]
        if layerwise_scaling and num_layers > 1:
            ffn_dropouts = [dropout * (0.5 + i / (num_layers - 1))
                            for i in range(num_layers)]
        else:
            ffn_dropouts = [dropout] * num_layers

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim, num_heads=num_heads, ffn_hidden=ffn_hidden,
                dropout=ffn_dropouts[i], drop_path_rate=dp_rates[i],
                attn_dropout=attn_dropout, kv_heads=kv_heads,
            )
            for i in range(num_layers)
        ])
        if self.use_unet_skip:
            self.skip_pairs = [(i, num_layers - 1 - i)
                               for i in range(num_layers // 2)]
            self.skip_projections = nn.ModuleDict({
                str(dst): nn.Linear(2 * dim, dim)
                for _, dst in self.skip_pairs
            })
        else:
            self.skip_pairs = []
            self.skip_projections = nn.ModuleDict()

    def forward(self, x, rope, uinf):
        skip_dict = {dst: src for src, dst in self.skip_pairs}
        cache = {}
        for i, layer in enumerate(self.layers):
            if i in skip_dict:
                src = skip_dict[i]
                x = self.skip_projections[str(i)](
                    torch.cat([x, cache[src]], dim=-1))
            x = layer(x, rope, uinf)
            cache[i] = x
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        freqs = (2.0 ** torch.arange(num_freqs,
                                      dtype=torch.float32)) * math.pi
        self.register_buffer('freqs', freqs)
        self.num_freqs = num_freqs

    @property
    def out_dim_factor(self):
        return 1 + 2 * self.num_freqs

    def forward(self, x):
        scaled = x[..., None] * self.freqs
        sin_f = torch.sin(scaled)
        cos_f = torch.cos(scaled)
        encoded = torch.cat([sin_f, cos_f], dim=-1).reshape(
            *x.shape[:-1], -1)
        return torch.cat([x, encoded], dim=-1)


class Decoder(nn.Module):
    def __init__(self, fourier_freqs=8, pos_hidden=256, pos_out=512,
                 pred_head_in_dim=512, pred_hidden=256, out_dim=4,
                 decoder_dropout=0.0):
        super().__init__()
        self.fourier = FourierFeatures(num_freqs=fourier_freqs)
        fourier_input_dim = 2 + 1 + 2
        fourier_dim = fourier_input_dim * self.fourier.out_dim_factor
        raw_feat_dim = 2
        combined_dim = fourier_dim + raw_feat_dim

        self.pos_mlp = nn.Sequential(
            nn.Linear(combined_dim, pos_hidden),
            nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(pos_hidden, pos_out),
        )
        self.pos_out = pos_out
        self.pred_head = nn.Sequential(
            nn.Linear(pred_head_in_dim, pred_hidden),
            nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(pred_hidden, out_dim),
        )

    def compute_pos_features(self, norm_pos, uinf, sdf, sdf_grad):
        fourier_in = torch.cat([norm_pos, sdf, sdf_grad], dim=-1)
        fourier_feats = self.fourier(fourier_in)
        return self.pos_mlp(torch.cat([fourier_feats, uinf], dim=-1))

    def predict(self, context):
        return self.pred_head(context)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class KDViT(nn.Module):
    """Batched KDViT. Forward takes explicit tensor args (no PyG Data)."""

    def __init__(
        self,
        n_leaves,
        patch_size=4,
        latent_dim=256,
        num_layers=5,
        num_heads=8,
        ffn_hidden=1024,
        fourier_freqs=8,
        pos_hidden=256,
        pos_out=512,
        pred_hidden=256,
        out_dim=4,
        pn_hidden=32,
        pn_dim=128,
        pn_layers=2,
        dropout=0.1,
        decoder_dropout=0.0,
        drop_path_rate=0.0,
        attn_dropout=0.0,
        layerwise_scaling=False,
        rope_scale=32.0,
        rope_base=100.0,
        use_did=True,
        did_bins=8,
        no_patchify=True,
        no_unet_skip=False,
        domain_x=(-2.0, 4.0),
        domain_y=(-1.5, 1.5),
        register_tokens=4,
        pool_kv_factor=0,
        gqa_kv_heads=0,
    ):
        super().__init__()

        if int(pool_kv_factor) != 0:
            raise ValueError(
                "pool_kv_factor has been removed from the active path "
                "(must be 0).")

        L = int(n_leaves)
        self.no_patchify = bool(no_patchify)
        self.num_registers = int(register_tokens)
        self.gqa_kv_heads = int(gqa_kv_heads)

        if not self.no_patchify:
            assert patch_size >= 1 and (patch_size & (patch_size - 1)) == 0, \
                f'patch_size must be a power of 2, got {patch_size}'
            assert L % patch_size == 0, \
                f'patch_size {patch_size} must divide n_leaves {L}'
            T = L // patch_size
            patch_groups = torch.arange(L).reshape(T, patch_size)
        else:
            T = L
            patch_groups = torch.arange(L).reshape(T, 1)

        self.n_leaves = L
        self.patch_size = patch_size
        self.num_tokens = T
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.use_did = bool(use_did)
        self.did_bins = int(did_bins)
        # Kept for caller (precomputing RoPE) — not used inside forward.
        self.rope_scale = float(rope_scale)
        self.rope_base = float(rope_base)
        self.domain_x = tuple(domain_x)
        self.domain_y = tuple(domain_y)

        self.__name__ = (f'KDViT_L{L}_P{patch_size}'
                         f'_D{latent_dim}_N{num_layers}')

        self.register_buffer('patch_groups', patch_groups.long())

        # === Encoder ===
        self.pointnet = PointNetEncoder(
            input_dim=7, hidden_dim=pn_hidden, out_dim=pn_dim,
            num_layers=pn_layers)
        encoder_in_dim = pn_dim + 19 + 1 + 2 + 2
        if self.use_did:
            encoder_in_dim += self.did_bins
        self.encoder_proj = nn.Linear(encoder_in_dim, latent_dim)
        self.encoder_uinf_film = nn.Linear(2, 2 * latent_dim)
        nn.init.zeros_(self.encoder_uinf_film.weight)
        nn.init.zeros_(self.encoder_uinf_film.bias)

        if not self.no_patchify:
            self.patchify = TreePatchify(patch_groups, latent_dim=latent_dim)
            self.unpatchify = TreeUnpatchify(
                patch_groups, L, latent_dim=latent_dim)

        self.processor = ViTProcessor(
            dim=latent_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_hidden=ffn_hidden, dropout=dropout,
            drop_path_rate=drop_path_rate, attn_dropout=attn_dropout,
            layerwise_scaling=layerwise_scaling,
            use_unet_skip=not bool(no_unet_skip),
            kv_heads=(self.gqa_kv_heads if self.gqa_kv_heads > 0 else None))

        if self.num_registers > 0:
            self.registers = nn.Parameter(
                torch.randn(self.num_registers, latent_dim) * 0.02)

        self.film_enc = nn.Linear(latent_dim, 2 * pos_out)
        self.film_vit = nn.Linear(latent_dim, 2 * pos_out)
        nn.init.zeros_(self.film_enc.weight)
        nn.init.zeros_(self.film_enc.bias)
        nn.init.zeros_(self.film_vit.weight)
        nn.init.zeros_(self.film_vit.bias)

        self.decoder = Decoder(
            fourier_freqs=fourier_freqs,
            pos_hidden=pos_hidden, pos_out=pos_out,
            pred_head_in_dim=pos_out,
            pred_hidden=pred_hidden, out_dim=out_dim,
            decoder_dropout=decoder_dropout)

    def forward(
        self,
        norm_pos,          # [B, N, 2]
        sdf,               # [B, N]
        sdf_grad,          # [B, N, 2]
        idw_indices,       # [B, N, 4]
        idw_weights,       # [B, N, 4]
        uinf,              # [B, 2]
        pn_input,          # [B, L, k, 7]
        pn_mask,           # [B, L, k]
        leaf_stats,        # [B, L, 19]
        leaf_sdf,          # [B, L]
        leaf_sdf_grad,     # [B, L, 2]
        leaf_norm_pos,     # [B, L, 2]
        rope_cos_x,        # [B, T+R, pair_d]
        rope_sin_x,
        rope_cos_y,
        rope_sin_y,
        leaf_did=None,     # [B, L, did_bins] (required if use_did)
    ):
        B = norm_pos.shape[0]
        N = norm_pos.shape[1]
        L = leaf_stats.shape[1]
        assert L == self.n_leaves, \
            f'leaf_stats L={L} != model n_leaves={self.n_leaves}'

        # === Encoder ===
        pn_feats = self.pointnet(pn_input, pn_mask)            # [B, L, pn_dim]
        enc_parts = [
            pn_feats,
            leaf_stats,
            leaf_sdf.unsqueeze(-1),                            # [B, L, 1]
            leaf_sdf_grad,
            leaf_norm_pos,
        ]
        if self.use_did:
            assert leaf_did is not None, 'use_did=True requires leaf_did'
            enc_parts.append(leaf_did)
        encoder_in = torch.cat(enc_parts, dim=-1)
        encoder_features = self.encoder_proj(encoder_in)       # [B, L, D]

        # Encoder uinf FiLM: broadcast over leaves via [B, 1, D].
        e_gamma, e_beta = self.encoder_uinf_film(uinf).chunk(2, dim=-1)
        encoder_features = (encoder_features
                            * (1.0 + e_gamma.unsqueeze(1))
                            + e_beta.unsqueeze(1))

        # === Patchify ===
        if self.no_patchify:
            tokens = encoder_features                          # [B, L, D]
        else:
            if B != 1:
                raise NotImplementedError(
                    "patchify path (no_patchify=False) only supports B=1; "
                    "use no_patchify=True for batching.")
            tokens = self.patchify(encoder_features.squeeze(0)).unsqueeze(0)

        # === Append register tokens ===
        R = self.num_registers
        if R > 0:
            regs = self.registers.unsqueeze(0).expand(B, -1, -1)
            tokens = torch.cat([tokens, regs], dim=1)          # [B, T+R, D]

        # === Processor (RoPE table includes registers already) ===
        rope = (rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)
        processed = self.processor(tokens, rope, uinf)

        # Strip register tokens before unpatchify / IDW.
        if R > 0:
            processed = processed[:, :self.num_tokens, :]

        if self.no_patchify:
            vit_features = processed                           # [B, L, D]
        else:
            vit_features = self.unpatchify(
                processed.squeeze(0)).unsqueeze(0)

        # === Batched IDW gather ===
        # idw_indices: [B, N, 4]; features: [B, L, D] -> [B, N, 4, D]
        D = vit_features.shape[-1]
        batch_idx = torch.arange(B, device=vit_features.device).view(
            B, 1, 1).expand(-1, N, idw_indices.shape[-1])
        vit_interp = vit_features[batch_idx, idw_indices]      # [B, N, 4, D]
        enc_interp = encoder_features[batch_idx, idw_indices]  # [B, N, 4, D]
        w = idw_weights.unsqueeze(-1)                          # [B, N, 4, 1]
        vit_interp = (vit_interp * w).sum(dim=-2)              # [B, N, D]
        enc_interp = (enc_interp * w).sum(dim=-2)

        # === Decoder ===
        uinf_query = uinf[:, None, :].expand(B, N, uinf.shape[-1])
        pos_feat = self.decoder.compute_pos_features(
            norm_pos, uinf_query, sdf.unsqueeze(-1), sdf_grad)

        g1, b1 = self.film_enc(enc_interp).chunk(2, dim=-1)
        pos_feat = pos_feat * (1.0 + g1) + b1
        g2, b2 = self.film_vit(vit_interp).chunk(2, dim=-1)
        pos_feat = pos_feat * (1.0 + g2) + b2

        return self.decoder.predict(pos_feat)                  # [B, N, 4]


# ---------------------------------------------------------------------------
# Per-case RoPE precompute helper (used by train.py at dataset load time)
# ---------------------------------------------------------------------------

def precompute_case_rope(leaf_centroids, head_dim, domain_x, domain_y,
                          rope_scale, rope_base, num_registers,
                          patch_groups=None):
    """Compute the per-case RoPE table.

    Args:
        leaf_centroids: ``[L, 2]`` cpu/gpu tensor.
        patch_groups: optional ``[T, P]`` long tensor. If given, token
            coords are the per-group mean of leaf centroids (used only
            when ``no_patchify=False``); otherwise tokens == leaves.

    Returns: tuple ``(cos_x, sin_x, cos_y, sin_y)`` each
        ``[T + num_registers, pair_d]``.
    """
    if patch_groups is None or patch_groups.shape[1] == 1:
        token_coords = leaf_centroids
    else:
        token_coords = leaf_centroids[patch_groups].mean(dim=1)
    return build_rope_table(
        head_dim, token_coords, domain_x, domain_y,
        device=leaf_centroids.device, dtype=leaf_centroids.dtype,
        rope_scale=rope_scale, rope_base=rope_base,
        num_registers=num_registers)
