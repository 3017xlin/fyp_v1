"""Batching helpers: per-case RoPE attachment + Data -> tensor packing.

Keeps the rest of the codebase ignorant of the model's exact forward
signature. The model takes explicit tensor args (so torch.compile is
happy), and these helpers translate from PyG ``Data`` (the dataset
format) to those args.
"""

import torch

from models.model import precompute_case_rope


_ROPE_KEYS = ('rope_cos_x', 'rope_sin_x', 'rope_cos_y', 'rope_sin_y')


def attach_rope(dataset, model, device=None):
    """Precompute & attach per-case RoPE tables to every Data in ``dataset``.

    Tables are kept on the same device as the corresponding ``leaf_centroids``
    (so when the dataset later moves to GPU, RoPE moves with it implicitly
    via the per-field device check in ``_move_dataset_to_device``).

    Idempotent: a second call is a no-op if rope is already attached.
    """
    raw = _unwrap_to_raw(model)
    head_dim = raw.head_dim
    R = raw.num_registers
    patch_groups = (None if raw.no_patchify
                    else raw.patch_groups.cpu())
    domain_x = raw.domain_x
    domain_y = raw.domain_y
    rope_scale = raw.rope_scale
    rope_base = raw.rope_base
    for data in dataset:
        if hasattr(data, 'rope_cos_x'):
            continue
        lc = data.leaf_centroids
        if device is not None and lc.device != torch.device(device):
            lc_dev = lc.to(device)
        else:
            lc_dev = lc
        pg = (patch_groups.to(lc_dev.device)
              if patch_groups is not None else None)
        cos_x, sin_x, cos_y, sin_y = precompute_case_rope(
            lc_dev, head_dim, domain_x, domain_y,
            rope_scale, rope_base, R, patch_groups=pg)
        # Store on the same device as leaf_centroids (CPU during pin phase,
        # GPU after _move_dataset_to_device).
        target_dev = lc.device
        data.rope_cos_x = cos_x.to(target_dev)
        data.rope_sin_x = sin_x.to(target_dev)
        data.rope_cos_y = cos_y.to(target_dev)
        data.rope_sin_y = sin_y.to(target_dev)


def _unwrap_to_raw(model):
    """Return the underlying KDViT through DDP and torch.compile wrappers."""
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def case_to_model_inputs(data, use_did):
    """Single-case Data -> kwargs dict with a leading B=1 dim.

    Used by eval / curve / FLOPs counting paths. Returns also ``y`` and
    ``surf`` for the caller's loss computation (kept separate from model
    kwargs).
    """
    def _u(t):
        return t.unsqueeze(0)
    inputs = dict(
        norm_pos=_u(data.norm_pos),
        sdf=_u(data.sdf),
        sdf_grad=_u(data.sdf_grad),
        idw_indices=_u(data.idw_indices),
        idw_weights=_u(data.idw_weights),
        uinf=_u(data.uinf),
        pn_input=_u(data.pn_input),
        pn_mask=_u(data.pn_mask),
        leaf_stats=_u(data.leaf_stats),
        leaf_sdf=_u(data.leaf_sdf),
        leaf_sdf_grad=_u(data.leaf_sdf_grad),
        leaf_norm_pos=_u(data.leaf_norm_pos),
        rope_cos_x=_u(data.rope_cos_x),
        rope_sin_x=_u(data.rope_sin_x),
        rope_cos_y=_u(data.rope_cos_y),
        rope_sin_y=_u(data.rope_sin_y),
    )
    if use_did:
        inputs['leaf_did'] = _u(data.leaf_did)
    return inputs


def make_train_batch(cases, subsampling, device, use_did, generator=None):
    """Stack ``cases`` into a training batch with per-case subsampling.

    Each case is independently subsampled to ``subsampling`` query points;
    the resulting per-point tensors are stacked into shape ``[B, S, ...]``,
    and the per-leaf tensors into ``[B, L, ...]``.

    Returns ``(inputs_kwargs, targets [B, S, 4], surf [B, S])``.
    """
    norm_pos_l, sdf_l, sdf_grad_l, idw_idx_l, idw_w_l = [], [], [], [], []
    y_l, surf_l = [], []
    uinf_l = []
    pn_in_l, pn_mask_l = [], []
    leaf_stats_l, leaf_sdf_l, leaf_sg_l, leaf_np_l, leaf_did_l = \
        [], [], [], [], []
    rcx, rsx, rcy, rsy = [], [], [], []
    for data in cases:
        N = data.norm_pos.shape[0]
        idx = torch.randperm(N, device=device, generator=generator)[
            :subsampling]
        norm_pos_l.append(data.norm_pos[idx])
        sdf_l.append(data.sdf[idx])
        sdf_grad_l.append(data.sdf_grad[idx])
        idw_idx_l.append(data.idw_indices[idx])
        idw_w_l.append(data.idw_weights[idx])
        y_l.append(data.y[idx])
        surf_l.append(data.surf[idx])
        uinf_l.append(data.uinf)
        pn_in_l.append(data.pn_input)
        pn_mask_l.append(data.pn_mask)
        leaf_stats_l.append(data.leaf_stats)
        leaf_sdf_l.append(data.leaf_sdf)
        leaf_sg_l.append(data.leaf_sdf_grad)
        leaf_np_l.append(data.leaf_norm_pos)
        if use_did:
            leaf_did_l.append(data.leaf_did)
        rcx.append(data.rope_cos_x)
        rsx.append(data.rope_sin_x)
        rcy.append(data.rope_cos_y)
        rsy.append(data.rope_sin_y)

    inputs = dict(
        norm_pos=torch.stack(norm_pos_l, dim=0),
        sdf=torch.stack(sdf_l, dim=0),
        sdf_grad=torch.stack(sdf_grad_l, dim=0),
        idw_indices=torch.stack(idw_idx_l, dim=0),
        idw_weights=torch.stack(idw_w_l, dim=0),
        uinf=torch.stack(uinf_l, dim=0),
        pn_input=torch.stack(pn_in_l, dim=0),
        pn_mask=torch.stack(pn_mask_l, dim=0),
        leaf_stats=torch.stack(leaf_stats_l, dim=0),
        leaf_sdf=torch.stack(leaf_sdf_l, dim=0),
        leaf_sdf_grad=torch.stack(leaf_sg_l, dim=0),
        leaf_norm_pos=torch.stack(leaf_np_l, dim=0),
        rope_cos_x=torch.stack(rcx, dim=0),
        rope_sin_x=torch.stack(rsx, dim=0),
        rope_cos_y=torch.stack(rcy, dim=0),
        rope_sin_y=torch.stack(rsy, dim=0),
    )
    if use_did:
        inputs['leaf_did'] = torch.stack(leaf_did_l, dim=0)
    targets = torch.stack(y_l, dim=0)
    surf = torch.stack(surf_l, dim=0)
    return inputs, targets, surf
