"""Equivalence tests (Part 8) — float32, AMP off.

Run as:
    pytest tests/test_batching.py -v
or:
    python -m tests.test_batching

(A) Golden fidelity (regression check):
    The first time this test runs it writes ``tests/golden_b1.pt`` from
    the current eager B=1 forward+backward on a fixed seed. Subsequent
    runs assert allclose against that file. **Before merging a model
    refactor that's supposed to be behavior-preserving, delete the
    golden, run the OLD code's test once to regenerate, then re-run on
    the NEW code.** That sandwich catches drift that (B) cannot.

(B) Batching self-consistency:
    A single B=2 forward must equal stacking two B=1 forwards. Same for
    parameter grads (averaging two B=1 backward passes vs one B=2).
"""

import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from models.model import KDViT, precompute_case_rope

GOLDEN_PATH = osp.join(osp.dirname(__file__), 'golden_b1.pt')

# Small synthetic config for speed.
L = 256
PN_K = 16
N_QUERY = 2000
DID_BINS = 8
DOMAIN_X = (-2.0, 4.0)
DOMAIN_Y = (-1.5, 1.5)

MODEL_KWARGS = dict(
    n_leaves=L,
    patch_size=1,
    latent_dim=128,
    num_layers=3,
    num_heads=4,
    ffn_hidden=256,
    fourier_freqs=4,
    pn_hidden=16, pn_dim=32, pn_layers=2,
    dropout=0.0, decoder_dropout=0.0, drop_path_rate=0.0, attn_dropout=0.0,
    use_did=True, did_bins=DID_BINS,
    no_patchify=True, no_unet_skip=False,
    domain_x=DOMAIN_X, domain_y=DOMAIN_Y,
    register_tokens=2, pool_kv_factor=0,
)


def _make_case(seed):
    g = torch.Generator().manual_seed(seed)
    leaf_centroids = torch.empty(L, 2).uniform_(-1.5, 3.5, generator=g)
    leaf_norm_pos = torch.empty(L, 2).uniform_(-1, 1, generator=g)
    pn_input = torch.randn(L, PN_K, 7, generator=g)
    pn_mask = torch.ones(L, PN_K, dtype=torch.bool)
    pn_mask[:, PN_K // 2:] = (
        torch.rand(L, PN_K // 2, generator=g) > 0.3)
    pn_mask[:, 0] = True
    leaf_stats = torch.randn(L, 19, generator=g)
    leaf_sdf = torch.randn(L, generator=g)
    leaf_sdf_grad = torch.randn(L, 2, generator=g)
    leaf_did = torch.randn(L, DID_BINS, generator=g)
    norm_pos = torch.empty(N_QUERY, 2).uniform_(-1, 1, generator=g)
    sdf = torch.randn(N_QUERY, generator=g)
    sdf_grad = torch.randn(N_QUERY, 2, generator=g)
    idw_indices = torch.randint(0, L, (N_QUERY, 4), generator=g)
    raw_w = torch.rand(N_QUERY, 4, generator=g) + 1e-3
    idw_weights = raw_w / raw_w.sum(dim=-1, keepdim=True)
    uinf = torch.randn(2, generator=g)
    return dict(
        leaf_centroids=leaf_centroids, leaf_norm_pos=leaf_norm_pos,
        pn_input=pn_input, pn_mask=pn_mask,
        leaf_stats=leaf_stats, leaf_sdf=leaf_sdf, leaf_sdf_grad=leaf_sdf_grad,
        leaf_did=leaf_did, norm_pos=norm_pos, sdf=sdf, sdf_grad=sdf_grad,
        idw_indices=idw_indices, idw_weights=idw_weights, uinf=uinf,
    )


def _build_inputs_b1(case, head_dim):
    cos_x, sin_x, cos_y, sin_y = precompute_case_rope(
        case['leaf_centroids'], head_dim, DOMAIN_X, DOMAIN_Y,
        rope_scale=32.0, rope_base=100.0, num_registers=2)

    def u(t):
        return t.unsqueeze(0)
    return dict(
        norm_pos=u(case['norm_pos']),
        sdf=u(case['sdf']), sdf_grad=u(case['sdf_grad']),
        idw_indices=u(case['idw_indices']),
        idw_weights=u(case['idw_weights']),
        uinf=u(case['uinf']),
        pn_input=u(case['pn_input']), pn_mask=u(case['pn_mask']),
        leaf_stats=u(case['leaf_stats']), leaf_sdf=u(case['leaf_sdf']),
        leaf_sdf_grad=u(case['leaf_sdf_grad']),
        leaf_norm_pos=u(case['leaf_norm_pos']),
        rope_cos_x=u(cos_x), rope_sin_x=u(sin_x),
        rope_cos_y=u(cos_y), rope_sin_y=u(sin_y),
        leaf_did=u(case['leaf_did']),
    )


def _stack_inputs(b1_inputs_list):
    return {k: torch.cat([d[k] for d in b1_inputs_list], dim=0)
            for k in b1_inputs_list[0]}


def _make_model():
    torch.manual_seed(42)
    return KDViT(**MODEL_KWARGS).double().float()


def test_b1_eq_b2_forward():
    model = _make_model().eval()
    head_dim = model.head_dim
    c0 = _make_case(0); c1 = _make_case(1)
    i0 = _build_inputs_b1(c0, head_dim)
    i1 = _build_inputs_b1(c1, head_dim)
    ib2 = _stack_inputs([i0, i1])
    with torch.no_grad():
        o0 = model(**i0)
        o1_ = model(**i1)
        ob2 = model(**ib2)
    ref = torch.cat([o0, o1_], dim=0)
    assert torch.allclose(ref, ob2, atol=1e-4, rtol=1e-4), (
        f'B=2 forward != stack(B=1): max abs diff '
        f'{(ref - ob2).abs().max().item():.3e}')


def test_b1_eq_b2_backward():
    model = _make_model()
    head_dim = model.head_dim
    c0 = _make_case(0); c1 = _make_case(1)
    i0 = _build_inputs_b1(c0, head_dim)
    i1 = _build_inputs_b1(c1, head_dim)

    # B=2 backward (mean over batch).
    model.zero_grad()
    ib2 = _stack_inputs([i0, i1])
    out = model(**ib2)
    loss = (out ** 2).mean(dim=(1, 2)).mean()
    loss.backward()
    g_b2 = {n: p.grad.detach().clone()
            for n, p in model.named_parameters() if p.grad is not None}

    # Two B=1 backward passes, averaged.
    model.zero_grad()
    o0 = model(**i0); l0 = (o0 ** 2).mean(); l0.backward()
    o1_ = model(**i1); l1 = (o1_ ** 2).mean(); l1.backward()
    g_b1 = {n: (p.grad.detach() / 2.0).clone()
            for n, p in model.named_parameters() if p.grad is not None}

    for n in g_b2:
        a = g_b2[n]; b = g_b1[n]
        if not torch.allclose(a, b, atol=1e-4, rtol=1e-4):
            raise AssertionError(
                f'grad mismatch on {n}: max abs '
                f'{(a - b).abs().max().item():.3e}')


def test_golden_regression():
    """Stable across refactors: writes golden_b1.pt on first run."""
    model = _make_model().eval()
    head_dim = model.head_dim
    c0 = _make_case(0)
    i0 = _build_inputs_b1(c0, head_dim)
    with torch.no_grad():
        out = model(**i0)
    # Tiny backward signature.
    model.train()
    out2 = model(**i0)
    loss = (out2 ** 2).mean()
    loss.backward()
    grad_fp = next((p.grad for p in model.parameters()
                    if p.grad is not None), None)
    grad_sig = grad_fp.detach().clone() if grad_fp is not None else None

    payload = {'forward_out': out.detach(), 'grad_first_param': grad_sig}
    if not osp.exists(GOLDEN_PATH):
        torch.save(payload, GOLDEN_PATH)
        print(f'[golden] wrote new baseline -> {GOLDEN_PATH}')
        return
    ref = torch.load(GOLDEN_PATH, map_location='cpu', weights_only=False)
    assert torch.allclose(out, ref['forward_out'], atol=1e-4, rtol=1e-4), \
        f'forward drift vs golden: max abs ' \
        f'{(out - ref["forward_out"]).abs().max().item():.3e}'
    if ref['grad_first_param'] is not None and grad_sig is not None:
        a = grad_sig; b = ref['grad_first_param']
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), \
            'grad drift vs golden'


if __name__ == '__main__':
    test_b1_eq_b2_forward()
    print('[OK] test_b1_eq_b2_forward')
    test_b1_eq_b2_backward()
    print('[OK] test_b1_eq_b2_backward')
    test_golden_regression()
    print('[OK] test_golden_regression')
