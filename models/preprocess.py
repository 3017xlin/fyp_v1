"""preprocess.py — One-command preprocessing for KDViT (E3-D2)."""

import argparse
import json
import math
import os
import os.path as osp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from models.build_tree import (build_kdtree_splits,
                               build_kdtree_splits_per_level,
                               compute_leaf_bounds)
from models.innerleaf import (assign_points_to_leaves, compute_did,
                              compute_idw_weights, gather_leaf_neighbors)
from utils.reorganize import reorganize

CACHE_SCHEMA_VERSION = 10
NUT_LOG_FLOOR = 1e-6
IDW_K = 4
POINT_CURV_K = 8
STATS_K_REF = 20

DID_BINS = 8
DID_DMAX = 5.0

STATS_ZSCORE_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15, 16]

PN_SDF_CHANNEL = 2
PN_CURV_CHANNEL = 6


_W = {}


def polygon_sdf_and_grad(query, polygon, chunk_size=20000):
    assert query.dim() == 2 and query.shape[-1] == 2
    assert polygon.dim() == 2 and polygon.shape[-1] == 2
    a = polygon
    b = torch.roll(polygon, -1, dims=0)
    seg = b - a
    seg_len_sq = (seg ** 2).sum(-1) + 1e-12
    N = query.shape[0]
    device = query.device
    dtype = query.dtype
    sdf_out = torch.empty(N, dtype=dtype, device=device)
    grad_out = torch.empty(N, 2, dtype=dtype, device=device)
    two_pi = 2.0 * math.pi
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        q = query[start:end]
        B = q.shape[0]
        qa = q[:, None, :] - a[None, :, :]
        t = (qa * seg[None, :, :]).sum(-1) / seg_len_sq[None, :]
        t = t.clamp(0.0, 1.0)
        closest_all = a[None, :, :] + t[..., None] * seg[None, :, :]
        diff_all = q[:, None, :] - closest_all
        dist_sq = (diff_all ** 2).sum(-1)
        min_dist_sq, min_idx = dist_sq.min(dim=1)
        min_dist = min_dist_sq.sqrt()
        closest = closest_all[torch.arange(B, device=device), min_idx]
        a_q = polygon - q[:, None, :]
        b_q = torch.roll(polygon, -1, dims=0) - q[:, None, :]
        cross = a_q[..., 0] * b_q[..., 1] - a_q[..., 1] * b_q[..., 0]
        dot = a_q[..., 0] * b_q[..., 0] + a_q[..., 1] * b_q[..., 1]
        winding = torch.atan2(cross, dot).sum(dim=1) / two_pi
        is_inside = winding.round().long().ne(0)
        signed = torch.where(is_inside, -min_dist, min_dist)
        safe = min_dist.clamp(min=1e-7)
        direction = (q - closest) / safe[..., None]
        grad = torch.where(is_inside[..., None], -direction, direction)
        sdf_out[start:end] = signed
        grad_out[start:end] = grad
    return sdf_out, grad_out


def compute_point_curvature(positions, sdf_grad, k=POINT_CURV_K):
    from scipy.spatial import cKDTree
    n = positions.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    kq = min(k + 1, n)
    tree = cKDTree(positions)
    _, nbr = tree.query(positions, k=kq)
    if nbr.ndim == 1:
        nbr = nbr[:, None]
    ang = np.arctan2(sdf_grad[:, 1], sdf_grad[:, 0])
    nbr_ang = ang[nbr]
    s = np.sin(nbr_ang).mean(axis=1)
    c = np.cos(nbr_ang).mean(axis=1)
    R = np.sqrt(s * s + c * c)
    return (1.0 - R).astype(np.float32)


def compute_leaf_stats(full_pos, leaf_indices, leaf_centroids,
                       mesh_sdf_np, mesh_sdf_grad_np, surf_np,
                       uinf_raw_np, n_leaves, k_ref=STATS_K_REF):
    stats = np.zeros((n_leaves, 19), dtype=np.float32)

    uinf_norm = np.linalg.norm(uinf_raw_np)
    if uinf_norm > 1e-8:
        u_hat = uinf_raw_np / uinf_norm
    else:
        u_hat = np.array([1.0, 0.0], dtype=np.float32)
    u_perp = np.array([-u_hat[1], u_hat[0]], dtype=np.float32)

    sort_order = np.argsort(leaf_indices)
    sorted_leaves = leaf_indices[sort_order]
    sorted_pos = full_pos[sort_order]
    sorted_sdf = mesh_sdf_np[sort_order]
    sorted_sdf_grad = mesh_sdf_grad_np[sort_order]
    sorted_surf = surf_np[sort_order]
    boundaries = np.searchsorted(sorted_leaves, np.arange(n_leaves + 1))

    for leaf_idx in range(n_leaves):
        start, end = int(boundaries[leaf_idx]), int(boundaries[leaf_idx + 1])
        n_members = end - start
        centroid = leaf_centroids[leaf_idx]
        if n_members == 0:
            continue

        members = sorted_pos[start:end]
        member_sdf = sorted_sdf[start:end]
        member_sdf_grad = sorted_sdf_grad[start:end]
        member_surf = sorted_surf[start:end]
        rel = members - centroid

        cov_xx = np.mean(rel[:, 0] ** 2)
        cov_yy = np.mean(rel[:, 1] ** 2)
        cov_xy = np.mean(rel[:, 0] * rel[:, 1])

        dists = np.linalg.norm(rel, axis=1)
        dist_mean = np.mean(dists)
        dist_std = np.std(dists)
        if dist_std > 1e-12 and n_members > 2:
            dist_skew = float(
                np.mean((dists - dist_mean) ** 3) / (dist_std ** 3))
        else:
            dist_skew = 0.0

        n_valid_norm = float(min(n_members, k_ref)) / k_ref
        area = math.pi * dist_mean ** 2 + 1e-8
        density = math.log1p(float(n_members) / area)

        com = np.mean(members, axis=0)
        com_dist = float(np.linalg.norm(com - centroid))

        sdf_min = float(np.min(member_sdf))
        sdf_max = float(np.max(member_sdf))
        sdf_range = sdf_max - sdf_min

        theta = np.arctan2(rel[:, 1], rel[:, 0])
        theta_sin = float(np.mean(np.sin(theta)))
        theta_cos = float(np.mean(np.cos(theta)))
        R = math.sqrt(theta_sin ** 2 + theta_cos ** 2)
        angular_span = 1.0 - R

        streamwise = float(np.dot(centroid, u_hat))
        crossflow = float(np.dot(centroid, u_perp))

        grad_norms = np.linalg.norm(member_sdf_grad, axis=1)
        valid_grad = grad_norms > 1e-8
        if np.sum(valid_grad) > 1:
            ga = np.arctan2(member_sdf_grad[valid_grad, 1],
                            member_sdf_grad[valid_grad, 0])
            gs = float(np.mean(np.sin(ga)))
            gc = float(np.mean(np.cos(ga)))
            R_grad = math.sqrt(gs ** 2 + gc ** 2)
            curvature_est = 1.0 - R_grad
        else:
            curvature_est = 0.0

        surf_ratio = float(np.mean(member_surf.astype(np.float32)))

        stats[leaf_idx] = [
            cov_xx, cov_yy, cov_xy,
            dist_mean, dist_std, dist_skew,
            n_valid_norm, density, com_dist,
            sdf_min, sdf_max, sdf_range,
            theta_sin, theta_cos, angular_span,
            streamwise, crossflow,
            curvature_est, surf_ratio,
        ]
    return stats


def _assemble_pn_input(rel_pos, valid_mask, neighbor_indices,
                       mesh_sdf, mesh_sdf_grad, surf_float, point_curv):
    safe = np.where(valid_mask, neighbor_indices, 0)
    nb_sdf = np.where(valid_mask, mesh_sdf[safe], 0.0).astype(np.float32)
    nb_grad = mesh_sdf_grad[safe].astype(np.float32)
    nb_grad[~valid_mask] = 0.0
    nb_surf = np.where(valid_mask, surf_float[safe], 0.0).astype(np.float32)
    nb_curv = np.where(valid_mask, point_curv[safe], 0.0).astype(np.float32)
    return np.concatenate([
        rel_pos,
        nb_sdf[..., None],
        nb_grad,
        nb_surf[..., None],
        nb_curv[..., None],
    ], axis=-1).astype(np.float32)


def _bbox_first_order_neighbors(leaf_bounds, eps=1e-6):
    bounds = np.asarray(leaf_bounds, dtype=np.float32)
    L = bounds.shape[0]
    x_lo = bounds[:, 0, 0]; y_lo = bounds[:, 0, 1]
    x_hi = bounds[:, 1, 0]; y_hi = bounds[:, 1, 1]

    x_touch = (np.abs(x_hi[:, None] - x_lo[None, :]) < eps) | \
              (np.abs(x_lo[:, None] - x_hi[None, :]) < eps)
    y_overlap = (np.minimum(y_hi[:, None], y_hi[None, :])
                 - np.maximum(y_lo[:, None], y_lo[None, :]))
    x_adj = x_touch & (y_overlap > eps)

    y_touch = (np.abs(y_hi[:, None] - y_lo[None, :]) < eps) | \
              (np.abs(y_lo[:, None] - y_hi[None, :]) < eps)
    x_overlap = (np.minimum(x_hi[:, None], x_hi[None, :])
                 - np.maximum(x_lo[:, None], x_lo[None, :]))
    y_adj = y_touch & (x_overlap > eps)

    A = x_adj | y_adj
    np.fill_diagonal(A, False)

    counts = A.sum(axis=1)
    max_k = int(counts.max()) if L > 0 else 0
    out = np.full((L, max(max_k, 1)), -1, dtype=np.int64)
    if max_k > 0:
        for i in range(L):
            nbrs = np.where(A[i])[0]
            if nbrs.size:
                out[i, :nbrs.size] = nbrs
    return out


class _Welford:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None

    def update(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        bn = arr.shape[0]
        if bn == 0:
            return
        b_mean = arr.mean(axis=0)
        b_M2 = ((arr - b_mean) ** 2).sum(axis=0)
        if self.n == 0:
            self.n, self.mean, self.M2 = bn, b_mean, b_M2
        else:
            delta = b_mean - self.mean
            total = self.n + bn
            self.mean = self.mean + delta * bn / total
            self.M2 = (self.M2 + b_M2
                       + delta * delta * self.n * bn / total)
            self.n = total

    def finalize(self):
        if self.n == 0:
            raise RuntimeError('Welford: no data')
        return self.mean, np.sqrt(self.M2 / self.n)


def build_tree_phase(n_leaves, output_dir, domain_x, domain_y,
                     did_bins, did_dmax, axis_mode):
    depth = int(round(math.log2(n_leaves)))
    assert 2 ** depth == n_leaves, 'n_leaves must be a power of 2'
    assert axis_mode in ('per_node', 'per_level'), \
        f'axis_mode must be per_node or per_level, got {axis_mode!r}'
    kdtree = {
        'version': CACHE_SCHEMA_VERSION,
        'depth': depth,
        'n_leaves': n_leaves,
        'domain_x_range': torch.tensor(domain_x, dtype=torch.float32),
        'domain_y_range': torch.tensor(domain_y, dtype=torch.float32),
        'did_bins': int(did_bins),
        'did_dmax': float(did_dmax),
        'axis_mode': str(axis_mode),
    }
    torch.save(kdtree, osp.join(output_dir, 'kdtree.pt'))
    print(f'[phase0] wrote metadata to {osp.join(output_dir, "kdtree.pt")} '
          f'(n_leaves={n_leaves}, depth={depth}, '
          f'did_bins={did_bins}, did_dmax={did_dmax}, '
          f'axis_mode={axis_mode})')
    return kdtree


def _init_base_worker(n_leaves, depth, domain_x, domain_y, idw_k,
                      my_path, base_dir, did_bins, did_dmax,
                      axis_mode):
    _W.clear()
    _W.update(dict(
        n_leaves=n_leaves, depth=depth,
        domain_x=domain_x, domain_y=domain_y, idw_k=idw_k,
        my_path=my_path, base_dir=base_dir,
        did_bins=did_bins, did_dmax=did_dmax,
        axis_mode=axis_mode))
    torch.set_num_threads(1)
    os.environ.setdefault('OMP_NUM_THREADS', '1')


def process_base_case(cname):
    import pyvista as pv
    n_leaves = _W['n_leaves']
    depth = _W['depth']
    domain_x = _W['domain_x']
    domain_y = _W['domain_y']
    idw_k = _W['idw_k']
    my_path = _W['my_path']
    did_bins = _W['did_bins']
    did_dmax = _W['did_dmax']

    internal = pv.read(osp.join(my_path, cname, cname + '_internal.vtu'))
    aerofoil = pv.read(osp.join(my_path, cname, cname + '_aerofoil.vtp'))

    airfoil_pos = torch.tensor(aerofoil.points[:, :2], dtype=torch.float32)
    full_pos = torch.tensor(internal.points[:, :2], dtype=torch.float32)
    surf_bool_np = (internal.point_data['U'][:, 0] == 0)
    surf = torch.from_numpy(surf_bool_np).bool()

    parts = cname.split('_')
    Uinf = float(parts[2])
    alpha_rad = float(parts[3]) * np.pi / 180.0
    uinf_raw = torch.tensor(
        [np.cos(alpha_rad) * Uinf, np.sin(alpha_rad) * Uinf],
        dtype=torch.float32)

    nut_raw = internal.point_data['nut']
    nut_log = np.log(np.maximum(nut_raw, NUT_LOG_FLOOR))
    U2 = internal.point_data['U'][:, :2]
    p1 = internal.point_data['p'][:, None]
    full_y_linear = torch.tensor(
        np.concatenate([U2, p1, nut_raw[:, None]], axis=-1),
        dtype=torch.float32)
    full_y_log = torch.tensor(
        np.concatenate([U2, p1, nut_log[:, None]], axis=-1),
        dtype=torch.float32)

    pos_np = full_pos.numpy()
    if _W.get('axis_mode', 'per_node') == 'per_level':
        percase_split_axes, percase_split_values = \
            build_kdtree_splits_per_level(pos_np, depth)
    else:
        percase_split_axes, percase_split_values = build_kdtree_splits(
            pos_np, depth)
    leaf_idx = assign_points_to_leaves(
        pos_np, percase_split_axes, percase_split_values, n_leaves)

    domain_lo = np.asarray([domain_x[0], domain_y[0]], dtype=np.float32)
    domain_hi = np.asarray([domain_x[1], domain_y[1]], dtype=np.float32)
    leaf_bounds_np = compute_leaf_bounds(
        percase_split_axes, percase_split_values, n_leaves,
        domain_lo, domain_hi)
    leaf_geo_neighbors_np = _bbox_first_order_neighbors(leaf_bounds_np)

    counts = np.bincount(leaf_idx.astype(np.int64), minlength=n_leaves)
    sums = np.zeros((n_leaves, 2), dtype=np.float64)
    np.add.at(sums, leaf_idx, pos_np.astype(np.float64))
    nonzero = counts > 0
    percase_centroids = np.zeros((n_leaves, 2), dtype=np.float32)
    percase_centroids[nonzero] = (
        sums[nonzero] / counts[nonzero, None]).astype(np.float32)

    mesh_sdf, mesh_sdf_grad = polygon_sdf_and_grad(full_pos, airfoil_pos)
    if surf_bool_np.any():
        outward_normals = -aerofoil.point_data['Normals'][:, :2]
        reordered = reorganize(
            aerofoil.points[:, :2],
            internal.points[surf_bool_np, :2],
            outward_normals)
        mesh_sdf[surf] = 0.0
        mesh_sdf_grad[surf] = torch.from_numpy(
            np.ascontiguousarray(reordered)).float()

    leaf_centroids_t = torch.from_numpy(percase_centroids).float()
    leaf_sdf, leaf_sdf_grad = polygon_sdf_and_grad(
        leaf_centroids_t, airfoil_pos)

    dx = domain_x[1] - domain_x[0]
    dy = domain_y[1] - domain_y[0]
    cx = leaf_centroids_t[:, 0]
    cy = leaf_centroids_t[:, 1]
    leaf_norm_x = (2.0 * (cx - domain_x[0]) / dx - 1.0).clamp(-1, 1)
    leaf_norm_y = (2.0 * (cy - domain_y[0]) / dy - 1.0).clamp(-1, 1)
    leaf_norm_pos = torch.stack([leaf_norm_x, leaf_norm_y], dim=-1)

    leaf_did_raw_np = compute_did(
        percase_centroids, airfoil_pos.numpy(),
        n_bins=did_bins, d_max=did_dmax)
    leaf_did_raw = torch.from_numpy(leaf_did_raw_np)

    mesh_sdf_np = mesh_sdf.numpy()
    mesh_sdf_grad_np = mesh_sdf_grad.numpy()
    surf_float_np = surf_bool_np.astype(np.float32)

    point_curv = compute_point_curvature(full_pos.numpy(), mesh_sdf_grad_np)

    leaf_stats = compute_leaf_stats(
        pos_np, leaf_idx, percase_centroids,
        mesh_sdf_np, mesh_sdf_grad_np, surf_float_np,
        uinf_raw.numpy(), n_leaves)

    idw_indices, idw_weights = compute_idw_weights(
        pos_np, percase_centroids, idw_k)

    x_norm = (2.0 * (full_pos[:, 0] - domain_x[0]) / dx - 1.0).clamp(-1, 1)
    y_norm = (2.0 * (full_pos[:, 1] - domain_y[0]) / dy - 1.0).clamp(-1, 1)
    norm_pos = torch.stack([x_norm, y_norm], dim=-1)

    base = {
        'version': CACHE_SCHEMA_VERSION,
        'case_name': cname,
        'pos': full_pos,
        'norm_pos': norm_pos,
        'surf': surf,
        'sdf_grad': mesh_sdf_grad,
        'leaf_centroids': torch.from_numpy(percase_centroids),
        'leaf_norm_pos': leaf_norm_pos,
        'leaf_sdf_grad': leaf_sdf_grad,
        'idw_indices': torch.from_numpy(idw_indices),
        'idw_weights': torch.from_numpy(idw_weights),
        'leaf_idx': torch.from_numpy(leaf_idx),
        'airfoil_pos': airfoil_pos,
        'full_y_linear': full_y_linear,
        'full_y_log': full_y_log,
        'uinf_raw': uinf_raw,
        'mesh_sdf_raw': mesh_sdf,
        'leaf_sdf_raw': leaf_sdf,
        'point_curv_raw': torch.from_numpy(point_curv),
        'leaf_stats_raw': torch.from_numpy(leaf_stats),
        'leaf_did_raw': leaf_did_raw,
        'leaf_bounds': torch.from_numpy(leaf_bounds_np),
        'leaf_geo_neighbors': torch.from_numpy(leaf_geo_neighbors_np),
    }
    return base, int(pos_np.shape[0])


def _base_worker(cname):
    try:
        base, n_points = process_base_case(cname)
        torch.save(base, osp.join(_W['base_dir'], cname + '.pt'))
        return ('ok', cname, None, n_points)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ('failed', cname, str(exc), 0)


def _init_pn_worker(base_dir, pn_dirs, k_levels):
    _W.clear()
    _W.update(dict(base_dir=base_dir, pn_dirs=pn_dirs, k_levels=k_levels))
    torch.set_num_threads(1)
    os.environ.setdefault('OMP_NUM_THREADS', '1')


_PN_LEVELS = {
    'full': ('full', 'random', 42),
}


def _pn_worker(cname):
    try:
        base = torch.load(osp.join(_W['base_dir'], cname + '.pt'),
                          map_location='cpu', weights_only=False)
        pos = base['pos'].numpy()
        leaf_idx = base['leaf_idx'].numpy()
        leaf_centroids = base['leaf_centroids'].numpy()
        mesh_sdf = base['mesh_sdf_raw'].numpy()
        mesh_sdf_grad = base['sdf_grad'].numpy()
        surf_float = base['surf'].numpy().astype(np.float32)
        point_curv = base['point_curv_raw'].numpy()

        for level, (k_key, mode, seed) in _PN_LEVELS.items():
            k = _W['k_levels'][k_key]
            rel_pos, valid_mask, nbr_idx = gather_leaf_neighbors(
                pos, leaf_idx, leaf_centroids, k, mode=mode, rng_seed=seed)
            pn_input = _assemble_pn_input(
                rel_pos, valid_mask, nbr_idx,
                mesh_sdf, mesh_sdf_grad, surf_float, point_curv)
            out = {
                'version': CACHE_SCHEMA_VERSION,
                'case_name': cname,
                'pn_input': torch.from_numpy(pn_input),
                'pn_mask': torch.from_numpy(valid_mask),
            }
            torch.save(out, osp.join(_W['pn_dirs'][level], cname + '.pt'))
        return ('ok', cname, None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ('failed', cname, str(exc))


def compute_coef_norm(base_dir, train_cases, task, idw_k):
    print(f'[phase3] computing coef_norm over {len(train_cases)} cases')
    uinfs = []
    sdf_acc = _Welford()
    y_lin_acc = _Welford()
    y_log_acc = _Welford()
    curv_acc = _Welford()
    stats_acc = _Welford()
    did_acc = _Welford()

    for c in tqdm(train_cases, desc='phase3:coef_norm'):
        case = torch.load(osp.join(base_dir, c + '.pt'),
                          map_location='cpu', weights_only=False)
        uinfs.append(case['uinf_raw'].numpy())
        sdf_pool = np.concatenate([
            case['mesh_sdf_raw'].numpy(), case['leaf_sdf_raw'].numpy()])
        sdf_acc.update(sdf_pool)
        y_lin_acc.update(case['full_y_linear'].numpy())
        y_log_acc.update(case['full_y_log'].numpy())
        curv_acc.update(case['point_curv_raw'].numpy())
        ls = case['leaf_stats_raw'].numpy()
        stats_acc.update(ls[:, STATS_ZSCORE_INDICES])
        did_acc.update(case['leaf_did_raw'].numpy())

    uinfs = np.stack(uinfs, axis=0).astype(np.float64)
    mean_uinf = uinfs.mean(axis=0).astype(np.float32)
    std_uinf = uinfs.std(axis=0).astype(np.float32)
    mean_sdf, std_sdf = sdf_acc.finalize()
    mean_out_lin, std_out_lin = y_lin_acc.finalize()
    mean_out_log, std_out_log = y_log_acc.finalize()
    mean_curv, std_curv = curv_acc.finalize()
    mean_stats_z, std_stats_z = stats_acc.finalize()
    mean_did, std_did = did_acc.finalize()

    mean_leaf_stats = np.zeros(19, dtype=np.float32)
    std_leaf_stats = np.ones(19, dtype=np.float32)
    for i_out, i_src in enumerate(STATS_ZSCORE_INDICES):
        mean_leaf_stats[i_src] = float(mean_stats_z[i_out])
        std_leaf_stats[i_src] = float(std_stats_z[i_out])

    surf_bc_phys_linear = torch.tensor([0., 0., 0., 0.], dtype=torch.float32)
    surf_bc_phys_log = torch.tensor(
        [0., 0., 0., float(np.log(NUT_LOG_FLOOR))], dtype=torch.float32)

    return {
        'version': CACHE_SCHEMA_VERSION,
        'task': task,
        'mean_uinf': torch.from_numpy(mean_uinf),
        'std_uinf': torch.from_numpy(std_uinf),
        'mean_sdf': torch.tensor(float(mean_sdf), dtype=torch.float32),
        'std_sdf': torch.tensor(float(std_sdf), dtype=torch.float32),
        'mean_out_linear': torch.from_numpy(mean_out_lin.astype(np.float32)),
        'std_out_linear': torch.from_numpy(std_out_lin.astype(np.float32)),
        'mean_out_log': torch.from_numpy(mean_out_log.astype(np.float32)),
        'std_out_log': torch.from_numpy(std_out_log.astype(np.float32)),
        'surf_bc_phys_linear': surf_bc_phys_linear,
        'surf_bc_phys_log': surf_bc_phys_log,
        'nut_log_floor': float(NUT_LOG_FLOOR),
        'idw_k': idw_k,
        'mean_curv': torch.tensor(float(mean_curv), dtype=torch.float32),
        'std_curv': torch.tensor(float(std_curv), dtype=torch.float32),
        'mean_leaf_stats': torch.from_numpy(mean_leaf_stats),
        'std_leaf_stats': torch.from_numpy(std_leaf_stats),
        'mean_did': torch.from_numpy(mean_did.astype(np.float32)),
        'std_did': torch.from_numpy(std_did.astype(np.float32)),
    }


def _init_norm_worker(coef_norm, base_dir, pn_dirs):
    _W.clear()
    _W.update(dict(coef_norm=coef_norm, base_dir=base_dir, pn_dirs=pn_dirs))
    torch.set_num_threads(1)


def _norm_worker(cname):
    try:
        cn = _W['coef_norm']
        eps = 1e-8
        mean_sdf = cn['mean_sdf']
        std_sdf = cn['std_sdf']
        mean_curv = cn['mean_curv']
        std_curv = cn['std_curv']
        mean_did = cn['mean_did']
        std_did = cn['std_did']

        bpath = osp.join(_W['base_dir'], cname + '.pt')
        case = torch.load(bpath, map_location='cpu', weights_only=False)
        case['y_linear'] = (case['full_y_linear'] - cn['mean_out_linear']) / (
            cn['std_out_linear'] + eps)
        case['y_log'] = (case['full_y_log'] - cn['mean_out_log']) / (
            cn['std_out_log'] + eps)
        case['uinf'] = (case['uinf_raw'] - cn['mean_uinf']) / (
            cn['std_uinf'] + eps)
        case['sdf'] = (case['mesh_sdf_raw'] - mean_sdf) / (std_sdf + eps)
        case['leaf_sdf'] = (case['leaf_sdf_raw'] - mean_sdf) / (std_sdf + eps)
        case['curv'] = (case['point_curv_raw'] - mean_curv) / (std_curv + eps)
        case['leaf_stats'] = (case['leaf_stats_raw'] - cn['mean_leaf_stats']) \
            / (cn['std_leaf_stats'] + eps)
        case['leaf_did'] = (case['leaf_did_raw'] - mean_did) / (std_did + eps)
        for k in ['full_y_linear', 'full_y_log', 'uinf_raw', 'mesh_sdf_raw',
                  'leaf_sdf_raw', 'point_curv_raw', 'leaf_stats_raw',
                  'leaf_did_raw']:
            case.pop(k, None)
        torch.save(case, bpath)

        for level in _PN_LEVELS:
            ppath = osp.join(_W['pn_dirs'][level], cname + '.pt')
            pn = torch.load(ppath, map_location='cpu', weights_only=False)
            x = pn['pn_input']
            mask = pn['pn_mask']
            x[..., PN_SDF_CHANNEL] = torch.where(
                mask, (x[..., PN_SDF_CHANNEL] - mean_sdf) / (std_sdf + eps),
                x[..., PN_SDF_CHANNEL])
            x[..., PN_CURV_CHANNEL] = torch.where(
                mask, (x[..., PN_CURV_CHANNEL] - mean_curv) / (std_curv + eps),
                x[..., PN_CURV_CHANNEL])
            pn['pn_input'] = x
            torch.save(pn, ppath)
        return ('ok', cname, None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ('failed', cname, str(exc))


def _run_pool(cases, worker_fn, initializer, initargs, workers, desc):
    results = []
    if workers <= 1:
        initializer(*initargs)
        for c in tqdm(cases, desc=desc):
            results.append(worker_fn(c))
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=initializer,
                                 initargs=initargs) as ex:
            futs = {ex.submit(worker_fn, c): c for c in cases}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                results.append(fut.result())
    return results


def _report_failures(results, phase):
    failures = [(r[1], r[2]) for r in results if r[0] == 'failed']
    if failures:
        print(f'[{phase}] {len(failures)} cases failed:')
        for name, msg in failures[:20]:
            print(f'  {name}: {msg}')
    return failures


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass
    ap = argparse.ArgumentParser(description='KDViT one-command preprocessing')
    ap.add_argument('--data_dir', default='data/Dataset')
    ap.add_argument('--task', default='full',
                    choices=['full', 'scarce', 'reynolds', 'aoa'])
    ap.add_argument('--n_leaves', type=int, required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--idw_k', type=int, default=IDW_K)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--domain_x_range', type=float, nargs=2,
                    default=[-2.0, 4.0])
    ap.add_argument('--domain_y_range', type=float, nargs=2,
                    default=[-1.5, 1.5])
    ap.add_argument('--did_bins', type=int, default=DID_BINS)
    ap.add_argument('--did_dmax', type=float, default=DID_DMAX)
    ap.add_argument('--axis_mode', type=str, default='per_level',
                    choices=['per_node', 'per_level'])
    args = ap.parse_args()

    base_dir = osp.join(args.output_dir, 'base')
    pn_dirs = {lv: osp.join(args.output_dir, f'pn_{lv}') for lv in _PN_LEVELS}
    os.makedirs(base_dir, exist_ok=True)
    for d in pn_dirs.values():
        os.makedirs(d, exist_ok=True)

    kdtree = build_tree_phase(
        args.n_leaves, args.output_dir,
        args.domain_x_range, args.domain_y_range,
        args.did_bins, args.did_dmax,
        args.axis_mode)
    n_leaves = int(kdtree['n_leaves'])
    depth = int(kdtree['depth'])
    domain_x = kdtree['domain_x_range'].numpy().tolist()
    domain_y = kdtree['domain_y_range'].numpy().tolist()

    with open(osp.join(args.data_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    if args.task == 'scarce':
        train_cases = manifest['scarce_train']
        test_cases = manifest['full_test']
    else:
        train_cases = manifest[args.task + '_train']
        test_cases = manifest[args.task + '_test']
    all_cases = list(dict.fromkeys(train_cases + test_cases))
    print(f'[main] train={len(train_cases)} test={len(test_cases)} '
          f'total={len(all_cases)}')

    print('\n=== PHASE 1: base data ===')
    results = _run_pool(
        all_cases, _base_worker, _init_base_worker,
        (n_leaves, depth, domain_x, domain_y,
         args.idw_k, args.data_dir, base_dir,
         args.did_bins, args.did_dmax,
         args.axis_mode),
        args.workers, 'phase1:base')
    _report_failures(results, 'phase1')

    ok = [r for r in results if r[0] == 'ok']
    if not ok:
        raise RuntimeError('Phase 1 produced no cases.')
    total_points = sum(r[3] for r in ok)
    global_mean = total_points / (len(ok) * n_leaves)
    k_full = max(1, int(round(global_mean * 1.0)))
    k_levels = {'full': int(k_full)}
    with open(osp.join(args.output_dir, 'k_levels.json'), 'w') as f:
        json.dump(k_levels, f, indent=2)
    print(f'[main] k_levels = {k_levels} '
          f'(global mean leaf occupancy = {global_mean:.1f})')

    print('\n=== PHASE 2: PointNet neighbors ===')
    cached = [r[1] for r in ok]
    results = _run_pool(
        cached, _pn_worker, _init_pn_worker,
        (base_dir, pn_dirs, k_levels),
        args.workers, 'phase2:pn')
    _report_failures(results, 'phase2')

    print('\n=== PHASE 3: coef_norm + normalize ===')
    train_ok = [c for c in train_cases
                if osp.exists(osp.join(base_dir, c + '.pt'))]
    coef_norm = compute_coef_norm(base_dir, train_ok, args.task, args.idw_k)
    coef_path = osp.join(args.output_dir, f'coef_norm_{args.task}.pt')
    torch.save(coef_norm, coef_path)
    print(f'[phase3] wrote {coef_path}')

    norm_cases = [c for c in all_cases
                  if osp.exists(osp.join(base_dir, c + '.pt'))]
    results = _run_pool(
        norm_cases, _norm_worker, _init_norm_worker,
        (coef_norm, base_dir, pn_dirs),
        args.workers, 'phase3:normalize')
    _report_failures(results, 'phase3')

    print(f'\nDone -> {args.output_dir}')


if __name__ == '__main__':
    main()
