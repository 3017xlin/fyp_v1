"""innerleaf.py — Leaf-level neighbor utilities for KDViT."""

import math

import numpy as np
from scipy.spatial import cKDTree


def assign_points_to_leaves(points, split_axes, split_values, n_leaves=4096):
    if hasattr(points, 'numpy'):
        points = points.numpy()
    if hasattr(split_axes, 'numpy'):
        split_axes = split_axes.numpy()
    if hasattr(split_values, 'numpy'):
        split_values = split_values.numpy()

    N = points.shape[0]
    depth = int(np.log2(n_leaves))
    n_internal = n_leaves - 1

    node_idx = np.zeros(N, dtype=np.int64)
    for _ in range(depth):
        axes = split_axes[node_idx].astype(np.intp)
        vals = split_values[node_idx]
        point_vals = points[np.arange(N), axes]
        go_left = point_vals < vals
        node_idx = np.where(go_left, 2 * node_idx + 1, 2 * node_idx + 2)

    return (node_idx - n_internal).astype(np.int64)


def gather_leaf_neighbors(positions, leaf_assignments, leaf_centroids,
                          k, mode='nearest', rng_seed=42):
    if hasattr(positions, 'numpy'):
        positions = positions.numpy()
    if hasattr(leaf_assignments, 'numpy'):
        leaf_assignments = leaf_assignments.numpy()
    if hasattr(leaf_centroids, 'numpy'):
        leaf_centroids = leaf_centroids.numpy()

    if mode not in ('nearest', 'random'):
        raise ValueError(f"mode must be 'nearest' or 'random', got {mode!r}")

    n_leaves = leaf_centroids.shape[0]
    rel_pos = np.zeros((n_leaves, k, 2), dtype=np.float32)
    valid_mask = np.zeros((n_leaves, k), dtype=np.bool_)
    neighbor_indices = np.full((n_leaves, k), -1, dtype=np.int64)

    rng = np.random.default_rng(rng_seed) if mode == 'random' else None

    sort_order = np.argsort(leaf_assignments)
    sorted_leaves = leaf_assignments[sort_order]
    sorted_pos = positions[sort_order]
    boundaries = np.searchsorted(sorted_leaves, np.arange(n_leaves + 1))

    for leaf_idx in range(n_leaves):
        start, end = int(boundaries[leaf_idx]), int(boundaries[leaf_idx + 1])
        n_members = end - start
        if n_members == 0:
            continue

        members = sorted_pos[start:end]
        global_ids = sort_order[start:end]
        centroid = leaf_centroids[leaf_idx]

        if n_members <= k:
            sel = np.arange(n_members)
        elif mode == 'nearest':
            dists_sq = ((members - centroid) ** 2).sum(axis=1)
            sel = np.argpartition(dists_sq, k)[:k]
        else:
            sel = rng.choice(n_members, size=k, replace=False)

        n_sel = len(sel)
        rel_pos[leaf_idx, :n_sel] = members[sel] - centroid
        valid_mask[leaf_idx, :n_sel] = True
        neighbor_indices[leaf_idx, :n_sel] = global_ids[sel]

    return rel_pos, valid_mask, neighbor_indices


def compute_idw_weights(query_pos, leaf_centroids, idw_k=4):
    if hasattr(query_pos, 'numpy'):
        query_pos = query_pos.numpy()
    if hasattr(leaf_centroids, 'numpy'):
        leaf_centroids = leaf_centroids.numpy()

    tree = cKDTree(leaf_centroids)
    dists, indices = tree.query(query_pos, k=idw_k)

    eps = 1e-8
    inv_dists = 1.0 / (dists.astype(np.float64) + eps)
    weights = (inv_dists / inv_dists.sum(axis=1, keepdims=True)).astype(
        np.float32)

    return indices.astype(np.int64), weights


def compute_did(query_pos, polygon, n_bins=8, d_max=5.0, chunk_size=4096):
    if hasattr(query_pos, 'numpy'):
        query_pos = query_pos.numpy()
    if hasattr(polygon, 'numpy'):
        polygon = polygon.numpy()

    query_pos = np.asarray(query_pos, dtype=np.float32)
    polygon = np.asarray(polygon, dtype=np.float32)

    N = query_pos.shape[0]
    bin_width = (2.0 * math.pi) / n_bins
    out = np.full((N, n_bins), d_max, dtype=np.float32)
    if N == 0 or polygon.shape[0] == 0:
        return out

    two_pi = 2.0 * math.pi

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        q = query_pos[start:end]
        rel = polygon[None, :, :] - q[:, None, :]
        dists = np.linalg.norm(rel, axis=-1)
        dists_capped = np.minimum(dists, d_max)
        angles = np.mod(np.arctan2(rel[..., 1], rel[..., 0]), two_pi)
        bin_idx = np.floor(angles / bin_width).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        rel_in_bin = angles - bin_idx.astype(np.float64) * bin_width

        for k in range(n_bins):
            mask = (bin_idx == k)
            count = mask.sum(axis=1)
            has = count > 0
            if not has.any():
                continue
            safe_count = np.maximum(count, 1)
            sum_d = (dists_capped * mask).sum(axis=1)
            mean_d = sum_d / safe_count

            masked_hi = np.where(mask, rel_in_bin, -np.inf)
            masked_lo = np.where(mask, rel_in_bin, np.inf)
            span = masked_hi.max(axis=1) - masked_lo.min(axis=1)
            span = np.where(has, span, 0.0)
            w = np.clip(span / bin_width, 0.0, 1.0)
            val = w * mean_d + (1.0 - w) * d_max
            out[start:end, k] = np.where(has, val, d_max).astype(np.float32)

    return out
