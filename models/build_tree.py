"""build_tree.py — KD-tree construction helpers (pure module).

Builds an equal-mass KD-tree from a pooled point cloud using recursive
median splits with **max-spread** axis selection: at every node the split
axis is whichever dimension (x or y) has the larger extent over that node's
subset. This lets sibling subtrees pick different axes while keeping leaf
numbering contiguous in space, which is what makes ``arange(L).reshape(T, P)``
patch grouping spatially coherent for any patch_size.

No CLI: tree construction is driven by ``models/preprocess.py`` (Phase 0),
which imports ``build_kdtree_splits``, ``compute_leaf_bounds`` and
``read_mesh_positions`` from here.
"""

import os.path as osp

import numpy as np


def read_mesh_positions(data_dir, case_name):
    import pyvista as pv
    vtu_path = osp.join(data_dir, case_name, case_name + '_internal.vtu')
    mesh = pv.read(vtu_path)
    return mesh.points[:, :2].astype(np.float32)


def build_kdtree_splits(points, depth=12):
    N = points.shape[0]
    n_internal = 2 ** depth - 1

    split_axes = np.zeros(n_internal, dtype=np.int8)
    split_values = np.zeros(n_internal, dtype=np.float32)

    indices = np.arange(N, dtype=np.int64)
    ranges = [(0, N)]

    for d in range(depth):
        next_ranges = []
        node_offset = 2 ** d - 1

        for i, (start, end) in enumerate(ranges):
            node_idx = node_offset + i
            count = end - start

            if count == 0:
                split_axes[node_idx] = 0
                split_values[node_idx] = 0.0
                next_ranges.append((start, start))
                next_ranges.append((start, end))
                continue

            seg = indices[start:end]
            sub = points[seg]

            spreads = sub.max(axis=0) - sub.min(axis=0)
            axis = int(np.argmax(spreads))
            split_axes[node_idx] = axis

            mid = count // 2
            if mid == 0:
                split_values[node_idx] = float(sub[0, axis])
                next_ranges.append((start, start))
                next_ranges.append((start, end))
                continue

            vals = points[seg, axis]
            order = np.argpartition(vals, mid)
            indices[start:end] = seg[order]
            split_values[node_idx] = float(points[indices[start + mid], axis])

            next_ranges.append((start, start + mid))
            next_ranges.append((start + mid, end))

        ranges = next_ranges

    return split_axes, split_values


def build_kdtree_splits_per_level(points, depth=12):
    N = points.shape[0]
    n_internal = 2 ** depth - 1

    split_axes = np.zeros(n_internal, dtype=np.int8)
    split_values = np.zeros(n_internal, dtype=np.float32)

    indices = np.arange(N, dtype=np.int64)
    ranges = [(0, N)]

    for d in range(depth):
        spreads = np.zeros(2, dtype=np.float64)
        for (s, e) in ranges:
            if e > s:
                sub = points[indices[s:e]]
                spreads += sub.max(axis=0) - sub.min(axis=0)
        axis = int(np.argmax(spreads))

        next_ranges = []
        node_offset = 2 ** d - 1

        for i, (start, end) in enumerate(ranges):
            node_idx = node_offset + i
            count = end - start
            split_axes[node_idx] = axis

            if count == 0:
                split_values[node_idx] = 0.0
                next_ranges.append((start, start))
                next_ranges.append((start, end))
                continue

            seg = indices[start:end]
            mid = count // 2
            if mid == 0:
                split_values[node_idx] = float(points[seg[0], axis])
                next_ranges.append((start, start))
                next_ranges.append((start, end))
                continue

            vals = points[seg, axis]
            order = np.argpartition(vals, mid)
            indices[start:end] = seg[order]
            split_values[node_idx] = float(points[indices[start + mid], axis])

            next_ranges.append((start, start + mid))
            next_ranges.append((start + mid, end))

        ranges = next_ranges

    return split_axes, split_values


def compute_leaf_bounds(split_axes, split_values, n_leaves,
                        domain_lo, domain_hi):
    n_internal = n_leaves - 1
    bounds = np.zeros((n_leaves, 2, 2), dtype=np.float32)

    for leaf_idx in range(n_leaves):
        lo = domain_lo.copy()
        hi = domain_hi.copy()
        tree_idx = leaf_idx + n_internal
        while tree_idx > 0:
            parent = (tree_idx - 1) // 2
            is_left = (tree_idx == 2 * parent + 1)
            axis = int(split_axes[parent])
            sv = split_values[parent]
            if is_left:
                hi[axis] = min(hi[axis], sv)
            else:
                lo[axis] = max(lo[axis], sv)
            tree_idx = parent
        bounds[leaf_idx, 0] = lo
        bounds[leaf_idx, 1] = hi

    return bounds
