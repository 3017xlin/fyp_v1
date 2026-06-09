"""dataset_cached.py — Load the KDViT split cache as PyG Data objects."""

import json
import os.path as osp

import torch
from torch_geometric.data import Data
from tqdm import tqdm

CACHE_SCHEMA_VERSION = 10
_VALID_ENCODINGS = ('linear', 'log')
_VALID_LEVELS = ('full',)


def _manifest_key(task, split):
    if task == 'scarce':
        return 'scarce_train' if split == 'train' else 'full_test'
    return f'{task}_{split}'


def _load_raw_coef_norm(cache_dir, task):
    coef_path = osp.join(cache_dir, f'coef_norm_{task}.pt')
    if not osp.exists(coef_path):
        raise FileNotFoundError(
            f"Missing {coef_path}. Run preprocess.py first.")
    return torch.load(coef_path, map_location='cpu', weights_only=False)


def _project_coef_norm(raw_coef_norm, nut_encoding):
    if nut_encoding not in _VALID_ENCODINGS:
        raise ValueError(
            f"nut_encoding must be one of {_VALID_ENCODINGS}, "
            f"got {nut_encoding!r}")
    return {
        'version':         raw_coef_norm['version'],
        'task':            raw_coef_norm.get('task'),
        'mean_uinf':       raw_coef_norm['mean_uinf'],
        'std_uinf':        raw_coef_norm['std_uinf'],
        'mean_sdf':        raw_coef_norm['mean_sdf'],
        'std_sdf':         raw_coef_norm['std_sdf'],
        'mean_out':        raw_coef_norm[f'mean_out_{nut_encoding}'],
        'std_out':         raw_coef_norm[f'std_out_{nut_encoding}'],
        'surf_bc_phys':    raw_coef_norm.get(f'surf_bc_phys_{nut_encoding}'),
        'mean_out_report': raw_coef_norm['mean_out_linear'],
        'std_out_report':  raw_coef_norm['std_out_linear'],
        'nut_encoding':    nut_encoding,
        'nut_log_floor':   raw_coef_norm.get('nut_log_floor', 1e-6),
    }


class DatasetCached:
    def __init__(self, cache_dir, split='train', task='full',
                 pn_level='full', nut_encoding='linear',
                 data_dir=None, names=None):
        if pn_level not in _VALID_LEVELS:
            raise ValueError(
                f"pn_level must be one of {_VALID_LEVELS}, got {pn_level!r}")
        if nut_encoding not in _VALID_ENCODINGS:
            raise ValueError(
                f"nut_encoding must be one of {_VALID_ENCODINGS}")
        if not osp.isdir(cache_dir):
            raise RuntimeError(
                f"cache_dir {cache_dir!r} does not exist. "
                f"Run preprocess.py first.")

        self.cache_dir = cache_dir
        self.split = split
        self.task = task
        self.pn_level = pn_level
        self.nut_encoding = nut_encoding
        self.base_dir = osp.join(cache_dir, 'base')
        self.pn_dir = osp.join(cache_dir, f'pn_{pn_level}')

        if names is None:
            if data_dir is None:
                raise ValueError(
                    "Pass either `names` or `data_dir` (to read manifest).")
            with open(osp.join(data_dir, 'manifest.json')) as f:
                manifest = json.load(f)
            names = manifest[_manifest_key(task, split)]
        self.names = list(names)

        missing = []
        self.data_list = []
        for cname in tqdm(self.names,
                          desc=f"[cache:{split}/{pn_level}/{nut_encoding}] "
                               f"{len(self.names)} cases"):
            data = self._load_one_case(cname, missing)
            if data is not None:
                self.data_list.append(data)
        if missing:
            head = ", ".join(missing[:5])
            raise RuntimeError(
                f"{len(missing)} cases missing from cache (first 5: {head}).")

    def _load_one_case(self, cname, missing):
        bpath = osp.join(self.base_dir, cname + '.pt')
        ppath = osp.join(self.pn_dir, cname + '.pt')
        if not (osp.exists(bpath) and osp.exists(ppath)):
            missing.append(cname)
            return None
        base = torch.load(bpath, map_location='cpu', weights_only=False)
        pn = torch.load(ppath, map_location='cpu', weights_only=False)

        y = base['y_log'] if self.nut_encoding == 'log' else base['y_linear']
        fields = dict(
            case_name=cname,
            pos=base['pos'].float(),
            norm_pos=base['norm_pos'].float(),
            y=y.float(),
            surf=base['surf'].bool(),
            sdf=base['sdf'].float(),
            sdf_grad=base['sdf_grad'].float(),
            curv=base['curv'].float(),
            idw_indices=base['idw_indices'].long(),
            idw_weights=base['idw_weights'].float(),
            leaf_idx=base['leaf_idx'].long(),
            leaf_centroids=base['leaf_centroids'].float(),
            leaf_norm_pos=base['leaf_norm_pos'].float(),
            leaf_stats=base['leaf_stats'].float(),
            leaf_sdf=base['leaf_sdf'].float(),
            leaf_sdf_grad=base['leaf_sdf_grad'].float(),
            pn_input=pn['pn_input'].float(),
            pn_mask=pn['pn_mask'].bool(),
            uinf=base['uinf'].float(),
            airfoil_pos=base['airfoil_pos'].float(),
        )
        if 'leaf_did' in base:
            fields['leaf_did'] = base['leaf_did'].float()
        if 'leaf_geo_neighbors' in base:
            fields['leaf_geo_neighbors'] = base['leaf_geo_neighbors'].long()
        return Data(**fields)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __iter__(self):
        return iter(self.data_list)
