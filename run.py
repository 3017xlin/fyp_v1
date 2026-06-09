"""Unified entry point: training and evaluation (E3-D2, Round 3).

Round 3 additions:
  * torchrun-aware DDP (NCCL), rank 0 owns I/O.
  * Auto sqrt(k) scaling of lr and weight_decay on B_eff = B * world,
    unless user explicitly overrode them.
  * num_heads / ffn_hidden derived from latent_dim if unset.
  * torch.compile wraps the training forward path (eval stays eager).
  * --full_eval gate; default fast eval (field MSE only).
  * Background resource monitor (CSV + 2x2 system plot).
"""

import argparse
import json
import math
import os
import os.path as osp
import random
import sys

import torch
import torch.distributed as dist
import yaml

import train as trainer
from dataset.dataset_cached import (DatasetCached, _load_raw_coef_norm,
                                    _project_coef_norm)
from models.model import KDViT
from utils.resource_monitor import ResourceMonitor


VAL_SPLIT_SEED = 12345
VAL_SPLIT_N_VAL = 80
VAL_SPLIT_N_EVAL = 50

LR_ANCHOR_DEFAULT = 3e-4
WD_ANCHOR_DEFAULT = 1e-2


def _load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _derive_arch_defaults(cfg):
    """Fill model.num_heads and model.ffn_hidden from latent_dim if null."""
    m = cfg['model']
    D = int(m['latent_dim'])
    if m.get('num_heads') in (None, 'null'):
        if D == 256:
            m['num_heads'] = 8
        elif D == 384:
            m['num_heads'] = 6
        else:
            raise ValueError(
                f"latent_dim={D} has no default num_heads; pass "
                f"--num_heads explicitly (must give head_dim%4==0).")
    head_dim = D // int(m['num_heads'])
    if head_dim % 4 != 0:
        raise ValueError(
            f"head_dim={head_dim} (D={D}, H={m['num_heads']}) "
            f"must be divisible by 4 for 2-D RoPE.")
    if m.get('ffn_hidden') in (None, 'null'):
        m['ffn_hidden'] = 4 * D


def _apply_sqrt_k_scaling(cfg, user_provided):
    """sqrt(k) scaling for lr and weight_decay (AdamW).

    B_eff = batch_size * world_size. If user passed --lr or --weight_decay,
    that one is taken verbatim (not scaled). The other defaults to
    ``anchor * sqrt(B_eff)``.
    """
    world = int(os.environ.get('WORLD_SIZE', '1'))
    B = int(cfg['training'].get('batch_size', 1))
    B_eff = B * world
    sqrtk = math.sqrt(B_eff)
    lr_anchor = float(cfg.get('scaling', {}).get('lr_anchor',
                                                   LR_ANCHOR_DEFAULT))
    wd_anchor = float(cfg.get('scaling', {}).get('wd_anchor',
                                                   WD_ANCHOR_DEFAULT))
    if not user_provided.get('lr'):
        cfg['training']['lr'] = float(lr_anchor * sqrtk)
    if not user_provided.get('weight_decay'):
        cfg['regularization']['weight_decay'] = float(wd_anchor * sqrtk)
    return B_eff


def _variant_suffix(cfg, args):
    m = cfg['model']
    parts = []
    if m.get('pn_hidden', 32) != 32:
        parts.append(f"H{m['pn_hidden']}")
    if m.get('pn_layers', 2) != 2:
        parts.append(f"PNl{m['pn_layers']}")
    if m.get('no_unet_skip', False):
        parts.append('NoSkip')
    if not m.get('no_patchify', True):
        parts.append('Patch')
    if not m.get('use_did', True):
        parts.append('NoDID')
    if int(m.get('register_tokens', 4)) != 4:
        parts.append(f"Reg{int(m['register_tokens'])}")
    if int(m.get('gqa_kv_heads', 0)) > 0:
        parts.append(f"GQA{int(m['gqa_kv_heads'])}")
    return '_'.join(parts)


def _auto_run_name(cfg, args, n_leaves, cache_dir, B_eff):
    m = cfg['model']
    base = osp.basename(cache_dir.rstrip('/')) or 'cache'
    lr = cfg['training']['lr']
    name = (f"{base}_L{n_leaves}_D{m['latent_dim']}_N{m['num_layers']}"
            f"_B{B_eff}_lr{lr:.2e}")
    suffix = _variant_suffix(cfg, args)
    if suffix:
        name = f"{name}_{suffix}"
    return name


def _apply_cli_overrides(cfg, args):
    mapping = {
        'patch_size':        ('model', 'patch_size'),
        'latent_dim':        ('model', 'latent_dim'),
        'num_layers':        ('model', 'num_layers'),
        'num_heads':         ('model', 'num_heads'),
        'ffn_hidden':        ('model', 'ffn_hidden'),
        'fourier_freqs':     ('model', 'fourier_freqs'),
        'pn_hidden':         ('model', 'pn_hidden'),
        'pn_layers':         ('model', 'pn_layers'),
        'ffn_dropout':       ('regularization', 'ffn_dropout'),
        'decoder_dropout':   ('regularization', 'decoder_dropout'),
        'attn_dropout':      ('regularization', 'attn_dropout'),
        'drop_path_rate':    ('regularization', 'drop_path_rate'),
        'weight_decay':      ('regularization', 'weight_decay'),
        'layerwise_scaling': ('regularization', 'layerwise_scaling'),
        'nut_encoding':      ('training', 'nut_encoding'),
        'nb_epochs':         ('training', 'nb_epochs'),
        'lr':                ('training', 'lr'),
        'subsampling':       ('training', 'subsampling'),
        'data_on_gpu':       ('training', 'data_on_gpu'),
        'compile':           ('training', 'compile'),
        'batch_size':        ('training', 'batch_size'),
        'swa_window':        ('swa', 'swa_window'),
        'cache_dir':         ('data', 'cache_dir'),
        'data_dir':          ('data', 'data_dir'),
        'task':              ('data', 'task'),
        'pn_level':          ('data', 'pn_level'),
        'use_did':           ('model', 'use_did'),
        'no_patchify':       ('model', 'no_patchify'),
        'no_unet_skip':      ('model', 'no_unet_skip'),
        'register_tokens':   ('model', 'register_tokens'),
        'gqa_kv_heads':      ('model', 'gqa_kv_heads'),
        'rope_base':         ('model', 'rope_base'),
        'full_eval':         ('evaluation', 'full_eval'),
    }
    user_provided = {}
    for cli_key, (section, key) in mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            cfg[section][key] = val
            user_provided[cli_key] = True
    return user_provided


def _load_kdtree(cache_dir):
    path = osp.join(cache_dir, 'kdtree.pt')
    if not osp.exists(path):
        print(f"ERROR: {path} not found (run preprocess.py first)")
        sys.exit(1)
    return torch.load(path, map_location='cpu', weights_only=False)


def _build_model(cfg, kdtree):
    m = cfg['model']
    r = cfg['regularization']
    did_bins = int(kdtree.get('did_bins', 8))
    dx = kdtree.get('domain_x_range')
    dy = kdtree.get('domain_y_range')
    domain_x = tuple(dx.tolist()) if dx is not None else (-2.0, 4.0)
    domain_y = tuple(dy.tolist()) if dy is not None else (-1.5, 1.5)
    return KDViT(
        n_leaves=int(kdtree['n_leaves']),
        patch_size=m['patch_size'],
        latent_dim=m['latent_dim'],
        num_layers=m['num_layers'],
        num_heads=m['num_heads'],
        ffn_hidden=m['ffn_hidden'],
        fourier_freqs=m['fourier_freqs'],
        out_dim=m['out_dim'],
        pn_hidden=m.get('pn_hidden', 32),
        pn_dim=m.get('pn_dim', 128),
        pn_layers=int(m.get('pn_layers', 2)),
        dropout=r['ffn_dropout'],
        decoder_dropout=r['decoder_dropout'],
        drop_path_rate=r['drop_path_rate'],
        attn_dropout=r['attn_dropout'],
        layerwise_scaling=r['layerwise_scaling'],
        rope_scale=float(m.get('rope_scale', 32)),
        rope_base=float(m.get('rope_base', 100.0)),
        use_did=bool(m.get('use_did', True)),
        did_bins=did_bins,
        no_patchify=bool(m.get('no_patchify', True)),
        no_unet_skip=bool(m.get('no_unet_skip', False)),
        domain_x=domain_x, domain_y=domain_y,
        register_tokens=int(m.get('register_tokens', 4)),
        pool_kv_factor=int(m.get('pool_kv_factor', 0)),
        gqa_kv_heads=int(m.get('gqa_kv_heads', 0)),
    )


def _load_coef_norm(cache_dir, task, nut_encoding):
    raw = _load_raw_coef_norm(cache_dir, task)
    return _project_coef_norm(raw, nut_encoding)


def _ensure_val_split(data_dir, task, run_dir, is_main,
                       seed=VAL_SPLIT_SEED,
                       n_val=VAL_SPLIT_N_VAL,
                       n_eval=VAL_SPLIT_N_EVAL):
    """Deterministic train/val partition. Only rank 0 writes the file."""
    path = osp.join(run_dir, 'val_split.json')
    if osp.exists(path):
        with open(path) as f:
            return json.load(f)
    with open(osp.join(data_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    key = 'scarce_train' if task == 'scarce' else f'{task}_train'
    all_train = sorted(manifest[key])
    rng = random.Random(seed)
    shuffled = list(all_train); rng.shuffle(shuffled)
    n_v = min(n_val, len(shuffled))
    val_names = sorted(shuffled[:n_v])
    train_names = sorted(shuffled[n_v:])
    rng2 = random.Random(seed + 1)
    train_eval = sorted(rng2.sample(train_names,
                                     min(n_eval, len(train_names))))
    val_eval = sorted(rng2.sample(val_names,
                                   min(n_eval, len(val_names))))
    out = {'seed': seed, 'train_names': train_names, 'val_names': val_names,
           'train_eval_names': train_eval, 'val_eval_names': val_eval}
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
    return out


def _run_eval_only(run_dir):
    config_path = osp.join(run_dir, 'config.json')
    if not osp.exists(config_path):
        print(f"ERROR: {config_path} not found"); sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)
    cache_dir = cfg['data']['cache_dir']
    task = cfg['data']['task']
    nut_enc = cfg['training']['nut_encoding']
    coef_norm = _load_coef_norm(cache_dir, task, nut_enc)
    kdtree = _load_kdtree(cache_dir)
    model = _build_model(cfg, kdtree)
    swa_path = osp.join(run_dir, 'swa_model.pt')
    if not osp.exists(swa_path):
        print(f"ERROR: {swa_path} not found.")
        sys.exit(1)
    state = torch.load(swa_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    trainer.run_priority1_eval(
        model=model, config=cfg, coef_norm=coef_norm,
        run_dir=run_dir, device=device, kdtree=kdtree)
    print("[eval_only] Done.")


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ('true', 't', '1', 'yes', 'y'):
        return True
    if s in ('false', 'f', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Expected bool, got {v!r}')


def parse_args():
    p = argparse.ArgumentParser(description='KDViT (E3-D2): train or evaluate')
    p.add_argument('--cache_dir', type=str, default=None)
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--task', type=str, default=None,
                   choices=['full', 'scarce', 'reynolds', 'aoa'])
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--run_dir', type=str, default=None)
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--patch_size', type=int, default=None)
    p.add_argument('--pn_level', type=str, default=None, choices=['full'])
    p.add_argument('--latent_dim', type=int, default=None)
    p.add_argument('--num_layers', type=int, default=None)
    p.add_argument('--num_heads', type=int, default=None)
    p.add_argument('--ffn_hidden', type=int, default=None)
    p.add_argument('--fourier_freqs', type=int, default=None)
    p.add_argument('--pn_hidden', type=int, default=None)
    p.add_argument('--pn_layers', type=int, default=None)
    p.add_argument('--ffn_dropout', type=float, default=None)
    p.add_argument('--decoder_dropout', type=float, default=None)
    p.add_argument('--attn_dropout', type=float, default=None)
    p.add_argument('--drop_path_rate', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--layerwise_scaling', action='store_true', default=None)
    p.add_argument('--nut_encoding', type=str, default=None,
                   choices=['linear', 'log'])
    p.add_argument('--nb_epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--subsampling', type=int, default=None)
    p.add_argument('--swa_window', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--use_did', dest='use_did', action='store_true',
                   default=None)
    p.add_argument('--no_did', dest='use_did', action='store_false',
                   default=None)
    p.add_argument('--no_patchify', action='store_true', default=None)
    p.add_argument('--no_unet_skip', action='store_true', default=None)
    p.add_argument('--register_tokens', type=int, default=None)
    p.add_argument('--gqa_kv_heads', type=int, default=None)
    p.add_argument('--rope_base', type=float, default=None)
    p.add_argument('--data_on_gpu', type=_str2bool, default=None)
    p.add_argument('--compile', dest='compile', type=_str2bool, default=None,
                   help='torch.compile training forward path')
    p.add_argument('--full_eval', type=_str2bool, default=None,
                   help='full eval (Cd/Cl + SDF bins). default fast eval.')
    return p.parse_args()


def _enable_line_buffering():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def _setup_ddp():
    """Initialize NCCL DDP if torchrun env is present.

    Returns (device, world_size, rank, local_rank, is_main).
    """
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        rank = int(os.environ.get('RANK', '0'))
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        return device, world, rank, local_rank, (rank == 0)
    # Single-process.
    if torch.cuda.is_available():
        device = 'cuda:0'; local_rank = 0
    else:
        device = 'cpu'; local_rank = 0
    return device, 1, 0, local_rank, True


def main():
    _enable_line_buffering()
    args = parse_args()
    cfg = _load_config()
    user_provided = _apply_cli_overrides(cfg, args)
    _derive_arch_defaults(cfg)

    device, world, rank, local_rank, is_main = _setup_ddp()

    if args.eval_only:
        if not is_main:
            return
        run_dir = args.run_dir or osp.join(cfg['data']['output_base'],
                                            args.run_name or '')
        _run_eval_only(run_dir)
        return

    B_eff = _apply_sqrt_k_scaling(cfg, user_provided)

    cache_dir = cfg['data']['cache_dir']
    data_dir = cfg['data']['data_dir']
    task = cfg['data']['task']
    pn_level = cfg['data']['pn_level']
    if cache_dir is None:
        if is_main:
            print("ERROR: --cache_dir is required")
        sys.exit(1)
    nut_enc = cfg['training']['nut_encoding']

    kdtree = _load_kdtree(cache_dir)
    n_leaves = int(kdtree['n_leaves'])
    if is_main:
        print(f"[run] KD-tree: {n_leaves} leaves (cache={cache_dir})")
        print(f"[run] world={world} batch_size="
              f"{cfg['training']['batch_size']} B_eff={B_eff} "
              f"lr={cfg['training']['lr']:.4e} "
              f"wd={cfg['regularization']['weight_decay']:.4e} "
              f"D={cfg['model']['latent_dim']} "
              f"H={cfg['model']['num_heads']} "
              f"FFN={cfg['model']['ffn_hidden']} "
              f"R={cfg['model']['register_tokens']}")

    coef_norm = _load_coef_norm(cache_dir, task, nut_enc)

    run_name = args.run_name or _auto_run_name(
        cfg, args, n_leaves, cache_dir, B_eff)
    run_dir = osp.join(cfg['data']['output_base'], run_name)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        print(f"[run] run_dir = {run_dir}")
        with open(osp.join(run_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
    if world > 1:
        dist.barrier()

    val_split = _ensure_val_split(data_dir, task, run_dir, is_main)
    if is_main:
        print(f"[run] train/val split: "
              f"{len(val_split['train_names'])} train, "
              f"{len(val_split['val_names'])} val "
              f"({len(val_split['train_eval_names'])} + "
              f"{len(val_split['val_eval_names'])} for curves)")

    if is_main:
        print(f"[run] loading train dataset (task={task}, "
              f"pn_level={pn_level}, {nut_enc})...")
    train_dataset = DatasetCached(
        cache_dir, split='train', task=task, pn_level=pn_level,
        nut_encoding=nut_enc, names=val_split['train_names'])
    if is_main:
        print(f"[run] training on {len(train_dataset)} cases")

    model = _build_model(cfg, kdtree).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"[run] model: {model.__name__}, {n_params:,} params "
              f"(use_did={model.use_did}, no_patchify={model.no_patchify})")

    # Wrap in DDP first, then torch.compile (per PyTorch recommendation).
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
        if is_main:
            print(f"[run] DDP wrapped (world={world})")

    do_compile = bool(cfg['training'].get('compile', True)) and \
        torch.cuda.is_available()
    if do_compile:
        try:
            model = torch.compile(model)
            if is_main:
                print("[run] torch.compile enabled (training path)")
        except Exception as e:
            if is_main:
                print(f"[run] torch.compile failed, falling back: {e}")

    # Resource monitor (rank 0 only).
    resource_monitor = None
    if is_main and bool(cfg['training'].get('resource_log', True)):
        resource_monitor = ResourceMonitor(
            csv_path=osp.join(run_dir, 'resource_log.csv'),
            interval_s=3.0)
        resource_monitor.start()
        resource_monitor.set_phase('train')

    try:
        trainer.main(
            device=device,
            train_dataset=train_dataset,
            model=model,
            config=cfg,
            run_dir=run_dir,
            coef_norm=coef_norm,
            kdtree=kdtree,
            val_split=val_split,
            resource_monitor=resource_monitor,
        )
    finally:
        if resource_monitor is not None:
            resource_monitor.stop()
            summary_extra = {}
            resource_monitor.summarize_and_plot(
                png_path=osp.join(run_dir, 'eval', 'resource_log.png'),
                summary_out=summary_extra)
            # Merge into eval_summary.json if it exists.
            sp = osp.join(run_dir, 'eval', 'eval_summary.json')
            if osp.exists(sp):
                try:
                    with open(sp) as f:
                        s = json.load(f)
                    s.update(summary_extra)
                    with open(sp, 'w') as f:
                        json.dump(s, f, indent=2,
                                   cls=trainer.NumpyEncoder)
                except Exception as e:
                    print(f'[resource] summary merge failed: {e}')
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
