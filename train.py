"""Training loop + post-training evaluation (E3-D2, Round 3).

Round 3 additions:
  * Real batching via utils.batch.make_train_batch.
  * DDP-aware (rank 0 owns all I/O; per-rank data sharding).
  * Scheduler counts true optimizer steps per epoch (per-rank).
  * Fused AdamW, optional torch.compile of training forward path.
  * Fast/full eval gate, per-case MSE arrays saved to JSON.
  * SWA window cleared before eval (OOM fix); training dataset deleted
    rather than moved back to CPU.
  * Lightweight background resource sampler (GPU + CPU + RSS).
"""

import gc
import glob
import json
import math
import os
import os.path as osp
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_N_TORCH_THREADS = int(os.environ.get(
    'TORCH_NUM_THREADS',
    str(max(1, min(8, (os.cpu_count() or 8) // 2)))
))
os.environ.setdefault('OMP_NUM_THREADS', str(_N_TORCH_THREADS))
os.environ.setdefault('MKL_NUM_THREADS', str(_N_TORCH_THREADS))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
torch.set_num_threads(_N_TORCH_THREADS)
import torch.distributed as dist

from tqdm import tqdm

from dataset.dataset_cached import DatasetCached
from utils.batch import (attach_rope, case_to_model_inputs,
                         make_train_batch, _unwrap_to_raw)
from utils.resource_monitor import ResourceMonitor

LOSS_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
_BG_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix='bg')

_EVAL_DEVICE_FIELDS = [
    'pos', 'norm_pos', 'y', 'surf', 'sdf', 'sdf_grad',
    'idw_indices', 'idw_weights', 'uinf', 'airfoil_pos',
    'leaf_centroids', 'leaf_norm_pos', 'leaf_stats', 'leaf_sdf',
    'leaf_sdf_grad', 'leaf_did',
    'pn_input', 'pn_mask',
    'rope_cos_x', 'rope_sin_x', 'rope_cos_y', 'rope_sin_y',
]
_TRAIN_DEVICE_FIELDS = [
    k for k in _EVAL_DEVICE_FIELDS if k not in ('pos', 'airfoil_pos')
]

_SDF_BIN_FRACTIONS = (0.0, 2.0 / 14.0, 6.0 / 14.0, 1.0)
_SDF_BIN_LABELS = ('near-wall', 'mid-field', 'far-field')

LIGHT_CKPT_INTERVAL = 10
LIGHT_CKPT_INTERVAL_SWA = 5
MAX_FULL_CKPTS = 3


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _is_dist() else 0


def _world():
    return dist.get_world_size() if _is_dist() else 1


def _is_main():
    return _rank() == 0


def _barrier():
    if _is_dist():
        dist.barrier()


# ---------------------------------------------------------------------------
# Dataset I/O helpers
# ---------------------------------------------------------------------------

def _move_dataset_to_device(dataset_list, device, non_blocking=True,
                             fields=None):
    if fields is None:
        fields = _EVAL_DEVICE_FIELDS
    if not dataset_list:
        return dataset_list
    for data in dataset_list:
        for k in fields:
            t = getattr(data, k, None)
            if isinstance(t, torch.Tensor) and t.device != torch.device(device):
                setattr(data, k, t.to(device, non_blocking=non_blocking))
    return dataset_list


def _pin_dataset(dataset_list, fields=None):
    if fields is None:
        fields = _EVAL_DEVICE_FIELDS
    if not dataset_list or not torch.cuda.is_available():
        return dataset_list
    for data in dataset_list:
        for k in fields:
            t = getattr(data, k, None)
            if (isinstance(t, torch.Tensor) and t.device.type == 'cpu'
                    and not t.is_pinned()):
                try:
                    setattr(data, k, t.pin_memory())
                except Exception:
                    pass
    return dataset_list


def _case_to_device(data, device, fields=None):
    """Return a shallow object with each tensor field moved to device."""
    if fields is None:
        fields = _EVAL_DEVICE_FIELDS
    from torch_geometric.data import Data
    out = Data()
    for k in fields:
        t = getattr(data, k, None)
        if isinstance(t, torch.Tensor):
            setattr(out, k, t.to(device, non_blocking=True))
    if getattr(data, 'case_name', None) is not None:
        out.case_name = data.case_name
    return out


# ---------------------------------------------------------------------------
# JSON / logging helpers
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return super().default(obj)


def _append_log_line(log_path, event_dict):
    with open(log_path, 'a') as f:
        f.write(json.dumps(event_dict, cls=NumpyEncoder) + '\n')


def _to_report_space(x, coef_norm):
    if coef_norm.get('nut_encoding') != 'log':
        return x
    device = x.device
    mi3 = coef_norm['mean_out'][3].to(device)
    si3 = coef_norm['std_out'][3].to(device)
    mr3 = coef_norm['mean_out_report'][3].to(device)
    sr3 = coef_norm['std_out_report'][3].to(device)
    out = x.clone()
    log_phys = x[..., 3] * si3 + mi3
    lin_phys = torch.exp(log_phys)
    out[..., 3] = (lin_phys - mr3) / (sr3 + 1e-8)
    return out


def _snapshot_state_dict_cpu(model):
    raw = _unwrap_to_raw(model)
    return {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}


def _make_swa_state_dict(window):
    assert len(window) > 0
    keys = list(window[0].keys())
    n = float(len(window))
    avg = {}
    for k in keys:
        first = window[0][k]
        if first.is_floating_point():
            acc = torch.zeros(first.shape, dtype=torch.float32)
            for sd in window:
                acc.add_(sd[k].to(torch.float32))
            acc.div_(n)
            avg[k] = acc.to(first.dtype)
        else:
            avg[k] = window[-1][k].clone()
    return avg


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _ckpt_dir(run_dir):
    d = osp.join(run_dir, 'checkpoints')
    os.makedirs(d, exist_ok=True)
    return d


def _light_ckpt_path(run_dir, epoch):
    return osp.join(_ckpt_dir(run_dir), f'epoch_{epoch:04d}.pt')


def _full_ckpt_path(run_dir, epoch):
    return osp.join(run_dir, f'full_ckpt_e{epoch:04d}.pt')


def _full_ckpt_glob(run_dir):
    return osp.join(run_dir, 'full_ckpt_e*.pt')


def _should_save_lightweight(epoch, total_epochs, swa_max):
    in_swa = epoch > (total_epochs - swa_max)
    interval = LIGHT_CKPT_INTERVAL_SWA if in_swa else LIGHT_CKPT_INTERVAL
    return (epoch % interval == 0) or (epoch == total_epochs)


def _save_lightweight(model, epoch, run_dir):
    torch.save(_snapshot_state_dict_cpu(model),
               _light_ckpt_path(run_dir, epoch))


def _save_full_checkpoint(model, optimizer, scheduler, epoch, run_dir,
                          keep=MAX_FULL_CKPTS):
    raw = _unwrap_to_raw(model)
    full = {
        'epoch': epoch,
        'model': {k: v.detach().cpu().clone()
                  for k, v in raw.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        full['rng_cuda'] = torch.cuda.get_rng_state_all()
    path = _full_ckpt_path(run_dir, epoch)
    tmp = path + '.tmp'
    torch.save(full, tmp)
    os.replace(tmp, path)
    fulls = sorted(glob.glob(_full_ckpt_glob(run_dir)))
    for f in fulls[:-keep]:
        try:
            os.remove(f)
        except Exception:
            pass


def _try_resume(model, optimizer, scheduler, run_dir, device):
    """All ranks must call this (load weights on every rank)."""
    fulls = sorted(glob.glob(_full_ckpt_glob(run_dir)))
    if not fulls:
        return 0
    path = fulls[-1]
    if _is_main():
        print(f'[resume] loading {path}', flush=True)
    state = torch.load(path, map_location='cpu', weights_only=False)
    raw = _unwrap_to_raw(model)
    raw.load_state_dict({k: v.to(device) for k, v in state['model'].items()})
    optimizer.load_state_dict(state['optimizer'])
    for s in optimizer.state.values():
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                s[k] = v.to(device)
    scheduler.load_state_dict(state['scheduler'])
    try:
        torch.set_rng_state(state['rng_torch'])
        if torch.cuda.is_available() and 'rng_cuda' in state:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
    except Exception:
        pass
    return int(state['epoch'])


# ---------------------------------------------------------------------------
# Inner training step (one epoch)
# ---------------------------------------------------------------------------

def _train_one_epoch(device, model, train_dataset, train_indices,
                      batch_size, subsampling, optimizer, scheduler,
                      max_grad_norm, loss_weights_tensor, coef_norm,
                      epoch_desc, data_on_gpu, use_did):
    model.train()
    is_log = ((coef_norm is not None)
              and (coef_norm.get('nut_encoding') == 'log'))

    avg_lsv_rep = torch.zeros(4, device=device)
    avg_lvv_rep = torch.zeros(4, device=device)
    avg_ls_rep = torch.zeros((), device=device)
    avg_lv_rep = torch.zeros((), device=device)
    grad_norm_sum_t = torch.zeros((), device=device)
    grad_norm_max_t = torch.zeros((), device=device)
    n_iter = 0
    use_amp = torch.cuda.is_available()

    w_norm = loss_weights_tensor.sum()

    # Chunk indices into batches of size `batch_size`.
    chunks = [train_indices[i:i + batch_size]
              for i in range(0, len(train_indices), batch_size)]

    pbar = tqdm(chunks, desc=epoch_desc, leave=False, mininterval=5.0,
                 disable=not _is_main())
    for chunk in pbar:
        cases = [train_dataset[i] for i in chunk]
        if not data_on_gpu:
            cases = [_case_to_device(c, device, fields=_TRAIN_DEVICE_FIELDS)
                     for c in cases]

        inputs, targets, surf = make_train_batch(
            cases, subsampling, device, use_did)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=use_amp):
            out = model(**inputs)
        out = out.float()                                        # [B, S, 4]

        surf_mask = surf.float().unsqueeze(-1)                   # [B, S, 1]
        vol_mask = 1.0 - surf_mask
        surf_count = surf_mask.sum(dim=1).clamp(min=1.0)         # [B, 1]
        vol_count = vol_mask.sum(dim=1).clamp(min=1.0)

        err_int = (out - targets) ** 2                            # [B, S, 4]
        lsv_int = (err_int * surf_mask).sum(dim=1) / surf_count   # [B, 4]
        lvv_int = (err_int * vol_mask).sum(dim=1) / vol_count

        lsw_b = (lsv_int * loss_weights_tensor).sum(dim=-1) / w_norm  # [B]
        lvw_b = (lvv_int * loss_weights_tensor).sum(dim=-1) / w_norm
        loss = (lvw_b + lsw_b).mean()                             # scalar
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm)
        grad_norm_sum_t = grad_norm_sum_t + grad_norm.detach()
        grad_norm_max_t = torch.maximum(grad_norm_max_t, grad_norm.detach())

        optimizer.step()
        scheduler.step()

        if is_log:
            out_rep = _to_report_space(out, coef_norm)
            tgt_rep = _to_report_space(targets, coef_norm)
            err_rep = (out_rep - tgt_rep) ** 2
            lsv_rep = (err_rep * surf_mask).sum(dim=1) / surf_count
            lvv_rep = (err_rep * vol_mask).sum(dim=1) / vol_count
        else:
            lsv_rep = lsv_int
            lvv_rep = lvv_int

        # Average across the batch for logging.
        avg_lsv_rep += lsv_rep.mean(dim=0).detach()
        avg_lvv_rep += lvv_rep.mean(dim=0).detach()
        avg_ls_rep += lsv_rep.mean().detach()
        avg_lv_rep += lvv_rep.mean().detach()
        n_iter += 1

    n = float(max(n_iter, 1))
    return {
        'loss_surf_var':  (avg_lsv_rep / n).cpu().numpy(),
        'loss_vol_var':   (avg_lvv_rep / n).cpu().numpy(),
        'loss_surf':      (avg_ls_rep / n).item(),
        'loss_vol':       (avg_lv_rep / n).item(),
        'grad_norm_avg':  (grad_norm_sum_t / n).item(),
        'grad_norm_max':  grad_norm_max_t.item(),
    }


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _save_cdcl_scatter(eval_dir, cd_true, cd_pred, cl_true, cl_pred):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for ax, true, pred, label in [
            (ax1, cd_true, cd_pred, 'Cd'),
            (ax2, cl_true, cl_pred, 'Cl'),
        ]:
            ax.scatter(true, pred, s=8, alpha=0.5)
            lo = min(true.min(), pred.min())
            hi = max(true.max(), pred.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    'r--', linewidth=1, label='y=x')
            ax.set_xlabel(f'True {label}'); ax.set_ylabel(f'Pred {label}')
            ax.set_title(label); ax.legend(); ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(osp.join(eval_dir, 'cdcl_scatter.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[WARN] Cd/Cl scatter plot failed: {e}')


def _save_per_case_mse_histograms(eval_dir, per_case_vol, per_case_surf,
                                    case_aoas=None):
    """Histograms (log x-axis) of per-case vol & surf MSE, with mean+median
    annotated. Long-tailed distributions are expected.
    """
    try:
        vol = np.asarray(per_case_vol)
        surf = np.asarray(per_case_surf)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, arr, label in [(axes[0], vol, 'volume'),
                                (axes[1], surf, 'surface')]:
            if arr.size == 0:
                continue
            arr_pos = arr[arr > 0]
            if arr_pos.size:
                bins = np.logspace(np.log10(arr_pos.min()),
                                    np.log10(arr_pos.max() + 1e-12), 40)
            else:
                bins = 40
            ax.hist(arr, bins=bins, color='steelblue', edgecolor='none')
            ax.set_xscale('log')
            m = float(np.mean(arr))
            med = float(np.median(arr))
            ax.axvline(m, ls='--', color='red', label=f'mean={m:.3g}')
            ax.axvline(med, ls=':', color='black', label=f'median={med:.3g}')
            ax.set_xlabel(f'per-case {label} MSE (report space)')
            ax.set_ylabel('count')
            ax.set_title(f'{label} MSE distribution ({arr.size} cases)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(osp.join(eval_dir, 'per_case_mse_hist.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[WARN] per-case MSE histogram failed: {e}')

    if case_aoas is None:
        return
    try:
        aoas = np.asarray(case_aoas)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, arr, label in [(axes[0], np.asarray(per_case_vol), 'volume'),
                                (axes[1], np.asarray(per_case_surf),
                                  'surface')]:
            ax.scatter(aoas, arr, s=10, alpha=0.6)
            ax.set_yscale('log')
            ax.set_xlabel('AoA (deg)')
            ax.set_ylabel(f'{label} MSE')
            ax.set_title(f'{label} MSE vs AoA')
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(osp.join(eval_dir, 'per_case_mse_vs_aoa.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[WARN] per-case MSE vs AoA failed: {e}')


def _save_sdf_binned_histograms(eval_dir, sdf_abs_all, mean_mse_all,
                                 sdf_max=None, n_bins_hist=60):
    if sdf_max is None:
        sdf_max = float(sdf_abs_all.max()) if sdf_abs_all.size else 0.0
    edges = [f * sdf_max for f in _SDF_BIN_FRACTIONS]
    bin_data, summary = [], {}
    for i, label in enumerate(_SDF_BIN_LABELS):
        lo, hi = edges[i], edges[i + 1]
        if i < len(_SDF_BIN_LABELS) - 1:
            mask = (sdf_abs_all >= lo) & (sdf_abs_all < hi)
        else:
            mask = (sdf_abs_all >= lo) & (sdf_abs_all <= hi)
        vals = mean_mse_all[mask]
        bin_data.append((label, lo, hi, vals))
        summary[f'bin{i}_{label.replace("-", "_")}'] = {
            'sdf_range': [float(lo), float(hi)],
            'n_points': int(vals.size),
            'mean_mse': float(vals.mean()) if vals.size else 0.0,
        }
    summary['sdf_max'] = float(sdf_max)
    if mean_mse_all.size:
        x_hi = float(np.percentile(mean_mse_all, 99.0))
    else:
        x_hi = 1.0
    if x_hi <= 0:
        x_hi = 1.0
    hist_edges = np.linspace(0.0, x_hi, n_bins_hist + 1)
    counts_per_bin = []
    for _, _, _, vals in bin_data:
        if vals.size:
            clipped = np.clip(vals, 0.0, x_hi)
            counts, _ = np.histogram(clipped, bins=hist_edges)
        else:
            counts = np.zeros(n_bins_hist, dtype=np.int64)
        counts_per_bin.append(counts)
    y_hi = max((c.max() for c in counts_per_bin if c.size), default=1)
    y_hi = max(int(y_hi * 1.05), 1)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                                  sharex=True, sharey=True)
        for ax, (label, lo, hi, vals), counts in zip(
                axes, bin_data, counts_per_bin):
            ax.bar(hist_edges[:-1], counts,
                   width=np.diff(hist_edges), align='edge',
                   color='steelblue', edgecolor='none')
            ax.set_xlim(0.0, x_hi)
            ax.set_ylim(0.0, y_hi)
            mean_str = (f'mean MSE = {vals.mean():.4g}'
                        if vals.size else 'no points')
            ax.set_title(f'{label}\nSDF in [{lo:.3g}, {hi:.3g}]\n'
                         f'{vals.size:,} pts, {mean_str}', fontsize=10)
            ax.set_xlabel('per-point 4-channel mean MSE')
        axes[0].set_ylabel('point count')
        fig.suptitle(
            'Per-point error histograms by |SDF| bin (report-space MSE)',
            fontsize=12)
        fig.tight_layout()
        fig.savefig(osp.join(eval_dir, 'sdf_error_histograms.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[WARN] SDF histogram plot failed: {e}')
    return summary


def _format_eval_summary(s):
    lines = ['=' * 72, 'Evaluation summary',
             f"mode         : {s.get('mode', '?')}",
             f"checkpoint   : {s.get('checkpoint', 'swa_model.pt')}",
             f"model        : {s.get('model_name', '?')}",
             f"n_params     : {s.get('n_params', '?')}",
             f"test cases   : {s.get('n_test_cases', '?')}",
             f"nut_encoding : {s.get('nut_encoding', '?')}",
             '=' * 72]
    if 'cd' in s:
        for name, key in (('Cd (drag)', 'cd'), ('Cl (lift)', 'cl')):
            d = s[key]
            lines.append(f"\n== {name} ==")
            lines.append(f"  Spearman rho       : {d['spearman_rho']:.4f}")
            lines.append(f"  MSE                : {d['mse']:.6e}")
            lines.append(f"  Median |rel err|   : {d['median_rel_err']*100:.2f}%")
            lines.append(f"  Mean   |rel err|   : {d['mean_rel_err']*100:.2f}%")
    fm = s['field_mse']
    lines.append(f"\n== Field MSE (report space) ==")
    lines.append(f"  Vol  mean +/- std : {fm['vol_mse_mean']:.6f} +/- {fm['vol_mse_std']:.6f}")
    lines.append(f"  Vol  median       : {fm['vol_mse_median']:.6f}")
    lines.append(f"  Surf mean +/- std : {fm['surf_mse_mean']:.6f} +/- {fm['surf_mse_std']:.6f}")
    lines.append(f"  Surf median       : {fm['surf_mse_median']:.6f}")
    lines.append(f"\n== Compute ==")
    for k, lab in (('peak_gpu_mem_used_gib', 'peak GPU mem'),
                    ('peak_cpu_rss_gib', 'peak CPU RSS'),
                    ('mean_train_gpu_util_pct', 'mean train GPU util'),
                    ('gpu_peak_training_gib', 'GPU peak (torch, train)'),
                    ('gpu_peak_eval_gib', 'GPU peak (torch, eval)')):
        v = s.get(k)
        if v is not None:
            unit = '%' if 'util' in k else ' GiB'
            lines.append(f"  {lab:<22}: {v:.2f}{unit}")
    if s.get('flops') is not None:
        lines.append(f"  FLOPs (1 sample)      : {s['flops']:,.0f}")
    if s.get('inference_time'):
        lat = s['inference_time']
        lines.append(f"  Inference time        : {lat['mean_s']:.3f} +/- {lat['std_s']:.3f} s/case")
    lines.append('=' * 72)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Fast eval: field MSE only
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_fast_eval(model, config, coef_norm, run_dir, device, kdtree):
    eval_dir = osp.join(run_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    data_cfg = config['data']
    cache_dir = data_cfg['cache_dir']
    data_dir = data_cfg['data_dir']
    task = data_cfg['task']
    pn_level = data_cfg['pn_level']
    nut_enc = coef_norm.get('nut_encoding', 'linear')
    data_on_gpu = bool(config['training'].get('data_on_gpu', True))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n[EVAL/fast] loading test dataset...")
    test_dataset = DatasetCached(
        cache_dir, split='test', task=task, pn_level=pn_level,
        nut_encoding=nut_enc, data_dir=data_dir)
    raw = _unwrap_to_raw(model)
    attach_rope(test_dataset, raw)
    if torch.cuda.is_available():
        if data_on_gpu:
            _move_dataset_to_device(test_dataset, device)
        else:
            _pin_dataset(test_dataset)

    per_case_vol, per_case_surf, case_aoas = [], [], []
    times = []
    raw.eval()
    for i, data in enumerate(test_dataset):
        if not data_on_gpu:
            data = _case_to_device(data, device)
        inputs = case_to_model_inputs(data, raw.use_did)
        t0 = time.time()
        out = raw(**inputs).squeeze(0).float()
        times.append(time.time() - t0)
        if nut_enc == 'log':
            out_rep = _to_report_space(out, coef_norm)
            tgt_rep = _to_report_space(data.y, coef_norm)
        else:
            out_rep = out
            tgt_rep = data.y
        se = (out_rep - tgt_rep) ** 2
        surf = data.surf
        per_case_surf.append(
            float(se[surf].mean().item()) if surf.any() else float('nan'))
        per_case_vol.append(
            float(se[~surf].mean().item()) if (~surf).any() else float('nan'))
        cname = getattr(data, 'case_name',
                         test_dataset.names[i] if i < len(test_dataset.names)
                         else '')
        try:
            case_aoas.append(float(cname.split('_')[3]))
        except Exception:
            case_aoas.append(float('nan'))
        if (i + 1) % 25 == 0:
            print(f"  [EVAL/fast] {i+1}/{len(test_dataset)}")

    per_case_vol = np.asarray(per_case_vol, dtype=np.float64)
    per_case_surf = np.asarray(per_case_surf, dtype=np.float64)

    fm = {
        'vol_mse_mean':   float(np.nanmean(per_case_vol)),
        'vol_mse_std':    float(np.nanstd(per_case_vol)),
        'vol_mse_median': float(np.nanmedian(per_case_vol)),
        'surf_mse_mean':  float(np.nanmean(per_case_surf)),
        'surf_mse_std':   float(np.nanstd(per_case_surf)),
        'surf_mse_median': float(np.nanmedian(per_case_surf)),
        'per_case_vol':   per_case_vol.tolist(),
        'per_case_surf':  per_case_surf.tolist(),
        'case_names':     list(test_dataset.names),
    }

    _save_per_case_mse_histograms(
        eval_dir, per_case_vol, per_case_surf, case_aoas=case_aoas)

    gpu_eval_peak = (float(torch.cuda.max_memory_allocated() / 2**30)
                     if torch.cuda.is_available() else None)
    n_params = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    summary = {
        'mode': 'fast',
        'checkpoint': 'swa_model.pt', 'task': task,
        'model_name': getattr(raw, '__name__', 'KDViT'),
        'n_params': int(n_params),
        'n_test_cases': len(test_dataset.names),
        'nut_encoding': nut_enc,
        'field_mse': fm,
        'gpu_peak_eval_gib': gpu_eval_peak,
        'inference_time': {
            'mean_s': float(np.mean(times)) if times else 0.0,
            'std_s':  float(np.std(times)) if times else 0.0,
            'total_s': float(np.sum(times)) if times else 0.0,
        },
    }
    with open(osp.join(eval_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    txt = _format_eval_summary(summary)
    with open(osp.join(eval_dir, 'eval_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print('\n' + txt)
    if data_on_gpu and torch.cuda.is_available():
        _move_dataset_to_device(test_dataset, 'cpu')
    return summary


# ---------------------------------------------------------------------------
# Full eval: + pyvista Cd/Cl + SDF binning + scatter
# ---------------------------------------------------------------------------

def _run_full_eval(model, config, coef_norm, run_dir, device, kdtree):
    import scipy.stats as _stats
    import pyvista as _pv
    import utils.metrics as metrics
    import collections

    eval_dir = osp.join(run_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    data_cfg = config['data']
    cache_dir = data_cfg['cache_dir']
    data_dir = data_cfg['data_dir']
    task = data_cfg['task']
    pn_level = data_cfg['pn_level']
    nut_enc = coef_norm.get('nut_encoding', 'linear')
    infer_hparams = {'subsampling': config['training']['subsampling']}
    data_on_gpu = bool(config['training'].get('data_on_gpu', True))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n[EVAL/full] loading test dataset...")
    test_dataset = DatasetCached(
        cache_dir, split='test', task=task, pn_level=pn_level,
        nut_encoding=nut_enc, data_dir=data_dir)
    raw = _unwrap_to_raw(model)
    attach_rope(test_dataset, raw)
    manifest_test = test_dataset.names
    if torch.cuda.is_available():
        if data_on_gpu:
            _move_dataset_to_device(test_dataset, device)
        else:
            _pin_dataset(test_dataset)

    PREFETCH_AHEAD = 8
    def _read_pv_pair(name):
        return (name,
                _pv.read(osp.join(data_dir, name, name + '_internal.vtu')),
                _pv.read(osp.join(data_dir, name, name + '_aerofoil.vtp')))

    pv_futures = collections.deque()
    for j in range(min(PREFETCH_AHEAD, len(manifest_test))):
        pv_futures.append(_BG_EXECUTOR.submit(_read_pv_pair, manifest_test[j]))

    cd_true_list, cd_pred_list, cl_true_list, cl_pred_list = [], [], [], []
    per_case_vol, per_case_surf, case_aoas = [], [], []
    inference_times = []
    cd_rel_err, cl_rel_err = [], []

    sdf_chunks, mse_chunks = [], []
    mean_sdf_phys = coef_norm['mean_sdf'].to(device)
    std_sdf_phys = coef_norm['std_sdf'].to(device)

    raw.eval()
    print(f"[EVAL/full] {len(manifest_test)} cases (nut={nut_enc})...")

    for i, (case_name, data) in enumerate(zip(manifest_test, test_dataset)):
        if not data_on_gpu:
            data = _case_to_device(data, device)
        t0 = time.time()
        outs, _ = metrics.Infer_test(device, [model], [infer_hparams], data,
                                      coef_norm=coef_norm)
        inference_times.append(float(time.time() - t0))
        out_internal = outs[0]
        if nut_enc == 'log':
            out_rep = _to_report_space(out_internal, coef_norm)
            tgt_rep = _to_report_space(data.y, coef_norm)
        else:
            out_rep = out_internal
            tgt_rep = data.y
        per_point_se = (out_rep - tgt_rep) ** 2
        loss_v = per_point_se[~data.surf].mean()
        loss_s = per_point_se[data.surf].mean()
        per_case_vol.append(float(loss_v.item()))
        per_case_surf.append(float(loss_s.item()))

        sdf_abs = (data.sdf * std_sdf_phys + mean_sdf_phys).abs()
        mean_mse = per_point_se.mean(dim=-1)
        sdf_chunks.append(sdf_abs.detach().cpu().numpy().astype(np.float32))
        mse_chunks.append(mean_mse.detach().cpu().numpy().astype(np.float32))

        pv_name, internal, aerofoil = pv_futures.popleft().result()
        assert pv_name == case_name
        next_j = i + PREFETCH_AHEAD
        if next_j < len(manifest_test):
            pv_futures.append(
                _BG_EXECUTOR.submit(_read_pv_pair, manifest_test[next_j]))

        parts = case_name.split('_')
        Uinf = float(parts[2]); angle = float(parts[3])
        case_aoas.append(angle)

        tc = metrics.Compute_coefficients(
            [internal], [aerofoil], data.surf, Uinf, angle, keep_vtk=False)
        cd_true_list.append(float(tc[0][0]))
        cl_true_list.append(float(tc[0][1]))

        intern_pred, aero_pred = metrics.Airfoil_test(
            internal, aerofoil, [out_internal], coef_norm, data.surf)
        pc = metrics.Compute_coefficients(
            intern_pred, aero_pred, data.surf, Uinf, angle, keep_vtk=False)
        cd_pred_list.append(float(pc[0][0]))
        cl_pred_list.append(float(pc[0][1]))

        if (i + 1) % 25 == 0:
            print(f"  [EVAL/full] {i+1}/{len(manifest_test)}")

    cd_true = np.asarray(cd_true_list); cd_pred = np.asarray(cd_pred_list)
    cl_true = np.asarray(cl_true_list); cl_pred = np.asarray(cl_pred_list)

    def _coef_stats(true, pred):
        sp, _ = _stats.spearmanr(true, pred)
        mse = float(np.mean((true - pred) ** 2))
        rel = np.abs(true - pred) / (np.abs(true) + 1e-8)
        return {
            'spearman_rho': float(sp), 'mse': mse,
            'mean_rel_err': float(np.mean(rel)),
            'median_rel_err': float(np.median(rel)),
            'max_rel_err': float(np.max(rel)),
            'rel_err': rel.tolist(),
        }

    cd_stats = _coef_stats(cd_true, cd_pred)
    cl_stats = _coef_stats(cl_true, cl_pred)

    _save_cdcl_scatter(eval_dir, cd_true, cd_pred, cl_true, cl_pred)
    sdf_binned = _save_sdf_binned_histograms(
        eval_dir,
        np.concatenate(sdf_chunks) if sdf_chunks else np.zeros(0, np.float32),
        np.concatenate(mse_chunks) if mse_chunks else np.zeros(0, np.float32))
    _save_per_case_mse_histograms(
        eval_dir, per_case_vol, per_case_surf, case_aoas=case_aoas)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, arr, label in [(axes[0], cd_stats['rel_err'], 'Cd'),
                                (axes[1], cl_stats['rel_err'], 'Cl')]:
            ax.hist(arr, bins=40, color='steelblue', edgecolor='none')
            ax.set_xlabel(f'{label} relative error')
            ax.set_title(f'{label} rel-err distribution')
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(osp.join(eval_dir, 'cdcl_relerr_hist.png'),
                    dpi=130, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[WARN] Cd/Cl rel-err hist failed: {e}')

    per_case_vol = np.asarray(per_case_vol)
    per_case_surf = np.asarray(per_case_surf)
    fm = {
        'vol_mse_mean':   float(np.mean(per_case_vol)),
        'vol_mse_std':    float(np.std(per_case_vol)),
        'vol_mse_median': float(np.median(per_case_vol)),
        'surf_mse_mean':  float(np.mean(per_case_surf)),
        'surf_mse_std':   float(np.std(per_case_surf)),
        'surf_mse_median': float(np.median(per_case_surf)),
        'per_case_vol':   per_case_vol.tolist(),
        'per_case_surf':  per_case_surf.tolist(),
        'case_names':     list(manifest_test),
    }
    gpu_eval_peak = (float(torch.cuda.max_memory_allocated() / 2**30)
                     if torch.cuda.is_available() else None)
    n_params = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    summary = {
        'mode': 'full',
        'checkpoint': 'swa_model.pt', 'task': task,
        'model_name': getattr(raw, '__name__', 'KDViT'),
        'n_params': int(n_params),
        'n_test_cases': len(manifest_test),
        'nut_encoding': nut_enc,
        'cd': cd_stats, 'cl': cl_stats, 'field_mse': fm,
        'gpu_peak_eval_gib': gpu_eval_peak,
        'inference_time': {
            'mean_s': float(np.mean(inference_times)),
            'std_s': float(np.std(inference_times)),
            'total_s': float(np.sum(inference_times)),
        },
        'sdf_binned': sdf_binned,
    }
    with open(osp.join(eval_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    txt = _format_eval_summary(summary)
    with open(osp.join(eval_dir, 'eval_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print('\n' + txt)
    if data_on_gpu and torch.cuda.is_available():
        _move_dataset_to_device(test_dataset, 'cpu')
    return summary


def run_priority1_eval(model, config, coef_norm, run_dir, device, kdtree,
                       loss_history=None):
    full_eval = bool(config.get('evaluation', {}).get('full_eval', False))
    if full_eval:
        summary = _run_full_eval(model, config, coef_norm, run_dir, device,
                                  kdtree)
    else:
        summary = _run_fast_eval(model, config, coef_norm, run_dir, device,
                                  kdtree)
    return summary


# ---------------------------------------------------------------------------
# Train/val curve from lightweight checkpoints
# ---------------------------------------------------------------------------

@torch.no_grad()
def _avg_report_space_mse(model, dataset, coef_norm, device,
                           data_on_gpu=True):
    raw = _unwrap_to_raw(model)
    raw.eval()
    nut_encoding = coef_norm.get('nut_encoding', 'linear')
    vols, surfs = [], []
    for data in dataset:
        if not data_on_gpu:
            data = _case_to_device(data, device)
        inputs = case_to_model_inputs(data, raw.use_did)
        out = raw(**inputs).squeeze(0)
        if nut_encoding == 'log':
            out_rep = _to_report_space(out, coef_norm)
            tgt_rep = _to_report_space(data.y, coef_norm)
        else:
            out_rep = out; tgt_rep = data.y
        se = (out_rep - tgt_rep) ** 2
        surf = data.surf
        if surf.any():
            surfs.append(float(se[surf].mean().item()))
        non = ~surf
        if non.any():
            vols.append(float(se[non].mean().item()))
    vm = float(np.mean(vols)) if vols else float('nan')
    sm = float(np.mean(surfs)) if surfs else float('nan')
    return vm, sm


def _run_train_val_curve(model, config, coef_norm, run_dir, device,
                          val_split, data_on_gpu=True):
    eval_dir = osp.join(run_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    ckpts = sorted(glob.glob(osp.join(run_dir, 'checkpoints', 'epoch_*.pt')))
    if not ckpts:
        print('[curve] no lightweight checkpoints found, skipping')
        return
    data_cfg = config['data']
    cache_dir = data_cfg['cache_dir']
    task = data_cfg['task']
    pn_level = data_cfg['pn_level']
    nut_enc = coef_norm.get('nut_encoding', 'linear')

    print(f"[curve] loading 50 train + 50 val cases...")
    train_eval_ds = DatasetCached(
        cache_dir, split='train', task=task, pn_level=pn_level,
        nut_encoding=nut_enc, names=val_split['train_eval_names'])
    val_eval_ds = DatasetCached(
        cache_dir, split='train', task=task, pn_level=pn_level,
        nut_encoding=nut_enc, names=val_split['val_eval_names'])
    raw = _unwrap_to_raw(model)
    attach_rope(train_eval_ds, raw)
    attach_rope(val_eval_ds, raw)
    if torch.cuda.is_available():
        if data_on_gpu:
            _move_dataset_to_device(train_eval_ds, device)
            _move_dataset_to_device(val_eval_ds, device)
        else:
            _pin_dataset(train_eval_ds)
            _pin_dataset(val_eval_ds)

    epoch_re = re.compile(r'epoch_(\d+)\.pt$')
    rows = []
    for ck in ckpts:
        m = epoch_re.search(ck)
        if not m:
            continue
        ep = int(m.group(1))
        state = torch.load(ck, map_location='cpu', weights_only=False)
        raw.load_state_dict({k: v.to(device) for k, v in state.items()})
        tv, ts = _avg_report_space_mse(raw, train_eval_ds, coef_norm, device,
                                         data_on_gpu=data_on_gpu)
        vv, vs = _avg_report_space_mse(raw, val_eval_ds, coef_norm, device,
                                         data_on_gpu=data_on_gpu)
        rows.append({'epoch': ep,
                     'train_vol_mse': tv, 'train_surf_mse': ts,
                     'val_vol_mse': vv, 'val_surf_mse': vs})
        print(f"[curve] e{ep}: train vol={tv:.4g} surf={ts:.4g} "
              f"| val vol={vv:.4g} surf={vs:.4g}", flush=True)
    rows.sort(key=lambda r: r['epoch'])
    with open(osp.join(eval_dir, 'train_val_curve.json'), 'w') as f:
        json.dump({'rows': rows,
                   'train_eval_names': val_split['train_eval_names'],
                   'val_eval_names': val_split['val_eval_names']},
                  f, indent=2)
    test_vol = test_surf = None
    sp = osp.join(eval_dir, 'eval_summary.json')
    if osp.exists(sp):
        with open(sp) as f:
            s = json.load(f)
        test_vol = float(s['field_mse']['vol_mse_mean'])
        test_surf = float(s['field_mse']['surf_mse_mean'])
    try:
        epochs = [r['epoch'] for r in rows]
        fig, (axv, axs) = plt.subplots(1, 2, figsize=(12, 5))
        axv.plot(epochs, [r['train_vol_mse'] for r in rows],
                 'o-', label='train')
        axv.plot(epochs, [r['val_vol_mse'] for r in rows],
                 's-', label='val')
        if test_vol is not None:
            axv.axhline(test_vol, ls='--', color='red',
                        label=f'SWA test={test_vol:.4g}')
        axv.set_yscale('log'); axv.set_xlabel('epoch')
        axv.set_ylabel('volume MSE (report)')
        axv.set_title('Volume MSE'); axv.legend(); axv.grid(True, alpha=0.3)
        axs.plot(epochs, [r['train_surf_mse'] for r in rows],
                 'o-', label='train')
        axs.plot(epochs, [r['val_surf_mse'] for r in rows],
                 's-', label='val')
        if test_surf is not None:
            axs.axhline(test_surf, ls='--', color='red',
                        label=f'SWA test={test_surf:.4g}')
        axs.set_yscale('log'); axs.set_xlabel('epoch')
        axs.set_ylabel('surface MSE (report)')
        axs.set_title('Surface MSE'); axs.legend(); axs.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(osp.join(eval_dir, 'train_val_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'[curve] plotting failed: {e}')
    if data_on_gpu and torch.cuda.is_available():
        _move_dataset_to_device(train_eval_ds, 'cpu')
        _move_dataset_to_device(val_eval_ds, 'cpu')
    print(f'[curve] done -> {eval_dir}')


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(device, train_dataset, model, config, run_dir, coef_norm, kdtree,
         val_split=None, resource_monitor=None):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    is_main = _is_main()
    world = _world()
    rank = _rank()
    data_on_gpu = bool(config['training'].get('data_on_gpu', True))
    train_cfg = config['training']
    reg_cfg = config['regularization']
    swa_cfg = config.get('swa', {})

    use_did = bool(_unwrap_to_raw(model).use_did)

    if is_main:
        print(f"[train] device={device}, world={world}, rank={rank}, "
              f"data_on_gpu={data_on_gpu}", flush=True)

    # Attach per-case RoPE.
    attach_rope(train_dataset, model)

    if torch.cuda.is_available():
        if data_on_gpu:
            if is_main:
                print(f"[train] moving train_dataset "
                      f"({len(train_dataset)} cases) to {device}...",
                      flush=True)
            _move_dataset_to_device(train_dataset, device,
                                     fields=_TRAIN_DEVICE_FIELDS)
            torch.cuda.synchronize()
            if is_main:
                free_b, total_b = torch.cuda.mem_get_info()
                print(f"[train] VRAM: {(total_b - free_b)/2**30:.2f} / "
                      f"{total_b/2**30:.2f} GiB used", flush=True)
        else:
            _pin_dataset(train_dataset, fields=_TRAIN_DEVICE_FIELDS)
    else:
        if is_main:
            print("[train] WARNING: CUDA not available -> training on CPU",
                  flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg['lr'],
        weight_decay=reg_cfg['weight_decay'],
        fused=torch.cuda.is_available())

    nb_epochs = train_cfg['nb_epochs']
    batch_size = int(train_cfg.get('batch_size', 1))
    swa_max = swa_cfg.get('swa_window', 100)

    # Per-rank optimizer steps per epoch.
    # All ranks see the same case order each epoch (deterministic seed) and
    # take their slice via train_indices[rank::world].
    n_train = len(train_dataset)
    per_rank_cases = math.ceil(n_train / world)
    steps_per_epoch = math.ceil(per_rank_cases / batch_size)
    total_steps = steps_per_epoch * nb_epochs
    warmup_steps = int(0.05 * total_steps)

    def _lr_lambda(step):
        if step < warmup_steps:
            return max(0.10, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=_lr_lambda)

    if is_main:
        print(f"[train] B={batch_size} world={world} "
              f"steps/epoch={steps_per_epoch} total_steps={total_steps} "
              f"warmup={warmup_steps}", flush=True)

    start_epoch = _try_resume(model, optimizer, scheduler, run_dir, device)

    log_path = osp.join(run_dir, 'training_log.jsonl')
    if is_main and start_epoch == 0 and not osp.exists(log_path):
        with open(log_path, 'w'):
            pass
        _append_log_line(log_path, {
            'event': 'training_start', 'ts': time.time(),
            'model': getattr(_unwrap_to_raw(model), '__name__', 'KDViT'),
            'n_params': int(sum(p.numel()
                                 for p in _unwrap_to_raw(model).parameters()
                                 if p.requires_grad)),
            'n_train': n_train, 'world': world,
            'batch_size': batch_size,
            'steps_per_epoch': steps_per_epoch,
        })
    elif is_main:
        _append_log_line(log_path, {
            'event': 'resume', 'ts': time.time(),
            'from_epoch': start_epoch,
        })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loss_weights_tensor = torch.tensor(LOSS_WEIGHTS, device=device,
                                       dtype=torch.float32)
    start_time = time.time()
    epochs_completed = start_epoch
    swa_window = []                                # rank 0 only

    if is_main:
        print(f"[train] entering loop: epoch {start_epoch+1} -> "
              f"{nb_epochs}", flush=True)
    pbar = tqdm(range(start_epoch, nb_epochs), position=0,
                 disable=not is_main)

    for epoch in pbar:
        epoch_t0 = time.time()
        # Deterministic shuffle that's identical across ranks.
        g = torch.Generator()
        g.manual_seed(0xDEADBEEF + epoch)
        perm = torch.randperm(n_train, generator=g).tolist()
        train_indices = perm[rank::world]

        res = _train_one_epoch(
            device, model, train_dataset, train_indices,
            batch_size, train_cfg['subsampling'],
            optimizer, scheduler,
            train_cfg['max_grad_norm'], loss_weights_tensor, coef_norm,
            f'epoch {epoch + 1}/{nb_epochs}', data_on_gpu, use_did)

        epochs_completed = epoch + 1
        loss_vol = res['loss_vol']; loss_surf = res['loss_surf']
        current_lr = optimizer.param_groups[0]['lr']
        if is_main:
            print(f'epoch {epochs_completed}: vol={loss_vol:.6f} '
                  f'surf={loss_surf:.6f} grad={res["grad_norm_avg"]:.3f} '
                  f'lr={current_lr:.6f} ({time.time() - epoch_t0:.1f}s)',
                  flush=True)
            _BG_EXECUTOR.submit(_append_log_line, log_path, {
                'event': 'epoch', 'epoch': epochs_completed,
                'lr': float(current_lr),
                'train_loss_vol': float(loss_vol),
                'train_loss_surf': float(loss_surf),
                'train_loss_vol_per_var': res['loss_vol_var'].tolist(),
                'train_loss_surf_per_var': res['loss_surf_var'].tolist(),
                'grad_norm_avg': float(res['grad_norm_avg']),
                'grad_norm_max': float(res['grad_norm_max']),
                'duration_s': float(time.time() - epoch_t0),
            })

        if is_main:
            swa_window.append(_snapshot_state_dict_cpu(model))
            while len(swa_window) > swa_max:
                swa_window.pop(0)
            if _should_save_lightweight(epochs_completed, nb_epochs, swa_max):
                _save_lightweight(model, epochs_completed, run_dir)
                _save_full_checkpoint(model, optimizer, scheduler,
                                       epochs_completed, run_dir)
            pbar.set_postfix(vol=loss_vol, surf=loss_surf,
                              swa=len(swa_window))
        _barrier()

    time_elapsed = time.time() - start_time
    if is_main:
        print(f'\nTraining complete: {epochs_completed} epochs in '
              f'{time_elapsed:.1f}s')

    gpu_train_alloc = 0.0
    if torch.cuda.is_available():
        gpu_train_alloc = float(torch.cuda.max_memory_allocated() / 2**30)
        if is_main:
            print(f'GPU peak (training): {gpu_train_alloc:.2f} GiB')

    swa_model_path = osp.join(run_dir, 'swa_model.pt')

    # rank-0-only post-training pipeline.
    if not is_main:
        _barrier()
        return model

    if len(swa_window) > 0:
        swa_state = _make_swa_state_dict(swa_window)
        swa_n = len(swa_window)
        print(f'SWA: averaged {swa_n} epoch snapshots')
    else:
        print('WARNING: SWA window empty, using final-epoch weights')
        swa_state = _snapshot_state_dict_cpu(model)
        swa_n = 0
    # Free the SWA window before eval (Round-3 OOM fix).
    swa_window.clear()
    del swa_window
    gc.collect()
    torch.save(swa_state, swa_model_path)

    raw = _unwrap_to_raw(model)
    raw_state_cpu = _snapshot_state_dict_cpu(model)
    raw.load_state_dict({k: v.to(device) for k, v in swa_state.items()})
    raw.eval()

    _append_log_line(log_path, {
        'event': 'training_end', 'epoch': epochs_completed,
        'swa_n': swa_n, 'total_time_s': float(time_elapsed),
        'gpu_peak_training_gib': gpu_train_alloc,
    })

    # Free training dataset (rank 0 only — other ranks already returned).
    del train_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if resource_monitor is not None:
        resource_monitor.set_phase('eval')

    eval_ok = False
    summary = None
    try:
        summary = run_priority1_eval(
            model=raw, config=config, coef_norm=coef_norm,
            run_dir=run_dir, device=device, kdtree=kdtree,
            loss_history={'gpu_peak_training_allocated_gib': gpu_train_alloc})
        eval_ok = True
    except Exception as e:
        import traceback
        print(f'\n[EVAL] FAILED (SWA weights kept): {e}')
        traceback.print_exc()

    if val_split is not None:
        try:
            _run_train_val_curve(
                model=raw, config=config, coef_norm=coef_norm,
                run_dir=run_dir, device=device, val_split=val_split,
                data_on_gpu=data_on_gpu)
        except Exception as e:
            import traceback
            print(f'\n[curve] FAILED: {e}')
            traceback.print_exc()

    # Inject torch-tracked train peak into summary too.
    if summary is not None:
        summary['gpu_peak_training_gib'] = gpu_train_alloc
        with open(osp.join(run_dir, 'eval', 'eval_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

    if eval_ok and osp.exists(swa_model_path):
        os.remove(swa_model_path)
        print(f'[cleanup] removed {swa_model_path}')

    raw.load_state_dict({k: v.to(device) for k, v in raw_state_cpu.items()})
    _barrier()
    return model
