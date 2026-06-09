"""Lightweight background resource sampler (GPU + CPU + RSS).

Writes a CSV every ~3s with one row per sample, and produces a 2x2
system plot at shutdown. Rank-0-only usage is the caller's job.
"""

import csv
import os
import threading
import time

import numpy as np


class ResourceMonitor:
    def __init__(self, csv_path, interval_s=3.0):
        self.csv_path = csv_path
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread = None
        self._phase = 'train'
        self._t0 = None
        self._fh = None
        self._writer = None
        self._nvml_ok = False
        self._nvml_handles = []
        self._psutil = None
        self._proc = None
        try:
            import pynvml
            pynvml.nvmlInit()
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            if visible:
                ids = [int(x) for x in visible.split(',') if x.strip() != '']
            else:
                ids = list(range(pynvml.nvmlDeviceGetCount()))
            self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                                   for i in ids]
            self._pynvml = pynvml
            self._nvml_ok = True
        except Exception as e:
            print(f'[resource] pynvml unavailable: {e}', flush=True)
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            self._proc.cpu_percent(None)
        except Exception as e:
            print(f'[resource] psutil unavailable: {e}', flush=True)

    def set_phase(self, phase):
        self._phase = phase

    def start(self):
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        self._fh = open(self.csv_path, 'w', newline='')
        n_gpu = len(self._nvml_handles)
        headers = ['t_s', 'phase', 'cpu_util_pct', 'cpu_rss_gib']
        for i in range(n_gpu):
            headers += [f'gpu{i}_util_pct', f'gpu{i}_mem_used_gib',
                        f'gpu{i}_mem_total_gib']
        self._writer = csv.writer(self._fh)
        self._writer.writerow(headers)
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        py = self._psutil
        nv = self._pynvml if self._nvml_ok else None
        while not self._stop.is_set():
            try:
                t = time.time() - self._t0
                cpu_util = py.cpu_percent(None) if py else float('nan')
                rss = (self._proc.memory_info().rss / 2**30
                       if self._proc else float('nan'))
                row = [f'{t:.2f}', self._phase,
                       f'{cpu_util:.1f}', f'{rss:.3f}']
                for h in self._nvml_handles:
                    u = nv.nvmlDeviceGetUtilizationRates(h).gpu
                    mem = nv.nvmlDeviceGetMemoryInfo(h)
                    row += [f'{u}', f'{mem.used/2**30:.3f}',
                            f'{mem.total/2**30:.3f}']
                self._writer.writerow(row)
                self._fh.flush()
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._fh is not None:
            self._fh.close()
        if self._nvml_ok:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def summarize_and_plot(self, png_path, summary_out=None):
        """Read the CSV back and emit a 2x2 system plot + summary dict."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f'[resource] matplotlib unavailable: {e}', flush=True)
            return None
        if not os.path.exists(self.csv_path):
            return None
        with open(self.csv_path) as f:
            r = csv.reader(f)
            headers = next(r)
            rows = list(r)
        if not rows:
            return None

        def col(name):
            i = headers.index(name)
            return np.array([
                float(row[i]) if row[i] not in ('', 'nan') else float('nan')
                for row in rows])

        t = col('t_s')
        phase = [row[headers.index('phase')] for row in rows]
        cpu_util = col('cpu_util_pct')
        rss = col('cpu_rss_gib')
        gpu_idxs = [int(h.replace('gpu', '').split('_')[0])
                    for h in headers if h.startswith('gpu') and
                    h.endswith('_util_pct')]
        gpu_utils = {i: col(f'gpu{i}_util_pct') for i in gpu_idxs}
        gpu_mems = {i: col(f'gpu{i}_mem_used_gib') for i in gpu_idxs}

        # Phase boundary (first eval index) for shading.
        eval_t = None
        for ts, ph in zip(t, phase):
            if ph == 'eval':
                eval_t = ts
                break

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        ax = axes[0, 0]
        for i, u in gpu_utils.items():
            ax.plot(t, u, label=f'gpu{i}')
        ax.set_ylabel('GPU util %')
        ax.set_xlabel('t (s)')
        ax.legend(); ax.grid(True, alpha=0.3)
        if eval_t is not None:
            ax.axvline(eval_t, ls='--', color='k', alpha=0.4)

        ax = axes[0, 1]
        for i, m in gpu_mems.items():
            ax.plot(t, m, label=f'gpu{i}')
        ax.set_ylabel('GPU mem used (GiB)')
        ax.set_xlabel('t (s)')
        ax.legend(); ax.grid(True, alpha=0.3)
        if eval_t is not None:
            ax.axvline(eval_t, ls='--', color='k', alpha=0.4)

        ax = axes[1, 0]
        ax.plot(t, cpu_util)
        ax.set_ylabel('CPU util %')
        ax.set_xlabel('t (s)')
        ax.grid(True, alpha=0.3)
        if eval_t is not None:
            ax.axvline(eval_t, ls='--', color='k', alpha=0.4)

        ax = axes[1, 1]
        ax.plot(t, rss)
        peak = float(np.nanmax(rss)) if rss.size else 0.0
        ax.axhline(peak, ls=':', color='r',
                    label=f'peak={peak:.2f} GiB')
        ax.set_ylabel('Process RSS (GiB)')
        ax.set_xlabel('t (s)')
        ax.legend(); ax.grid(True, alpha=0.3)
        if eval_t is not None:
            ax.axvline(eval_t, ls='--', color='k', alpha=0.4)

        fig.tight_layout()
        fig.savefig(png_path, dpi=130, bbox_inches='tight')
        import matplotlib.pyplot as _plt
        _plt.close(fig)

        train_mask = np.array([p == 'train' for p in phase])
        if train_mask.any() and gpu_utils:
            mean_train_util = float(np.nanmean(
                np.stack([gpu_utils[i][train_mask] for i in gpu_idxs])))
        else:
            mean_train_util = None
        peak_gpu_mem = (float(np.nanmax(
            np.stack([gpu_mems[i] for i in gpu_idxs])))
            if gpu_mems else None)
        peak_rss = peak

        summary = {
            'peak_gpu_mem_used_gib': peak_gpu_mem,
            'peak_cpu_rss_gib': peak_rss,
            'mean_train_gpu_util_pct': mean_train_util,
        }
        if summary_out is not None:
            summary_out.update(summary)
        return summary
