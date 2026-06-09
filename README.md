# KDViT — KD-Tree Vision Transformer Surrogate for 2D Airfoil CFD

A ViT-based neural surrogate for the [AirfRANS](https://github.com/Extrality/AirfRANS) benchmark. KD-tree equal-mass partition concentrates representational capacity near the airfoil where the mesh is densest.

The architecture is fixed to the experimentally-best **E3-D2** configuration; this round (Round 3) is an ablation-friendly refactor: real batching, DDP, `torch.compile`, fast/full eval, and lightweight resource monitoring.

```
Mesh points ─► assign to KD-tree leaves (max-spread splits)
                    │
              PointNet over 7-d enriched leaf neighbors ─► L leaf features
                    │  (+ 19-d leaf stats, leaf SDF/∇SDF, leaf_norm_pos,
                    │     optional leaf DID, inlet velocity)
              TreePatchify (group P leaves) ─► L/P tokens
                    │
              ViT (continuous RoPE + U-Net skips + per-block FiLM
                    + R register tokens)
                    │
              TreeUnpatchify ─► L leaf features
                    │
              IDW-4 interpolation + dual-FiLM Fourier MLP ─► [N, 4]
```

## What changed in Round 3

- **Real batching**: forward takes explicit tensor args, all activations carry a leading `B` dim. `B=1` is just `[1,...]` — no separate code path.
- **DDP**: launch with `torchrun --nproc-per-node N`. Rank 0 owns all I/O; data is sharded `[rank::world]`.
- **`torch.compile`** is enabled by default on the training forward path. Eval / curves stay eager (variable N would force recompile).
- **Per-case RoPE precompute** (in-memory, not disk-cached). Pooled-KV is removed from the active path.
- **lr/wd sqrt(k) scaling** on `B_eff = batch_size * world_size`. User-supplied `--lr` / `--weight_decay` are taken verbatim (not scaled).
- **Scheduler fix**: `steps_per_epoch = ceil(per-rank-cases / batch_size)` — cosine in epoch space stays the same shape.
- **Fast vs full eval** (`--full_eval`): default is fast (field MSE only, per-case arrays saved). Full adds Cd/Cl + SDF binning + scatter.
- **Per-case error histograms** (vol/surf MSE, vs AoA) for every run.
- **Background resource sampler** (GPU util/mem, CPU util/RSS) → CSV + 2×2 system plot.
- **OOM/perf fixes**: SWA window cleared before eval; training dataset deleted (not moved back to CPU).
- **Equivalence tests** in `tests/test_batching.py` (golden + B=1 vs B=2).

## Config defaults

| | default |
|---|---|
| `latent_dim` | 256 |
| `num_heads` | derived: D=256→8, D=384→6 |
| `ffn_hidden` | 4·D |
| `register_tokens` | 4 |
| `pool_kv_factor` | 0 (removed from active path) |
| `gqa_kv_heads` | 0 (dormant) |
| `compile` | true |
| `full_eval` | false (fast eval) |

## Setup

```bash
pip install -r requirements.txt        # adds psutil + nvidia-ml-py
# Or: bash scripts/setup_env.sh
```

Download [AirfRANS](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip), unpack to `data/Dataset/`.

## Preprocessing

```bash
python -m models.preprocess \
    --data_dir data/Dataset --task full \
    --n_leaves 4096 --output_dir cache_4096 --workers 8
```

## Training — single GPU

```bash
python run.py --cache_dir cache_4096 --data_dir data/Dataset \
    --batch_size 2 --latent_dim 256 --num_layers 6
```

## Training — 2 GPUs (DDP)

```bash
torchrun --nproc-per-node 2 run.py \
    --cache_dir cache_4096 --data_dir data/Dataset \
    --batch_size 2 --latent_dim 256 --num_layers 6
```

`B_eff = batch_size * world_size = 4` in the 2-GPU example above, so lr is auto-scaled to `3e-4 * sqrt(4) = 6e-4` and wd to `0.01 * sqrt(4) = 0.02`. Pass `--lr` or `--weight_decay` to disable scaling for that one knob.

## Tests

```bash
pytest tests/test_batching.py -v
```

The golden file regenerates on first run. To validate a behavior-preserving refactor: delete `tests/golden_b1.pt`, run the old code's test once, then run the new code's test — drift will fail loudly.

## Acknowledgement

Built on [AirfRANS](https://github.com/Extrality/AirfRANS) (Bonnet et al., NeurIPS 2022).
