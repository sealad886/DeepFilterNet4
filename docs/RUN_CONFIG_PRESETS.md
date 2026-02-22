# Run-Config Default Presets

Ready-to-use hardware profiles for `train_dynamic.py` on Apple Silicon.
Each preset combines batch size, data-pipeline tuning, and sync-barrier
settings from the [Data Pipeline Tuning](DATA_PIPELINE_TUNING.md) and
[Sync Barrier Policy](SYNC_BARRIER_POLICY.md) guides into a single TOML file.

## Preset Table

| Preset | Hardware | `batch_size` | `eval_frequency` | `num_workers` | `prefetch_size` | `sync_mode` | `use_mlx_data` | `fp16` | Notes |
|--------|----------|-------------|-------------------|---------------|-----------------|-------------|----------------|--------|-------|
| **entry** | M1/M2, ≤16 GB | 2 | 10 | 2 | 4 | normal | false | auto | Conservative for memory |
| **pro** | M1–M3 Pro | 4 | 10 | 4 | 8 | normal | false | auto | Balanced throughput |
| **max** | M1–M3 Max | 8 | 25 | 6 | 12 | fast | true | auto | High throughput |
| **ultra** | M1–M2 Ultra | 8 | 50 | 8 | 16 | fast | true | auto | Maximum throughput |
| **debug** | Any | 2 | 1 | 2 | 4 | debug | false | off | Full observability |

Preset TOML files live in `schemas/presets/`.

## Usage

### Selecting a Preset via CLI

```bash
python -m df_mlx.train_dynamic --preset pro --cache-dir /path/to/cache
```

The preset is loaded as the base config.  Any `--run-config` TOML and
explicit CLI flags override the preset values with the standard
precedence chain:

```
defaults  <  preset  <  run-config TOML  <  CLI flags
```

### Combining a Preset with a Run-Config

```bash
python -m df_mlx.train_dynamic \
    --preset max \
    --run-config my_overrides.toml \
    --batch-size 12
```

Here `max` supplies the base, `my_overrides.toml` overrides any keys it
sets, and `--batch-size 12` wins over both.

### Overriding Individual Values

Any CLI flag or TOML key can override the preset:

```bash
# Start from "pro" but bump workers
python -m df_mlx.train_dynamic --preset pro --num-workers 6
```

## Override Safety Guide

### Safe to Change

| Setting | Why |
|---------|-----|
| `batch_size` | Adjust for available memory; larger batches improve GPU utilisation |
| `num_workers` | Tune to match I/O vs compute balance on your workload |
| `prefetch_size` | Increase to ~3× `num_workers` if GPU stalls are visible |
| `eval_frequency` | Trade logging granularity for throughput |
| `fp16` | Override to `false` if you hit precision issues |

### Change with Caution

| Setting | Risk |
|---------|------|
| `sync_mode` | Changing to `"fast"` hides per-step loss; `"debug"` can halve throughput |
| `use_mlx_data` | Requires the `mlx-data` package; entry devices may OOM from its higher RSS |
| `batch_size > 4` on entry | Likely triggers OOM on ≤16 GB devices |

## Compatibility Notes

1. **Entry devices may OOM with `batch_size > 4`** — the M1/M2 base chips
   share unified memory between CPU and GPU.  Leave headroom for the OS and
   other apps.

2. **`use_mlx_data = true` requires `mlx-data`** — install with
   `pip install mlx-data`.  If the package is missing, training falls back to
   `PrefetchDataLoader` automatically.

3. **`fp16 = "auto"` falls back to FP32** on hardware that doesn't expose
   half-precision matrix operations (unlikely on any Apple Silicon, but
   future-proofed).

4. **GAN training benefits from `sync_mode = "normal"`** even on Max/Ultra.
   The discriminator update step interleaves `mx.eval()` calls, so the
   throughput gain from `fast` mode is marginal while `normal` gives better
   loss visibility during the sensitive GAN ramp-up phase.

5. **The `debug` preset disables FP16** (`fp16 = false`) to eliminate
   precision-related noise during debugging.  Switch back to `"auto"` for
   production runs.

## Rationale

The profiles originate from benchmarking described in
[DATA_PIPELINE_TUNING.md](DATA_PIPELINE_TUNING.md) (worker/prefetch
saturation curves) and [SYNC_BARRIER_POLICY.md](SYNC_BARRIER_POLICY.md)
(eval-frequency impact on throughput).  The `batch_size` values are
chosen to keep peak RSS well below 80 % of total unified memory for each
hardware class.
