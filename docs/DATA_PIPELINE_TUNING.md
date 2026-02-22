# Data Pipeline Saturation Tuning Profiles

Recommended worker/prefetch settings for MLX-based training on Apple Silicon,
tuned per hardware class. These profiles target `DataloaderConfig` in
`df_mlx/run_config.py` and cover both `PrefetchDataLoader` (thread-pool) and
`MLXDataStream` (mlx-data native C++) backends.

## Hardware Profile Table

| Hardware Class | Loader | `num_workers` | `prefetch_size` | Rationale |
|---|---|---|---|---|
| **Entry** (M1/M2, ≤16 GB) | PrefetchDataLoader | 2 | 4 | Limited memory bandwidth; excess workers cause contention on the shared memory bus and raise RSS above safe headroom. |
| **Pro** (M1–M3 Pro, 16–36 GB) | PrefetchDataLoader | 4 | 8 | Balanced: 4 workers saturate NVMe throughput without starving GPU-side memory. Prefetch of 8 keeps the queue full at typical batch latencies. |
| **Max** (M1–M3 Max, 32–96 GB) | MLXDataStream | 6 | 12 | Higher memory bandwidth benefits from the mlx-data native path; 6 workers avoid the diminishing-returns regime observed above ~8 on Max chips. |
| **Ultra** (M1–M2 Ultra, 64–192 GB) | MLXDataStream | 8 | 16 | Maximum I/O parallelism; large prefetch buffer prevents GPU stalls on the dual-die interconnect. |

## Tradeoff Notes

### Loader Choice

| Dimension | PrefetchDataLoader | MLXDataStream |
|---|---|---|
| Threading | Python thread-pool (GIL-bound for CPU transforms) | Native C++ threads, GIL-free |
| Memory copy | Python → NumPy → mlx conversion per batch | Zero-copy to MLX tensors |
| Baseline RSS | Lower (~100–200 MB overhead) | Higher (~300–500 MB overhead from internal buffers) |
| Best for | Entry/Pro configs, small batch sizes | Max/Ultra configs, large batch sizes |

### Prefetch Depth

- **Larger prefetch** → fewer GPU stalls, smoother throughput.
- **Cost**: proportionally higher RSS (each prefetched batch stays resident).
- **Staleness risk**: on error/reset the queued batches are discarded; deeper queues waste more work.
- Rule of thumb: `prefetch_size ≈ 2 × num_workers` keeps the queue full without excessive memory.

### Worker Count

- **More workers** → higher throughput **up to I/O saturation**, after which OS-thread scheduling overhead dominates.
- On entry-class chips the crossover is ≈2–3 workers; on Max it is ≈6–8.
- Workers beyond the crossover increase p95 latency jitter without improving median throughput.

## Measurement Protocol

Use `benchmark_pipeline.py` to sweep configurations:

```bash
python -m df_mlx.benchmark_pipeline \
    --cache-dir /path/to/audio_cache \
    --batch-size 8 \
    --batches 200 \
    --workers 1,2,4,6,8 \
    --backends prefetch,mlx_stream
```

**Metrics collected per configuration:**

| Metric | Description |
|---|---|
| `batches/sec` | Sustained throughput after warmup |
| `p95 latency` | 95th-percentile per-batch fetch time |
| `RSS memory` | Resident set size delta from baseline |

**Protocol rules:**

1. Run **3 trials** per configuration; report **median**.
2. **Warmup**: first 10 batches are discarded from statistics.
3. Pin to a single performance core cluster (if possible) for reproducibility.
4. Close competing GPU workloads during measurement.

## Auto-Tuning

Set `auto_tune_dataloader = true` in the `[dataloader]` section of your
run-config TOML to let the trainer detect your chip class and apply the
recommended profile automatically. Explicit `num_workers` / `prefetch_size` /
`use_mlx_data` values always take precedence over auto-tuned defaults.

```toml
[dataloader]
auto_tune_dataloader = true
```

See `get_hardware_tuning_profile()` in `df_mlx/run_config.py` for detection
logic and the `HARDWARE_PROFILES` constant for the full profile table.
