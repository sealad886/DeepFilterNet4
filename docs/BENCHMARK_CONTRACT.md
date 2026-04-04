# Benchmark Contract & Baseline Matrix

This document defines the current contract sweep, reproducibility metadata schema,
baseline metrics, and pass/fail threshold policy for df_mlx training-step benchmarks.

## Program authority

This contract is the non-negotiable benchmark authority for the staged `df_mlx`
optimization program inside `DeepFilterNet/df_mlx/`. Promotion decisions for the
approved sequence — baseline / gate lock, compiled `train_dynamic.py` fast path,
`MLXDataStream` data path, residual hotspot re-profiling, optional flagged advanced
acceleration, and release-candidate hardening — are made from
`DeepFilterNet/df_mlx/benchmark_train_step.py` running this contract.
`DeepFilterNet/df_mlx/benchmark_pipeline.py` and
`DeepFilterNet/df_mlx/benchmark_hotspots.py` remain secondary diagnostic surfaces:
they can explain or localize a win/regression for a specific phase, but they do not
replace the train-step benchmark for promotion.

### Stage-inherited expectations

Every later optimization phase inherits the same acceptance checks from this
contract:

- **Throughput**: compare `samples_per_sec_p5` against the active baseline using the
  regression gate defined below.
- **Tail latency**: enforce the `step_p95_ms` gate below and retain `step_p99_ms`
  in every artifact for spike review and release-candidate sign-off.
- **Variance**: require coefficient of variation (`samples_per_sec_std /
  samples_per_sec_mean`) to stay at or below 0.20 before results can be used for
  promotion.
- **Secondary diagnostics**: supporting benchmarks may add stage-specific evidence,
  but they cannot waive or replace these train-step promotion rules.

## Canonical Matrix

With `--contract`, the benchmark currently pins the following promotion-critical
dimensions while leaving the remaining CLI sweep arguments at the values you pass
(or their defaults):

| Dimension | Values | Notes |
|-----------|--------|-------|
| Batch size | 1, 4, 8 | Overridden by `--contract` |
| Compile mode | `compiled`, `eager` | Overridden by `--contract` |
| Warmup steps | 5 | Overridden by `--contract` |
| Measured steps | 50 | Overridden by `--contract` |
| Repeats | 3 | Overridden by `--contract` |

Other sweep axes — backend, workers, backend-specific prefetch setting, split,
model variant, optimizer hyperparameters, DSP configuration, and seed — continue
to come from the active CLI arguments in `benchmark_train_step.py`. As a result,
the exact case count depends on those settings and on whether optional backends
such as `mlx_stream` are available in the environment.

## Warmup Policy

- **5 warmup steps** are executed and discarded before measurement begins.
- Warmup allows JIT compilation, Metal shader caching, and memory pool stabilization.

## Measurement Window

- **50 measured steps** minimum per configuration point.
- Each configuration is run **3 independent times** (repeats) for variance estimation.
- Total measured step-count per config: ≥ 150 steps (3 × 50).

## Reproducibility Metadata Schema

When `benchmark_train_step.py` is run with `--metadata`, it emits the following
run-level metadata alongside `results[]`. Each `results[]` entry is the flat
`BenchmarkResult` record written by the benchmark today.

```json
{
  "hardware": {
    "chip": "Apple M3 Max",
    "gpu_cores": 40,
    "memory_gb": 48
  },
  "os": {
    "name": "macOS",
    "version": "15.2"
  },
  "runtime": {
    "python": "3.11.12",
    "mlx": "0.24.1",
    "mlx_nn": "0.24.0"
  },
  "commit": "abcdef1",
  "timestamp": "2026-02-14T12:00:00+00:00"
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `hardware.chip` | string | Apple Silicon chip name (e.g. "Apple M3 Max") |
| `hardware.gpu_cores` | int | Number of GPU cores |
| `hardware.memory_gb` | int | Total unified memory in GB |
| `os.name` | string | Operating system name |
| `os.version` | string | OS version string |
| `runtime.python` | string | Python version |
| `runtime.mlx` | string | MLX framework version |
| `runtime.mlx_nn` | string | MLX neural network module version (if available) |
| `commit` | string | Short git commit hash |
| `timestamp` | string | ISO 8601 UTC timestamp |

## Baseline Metrics

Each configuration point records the following key aggregated metrics used by the
contract and promotion gate:

| Metric | Unit | Description |
|--------|------|-------------|
| `samples_per_sec_mean` | samples/s | Mean throughput across repeats |
| `samples_per_sec_std` | samples/s | Standard deviation of throughput |
| `samples_per_sec_p5` | samples/s | 5th percentile throughput |
| `samples_per_sec_p95` | samples/s | 95th percentile throughput |
| `step_mean_ms` | ms | Mean step latency |
| `step_p95_ms` | ms | 95th percentile step latency |
| `step_p99_ms` | ms | 99th percentile step latency |
| `data_mean_ms` | ms | Mean data-loading latency |
| `data_p95_ms` | ms | 95th percentile data-loading latency |
| `data_p99_ms` | ms | 99th percentile data-loading latency |
| `loss_mean` | scalar | Mean loss value |
| `loss_std` | scalar | Loss standard deviation |
| `loss_last` | scalar | Final measured loss value |

## Pass/Fail Threshold Policy

### Regression Gates

| Gate | Condition | Description |
|------|-----------|-------------|
| **Throughput** | `new_p5 < baseline_p5 × 0.90` | Fail if >10% throughput regression at p5 |
| **Tail latency** | `new_p95 > baseline_p95 × 1.15` | Fail if >15% tail latency increase at p95 |
| **Variance** | `new_std / new_mean > 0.20` | Fail if coefficient of variation exceeds 20% |

### Decision Logic

```
PASS  — all three gates pass
FAIL  — any gate fails
```

The current standalone promotion path (`scripts/perf_gate.py`) has no
`BENCHMARK_OVERRIDE`-style force-pass mode, although its throughput and latency
threshold factors can still be adjusted with CLI flags.

### Tolerance Rationale

- **10% throughput margin**: Accounts for thermal throttling, background processes,
  and minor OS scheduling variance on consumer Apple Silicon.
- **15% tail latency margin**: Tail latencies (p95) are inherently noisier than means;
  the wider band prevents false positives from GC pauses or Metal shader recompilation.
- **20% CV cap**: A coefficient of variation above 0.20 indicates the benchmark
  environment is too noisy for reliable comparison. The run should be retried under
  quieter conditions.

## Running the Canonical Matrix

```bash
cd DeepFilterNet
python -m df_mlx.benchmark_train_step \
    --contract \
    --metadata \
    --cache-dir /path/to/audio_cache \
    --json-out logs/benchmark_contract.json
```

The `--contract` flag pins the promotion-critical sweep parameters listed above by
overriding `--batch-size`, `--compiled`, `--warmup-steps`, `--steps`, and
`--repeats`. Other sweep axes continue to use the CLI values in effect. The
`--metadata` flag attaches reproducibility metadata to the JSON output. This
command is the primary promotion authority used by
[PERF_REGRESSION_GATE.md](PERF_REGRESSION_GATE.md).

## Baseline Artifact Format

With `--metadata`, `benchmark_train_step.py` writes JSON with the following
top-level structure:

```json
{
  "metadata": {
    "hardware": {
      "chip": "Apple M3 Max",
      "gpu_cores": 40,
      "memory_gb": 48
    },
    "os": {
      "name": "macOS",
      "version": "15.2"
    },
    "runtime": {
      "python": "3.11.12",
      "mlx": "0.24.1",
      "mlx_nn": "0.24.0"
    },
    "commit": "abcdef1",
    "timestamp": "2026-02-14T12:00:00+00:00"
  },
  "results": [
    {
      "backend": "mlx_stream",
      "split": "train",
      "epoch": 0,
      "workers": 4,
      "prefetch": 16,
      "batch_size": 4,
      "warmup_steps": 5,
      "steps_requested": 50,
      "repeats": 3,
      "compiled": true,
      "model_variant": "full",
      "learning_rate": 0.001,
      "weight_decay": 0.0,
      "grad_clip": 0.0,
      "sample_rate": 48000,
      "segment_length": 5.0,
      "fft_size": 960,
      "hop_size": 480,
      "nb_erb": 32,
      "nb_df": 96,
      "seed": 42,
      "measured_steps": 150,
      "measured_samples": 600,
      "total_seconds": 48.0,
      "data_mean_ms": 5.0,
      "data_p95_ms": 8.0,
      "data_p99_ms": 12.0,
      "step_mean_ms": 80.0,
      "step_p95_ms": 95.0,
      "step_p99_ms": 110.0,
      "total_mean_ms": 85.0,
      "total_p95_ms": 101.0,
      "total_p99_ms": 118.0,
      "steps_per_sec": 3.1,
      "samples_per_sec": 12.5,
      "samples_per_sec_mean": 12.6,
      "samples_per_sec_std": 0.3,
      "samples_per_sec_p5": 11.8,
      "samples_per_sec_p95": 13.1,
      "loss_mean": 0.31,
      "loss_std": 0.02,
      "loss_last": 0.28
    }
  ]
}
```

Current `results[]` entries are flat `BenchmarkResult` records. For compatibility,
`scripts/perf_gate.py` accepts both this flat shape and the older nested
`{"config": {...}, "metrics": {...}}` record format. Gate config keys are derived
from whatever fields are present in each record, so flat and nested artifacts only
compare cleanly when they encode the same dimensions.

## References

- [BENCHMARKS.md](BENCHMARKS.md) — Historical benchmark results and methodology
- [benchmark_train_step.py](../DeepFilterNet/df_mlx/benchmark_train_step.py) — Primary promotion benchmark entrypoint
- [benchmark_pipeline.py](../DeepFilterNet/df_mlx/benchmark_pipeline.py) — Secondary data-path diagnostic benchmark
- [benchmark_hotspots.py](../DeepFilterNet/df_mlx/benchmark_hotspots.py) — Secondary hotspot diagnostic benchmark
- [benchmark_common.py](../DeepFilterNet/df_mlx/benchmark_common.py) — Shared helpers
