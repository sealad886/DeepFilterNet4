# Benchmark Contract & Baseline Matrix

This document defines the canonical benchmark matrix, reproducibility metadata schema,
baseline metrics, and pass/fail threshold policy for df_mlx training-step benchmarks.

## Canonical Matrix

Every official benchmark run sweeps the following dimensions:

| Dimension | Values | Notes |
|-----------|--------|-------|
| Backbone | `dfnet4` (primary), `mamba` (secondary) | `dfnet4` is the default model variant |
| Batch size | 1, 4, 8 | Sweep across small/medium workloads |
| Compile mode | `compiled`, `eager` | MLX `mx.compile` on/off |
| Grad accumulation | 1, 2 | Effective batch scaling |
| FP16 | on, off | Half-precision training toggle |

This produces **2 × 3 × 2 × 2 × 2 = 48** configuration points per backend.

## Warmup Policy

- **5 warmup steps** are executed and discarded before measurement begins.
- Warmup allows JIT compilation, Metal shader caching, and memory pool stabilization.

## Measurement Window

- **50 measured steps** minimum per configuration point.
- Each configuration is run **3 independent times** (repeats) for variance estimation.
- Total measured step-count per config: ≥ 150 steps (3 × 50).

## Reproducibility Metadata Schema

Every benchmark run **must** capture the following metadata alongside results:

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
  "config": {
    "backbone": "dfnet4",
    "batch_size": 4,
    "compiled": true,
    "grad_accumulation": 1,
    "fp16": false
  },
  "timestamp": "2026-02-14T12:00:00+00:00",
  "reproducibility_hash": "<sha256(commit+config_json+chip)>"
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
| `config` | object | Full benchmark configuration for this run |
| `timestamp` | string | ISO 8601 UTC timestamp |
| `reproducibility_hash` | string | SHA-256 of `commit + sorted(config) + chip` |

## Baseline Metrics

Each configuration point produces the following aggregated metrics:

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
| `peak_rss_mb` | MB | Peak resident set size |

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
FAIL  — any gate fails (unless overridden)
OVERRIDE — set BENCHMARK_OVERRIDE=1 env var to force pass with warning
```

### Override Mechanism

Setting the environment variable `BENCHMARK_OVERRIDE=1` suppresses failures and emits
a warning instead. This is intended for known-noisy environments (e.g. CI with shared
hardware) and **must not** be used for release baselines.

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

The `--contract` flag overrides individual sweep arguments and runs the full canonical
matrix defined above. The `--metadata` flag attaches reproducibility metadata to the
JSON output.

## Baseline Artifact Format

Baseline results are stored as JSON with the following top-level structure:

```json
{
  "metadata": { "...reproducibility fields..." },
  "results": [
    {
      "config": { "backbone": "dfnet4", "batch_size": 4, "compiled": true, "...": "..." },
      "metrics": {
        "samples_per_sec_mean": 12.5,
        "samples_per_sec_std": 0.3,
        "samples_per_sec_p5": 11.8,
        "samples_per_sec_p95": 13.1,
        "step_mean_ms": 80.0,
        "step_p95_ms": 95.0,
        "step_p99_ms": 110.0,
        "data_mean_ms": 5.0,
        "data_p95_ms": 8.0,
        "data_p99_ms": 12.0,
        "loss_mean": 0.31,
        "loss_std": 0.02,
        "peak_rss_mb": 1200
      }
    }
  ]
}
```

## References

- [BENCHMARKS.md](BENCHMARKS.md) — Historical benchmark results and methodology
- [benchmark_train_step.py](../DeepFilterNet/df_mlx/benchmark_train_step.py) — Benchmark entrypoint
- [benchmark_common.py](../DeepFilterNet/df_mlx/benchmark_common.py) — Shared helpers
