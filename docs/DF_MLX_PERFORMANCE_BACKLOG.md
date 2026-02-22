# DF-MLX Performance Backlog (Execution Plan)

Last updated: 2026-02-15

This backlog translates the identified performance program into implementable work for this repository.

## Scope

- Target module: `DeepFilterNet/df_mlx/`
- Primary goals:
  1. Increase train-step throughput (samples/s)
  2. Reduce p95/p99 step latency jitter
  3. Reduce avoidable host sync / Python overhead
  4. Keep behavior and convergence stable
- Validation standard:
  - `docs/BENCHMARK_CONTRACT.md`
  - `docs/PERF_REGRESSION_GATE.md`

---

## Delivery principles

1. Ship high-ROI pure-MLX and control-plane reductions first.
2. Use CPython/Rust extensions for CPU-heavy data-path loops.
3. Use custom MLX kernels for fused tensor math that remains dominant after refactors.
4. Every item must define:
   - concrete files/symbols
   - explicit acceptance criteria
   - required tests/benchmarks

---

## Short-term program (0-4 weeks)

### Epic S0 — Baseline and measurement hardening

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| S0.1 | Capture baseline 48-point benchmark matrix | Validation | `df_mlx/benchmark_train_step.py`, `logs/` | Contract run completes; reproducibility metadata and baseline artifact stored |
| S0.2 | Add hotspot microbench harness for feature frontends and DF op | Pure-MLX infra | `df_mlx/benchmark_pipeline.py` (or new `benchmark_hotspots.py`) | Repeatable per-op timing for mel frontend, DfOp, iSTFT paths |

### Epic S1 — Pure-MLX hot-path vectorization

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| S1.1 | Vectorize DNSMOS mel frontend (remove nested Python loops) | Pure-MLX | `df_mlx/dnsmos_proxy.py` | Numerical parity test passes; reduced CPU overhead in hotspot benchmark |
| S1.2 | Vectorize DfOp tap-window construction | Pure-MLX | `df_mlx/modules.py`, `df_mlx/ops.py` | Output parity test passes; no model-shape regressions |
| S1.3 | Reduce redundant train/validation host conversions in non-debug modes | Pure-MLX | `df_mlx/train_dynamic.py` | Metric parity preserved for required outputs; fewer host conversions |

### Epic S2 — Data path optimization

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| S2.1 | Reduce `np.stack`/Python list churn in batch assembly | Pure-MLX / CPython | `df_mlx/dynamic_dataset.py` | Throughput improves in loader-only benchmark, no resume regressions |
| S2.2 | Batch packing acceleration in extension-backed path | CPython (PyO3/Rust) | `pyDF-data/`, `df_mlx/dynamic_dataset.py` | Equivalent output tensors, lower CPU per batch |
| S2.3 | Resume/determinism hardening after loader changes | Validation | `tests/test_checkpoint_resume_dynamic.py`, `tests/test_dynamic_dataset_failure_modes.py` | All resume semantics tests pass |

---

## Mid-term program (1-2 months)

### Epic M1 — Compile strategy and mode partitioning

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| M1.1 | Separate throughput mode vs diagnostic mode execution surfaces | Pure-MLX | `df_mlx/train_dynamic.py`, run-config docs | Fast mode avoids expensive diagnostics while preserving required checkpoints |
| M1.2 | Gen-only compiled GAN experiment implementation (guarded) | Pure-MLX compile | `df_mlx/train_dynamic.py`, `tests/test_gan_compile_experiment.py` | Meets experiment guardrails and no correctness regressions |

### Epic M2 — iSTFT and spectral pipeline optimization

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| M2.1 | Improve iSTFT overlap-add paths for common ratios | Pure-MLX | `df_mlx/ops.py` | Existing iSTFT vectorization tests pass; latency improvement in microbench |
| M2.2 | Optional fused spectral frontend for loss paths | Pure-MLX / kernel-ready | `df_mlx/loss.py`, `df_mlx/ops.py` | Equal loss values within tolerance; lower kernel launch count |

---

## Long-term program (2-4+ months)

### Epic L1 — Custom MLX kernel track

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| L1.1 | Custom Metal kernel for DfOp gather+complex MAC | Custom MLX kernel | new kernel module + `df_mlx/modules.py` integration | Numerical parity, measurable speedup over pure-MLX implementation |
| L1.2 | Custom Metal kernel for iSTFT overlap-add/normalization | Custom MLX kernel | new kernel module + `df_mlx/ops.py` integration | Numerical parity and improved p95 latency |
| L1.3 | Optional kernelized mel frontend for DNSMOS | Custom MLX kernel | kernel module + `df_mlx/dnsmos_proxy.py` | Faster mel extraction under DNSMOS-heavy workloads |

### Epic L2 — Full CPython acceleration lane

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| L2.1 | Move high-frequency data augment/mix operations into Rust-backed extension | CPython (PyO3/Rust) | `pyDF-data/src/`, `df_mlx/dynamic_dataset.py` | Functional parity with Python path, reduced CPU wall-time |
| L2.2 | Introduce guarded fallback architecture for extension path | CPython infra | `df_mlx/dynamic_dataset.py`, docs | Works with/without extension installed; tests cover both paths |

---

## Dependency order (execution sequence)

1. S0.1 → S0.2 (baseline + measurement)
2. S1.1 → S1.2 → S1.3 (vectorize compute/control-plane hotspots)
3. S2.1 → S2.2 → S2.3 (data path + extension + determinism)
4. M1.x and M2.x once short-term wins stabilize
5. L1/L2 after profiling confirms residual bottlenecks justify complexity

---

## Definition of done per backlog item

An item is done only when all are true:

1. Code merged locally with tests for behavior parity/safety.
2. Focused tests pass (`pytest`) for changed components.
3. Benchmark evidence recorded (before/after) and does not trip perf gate.
4. No convention violations (`docs/CONVENTIONS.md` invariants maintained).

---

## Current implementation status

- S0.1: complete (benchmark contract + gate infrastructure already in repo)
- S0.2: not started
- S1.1: in progress (starting now)
- S1.2: in progress (starting now)
- Remaining items: not started
