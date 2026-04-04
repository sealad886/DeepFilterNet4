# DF-MLX Performance Backlog (Execution Plan)

Last updated: 2026-04-04

This backlog reflects the approved current-state optimization program for `DeepFilterNet/df_mlx/`.
It is intentionally grounded in the repository as it exists today, not in older backlog assumptions.

## Scope

- Target module: `DeepFilterNet/df_mlx/`
- Optimization scope: MLX/Metal-only surfaces inside `df_mlx`
- Primary goals:
  1. Increase train-step throughput (samples/s)
  2. Reduce p95/p99 step latency jitter
  3. Reduce avoidable host sync / Python overhead on the MLX training path
  4. Keep behavior, convergence, and resume semantics stable
- Validation standard:
  - `docs/BENCHMARK_CONTRACT.md`
  - `docs/PERF_REGRESSION_GATE.md`

---

## Delivery principles

1. `DeepFilterNet/df_mlx/benchmark_train_step.py` is the canonical authority for train-step throughput, tail latency, and perf-gate promotion decisions.
2. `benchmark_train_step.py` does not execute the `train_dynamic.py` batch loop, `debug.sync_mode`, or sync-window metric collection, so loop-level fast-path changes on that surface require supplemental verification from focused tests and short controlled `train_dynamic.py` runs.
3. `DeepFilterNet/benchmark_sync_barriers.py` is an optional diagnostic microbenchmark for sync-barrier hypotheses only; it does not replace either the canonical train-step benchmark or loop-level `train_dynamic.py` verification.
4. `DeepFilterNet/df_mlx/benchmark_pipeline.py` already exists and should be used primarily to validate the `MLXDataStream` path against train-step behavior.
5. `DeepFilterNet/df_mlx/benchmark_hotspots.py` already exists and should be used only for residual hotspot re-profiling after earlier stages land.
6. `StreamingDfNet4.process_frame()` is already compiled; any later streaming follow-up targets `process_audio()` orchestration or other residual overhead, not re-adding compilation to the inner frame kernel.
7. `PrefetchDataLoader` remains a comparison backend in existing benchmarks, but it is out of scope as an optimization target for this program.
8. Advanced compile or Metal-kernel work stays feature-flagged until benchmark and parity evidence justify promotion.

---

## Benchmark surfaces (current repository state)

| Surface | File | Current role |
|---|---|---|
| Canonical train-step benchmark | `DeepFilterNet/df_mlx/benchmark_train_step.py` | Primary authority for train-step throughput, tail latency, and perf-gate promotion decisions |
| `train_dynamic.py` loop verification | `DeepFilterNet/tests/test_fast_sync_metric_suppression.py`, `DeepFilterNet/tests/test_sync_cadence_integration.py`, `DeepFilterNet/tests/test_sync_mode_partitioning.py`, `docs/IMPLEMENTATION_HANDOFF.md` | Required supplemental evidence for changes that affect `sync_mode` cadence or sync-window metric collection inside the real training loop |
| Sync-barrier microbenchmark | `DeepFilterNet/benchmark_sync_barriers.py` | Optional synthetic diagnostic for barrier-pattern hypotheses; not a substitute for the real `train_dynamic.py` loop checks |
| Data-path benchmark | `DeepFilterNet/df_mlx/benchmark_pipeline.py` | Existing harness for loader/data-wait behavior; use primarily for `MLXDataStream` validation alongside the train-step benchmark |
| Hotspot microbench | `DeepFilterNet/df_mlx/benchmark_hotspots.py` | Existing per-op harness for residual hotspot re-profiling after Stage 1 and Stage 2 work |

---

## Approved execution sequence

1. Baseline / gate lock
2. Compiled `train_dynamic.py` fast path
3. `MLXDataStream` data path
4. Residual hotspot re-profiling
5. Optional flagged advanced acceleration
6. Release-candidate hardening

---

## Stage 0 — Baseline and stale-assumption lock

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| S0.1 | Lock the benchmark contract and baseline measurement surfaces | Validation | `docs/BENCHMARK_CONTRACT.md`, `docs/PERF_REGRESSION_GATE.md`, `DeepFilterNet/df_mlx/benchmark_train_step.py`, `logs/` | Contract run metadata is captured and the canonical baseline is recorded for the active program |
| S0.2 | Reconcile roadmap and benchmark surfaces with current code | Docs / validation | `docs/DF_MLX_PERFORMANCE_BACKLOG.md`, `docs/PERFORMANCE_AUDIT.md`, `DeepFilterNet/df_mlx/benchmark_pipeline.py`, `DeepFilterNet/df_mlx/benchmark_hotspots.py` | Docs enumerate the current benchmark entrypoints and contain no stale claims about missing hotspot harnesses, an uncompiled `process_frame()`, or `PrefetchDataLoader` optimization scope |

## Stage 1 — Compiled `train_dynamic.py` fast path

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| S1.1 | Separate throughput-oriented fast execution from diagnostic/control-plane work | Pure-MLX compile | `DeepFilterNet/df_mlx/train_dynamic.py`, related training helpers/docs | `benchmark_train_step.py` still clears the train-step contract, while focused `sync_mode` tests plus short controlled `train_dynamic.py` runs confirm that loop-level fast-mode cleanup suppresses intentionally skipped sync-window metrics without removing explicit diagnostic modes |

## Stage 2 — `MLXDataStream` data path

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| S2.1 | Reduce `MLXDataStream`-side Python batch materialization and MLX conversion overhead | MLX data path | `DeepFilterNet/df_mlx/dynamic_dataset.py`, `DeepFilterNet/df_mlx/benchmark_pipeline.py` | `benchmark_pipeline.py` (`mlx_stream`) and `benchmark_train_step.py` both improve or hold the regression gate |
| S2.2 | Harden resume/determinism after `MLXDataStream` changes | Validation | `DeepFilterNet/tests/test_checkpoint_resume_dynamic.py`, `DeepFilterNet/tests/test_dynamic_dataset_failure_modes.py`, `DeepFilterNet/df_mlx/test_dynamic_dataset_safety.py` | Resume/determinism checks pass with the optimized `MLXDataStream` path |

## Stage 3 — Residual hotspot re-profiling

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| S3.1 | Re-profile train-step, data-path, and hotspot surfaces after Stage 1 and Stage 2 | Validation | `DeepFilterNet/df_mlx/benchmark_train_step.py`, `DeepFilterNet/df_mlx/benchmark_pipeline.py`, `DeepFilterNet/df_mlx/benchmark_hotspots.py`, `logs/` | Residual hotspots are ranked with train-step data first and microbench data second |
| S3.2 | Optimize only residual MLX hotspots that still move the train-step benchmark | Pure-MLX / MLX runtime | `DeepFilterNet/df_mlx/dnsmos_proxy.py`, `DeepFilterNet/df_mlx/modules.py`, `DeepFilterNet/df_mlx/ops.py`, `DeepFilterNet/df_mlx/model.py` | Any targeted hotspot change preserves parity and produces a measurable train-step win |

Stage 3 note: older candidate surfaces such as the DNSMOS mel frontend, DfOp tap-window work, or streaming follow-up stay deferred until re-profiling confirms they still matter. If streaming work is promoted, it should target `StreamingDfNet4.process_audio()` orchestration/output accumulation rather than the already-compiled `process_frame()` inner loop.

## Stage 4 — Optional flagged advanced acceleration

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| S4.1 | Add guarded compile or Metal-kernel experiments only for benchmark-proven residual bottlenecks | Feature-flagged MLX/Metal | `DeepFilterNet/df_mlx/` kernel/compile surfaces plus supporting tests/docs | Experimental paths are disabled by default, preserve parity, and show a benchmark win before any promotion |

## Stage 5 — Release-candidate hardening

| ID | Item | Type | Files | Acceptance criteria |
|---|---|---|---|---|
| S5.1 | Re-run the contract, perf gate, and focused safety suite for the chosen fast path | Validation | `docs/BENCHMARK_CONTRACT.md`, `docs/PERF_REGRESSION_GATE.md`, `DeepFilterNet/df_mlx/benchmark_train_step.py`, targeted `df_mlx` tests | The release-candidate path clears the benchmark gate and focused correctness/resume verification |

---

## Definition of done per backlog item

An item is done only when all are true:

1. Code merged locally with tests for behavior parity/safety.
2. Focused tests pass (`pytest`) for changed components.
3. Benchmark evidence recorded (before/after) and does not trip the perf gate.
4. No convention violations (`docs/CONVENTIONS.md` invariants maintained).

---

## Current implementation status

- `benchmark_train_step.py`: present and canonical.
- `benchmark_pipeline.py`: present and available for `MLXDataStream` validation.
- `benchmark_hotspots.py`: present and available for residual hotspot re-profiling.
- `StreamingDfNet4.process_frame()`: already compiled; any future streaming work should target `process_audio()` or other residual orchestration surfaces.
- S0.1: complete (benchmark contract + gate infrastructure already in repo).
- S0.2: complete (current-state reconciliation landed on 2026-04-04).
- S1.x and later: not started under the approved execution sequence.
