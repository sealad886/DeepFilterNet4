<!-- markdownlint-disable-file -->
# df_mlx Optimization Program Design

## Goal

Deliver a release-candidate-quality, multi-stage optimization program for `DeepFilterNet/df_mlx/` that improves **training throughput** and **p95/p99 step latency** on Apple Silicon while keeping correctness, convergence behavior, and resume semantics stable.

## Scope

### In Scope

- MLX- and Metal-related paths inside `DeepFilterNet/df_mlx/`
- `MLXDataStream` and its MLX-facing batch materialization path
- compiled training-path execution in `train_dynamic.py`
- existing `df_mlx` benchmark and performance-gate tooling
- residual MLX hotspots relevant to training throughput and tail latency
- feature-flagged Metal kernel and compile experiments implemented inside `df_mlx`

### Out of Scope

- `PrefetchDataLoader` as an optimization target
- non-MLX or non-Metal optimization work outside `df_mlx`
- speculative cross-module refactors outside the release-candidate path

## Design Principles

1. **Measurement first** — no optimization is promoted without benchmark evidence.
2. **Training-path first** — decisions are driven by `benchmark_train_step.py`, not isolated microbenchmarks alone.
3. **MLX/Metal only** — all implementation work stays inside `DeepFilterNet/df_mlx/`.
4. **Flag advanced work** — invasive kernel or compile experiments remain behind feature flags until verified.
5. **Release-candidate discipline** — each stage must preserve correctness and clear the existing performance gate.

## Program Architecture

The project is organized as one overall program with five gated stages:

### Stage 0 — Re-baseline and Correct Stale Assumptions

Reconcile the current repository state with older roadmap and audit material so the program does not re-plan already-landed work. The output of this stage is a corrected execution baseline and a single release-candidate scorecard grounded in:

- `docs/BENCHMARK_CONTRACT.md`
- `docs/PERF_REGRESSION_GATE.md`
- `DeepFilterNet/df_mlx/benchmark_train_step.py`
- `DeepFilterNet/df_mlx/benchmark_pipeline.py`
- `DeepFilterNet/df_mlx/benchmark_hotspots.py`

### Stage 1 — MLX Data-Path Optimization

Optimize only the `MLXDataStream` training path and the MLX-facing conversion surfaces inside `dynamic_dataset.py`. The intent is to reduce:

- batch fetch latency
- data wait p95/p99
- avoidable batch materialization overhead
- resume/jitter regressions on the canonical training path

This stage deliberately excludes `PrefetchDataLoader`, even if adjacent code is similar, to preserve scope discipline.

### Stage 2 — Compiled Training-Path Optimization

Refine the main compiled training path in `train_dynamic.py` so the throughput-oriented execution surface pays only for work required on the release-candidate path. This stage targets:

- optional diagnostic/control-plane work inside the hot path
- fast-mode vs diagnostic-mode execution separation
- reduction of avoidable syncs, logging, or non-essential per-step computation

The existing closure-based compile architecture remains intact; the design assumes the current MLX compile boundaries are correct and should be optimized, not re-architected.

### Stage 3 — Residual MLX Hotspot Optimization

Re-profile the remaining operations that still materially affect train-step throughput or tail jitter after Stages 1 and 2. Candidate surfaces include:

- STFT / iSTFT
- Mel frontend
- DfOp
- Spectral loss path
- end-to-end streaming follow-up work that shares the same MLX/Metal execution surfaces

This stage is benchmark-driven and may remove items from scope if prior stages make them irrelevant.

### Stage 4 — Flagged Advanced Acceleration

Implement advanced MLX/Metal acceleration experiments only when the earlier benchmark evidence still shows a meaningful residual bottleneck. This includes:

- additional fused Metal kernels
- compile experiments
- branch-isolated, feature-flagged acceleration paths in `df_mlx`

Promotion from experiment to mainline requires parity tests and benchmark-gate clearance.

### Stage 5 — Release-Candidate Hardening

Consolidate the optimized path into a release-candidate standard deliverable by requiring:

- benchmark contract runs
- performance regression gate review
- focused correctness and resume-safety tests
- docs/config cleanup for the chosen fast path

## Data Flow and Execution Focus

The primary optimized path for this program is:

`MLXDataStream -> batched MLX tensors -> compiled training step in train_dynamic.py -> benchmark_train_step / perf gate`

All recommendations in the later implementation plan should be judged primarily by their effect on this path.

## Feature-Flag Policy

All invasive or experimental acceleration work must:

1. live behind an explicit feature flag
2. have a safe default path
3. include parity verification against the default path
4. demonstrate a benchmark win before promotion

This is especially important for kernel work, because the repository’s prior audit already showed that not all low-level graph optimizations translate into end-to-end throughput gains.

## Testing and Verification Strategy

Each stage must define its own verification, but the overall project requires:

- component-level tests for changed `df_mlx` behavior
- parity tests for advanced acceleration paths and fallbacks
- benchmark evidence against the canonical contract
- p95/p99 latency review, not averages alone
- resume/determinism checks for any data-path change touching `MLXDataStream`

## Risks and Mitigations

### Risk: stale roadmap assumptions drive unnecessary work
**Mitigation:** Stage 0 explicitly corrects docs/backlog before implementation sequencing.

### Risk: microbench wins do not improve training throughput
**Mitigation:** treat `benchmark_train_step.py` as the primary authority for promotion decisions.

### Risk: invasive acceleration destabilizes the RC path
**Mitigation:** keep advanced work feature-flagged and branch-isolated until benchmark + parity proof exists.

### Risk: optimization scope expands beyond `df_mlx`
**Mitigation:** reject tasks that require moving outside MLX/Metal surfaces in `DeepFilterNet/df_mlx/`.

## Final Design Decision

The selected approach is a **measurement-first staged optimization program** focused on the MLX/Metal training path in `df_mlx`, with `MLXDataStream` favored over `PrefetchDataLoader`, and advanced acceleration restricted to feature-flagged work that earns promotion through benchmark evidence.
