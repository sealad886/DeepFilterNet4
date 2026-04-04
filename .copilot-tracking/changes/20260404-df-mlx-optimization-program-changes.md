<!-- markdownlint-disable-file -->
# df_mlx optimization program changes

## Scope

Implement the approved, measurement-first `df_mlx` optimization program inside MLX/Metal-related paths only.

## Milestone log

### 2026-04-04

- Completed Task 1.1 doc reconciliation for the active `df_mlx` optimization program.
- Completed Task 1.2 benchmark-authority lock so `benchmark_train_step.py` is the
  explicit promotion authority, `benchmark_pipeline.py` / `benchmark_hotspots.py`
  stay secondary diagnostics, and later phases inherit the same throughput,
  tail-latency, and variance expectations.
- Corrected the backlog and performance audit so they now reflect the current benchmark surfaces (`benchmark_train_step.py`, `benchmark_pipeline.py`, `benchmark_hotspots.py`) and the approved execution sequence.
- Removed stale assumptions that `benchmark_hotspots.py` still needed to be added, that `StreamingDfNet4.process_frame()` was uncompiled, or that `PrefetchDataLoader` was in scope as an optimization target.
- Review-fix pass: trimmed host-local/setup-history details from this milestone log and clarified `docs/PERFORMANCE_AUDIT.md` historical implementation sections as prior repository work that informs, but does not replace, the newly approved staged program.
- Task 1.2 follow-up: taught `benchmark_train_step.py` to emit per-repeat throughput summary fields (`samples_per_sec_mean`, `samples_per_sec_std`, `samples_per_sec_p5`, `samples_per_sec_p95`) while preserving the legacy aggregate `samples_per_sec` field for downstream compatibility.
- Task 1.2 final review fix: aligned `generate_contract_matrix()` with the live `--contract` override sweep (`batch_size`, `compiled`, `warmup_steps`, `steps`, `repeats`) and added a direct flat-record compatibility regression test for `scripts/perf_gate.py`.
- Phase 2 slice: `sync_mode=fast` now skips sync-window component-metric recomputation in `training_metrics.collect_sync_metrics()` and suppresses detailed epoch component summaries when those metrics were intentionally not collected.
- Phase 2.2 doc alignment: clarified that `benchmark_train_step.py` remains the canonical program-level train-step authority, but its `_build_train_step()` helper uses local `loss_fn` / `step_fn` closures and therefore does not cover `train_dynamic.py` loop-control overhead. The docs now point Phase 2 verification at focused sync-mode tests, short controlled `train_dynamic.py` runs, and optional `benchmark_sync_barriers.py` diagnostics without demoting the canonical contract benchmark.
