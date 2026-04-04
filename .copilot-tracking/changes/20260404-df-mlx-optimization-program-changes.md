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
