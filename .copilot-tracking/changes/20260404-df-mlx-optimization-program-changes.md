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
- Phase 3 slice: `MLXDataStream` now precomputes wrapped retry metadata once per epoch and reuses fallback-array views in `_create_stream()` / `_sample_transform()` instead of allocating Python fallback lists and per-sample retry arrays in the hot path. The safety suite now covers wrapped fallback order, explicit all-fail retry errors, real mlx-data iteration, and resume-prefix determinism for the optimized path.
- Phase 3 evidence: on a controlled raw-list benchmark against base commit `35f64e6`, `benchmark_pipeline.py` (`mlx_stream`, 100 measured batches, 5 repeats) improved from 2611.13 to 2707.54 samples/s (+3.69%), while p95 fell from 2.06 ms to 1.98 ms (-3.87%) and p99 fell from 2.44 ms to 2.32 ms (-4.84%). `benchmark_train_step.py` (`mlx_stream`, compiled, 6 measured steps, 2 repeats) held/improved from 5.84 to 5.94 samples/s (+1.79%), with step mean 682.81 ms to 669.32 ms (-1.98%) and step p95 effectively flat at 865.97 ms to 870.35 ms (+0.51%).
- Phase 4 re-profile: `benchmark_hotspots.py` at batch sizes 4 and 8 still ranks `spectral_loss` above the other residual microbench surfaces, but the absolute residual costs remain too small relative to the current train-step totals to justify another optimization slice. No residual hotspot cleared the “must move benchmark_train_step.py” bar.
- Phase 5 decision: no optional advanced acceleration or Metal-kernel experiment was promoted. Residual profiling did not uncover a benchmark-justified candidate, so the program explicitly stops here instead of adding speculative complexity behind new flags.
- Phase 6 RC hardening: the closeout acceptance bundle passed with 150 focused Python tests, `cargo test -p deep_filter -q`, a fresh current `benchmark_pipeline.py` run at 2446.79 samples/s (p95 2.21 ms, p99 2.61 ms), and a fresh current `benchmark_train_step.py` run whose best measured `mlx_stream` variant reached 5.71 samples/s at prefetch 32 with total p95 1008.71 ms.
