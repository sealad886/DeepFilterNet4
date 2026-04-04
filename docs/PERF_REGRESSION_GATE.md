# Performance Regression Gate

## Gate Overview

This gate runs before merging performance-sensitive changes to catch throughput and
latency regressions in df_mlx training. It compares benchmark results from a candidate
branch against a known baseline using the thresholds defined in
[BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md).

## Program promotion authority

This gate is the operational enforcement mechanism for the staged `df_mlx`
optimization program. Every promotion checkpoint after baseline / gate lock —
compiled `train_dynamic.py` fast path, `MLXDataStream` data path, residual hotspot
re-profiling, optional flagged advanced acceleration, and release-candidate
hardening — uses the same primary train-step authority for throughput and tail
latency: `python -m df_mlx.benchmark_train_step --contract --metadata`.
`benchmark_pipeline.py` and `benchmark_hotspots.py` may be run to diagnose a
stage-specific regression or validate a hypothesis, but they never replace the
train-step benchmark for merge or promotion decisions.

`benchmark_train_step.py` also has an explicit scope boundary: `_build_train_step()`
creates its own local `loss_fn` / `step_fn` pair and does not execute the
`train_dynamic.py` batch loop, `debug.sync_mode`, or sync-window metric
collection. For loop-control changes on that surface, the canonical gate
benchmark remains required but is not sufficient on its own.

### Supplemental verification for `train_dynamic.py` loop-control changes

When a candidate changes `train_dynamic.py` behavior that depends on
`sync_mode`, eval cadence, or sync-window metric collection, promotion evidence
must also include:

1. Focused sync-mode behavior tests:
   - `DeepFilterNet/tests/test_fast_sync_metric_suppression.py`
   - `DeepFilterNet/tests/test_sync_cadence_integration.py`
   - `DeepFilterNet/tests/test_sync_mode_partitioning.py`
2. Short controlled `python -m df_mlx.train_dynamic` runs using
   `sync_mode=fast` and `sync_mode=debug` or `sync_mode=profile` for loop-level
   observability, using the existing checks documented in
   [IMPLEMENTATION_HANDOFF.md](IMPLEMENTATION_HANDOFF.md) and
   [SYNC_BARRIER_POLICY.md](SYNC_BARRIER_POLICY.md).
3. Optional `DeepFilterNet/benchmark_sync_barriers.py` runs when you need a
   synthetic barrier-pattern diagnostic. This benchmark is supporting evidence
   only and does not replace either the canonical gate or the real training-loop
   checks.

### Inherited promotion rules

All later phases inherit the same rules from
[BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md):

- **Throughput**: compare `samples_per_sec_p5` to the active baseline and fail if it
  crosses the contract threshold.
- **Tail latency**: enforce the `step_p95_ms` threshold and keep `step_p99_ms` in
  every artifact for spike review and release-candidate sign-off.
- **Variance**: reject runs with coefficient of variation above 0.20 as too noisy
  for promotion.

## When to Run

- Any change touching `train_dynamic.py`, `dynamic_dataset.py`, `run_config.py`, or model code
- Before release tags
- After dependency upgrades (MLX, mlx-data, mlx-whisper, etc.)
- Any PR marked with the `perf` label
- At each staged-program promotion checkpoint after the baseline / gate lock
- Any `train_dynamic.py` change that affects `sync_mode`, eval cadence, or
  sync-window metric collection (run the supplemental verification above in
  addition to the canonical gate)

## Gate Procedure

1. **Checkout baseline** (main branch HEAD or tagged release).
2. Run the canonical benchmark:
   ```bash
   cd DeepFilterNet
   python -m df_mlx.benchmark_train_step \
        --contract --metadata \
        --cache-dir /path/to/audio_cache \
        --json-out baseline.json
   ```
3. **Checkout candidate** branch.
4. Run the same benchmark:
   ```bash
   python -m df_mlx.benchmark_train_step \
        --contract --metadata \
        --cache-dir /path/to/audio_cache \
        --json-out candidate.json
   ```
5. **Compare** with the perf gate script:
   ```bash
   python scripts/perf_gate.py \
        --baseline baseline.json \
        --candidate candidate.json \
        --report gate_report.md
   ```
6. Review the generated report. Exit code 0 = pass, 1 = fail, 2 = error.

## Reproducibility Controls

- **Same hardware**: Baseline and candidate should run on the same machine. The gate
  report surfaces recorded metadata, but `scripts/perf_gate.py` does not currently
  auto-fail on hardware or runtime metadata mismatches; compare both artifacts
  manually before treating the result as authoritative.
- **Thermal settle**: If you run baseline and candidate back-to-back on one machine,
  allow manual cooldown as needed. `benchmark_train_step.py` does not currently
  inject a fixed 30-second pause between repeats or between the two runs.
- **Close resource-heavy apps**: Browsers, IDEs, and other GPU-consuming applications
  should be closed during benchmark runs.
- **3 repeats feed the contract metrics**: Each config point runs 3 independent
  repeats, and the emitted artifact records the aggregated `samples_per_sec_p5`,
  `step_p95_ms`, `samples_per_sec_mean`, and `samples_per_sec_std` values. The gate
  compares those reported fields directly; it does not recompute a median from raw
  repeats.
- **Metadata audit trail**: Every run records chip, GPU cores, memory, OS version,
  Python version, MLX version, and git commit. This metadata is stored in the JSON
  artifact for post-hoc auditing.

## Variance Policy

- **CV > 20%**: Results are unreliable. Re-run with additional warmup steps or under
  quieter conditions (fewer background processes, cooler hardware).
- **No automatic outlier filtering**: The current benchmark/gate path uses the
  emitted aggregates as-is. It does not apply 3σ trimming or enforce a minimum
  surviving-run count after filtering.

## Triage Protocol

When the gate fails:

1. **Check environment drift**: Compare the recorded hardware, OS, and runtime
   metadata between baseline and candidate artifacts. If the machine or runtime
   changed, re-run on matching hardware before treating the comparison as
   authoritative.
2. **Isolate the regression commit**: Use `git bisect` with the benchmark script:
   ```bash
   git bisect start <bad-commit> <good-commit>
   git bisect run bash -lc '
     python -m df_mlx.benchmark_train_step \
       --contract --metadata \
       --cache-dir /path/to/audio_cache \
       --json-out candidate.json &&
     python scripts/perf_gate.py \
       --baseline baseline.json \
       --candidate candidate.json
   '
   ```
3. **Profile the regressed path**: If the suspected regression lives in
   `train_dynamic.py` loop control or observability, reproduce it with a short
   controlled `python -m df_mlx.train_dynamic` run using `sync_mode=profile`.
   Use `benchmark_pipeline.py`, `benchmark_hotspots.py`, or
   `benchmark_sync_barriers.py` only as supporting diagnostics for narrower
   hypotheses.
4. **Small regressions (< 5%)**: Request perf-team review. May be acceptable with
   documented justification.
5. **Large regressions (≥ 5%)**: Block merge until the regression is resolved or an
   equivalent performance improvement is identified elsewhere.
6. **No scripted override**: `scripts/perf_gate.py` does not currently honor
   `BENCHMARK_OVERRIDE` or emit an overridden PASS result. If you need to proceed
   after a noisy run, re-benchmark or record the manual decision outside the gate
   report.

## Report Format

The gate script generates a markdown report:

```
=== Performance Regression Gate ===
Baseline: commit abc1234 (2026-02-13)
Candidate: commit def5678 (2026-02-14)
Hardware: Apple M3 Max (40 cores, 48GB)

| Config | Metric | Baseline | Candidate | Delta | Status |
|--------|--------|----------|-----------|-------|--------|
| full/bs4/compiled/ga1/fp32 | samples/s | 120.5 | 118.2 | -1.9% | PASS |
| full/bs4/eager/ga1/fp32 | step_p95_ms | 33.2 | 38.1 | +14.8% | PASS |
| full/bs8/compiled/ga1/fp32 | samples/s | 230.1 | 195.3 | -15.1% | FAIL |

Result: FAIL (1 regression detected)
```

## Thresholds

| Gate | Condition | Description |
|------|-----------|-------------|
| **Throughput** | `new_p5 < baseline_p5 × 0.90` | Fail if >10% throughput regression at p5 |
| **Tail latency** | `new_p95 > baseline_p95 × 1.15` | Fail if >15% tail latency increase at p95 |
| **Variance** | `CV > 0.20` | Fail if coefficient of variation exceeds 20% |

`step_p99_ms` remains a required review metric for every stage promotion and for
release-candidate sign-off, but the hard fail tail-latency threshold stays the p95
rule above unless this gate document is explicitly revised.

See [BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md) for full rationale.

## References

- [BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md) — Canonical matrix, metadata schema, thresholds
- [benchmark_train_step.py](../DeepFilterNet/df_mlx/benchmark_train_step.py) — Primary promotion benchmark entrypoint
- [benchmark_pipeline.py](../DeepFilterNet/df_mlx/benchmark_pipeline.py) — Secondary data-path diagnostic benchmark
- [benchmark_hotspots.py](../DeepFilterNet/df_mlx/benchmark_hotspots.py) — Secondary hotspot diagnostic benchmark
- [benchmark_sync_barriers.py](../DeepFilterNet/benchmark_sync_barriers.py) — Supplemental sync-barrier microbenchmark
- [scripts/perf_gate.py](../scripts/perf_gate.py) — Gate automation script
- [SYNC_BARRIER_POLICY.md](SYNC_BARRIER_POLICY.md) — Sync modes for profiling
- [IMPLEMENTATION_HANDOFF.md](IMPLEMENTATION_HANDOFF.md) — Existing short `train_dynamic.py` verification runs
- [DATA_PIPELINE_TUNING.md](DATA_PIPELINE_TUNING.md) — Hardware profiles
