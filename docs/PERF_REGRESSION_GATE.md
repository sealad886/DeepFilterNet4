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
hardening — uses the same primary authority:
`python -m df_mlx.benchmark_train_step --contract --metadata`.
`benchmark_pipeline.py` and `benchmark_hotspots.py` may be run to diagnose a
stage-specific regression or validate a hypothesis, but they never replace the
train-step benchmark for merge or promotion decisions.

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

## Gate Procedure

1. **Checkout baseline** (main branch HEAD or tagged release).
2. Run the canonical benchmark:
   ```bash
   cd DeepFilterNet
   python -m df_mlx.benchmark_train_step \
       --contract --metadata \
       --cache-dir /path/to/audio_cache \
       --json-out baseline.jsonl
   ```
3. **Checkout candidate** branch.
4. Run the same benchmark:
   ```bash
   python -m df_mlx.benchmark_train_step \
       --contract --metadata \
       --cache-dir /path/to/audio_cache \
       --json-out candidate.jsonl
   ```
5. **Compare** with the perf gate script:
   ```bash
   python scripts/perf_gate.py \
       --baseline baseline.jsonl \
       --candidate candidate.jsonl \
       --report gate_report.md
   ```
6. Review the generated report. Exit code 0 = pass, 1 = fail, 2 = error.

## Reproducibility Controls

- **Same hardware**: Baseline and candidate must run on the same machine. The gate
  script compares hardware metadata hashes and warns on mismatch.
- **Thermal settle**: Allow a 30-second cooldown between benchmark runs. The benchmark
  script inserts this automatically between repeats.
- **Close resource-heavy apps**: Browsers, IDEs, and other GPU-consuming applications
  should be closed during benchmark runs.
- **3 trials, use median**: Each config point runs 3 independent repeats. The median
  is used for comparison to reduce outlier impact.
- **Metadata audit trail**: Every run records chip, GPU cores, memory, OS version,
  Python version, MLX version, and git commit. This metadata is stored in the JSONL
  output for post-hoc auditing.

## Variance Policy

- **CV > 20%**: Results are unreliable. Re-run with additional warmup steps or under
  quieter conditions (fewer background processes, cooler hardware).
- **Outlier detection**: Drop any run where a metric deviates more than 3σ from the
  group median. This prevents a single thermal-throttle spike from poisoning results.
- **Minimum valid runs**: At least 3 valid runs must remain after outlier filtering.
  If fewer survive, the gate reports an error (exit code 2) and the benchmark must be
  re-run.

## Triage Protocol

When the gate fails:

1. **Check environment drift**: Compare hardware metadata between baseline and
   candidate. If `reproducibility_hash` differs, re-run on matching hardware.
2. **Isolate the regression commit**: Use `git bisect` with the benchmark script:
   ```bash
   git bisect start <bad-commit> <good-commit>
   git bisect run python scripts/perf_gate.py \
       --baseline baseline.jsonl \
       --candidate <(python -m df_mlx.benchmark_train_step --contract --metadata --json-out /dev/stdout)
   ```
3. **Profile the regressed path**: Run the benchmark with `sync_mode=profile` to
   collect detailed timing breakdowns.
4. **Small regressions (< 5%)**: Request perf-team review. May be acceptable with
   documented justification.
5. **Large regressions (≥ 5%)**: Block merge until the regression is resolved or an
   equivalent performance improvement is identified elsewhere.
6. **Override escape hatch**: Set `BENCHMARK_OVERRIDE=1` with a documented
   justification in the PR description. Overrides are logged in the report and must
   never be used for release baselines.

## Report Format

The gate script generates a markdown report:

```
=== Performance Regression Gate ===
Baseline: commit abc1234 (2026-02-13)
Candidate: commit def5678 (2026-02-14)
Hardware: Apple M3 Max (40 cores, 48GB)

| Config | Metric | Baseline | Candidate | Delta | Status |
|--------|--------|----------|-----------|-------|--------|
| dfnet4/bs4/compiled | samples/s | 120.5 | 118.2 | -1.9% | PASS |
| dfnet4/bs4/eager | step_p95_ms | 33.2 | 38.1 | +14.8% | PASS |
| dfnet4/bs8/compiled | samples/s | 230.1 | 195.3 | -15.1% | FAIL |

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
- [scripts/perf_gate.py](../scripts/perf_gate.py) — Gate automation script
- [SYNC_BARRIER_POLICY.md](SYNC_BARRIER_POLICY.md) — Sync modes for profiling
- [DATA_PIPELINE_TUNING.md](DATA_PIPELINE_TUNING.md) — Hardware profiles
