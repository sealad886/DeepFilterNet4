# MLX JIT Optimization — Implementation Handoff

> **Created:** 2026-02-14
> **Status:** Ready for implementation agent pickup
> **Scope:** Validate, baseline, and production-qualify the MLX JIT optimization
> infrastructure delivered by the planning program.

## Status Summary

| Area | Status |
|------|--------|
| Planning & design (10 child tasks) | **COMPLETE** |
| Code infrastructure | **INTEGRATED** into `run_config.py`, `train_dynamic.py`, `benchmark_train_step.py`, `scripts/perf_gate.py` |
| Test suite | **PASSING** (~170 new tests across 10+ test files) |
| Documentation | **CURRENT** (7 new docs, 5 new conventions) |
| Preset TOML files | **SHIPPED** (`schemas/presets/{entry,pro,max,ultra,debug}.toml`) |

All planning artifacts are design-complete. No further design decisions are
required for Phase 1–4 below. Phase 5 (GAN compile experiment) is explicitly
non-blocking R&D.

---

## Implementation Checklist

### Phase 1: Validation (Day 1)

Confirm that all delivered infrastructure is functional on the target machine.

- [ ] **1.1** Run the full test suite:
  ```bash
  cd DeepFilterNet
  python -m pytest tests/ -q
  ```
  _Exit criterion: 0 failures, 0 errors._

- [ ] **1.2** Run the benchmark contract matrix:
  ```bash
  python -m df_mlx.benchmark_train_step --contract --metadata --json-out logs/validation_run.jsonl
  ```
  _Exit criterion: all 48 config points produce results; no crashes._

- [ ] **1.3** Verify hardware detection:
  ```bash
  python -c "from df_mlx.run_config import detect_hardware_class; print(detect_hardware_class())"
  ```
  _Exit criterion: prints one of `entry`, `pro`, `max`, `ultra`._

- [ ] **1.4** Verify hardware tuning profile:
  ```bash
  python -c "from df_mlx.run_config import get_hardware_tuning_profile; import json; print(json.dumps(get_hardware_tuning_profile(), indent=2))"
  ```
  _Exit criterion: returns a dict with `num_workers`, `prefetch_size`, `use_mlx_data` keys._

- [ ] **1.5** Verify preset loading (one per available hardware class):
  ```bash
  python -m df_mlx.train_dynamic --preset pro --print-run-config
  ```
  _Exit criterion: prints a TOML config with `batch_size`, `sync_mode`, `num_workers` matching the [preset table](RUN_CONFIG_PRESETS.md#preset-table)._

- [ ] **1.6** Verify `auto_tune_dataloader`:
  ```bash
  python -c "
  from df_mlx.run_config import RunConfig, resolve_run_config
  cfg = RunConfig()
  cfg.dataloader.auto_tune_dataloader = True
  resolve_run_config(cfg)
  print(f'workers={cfg.dataloader.num_workers} prefetch={cfg.dataloader.prefetch_size}')
  "
  ```
  _Exit criterion: values match the hardware profile for the detected chip class._

- [ ] **1.7** Confirm no regressions in existing training behavior:
  ```bash
  python -m pytest tests/test_run_config.py tests/test_train_dynamic_gan_config.py \
      tests/test_train_dynamic_cli_toml_parity.py tests/test_train_dynamic_pipeline_stages.py \
      tests/test_checkpoint_resume_dynamic.py -q
  ```
  _Exit criterion: all pass._

**Phase 1 gate:** ALL items green. If any fail, diagnose and fix before
proceeding. Do NOT skip failures.

---

### Phase 2: Baseline Collection (Day 1–2)

Establish the canonical performance baseline on target hardware.

- [ ] **2.1** Run full canonical benchmark (3 repeats × 50 steps × 48 configs):
  ```bash
  cd DeepFilterNet
  python -m df_mlx.benchmark_train_step \
      --contract --metadata \
      --json-out ../logs/baseline.jsonl
  ```
  _Exit criterion: `logs/baseline.jsonl` exists and contains 48 result entries with metadata._

- [ ] **2.2** Validate baseline metadata:
  ```bash
  python -c "
  import json
  with open('logs/baseline.jsonl') as f:
      data = json.load(f)
  meta = data['metadata']
  print(f\"Chip: {meta['hardware']['chip']}\")
  print(f\"MLX: {meta['runtime']['mlx']}\")
  print(f\"Commit: {meta['commit']}\")
  print(f\"Hash: {meta['reproducibility_hash']}\")
  "
  ```
  _Exit criterion: all four fields populated; hash is non-empty._

- [ ] **2.3** Record the `reproducibility_hash` for future comparisons.
  Store in `logs/baseline_metadata.txt`:
  ```bash
  python -c "
  import json
  with open('logs/baseline.jsonl') as f:
      meta = json.load(f)['metadata']
  print(f\"hardware_hash={meta['reproducibility_hash']}\")
  print(f\"chip={meta['hardware']['chip']}\")
  print(f\"commit={meta['commit']}\")
  " > logs/baseline_metadata.txt
  ```

- [ ] **2.4** Spot-check variance: ensure no config point has CV > 20%.
  ```bash
  python -c "
  import json
  with open('logs/baseline.jsonl') as f:
      data = json.load(f)
  for r in data['results']:
      mean = r['metrics']['samples_per_sec_mean']
      std = r['metrics']['samples_per_sec_std']
      cv = std / mean if mean > 0 else 0
      cfg = r['config']
      tag = f\"{cfg['backbone']}/bs{cfg['batch_size']}/{'compiled' if cfg['compiled'] else 'eager'}\"
      if cv > 0.20:
          print(f'WARNING: {tag} CV={cv:.2%} — re-run under quieter conditions')
  print('Variance check complete.')
  "
  ```
  _Exit criterion: no CV > 20% warnings. If any appear, re-run those configs._

**Phase 2 gate:** `logs/baseline.jsonl` exists, all metadata populated,
CV < 20% for all config points.

---

### Phase 3: Integration Testing (Day 2–3)

Test the new infrastructure features in realistic training scenarios.

- [ ] **3.1** `sync_mode=fast` — short training run (1–2 epochs):
  ```bash
  cat > /tmp/fast_test.toml << 'EOF'
  [debug]
  sync_mode = "fast"

  [training]
  epochs = 2
  EOF

  python -m df_mlx.train_dynamic \
      --preset pro \
      --run-config /tmp/fast_test.toml \
      --cache-dir /path/to/audio_cache
  ```
  _Exit criterion: training completes without errors; loss is logged at eval_frequency=50 intervals._

- [ ] **3.2** `sync_mode=debug` — verify full observability:
  ```bash
  python -m df_mlx.train_dynamic \
      --preset debug \
      --run-config /tmp/fast_test.toml \
      --cache-dir /path/to/audio_cache
  ```
  _Exit criterion: loss logged every step (eval_frequency=1); component losses visible._

- [ ] **3.3** Test each hardware preset (where hardware is available):

  | Preset | Test |
  |--------|------|
  | `entry` | `--preset entry --cache-dir ...` (≤16 GB device) |
  | `pro` | `--preset pro --cache-dir ...` (Pro chip) |
  | `max` | `--preset max --cache-dir ...` (Max chip) |
  | `ultra` | `--preset ultra --cache-dir ...` (Ultra chip) |

  _Exit criterion: each preset trains for ≥1 epoch without OOM or crash._

- [ ] **3.4** Verify `auto_tune_dataloader=true` in a real run:
  ```bash
  cat > /tmp/auto_tune.toml << 'EOF'
  [dataloader]
  auto_tune_dataloader = true

  [training]
  epochs = 1
  EOF

  python -m df_mlx.train_dynamic \
      --run-config /tmp/auto_tune.toml \
      --cache-dir /path/to/audio_cache
  ```
  _Exit criterion: log line shows `auto_tune_dataloader: hw_class=... num_workers=... prefetch_size=...`._

- [ ] **3.5** Run the perf regression gate (self-comparison as smoke test):
  ```bash
  python scripts/perf_gate.py \
      --baseline logs/baseline.jsonl \
      --candidate logs/baseline.jsonl \
      --report logs/gate_selfcheck.md
  ```
  _Exit criterion: exit code 0 (self-comparison should always pass)._

- [ ] **3.6** Run compile-boundary shape guardrail tests:
  ```bash
  python -m pytest tests/test_compile_boundary_audit.py -q
  ```
  _Exit criterion: all 16 tests pass._

- [ ] **3.7** Run sync-cadence integration tests:
  ```bash
  python -m pytest tests/test_sync_cadence_integration.py tests/test_sync_barrier_policy.py -q
  ```
  _Exit criterion: all pass._

**Phase 3 gate:** ALL sync modes tested, at least one preset validated on
real hardware, perf gate self-check passes, all integration tests green.

---

### Phase 4: Production Validation (Day 3–5)

Full-duration training to confirm convergence and throughput at scale.

- [ ] **4.1** Full training run (50+ epochs) with optimal preset:
  ```bash
  python -m df_mlx.train_dynamic \
      --preset max \
      --cache-dir /path/to/audio_cache \
      --run-config production.toml
  ```
  Where `production.toml` includes:
  ```toml
  [debug]
  sync_mode = "fast"

  [training]
  epochs = 50
  ```
  _Exit criterion: training completes successfully; no NaN losses; no OOM._

- [ ] **4.2** Verify convergence quality:
  Compare final validation loss against a known-good eager baseline run.
  _Exit criterion: validation loss within 5% of the eager baseline at the same epoch count._

- [ ] **4.3** Run post-training benchmark and perf gate:
  ```bash
  python -m df_mlx.benchmark_train_step \
      --contract --metadata \
      --json-out logs/post_production.jsonl

  python scripts/perf_gate.py \
      --baseline logs/baseline.jsonl \
      --candidate logs/post_production.jsonl \
      --report logs/gate_production.md
  ```
  _Exit criterion: perf gate exit code 0 (no regressions > 10% throughput or > 15% tail latency)._

- [ ] **4.4** Archive results:
  ```bash
  cp logs/baseline.jsonl logs/baseline_$(date +%Y%m%d).jsonl
  cp logs/post_production.jsonl logs/production_$(date +%Y%m%d).jsonl
  cp logs/gate_production.md logs/gate_production_$(date +%Y%m%d).md
  ```

- [ ] **4.5** Document findings:
  Update `docs/BENCHMARKS.md` with production results if they improve on
  existing published numbers.

**Phase 4 gate:** 50+ epoch run completes, convergence verified, perf gate
passes, results archived.

---

### Phase 5: Optional R&D (Non-blocking)

> **This phase does NOT block Phase 1–4 signoff.**
> Results feed into the _next_ planning cycle, not this one.

- [ ] **5.1** Review the GAN compile experiment spec: [GAN_COMPILE_EXPERIMENT.md](GAN_COMPILE_EXPERIMENT.md)
- [ ] **5.2** Enable the feature flag and run Variant A (baseline control):
  ```toml
  [gan]
  experimental_compile = false  # Variant A is the eager control
  ```
- [ ] **5.3** Run Variant B (gen-only compiled) for 20 epochs past GAN activation.
- [ ] **5.4** Compare using the [comparison table template](GAN_COMPILE_EXPERIMENT.md#9-comparison-table-template).
- [ ] **5.5** Write the [recommendation memo](GAN_COMPILE_EXPERIMENT.md#10-recommendation-memo-template).

_Phase 5 has no gate — it is exploratory. Results inform future planning._

---

## Readiness Gates Summary

| Gate | Pass Criteria | Verification Command | Blocks |
|------|---------------|---------------------|--------|
| **Unit Tests** | All tests pass, 0 failures | `cd DeepFilterNet && python -m pytest tests/ -q` | Phase 2+ |
| **Lint** | No black/isort/flake8 violations | Pre-commit hooks (automatic) | All commits |
| **Benchmark Run** | 48 config points, no crashes | `python -m df_mlx.benchmark_train_step --contract --metadata` | Phase 3+ |
| **Variance** | CV < 20% on all config points | Spot-check script (Phase 2.4) | Phase 3+ |
| **Baseline Stored** | `logs/baseline.jsonl` exists with metadata | `ls logs/baseline.jsonl` | Phase 3+ |
| **Perf Gate** | Exit code 0, no regressions above threshold | `python scripts/perf_gate.py --baseline ... --candidate ...` | Phase 4 signoff |
| **Convergence** | Val loss within 5% of eager baseline | Manual comparison | Phase 4 signoff |
| **Documentation** | All docs reflect current behavior | Review `docs/` directory | Phase 4 signoff |

---

## Exit Criteria

The implementation is **complete** when ALL of the following hold:

1. **Phase 1–4** checklist items are checked off.
2. **All readiness gates** in the table above show PASS.
3. Baseline benchmarks are stored in `logs/` with reproducibility metadata.
4. At least one production training run (50+ epochs) completed with
   acceptable quality (convergence within 5% of eager baseline).
5. No outstanding test failures (`pytest tests/ -q` exits 0).
6. Perf regression gate passes (`scripts/perf_gate.py` exit code 0).
7. Git repo is clean and pushed (`git status` shows "up to date with origin").

Phase 5 (GAN compile experiment) is **explicitly excluded** from exit
criteria. It runs on a separate timeline and its results feed into future
planning.

---

## Dependency Graph

```
Phase 1 (Validation)
  │
  ├─── gate: all tests pass
  │
  ▼
Phase 2 (Baseline Collection)
  │
  ├─── gate: baseline.jsonl exists, CV < 20%
  │
  ▼
Phase 3 (Integration Testing)   ←── depends on baseline for perf gate self-check
  │
  ├─── gate: sync modes tested, preset validated, integration tests green
  │
  ▼
Phase 4 (Production Validation) ←── depends on baseline for regression comparison
  │
  ├─── gate: convergence verified, perf gate passes
  │
  ▼
DONE ─── all exit criteria met

Phase 5 (R&D) ←── independent; may start after Phase 4 but does NOT block DONE
```

---

## Key Commands Reference

```bash
# ──── Test Suite ────
cd DeepFilterNet
python -m pytest tests/ -q                          # Full suite
python -m pytest tests/test_compile_boundary_audit.py tests/test_sync_cadence_integration.py \
    tests/test_sync_barrier_policy.py tests/test_benchmark_contract.py \
    tests/test_run_config_presets.py tests/test_perf_regression_gate.py \
    tests/test_gan_compile_experiment.py -q          # JIT-specific tests only

# ──── Benchmarking ────
python -m df_mlx.benchmark_train_step --contract --metadata --json-out logs/baseline.jsonl
python scripts/perf_gate.py --baseline logs/baseline.jsonl --candidate logs/candidate.jsonl --report logs/gate.md

# ──── Presets ────
python -m df_mlx.train_dynamic --preset max --print-run-config
python -m df_mlx.train_dynamic --preset pro --run-config overrides.toml --cache-dir /path/to/cache
python -m df_mlx.train_dynamic --preset debug --cache-dir /path/to/cache  # Full observability

# ──── Hardware Detection ────
python -c "from df_mlx.run_config import detect_hardware_class; print(detect_hardware_class())"
python -c "from df_mlx.run_config import get_hardware_tuning_profile; import json; print(json.dumps(get_hardware_tuning_profile(), indent=2))"

# ──── Sync Modes ────
# Set in run-config TOML under [debug] sync_mode = "fast" | "normal" | "debug" | "profile"
# Or use a preset that sets it: --preset max (fast), --preset debug (debug)
```

---

## Reference Documents

| Document | Purpose | Key Sections |
|----------|---------|-------------|
| [BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md) | 48-point canonical matrix, metadata schema, pass/fail thresholds | Matrix, Metadata Schema, Threshold Policy |
| [SYNC_BARRIER_POLICY.md](SYNC_BARRIER_POLICY.md) | 4 operating modes, sync trigger inventory, metric gating | Operating Modes, Sync Trigger Inventory |
| [COMPILE_BOUNDARY_AUDIT.md](COMPILE_BOUNDARY_AUDIT.md) | Compiled function inventory, retrace risks, shape guardrails | Retrace Risk Inventory, Shape Invariants, Guardrails |
| [DATA_PIPELINE_TUNING.md](DATA_PIPELINE_TUNING.md) | Per-hardware worker/prefetch profiles, auto-tuning | Hardware Profile Table, Auto-Tuning |
| [RUN_CONFIG_PRESETS.md](RUN_CONFIG_PRESETS.md) | 5 TOML presets, `--preset` CLI, override safety guide | Preset Table, Override Safety Guide |
| [PERF_REGRESSION_GATE.md](PERF_REGRESSION_GATE.md) | Gate procedure, triage protocol, thresholds | Gate Procedure, Triage Protocol, Thresholds |
| [GAN_COMPILE_EXPERIMENT.md](GAN_COMPILE_EXPERIMENT.md) | 4-variant experiment matrix, guardrails, abort/success criteria | Experiment Matrix, Guardrails, Success Criteria |
| [CONVENTIONS.md](CONVENTIONS.md) | Repository conventions (5 new JIT-related entries) | Sync Mode Selection, Compile Boundary Shape Invariants, Hardware Profile Presets, Performance Regression Gate, GAN-Phase Eager Mode |
| [BENCHMARKS.md](BENCHMARKS.md) | Historical benchmark results and methodology | Training Performance, Data Loading Performance |

---

## Known Incomplete Items (Future Work)

These items are documented in upstream specs but are **out of scope** for this
implementation pass. They should be tracked as separate tasks.

| Item | Source | Priority |
|------|--------|----------|
| `--sync-mode` CLI flag passthrough | [SYNC_BARRIER_POLICY.md](SYNC_BARRIER_POLICY.md) checklist | Low (TOML config is sufficient) |
| GAN compile experiment _implementation_ | [GAN_COMPILE_EXPERIMENT.md](GAN_COMPILE_EXPERIMENT.md) §11 | R&D (non-blocking) |
| `benchmark_pipeline.py` entrypoint | [DATA_PIPELINE_TUNING.md](DATA_PIPELINE_TUNING.md) | Nice-to-have |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `detect_hardware_class()` returns `entry` on a Pro chip | Missing `sysctl` access or unexpected chip name | Check `sysctl -n machdep.cpu.brand_string` output |
| Benchmark CV > 20% | Noisy environment (background apps, thermal throttling) | Close GPU-heavy apps; allow 30s cooldown between runs |
| Perf gate exit code 2 | Fewer than 3 valid runs after outlier filtering | Re-run benchmark with more repeats |
| OOM on `--preset max` | Insufficient unified memory for batch_size=8 | Use `--preset pro` or `--batch-size 4` |
| `ModuleNotFoundError: mlx_data` | `mlx-data` not installed | `pip install mlx-data` (required for `use_mlx_data=true`) |
| Compile retrace warnings during training | Non-constant batch size or mid-run config change | Ensure `drop_last=True`; do NOT change `max_grad_norm` mid-run |
