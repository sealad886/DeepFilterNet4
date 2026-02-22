# GAN-Phase Compile Experiment

> **Status:** OPTIONAL R&D — does NOT block the primary integration path
> **Date:** 2026-02-14
> **Feature flag:** `gan.experimental_compile` (default `false`)
> **Related:**
> - [CONVENTIONS.md](CONVENTIONS.md) — epoch-boundary mode switch convention
> - [COMPILE_BOUNDARY_AUDIT.md](COMPILE_BOUNDARY_AUDIT.md) — compiled step inventory
> - [SYNC_BARRIER_POLICY.md](SYNC_BARRIER_POLICY.md) — sync trigger budgets per mode

## 1. Objective

Evaluate whether partial compilation during GAN-active training can improve
throughput without sacrificing convergence quality. The current convention
enforces a one-way compiled → eager switch at GAN activation
(`docs/CONVENTIONS.md`). This experiment explores relaxing that constraint
under controlled conditions.

## 2. Background

- **Current behavior:** compiled mode runs pre-GAN; once `gan.start_epoch` is
  reached, the trainer switches to eager and never returns to compiled mode.
- **Motivation:** the compiled step yields measurable throughput gains; if parts
  of the GAN loop (e.g., generator forward/backward) can remain compiled, the
  overall training wall-time may decrease.
- **Risk:** the discriminator update at L4983 performs a per-step `mx.eval()`
  sync (see `COMPILE_BOUNDARY_AUDIT.md` §2, risk #8). Wrapping both generator
  and discriminator in a single compiled graph may invalidate that sync or
  introduce retrace instability.

## 3. Experiment Matrix

| Variant | Generator | Discriminator | Compile Scope | Risk Level |
|---------|-----------|---------------|---------------|------------|
| **A: Baseline** | Eager | Eager | None | None (control) |
| **B: Gen-only compiled** | Compiled (gen fwd+bwd) | Eager | Generator loss+grad only | Medium |
| **C: Full compiled** | Compiled (gen+disc) | Compiled | Both paths | High |
| **D: Alternating** | Compiled (gen steps) | Eager (disc steps) | Generator steps only | Medium |

### Variant Details

- **A (Baseline):** current production path — full eager mode after GAN
  activation. Acts as the control for all comparisons.
- **B (Gen-only compiled):** compile only the generator loss+grad computation;
  discriminator update remains eager with its existing per-step sync. The
  generator compiled graph excludes `disc_loss` and `fm_loss` branches.
- **C (Full compiled):** compile both the generator and discriminator forward
  and backward passes into a single graph. Highest risk due to loss of the
  per-step disc sync barrier.
- **D (Alternating):** on generator-update steps, use a compiled generator
  step; on discriminator-update steps, fall back to eager. Compile scope
  alternates per step, which may trigger frequent retraces if MLX doesn't
  cache both graph variants.

## 4. Feature Flag

```toml
[gan]
experimental_compile = false  # Enable experimental GAN-phase compilation (R&D only)
```

- **Default:** `false` — existing eager-only GAN path is preserved.
- **When `true`:** unlocks the experimental compiled-GAN code path (once
  implemented). The specific variant is selected by an additional config
  key (TBD during implementation).
- **Scope:** R&D only. This flag MUST remain off in production configs until
  the experiment produces a positive recommendation.

## 5. Correctness Guardrails

All guardrails apply when `gan.experimental_compile = true`. Any single
violation triggers an immediate abort back to eager mode.

| # | Guardrail | Threshold | Action |
|---|-----------|-----------|--------|
| 1 | **Loss divergence** | `gen_loss` or `disc_loss` > 10× initial value | Abort |
| 2 | **Gradient explosion** | `grad_norm > 100` for 5 consecutive steps | Abort |
| 3 | **Discriminator balance** | `disc_accuracy < 10%` or `> 90%` for a full epoch | Abort |
| 4 | **NaN/Inf detector** | Any non-finite value in loss or gradients | Abort immediately |
| 5 | **Convergence comparison** | Final validation loss > 5% above Variant A baseline | Flag for review |

### Guardrail Constants

```python
LOSS_DIVERGENCE_FACTOR = 10.0
GRAD_NORM_EXPLOSION_THRESHOLD = 100.0
GRAD_NORM_EXPLOSION_WINDOW = 5
DISC_ACCURACY_LOW = 0.10
DISC_ACCURACY_HIGH = 0.90
CONVERGENCE_TOLERANCE = 0.05
THROUGHPUT_MIN_RATIO = 0.80
PESQ_SISDR_MAX_DROP = 0.10
```

## 6. Abort Criteria

Any of the following triggers an immediate abort of the experiment variant:

1. **Loss divergence:** `gen_loss` or `disc_loss` exceeds 10× the value
   recorded at the first GAN-active step, within the first 5 GAN epochs.
2. **Gradient explosion:** `grad_norm > 100` for 5 or more consecutive
   optimizer steps.
3. **NaN/Inf:** any non-finite value detected in loss, gradients, or model
   parameters.
4. **Throughput regression:** training throughput drops below 80% of the
   Variant A (eager) baseline, indicating compile overhead dominates.
5. **Spectral quality divergence:** PESQ or SI-SDR metrics drop more than 10%
   compared to Variant A at the same epoch.

On abort, the trainer:
- Logs the abort reason and the step/epoch at which it occurred.
- Falls back to eager mode for the remainder of the run.
- Writes abort metadata to `ablation_metrics.jsonl`.

## 7. Success Criteria

A variant is recommended for adoption if ALL of the following hold:

- **Throughput:** ≥ 15% improvement in samples/s over Variant A.
- **Convergence:** final validation loss within 5% of Variant A.
- **Stability:** zero aborts triggered across all experiment runs.
- **Quality:** PESQ and SI-SDR within 5% of Variant A on the evaluation set.

## 8. Measurement Protocol

1. Run all four variants for **20 epochs past GAN activation**.
2. Use identical seeds, data, and hyperparameters across variants.
3. Record per-step:
   - Throughput (samples/s)
   - Generator loss, discriminator loss
   - Generator and discriminator gradient norms
   - Discriminator accuracy
4. Record per-epoch:
   - Validation loss
   - PESQ and SI-SDR on the evaluation set (if available)
5. Output:
   - Per-variant loss/throughput curves (CSV + plots)
   - Comparison table (see §9)
   - Recommendation memo (see §10)

## 9. Comparison Table Template

| Metric | A (Baseline) | B (Gen-only) | C (Full) | D (Alternating) |
|--------|-------------|-------------|---------|-----------------|
| Throughput (samples/s) | — | — | — | — |
| Throughput Δ vs A | — | — | — | — |
| Final val loss | — | — | — | — |
| Val loss Δ vs A | — | — | — | — |
| Aborts triggered | — | — | — | — |
| PESQ | — | — | — | — |
| SI-SDR | — | — | — | — |

## 10. Recommendation Memo Template

```
GAN-Phase Compile Experiment — Recommendation

Based on {N} experiment runs over {E} epochs post-GAN activation:

Variant {X} results:
  - Throughput: {Y} samples/s ({Z}% vs baseline)
  - Final validation loss: {L} ({within/outside} 5% tolerance)
  - Stability: {stable/unstable}, {N_aborts} aborts triggered
  - PESQ: {P} ({within/outside} 5% tolerance)
  - SI-SDR: {S} ({within/outside} 5% tolerance)

Recommendation: {ADOPT / REJECT / FURTHER STUDY}

If ADOPT:
  - Required guardrails for production use: {list}
  - Proposed convention change: {description}
  - Migration path: {steps}

If REJECT:
  - Rationale: {explanation}
  - The current eager-only GAN convention remains unchanged.

If FURTHER STUDY:
  - Open questions: {list}
  - Proposed follow-up experiments: {description}
```

## 11. Implementation Notes

- **Variant B (gen-only compiled) is implemented** in `train_dynamic.py`.
  The implementation creates a separate `loss_fn_gan` with GAN generator
  paths always active (hardcoded True), avoiding the `mx.compile` trace-time
  Python boolean problem where `gan_active` would be captured as False.
- Implementation reuses the existing `compiled_step` and
  `compiled_loss_and_grad_step` patterns from pre-GAN training
  (`COMPILE_BOUNDARY_AUDIT.md` §1), with GAN-specific variants
  (`compiled_gan_step`, `compiled_gan_loss_and_grad_step`).
- The discriminator update remains fully eager with its existing per-step
  `mx.eval()` sync, unchanged from production.
- A one-time correctness verification runs on the first compiled-GAN step:
  it compares the compiled loss against an eager forward pass and logs
  PASS/FAIL with tolerances (rtol=1e-4, atol=1e-5).
- The `resolve_epoch_train_mode()` function accepts
  `experimental_compiled_gan` to relax the one-way COMPILED→EAGER invariant.
- All experiment code is gated behind `gan.experimental_compile` and
  removable without affecting the production path.
- Variants C and D are **NOT yet implemented** and would require further work.
