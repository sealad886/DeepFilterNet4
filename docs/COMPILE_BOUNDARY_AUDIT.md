# Compile Boundary & Shape Stability Audit

> **Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` — MLX compiled training steps  
> **Date:** 2026-02-14  
> **Status:** Complete

## 1. Compiled Functions Inventory

The training loop defines **two compile boundaries** inside `train()`:

### 1.1 `compiled_step` (fwd + bwd + optimizer update)

| Property | Value |
|----------|-------|
| **Location** | `train_dynamic.py` ~L3778 |
| **Decorator** | `@partial(mx.compile, inputs=state, outputs=state)` |
| **Captured state** | `[model.state, optimizer.state]` |
| **Arguments (14)** | `noisy_real`, `noisy_imag`, `feat_erb`, `feat_spec`, `clean_real`, `clean_imag`, `snr`, `vad_weight`, `speech_weight`, `awesome_weight`, `vad_reg_weight`, `gan_weight`, `fm_weight`, `max_grad_norm_val` |
| **Returns** | `(loss, out)` |
| **Used when** | `epoch_use_compiled_step=True` AND `grad_accumulation_steps == 1` |

### 1.2 `compiled_loss_and_grad_step` (fwd + bwd only)

| Property | Value |
|----------|-------|
| **Location** | `train_dynamic.py` ~L3825 |
| **Decorator** | `@partial(mx.compile, inputs=[model.state], outputs=[model.state])` |
| **Captured state** | `[model.state]` |
| **Arguments (13)** | `noisy_real`, `noisy_imag`, `feat_erb`, `feat_spec`, `clean_real`, `clean_imag`, `snr`, `vad_weight`, `speech_weight`, `awesome_weight`, `vad_reg_weight`, `gan_weight`, `fm_weight` |
| **Returns** | `(loss, out, grads)` |
| **Used when** | `epoch_use_compiled_step=True` AND `grad_accumulation_steps > 1` |

**Key difference:** `compiled_step` includes `max_grad_norm_val` and calls `optimizer.update` inside the graph. `compiled_loss_and_grad_step` omits optimizer update so that gradient accumulation can happen eagerly outside the compiled graph.

## 2. Retrace Risk Inventory

| # | Risk Factor | Location | Current Mitigation | Risk Level | Recommendation |
|---|-------------|----------|--------------------|------------|----------------|
| 1 | Variable batch size | DataLoader | `drop_last=True` (default on both `PrefetchDataLoader` and `MLXDataStream`) | **LOW** | Shape assertion guard added ✅ |
| 2 | Loss branch flags | `compiled_step` args (`vad_weight`, `speech_weight`, etc.) | Resolved as `mx.array` before compile; traced as graph inputs, not Python bools | **MINIMAL** | Document as invariant (this doc) |
| 3 | Grad clip branch | `compiled_step` body: `if max_grad_norm_val > 0` | Python-level condition at trace time; value constant within a run | **MINIMAL** | Document as invariant; do NOT change `max_grad_norm` mid-run |
| 4 | Dynamic weight scaling | Loss weight arrays (`awesome_weight_mx`, etc.) | Passed as `mx.array` graph inputs each step | **MINIMAL** | Safe: values change per step but shapes/dtypes are constant |
| 5 | Model architecture change | State capture list | Model architecture is constant within a run | **NONE** | N/A |
| 6 | FP16 dtype mismatch | Input tensors | `use_fp16` cast applied once per batch before compile boundary | **MINIMAL** | Dtype assertion guard added ✅ |
| 7 | Train/valid shape divergence | Validation loop | Validation uses separate loader, not compiled step | **NONE** | N/A — validation always runs eagerly |
| 8 | GAN mode switch | Epoch-boundary mode transition | One-way compiled→eager switch; compiled never re-entered after GAN activation (see `docs/CONVENTIONS.md`) | **NONE** | Existing convention is sufficient |

### Overall Assessment

**Retrace risk is LOW.** The main potential source — variable batch size — is mitigated by `drop_last=True` defaults and now additionally by a runtime shape assertion before compiled function entry. All other traced branches use constant Python values or `mx.array` graph inputs that change value but not shape.

## 3. Shape Invariants

The following shape contracts must hold at every compiled step invocation:

```
noisy_real.shape == (B, F, T)  == clean_real.shape
noisy_imag.shape == (B, F, T)  == clean_imag.shape
feat_erb.shape   == (B, E, T)
feat_spec.shape  == (B, D, T)
snr.shape        == (B,)

Where:
  B = batch_size (constant within a run, enforced by drop_last=True)
  F = n_freqs = fft_size // 2 + 1
  T = time_steps = ceil(segment_samples / hop_size)
  E = nb_erb
  D = nb_df
```

All scalar weight arguments (`vad_weight`, `speech_weight`, etc.) are `mx.array` with shape `()` and dtype `float32`. `max_grad_norm_val` (in `compiled_step` only) is a Python `float` constant.

## 4. Guardrails Implemented

### 4.1 `_assert_compile_boundary_shapes()`

**Location:** `train_dynamic.py`, defined inside `train()` before the compiled step definitions.

Validates at every compiled step invocation:
1. `noisy.shape[0] == expected_batch_size` — prevents batch-dimension retrace
2. `noisy.shape == clean.shape` — prevents shape mismatch between paired inputs
3. `noisy.dtype == expected_dtype` — prevents dtype-induced retrace (optional, enabled when `use_fp16=True`)

Raises `ValueError` with diagnostic message on violation.

### 4.2 `_log_compile_retrace_warning()`

**Location:** Same scope as above.

Logs a numbered warning via `tqdm.write()` when a retrace is detected. Tracks cumulative retrace count via closure variable `_compile_retrace_count`.

### 4.3 Call Site

The shape assertion is invoked immediately before the `if grad_accumulation_steps > 1` branch inside the `epoch_use_compiled_step` block, ensuring both `compiled_step` and `compiled_loss_and_grad_step` are covered by the same guard.

## 5. Compile Warmup

MLX's `mx.compile` traces lazily on first invocation. The first batch of each run (or mode transition) incurs a one-time tracing cost. No explicit warmup pass is currently performed.

**Recommendation:** For benchmarking scenarios, consider adding a single dummy forward pass before timing begins. For training, the overhead is negligible relative to total epoch time and is not worth the complexity of a warmup pass.

## 6. Testing

Test file: `DeepFilterNet/tests/test_compile_boundary_audit.py` (16 tests)

| Test Class | Coverage |
|------------|----------|
| `TestAssertCompileBoundaryShapes` | Valid shapes accepted; batch mismatch raises; shape mismatch raises; dtype mismatch raises; dtype check disabled; fp16 consistent passes |
| `TestGuardrailFunctionsExist` | Both guardrail functions exist in source; retrace warning has `context` param |
| `TestCompiledFunctionSignatures` | Both compiled functions exist; arg counts match (14 and 13) |
| `TestDropLastDefaults` | `PrefetchDataLoader.drop_last=True`; `MLXDataStream.drop_last=True` |
| `TestShapeAssertionIntegration` | Shape assertion appears before compiled step call in source |

## 7. Related Conventions

- **Epoch-Boundary Training Mode Switch** (`docs/CONVENTIONS.md`): Compiled mode is only used pre-GAN; one-way switch to eager prevents retrace risk from mode oscillation.
- **Counter Semantics** (`docs/CONVENTIONS.md`): Micro-batch counters and optimizer-step `global_step` are orthogonal to compile boundaries.
