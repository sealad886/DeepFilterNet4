# Sync Barrier Policy

> Defines operating modes, sync trigger budgets, and metric classification for
> `train_dynamic.py` on the MLX backend.

## Operating Modes

| Mode | `sync_mode` | `eval_frequency` | Sync Budget | Use Case |
|------|-------------|-------------------|-------------|----------|
| **Fast (production)** | `fast` | 50 | Minimize: only periodic eval + checkpoint + epoch boundary | Maximum throughput |
| **Normal (default)** | `normal` | 10 | Balanced: regular loss readout, moderate overhead | Development |
| **Debug** | `debug` | 1 | Maximum: every-step sync for full observability | Debugging issues |
| **Profile** | `profile` | 5 | Instrumented: sync + timing metadata | Performance analysis |

### Mode Selection

Set via `debug.sync_mode` in the run-config TOML:

```toml
[debug]
sync_mode = "fast"
```

When `sync_mode` is set and `training.eval_frequency` is still at the default (10),
`eval_frequency` is automatically overridden to the mode's recommended value.
An explicit `eval_frequency` in the TOML or CLI always takes precedence.

### Interaction with Debug Flags

- `debug.debug_numerics = true` forces eager mode regardless of `sync_mode`.
- `debug.nan_skip_batch = true` forces eager mode regardless of `sync_mode`.
- These flags do **not** change `eval_frequency`; they affect the execution mode
  (compiled vs eager) independently.

## Sync Trigger Inventory

All `mx.eval()` call sites in `train_dynamic.py`, classified by cost and mode
applicability.

### One-Time Triggers

| Trigger | Location | Mode Applicability | Cost | Purpose |
|---------|----------|-------------------|------|---------|
| Weight init | L2301 | All | One-time | Materialize model params |
| Disc init | L2376 | All (if GAN) | One-time | Materialize disc params |
| Training end | L5720 | All | One-time | Final sync before exit |

### Per-Step / Per-eval_frequency Triggers

| Trigger | Location | Mode Applicability | Cost | Purpose |
|---------|----------|-------------------|------|---------|
| Compiled periodic (loss+params+opt) | L4818 | Normal, Fast | Per-eval_freq | Loss readout + param/opt sync |
| Compiled periodic (loss only) | L4820 | Normal, Fast | Per-eval_freq | Loss readout (no accum) |
| Compiled state sync | L4843 | Normal, Fast | Per-eval_freq | Accumulated state sync |
| Eager periodic (loss) | L4901 | Debug, Normal | Per-eval_freq | Loss readout |
| Eager periodic (params+opt) | L4933 | Debug, Normal | Per-eval_freq | Param + optimizer sync |
| Disc step | L4983 | All (if GAN) | Per-step | Sync disc params |
| Validation forward pass | L4225 | All | Per-validate_every | Materialize validation audio |

### Periodic / Boundary Triggers

| Trigger | Location | Mode Applicability | Cost | Purpose |
|---------|----------|-------------------|------|---------|
| Steps checkpoint | L5413 | All | Per-save | Ensure consistent state for save |
| Epoch end | L5447 | All | Per-epoch | Accurate epoch loss accumulation |

## Metric Classification

Metrics are classified by sync cost and gated by the operating mode.

### Free (No Sync Required)

| Metric | Sync Required | Mode Threshold |
|--------|---------------|----------------|
| `lr` | No (Python value) | All modes |
| `epoch` | No (Python counter) | All modes |
| `global_step` | No (Python counter) | All modes |

### Lightweight (Scalar Sync)

| Metric | Sync Required | Mode Threshold |
|--------|---------------|----------------|
| `loss_val` | Yes (scalar) | All modes |
| `grad_norm` | Yes (scalar) | Normal, Debug |
| `samples_per_sec` | Timing only | All modes |

### Medium (Multiple Scalar Syncs)

| Metric | Sync Required | Mode Threshold |
|--------|---------------|----------------|
| Component losses (spec, vad, etc.) | Yes (multiple scalars) | Normal, Debug |
| `music_suppression` | Forward pass | Normal, Debug |
| `mask_saturation` | Forward pass | Normal, Debug |

### Expensive (Full Tensor / Validation Sync)

| Metric | Sync Required | Mode Threshold |
|--------|---------------|----------------|
| Weight histogram | Full param sync | Debug only |
| Gradient histogram | Full grad sync | Debug only |
| SI-SDR eval | Full validation pass | Per `validate_every` |

## Implementation Checklist

- [x] Document sync barrier policy (`docs/SYNC_BARRIER_POLICY.md`)
- [x] Add `sync_mode` field to `DebugConfig` in `run_config.py`
- [x] Add `resolve_run_config()` to apply mode-based `eval_frequency` override
- [x] Add tests for sync_mode choices, overrides, and debug_numerics interaction
- [x] Wire `resolve_run_config()` into `train_dynamic.py` startup (future task)
- [x] Gate medium/expensive metrics behind mode checks in training loop (future task)
- [ ] Add `--sync-mode` CLI flag passthrough (future task)
