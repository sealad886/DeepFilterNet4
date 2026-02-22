# Repository Standards & Conventions

## 1. Scope and Purpose

This file captures non-obvious, repo-specific rules that matter for correctness, maintainability, and team sanity. These are patterns that a new contributor would benefit from knowing explicitly.

For general coding standards (formatting, linting, commit messages), see [CONTRIBUTING.md](../CONTRIBUTING.md).

## 2. Core Conventions

### Model Version Architecture Pattern

**Status:** REQUIRED

**Scope:** All `DeepFilterNet/df/deepfilternet*.py` files

**Rule:**

- Each model version (DFNet, DFNet2, DFNet3, DFNet4) has its own module file following the naming pattern `deepfilternet{N}.py`.
- New model versions extend the architecture via composition, not inheritance from previous versions.
- Model-specific configuration uses the `DfnetConfig` section in config files.

**Rationale:**

- Keeps model architectures isolated and independently testable.
- Allows comparing performance across versions without coupling.
- Prevents regression in older models when experimenting with new approaches.

**Related Files:**

- [DeepFilterNet/df/deepfilternet.py](../DeepFilterNet/df/deepfilternet.py)
- [DeepFilterNet/df/deepfilternet4.py](../DeepFilterNet/df/deepfilternet4.py)

---

### Configuration Hierarchy

**Status:** REQUIRED

**Scope:** All training, evaluation, and inference code

**Rule:**

- Model checkpoints expect a directory containing `config.ini` (or `config.yaml`) plus a `checkpoints/` subdirectory.
- Configuration is loaded via `df.config.DfParams` using INI or YAML parsers.
- Command-line arguments override config file values.

**Rationale:**

- Ensures reproducibility: models are always paired with their training config.
- Standard directory layout allows scripts to auto-discover model parameters.

**Examples:**

- Good: `model_dir/config.ini` + `model_dir/checkpoints/model_0001.pth`
- Bad: Loose `.pth` files without accompanying config

**Related Files:**

- [DeepFilterNet/df/config.py](../DeepFilterNet/df/config.py)
- [DeepFilterNet/df/checkpoint.py](../DeepFilterNet/df/checkpoint.py)

---

### Dual Language Crate Pattern

**Status:** REQUIRED

**Scope:** All Rust crates with Python bindings

**Rule:**

- `libDF/` contains pure Rust DSP and runtime code.
- `pyDF/` wraps `libDF` as Python bindings via PyO3/Maturin.
- `pyDF-data/` provides Rust-backed data loading for training.
- `pyDF-augment/` provides Rust-accelerated augmentation ops (biquad, mix, combine).
- Never put Python-specific logic in `libDF/`.

**Build Rule:** All PyO3 `cdylib` crates (pyDF, pyDF-data, pyDF-augment) use
`pyo3/extension-module` which defers CPython symbol resolution to load-time.
They **must** be excluded from standalone `cargo build --workspace` and built
only via `maturin develop`. The `setup.sh` script enforces this automatically
with `--exclude` flags. When adding a new PyO3 crate, add its package name to
the exclusion list in `setup.sh` and add a `--with-<crate>` maturin flag.

**Rationale:**

- Keeps the Rust core portable (WebAssembly, C FFI, standalone CLI).
- Python bindings are a separate concern that shouldn't pollute core algorithms.
- `extension-module` prevents direct linking against libpython; the Python
  interpreter provides those symbols when loading the `.so`/`.dylib` at runtime.

**Related Files:**

- [libDF/](../libDF/)
- [pyDF/](../pyDF/)
- [pyDF-data/](../pyDF-data/)
- [pyDF-augment/](../pyDF-augment/)

---

### ERB (Equivalent Rectangular Bandwidth) Scale

**Status:** REQUIRED

**Scope:** All spectral feature code

**Rule:**

- Spectral features use ERB-scale compression (default 32 bands) for the encoder.
- DF (Deep Filtering) operates on linear-frequency bins (default 96 lowest bins).
- ERB band count and DF bin count are configurable but defaults should be used unless experimenting.

**Rationale:**

- ERB scale matches human auditory perception, improving model efficiency.
- Linear DF bins focus compute on perceptually important low frequencies.

**Related Files:**

- [DeepFilterNet/df/modules.py](../DeepFilterNet/df/modules.py)
- [docs/ARCHITECTURE.md](ARCHITECTURE.md)

---

### Test Markers for Hardware-Specific Tests

**Status:** REQUIRED

**Scope:** All pytest tests

**Rule:**

- Use `@pytest.mark.mps` for Apple Silicon (MPS) specific tests.
- Tests that require GPU should be skippable via markers or environment checks.

**Rationale:**

- CI runs on Linux CPU instances; MPS tests would fail.
- Allows developers on Apple Silicon to run the full test suite with appropriate filtering.

**Examples:**

- Good: `@pytest.mark.mps` on tests that use `torch.device("mps")`
- Bad: Unconditionally creating MPS tensors in shared test fixtures

**Related Files:**

- [DeepFilterNet/tests/](../DeepFilterNet/tests/)
- [DeepFilterNet/pyproject.toml](../DeepFilterNet/pyproject.toml)

---

### Issue Tracking with bd (beads)

**Status:** REQUIRED

**Scope:** All AI agents and contributors

**Rule:**

- Use `bd` for issue tracking, not GitHub Issues directly.
- Run `bd prime` at session start for workflow context.
- Run `bd sync` before ending a work session.
- Reference issue IDs in commit messages when applicable.

**Rationale:**

- Git-backed issues stay with the repository and work offline.
- AI agents get structured context injection via bd.
- See `.claude/skills/beads/` for comprehensive AI integration patterns.

**Related Files:**

- [.beads/](../.beads/)
- [.claude/skills/beads/](../.claude/skills/beads/)
- [AGENTS.md](../AGENTS.md)

---

### Training Counter Semantics (MLX Dynamic Trainer)

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py`, `DeepFilterNet/df_mlx/dynamic_dataset.py`, and related checkpoint/resume tests

**Rule:**

- Treat epoch progress in **micro-batches** (dataloader iterations), and treat `global_step` in **optimizer steps** (parameter updates).
- In checkpoint state, `batch_idx` represents **micro-batches completed in the current epoch** (count), not the 0-based index of the last seen batch.
- Progress bars must use micro-batch totals and iterate over a bounded iterator to avoid overshooting epoch boundaries.
- When both model and data checkpoints are present for in-progress resume, their `(epoch, micro-batch)` positions must match exactly; otherwise fail loudly.

**Rationale:**

- Mixing units (optimizer-step totals vs micro-batch iteration) causes misleading progress bars and apparent epoch overruns.
- Count-vs-index ambiguity in `batch_idx` causes off-by-one resume behavior and can duplicate or skip data after interruption.
- Strict model/data checkpoint reconciliation prevents silent divergence during recovery and makes incidents reproducible.

**Related Files:**

- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)
- [DeepFilterNet/df_mlx/dynamic_dataset.py](../DeepFilterNet/df_mlx/dynamic_dataset.py)
- [DeepFilterNet/tests/test_train_control_semantics.py](../DeepFilterNet/tests/test_train_control_semantics.py)

---

### Epoch-Boundary Training Mode Switch (Compiled → Eager with GAN)

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` and trainer control-flow tests

**Rule:**

- Determine training mode once per epoch (before the batch loop), never mid-epoch.
- Allow compiled mode only while GAN is inactive and compiled-step base constraints are satisfied.
- Once GAN-active epochs begin, switch to eager mode and do not switch back to compiled mode later in the run.
- Emit explicit mode markers (`TRAIN_MODE=COMPILED` / `TRAIN_MODE=EAGER`) and fail fast if a GAN-active epoch tries to run the compiled step.

**Rationale:**

- Prevents mixed execution semantics within a single epoch and keeps checkpoint/resume behavior deterministic.
- Preserves pre-GAN performance gains from compiled training while ensuring GAN/discriminator updates run on the eager path.
- One-way switching avoids subtle resume-dependent mode oscillation and reduces incident surface area.

**Related Files:**

- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)
- [DeepFilterNet/tests/test_train_control_semantics.py](../DeepFilterNet/tests/test_train_control_semantics.py)

---

### Sync Mode Selection

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` and run-config TOML files

**Rule:**

- Every training run must specify a `sync_mode` (`fast`, `normal`, `debug`, or `profile`) that controls eval barrier frequency and metric verbosity.
- When `sync_mode` is set and `eval_frequency` is at its default, `eval_frequency` is automatically overridden to the mode's recommended value (`fast`→50, `normal`→10, `debug`→1, `profile`→5).
- An explicit `eval_frequency` in the TOML or CLI always takes precedence over the mode default.

**Rationale:**

- Prevents accidental performance loss from too-frequent synchronization in production runs.
- Provides a single knob that coordinates eval frequency, logging verbosity, and sync budget so developers don't need to tune multiple flags independently.

**Examples:**

- Set `sync_mode = "fast"` for throughput-critical production runs (eval every 50 steps).
- Set `sync_mode = "debug"` when investigating training anomalies (eval every step, full observability).

```toml
[debug]
sync_mode = "fast"
```

**Related Files:**

- [docs/SYNC_BARRIER_POLICY.md](SYNC_BARRIER_POLICY.md)
- [docs/RUN_CONFIG_PRESETS.md](RUN_CONFIG_PRESETS.md)

---

### Compile Boundary Shape Invariants

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` — all `mx.compile()`-wrapped functions

**Rule:**

- All inputs to `mx.compile()`-wrapped functions must have fixed shapes within a run.
- `drop_last=True` is mandatory on all data loaders feeding compiled paths to prevent tail batches with different shapes.
- Shape changes trigger expensive retraces that silently degrade throughput.

**Rationale:**

- MLX's `mx.compile()` traces the computation graph for a given set of input shapes. A shape change forces a full retrace, which is as expensive as the first compilation and silently destroys any throughput gains.
- Enforcing `drop_last=True` eliminates the most common source of shape variation (short tail batches at epoch boundaries).

**Examples:**

- Good: `PrefetchDataLoader(dataset, batch_size=4, drop_last=True)`
- Bad: `PrefetchDataLoader(dataset, batch_size=4, drop_last=False)` — the last batch may have fewer samples, triggering a retrace.

**Related Files:**

- [docs/COMPILE_BOUNDARY_AUDIT.md](COMPILE_BOUNDARY_AUDIT.md)
- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)

---

### Hardware Profile Presets

**Status:** STRONGLY RECOMMENDED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` training configuration

**Rule:**

- Use `--preset` with the appropriate hardware class (`entry`, `pro`, `max`, `ultra`, `debug`) instead of manually tuning worker/prefetch/batch parameters.
- Presets encode tested, hardware-specific defaults that balance throughput and memory.
- Preset values can still be overridden by explicit CLI flags or run-config TOML entries.

**Rationale:**

- Manual tuning of `num_workers`, `prefetch_size`, `batch_size`, and `sync_mode` is error-prone and requires hardware-specific knowledge.
- Presets were validated against the benchmark contract matrix and represent known-good configurations for each Apple Silicon tier.

**Examples:**

```bash
# For M3 Max hardware
python -m df_mlx.train_dynamic --preset max --run-config my_config.toml

# For debugging on any hardware
python -m df_mlx.train_dynamic --preset debug --run-config my_config.toml
```

**Related Files:**

- [docs/RUN_CONFIG_PRESETS.md](RUN_CONFIG_PRESETS.md)
- [docs/DATA_PIPELINE_TUNING.md](DATA_PIPELINE_TUNING.md)

---

### Performance Regression Gate

**Status:** REQUIRED

**Scope:** Changes to training loop, data pipeline, model code in `DeepFilterNet/df_mlx/`

**Rule:**

- Any performance-sensitive change must pass the benchmark regression gate before merge.
- The gate compares candidate benchmark results against a known baseline using the thresholds defined in the benchmark contract.
- Regressions exceeding the threshold require explicit justification or mitigation before merge.

**Rationale:**

- Prevents silent throughput degradation that accumulates across multiple changes.
- The automated gate (`scripts/perf_gate.py`) provides objective pass/fail signals, removing subjective judgment from performance review.

**Examples:**

```bash
# Generate baseline, then candidate, then compare
python -m df_mlx.benchmark_train_step --contract --metadata --json-out baseline.jsonl
python -m df_mlx.benchmark_train_step --contract --metadata --json-out candidate.jsonl
python scripts/perf_gate.py --baseline baseline.jsonl --candidate candidate.jsonl
```

**Related Files:**

- [docs/PERF_REGRESSION_GATE.md](PERF_REGRESSION_GATE.md)
- [docs/BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md)
- [scripts/perf_gate.py](../scripts/perf_gate.py)

---

### GAN-Phase Eager Mode

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` — GAN training phase

**Rule:**

- GAN-active training must use eager mode unless `gan.experimental_compile=true` is explicitly set in the run-config TOML.
- The compiled→eager mode switch at GAN activation is one-way and irreversible within a run.
- The `gan.experimental_compile` flag gates an R&D experiment path and is not recommended for production training.

**Rationale:**

- Discriminator updates introduce additional sync barriers (`mx.eval()` per discriminator step) that are incompatible with the current compile boundary.
- One-way switching prevents resume-dependent mode oscillation and keeps checkpoint semantics deterministic.
- The experimental compile flag provides a controlled path for future R&D without relaxing the default safety constraint.

**Examples:**

```toml
# Default (recommended): GAN phase uses eager mode
[gan]
start_epoch = 10
# experimental_compile defaults to false

# R&D only: attempt compiled GAN phase
[gan]
start_epoch = 10
experimental_compile = true
```

**Related Files:**

- [docs/GAN_COMPILE_EXPERIMENT.md](GAN_COMPILE_EXPERIMENT.md)
- [docs/COMPILE_BOUNDARY_AUDIT.md](COMPILE_BOUNDARY_AUDIT.md)
- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)

---

### GAN Discriminator Update Cadence with Gradient Accumulation

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` — GAN-active training loop

**Rule:**

- When gradient accumulation is enabled, discriminator updates must follow **optimizer-step cadence**, not micro-batch cadence.
- `gan.disc_update_freq` is interpreted in optimizer steps (`global_step`), consistent with trainer counter semantics.
- GAN discriminator waveform tensors should use model precision (FP16 when enabled), while MRSTFT may remain FP32 for numeric stability.

**Rationale:**

- Running discriminator updates per micro-batch under accumulation multiplies adversarial memory/compute pressure and can trigger OOM at GAN activation.
- Aligning cadence to optimizer steps preserves intended update semantics and avoids surprise behavior from micro-batch internals.
- Mixed precision on GAN discriminator paths reduces peak memory at GAN onset without changing MRSTFT stability policy.

**Related Files:**

- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)
- [DeepFilterNet/tests/test_gan_memory_path.py](../DeepFilterNet/tests/test_gan_memory_path.py)

---

## 3. Known Exceptions

_None documented yet._

## 4. Change History (Human-Readable)

- **2025-01-06**: Initial conventions document created during AI integration optimization.
- **2026-02-13**: Added MLX training counter semantics convention (micro-batch vs optimizer-step units, checkpoint resume invariants).
- **2026-02-13**: Added epoch-boundary compiled→eager training mode convention for delayed GAN activation.
- **2026-02-14**: Added sync mode selection, compile boundary shape invariants, hardware profile presets, performance regression gate, and GAN-phase eager mode conventions.
- **2026-02-15**: Added GAN discriminator optimizer-step cadence and GAN mixed-precision waveform-path conventions to prevent GAN-onset OOM under gradient accumulation.
- **2026-02-15**: Added Augmentation Extension Fallback Architecture convention documenting `augment_ext.py` bridge pattern and `pyDF-augment` Rust extension.

---

### Augmentation Extension Fallback Architecture

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/augment_ext.py`, `DeepFilterNet/df_mlx/dynamic_dataset.py`, `pyDF-augment/`

**Rule:**

- All augmentation operations (`biquad_filter`, `mix_audio`, `combine_noises`) MUST route through the bridge module `df_mlx.augment_ext`.
- Callers must never check `_RUST_AVAILABLE` directly — the bridge handles Rust/Python dispatch internally.
- When the Rust extension `libdfaugment` is installed, the bridge uses accelerated Rust implementations; otherwise it falls back transparently to pure-Python/SciPy.

**Extension Location:** `pyDF-augment/` — a Maturin-based Rust crate exposing `libdfaugment`.

**Building the Extension:**
```bash
cd pyDF-augment && maturin develop --release
```

**How Fallback Works:**
- At import time, `augment_ext.py` attempts `from libdfaugment import ...`.
- If the import succeeds, `_RUST_AVAILABLE = True` and Rust functions are used.
- If the import fails, `_RUST_AVAILABLE = False` and Python fallbacks are used.
- The dispatch is transparent to callers — API signatures are identical.

**Accelerated Operations:**

| Operation | Rust backend | Fallback backend |
|-----------|-------------|-----------------|
| `biquad_filter` | `libdfaugment.biquad_filter` | `scipy.signal.lfilter` |
| `mix_audio` | `libdfaugment.mix_audio` | NumPy arithmetic |
| `combine_noises` | `libdfaugment.combine_noises` | NumPy arithmetic |

**Checking Active Backend:**
```python
from df_mlx.augment_ext import augment_capabilities, rust_augment_available
print(rust_augment_available())   # True/False
print(augment_capabilities())     # {'rust_extension': False, 'biquad_backend': 'scipy', ...}
```

**Rationale:**

- Centralised dispatch avoids scattered `try/except ImportError` blocks across the codebase.
- Pure-Python fallbacks guarantee the training pipeline always works, even without compiling the Rust extension.
- A single bridge module makes it easy to add new accelerated operations.

**Related Files:**

- [DeepFilterNet/df_mlx/augment_ext.py](../DeepFilterNet/df_mlx/augment_ext.py)
- [DeepFilterNet/df_mlx/dynamic_dataset.py](../DeepFilterNet/df_mlx/dynamic_dataset.py)
- [DeepFilterNet/tests/test_guarded_fallback.py](../DeepFilterNet/tests/test_guarded_fallback.py)
- `pyDF-augment/` (Rust extension crate)

---

### Metal Kernels: Differentiable via Custom VJP

- **Short Name:** metal-kernel-custom-vjp
- **Status:** REQUIRED
- **Scope:** All `df_mlx/` code that calls `mx.fast.metal_kernel`

**Rule:** Custom Metal kernels (`mx.fast.metal_kernel`) must be wrapped with
`mx.custom_function` and a corresponding `.vjp` decorator so they are fully
differentiable and can execute on the `nn.value_and_grad` computation graph
during training.

**Implementation pattern:**

```python
@mx.custom_function
def _my_custom(input_a, input_b):
    """Forward: dispatch the Metal kernel."""
    return _my_metal_kernel_dispatch(input_a, input_b)

@_my_custom.vjp
def _my_vjp(primals, cotangents, _outputs):
    """Backward: pure-MLX gradient computation."""
    a, b = primals
    d_out = cotangents
    # ... compute d_a, d_b using standard MLX ops ...
    return d_a, d_b
```

**Rationale:**

- MLX's `mx.fast.metal_kernel` creates opaque `CustomKernel` primitives with no
  automatic VJP.  Wrapping with `mx.custom_function` provides a custom backward
  pass so the Metal kernel can be used in both training and inference.
- Forward passes use the fused Metal kernel for speed; backward passes use
  pure-MLX ops derived from the mathematical chain rule.
- All three kernels (DfOp gather+CMAC, iSTFT overlap-add, mel power+log) have
  VJP implementations verified against pure-MLX fallback gradients via tests.
- The `_dfop_fallback` / vectorized-overlap-add / power+mel+log pure-MLX paths
  remain as fallbacks for platforms without Metal kernel support.

**Related Files:**

- [DeepFilterNet/df_mlx/kernels.py](../DeepFilterNet/df_mlx/kernels.py) — Metal kernel source + VJP implementations
- [DeepFilterNet/df_mlx/modules.py](../DeepFilterNet/df_mlx/modules.py) — `DfOp` uses Metal kernel in all modes
- [DeepFilterNet/df_mlx/ops.py](../DeepFilterNet/df_mlx/ops.py) — `istft` Metal kernel path (differentiable)
- [DeepFilterNet/df_mlx/dnsmos_proxy.py](../DeepFilterNet/df_mlx/dnsmos_proxy.py) — `MelSpectrogram` uses Metal kernel in all modes
- [DeepFilterNet/tests/test_metal_kernel_training_guard.py](../DeepFilterNet/tests/test_metal_kernel_training_guard.py) — VJP correctness tests

---

### Training Module Structure

**Status:** REQUIRED
**Scope:** All `DeepFilterNet/df_mlx/training_*.py` modules and `train_dynamic.py`

**Rule:** Training code is organized into focused, single-responsibility modules
extracted from the original `train_dynamic.py` monolith. New training code must
be placed in the appropriate module rather than added back to `train_dynamic.py`.

**Module Layout:**

| Module | Responsibility |
|--------|---------------|
| `training_losses.py` | Loss constants and `_compute_*` loss functions |
| `training_checkpoints.py` | `CheckpointManifest`, `CheckpointRecord`, save/load/prune |
| `training_cli.py` | CLI flag parsing, pipeline-stage resolution, CLI overrides |
| `training_ops.py` | `NumericDebugger`, `NumericDebugConfig`, gradient utilities |
| `training_signals.py` | Signal handlers (SIGINT, SIGUSR1) and graceful-stop flags |
| `training_waveform.py` | Waveform synthesis and GAN waveform utilities |
| `training_cli_main.py` | `main()` entry point: argparse, config layering, `train()` dispatch |
| `training_session.py` | `TrainingSession` facade: `from_run_config()`, `setup()`, `run()` |
| `train_dynamic.py` | `train()` loop, re-export shim, dataset/model setup |

**Re-export Policy:**

- `train_dynamic.py` re-exports all public symbols from the `training_*` modules
  so that existing external imports continue to work.
- **New code** should import from the canonical `training_*.py` module directly
  (e.g., `from df_mlx.training_losses import _compute_stft_loss`).
- Re-exports exist solely for backward compatibility and will not be extended
  with new symbols.

**Naming Conventions:**

- Internal loss functions use the `_compute_*` prefix (e.g., `_compute_multi_res_stft_loss`).
- Private helpers use a leading underscore; public API symbols do not.
- Module names use `training_` prefix to group them alphabetically.

**Rationale:**

The original `train_dynamic.py` grew to 7 631 lines. Decomposition into focused
modules enables targeted testing, faster code navigation, clearer ownership, and
reduces merge-conflict surface area. The re-export shim preserves backward
compatibility while the codebase migrates to canonical imports.

**Related Files:**

- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py) — re-export shim + `train()` loop
- [DeepFilterNet/df_mlx/training_losses.py](../DeepFilterNet/df_mlx/training_losses.py) — loss functions
- [DeepFilterNet/df_mlx/training_checkpoints.py](../DeepFilterNet/df_mlx/training_checkpoints.py) — checkpoint management
- [DeepFilterNet/df_mlx/training_cli.py](../DeepFilterNet/df_mlx/training_cli.py) — CLI parsing
- [DeepFilterNet/df_mlx/training_ops.py](../DeepFilterNet/df_mlx/training_ops.py) — numeric debug + gradient ops
- [DeepFilterNet/df_mlx/training_signals.py](../DeepFilterNet/df_mlx/training_signals.py) — signal handling
- [DeepFilterNet/df_mlx/training_waveform.py](../DeepFilterNet/df_mlx/training_waveform.py) — waveform utilities
- [DeepFilterNet/df_mlx/training_cli_main.py](../DeepFilterNet/df_mlx/training_cli_main.py) — main entry point
- [DeepFilterNet/df_mlx/training_session.py](../DeepFilterNet/df_mlx/training_session.py) — TrainingSession class
- [DeepFilterNet/tests/test_train_dynamic_reexports.py](../DeepFilterNet/tests/test_train_dynamic_reexports.py) — re-export coverage test
