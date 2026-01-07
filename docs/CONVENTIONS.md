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
- Never put Python-specific logic in `libDF/`.

**Rationale:**

- Keeps the Rust core portable (WebAssembly, C FFI, standalone CLI).
- Python bindings are a separate concern that shouldn't pollute core algorithms.

**Related Files:**

- [libDF/](../libDF/)
- [pyDF/](../pyDF/)
- [pyDF-data/](../pyDF-data/)

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

## 3. Known Exceptions

_None documented yet._

## 4. Change History (Human-Readable)

- **2025-01-06**: Initial conventions document created during AI integration optimization.
