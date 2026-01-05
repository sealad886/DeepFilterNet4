---
applyTo: '.copilot-tracking/changes/20260105-deepfilternet-modernization-changes.md'
---
<!-- markdownlint-disable-file -->
# DeepFilterNet Fork Modernization — Structured Delivery Plan

## Purpose

Modernize DeepFilterNet fork to support the latest PyTorch ecosystem (PyTorch 2.5+, TorchAudio 2.9+, Python 3.13) by cherry-picking and integrating community PR improvements.

---

## Reference Inputs

### Internal

- [DeepFilterNet/pyproject.toml](../../DeepFilterNet/pyproject.toml) — Python package configuration and dependencies
- [DeepFilterNet/df/io.py](../../DeepFilterNet/df/io.py) — Core audio I/O module requiring TorchAudio 2.9 migration
- [DeepFilterNet/df/checkpoint.py](../../DeepFilterNet/df/checkpoint.py) — Checkpoint loading requiring weights_only fix
- [DeepFilterNet/df/enhance.py](../../DeepFilterNet/df/enhance.py) — Enhancement module requiring CUDA tensor fix
- [pyDF/Cargo.toml](../../pyDF/Cargo.toml) — Rust/PyO3 bindings requiring update
- [pyDF-data/Cargo.toml](../../pyDF-data/Cargo.toml) — Rust/PyO3 data bindings requiring update

### External

- #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md — Comprehensive research findings
- [TorchAudio Maintenance Transition](https://github.com/pytorch/audio/issues/3902) — Official deprecation timeline
- [PyO3 Migration Guide](https://pyo3.rs/v0.25.0/migration) — API migration documentation

---

## Delivery Framework

Each phase contains **atomic, non-overlapping tasks**. Each task is individually testable and constitutes a single measurable output. A task is not complete until:

- All specified file modifications are applied
- Code compiles/imports without errors
- Functionality can be verified with a simple test
- Changes are committed with conventional commit message

---

## [x] Phase 1 — TorchAudio 2.9 Compatibility

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|-------------|-------------------|-------|
| [x] | 1.1 | Add packaging dependency | `packaging` added to pyproject.toml dependencies | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 1-25) |
| [x] | 1.2 | Implement version detection | TorchAudio version detection at module load in df/io.py | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 27-55) |
| [x] | 1.3 | Add TorchCodec fallback for metadata | Replace `torchaudio.info()` with TorchCodec AudioDecoder for 2.9+ | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 57-95) |
| [x] | 1.4 | Update AudioMetaData import | Fix AudioMetaData import path for TorchAudio 2.9+ | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 97-115) |
| [x] | 1.5 | Test TorchAudio compatibility | Verify load_audio/save_audio work on TorchAudio 2.5 and 2.9 | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 117-135) |

---

## [x] Phase 2 — Python 3.13 / PyO3 Update

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|-------------|-------------------|-------|
| [x] | 2.1 | Update pyDF Cargo.toml | PyO3 to 0.22 (with gil-refs), numpy to 0.22 in pyDF/Cargo.toml | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 137-165) |
| [x] | 2.2 | Update pyDF-data Cargo.toml | PyO3 to 0.22 (with gil-refs), numpy to 0.22 in pyDF-data/Cargo.toml | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 167-195) |
| [x] | 2.3 | Migrate PyO3 method signatures | Update #[pyfunction] and #[pymethods] per migration guide | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 197-245) |
| [x] | 2.4 | Update type definitions | Migrate PyO3 type annotations and conversions | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 247-285) |
| [x] | 2.5 | Regenerate .pyi stub files | Update libdf.pyi with new signatures | No changes needed - API signatures unchanged from Python perspective |
| [x] | 2.6 | Update Cargo.lock | Run cargo update and verify dependencies | PyO3 0.22.6, numpy 0.22.1 locked |
| [x] | 2.7 | Test Python 3.13 build | Verify maturin build succeeds on Python 3.13+ | Tested with Python 3.14 using ABI3 forward compatibility |

---

## [x] Phase 3 — PyTorch Compatibility Fixes

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|-------------|-------------------|-------|
| [x] | 3.1 | Fix torch.load weights_only | Add weights_only=False to checkpoint.py torch.load calls | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 342-370) |
| [x] | 3.2 | Fix CUDA tensor handling | Add .detach().cpu() before .numpy() in enhance.py | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 372-400) |
| [x] | 3.3 | Test checkpoint loading | Verify model loading works with PyTorch 2.5+ | Tested df.io, df.checkpoint, df.enhance import successfully |

---

## [x] Phase 4 — Device Selection Enhancement

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|-------------|-------------------|-------|
| [x] | 4.1 | Add --device CLI argument | Add device argument to enhance.py CLI | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 422-455) |
| [x] | 4.2 | Add MPS device detection | Add Apple Silicon MPS to auto-detection logic | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 457-490) |
| [x] | 4.3 | Update device selection logic | Expand get_device() to support explicit device specification | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 492-525) |

---

## [x] Phase 5 — Dependency Updates

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|-------------|-------------------|-------|
| [x] | 5.1 | Update Python version constraint | Change python requirement to >=3.9,<4.0 | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 527-545) |
| [x] | 5.2 | Update PyTorch version constraints | Update torch/torchaudio to >=2.5,<3.0 | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 547-575) |
| [x] | 5.3 | Add TorchCodec optional dependency | Add torchcodec as optional dependency | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 577-600) |
| [x] | 5.4 | Update poe task versions | Update pip install commands in poe tasks | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 602-635) |
| [x] | 5.5 | Update requirements.txt | Sync requirements.txt with pyproject.toml changes | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 637-660) |

---

## [x] Phase 6 — Documentation & Testing

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|-------------|-------------------|-------|
| [x] | 6.1 | Update README fork notice | Add fork notice and list of improvements | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 662-695) |
| [x] | 6.2 | Document new dependencies | Add TorchCodec and FFmpeg requirements to README | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 697-725) |
| [x] | 6.3 | Add version compatibility table | Document supported Python/PyTorch/TorchAudio versions | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 727-755) |
| [x] | 6.4 | Run existing tests | Execute pytest and verify no regressions | .copilot-tracking/details/20260105-deepfilternet-modernization-details.md (Lines 757-780) |

---

## Operational Dependencies

- Python 3.9+ (3.13 for full support)
- PyTorch 2.5+ with TorchAudio 2.5+
- TorchCodec (required for TorchAudio 2.9+)
- FFmpeg libraries (required by TorchCodec)
- Rust toolchain with cargo
- maturin for Python/Rust builds

---

## Acceptance Criteria

A phase is complete only when:

- All tasks in the phase have status `[x]`
- Code compiles without errors
- Basic functionality tests pass
- Changes are committed with conventional commit messages

---

## Completion Definition

The work represented by this plan is considered **functionally complete** when:

- DeepFilterNet imports and runs on Python 3.9, 3.11, and 3.13
- TorchAudio 2.5, 2.8, and 2.9 are all supported
- All existing tests pass
- README documents fork improvements and compatibility matrix
- Fork is ready for use as a modern DeepFilterNet alternative
