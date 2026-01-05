<!-- markdownlint-disable-file -->
# Release Changes: DeepFilterNet Fork Modernization

**Related Plan**: 20260105-deepfilternet-modernization-plan.instructions.md
**Implementation Date**: 2026-01-05

## Summary

Modernize DeepFilterNet fork to support the latest PyTorch ecosystem (PyTorch 2.5+, TorchAudio 2.9+, Python 3.13) by integrating improvements from community PRs #670, #648, #653, #666, and #619.

## Changes

### Added

- DeepFilterNet/df/io.py - Added `get_audio_metadata()` function providing TorchCodec fallback for TorchAudio 2.9+ compatibility (Tasks 1.3)
- DeepFilterNet/df/io.py - Added version detection constants `TORCHAUDIO_VERSION`, `USE_TORCHCODEC`, `HAS_TORCHCODEC` (Task 1.2)
- DeepFilterNet/df/io.py - Added fallback `AudioMetaData` dataclass for TorchAudio 2.9+ when native class not available (Task 1.4)

### Modified

- DeepFilterNet/df/io.py - Updated `load_audio()` to use new `get_audio_metadata()` instead of direct `ta.info()` call (Task 1.3)
- DeepFilterNet/df/io.py - Updated AudioMetaData import with three-tier fallback: torchaudio → torchaudio.backend.common → dataclass (Task 1.4)
- pyDF/Cargo.toml - Updated PyO3 from 0.20 to 0.22 with `gil-refs` feature for backward compatibility (Task 2.1)
- pyDF/Cargo.toml - Updated numpy from 0.20 to 0.22 (Task 2.1)
- pyDF-data/Cargo.toml - Updated PyO3 from 0.20 to 0.22 with `gil-refs` feature for backward compatibility (Task 2.2)
- pyDF-data/Cargo.toml - Updated numpy from 0.20 to 0.22 (Task 2.2)
- pyDF/src/lib.rs - Migrated `#[pymodule]` signature from `&PyModule` to `&Bound<'_, PyModule>` (Task 2.3)
- pyDF/src/lib.rs - Updated all array return types from `&'py PyArrayX<T>` to `Bound<'py, PyArrayX<T>>` for numpy 0.22 API (Task 2.4)
- pyDF/src/lib.rs - Replaced `into_pyarray(py)` with `into_pyarray_bound(py)` throughout (Task 2.4)
- pyDF/src/lib.rs - Added `PyArrayMethods` and `PyUntypedArrayMethods` trait imports for numpy 0.22 (Task 2.4)
- pyDF-data/src/lib.rs - Migrated `#[pymodule]` signature from `&PyModule` to `&Bound<'_, PyModule>` (Task 2.3)
- pyDF-data/src/lib.rs - Updated `FdBatch` type alias from reference types to `Bound<'py, PyArrayX<T>>` (Task 2.4)
- pyDF-data/src/lib.rs - Updated `get_batch()` to use `into_pyarray_bound(py)` throughout (Task 2.4)
- Cargo.lock - Updated to PyO3 0.22.6 and numpy 0.22.1 (Task 2.6)

### Removed


### Notes

**Deviation from plan (Task 2.1, 2.2):** Original plan specified PyO3 0.25 and numpy 0.25, but PyO3 0.23+ removed the `gil-refs` feature required for backward compatibility. Used PyO3 0.22 with `gil-refs` feature instead, which supports Python 3.13 per the PyO3 changelog.

**Task 2.5 (libdf.pyi):** No changes needed - the .pyi stub file documents the Python API which is unchanged. The internal Rust type changes (`&'py PyArray` → `Bound<'py, PyArray>`) don't affect the Python type hints.

**Task 2.7 (Python 3.13+ build):** Tested with Python 3.14 using `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` flag. Both pyDF and pyDF-data wheels built successfully.

## Phase 3 Changes

### Modified

- DeepFilterNet/df/checkpoint.py - Added `weights_only=False` to `torch.load()` call to fix FutureWarning on PyTorch 2.5+ (Task 3.1)
- DeepFilterNet/df/enhance.py - Changed `audio.numpy()` to `audio.detach().cpu().numpy()` in `df_features()` to handle CUDA tensors (Task 3.2)

## Phase 4 Changes

### Added

- DeepFilterNet/df/enhance.py - Added `--device` / `-D` CLI argument to `setup_df_argument_parser()` for explicit device selection (Task 4.1)
- DeepFilterNet/df/enhance.py - Added `device` parameter to `init_df()` function signature with docstring update (Task 4.3)
- DeepFilterNet/df/enhance.py - Added `device` parameter to `enhance()` function signature with docstring update (Task 4.3)

### Modified

- DeepFilterNet/df/utils.py - Updated `get_device()` to accept optional `device` parameter for explicit device specification (Task 4.2, 4.3)
- DeepFilterNet/df/utils.py - Added MPS device detection for Apple Silicon support in auto-detection logic (Task 4.2)
- DeepFilterNet/df/enhance.py - Updated `init_df()` to pass device parameter through to `get_device()` (Task 4.3)
- DeepFilterNet/df/enhance.py - Updated `enhance()` to pass device parameter through to `get_device()` (Task 4.3)
- DeepFilterNet/df/enhance.py - Updated `main()` to extract and pass device argument to `init_df()` and `enhance()` (Task 4.3)

## Phase 5 Changes

### Modified

- DeepFilterNet/pyproject.toml - Updated Python version constraint from `>=3.8,<4.0` to `>=3.9,<4.0` (Task 5.1)
- DeepFilterNet/pyproject.toml - Updated tool.black target-version to `["py39", "py310", "py311", "py312", "py313"]` (Task 5.1)
- DeepFilterNet/pyproject.toml - Updated poe tasks from PyTorch 2.1 to 2.5.1 with CUDA 121/124 options (Task 5.4)
- DeepFilterNet/pyproject.toml - Added `install-torch-mps` poe task for Apple Silicon (Task 5.4)

### Added

- DeepFilterNet/pyproject.toml - Added `torchcodec` optional dependency `{version = ">=0.1", optional = true}` (Task 5.3)
- DeepFilterNet/pyproject.toml - Added `torchcodec` extra: `torchcodec = ["torchcodec"]` (Task 5.3)

### Modified

- DeepFilterNet/requirements.txt - Updated torch version constraint from `>=2.0` to `>=2.5` (Task 5.2, 5.5)
- DeepFilterNet/requirements.txt - Updated torchaudio version constraint from `>=2.0` to `>=2.5` (Task 5.2, 5.5)
- DeepFilterNet/requirements.txt - Updated numpy version constraint from `>=1.20` to `>=1.22` (Task 5.5)

### Notes

**Task 5.2 deviation:** PyTorch/TorchAudio are not listed as poetry dependencies in pyproject.toml; they're installed via poe tasks. Updated poe tasks to install PyTorch 2.5.1 and requirements.txt minimum to 2.5.

## Phase 6 Changes

### Added

- README.md - Added "Fork Notice" section at top documenting modernization improvements (Task 6.1)
- README.md - Added "Version Compatibility" table with Python/PyTorch/TorchAudio/TorchCodec versions (Task 6.3)
- README.md - Added "TorchAudio 2.9+ Users" section with TorchCodec/FFmpeg installation instructions (Task 6.2)

### Modified

- DeepFilterNet/df/utils.py - Added exception handling for `get_device()` when config is not loaded (Task 6.4 - discovered during testing)

### Notes

**Task 6.4 (testing):** Verified core functionality:
- All modified modules import successfully (df.io, df.checkpoint, df.enhance, df.utils)
- TorchCodec fallback works with TorchAudio 2.9.1 (`USE_TORCHCODEC=True`, `HAS_TORCHCODEC=True`)
- Audio loading via TorchCodec works correctly
- Device selection works with auto-detection and explicit device specification
- MPS detection works on Apple Silicon
- Existing test suite has pre-existing issues unrelated to modernization (missing fixtures, missing directories)

---

## Release Summary

**Total Files Affected**: 12

### Files Created (0)

(No new files created - all changes are modifications to existing files)

### Files Modified (12)

- DeepFilterNet/df/io.py - TorchAudio 2.9 compatibility with TorchCodec fallback
- DeepFilterNet/df/checkpoint.py - PyTorch 2.5+ weights_only fix
- DeepFilterNet/df/enhance.py - CUDA tensor handling and device selection
- DeepFilterNet/df/utils.py - MPS device detection and explicit device selection
- DeepFilterNet/pyproject.toml - Python 3.9+ requirement, torchcodec optional, updated poe tasks
- DeepFilterNet/requirements.txt - PyTorch 2.5+ minimum versions
- pyDF/Cargo.toml - PyO3 0.22, numpy 0.22
- pyDF/src/lib.rs - numpy 0.22 API migration (Bound types, into_pyarray_bound)
- pyDF-data/Cargo.toml - PyO3 0.22, numpy 0.22
- pyDF-data/src/lib.rs - numpy 0.22 API migration (Bound types, into_pyarray_bound)
- Cargo.lock - Updated dependencies locked to PyO3 0.22.6, numpy 0.22.1
- README.md - Fork notice, version compatibility, TorchCodec documentation

### Files Removed (0)

(No files removed)

### Dependencies & Infrastructure

- **New Dependencies**:
  - `torchcodec>=0.1` (optional, for TorchAudio 2.9+ users)
  - `packaging>=23,<25` (for version detection)
  
- **Updated Dependencies**:
  - PyO3: 0.20 → 0.22.6 (with `gil-refs` feature)
  - numpy (Rust): 0.20 → 0.22.1
  - Python: >=3.8 → >=3.9
  - PyTorch: 2.1 → 2.5+ (poe tasks)
  - TorchAudio: 2.1 → 2.5+ (poe tasks)

- **Infrastructure Changes**:
  - Added `install-torch-mps` poe task for Apple Silicon
  - Updated `install-torch-cuda11` → `install-torch-cuda121` and `install-torch-cuda124`
  
- **Configuration Updates**:
  - Black target versions updated for Python 3.9-3.13

### Deployment Notes

- Users with TorchAudio 2.9+ must install TorchCodec: `pip install torchcodec`
- TorchCodec requires FFmpeg libraries (`ffmpeg`, `libavcodec-dev`, `libavformat-dev`)
- Python 3.14 builds require `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` environment variable
- PyO3 0.22 is used instead of 0.25 due to `gil-refs` requirement (PyO3 0.23+ removed `gil-refs`)

### Verified Functionality

- ✅ DeepFilterNet modules import on Python 3.14
- ✅ TorchAudio 2.9 audio loading via TorchCodec
- ✅ Device auto-detection (CUDA → MPS → CPU)
- ✅ Explicit device selection via `--device` CLI argument
- ✅ MPS detection on Apple Silicon
- ✅ pyDF and pyDF-data wheels build successfully with maturin
