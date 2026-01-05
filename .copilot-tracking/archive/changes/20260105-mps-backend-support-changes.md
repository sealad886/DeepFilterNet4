<!-- markdownlint-disable-file -->
# Release Changes: MPS Backend Support Enhancement

**Related Plan**: 20260105-mps-backend-support-plan.instructions.md
**Implementation Date**: 2026-01-05

## Summary

Add comprehensive Apple Silicon MPS backend support with runtime detection, graceful degradation for unsupported operations, documentation, and validation testing.

## Changes

### Added

- DeepFilterNet/df/utils.py - Added `get_macos_version()` function to detect macOS version for MPS compatibility checking
- DeepFilterNet/df/utils.py - Added `mps_supports_complex()` function to verify macOS 14+ for complex tensor operations
- DeepFilterNet/df/utils.py - Added `mps_safe_norm()` utility function for safe norm computation with MPS CPU fallback for complex tensors
- DeepFilterNet/tests/conftest.py - Created pytest configuration with MPS fixtures (`mps_device`, `mps_complex_device`) and auto-skip for MPS tests on non-macOS
- DeepFilterNet/tests/test_mps.py - Created comprehensive MPS test suite with device detection, safe norm, and DfOp forward method tests

### Modified

- DeepFilterNet/df/utils.py - Updated `get_device()` with runtime warning for MPS on macOS < 14 and comprehensive docstring documenting PYTORCH_ENABLE_MPS_FALLBACK
- DeepFilterNet/df/modules.py - Updated `DfOp.__init__()` to auto-select `forward_real_unfold` when complex ops unsupported on MPS, with DF_USE_COMPLEX config override
- DeepFilterNet/df/stoi.py - Added MPS compatibility note documenting that torch.norm calls operate on real tensors
- README.md - Added comprehensive Apple Silicon (MPS) Support section with requirements, usage examples, compatibility table, performance expectations, and troubleshooting guide
- DeepFilterNet/pyproject.toml - Added pytest.ini_options with MPS marker registration for test suite

### Removed

## Release Summary

**Total Files Affected**: 7

### Files Created (2)

- DeepFilterNet/tests/conftest.py - Pytest configuration with MPS fixtures and markers
- DeepFilterNet/tests/test_mps.py - Comprehensive MPS test suite (15 tests)

### Files Modified (5)

- DeepFilterNet/df/utils.py - Added MPS detection utilities and safe norm function
- DeepFilterNet/df/modules.py - DfOp auto-selects forward method based on MPS support
- DeepFilterNet/df/stoi.py - Added MPS compatibility documentation
- README.md - Added MPS usage section with compatibility table and troubleshooting
- DeepFilterNet/pyproject.toml - Registered MPS pytest marker

### Files Removed (0)

None

### Dependencies & Infrastructure

- **New Dependencies**: None
- **Updated Dependencies**: None
- **Infrastructure Changes**: None
- **Configuration Updates**: Added `tool.pytest.ini_options` with MPS marker

### Deployment Notes

MPS backend requires:
- macOS 14 (Sonoma) or later for full complex tensor operation support
- PyTorch 2.5.1+ with MPS enabled
- Apple Silicon Mac (M1/M2/M3/M4)

For older macOS versions, set `PYTORCH_ENABLE_MPS_FALLBACK=1` or use `--device cpu`.
