<!-- markdownlint-disable-file -->
# Task Details: MPS Backend Support Enhancement

## Research Reference

**Source Research**: #file:../research/20260713-mps-backend-support-research.md

---

## Phase 1: Runtime Detection and Warnings

### Task 1.1: macOS Version Detection Utility

Add a utility function to detect macOS version for MPS compatibility checking.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df/utils.py` - Add `get_macos_version()` function

- **Implementation**:
  ```python
  import platform
  from typing import Optional, Tuple

  def get_macos_version() -> Optional[Tuple[int, int]]:
      """Get macOS version as (major, minor) tuple, or None if not macOS."""
      if platform.system() != "Darwin":
          return None
      try:
          version = platform.mac_ver()[0]
          parts = version.split(".")
          return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
      except (ValueError, IndexError):
          return None
  ```

- **Success**:
  - Function returns (14, 0) or higher on macOS Sonoma+
  - Function returns None on Linux/Windows
  - No external dependencies added

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 173-176) - Technical requirements

- **Dependencies**: None (first task)

---

### Task 1.2: MPS Complex Operation Compatibility Check

Add function to verify macOS version supports complex tensor operations on MPS.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df/utils.py` - Add `mps_supports_complex()` function

- **Implementation**:
  ```python
  def mps_supports_complex() -> bool:
      """Check if MPS backend supports complex tensor operations.
      
      Complex operations require macOS 14 (Sonoma) or later.
      Returns True if on macOS 14+ or if not using MPS.
      """
      if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
          return True  # Not using MPS, so no restriction
      macos_version = get_macos_version()
      if macos_version is None:
          return True  # Not macOS, shouldn't happen if MPS available
      return macos_version[0] >= 14
  ```

- **Success**:
  - Returns True on macOS 14+
  - Returns True on non-MPS platforms
  - Returns False on macOS 12/13 with MPS available

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 108-111) - macOS version requirement

- **Dependencies**: Task 1.1 completion

---

### Task 1.3: Runtime Warning for Incompatible Configurations

Add warning when MPS is selected but complex operations may fail.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df/utils.py` - Update `get_device()` function
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df/modules.py` - Add warning in DfOp initialization

- **Implementation**:
  In `get_device()`, after MPS detection:
  ```python
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      if not mps_supports_complex():
          logger.warning(
              "MPS backend detected on macOS < 14. Complex tensor operations may fail. "
              "Consider using --device cpu or set PYTORCH_ENABLE_MPS_FALLBACK=1"
          )
      return torch.device("mps")
  ```

- **Success**:
  - Warning logged on macOS < 14 with MPS
  - No warning on macOS 14+ or CPU/CUDA
  - Warning includes actionable guidance

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 57-59) - MPS fallback env var

- **Dependencies**: Task 1.2 completion

---

### Task 1.4: Environment Variable Documentation

Document the PYTORCH_ENABLE_MPS_FALLBACK environment variable in code comments.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df/utils.py` - Add docstring to `get_device()`

- **Implementation**:
  Update `get_device()` docstring:
  ```python
  def get_device(device: Optional[str] = None) -> torch.device:
      """Get the compute device for model inference.
      
      Args:
          device: Explicit device string ('cpu', 'cuda', 'cuda:0', 'mps').
                  If None, auto-detects in order: CUDA → MPS → CPU.
      
      Returns:
          torch.device for model placement.
      
      Note:
          MPS backend requires macOS 14+ for complex tensor operations.
          On older macOS, set PYTORCH_ENABLE_MPS_FALLBACK=1 to fall back
          to CPU for unsupported operations (slower but compatible).
      """
  ```

- **Success**:
  - Docstring includes MPS requirements
  - Environment variable documented
  - Help text accessible via `help(get_device)`

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 57-59) - Fallback documentation

- **Dependencies**: Task 1.3 completion

---

## Phase 2: Graceful Degradation Implementation

### Task 2.1: DfOp Forward Method Selection Logic

Auto-select real-valued forward method when complex operations are unsupported.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df/modules.py` - Update `DfOp` class

- **Implementation**:
  In `DfOp.__init__()`, add logic to select forward method:
  ```python
  def __init__(self, ...):
      ...
      # Select forward method based on MPS complex support
      self._use_complex = mps_supports_complex()
      if not self._use_complex:
          logger.info("Using real-valued forward method for MPS compatibility")
  
  def forward(self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None) -> Tensor:
      if self._use_complex:
          return self.forward_complex_strided(spec, coefs, alpha)
      else:
          return self.forward_real_unfold(spec, coefs, alpha)
  ```

- **Success**:
  - `forward_real_unfold` used on macOS < 14 with MPS
  - `forward_complex_strided` used on macOS 14+ or CPU/CUDA
  - Info log indicates which method selected

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 10-14) - Forward method alternatives

- **Dependencies**: Phase 1 completion

---

### Task 2.2: STOI Evaluation CPU Fallback

Ensure STOI computation works by moving complex tensors to CPU before norm.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df/stoi.py` - Update norm computation

- **Implementation**:
  In lines using `torch.norm()` on potentially complex tensors:
  ```python
  # Before: norm = torch.norm(x, dim=1, keepdim=True)
  # After:
  if x.is_complex() and x.device.type == "mps":
      norm = torch.norm(x.cpu(), dim=1, keepdim=True).to(x.device)
  else:
      norm = torch.norm(x, dim=1, keepdim=True)
  ```

- **Success**:
  - STOI computation works on MPS without errors
  - No performance regression on CPU/CUDA
  - Complex norm computed on CPU, result moved back

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 100-103) - torch.linalg.norm limitation

- **Dependencies**: Task 2.1 completion

---

### Task 2.3: Configuration Option for Forward Method

Add config option to override automatic forward method selection.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df/modules.py` - Add config check
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/df/config.py` - Document option

- **Implementation**:
  ```python
  # In DfOp.__init__()
  use_complex_override = config(
      "DF_USE_COMPLEX", cast=bool, section="df", default=None, save=False
  )
  if use_complex_override is not None:
      self._use_complex = use_complex_override
      logger.info(f"Forward method overridden via config: use_complex={use_complex_override}")
  ```

- **Success**:
  - Config `DF_USE_COMPLEX=true` forces complex method
  - Config `DF_USE_COMPLEX=false` forces real method
  - No config uses auto-detection

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 203-207) - Configuration guidance

- **Dependencies**: Task 2.2 completion

---

## Phase 3: Documentation Updates

### Task 3.1: README MPS Section

Add dedicated MPS usage section to README.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/README.md` - Add MPS section after Device Selection

- **Content**:
  ```markdown
  ### Apple Silicon (MPS) Support
  
  DeepFilterNet supports Apple Silicon GPUs via PyTorch's MPS backend for accelerated inference.
  
  **Requirements:**
  - macOS 14 (Sonoma) or later for full support
  - PyTorch 2.5.1+ with MPS enabled
  
  **Usage:**
  ```bash
  # Explicit MPS device
  deep-filter --device mps input.wav
  
  # Auto-detection (uses MPS if available)
  deep-filter input.wav
  ```
  
  **Note:** Complex tensor operations require macOS 14+. On older versions, set:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
  ```

- **Success**:
  - MPS section visible in README
  - Requirements clearly stated
  - Usage examples provided

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 68-73) - Current MPS support status

- **Dependencies**: Phase 2 completion

---

### Task 3.2: MPS Compatibility Table

Document which operations are supported on MPS.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/README.md` - Add compatibility table

- **Content**:
  ```markdown
  #### MPS Operation Compatibility
  
  | Operation | macOS 12-13 | macOS 14+ |
  |-----------|-------------|-----------|
  | Model inference | ✅ (real mode) | ✅ |
  | Complex filtering | ❌ | ✅ |
  | STOI evaluation | ⚠️ CPU fallback | ⚠️ CPU fallback |
  | Training | ⚠️ Limited | ✅ |
  ```

- **Success**:
  - Table renders correctly in GitHub
  - Compatibility clearly indicated
  - Footnotes explain limitations

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 90-103) - Operation compatibility

- **Dependencies**: Task 3.1 completion

---

### Task 3.3: Performance Expectations Documentation

Document expected performance characteristics.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/README.md` - Add performance section

- **Content**:
  ```markdown
  #### Performance Expectations
  
  - **Batch processing:** Significant speedup over CPU (2-4x typical)
  - **Real-time streaming:** Marginal improvement due to CPU↔MPS transfer overhead
  
  The STFT/ISTFT operations run on CPU (libDF/Rust) for optimal real-time latency.
  GPU acceleration benefits the neural network inference portion.
  ```

- **Success**:
  - Performance expectations set appropriately
  - Architecture explained simply
  - No false promises about speedup

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 132-142) - Performance considerations

- **Dependencies**: Task 3.2 completion

---

### Task 3.4: Troubleshooting Guide

Add troubleshooting section for common MPS issues.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/README.md` - Add troubleshooting section

- **Content**:
  ```markdown
  #### MPS Troubleshooting
  
  **"ComplexFloat not supported" error:**
  - Requires macOS 14+, or use `--device cpu`
  - Or set `PYTORCH_ENABLE_MPS_FALLBACK=1`
  
  **Slower than expected:**
  - Real-time streaming has CPU↔GPU overhead
  - Batch processing benefits most from MPS
  
  **"MPS backend not available":**
  - Ensure PyTorch installed with MPS support: `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`
  - Requires Apple Silicon Mac (M1/M2/M3)
  ```

- **Success**:
  - Common errors addressed
  - Solutions actionable
  - Hardware requirements clear

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 108-125) - Compatibility issues

- **Dependencies**: Task 3.3 completion

---

## Phase 4: Testing and Validation

### Task 4.1: MPS Pytest Markers

Add pytest markers for MPS-specific tests.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/tests/conftest.py` - Create or update with markers
  - `/Users/andrew/zRepos/DeepFilterNet/pyproject.toml` - Register markers

- **Implementation**:
  ```python
  # conftest.py
  import pytest
  import torch
  
  def pytest_configure(config):
      config.addinivalue_line("markers", "mps: mark test as requiring MPS backend")
  
  @pytest.fixture
  def mps_device():
      if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
          pytest.skip("MPS not available")
      return torch.device("mps")
  ```

- **Success**:
  - `@pytest.mark.mps` available for tests
  - `mps_device` fixture skips on non-MPS systems
  - Markers registered in pyproject.toml

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 203-207) - Testing guidance

- **Dependencies**: Phase 3 completion

---

### Task 4.2: Device Detection Unit Tests

Add tests for get_device() MPS scenarios.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/tests/test_utils.py` - Create or update

- **Implementation**:
  ```python
  def test_get_device_explicit_mps():
      device = get_device("mps")
      assert device.type == "mps"
  
  def test_get_device_auto_detection():
      device = get_device()
      assert device.type in ("cuda", "mps", "cpu")
  
  def test_mps_supports_complex_detection():
      result = mps_supports_complex()
      assert isinstance(result, bool)
  
  def test_get_macos_version():
      version = get_macos_version()
      if platform.system() == "Darwin":
          assert version is not None
          assert isinstance(version, tuple)
          assert len(version) == 2
      else:
          assert version is None
  ```

- **Success**:
  - All device selection paths tested
  - Version detection validated
  - Tests pass on macOS and non-macOS

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 143-164) - get_device implementation

- **Dependencies**: Task 4.1 completion

---

### Task 4.3: Forward Method Compatibility Tests

Test both forward methods work on MPS.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/tests/test_modules.py` - Add MPS tests

- **Implementation**:
  ```python
  @pytest.mark.mps
  def test_dfop_forward_complex_strided_mps(mps_device):
      if not mps_supports_complex():
          pytest.skip("Complex ops require macOS 14+")
      # Test forward_complex_strided on MPS
      ...
  
  @pytest.mark.mps
  def test_dfop_forward_real_unfold_mps(mps_device):
      # Test forward_real_unfold on MPS (should always work)
      ...
  ```

- **Success**:
  - Both forward methods tested on MPS
  - Complex method skipped on macOS < 14
  - Real method works on all MPS versions

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 165-172) - Forward method code

- **Dependencies**: Task 4.2 completion

---

### Task 4.4: End-to-End Inference Test

Test full enhance() pipeline on MPS.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/DeepFilterNet/tests/test_enhance.py` - Add MPS test

- **Implementation**:
  ```python
  @pytest.mark.mps
  def test_enhance_mps(mps_device, sample_audio, pretrained_model):
      model, df_state, _, _ = pretrained_model
      enhanced = enhance(model, df_state, sample_audio, device="mps")
      assert enhanced.shape == sample_audio.shape
      assert not torch.isnan(enhanced).any()
  ```

- **Success**:
  - Full inference pipeline works on MPS
  - Output shape correct
  - No NaN values in output

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 75-94) - Data flow architecture

- **Dependencies**: Task 4.3 completion

---

### Task 4.5: CI/CD MPS Skip Configuration

Configure CI to skip MPS tests on non-macOS runners.

- **Files**:
  - `/Users/andrew/zRepos/DeepFilterNet/pyproject.toml` - Add pytest configuration
  - `/Users/andrew/zRepos/DeepFilterNet/.github/workflows/*.yml` - Update if exists

- **Implementation**:
  In pyproject.toml:
  ```toml
  [tool.pytest.ini_options]
  markers = [
      "mps: mark test as requiring MPS backend (skip on non-macOS)",
  ]
  ```

- **Success**:
  - MPS tests skip gracefully on Linux CI
  - MPS tests run on macOS CI (if available)
  - No CI failures due to MPS unavailability

- **Research References**:
  - #file:../research/20260713-mps-backend-support-research.md (Lines 203-207) - CI guidance

- **Dependencies**: Task 4.4 completion

---

## Dependencies

- PyTorch 2.5.1+ with MPS support
- macOS 14 (Sonoma) or later for complex operations
- torchaudio 2.5.1+
- pytest with markers support

## Success Criteria

- MPS inference works for batch processing on macOS 14+
- Graceful degradation on macOS 12-13 with warnings
- Comprehensive documentation in README
- Test suite validates all MPS code paths
- CI configured to handle MPS tests appropriately
