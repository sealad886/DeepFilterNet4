# Task Research Notes: MPS Backend Support for DeepFilterNet

## Research Executed

### File Analysis
- `DeepFilterNet/df/utils.py`
  - Contains `get_device()` function with MPS detection (lines 37-47)
  - MPS detection order: CUDA → MPS → CPU (auto-detection fallback chain)
  - `as_complex()` and `as_real()` utility functions that wrap `torch.view_as_complex()` and `torch.view_as_real()` (lines 48-61)

- `DeepFilterNet/df/modules.py`
  - Contains `DfOp` class with multiple forward implementations (lines 320-480)
  - `forward_complex_strided()` uses `torch.view_as_complex()` and `torch.view_as_real()` for spectral filtering (lines 408-420)
  - `forward_real_unfold()` and `forward_real_no_pad_one_step()` provide non-complex alternatives
  - Model components use `get_device()` for device selection

- `DeepFilterNet/df/enhance.py`
  - Core data flow: `df_state.analysis()` (CPU/libDF) → model inference (MPS possible) → `df_state.synthesis()` (CPU/libDF)
  - Features are computed on CPU via `libdf` (STFT/ERB), then moved to compute device for model inference
  - Results are moved back to CPU for synthesis (ISTFT)

- `libDF/src/transforms.rs`
  - STFT/ISTFT implemented in Rust using RealFftPlanner
  - All spectral transforms run on CPU via numpy/ndarray interop
  - This is by design for real-time streaming with minimal latency

- `DeepFilterNet/df/deepfilternet3.py`
  - Model uses `as_complex()` for mask computation (line 429+)
  - Pattern: `mask = (as_complex(spec_e).abs() / as_complex(spec).abs().add(eps)).clamp(eps, 1)`

- `DeepFilterNet/df/stoi.py`
  - Uses `torch.norm()` for STOI metric computation
  - Not part of inference path; used only for evaluation

### Code Search Results
- `mps|MPS|metal` pattern
  - 20+ matches: README mentions, utils.py device detection, pyproject.toml poe task
  - `install-torch-mps = "pip install torch==2.5.1 torchaudio==2.5.1"` in pyproject.toml

- `.to(device|get_device` pattern
  - 20+ matches: deepfilternet3.py (line 87), modules.py (line 223), enhance.py (line 186)
  - Models are moved to device via `model.to(get_device(device))`
  - Feature tensors moved to device in `df_features()` function

- `view_as_complex|view_as_real|as_complex|as_real` pattern
  - Extensive usage in modules.py, deepfilternet2.py, deepfilternet3.py
  - Critical for deep filtering spectral operations

### External Research
- PyTorch GitHub Issues
  - [#78044](https://github.com/pytorch/pytorch/issues/78044): FFT operators added to MPS in Feb 2024 (PR #119670)
  - [#105665](https://github.com/pytorch/pytorch/issues/105665): Complex support (view_as_complex, polar, mul) added Sept 2023-Feb 2024
  - [#115513](https://github.com/pytorch/pytorch/pull/115513): General complex op support enabled for macOS 14
  - [#77764](https://github.com/pytorch/pytorch/issues/77764): General MPS op coverage tracking (ongoing)
  - [#146691](https://github.com/pytorch/pytorch/issues/146691): `torch.linalg.norm` on complex still NOT supported (Feb 2025)

- PyTorch MPS Documentation
  - Requires macOS 12.3+ for basic MPS support
  - Requires macOS 14+ for complex number operations
  - `PYTORCH_ENABLE_MPS_FALLBACK=1` env var enables CPU fallback for unsupported ops

### Project Conventions
- Standards referenced: Device selection via `get_device()` in utils.py
- Instructions followed: `--device` CLI argument support added in modernization

## Key Discoveries

### Current MPS Support Status
DeepFilterNet already has basic MPS support implemented:
1. **Device Detection**: `get_device()` auto-detects MPS availability
2. **CLI Support**: `--device mps` argument for explicit device selection
3. **Poe Task**: `install-torch-mps` for macOS-specific PyTorch installation

### Data Flow Architecture
```
Audio Input
    │
    ▼
df_state.analysis() ─────► CPU (libDF/Rust)
    │                        STFT computation
    ▼
spec, erb_feat, spec_feat ─► .to(device)
    │                        Move to MPS
    ▼
model(spec, erb, spec_feat) ► MPS (PyTorch)
    │                        Neural network inference
    ▼
enhanced = output.cpu() ────► CPU
    │
    ▼
df_state.synthesis() ───────► CPU (libDF/Rust)
    │                        ISTFT computation
    ▼
Audio Output
```

### MPS Operation Compatibility Analysis

| Operation | MPS Support | Notes |
|-----------|-------------|-------|
| `torch.view_as_complex()` | ✅ Supported | Added in PyTorch ~2.1 for macOS 14+ |
| `torch.view_as_real()` | ✅ Supported | Added in PyTorch ~2.1 for macOS 14+ |
| Complex `*` (mul) | ✅ Supported | PR #108395 (Sept 2023) |
| Complex `.abs()` | ✅ Supported | Via view_as_real decomposition |
| Complex `.sum()` | ✅ Supported | For macOS 14+ |
| `torch.atan2()` | ✅ Supported | Used in angle computation |
| `torch.fft.rfft()` | ✅ Supported | PR #119670 (Feb 2024) |
| `torch.linalg.norm()` on complex | ❌ NOT Supported | Issue #146691 (Feb 2025) |
| Conv2d, BatchNorm, ReLU | ✅ Supported | Standard ops work |
| GRU/LSTM | ✅ Supported | Standard RNN ops work |

### Potential MPS Compatibility Issues

1. **macOS Version Requirement**
   - Complex number operations require macOS 14 (Sonoma) or later
   - Basic MPS works on macOS 12.3+, but complex ops will fail

2. **CPU↔MPS Data Transfer Overhead**
   - STFT/ISTFT runs on CPU via libDF (by design for real-time streaming)
   - Model inference on MPS requires data transfer each frame
   - For real-time use, this overhead may negate GPU speedup

3. **`torch.linalg.norm()` on Complex**
   - Not used in inference path, only in `stoi.py` for evaluation
   - STOI computation would need CPU fallback or workaround

4. **Potential Strided View Issues**
   - `as_strided()` views with complex tensors may have edge cases
   - The `forward_complex_strided()` method is the most MPS-sensitive code path

### Performance Considerations

**Inference Path (enhance.py)**:
- Feature extraction: CPU (libDF, unavoidable)
- Model forward pass: MPS-capable
- Synthesis: CPU (libDF, unavoidable)

**Expected Speedup**:
- Batch processing: Significant speedup (model is the bottleneck)
- Real-time/streaming: Marginal speedup (CPU↔MPS transfer overhead)

### Complete Examples

**MPS Device Selection (current implementation)**:
```python
# DeepFilterNet/df/utils.py
def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        cfg_device = config("DEVICE", cast=str, section="train", default="", save=False)
        if cfg_device:
            return torch.device(cfg_device)
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

**Complex Forward (modules.py)**:
```python
def forward_complex_strided(self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None):
    padded = as_strided(spec[..., :self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3)
    spec_f = torch.sum(torch.view_as_complex(padded) * torch.view_as_complex(coefs), dim=2)
    spec_f = torch.view_as_real(spec_f)
    return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)
```

### Technical Requirements
- **Minimum PyTorch Version**: 2.5.1 (already specified in pyproject.toml)
- **Minimum macOS Version**: 14 (Sonoma) for complex operations
- **Xcode Version**: 13.3.1+ for building PyTorch with MPS support

## Recommended Approach

**MPS support is already substantially complete** for DeepFilterNet inference. The key remaining work is:

1. **Documentation and Validation**
   - Document macOS 14+ requirement for complex operations
   - Add runtime warning if macOS < 14 and MPS selected with complex ops
   - Create MPS-specific test suite to validate operations

2. **Graceful Degradation**
   - Add version detection for macOS to warn about complex op limitations
   - Implement automatic fallback to `forward_real_unfold()` if complex ops fail
   - Use `PYTORCH_ENABLE_MPS_FALLBACK=1` as documented fallback strategy

3. **Evaluation Code Updates** (optional)
   - STOI computation uses `torch.norm()` on complex - needs CPU fallback
   - Training losses may have similar issues

4. **Performance Testing**
   - Benchmark batch processing vs CPU baseline
   - Benchmark real-time streaming with CPU↔MPS overhead
   - Document expected speedup scenarios

## Implementation Guidance
- **Objectives**: Ensure DeepFilterNet inference works reliably on MPS with appropriate user warnings and fallbacks
- **Key Tasks**:
  1. Add macOS version detection and warning for complex operation requirements
  2. Document MPS usage in README with macOS 14+ requirement
  3. Test all forward methods (`forward_complex_strided`, `forward_real_unfold`) on MPS
  4. Add MPS-specific pytest markers for CI/CD
  5. Consider `forward_real_unfold` as default for better compatibility
- **Dependencies**: PyTorch 2.5.1+, macOS 14+ (for complex ops), torchaudio 2.5.1
- **Success Criteria**: 
  - Inference runs on MPS without errors for batch processing
  - Appropriate warnings displayed for incompatible configurations
  - Performance benchmarks documented
