# Lint Ignore Tracking

This file tracks inline lint/type-check ignores added to the codebase and their rationale.

## Python Type Ignores (pyright/mypy)

### DeepFilterNet/df/config.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 153 | `# type: ignore[call-arg]` | `cast` is a `Type[T]` callable - pyright can't track this dynamic type coercion pattern |
| 286 | `# type: ignore[misc]` | Type variable in class attribute is intentional for generic `Csv` class |

### DeepFilterNet/df/discriminator.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 336 | `# type: ignore[return-value]` | Accumulator starts as `float` (0.0) but becomes `Tensor` after first addition with `torch.mean()`. Runtime always returns Tensor. |
| 358 | `# type: ignore[return-value]` | Same pattern - float accumulator becomes Tensor at runtime |
| 382 | `# type: ignore[return-value]` | Same pattern - float accumulator becomes Tensor at runtime |

### DeepFilterNet/df/quantization.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 122 | `# type: ignore[possibly-undefined]` | `get_default_qat_qconfig` is conditionally imported when `QUANTIZATION_AVAILABLE=True`. Function is only called inside a guard that checks availability. |
| 124 | `# type: ignore[possibly-undefined]` | `get_default_qconfig` - same pattern |
| 137 | `# type: ignore[possibly-undefined]` | `QuantStub` - conditionally imported, guarded by `QUANTIZATION_AVAILABLE` ternary |
| 139 | `# type: ignore[possibly-undefined]` | `DeQuantStub` - same pattern |
| 190 | `# type: ignore[possibly-undefined]` | `get_default_qat_qconfig` - guarded by early return on line 182 |
| 200 | `# type: ignore[possibly-undefined]` | `prepare_qat` - guarded by early return |
| 227 | `# type: ignore[possibly-undefined]` | `convert` - guarded by early return on line 219 |
| 385 | `# type: ignore[possibly-undefined]` | `get_default_qconfig` - guarded by early return on line 377 |

### DeepFilterNet/df/io.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 31 | `# type: ignore[import-unresolved]` | `torchaudio.backend.common` is version-dependent - fallback import for older TorchAudio versions |
| 64 | `# type: ignore[possibly-undefined]` | `AudioDecoder` is conditionally imported when `HAS_TORCHCODEC=True` |

### DeepFilterNet/df/enhance.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 233 | `# type: ignore[operator]` | `reset_h0` is dynamically checked with `hasattr()` - pyright can't verify method exists |

### DeepFilterNet/df/modules.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 567 | `# type: ignore[operator]` | `flatten_parameters()` called on `nn.GRU` layer in ModuleList - pyright thinks it's a Tensor |
| 721 | `# type: ignore[return-value]` | GRU returns `Tuple[Tensor, Optional[Tensor]]` but h is always Tensor when h0 is provided |
| 760 | `# type: ignore[return-value]` | Same pattern - GRU hidden state is always Tensor at runtime |
| 917 | `# type: ignore[call-arg]` | `example_outputs` param was deprecated and removed in newer PyTorch - legacy test code |
| 922 | `# type: ignore[call-arg]` | Same - deprecated torch.onnx.export parameter |
| 927 | `# type: ignore[call-arg]` | Same - deprecated torch.onnx.export parameter |
| 941 | `# type: ignore[call-arg]` | Same - deprecated torch.onnx.export parameter |
| 946 | `# type: ignore[call-arg]` | Same - deprecated torch.onnx.export parameter |

### DeepFilterNet/df/scripts/export_onnx.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 67 | `# type: ignore[possibly-undefined]` | `onnxsim` conditionally imported in try block |
| 72 | `# type: ignore[possibly-undefined]` | `onnx.load` - onnx conditionally imported at module level |
| 73 | `# type: ignore[possibly-undefined]` | `onnxsim.simplify` - conditionally imported |
| 85 | `# type: ignore[possibly-undefined]` | `onnx.checker.check_model` - conditionally imported |
| 90 | `# type: ignore[possibly-undefined]` | `onnx.save_model` - conditionally imported |
| 105 | `# type: ignore[possibly-undefined]` | `onnx.load` in onnx_check() |
| 106 | `# type: ignore[possibly-undefined]` | `onnx.helper.printable_graph` |
| 107 | `# type: ignore[possibly-undefined]` | `onnx.checker.check_model` |
| 109 | `# type: ignore[possibly-undefined]` | `ort.InferenceSession` - ort conditionally imported |
| 224 | `# type: ignore[possibly-undefined]` | `onnx.load` in print_graph section |
| 225 | `# type: ignore[possibly-undefined]` | `onnx.helper.printable_graph` |

### DeepFilterNet/df/scripts/dnsmos.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 156 | `# type: ignore[assignment]` | `np.pad` returns ndarray but audio was typed as Tensor - runtime coercion handles this |
| 186 | `# type: ignore[return-value]` | Return type annotation says `List[float]` but actually returns `np.floating` scalars |

### DeepFilterNet/df/evaluation_utils.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 172 | `# type: ignore[assignment]` | Intentional `[None] * len(...)` for optional noisy files placeholder |
| 478 | `# type: ignore[override]` | `NoisyMetric.add()` intentionally has different signature - doesn't need clean reference |
| 498 | `# type: ignore[override]` | `NoisyMetric.compute_metric()` intentionally has different signature |

### DeepFilterNet/df/loss.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 131 | `# type: ignore[operator]` | `angle.apply()` returns Tensor at runtime - pyright can't track custom autograd Function |
| 132 | `# type: ignore[operator]` | Same - `angle.apply()` custom autograd Function |
| 171 | `# type: ignore[operator]` | Same - `angle.apply()` custom autograd Function |
| 172 | `# type: ignore[operator]` | Same - `angle.apply()` custom autograd Function |
| 525 | `# type: ignore[return-value]` | `loss` can be `0` literal initially but becomes `Tensor` at runtime |
| 528 | `# type: ignore[assignment]` | `tokens` list comprehension changes type from `Tensor` to `List[Tensor]` |
| 769 | `# type: ignore[possibly-undefined]` | `noisy_emb` is defined in earlier `if noisy is not None` block - control flow is correct |
| 1070 | `# type: ignore[misc]` | `self.asrl` is `Optional` but guarded by `asrl_f > 0` check which sets it |
| 1072 | `# type: ignore[misc]` | `self.lsnrl` is `Optional` but guarded by `lsnr_f != 0` check |
| 1074 | `# type: ignore[misc]` | `self.sdrl` is `Optional` but guarded by `sdrl_f != 0` check |
| 1338 | `# type: ignore[call-arg]` | librosa API changed - using keyword args for compatibility |

### DeepFilterNet/df/utils.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 129 | `# type: ignore[override]` | Custom autograd Function backward signature - pyright doesn't understand ctx pattern |
| 144 | `# type: ignore[override]` | Same - custom autograd Function backward signature |

### DeepFilterNet/df/stoi.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 218 | `# type: ignore[operator]` | `EPS` is `np.finfo` float but torch handles addition at runtime |
| 227 | `# type: ignore[operator]` | Same - `EPS` numpy float with torch division |
| 228 | `# type: ignore[operator]` | Same - `EPS` numpy float with torch division |

### DeepFilterNet/df/logger.py

| Line | Ignore | Rationale |
|------|--------|-----------|
| 218 | `# type: ignore[union-attr]` | `module.weight` exists on `GroupedLinearEinsum` but pyright sees base `Module` type |

## Code Fixes (Not Ignores)

### DeepFilterNet/df/mamba.py

| Line | Change | Rationale |
|------|--------|-----------|
| 430 | Added `assert self.out_proj is not None` | Type narrowing - `out_proj` is only None when `merge != "proj"`, but we're in the `elif merge == "proj"` branch, so this assertion is always true at runtime. |

### DeepFilterNet/df/modules.py

| Line | Change | Rationale |
|------|--------|-----------|
| ~863 | Refactored `LocalSnrTarget.forward()` to check `self.range` before calling `.clamp()` | Bug fix - `self.range` could be None, causing runtime error |

### DeepFilterNet/df/scripts/export_onnx.py

| Line | Change | Rationale |
|------|--------|-----------|
| 311 | Added `enc_outputs: Optional[Tuple[Tensor, ...]] = None` | Initialize variable before conditional block to satisfy type checker |
| 580 | Changed `log_level=` to `level=` | Bug fix - `init_logger()` parameter is `level`, not `log_level` |

## Configuration Files

### .markdownlint.json

Disables all markdown linting rules repo-wide. The user will fine-tune this later.

Disabled rules:
- MD001: Heading increment
- MD004: Unordered list style
- MD009: Trailing spaces
- MD013: Line length
- MD022: Blanks around headings
- MD024: Multiple headings same content
- MD031: Blanks around fences
- MD032: Blanks around lists
- MD033: Inline HTML
- MD034: Bare URLs
- MD040: Fenced code language
- MD041: First line heading
- MD059: Descriptive link text
- MD060: Table column style

### pyrightconfig.json

Sets pyright to `basic` mode with warnings (not errors) for:
- `reportMissingImports`: Optional dependencies (onnx, onnxruntime, onnxsim, whisper, resemblyzer)
- `reportOptionalMemberAccess`: Conditional member access patterns
- `reportPossiblyUnbound`: Conditionally imported symbols
- `reportGeneralTypeIssues`: General type mismatches

Excludes test files from strict checking.

---

Last updated: 2026-01-05
