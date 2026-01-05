<!-- markdownlint-disable-file -->
# Release Changes: DeepFilterNet4 Architecture

**Related Plan**: 20260105-deepfilternet4-architecture-plan.instructions.md
**Implementation Date**: 2026-01-05

## Summary

Implementation of DeepFilterNet4, a next-generation speech enhancement model featuring Mamba state-space models, hybrid time-frequency processing, multi-resolution deep filtering, and perceptual training losses.

## Changes

### Added

- `DeepFilterNet/df/mamba.py` - Mamba state space model implementation with MambaBlock, Mamba, SqueezedMamba, and BidirectionalMamba classes
- `DeepFilterNet/tests/test_mamba.py` - Comprehensive test suite for Mamba modules (26 tests covering shapes, causality, API compatibility, MPS)
- `DeepFilterNet/df/scripts/benchmark_mamba.py` - Benchmark script comparing Mamba vs GRU performance metrics
- `DeepFilterNet/df/hybrid_encoder.py` - Hybrid time-frequency encoder with WaveformEncoder, PhaseEncoder, CrossDomainAttention, SimpleCrossAttention, MagnitudeEncoder, HybridEncoder, and LightweightHybridEncoder
- `DeepFilterNet/tests/test_hybrid_encoder.py` - Comprehensive test suite for hybrid encoder components (32 tests, 31 passed, 1 CUDA skip)
- `DeepFilterNet/tests/test_multires_df.py` - Comprehensive test suite for multi-resolution deep filtering (39 tests, 3 skipped for config/CUDA)
- `DeepFilterNet/df/discriminator.py` - GAN discriminators with MultiPeriodDiscriminator, MultiScaleDiscriminator, CombinedDiscriminator, and helper loss functions (Phase 4.1)
- `DeepFilterNet/df/dnsmos_proxy.py` - Differentiable DNSMOS proxy model with MelSpectrogram, DNSMOSProxy, LightweightDNSMOSProxy, and DNSMOSLoss (Phase 4.3)
- `DeepFilterNet/tests/test_phase4_training.py` - Comprehensive test suite for Phase 4 training enhancements (43 tests, 42 passed, 1 CUDA skip)

### Modified

- `DeepFilterNet/df/multiframe.py` - Added MultiResolutionDF for multi-resolution deep filtering with learnable weights, added AdaptiveOrderPredictor for Gumbel-Softmax filter order selection
- `DeepFilterNet/df/deepfilternet4.py` - Added DfOutputReshape, MultiResDfDecoder, SingleResDfDecoder, AdaptiveDfDecoder for multi-resolution coefficient prediction; completed DfNet4 full model integration with Encoder4, ErbDecoder4, init_model(), and ModelParams4 (Phase 5.1); added DfNet4Lite, Encoder4Lite, ErbDecoder4Lite for ~50% parameter reduction variant with reduced conv_ch, emb_hidden_dim, and fewer layers (Phase 5.2); enhanced ModelParams4 docstring with comprehensive parameter documentation and added generate_config_template() class method (Phase 5.3)
- `DeepFilterNet/df/loss.py` - Added FeatureMatchingLoss, SpeakerContrastiveLoss, GeneratorLoss, DiscriminatorLoss; updated Loss class with discriminator parameter and GAN wrapper methods (compute_d_loss_with_disc, compute_g_loss_with_disc, run_discriminator) (Phase 4.2, 4.4, 4.5, 4.6)
- `DeepFilterNet/df/train.py` - Added GAN training support with setup_discriminator(), load_discriminator_opt(), setup_discriminator_lrs(), synthesize_waveform(), train_discriminator_step(); updated run_epoch() for alternating G/D training; updated main() for discriminator checkpoint saving; added QAT support with --qat, --qat-start-epoch, --qat-backend arguments and QATCallback integration for quantization-aware training (Phase 4.6, 5.4)
- `DeepFilterNet/df/config.py` - Added _fix_dfnet4() backward compatibility method for DFNet4 configs, including auto-creation of deepfilternet4 section and migration of compatible params from DFNet3 (Phase 5.3)
- `DeepFilterNet/tests/conftest.py` - Added df_config fixture for tests requiring config initialization
- `DeepFilterNet/tests/test_phase4_training.py` - Added TestLossGANIntegration test class with 5 tests for Loss GAN methods (48 tests total, 47 passed, 1 CUDA skip)
- `DeepFilterNet/df/hybrid_encoder.py` - Added time_fallback_proj and phase_fallback_proj Linear layers to HybridEncoder for proper dimension handling when waveform input is not provided (Phase 5.1)
- `DeepFilterNet/tests/test_deepfilternet4.py` - Comprehensive test suite for DfNet4 full model (51 tests, 49 passed, 2 skipped); tests cover standard, hybrid encoder, multi-res DF, adaptive order, lite variant, GRU backbone, gradient flow, eval mode, batch sizes, sequence lengths, components, init_model, devices, edge cases, ModelParams4 config, config compatibility, and quantization (Phase 5.1, 5.2, 5.3, 5.4)

### Added (Phase 5.4)

- `DeepFilterNet/df/quantization.py` - Quantization-aware training (QAT) and post-training quantization (PTQ) support with QuantizationConfig, QATCallback, prepare_model_for_qat(), convert_qat_model(), quantize_dynamic(), quantize_static(), export_quantized_model(), get_model_size_mb(), compare_model_sizes(), find_fusable_patterns(), fuse_modules(), and QuantizedModelWrapper classes

### Added (Phase 5.5)

- `DeepFilterNet/df/scripts/distill.py` - Knowledge distillation training script with DistillationLoss (temperature scaling, KL divergence, feature matching), DistillationTrainer class for teacher-student training loop, compare_models() for teacher/student comparison metrics, and CLI with configurable temperature, alpha, feature-weight, epochs, batch-size, lr, and compare-only mode

### Modified (Phase 6.1)

- `DeepFilterNet/tests/test_deepfilternet4.py` - Added comprehensive integration tests including TestDfNet4Integration (10 tests: training step, optimizer, scheduler, gradient clipping, checkpoint save/load, model variants, hybrid encoder variants, DF variants, memory efficiency, no_grad inference), TestDfNet4WithLoss (3 tests: spectral, mask, lsnr loss), TestDfNet4FeatureExtraction (1 test: libdf feature compatibility). Total: 65 tests pass, 2 skipped.

### Added (Phase 6.2)

- `DeepFilterNet/df/scripts/benchmark_dfnet4.py` - Benchmark script for DFNet4 performance evaluation with load_model(), measure_rtf(), count_parameters(), get_voicebank_files(), run_evaluation(), format_results(), compare_models() functions. Supports PESQ/STOI/DNSMOS metrics, RTF measurement, DFNet3 comparison, and JSON output.

### Added (Phase 6.3)

- `DeepFilterNet/df/scripts/export_onnx.py` - ONNX export script for DFNet4 models with export_dfnet4(), export_impl(), onnx_check(), onnx_simplify(), create_export_archive() functions. Supports full model export, component-wise export (enc, erb_dec, df_dec), dynamic axes for variable length, validation against PyTorch outputs, and archive creation.

### Modified (Phase 6.4)

- `README.md` - Added DFNet4 announcement in News section with feature highlights
- `DeepFilterNet/README.md` - Added comprehensive DeepFilterNet4 documentation section (~230 lines) covering key features, model variants, quick start, architecture components (Mamba encoder, hybrid encoder, multi-resolution DF, adaptive filter order), training (basic, GAN, QAT, distillation), ONNX export, benchmarking, configuration reference, and migration link

### Added (Phase 6.4)

- `docs/ARCHITECTURE.md` - Comprehensive architecture documentation (~580 lines) covering architecture evolution (DFNet1-4), high-level diagrams, Mamba backbone (SSM equations, block structure), encoder/decoder architecture, deep filtering (standard, multi-resolution, adaptive), feature extraction (ERB, DF), training components (losses, discriminator, distillation, QAT), configuration reference, model size comparison, file structure
- `docs/MIGRATION.md` - Migration guide from DFNet3 to DFNet4 (~350 lines) covering key differences, configuration migration (INI to dataclass with helper function), model loading, training migration, inference migration, common issues and solutions, API reference changes, migration checklist

### Removed

- None

---

## Release Summary

**Total Files Affected**: 23

### Files Created (13)

- `DeepFilterNet/df/mamba.py` - Mamba state-space model with MambaBlock, Mamba, SqueezedMamba, BidirectionalMamba classes
- `DeepFilterNet/df/hybrid_encoder.py` - Hybrid time-frequency encoder with WaveformEncoder, PhaseEncoder, CrossDomainAttention
- `DeepFilterNet/df/discriminator.py` - GAN discriminators (MPD, MSD, Combined) with loss functions
- `DeepFilterNet/df/dnsmos_proxy.py` - Differentiable DNSMOS approximation model
- `DeepFilterNet/df/quantization.py` - Quantization-aware training (QAT) and post-training quantization (PTQ) support
- `DeepFilterNet/df/scripts/distill.py` - Knowledge distillation training script
- `DeepFilterNet/df/scripts/benchmark_mamba.py` - Mamba vs GRU performance benchmark
- `DeepFilterNet/df/scripts/benchmark_dfnet4.py` - DFNet4 performance evaluation with PESQ/STOI/DNSMOS
- `DeepFilterNet/df/scripts/export_onnx.py` - ONNX export for DFNet4 models
- `DeepFilterNet/tests/test_mamba.py` - Mamba module tests (26 tests)
- `DeepFilterNet/tests/test_hybrid_encoder.py` - Hybrid encoder tests (32 tests)
- `DeepFilterNet/tests/test_multires_df.py` - Multi-resolution DF tests (39 tests)
- `DeepFilterNet/tests/test_phase4_training.py` - Training enhancement tests (48 tests)
- `docs/ARCHITECTURE.md` - Comprehensive architecture documentation
- `docs/MIGRATION.md` - DFNet3 → DFNet4 migration guide

### Files Modified (8)

- `DeepFilterNet/df/deepfilternet4.py` - Added DfNet4, DfNet4Lite, encoders, decoders, multi-res DF
- `DeepFilterNet/df/multiframe.py` - Added MultiResolutionDF, AdaptiveOrderPredictor
- `DeepFilterNet/df/loss.py` - Added FeatureMatchingLoss, SpeakerContrastiveLoss, GAN losses
- `DeepFilterNet/df/train.py` - Added GAN training support and QAT integration
- `DeepFilterNet/df/config.py` - Added DFNet4 backward compatibility
- `DeepFilterNet/tests/conftest.py` - Added df_config fixture
- `DeepFilterNet/tests/test_deepfilternet4.py` - Comprehensive DFNet4 tests (65 tests)
- `README.md` - Added DFNet4 announcement
- `DeepFilterNet/README.md` - Added DFNet4 documentation section

### Files Removed (0)

- None

### Dependencies & Infrastructure

- **New Dependencies**: einops (tensor operations), resemblyzer (optional, speaker embeddings)
- **Updated Dependencies**: PyTorch >= 2.0 (required for Mamba SSM)
- **Infrastructure Changes**: None
- **Configuration Updates**: New ModelParams4 dataclass configuration system

### Deployment Notes

1. **Python Version**: Minimum Python 3.9 required (up from 3.8)
2. **PyTorch Version**: PyTorch 2.0+ required for Mamba SSM operations
3. **Model Compatibility**: DFNet4 checkpoints are not compatible with DFNet3 loader
4. **ONNX Export**: New export script at `df/scripts/export_onnx.py`
5. **Quantization**: INT8 deployment supported via QAT or PTQ
6. **Apple Silicon**: Full MPS support for DFNet4 inference

### Test Coverage

| Module | Tests | Passed | Skipped |
|--------|-------|--------|---------|
| test_mamba.py | 26 | 26 | 0 |
| test_hybrid_encoder.py | 32 | 31 | 1 (CUDA) |
| test_multires_df.py | 39 | 36 | 3 (config/CUDA) |
| test_phase4_training.py | 48 | 47 | 1 (CUDA) |
| test_deepfilternet4.py | 65 | 63 | 2 (CUDA/quant) |
| **Total** | **210** | **203** | **7** |

