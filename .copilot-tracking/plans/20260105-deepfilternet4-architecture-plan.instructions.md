---
applyTo: '.copilot-tracking/changes/20260105-deepfilternet4-architecture-changes.md'
---
<!-- markdownlint-disable-file -->
# DeepFilterNet4 Architecture Implementation — Structured Delivery Plan

## Purpose

Implement DeepFilterNet4, a next-generation speech enhancement model featuring Mamba state-space models, hybrid time-frequency processing, multi-resolution deep filtering, and perceptual training losses to achieve SOTA performance (PESQ 3.5+, STOI 0.96+).

---

## Reference Inputs

### Internal
- `DeepFilterNet/df/deepfilternet3.py` — Current SOTA model architecture
- `DeepFilterNet/df/modules.py` — Core neural network building blocks
- `DeepFilterNet/df/multiframe.py` — Multi-frame deep filtering operations
- `DeepFilterNet/df/loss.py` — Current loss function implementations
- `DeepFilterNet/df/train.py` — Training loop and optimization

### External
- #file:../research/20260105-deepfilternet4-architecture-research.md — Comprehensive architecture research
- Mamba paper: "Linear-Time Sequence Modeling with Selective State Spaces"
- MH-SENet (ISCA 2025): Hybrid time-frequency speech enhancement

---

## Delivery Framework

Each phase contains **atomic, non-overlapping tasks**. Each task is individually testable and constitutes a single measurable output. A task is not complete until:

- All specified files are created/modified
- Unit tests pass for the component
- Integration with existing codebase verified
- No regressions in existing functionality

---

## Phase 1 — Core Mamba Integration

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 1.1 | Mamba module implementation | `mamba.py` with `MambaBlock`, `Mamba` classes | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1-80) |
| [x] | 1.2 | SqueezedMamba layer | Drop-in replacement for `SqueezedGRU_S` | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 81-130) |
| [x] | 1.3 | Mamba unit tests | `tests/test_mamba.py` with comprehensive coverage | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 131-170) |
| [x] | 1.4 | Mamba benchmark script | Performance comparison vs GRU | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 171-200) |
| [x] | 1.5 | MPS compatibility for Mamba | Ensure Mamba works on Apple Silicon | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 201-230) |

---

## [x] Phase 2 — Hybrid Encoder Architecture

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 2.1 | Time-domain encoder branch | `WaveformEncoder` in `hybrid_encoder.py` | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 231-290) |
| [x] | 2.2 | Phase encoder branch | `PhaseEncoder` processing unwrapped phase | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 291-340) |
| [x] | 2.3 | Cross-domain attention fusion | `CrossDomainAttention` module | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 341-400) |
| [x] | 2.4 | HybridEncoder integration | Complete encoder with all branches | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 401-460) |
| [x] | 2.5 | Hybrid encoder tests | Unit tests for all encoder components | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 461-500) |

---

## [x] Phase 3 — Multi-Resolution Deep Filtering

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 3.1 | MultiResolutionDF module | Multi-resolution DF with learnable weights | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 501-560) |
| [x] | 3.2 | AdaptiveOrderPredictor | SNR-based filter order selection | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 561-610) |
| [x] | 3.3 | Multi-res DF decoder | Decoder producing multi-resolution coefficients | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 611-670) |
| [x] | 3.4 | Multi-resolution DF tests | Integration tests for DF components | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 671-710) |

---

## [x] Phase 4 — Training Enhancements

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 4.1 | MultiPeriodDiscriminator | `discriminator.py` with MPD implementation | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 711-780) |
| [x] | 4.2 | Feature matching loss | Discriminator feature matching for stability | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 781-820) |
| [x] | 4.3 | DNSMOSProxy model | Differentiable DNSMOS approximation | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 821-890) |
| [x] | 4.4 | SpeakerContrastiveLoss | Speaker embedding preservation loss | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 891-940) |
| [x] | 4.5 | Loss class updates | Integrate new losses into training | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 941-1000) |
| [x] | 4.6 | GAN training loop | Update train.py for adversarial training | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1001-1060) |

---

## [x] Phase 5 — Model Variants & Optimization

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 5.1 | DFNet4 full model | Complete `deepfilternet4.py` integration | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1061-1150) |
| [x] | 5.2 | DFNet4Lite variant | 50% parameter reduction variant | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1151-1200) |
| [x] | 5.3 | Config parameters | `ModelParams4` with all new config options | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1201-1260) |
| [x] | 5.4 | Quantization-aware training | QAT support for INT8 deployment | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1261-1310) |
| [x] | 5.5 | Knowledge distillation | Training script for Lite variant | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1311-1360) |

---

## [x] Phase 6 — Integration & Testing

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 6.1 | Full integration tests | End-to-end model tests | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1361-1410) |
| [x] | 6.2 | Performance benchmarks | PESQ/STOI/DNSMOS evaluation scripts | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1411-1460) |
| [x] | 6.3 | Model export | ONNX export with validation | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1461-1500) |
| [x] | 6.4 | Documentation | README updates and architecture docs | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1501-1550) |
| [x] | 6.5 | Migration guide | Guide for DFNet3 → DFNet4 users | .copilot-tracking/details/20260105-deepfilternet4-architecture-details.md (Lines 1551-1600) |

---

## Operational Dependencies

- PyTorch >= 2.0 with MPS support
- mamba-ssm >= 1.2.0 (or pure PyTorch implementation)
- einops >= 0.7.0 for tensor operations
- resemblyzer >= 0.1.3 for speaker embeddings (optional)
- Existing DeepFilterNet infrastructure (libDF, pyDF, etc.)

---

## Acceptance Criteria

A phase is complete only when:

- All tasks in the phase marked `[x]`
- Unit tests achieve >90% coverage for new code
- Integration tests pass with existing DFNet3 tests
- No performance regression in inference speed
- Memory usage within acceptable bounds

---

## Completion Definition

The work represented by this plan is considered **functionally complete** when:

- DFNet4 model achieves PESQ >= 3.45 on VoiceBank-DEMAND test set
- All 6 phases completed with passing tests
- Documentation complete for all new modules
- Both full and Lite variants operational
- MPS (Apple Silicon) support verified

