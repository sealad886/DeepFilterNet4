---
applyTo: '.copilot-tracking/changes/20260105-mps-backend-support-changes.md'
---
<!-- markdownlint-disable-file -->
# MPS Backend Support Enhancement — Structured Delivery Plan

## Purpose

Ensure DeepFilterNet inference works reliably on Apple Silicon MPS backend with appropriate runtime warnings, graceful degradation for unsupported operations, comprehensive documentation, and validation testing.

---

## Reference Inputs

### Internal

- [DeepFilterNet/df/utils.py](DeepFilterNet/df/utils.py) — Device detection with existing MPS support
- [DeepFilterNet/df/modules.py](DeepFilterNet/df/modules.py) — DfOp class with complex and real forward methods
- [DeepFilterNet/df/enhance.py](DeepFilterNet/df/enhance.py) — Core inference pipeline with CPU↔device data flow
- [DeepFilterNet/df/stoi.py](DeepFilterNet/df/stoi.py) — STOI evaluation using torch.norm (MPS incompatible for complex)

### External

- #file:../research/20260713-mps-backend-support-research.md — Comprehensive MPS compatibility research
- [PyTorch MPS Backend Documentation](https://docs.pytorch.org/docs/stable/notes/mps.html) — Official MPS guidance
- [PyTorch MPS Op Coverage Issue #77764](https://github.com/pytorch/pytorch/issues/77764) — Operation support tracking

---

## Delivery Framework

Each phase contains **atomic, non-overlapping tasks**. Each task is individually testable and constitutes a single measurable output. A task is not complete until:

- Code compiles without errors
- Existing tests pass (no regressions)
- New functionality is validated on target platform or with appropriate mocks
- Documentation accurately reflects implementation

---

## Phase 1 — Runtime Detection and Warnings

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 1.1 | macOS version detection utility | Function to detect macOS version and return tuple (major, minor) | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 10-30) |
| [x] | 1.2 | MPS complex operation compatibility check | Function to verify macOS 14+ for complex tensor operations | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 32-55) |
| [x] | 1.3 | Runtime warning for incompatible configurations | Warning logged when MPS selected on macOS < 14 with complex ops | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 57-80) |
| [x] | 1.4 | Environment variable documentation | Document PYTORCH_ENABLE_MPS_FALLBACK=1 as fallback strategy | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 82-95) |

---

## Phase 2 — Graceful Degradation Implementation

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 2.1 | DfOp forward method selection logic | Auto-select `forward_real_unfold` when complex ops unsupported | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 100-130) |
| [x] | 2.2 | STOI evaluation CPU fallback | Move tensors to CPU before torch.norm on complex in stoi.py | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 132-155) |
| [x] | 2.3 | Configuration option for forward method | Add config option to override forward method selection | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 157-180) |

---

## Phase 3 — Documentation Updates

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 3.1 | README MPS section | Add MPS usage section with macOS 14+ requirement | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 185-215) |
| [x] | 3.2 | MPS compatibility table | Document supported/unsupported operations | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 217-245) |
| [x] | 3.3 | Performance expectations documentation | Document batch vs real-time speedup scenarios | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 247-275) |
| [x] | 3.4 | Troubleshooting guide | Common MPS issues and solutions | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 277-305) |

---

## Phase 4 — Testing and Validation

| Status | ID | Deliverable | Output Definition | Notes |
|--------|----|------------|------------------|-------|
| [x] | 4.1 | MPS pytest markers | Add @pytest.mark.mps for MPS-specific tests | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 310-335) |
| [x] | 4.2 | Device detection unit tests | Tests for get_device() with MPS scenarios | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 337-365) |
| [x] | 4.3 | Forward method compatibility tests | Test forward_complex_strided and forward_real_unfold on MPS | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 367-400) |
| [x] | 4.4 | End-to-end inference test | Full enhance() pipeline test on MPS device | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 402-430) |
| [x] | 4.5 | CI/CD MPS skip configuration | Configure tests to skip MPS tests when not on macOS | .copilot-tracking/details/20260105-mps-backend-support-details.md (Lines 432-455) |

---

## Operational Dependencies

- PyTorch 2.5.1+ with MPS support
- macOS 14 (Sonoma) or later for complex number operations
- torchaudio 2.5.1+ for audio processing
- pytest with markers support for conditional test execution

---

## Acceptance Criteria

A phase is complete only when:

- All tasks in the phase have passing tests
- No regressions in existing functionality
- Documentation accurately reflects implementation
- Code follows existing project conventions (get_device pattern, logging via loguru)

---

## Completion Definition

The work represented by this plan is considered **functionally complete** when:

- MPS backend works reliably for batch inference on macOS 14+
- Runtime warnings guide users on incompatible configurations
- Graceful fallback to real-valued operations when complex ops unsupported
- Documentation clearly explains MPS requirements and limitations
- Test suite validates MPS functionality with appropriate skip markers for CI
