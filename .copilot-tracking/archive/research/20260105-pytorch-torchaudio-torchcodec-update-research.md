# Task Research Notes: Update DeepFilterNet to Latest PyTorch/TorchAudio/TorchCodec

## Research Executed

### File Analysis
- [DeepFilterNet/pyproject.toml](../DeepFilterNet/pyproject.toml)
  - Uses Poetry for package management
  - PyTorch/TorchAudio installed via `poe` tasks (pip commands)
  - Current version pins: `torch==2.1+cu118/cu121`, `torchaudio==2.1`
  - Python support: `>=3.8,<4.0`

- [DeepFilterNet/requirements.txt](../DeepFilterNet/requirements.txt)
  - Specifies `torch >=2.0, <3.0` and `torchaudio >=2.0, <3.0`
  - Loose version constraints allow upgrade

- [DeepFilterNet/df/io.py](../DeepFilterNet/df/io.py)
  - Core audio I/O module using TorchAudio
  - Uses `torchaudio.load()`, `torchaudio.save()`, `torchaudio.info()`
  - Has backward-compatible try/except for `AudioMetaData` import
  - Handles `resample()` via `torchaudio.functional.resample`
  - Affected by TorchAudio 2.9+ deprecations

- [DeepFilterNet/df/evaluation_utils.py](../DeepFilterNet/df/evaluation_utils.py)
  - Uses `torchaudio.functional.highpass_biquad`
  - Uses `torchaudio.transforms.Resample`
  - These transforms are preserved in maintenance mode

- [DeepFilterNet/df/train.py](../DeepFilterNet/df/train.py)
  - Imports `torchaudio` for data loading
  - Core training pipeline

### Code Search Results
- `import torchaudio` / `from torchaudio`
  - 18 matches across codebase
  - Main files: `df/io.py`, `df/train.py`, `df/evaluation_utils.py`, various scripts

- `ta.load`, `ta.save`, `ta.info`
  - 11 matches across scripts and io.py
  - All need migration path for TorchAudio 2.9+

- `torch.stft`, `torch.istft`
  - 11 matches in loss.py, stoi.py, visualization.py, tests
  - These are PyTorch core functions - no migration needed

### External Research
- #tool:githubRepo:"Rikorose/DeepFilterNet torchaudio 2.9 torchcodec"
  - **PR #670 already exists**: `df/io: Add support for torchaudio 2.9.` by @lubosz
  - Implements version-checking and TorchCodec fallback path
  - Status: Open, unmerged

- #tool:fetch:https://github.com/pytorch/audio/issues/3902
  - Official TorchAudio maintenance transition announcement
  - TorchAudio 2.8: Deprecation warnings added
  - TorchAudio 2.9: Deprecated APIs removed, `load()`/`save()` now use TorchCodec internally
  - TorchAudio 2.10+: Using LibTorch Stable ABI for remaining C++ extensions

- #tool:context7:/pytorch/pytorch
  - PyTorch 2.5 released Oct 2024
  - PyTorch 2.6 released Jan 2025
  - Current stable: 2.9.x (as of Jan 2026)

- #tool:context7:/pytorch/audio
  - TorchAudio 2.9 migration to maintenance phase
  - `torchaudio.load()` and `torchaudio.save()` now backed by TorchCodec
  - Preserved APIs: `transforms`, `functional`, `compliance.kaldi`, `models`, `pipelines`
  - Removed: `info()`, backend selection, various I/O utilities

- #tool:context7:/pytorch/torchcodec
  - TorchCodec: New unified media decoding/encoding library
  - `AudioDecoder` for loading audio
  - `AudioEncoder` for saving audio
  - Supports resampling, channel conversion, format detection
  - FFmpeg-based, supports various formats

### Project Conventions
- Standards referenced: PyTorch ecosystem conventions, Rust/Python hybrid project
- Instructions followed: Version compatibility patterns via try/except imports

## Key Discoveries

### TorchAudio Maintenance Phase (CRITICAL)

**Timeline:**
- TorchAudio 2.7: No new features, maintenance announcement
- TorchAudio 2.8 (Aug 2025): Deprecation warnings for APIs to be removed
- TorchAudio 2.9 (Oct 2025): Deprecated APIs removed, `load()`/`save()` use TorchCodec internally
- TorchAudio 2.10+: Stable ABI for remaining C++ extensions

**What's Preserved:**
- `torchaudio.transforms.*` (MelSpectrogram, Resample, MFCC, etc.)
- `torchaudio.functional.*` (resample, highpass_biquad, etc.)
- `torchaudio.compliance.kaldi.*`
- `torchaudio.models.*`
- `torchaudio.pipelines.*`

**What's Removed/Changed:**
- `torchaudio.info()` → Use TorchCodec's `AudioDecoder.metadata`
- `torchaudio.backend.*` → Removed (FFmpeg/SoundFile selection)
- `torchaudio.list_audio_backends()` → Removed
- `torchaudio.load()` / `torchaudio.save()` → Still exist but use TorchCodec internally

### Current DeepFilterNet Usage Analysis

**Files Requiring Updates:**

1. **[df/io.py](../DeepFilterNet/df/io.py)** (HIGH PRIORITY)
   - `torchaudio.info()` - DEPRECATED in 2.8, REMOVED in 2.9
   - `torchaudio.load()` - Works but internally uses TorchCodec
   - `torchaudio.save()` - Works but internally uses TorchCodec
   - `AudioMetaData` import path changed

2. **Scripts using `ta.load()`** (MEDIUM PRIORITY)
   - `scripts/sample_from_hdf5.py`
   - `scripts/trim_silence_hdf5.py`
   - `scripts/fix_n_samples_hdf5.py`
   - `scripts/filter_dnsmos.py`
   - `scripts/dnsmos_dns5.py`
   - `scripts/list_attrs_in_hdf5.py`
   - `scripts/plot_spec.py`

**Files Using Preserved APIs (NO CHANGE NEEDED):**
- `df/evaluation_utils.py` - Uses `torchaudio.functional.highpass_biquad`, `torchaudio.transforms.Resample`
- `df/multiframe.py` - Comment reference only
- Files using `torch.stft/istft` - PyTorch core, not TorchAudio

### Existing PR #670 Analysis

PR #670 by @lubosz implements TorchAudio 2.9 support:
- Adds version check for TorchAudio version
- Implements TorchCodec fallback for metadata retrieval
- Provides graceful degradation path

### Version Compatibility Matrix

| Library | DeepFilterNet Current | Latest Stable | Recommended |
|---------|----------------------|---------------|-------------|
| PyTorch | 2.1 | 2.9.x | 2.5+ |
| TorchAudio | 2.1 | 2.9.1 | 2.5+ (with TorchCodec for 2.9+) |
| TorchCodec | Not used | 0.x | Required for TorchAudio 2.9+ |
| Python | >=3.8,<4.0 | 3.8-3.12 | >=3.9,<4.0 |

### TorchCodec API Examples

**Loading Audio (replacing `torchaudio.info()` + `torchaudio.load()`):**
```python
from torchcodec.decoders import AudioDecoder

decoder = AudioDecoder("audio.mp3")

# Get metadata (replaces torchaudio.info())
print(decoder.metadata.sample_rate)      # 44100
print(decoder.metadata.num_channels)     # 2
print(decoder.metadata.duration_seconds) # 180.5

# Load samples (replaces torchaudio.load())
samples = decoder.get_all_samples()
audio_tensor = samples.data  # shape: [channels, samples]
sample_rate = samples.sample_rate

# Load with resampling
decoder = AudioDecoder("audio.mp3", sample_rate=16000)
samples = decoder.get_all_samples()
```

**Saving Audio (replacing `torchaudio.save()`):**
```python
from torchcodec.encoders import AudioEncoder

encoder = AudioEncoder(samples=audio_tensor, sample_rate=16000)
encoder.to_file("output.wav")
encoder.to_file("output.mp3", bit_rate=128000)
```

### Migration Strategy Options

**Option A: Minimal Update (Version-Conditional)**
- Keep existing TorchAudio API calls
- Add version checking to use TorchCodec for 2.9+
- Pros: Backward compatible, minimal code changes
- Cons: Maintains two code paths

**Option B: Full TorchCodec Migration**
- Replace all TorchAudio I/O with TorchCodec directly
- Keep TorchAudio only for transforms/functional
- Pros: Future-proof, cleaner code
- Cons: Breaking change for users on older TorchAudio

**Option C: Leverage PR #670**
- Review and merge existing PR #670
- Extend if needed for additional scripts
- Pros: Community-contributed, already tested
- Cons: May need enhancement

## Recommended Approach

**Fork-Based Modernization Strategy**

This fork will directly integrate improvements from open PRs and modernize the codebase without waiting for upstream merges. The goal is a maintained, up-to-date fork supporting the latest PyTorch ecosystem.

**Implementation Phases:**

### Phase 1: Core Compatibility Updates (CRITICAL)

Cherry-pick or reimplement changes from these PRs:

1. **PR #670 - TorchAudio 2.9 Support**
   - Reimplement version-checking logic in `df/io.py`
   - Add TorchCodec fallback for metadata retrieval
   - Test with TorchAudio 2.5, 2.8, and 2.9

2. **PR #648 - Python 3.13 Support**
   - Update PyO3 to v0.25.0 in `pyDF/Cargo.toml` and `pyDF-data/Cargo.toml`
   - Update numpy Rust binding to v0.25.0
   - Migrate method signatures per PyO3 migration guide
   - Update `.pyi` stub files

3. **PR #653 - torch.load Compatibility**
   - Add `weights_only=False` to `torch.load()` calls in `checkpoint.py`
   - Future-proofs against PyTorch's upcoming default change

### Phase 2: Bug Fixes & Improvements

4. **PR #666 - CUDA Tensor Fix**
   - Add `.detach().cpu()` before `.numpy()` in `df_features()`
   - Enables CUDA tensor workflows

5. **PR #619 - Device Selection**
   - Add `--device` CLI argument
   - Add MPS (Apple Silicon) auto-detection
   - Expand device selection logic

### Phase 3: Performance Enhancements

6. **PR #617 - LADSPA Plugin Refactor** (if using LADSPA)
   - Replace delay mechanism with ring buffer
   - Remove spin loops
   - Fix resource leak

7. **PR #664 - Streaming CLI**
   - Implement O(1) memory streaming for Rust CLI
   - Chunked resampling helpers

### Phase 4: Platform Expansion (Optional)

8. **PR #641 - Android & iOS Support** (if needed)
   - JNI bindings for Android
   - iOS framework
   - Mobile CI/CD pipelines

### Phase 5: Fork Maintenance

- Update `pyproject.toml` with modern dependency versions
- Update README to reflect fork status and improvements
- Set up CI/CD for the fork
- Consider renaming package to avoid confusion with upstream

## Implementation Guidance

### Objectives

- Create a modernized fork of DeepFilterNet
- Support PyTorch 2.5+ and TorchAudio 2.5+ through 2.9+
- Support Python 3.9 through 3.13
- Maintain backward compatibility where practical
- Integrate fixes and improvements from community PRs

### Fork Setup Tasks

1. Fork repository and set up development environment
2. Cherry-pick or reimplement PR #670 (TorchAudio 2.9)
3. Cherry-pick or reimplement PR #648 (Python 3.13 / PyO3 update)
4. Apply PR #653 fix (torch.load weights_only)
5. Apply PR #666 fix (CUDA tensor handling)
6. Optionally apply PR #619 (device selection enhancement)
7. Update `DeepFilterNet/pyproject.toml`:
   - Bump Python version to `>=3.9,<4.0`
   - Update poe task versions for torch/torchaudio
   - Add `torchcodec` as optional dependency
8. Update Cargo.toml files with modern Rust dependencies
9. Test across Python 3.9, 3.11, 3.13 and TorchAudio versions
10. Update README to document fork improvements

### Dependency Updates for Fork

**Python (`pyproject.toml`):**
```toml
python = ">=3.9,<4.0"
torch = ">=2.5,<3.0"
torchaudio = ">=2.5,<3.0"
torchcodec = {version = ">=0.1", optional = true}
```

**Rust (`Cargo.toml`):**
```toml
pyo3 = "0.25"
numpy = "0.25"
```

### Code Pattern for Version-Aware Audio Loading

```python
import torchaudio
from packaging import version

TORCHAUDIO_VERSION = version.parse(torchaudio.__version__)
USE_TORCHCODEC = TORCHAUDIO_VERSION >= version.parse("2.9.0")

if USE_TORCHCODEC:
    try:
        from torchcodec.decoders import AudioDecoder
        HAS_TORCHCODEC = True
    except ImportError:
        HAS_TORCHCODEC = False
        # Fallback or raise helpful error
else:
    HAS_TORCHCODEC = False

def load_audio(file: str, sr: Optional[int] = None) -> Tuple[Tensor, int]:
    if USE_TORCHCODEC and HAS_TORCHCODEC:
        decoder = AudioDecoder(file, sample_rate=sr) if sr else AudioDecoder(file)
        samples = decoder.get_all_samples()
        return samples.data, samples.sample_rate
    else:
        audio, orig_sr = torchaudio.load(file)
        if sr and orig_sr != sr:
            audio = torchaudio.functional.resample(audio, orig_sr, sr)
        return audio, sr or orig_sr
```

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| TorchCodec not installed on 2.9+ | Clear error message with install instructions |
| FFmpeg not available | Document FFmpeg installation requirements |
| Windows support (TorchCodec) | TorchCodec Windows support is experimental; document limitation |
| API differences | Version-aware wrapper functions |

---

## Open Pull Requests Analysis

This section documents unmerged PRs that are relevant to modernizing DeepFilterNet.

### PR #670: TorchAudio 2.9 Support (CRITICAL)

| Field | Value |
|-------|-------|
| **Title** | df/io: Add support for torchaudio 2.9. |
| **Author** | @lubosz |
| **Status** | Open |
| **Files Changed** | 1 (df/io.py) |
| **Lines** | +34/-11 |
| **URL** | https://github.com/Rikorose/DeepFilterNet/pull/670 |

**Description:**
Implements version-checking for TorchAudio version and provides TorchCodec fallback path for `torchaudio.info()` which was removed in TorchAudio 2.9.

**Key Changes:**
- Adds `packaging` dependency for version parsing
- Detects TorchAudio version at import time
- For TorchAudio 2.9+: Uses `torchcodec.decoders.AudioDecoder` for metadata
- Maintains backward compatibility with older TorchAudio versions

**Relevance:** ESSENTIAL - Directly addresses the TorchAudio 2.9 breaking change.

---

### PR #648: Python 3.13 Support (HIGH PRIORITY)

| Field | Value |
|-------|-------|
| **Title** | Update pyo3 to build for Python 3.13 |
| **Author** | @benniekiss |
| **Status** | Open |
| **Files Changed** | 6 |
| **Lines** | +711/-713 |
| **URL** | https://github.com/Rikorose/DeepFilterNet/pull/648 |

**Description:**
Updates PyO3 from older version to v0.25.0 and numpy binding to v0.25.0, enabling builds for Python 3.13.

**Key Changes:**
- Updates `pyo3` to v0.25.0 (pyDF/Cargo.toml, pyDF-data/Cargo.toml)
- Updates `numpy` (Rust binding) to v0.25.0
- Migrates method signatures (v21 → v22 migration)
- Updates type definitions
- Updates Cargo.lock
- Updates `.pyi` stub files

**Migration Notes:**
- PyO3 v0.21 → v0.25 includes breaking API changes
- Method signatures need migration per PyO3 migration guide
- Community-tested and reported working on Python 3.13

**Relevance:** HIGH - Enables Python 3.13 support, which is current stable Python.

---

### PR #641: Android & iOS Implementation (FEATURE)

| Field | Value |
|-------|-------|
| **Title** | Android & IOS implementation |
| **Author** | @s-sciacovelli (KaleyraVideo) |
| **Status** | Open |
| **Files Changed** | 14 |
| **Lines** | +1,703/-667 |
| **URL** | https://github.com/Rikorose/DeepFilterNet/pull/641 |

**Description:**
Comprehensive mobile platform support adding real-time noise suppression for Android and iOS applications.

**Key Features:**
- Rust-based JNI bindings for Android
- iOS framework with Swift/ObjC compatibility
- CI/CD pipelines for mobile builds (build_ios.yml, build_android.yml)
- Model optimization scripts for mobile performance
- 16KB alignment enforcement for Android .so files (required for newer Android)

**Implementation Details:**
- Uses libDF Rust core for processing
- JNI bridge for Android integration
- XCFramework for iOS distribution
- `df_process_frame()` function with buffer size parameter
- Real-time processing optimizations

**Technical Notes:**
- 24 commits from multiple contributors (KaleyraVideo team)
- No benchmarks provided vs. base DeepFilterNet3
- Uses optimized model for mobile (quality tradeoffs not documented)

**Relevance:** FEATURE - Expands platform support significantly, but large PR requiring careful review.

---

### PR #619: Device Selection Enhancement (FEATURE)

| Field | Value |
|-------|-------|
| **Title** | [feat] allow specifying compute device |
| **Author** | @benniekiss |
| **Status** | Open |
| **Files Changed** | 2 |
| **Lines** | +46/-12 |
| **URL** | https://github.com/Rikorose/DeepFilterNet/pull/619 |

**Description:**
Allows users to explicitly specify the PyTorch compute device (cpu, cuda, mps) instead of relying only on automatic selection.

**Key Changes:**
- Adds `--device` CLI argument to specify device
- Adds support for `mps` (Apple Silicon) device auto-detection
- Expands automatic device selection to include more device types
- Maintains backward compatibility with automatic selection

**Use Case:**
Addresses issue where users needed to run on specific devices (e.g., MPS on Apple Silicon) but couldn't override automatic device selection.

**Related Issues:**
- Fixes #624: "Ability to run on CPU"

**Relevance:** USEFUL - Improves flexibility for users on different hardware, especially Apple Silicon.

---

### PR #617: LADSPA Plugin Refactor (BUGFIX/PERFORMANCE)

| Field | Value |
|-------|-------|
| **Title** | Refactor noise processing |
| **Author** | @danielhuang |
| **Status** | Open |
| **Files Changed** | 3 |
| **Lines** | +757/-829 |
| **URL** | https://github.com/Rikorose/DeepFilterNet/pull/617 |

**Description:**
Major refactor of the LADSPA plugin's audio processing pipeline to address latency, stuttering, and resource leak issues.

**Problems Addressed:**
1. **Latency accumulation**: Original implementation introduced delay that didn't recover when CPU load decreased
2. **Cross-stream stuttering**: Processing delays caused EasyEffects output streams to stutter
3. **Resource leak**: Background thread continued polling after deactivation
4. **Spin loops**: Inefficient CPU usage when idle

**Key Changes:**
- Replaces sample dropping/delay mechanism with ring buffer approach
- Samples only dropped when ring buffer is full
- Removes spin loops (thread sleeps when no activity)
- Fixes memory/resource leak in background processing thread
- Non-blocking `run` method for LADSPA plugin

**Caveats:**
- Designed for real-time (online) processing
- May not work optimally with offline processing (FFmpeg batch mode)
- Some users report stuttering on lower-power devices (Fedora balanced mode)

**Related Issues:**
- Fixes wwmm/easyeffects#3154: "Sound stutter under CPU load when using DNR"
- Related: wwmm/easyeffects#3281: "easyeffects causes unusual events/s in powertop"

**Relevance:** HIGH - Significant quality-of-life improvement for LADSPA plugin users.

---

### PR #666: CUDA Tensor Fix (BUGFIX)

| Field | Value |
|-------|-------|
| **Title** | Fix: handle CUDA tensors in df_features() by moving audio to CPU before NumPy conversion |
| **Author** | @F1xxs |
| **Status** | Open |
| **Files Changed** | 1 |
| **Lines** | +1/-1 |
| **URL** | https://github.com/Rikorose/DeepFilterNet/pull/666 |

**Description:**
Fixes a bug where CUDA tensors couldn't be passed to `df_features()` because the code attempted direct NumPy conversion without moving to CPU first.

**Problem:**
```python
spec = df.analysis(audio.numpy())  # [C, Tf] -> [C, Tf, F]
# TypeError: can't convert cuda:0 device type tensor to numpy
```

**Fix:**
```python
spec = df.analysis(audio.detach().cpu().numpy())
```

**Key Changes:**
- Adds `.detach().cpu()` before `.numpy()` conversion
- Enables usage with external models operating on CUDA tensors

**Relevance:** USEFUL - Simple fix enabling CUDA tensor workflows.

---

### PR #664: Streaming CLI Enhancement (PERFORMANCE)

| Field | Value |
|-------|-------|
| **Title** | Stream enhance CLI processing - O(1) memory |
| **Author** | @otto-dev |
| **Status** | Open |
| **Files Changed** | 1 |
| **Lines** | +511/-29 |
| **URL** | https://github.com/Rikorose/DeepFilterNet/pull/664 |

**Description:**
Refactors the Rust CLI (`deep-filter` binary) to use streaming I/O, achieving O(1) memory usage instead of loading entire files into memory.

**Key Changes:**
- Streams WAV I/O in hop-sized chunks (avoids multi-GB buffers)
- Adds chunked resampling helpers for incremental model feeding
- Writes enhanced samples directly to WavWriter with delay compensation
- Significant memory reduction for large files

**Build Command:**
```bash
cargo check -p deep_filter --bin deep-filter --features "bin tract wav-utils transforms"
```

**Relevance:** HIGH - Major improvement for processing large audio files.

---

### PR #653: torch.load weights_only Fix (COMPATIBILITY)

| Field | Value |
|-------|-------|
| **Title** | [checkpoint.py] Set torch.load weights_only to False explicitly |
| **Author** | @bukshuk |
| **Status** | Open |
| **Files Changed** | 1 |
| **Lines** | +1/-1 |
| **URL** | https://github.com/Rikorose/DeepFilterNet/pull/653 |

**Description:**
Addresses PyTorch's change where `torch.load()` defaults to `weights_only=True` starting from a future version, which breaks checkpoint loading.

**Problem:**
PyTorch is transitioning to `weights_only=True` as the default for security reasons. DeepFilterNet checkpoints contain more than just weights (optimizer state, config, etc.), requiring `weights_only=False`.

**Fix:**
```python
# Before
torch.load(checkpoint_path)

# After
torch.load(checkpoint_path, weights_only=False)
```

**Relevance:** COMPATIBILITY - Required for future PyTorch versions.

---

## PR Integration Priority Matrix (Fork Strategy)

| PR | Priority | Category | Action | Complexity |
|----|----------|----------|--------|------------|
| #670 | CRITICAL | TorchAudio 2.9 | Cherry-pick | Low |
| #648 | CRITICAL | Python 3.13 | Cherry-pick | Medium |
| #653 | HIGH | PyTorch Compat | Cherry-pick | Low |
| #666 | HIGH | CUDA Fix | Cherry-pick | Low |
| #619 | MEDIUM | Feature | Cherry-pick | Low |
| #617 | MEDIUM | LADSPA Perf | Evaluate | High |
| #664 | LOW | CLI Perf | Evaluate | Medium |
| #641 | OPTIONAL | Mobile | Evaluate | High |

### Implementation Order for Fork:

**Sprint 1: Core Modernization**
1. **#670** - TorchAudio 2.9 support (unblocks modern PyTorch)
2. **#648** - Python 3.13 + PyO3 update (enables modern Python)
3. **#653** - weights_only fix (simple, prevents future breakage)

**Sprint 2: Bug Fixes & Enhancements**
4. **#666** - CUDA tensor fix (enables GPU workflows)
5. **#619** - Device selection (user-requested feature)

**Sprint 3: Performance (If Needed)**
6. **#617** - LADSPA refactor (only if using LADSPA plugin)
7. **#664** - Streaming CLI (only if processing large files)

**Future/Optional:**
8. **#641** - Android/iOS (only if targeting mobile platforms)

---

## References

- [TorchAudio Maintenance Transition Issue #3902](https://github.com/pytorch/audio/issues/3902)
- [TorchAudio 2.9 Release Notes](https://github.com/pytorch/audio/releases/tag/v2.9.0)
- [TorchCodec Documentation](https://docs.pytorch.org/torchcodec/)
- [DeepFilterNet PR #670](https://github.com/Rikorose/DeepFilterNet/pull/670)
- [DeepFilterNet PR #648](https://github.com/Rikorose/DeepFilterNet/pull/648)
- [DeepFilterNet PR #641](https://github.com/Rikorose/DeepFilterNet/pull/641)
- [DeepFilterNet PR #619](https://github.com/Rikorose/DeepFilterNet/pull/619)
- [DeepFilterNet PR #617](https://github.com/Rikorose/DeepFilterNet/pull/617)
- [DeepFilterNet PR #666](https://github.com/Rikorose/DeepFilterNet/pull/666)
- [DeepFilterNet PR #664](https://github.com/Rikorose/DeepFilterNet/pull/664)
- [DeepFilterNet PR #653](https://github.com/Rikorose/DeepFilterNet/pull/653)
- [PyTorch Release Schedule](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [PyO3 Migration Guide](https://pyo3.rs/v0.25.0/migration)
