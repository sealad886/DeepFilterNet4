<!-- markdownlint-disable-file -->
# Task Details: DeepFilterNet Fork Modernization

## Research Reference

**Source Research**: #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md

---

## Phase 1: TorchAudio 2.9 Compatibility

### Task 1.1: Add packaging dependency

Add the `packaging` library to pyproject.toml dependencies for version parsing.

- **Files**:
  - `DeepFilterNet/pyproject.toml` - Add `packaging` to dependencies section

- **Success**:
  - `packaging` appears in `[tool.poetry.dependencies]`
  - `poetry lock` succeeds

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 300-315) - Version detection pattern

- **Dependencies**:
  - None (first task)

### Task 1.2: Implement version detection

Add TorchAudio version detection at module load time in df/io.py.

- **Files**:
  - `DeepFilterNet/df/io.py` - Add version detection near imports

- **Code Pattern**:
```python
import torchaudio
from packaging import version

TORCHAUDIO_VERSION = version.parse(torchaudio.__version__)
USE_TORCHCODEC = TORCHAUDIO_VERSION >= version.parse("2.9.0")
```

- **Success**:
  - `USE_TORCHCODEC` constant is defined
  - Importing `df.io` does not raise errors

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 300-330) - Version detection implementation

- **Dependencies**:
  - Task 1.1 completion (packaging dependency)

### Task 1.3: Add TorchCodec fallback for metadata

Replace `torchaudio.info()` calls with TorchCodec AudioDecoder for TorchAudio 2.9+.

- **Files**:
  - `DeepFilterNet/df/io.py` - Modify `get_audio_info()` or equivalent function

- **Code Pattern**:
```python
if USE_TORCHCODEC:
    try:
        from torchcodec.decoders import AudioDecoder
        HAS_TORCHCODEC = True
    except ImportError:
        HAS_TORCHCODEC = False
else:
    HAS_TORCHCODEC = False

def get_audio_metadata(file: str):
    if USE_TORCHCODEC and HAS_TORCHCODEC:
        decoder = AudioDecoder(file)
        return decoder.metadata.sample_rate, decoder.metadata.num_channels
    else:
        info = torchaudio.info(file)
        return info.sample_rate, info.num_channels
```

- **Success**:
  - `get_audio_metadata()` returns correct sample rate and channels
  - Works with both TorchAudio <2.9 and >=2.9

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 145-165) - TorchCodec API examples
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 340-365) - PR #670 analysis

- **Dependencies**:
  - Task 1.2 completion (version detection)

### Task 1.4: Update AudioMetaData import

Fix the AudioMetaData import path which changed in TorchAudio 2.9.

- **Files**:
  - `DeepFilterNet/df/io.py` - Update try/except import block

- **Current Code**:
```python
try:
    from torchaudio.backend.common import AudioMetaData
except ImportError:
    from torchaudio import AudioMetaData
```

- **Updated Code**:
```python
try:
    from torchaudio import AudioMetaData
except ImportError:
    try:
        from torchaudio.backend.common import AudioMetaData
    except ImportError:
        # TorchAudio 2.9+: AudioMetaData may not be needed if using TorchCodec
        AudioMetaData = None
```

- **Success**:
  - No ImportError when importing df.io
  - AudioMetaData is available or gracefully handled

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 95-105) - AudioMetaData changes

- **Dependencies**:
  - Task 1.2 completion

### Task 1.5: Test TorchAudio compatibility

Verify load_audio and save_audio work correctly across TorchAudio versions.

- **Files**:
  - None (testing task)

- **Test Commands**:
```bash
# Test with current TorchAudio
python -c "from df.io import load_audio, save_audio; print('Import OK')"

# Test loading audio
python -c "from df.io import load_audio; audio, sr = load_audio('test.wav'); print(f'Loaded: {audio.shape}, {sr}Hz')"
```

- **Success**:
  - `load_audio()` returns tensor and sample rate
  - `save_audio()` writes valid audio file
  - No deprecation warnings on TorchAudio 2.8
  - No errors on TorchAudio 2.9

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 80-95) - TorchAudio timeline

- **Dependencies**:
  - Tasks 1.1-1.4 completion

---

## Phase 2: Python 3.13 / PyO3 Update

### Task 2.1: Update pyDF Cargo.toml

Update PyO3 and numpy versions in pyDF/Cargo.toml.

- **Files**:
  - `pyDF/Cargo.toml` - Update dependency versions

- **Changes**:
```toml
[dependencies]
pyo3 = { version = "0.25", features = ["extension-module"] }
numpy = "0.25"
```

- **Success**:
  - `cargo check -p pydf` succeeds (may have warnings until signatures migrated)

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 370-395) - PR #648 analysis
  - [PyO3 Migration Guide](https://pyo3.rs/v0.25.0/migration)

- **Dependencies**:
  - Phase 1 completion recommended

### Task 2.2: Update pyDF-data Cargo.toml

Update PyO3 and numpy versions in pyDF-data/Cargo.toml.

- **Files**:
  - `pyDF-data/Cargo.toml` - Update dependency versions

- **Changes**:
```toml
[dependencies]
pyo3 = { version = "0.25", features = ["extension-module"] }
numpy = "0.25"
```

- **Success**:
  - `cargo check -p pydf-data` succeeds

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 370-395) - PR #648 analysis

- **Dependencies**:
  - Task 2.1 completion

### Task 2.3: Migrate PyO3 method signatures

Update #[pyfunction] and #[pymethods] attributes per PyO3 0.25 migration guide.

- **Files**:
  - `pyDF/src/lib.rs` - Update function signatures
  - `pyDF-data/src/lib.rs` - Update function signatures

- **Key Changes (PyO3 v0.21 → v0.25)**:
  - `#[new]` methods now return `Self` directly instead of `PyResult<Self>`
  - `#[getter]` and `#[setter]` syntax changes
  - `PyAny` → `Bound<'_, PyAny>` for bound references
  - `Python::with_gil()` patterns may need updates

- **Success**:
  - `cargo check` passes without errors
  - All #[pyfunction] and #[pymethods] compile

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 380-395) - Migration notes
  - [PyO3 Migration Guide](https://pyo3.rs/v0.25.0/migration)

- **Dependencies**:
  - Tasks 2.1, 2.2 completion

### Task 2.4: Update type definitions

Migrate PyO3 type annotations and conversions for 0.25 compatibility.

- **Files**:
  - `pyDF/src/lib.rs` - Update type conversions
  - `pyDF-data/src/lib.rs` - Update type conversions

- **Key Changes**:
  - `PyReadonlyArray` → Use new numpy 0.25 API
  - `PyArray::from_owned_array` patterns
  - `IntoPy<PyObject>` trait implementations

- **Success**:
  - All type conversions compile
  - numpy array passing works correctly

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 370-395)

- **Dependencies**:
  - Task 2.3 completion

### Task 2.5: Regenerate .pyi stub files

Update Python type stub files to reflect new signatures.

- **Files**:
  - `pyDF/libdf.pyi` - Update type stubs

- **Success**:
  - `.pyi` file matches actual function signatures
  - Type checkers (mypy, pyright) don't report errors

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 370-395)

- **Dependencies**:
  - Tasks 2.3, 2.4 completion

### Task 2.6: Update Cargo.lock

Run cargo update to refresh all dependency versions.

- **Files**:
  - `Cargo.lock` - Will be auto-generated

- **Commands**:
```bash
cargo update
cargo check --workspace
```

- **Success**:
  - `Cargo.lock` updated with new versions
  - `cargo check --workspace` passes

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 370-395)

- **Dependencies**:
  - Tasks 2.1-2.4 completion

### Task 2.7: Test Python 3.13 build

Verify maturin build succeeds on Python 3.13.

- **Files**:
  - None (testing task)

- **Commands**:
```bash
# Ensure Python 3.13 environment
python --version  # Should show 3.13.x

# Build with maturin
cd pyDF && maturin develop
cd ../pyDF-data && maturin develop

# Test import
python -c "import libdf; print('libdf OK')"
python -c "import libdfdata; print('libdfdata OK')"
```

- **Success**:
  - maturin build completes without errors
  - Modules import successfully on Python 3.13

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 370-395)

- **Dependencies**:
  - Tasks 2.1-2.6 completion

---

## Phase 3: PyTorch Compatibility Fixes

### Task 3.1: Fix torch.load weights_only

Add `weights_only=False` to all `torch.load()` calls in checkpoint.py.

- **Files**:
  - `DeepFilterNet/df/checkpoint.py` - Update torch.load calls

- **Search Pattern**: `torch.load(`

- **Change**:
```python
# Before
state = torch.load(checkpoint_path)

# After
state = torch.load(checkpoint_path, weights_only=False)
```

- **Success**:
  - No FutureWarning about weights_only
  - Checkpoint loading works on PyTorch 2.5+

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 570-600) - PR #653 analysis

- **Dependencies**:
  - Phase 2 completion recommended

### Task 3.2: Fix CUDA tensor handling

Add `.detach().cpu()` before `.numpy()` conversion in enhance.py.

- **Files**:
  - `DeepFilterNet/df/enhance.py` - Fix df_features() or similar function

- **Search Pattern**: `.numpy()` on tensors that might be on GPU

- **Change**:
```python
# Before
spec = df.analysis(audio.numpy())

# After
spec = df.analysis(audio.detach().cpu().numpy())
```

- **Success**:
  - CUDA tensors can be processed without TypeError
  - CPU tensors still work correctly

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 505-535) - PR #666 analysis

- **Dependencies**:
  - Task 3.1 completion

### Task 3.3: Test checkpoint loading

Verify model loading works correctly with PyTorch 2.5+.

- **Files**:
  - None (testing task)

- **Test Commands**:
```bash
python -c "
from df.enhance import init_df
model, df_state, _ = init_df()
print(f'Model loaded: {type(model).__name__}')
"
```

- **Success**:
  - Model loads without warnings
  - Inference produces valid output

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 570-600)

- **Dependencies**:
  - Tasks 3.1, 3.2 completion

---

## Phase 4: Device Selection Enhancement

### Task 4.1: Add --device CLI argument

Add a `--device` argument to the enhance.py CLI.

- **Files**:
  - `DeepFilterNet/df/enhance.py` - Add argparse argument

- **Code Addition**:
```python
parser.add_argument(
    "--device",
    "-D",
    type=str,
    default=None,
    help="Compute device: cpu, cuda, cuda:0, mps, or auto (default: auto)"
)
```

- **Success**:
  - `--device` argument appears in `--help`
  - Argument value is passed to device selection logic

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 440-465) - PR #619 analysis

- **Dependencies**:
  - Phase 3 completion

### Task 4.2: Add MPS device detection

Add Apple Silicon MPS to automatic device detection logic.

- **Files**:
  - `DeepFilterNet/df/utils.py` or `df/enhance.py` - Update get_device()

- **Code Pattern**:
```python
def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None and device != "auto":
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

- **Success**:
  - MPS device selected on Apple Silicon when available
  - CUDA still preferred when available
  - Explicit device override works

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 440-465) - PR #619 analysis

- **Dependencies**:
  - Task 4.1 completion

### Task 4.3: Update device selection logic

Ensure device selection respects the --device argument throughout the codebase.

- **Files**:
  - `DeepFilterNet/df/enhance.py` - Wire up argument to init_df() and model loading

- **Success**:
  - `deepFilter -D cpu` forces CPU processing
  - `deepFilter -D cuda:1` uses specific GPU
  - `deepFilter -D mps` uses Apple Silicon

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 440-465)

- **Dependencies**:
  - Task 4.2 completion

---

## Phase 5: Dependency Updates

### Task 5.1: Update Python version constraint

Change Python requirement from `>=3.8` to `>=3.9`.

- **Files**:
  - `DeepFilterNet/pyproject.toml` - Update python version

- **Change**:
```toml
[tool.poetry.dependencies]
python = ">=3.9,<4.0"
```

- **Success**:
  - `poetry check` passes
  - Python 3.8 no longer listed as supported

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 280-295) - Dependency updates

- **Dependencies**:
  - Phase 4 completion

### Task 5.2: Update PyTorch version constraints

Update torch and torchaudio version requirements.

- **Files**:
  - `DeepFilterNet/pyproject.toml` - Update dependency versions
  - `DeepFilterNet/requirements.txt` - Update version constraints

- **Changes (pyproject.toml)**:
```toml
torch = ">=2.5,<3.0"
torchaudio = ">=2.5,<3.0"
```

- **Changes (requirements.txt)**:
```
torch >=2.5, <3.0
torchaudio >=2.5, <3.0
```

- **Success**:
  - Version constraints updated in both files
  - `pip install` resolves correctly

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 280-295)

- **Dependencies**:
  - Task 5.1 completion

### Task 5.3: Add TorchCodec optional dependency

Add torchcodec as an optional dependency for TorchAudio 2.9+ users.

- **Files**:
  - `DeepFilterNet/pyproject.toml` - Add optional dependency

- **Change**:
```toml
[tool.poetry.dependencies]
torchcodec = {version = ">=0.1", optional = true}

[tool.poetry.extras]
torchcodec = ["torchcodec"]
```

- **Success**:
  - `pip install deepfilterlib[torchcodec]` installs torchcodec
  - Base install works without torchcodec

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 280-295)

- **Dependencies**:
  - Task 5.2 completion

### Task 5.4: Update poe task versions

Update pip install commands in pyproject.toml poe tasks.

- **Files**:
  - `DeepFilterNet/pyproject.toml` - Update poe tasks

- **Current Tasks** (example):
```toml
[tool.poe.tasks]
install-torch-cuda118 = "pip install torch==2.1+cu118 torchaudio==2.1+cu118 ..."
```

- **Updated Tasks**:
```toml
[tool.poe.tasks]
install-torch-cuda121 = "pip install torch==2.5.0+cu121 torchaudio==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121"
install-torch-cuda124 = "pip install torch==2.5.0+cu124 torchaudio==2.5.0+cu124 --index-url https://download.pytorch.org/whl/cu124"
install-torch-cpu = "pip install torch==2.5.0+cpu torchaudio==2.5.0+cpu --index-url https://download.pytorch.org/whl/cpu"
```

- **Success**:
  - `poe install-torch-cuda121` installs PyTorch 2.5+
  - All poe tasks work correctly

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 5-15) - Current poe tasks

- **Dependencies**:
  - Task 5.3 completion

### Task 5.5: Update requirements.txt

Sync requirements.txt with pyproject.toml changes.

- **Files**:
  - `DeepFilterNet/requirements.txt` - Update all version constraints

- **Success**:
  - requirements.txt matches pyproject.toml
  - `pip install -r requirements.txt` works

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 15-20)

- **Dependencies**:
  - Task 5.4 completion

---

## Phase 6: Documentation & Testing

### Task 6.1: Update README fork notice

Add a notice explaining this is a modernized fork.

- **Files**:
  - `README.md` - Add fork notice near top

- **Content**:
```markdown
## Fork Notice

This is a modernized fork of [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) 
with the following improvements:

- **Python 3.13 support** (PyO3 0.25)
- **TorchAudio 2.9 support** (TorchCodec integration)
- **PyTorch 2.5+ compatibility** (weights_only fix)
- **Apple Silicon MPS support**
- **CUDA tensor handling fix**
```

- **Success**:
  - Fork notice visible at top of README
  - Links to original repository

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 600-630) - PR integration matrix

- **Dependencies**:
  - Phase 5 completion

### Task 6.2: Document new dependencies

Add TorchCodec and FFmpeg requirements to README.

- **Files**:
  - `README.md` - Update installation section

- **Content**:
```markdown
### TorchAudio 2.9+ Users

For TorchAudio 2.9+, install TorchCodec for audio I/O:

```bash
pip install torchcodec
```

TorchCodec requires FFmpeg libraries. On Ubuntu/Debian:
```bash
sudo apt install ffmpeg libavcodec-dev libavformat-dev
```
```

- **Success**:
  - Installation instructions are clear
  - TorchCodec requirement documented

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 330-340) - Risks and mitigations

- **Dependencies**:
  - Task 6.1 completion

### Task 6.3: Add version compatibility table

Document supported version matrix in README.

- **Files**:
  - `README.md` - Add compatibility section

- **Content**:
```markdown
### Version Compatibility

| Component | Minimum | Recommended | Maximum |
|-----------|---------|-------------|---------|
| Python | 3.9 | 3.11+ | 3.13 |
| PyTorch | 2.5 | 2.5+ | <3.0 |
| TorchAudio | 2.5 | 2.5+ | <3.0 |
| TorchCodec | 0.1 | Latest | - |
```

- **Success**:
  - Compatibility table is accurate
  - Users can quickly check requirements

- **Research References**:
  - #file:../research/20260105-pytorch-torchaudio-torchcodec-update-research.md (Lines 135-145) - Version matrix

- **Dependencies**:
  - Task 6.2 completion

### Task 6.4: Run existing tests

Execute pytest to verify no regressions.

- **Files**:
  - None (testing task)

- **Commands**:
```bash
cd DeepFilterNet
pytest tests/ -v
```

- **Success**:
  - All existing tests pass
  - No new failures introduced

- **Research References**:
  - N/A

- **Dependencies**:
  - Tasks 6.1-6.3 completion

---

## Dependencies Summary

- Python 3.9+ (3.13 for full support)
- PyTorch 2.5+ with TorchAudio 2.5+
- TorchCodec (required for TorchAudio 2.9+)
- FFmpeg libraries (required by TorchCodec)
- Rust toolchain (for PyO3 builds)
- maturin (for Python/Rust wheel builds)
- packaging (Python library for version parsing)

## Success Criteria

- DeepFilterNet imports and runs on Python 3.9, 3.11, and 3.13
- TorchAudio 2.5, 2.8, and 2.9 are all supported
- Model checkpoint loading works on PyTorch 2.5+
- CUDA and MPS device selection works correctly
- All existing tests pass
- README documents fork improvements and compatibility
