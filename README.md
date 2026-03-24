> [!CAUTION]
> The most actively developed path in this repo is `DeepFilterNet/df_mlx/`: native MLX training and inference on Apple Silicon plus speech-aware VAD gating. Some older sections further down in this README still describe the broader framework and are being refreshed.

# DeepFilterNet

This repository contains the broader DeepFilterNet codebase together with an actively developed MLX implementation of DeepFilterNet4 in `DeepFilterNet/df_mlx/`.

If you want to get started on Apple Silicon, use the helper script:

```sh
# Omit `--venv` if you want to install in the current virtual environment.
# Use `-h/--help` for help.
./setup.sh --all [--venv /path/to/venv]
```

## `df_mlx` today

The current `df_mlx` module is centered on:

- native MLX `DfNet4` / `DfNet4Lite` models for Apple Silicon
- a dynamic training entrypoint at `python -m df_mlx.train_dynamic`
- a structured TOML `RunConfig` system with hardware presets: `entry`, `pro`, `max`, `ultra`, and `debug`
- a new VAD path built around `VadHead` plus inference-time soft gating
- batch and streaming inference paths that both apply the same VAD gating behavior

For module-level detail, see `DeepFilterNet/df_mlx/README.md`.

## Running `df_mlx` training

The supported CLI lives in `DeepFilterNet/df_mlx/training_cli_main.py`, and `python -m df_mlx.train_dynamic` is the supported module entrypoint.

Start by generating a commented run-config template:

```bash
python -m df_mlx.train_dynamic --print-run-config
```

A typical preset-based run looks like this:

```bash
python -m df_mlx.train_dynamic \
    --preset pro \
    --cache-dir /path/to/audio_cache \
    --run-config /path/to/run_config.toml
```

If you are not training from a prebuilt cache, `train_dynamic` can also read a dataset/mixer JSON or plain file lists:

```bash
python -m df_mlx.train_dynamic \
    --config /path/to/config.json \
    --run-config /path/to/run_config.toml

python -m df_mlx.train_dynamic \
    --speech-list /path/to/speech.txt \
    --noise-list /path/to/noise.txt \
    --rir-list /path/to/rirs.txt \
    --run-config /path/to/run_config.toml
```

The CLI distinguishes three different configuration inputs:

- `--config`: dataset/mixer JSON for dynamic data generation
- `--run-config`: TOML runtime and training settings for `RunConfig`
- `--train-config`: optional train.py-compatible legacy INI, translated by `df_mlx/train_dynamic_config.py`

For new runs, prefer the TOML flow. The documented precedence is:

```text
defaults < preset < train-config INI compatibility < run-config TOML < CLI flags
```

The preset layer is optional; when you use one, it supplies the base `RunConfig`. The run-config TOML and explicit CLI flags still win.

See also:

- `docs/RUN_CONFIG_PRESETS.md`
- `DeepFilterNet/df_mlx/README.md`

## New config system: `run_config.toml` vs legacy `.ini`

`df_mlx` no longer expects you to drive training from a single `config.ini`. The current configuration system is the `RunConfig` dataclass in `DeepFilterNet/df_mlx/run_config.py`, loaded from TOML and validated before training starts.

| Concern | Legacy flow | Current `df_mlx` flow |
| --- | --- | --- |
| Main runtime / training settings | train.py-style `config.ini` | `run_config.toml` (`RunConfig`) |
| Parser / adapter | `configparser`-style INI | TOML via `tomllib` / `tomli` |
| Presets | none | `--preset entry\|pro\|max\|ultra\|debug` |
| Config shape | flat sections | nested tables such as `[dataset]`, `[training]`, `[checkpoint]`, `[loss]`, and `[vad]` |
| Stage scheduling | ad hoc | `loss.pipeline_stages` in TOML |
| Backward compatibility | native | `--train-config` or `[train_ini.*]` compatibility tables |
| Final override point | CLI | CLI |

If you still have an older `config.ini`, `df_mlx/train_dynamic_config.py` can map supported sections such as `[df]`, `[train]`, `[optim]`, `[distortion]`, and `[deepfilternet4]` into the MLX training path. Unsupported legacy sections are ignored with warnings rather than silently misapplied.

## VAD head and soft mask path

This fork's headline model addition is the `VadHead` path in `DeepFilterNet/df_mlx/model.py`.

- `DfNet4` now instantiates `VadHead` alongside `ErbDecoder4` and `DfDecoder4`.
- `VadHead` predicts raw logits from the shared backbone embedding.
- During training, `train_dynamic.py` combines:
  - the proxy VAD consistency loss from `df_mlx/training_losses.py`
  - a BCE-with-logits head loss against the clean-signal VAD proxy target
- During inference, the logits become a soft VAD mask / gate:
  - `mx.maximum(mx.sigmoid(vad_logits), 0.01)`
  - that gate is applied to the enhanced spectrum in both `DfNet4.__call__` and `StreamingDfNet4`

![Diagram of the `df_mlx` VAD head and soft mask path](docs/images/df_mlx_vad_mask_layer.svg)

The diagram uses “VAD mask” as shorthand for the inference-time soft gate. In code, the concrete pieces are `VadHead`, the BCE-with-logits supervision path, and the final sigmoid gate applied to the output spectrum.

## Other `df_mlx` changes worth knowing about

In addition to the VAD path, the current MLX implementation exposes:

- selectable backbones via `RunConfig.model.backbone_type`: `mamba`, `gru`, or `attention`
- `full` and `lite` model variants via `RunConfig.model.variant`
- LSNR estimation in `Encoder4`, plus optional LSNR-based dropout during training
- streaming inference via `StreamingDfNet4`
- hardware-aware run presets and dataloader tuning in `RunConfig`

## Loss docs and background reading

If you want the math and rationale behind the training objectives, start here:

- `docs/LOSSES.md` — the technical reference for the composite loss, checkpoints, and weight-update paths
- `docs/LOSSES_AN_INTRODUCTION_TO_LOSS.md` — higher-level background on how the loss stack fits together
- `docs/LOSS_LANDSCAPE_ANALYSIS.md` — follow-on analysis of weighting tradeoffs and VAD-head gradient pressure
- `docs/INCIDENT_LOSS_VAD_AUDIT_2025_01.md` — audit history for the VAD/loss correctness fixes that shaped the current implementation

Those documents are the best place to go before changing `awesome`, `pipeline_awesome`, GAN, speech-band, or VAD-related weights.

## Usage

### deep-filter

Download a pre-compiled `deep-filter` binary from this repository's Releases page.
You can use `deep-filter` to suppress noise in noisy .wav audio files. Currently, only wav files with a sampling rate of 48kHz are supported.

```bash
USAGE:
    deep-filter [OPTIONS] [FILES]...

ARGS:
    <FILES>...

OPTIONS:
    -D, --compensate-delay
            Compensate delay of STFT and model lookahead
    -h, --help
            Print help information
    -m, --model <MODEL>
            Path to model tar.gz. Defaults to DeepFilterNet2.
    -o, --out-dir <OUT_DIR>
            [default: out]
    --pf
            Enable postfilter
    -v, --verbose
            Logging verbosity
    -V, --version
            Print version information
```

If you want to use the pytorch backend e.g. for GPU processing, see further below for the Python usage.

### DeepFilterNet Framework

This framework supports Linux, MacOS and Windows. Training is only tested under Linux. The framework is structured as follows:

* `libDF` contains Rust code used for data loading and augmentation.
* `DeepFilterNet` contains DeepFilterNet code training, evaluation and visualization as well as pretrained model weights.
* `pyDF` contains a Python wrapper of libDF STFT/ISTFT processing loop.
* `pyDF-data` contains a Python wrapper of libDF dataset functionality and provides a pytorch data loader.
* `ladspa` contains a LADSPA plugin for real-time noise suppression.
* `models` contains pretrained for usage in DeepFilterNet (Python) or libDF/deep-filter (Rust)

### DeepFilterNet Python: PyPI

Install the DeepFilterNet Python wheel via pip:
```bash
# Install cpu/cuda pytorch (>=1.9) dependency from pytorch.org, e.g.:
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Install DeepFilterNet
pip install deepfilternet
# Or install DeepFilterNet including data loading functionality for training (Linux only)
pip install deepfilternet[train]
```

To enhance noisy audio files using DeepFilterNet run
```bash
# Specify an output directory with --output-dir [OUTPUT_DIR]
deepFilter path/to/noisy_audio.wav
```

### Manual Installation

Install cargo via [rustup](https://rustup.rs/). Use a local `virtualenv`/`venv`
or conda environment; do not install repository dependencies into the global
Python.
Please read the comments and only execute the commands that you need.

Recommended developer setup:
```bash
cd path/to/DeepFilterNet/
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ./DeepFilterNet[train,eval]

# Apple Silicon / df_mlx development:
python3 -m pip install -r DeepFilterNet/requirements_mlx.txt
```

Installation of python dependencies and libDF:
```bash
cd path/to/DeepFilterNet/  # cd into repository
# Recommended: Install or activate a python env
# Mandatory: Install cpu/cuda pytorch (>=1.8) dependency from pytorch.org, e.g.:
python3 -m pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Install build dependencies used to compile libdf and DeepFilterNet python wheels
python3 -m pip install maturin setuptools wheel

#  Install remaining DeepFilterNet python dependencies
# *Option A:* Install DeepFilterNet python wheel globally within your environment. Do this if you want use
# this repos as is, and don't want to develop within this repository.
python3 -m pip install ./DeepFilterNet[train,eval]
# *Option B:* If you want to develop within this repo, install only dependencies and work with the repository version
python3 -m pip install -e ./DeepFilterNet[train,eval]

# If you are modifying the Rust bindings directly, rebuild them in-place:
maturin develop --release -m pyDF/Cargo.toml
# Required build dependency for pyDF-data: HDF5 headers (e.g. ubuntu: libhdf5-dev)
maturin develop --release -m pyDF-data/Cargo.toml
# If you have troubles with hdf5 you may try to build and link hdf5 statically:
# (This is required on macOS/Homebrew when only hdf5 2.x is installed.)
maturin develop --release --features hdf5-static -m pyDF-data/Cargo.toml
```

### Development Setup

#### Pre-commit Hooks

This repository uses [pre-commit](https://pre-commit.com) to run code quality checks before commits. The hooks match our CI workflow, catching issues locally before push.

**Installation:**

```bash
# Install pre-commit
python3 -m pip install pre-commit

# Install git hooks
pre-commit install

# Run against all files (first-time setup)
pre-commit run --all-files
```

**Requirements:**

- Python 3.10+
- Rust toolchain with nightly (`rustup toolchain install nightly`)

**Hooks included:**

- **Utility hooks**: trailing whitespace, end-of-file, YAML/TOML/JSON validation
- **Python**: black (formatting), isort (import sorting), flake8 (linting)
- **Rust**: cargo fmt (formatting), cargo clippy (linting)

**Skipping hooks** (not recommended):

```bash
git commit --no-verify -m "message"
```

### Use DeepFilterNet from command line

To enhance noisy audio files using DeepFilterNet run
```bash
$ python DeepFilterNet/df/enhance.py --help
usage: enhance.py [-h] [--model-base-dir MODEL_BASE_DIR] [--pf] [--output-dir OUTPUT_DIR] [--log-level LOG_LEVEL] [--compensate-delay]
                  noisy_audio_files [noisy_audio_files ...]

positional arguments:
  noisy_audio_files     List of noise files to mix with the clean speech file.

optional arguments:
  -h, --help            show this help message and exit
  --model-base-dir MODEL_BASE_DIR, -m MODEL_BASE_DIR
                        Model directory containing checkpoints and config.
                        To load a pretrained model, you may just provide the model name, e.g. `DeepFilterNet`.
                        By default, the pretrained DeepFilterNet2 model is loaded.
  --pf                  Post-filter that slightly over-attenuates very noisy sections.
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory in which the enhanced audio files will be stored.
  --log-level LOG_LEVEL
                        Logger verbosity. Can be one of (debug, info, error, none)
  --compensate-delay, -D
                        Add some paddig to compensate the delay introduced by the real-time STFT/ISTFT implementation.

# Enhance audio with original DeepFilterNet
python DeepFilterNet/df/enhance.py -m DeepFilterNet path/to/noisy_audio.wav

# Enhance audio with DeepFilterNet2
python DeepFilterNet/df/enhance.py -m DeepFilterNet2 path/to/noisy_audio.wav
```

### Use DeepFilterNet within your Python script

```py
from df import enhance, init_df

model, df_state, _ = init_df()  # Load default model
enhanced_audio = enhance(model, df_state, noisy_audio)
```

See `scripts/external_usage.py` for a full example.

### Training

The entry point is `DeepFilterNet/df/train.py`. It expects a data directory containing HDF5 dataset
as well as a dataset configuration json file.

So, you first need to create your datasets in HDF5 format. Each dataset typically only
holds training, validation, or test set of noise, speech or RIRs.
```py
# Install additional dependencies for dataset creation
pip install h5py librosa soundfile
# Go to DeepFilterNet python package
cd path/to/DeepFilterNet/DeepFilterNet
# Prepare text file (e.g. called training_set.txt) containing paths to .wav files
#
# usage: prepare_data.py [-h] [--num_workers NUM_WORKERS] [--max_freq MAX_FREQ] [--sr SR] [--dtype DTYPE]
#                        [--codec CODEC] [--mono] [--compression COMPRESSION]
#                        type audio_files hdf5_db
#
# where:
#   type: One of `speech`, `noise`, `rir`
#   audio_files: Text file containing paths to audio files to include in the dataset
#   hdf5_db: Output HDF5 dataset.
python df/scripts/prepare_data.py --sr 48000 speech training_set.txt TRAIN_SET_SPEECH.hdf5
```
All datasets should be made available in one dataset folder for the train script.

The dataset configuration file should contain 3 entries: "train", "valid", "test". Each of those
contains a list of datasets (e.g. a speech, noise and a RIR dataset). You can use multiple speech
or noise dataset. Optionally, a sampling factor may be specified that can be used to over/under-sample
the dataset. Say, you have a specific dataset with transient noises and want to increase the amount
of non-stationary noises by oversampling. In most cases you want to set this factor to 1.

<details>
  <summary>Dataset config example:</summary>
<p>

`dataset.cfg`

```json
{
  "train": [
    [
      "TRAIN_SET_SPEECH.hdf5",
      1.0
    ],
    [
      "TRAIN_SET_NOISE.hdf5",
      1.0
    ],
    [
      "TRAIN_SET_RIR.hdf5",
      1.0
    ]
  ],
  "valid": [
    [
      "VALID_SET_SPEECH.hdf5",
      1.0
    ],
    [
      "VALID_SET_NOISE.hdf5",
      1.0
    ],
    [
      "VALID_SET_RIR.hdf5",
      1.0
    ]
  ],
  "test": [
    [
      "TEST_SET_SPEECH.hdf5",
      1.0
    ],
    [
      "TEST_SET_NOISE.hdf5",
      1.0
    ],
    [
      "TEST_SET_RIR.hdf5",
      1.0
    ]
  ]
}
```

</p>
</details>

Finally, start the training script. The training script may create a model `base_dir` if not
existing used for logging, some audio samples, model checkpoints, and config. If no config file is
found, it will create a default config. For the older INI-based training flow, point the script at
a model directory that already contains a `config.ini`.
```py
# usage: train.py [-h] [--debug] data_config_file data_dir base_dir
python df/train.py path/to/dataset.cfg path/to/data_dir/ path/to/base_dir/
```

## Apple Silicon Optimization

DeepFilterNet supports optimized whisper inference on Apple Silicon Macs (M1/M2/M3/M4) using the [MLX framework](https://github.com/ml-explore/mlx). This can provide **5-10x speedup** for ASR-based loss computation during training.

### Installation

```bash
# Install with MLX support (Apple Silicon only)
pip install deepfilternet[asr-mlx]

# Or install MLX dependencies separately
pip install mlx mlx-whisper
```

### Usage

The whisper backend is automatically selected based on your platform:
- **Apple Silicon**: Uses mlx-whisper (if installed) for optimal performance
- **CUDA/CPU**: Uses openai-whisper (PyTorch)

You can also explicitly select a backend:

```python
from df.whisper_adapter import get_whisper_backend

# Auto-detect (recommended)
backend = get_whisper_backend("base")

# Force PyTorch backend
backend = get_whisper_backend("base", backend="pytorch")

# Force MLX backend (Apple Silicon only)
backend = get_whisper_backend("base", backend="mlx")
```

For ASRLoss in training:

```python
from df.loss import ASRLoss

# Auto-detect optimal backend
loss_fn = ASRLoss(model="base", backend="auto")

# Explicit MLX for Apple Silicon
loss_fn = ASRLoss(model="base", backend="mlx")
```

### Requirements

- macOS 13.3+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- mlx >= 0.0.6
- mlx-whisper >= 0.4.0

## MLX DeepFilterNet4 (Native Apple Silicon)

In addition to the whisper optimization, DeepFilterNet includes a **full native MLX implementation** of the DfNet4 model for Apple Silicon. This provides maximum performance for both training and inference on M1/M2/M3/M4 chips.

### Features

- **100% MLX native**: Full model implementation using MLX framework
- **Feature parity**: All DfNet4 features including Mamba backbone, hybrid encoder, multi-resolution DF
- **Streaming API**: Frame-by-frame processing for real-time applications
- **Training support**: Full training loop with loss functions and checkpointing
- **PyTorch compatibility**: Load weights from PyTorch checkpoints

### Installation

```bash
# Install MLX dependencies
pip install mlx mlx-lm soundfile h5py

# Or use the requirements file
pip install -r requirements_mlx.txt
```

### Examples

See the `examples/` directory for complete usage examples:

| Example | Description |
|---------|-------------|
| [mlx_inference.py](examples/mlx_inference.py) | Basic single-file audio enhancement |
| [mlx_batch_enhance.py](examples/mlx_batch_enhance.py) | Process directory of audio files |
| [mlx_training.py](examples/mlx_training.py) | Training from scratch with checkpointing |
| [mlx_streaming.py](examples/mlx_streaming.py) | Real-time frame-by-frame processing |

### Quick Start

```python
from df_mlx.model import enhance, init_model, load_checkpoint

# Initialize model
model = init_model()

# Optionally load trained weights
load_checkpoint(model, "path/to/checkpoint.safetensors")

# Enhance audio (1D array of samples at 48kHz)
enhanced = enhance(model, noisy_audio)
```

### Streaming Processing

For real-time applications:

```python
from df_mlx.model import StreamingDfNet4, init_model

model = init_model()
streaming = StreamingDfNet4(model)
state = streaming.init_state(batch_size=1)

# Process frame by frame
for chunk in audio_chunks:
    enhanced_chunk, state = streaming.process_frame(chunk, state)
    # Use enhanced_chunk immediately
```

### Training

```python
from df_mlx.train import Trainer
from df_mlx.config import TrainConfig
from df_mlx.model import init_model

model = init_model()
config = TrainConfig(
    learning_rate=1e-4,
    warmup_steps=500,
    checkpoint_dir="./checkpoints",
)
trainer = Trainer(model, config)

# Training loop
for spec, feat_erb, feat_spec, target in data_loader:
    loss = trainer.train_step(spec, feat_erb, feat_spec, target)
```

### Performance

On Apple Silicon, the MLX implementation typically achieves:
- **Inference**: 0.02-0.05x RTF (20-50x faster than real-time)
- **Training**: Competitive with PyTorch on CUDA for small batches
- **Memory**: Efficient unified memory usage

See [benchmark_mlx_vs_pytorch.py](benchmark_mlx_vs_pytorch.py) for detailed benchmarks.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute to this project.

### Code Quality and Security Checks

This repository uses automated checks to maintain code quality and security:

#### Required Checks (must pass for PRs)
- **Python Linting**: Code must pass `black`, `isort`, and `flake8` checks
- **Rust Linting**: Code must pass `rustfmt` and `clippy` checks  
- **CodeQL Security Scan**: Automated security vulnerability detection for Python and Rust
- **Dependency Review**: Checks for vulnerable or incompatible dependencies in PRs

#### Automated Tools
- **Dependabot**: Automatically creates PRs for dependency updates
- **Stale Issue Management**: Helps keep issue tracker organized

#### Running Checks Locally

**Python:**
```bash
# Format code
black .
isort .

# Check linting
flake8
```

**Rust:**
```bash
# Format code
cargo fmt

# Check linting
cargo clippy --all-features -- -D warnings

# Run tests
cargo test --all-features
```

### Issue Templates

When reporting bugs or requesting features, please use the provided templates:
- [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yml)
- [Feature Request](.github/ISSUE_TEMPLATE/feature_request.yml)

### Security

For security vulnerabilities, please see our [Security Policy](SECURITY.md) and report issues privately through GitHub Security Advisories.
## Citation Guide

To reproduce any metrics, we recomend to use the python implementation via `pip install deepfilternet`.

If you use this framework, please cite: *DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering*
```bibtex
@inproceedings{schroeter2022deepfilternet,
  title={{DeepFilterNet}: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering},
  author = {Schröter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
  booktitle={ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
  organization={IEEE}
}
```

If you use the DeepFilterNet2 model, please cite: *DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio*

```bibtex
@inproceedings{schroeter2022deepfilternet2,
  title = {{DeepFilterNet2}: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio},
  author = {Schröter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
  booktitle={17th International Workshop on Acoustic Signal Enhancement (IWAENC 2022)},
  year = {2022},
}
```

If you use the DeepFilterNet3 model, please cite: *DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement*

```bibtex
@inproceedings{schroeter2023deepfilternet3,
  title = {{DeepFilterNet}: Perceptually Motivated Real-Time Speech Enhancement},
  author = {Schröter, Hendrik and Rosenkranz, Tobias and Escalante-B., Alberto N. and Maier, Andreas},
  booktitle={INTERSPEECH},
  year = {2023},
}
```

If you use the multi-frame beamforming algorithms. please cite *Deep Multi-Frame Filtering for Hearing Aids*

```bibtex
@inproceedings{schroeter2023deep_mf,
  title = {Deep Multi-Frame Filtering for Hearing Aids},
  author = {Schröter, Hendrik and Rosenkranz, Tobias and Escalante-B., Alberto N. and Maier, Andreas},
  booktitle={INTERSPEECH},
  year = {2023},
}
```

## License

DeepFilterNet is free and open source! All code in this repository is dual-licensed under either:

* MIT License ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option. This means you can select the license you prefer!

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
