# DeepFilterNet
A Low Complexity Speech Enhancement Framework for Full-Band Audio (48kHz) using on Deep Filtering.

## Fork Notice

This is a modernized fork of [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) with the following improvements:

- **Python 3.9–3.13 support** — Updated PyO3 bindings for latest Python versions
- **TorchAudio 2.9 support** — TorchCodec integration for deprecated `torchaudio.info()` API
- **PyTorch 2.5+ compatibility** — Fixed `weights_only` deprecation warnings
- **Apple Silicon MPS support** — Automatic Metal Performance Shaders detection
- **Device selection CLI** — New `--device` argument for explicit compute device control
- **CUDA tensor handling fix** — Proper `.detach().cpu()` before `.numpy()` conversion

### Version Compatibility

| Component | Minimum | Recommended | Maximum |
|-----------|---------|-------------|---------|
| Python | 3.9 | 3.11+ | 3.13 |
| PyTorch | 2.5 | 2.5+ | <3.0 |
| TorchAudio | 2.5 | 2.5+ | <3.0 |
| TorchCodec | 0.1 | Latest | (for TorchAudio 2.9+) |

### TorchAudio 2.9+ Users

TorchAudio 2.9 removed the `torchaudio.info()` API. This fork automatically uses TorchCodec as a fallback when available:

```bash
# Install TorchCodec for TorchAudio 2.9+
pip install torchcodec
```

TorchCodec requires FFmpeg libraries:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg libavcodec-dev libavformat-dev

# macOS (Homebrew)
brew install ffmpeg
```

### Apple Silicon (MPS) Support

DeepFilterNet supports Apple Silicon GPUs via PyTorch's MPS backend for accelerated inference.

**Requirements:**
- macOS 14 (Sonoma) or later for full support
- PyTorch 2.5.1+ with MPS enabled
- Apple Silicon Mac (M1/M2/M3/M4)

**Usage:**
```bash
# Explicit MPS device
deep-filter --device mps input.wav

# Auto-detection (uses MPS if available)
deep-filter input.wav

# Python API
from df import enhance, init_df
model, df_state, _ = init_df(device="mps")
enhanced = enhance(model, df_state, noisy_audio)
```

**Note:** Complex tensor operations require macOS 14+. On older versions, set:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### MPS Operation Compatibility

| Operation | macOS 12-13 | macOS 14+ |
|-----------|-------------|-----------|
| Model inference | ✅ (real mode) | ✅ |
| Complex filtering | ❌ | ✅ |
| STOI evaluation | ✅ | ✅ |
| Training | ⚠️ Limited | ✅ |

#### Performance Expectations

- **Batch processing:** Significant speedup over CPU (2-4x typical)
- **Real-time streaming:** Marginal improvement due to CPU↔MPS transfer overhead

The STFT/ISTFT operations run on CPU (libDF/Rust) for optimal real-time latency.
GPU acceleration benefits the neural network inference portion.

#### MPS Troubleshooting

**"ComplexFloat not supported" error:**
- Requires macOS 14+, or use `--device cpu`
- Or set `PYTORCH_ENABLE_MPS_FALLBACK=1`

**Slower than expected:**
- Real-time streaming has CPU↔GPU overhead
- Batch processing benefits most from MPS

**"MPS backend not available":**
- Ensure PyTorch installed with MPS support
- Requires Apple Silicon Mac (M1/M2/M3/M4)
- Check with: `python -c "import torch; print(torch.backends.mps.is_available())"`

---

## DeepFilterNet4

DeepFilterNet4 (DFNet4) is the next generation architecture that combines state-space models (Mamba) with advanced training techniques for improved speech enhancement quality and efficiency.

### Key Features

| Feature | Description |
|---------|-------------|
| **Mamba Backbone** | State-space model for efficient long-range temporal modeling with O(n) complexity |
| **Hybrid Encoder** | Optional time-domain and phase branches for multi-domain feature fusion |
| **Multi-Resolution DF** | Deep filtering at multiple frequency resolutions for better detail preservation |
| **Adaptive Filter Order** | Signal-complexity-based filter order selection for efficiency |
| **GAN Training** | Discriminator and feature matching losses for perceptual quality |
| **Quantization-Aware** | INT8 quantization with negligible quality degradation |
| **Knowledge Distillation** | DfNet4Lite variant (~1.3M params) for embedded deployment |

### Model Variants

```text
DfNet4        ~2.6M params  Full-featured model with all components
DfNet4Lite    ~1.3M params  Lightweight variant for edge deployment
```

### Quick Start

```python
from df.deepfilternet4 import DfNet4, DfNet4Config
import torch

# Create model with default config
config = DfNet4Config()
model = DfNet4(config)

# Or use DfNet4Lite for embedded deployment
from df.deepfilternet4 import DfNet4Lite
lite_model = DfNet4Lite(config)

# Forward pass
batch_size, n_frames = 1, 100
erb = torch.randn(batch_size, 1, config.nb_erb, n_frames)       # ERB features
spec = torch.randn(batch_size, 1, config.nb_df, n_frames, 2)    # Complex spectrum
out_erb, out_df = model(erb, spec)
```

### Architecture Components

#### 1. Mamba Encoder

The Mamba-based encoder replaces traditional GRU/LSTM with state-space models for better long-range temporal modeling:

```python
config = DfNet4Config(
    mamba_d_state=64,      # State space dimension
    mamba_d_conv=4,        # Local convolution width
    mamba_expand=2,        # Expansion factor
    num_mamba_layers=4,    # Number of Mamba blocks
)
```

#### 2. Hybrid Encoder

Optional multi-domain feature fusion:

```python
config = DfNet4Config(
    hybrid_encoder=True,       # Enable hybrid encoder
    use_time_domain_enc=True,  # Time-domain branch
    use_phase_enc=True,        # Phase information branch
)
```

#### 3. Multi-Resolution Deep Filtering

Process different frequency bands at different resolutions:

```python
config = DfNet4Config(
    multi_res_df=True,     # Enable multi-resolution DF
    df_resolutions=[2, 4], # Resolution factors for different bands
)
```

#### 4. Adaptive Filter Order

Dynamically select filter order based on signal complexity:

```python
config = DfNet4Config(
    adaptive_df_order=True,     # Enable adaptive order
    df_order_range=(3, 8),      # Min and max filter orders
)
```

### Training

#### Basic Training

```bash
python df/train.py --model-type dfnet4 path/to/dataset.cfg path/to/data/ path/to/output/
```

#### GAN Training

Enable adversarial training for improved perceptual quality:

```python
from df.deepfilternet4 import DfNet4, DfNet4Discriminator

# Generator
model = DfNet4(config)

# Discriminator
discriminator = DfNet4Discriminator(
    in_channels=1,
    hidden_channels=[64, 128, 256, 512],
    num_scales=3,
    num_layers_per_scale=3,
)
```

#### Quantization-Aware Training (QAT)

Train models for INT8 deployment:

```python
from df.deepfilternet4 import DfNet4, prepare_qat

model = DfNet4(config)
qat_model = prepare_qat(model)

# Train as usual, then convert
quantized_model = torch.quantization.convert(qat_model)
```

#### Knowledge Distillation

Train a lightweight student model:

```python
from df.deepfilternet4 import DfNet4, DfNet4Lite, DistillationLoss

teacher = DfNet4(config).eval()
student = DfNet4Lite(config)

loss_fn = DistillationLoss(
    teacher_model=teacher,
    alpha_kd=0.5,      # Distillation weight
    temperature=4.0,   # Softmax temperature
)
```

### ONNX Export

Export DFNet4 for deployment:

```bash
# Full model export
python df/scripts/export_onnx.py output_dir/ \
    --checkpoint path/to/checkpoint.pt \
    --full-model \
    --simplify

# Component-wise export (encoder, ERB decoder, DF decoder)
python df/scripts/export_onnx.py output_dir/ \
    --checkpoint path/to/checkpoint.pt

# Create deployment archive
python df/scripts/export_onnx.py output_dir/ \
    --checkpoint path/to/checkpoint.pt \
    --archive
```

### Benchmarking

Compare performance with previous versions:

```bash
# Run benchmarks on VoiceBank-DEMAND
python df/scripts/benchmark_dfnet4.py \
    --checkpoint path/to/dfnet4.pt \
    --test-dir path/to/voicebank/ \
    --metrics pesq stoi dnsmos \
    --compare-dfnet3

# RTF-only benchmark
python df/scripts/benchmark_dfnet4.py \
    --checkpoint path/to/dfnet4.pt \
    --rtf-only \
    --device cuda
```

### Configuration Reference

Full configuration options:

```python
DfNet4Config(
    # ERB/Spectral dimensions
    nb_erb=32,             # Number of ERB bands
    nb_df=96,              # DF frequency bins
    df_order=5,            # Deep filter order
    df_lookahead=2,        # Lookahead frames

    # Mamba parameters
    mamba_d_model=256,     # Model dimension
    mamba_d_state=64,      # State space dimension
    mamba_d_conv=4,        # Convolution width
    mamba_expand=2,        # Expansion factor
    num_mamba_layers=4,    # Number of layers

    # Optional features
    hybrid_encoder=False,       # Multi-domain encoder
    use_time_domain_enc=False,  # Time-domain branch
    use_phase_enc=False,        # Phase branch
    multi_res_df=False,         # Multi-resolution DF
    adaptive_df_order=False,    # Adaptive filter order

    # Decoder
    hidden_dim=256,        # Hidden dimension
    num_hidden_layers=2,   # Decoder layers
)
```

### Migration from DFNet3

See the [Migration Guide](docs/MIGRATION.md) for detailed instructions on migrating from DeepFilterNet3 to DeepFilterNet4.

Key changes:
- Configuration format: INI → Python dataclass
- Model class: `ModelParams` → `DfNet4Config`
- Encoder: GRU → Mamba SSM
- New optional components: hybrid encoder, multi-resolution DF, adaptive order

---

![deepfilternet3](https://user-images.githubusercontent.com/16517898/225623209-a54fea75-ca00-404c-a394-c91d2d1146d2.svg)

For PipeWire integration as a virtual noise suppression microphone look [here](https://github.com/Rikorose/DeepFilterNet/blob/main/ladspa/README.md).

### Demo

https://github.com/Rikorose/DeepFilterNet/assets/16517898/79679fd7-de73-4c22-948c-891927c7d2ca

To run the demo (linux only) use:
```bash
cargo +nightly run -p df-demo --features ui --bin df-demo --release
```

### News

- **DeepFilterNet4** - Next generation architecture with improved quality and efficiency:
  - Mamba backbone (state-space models) for efficient long-range temporal modeling
  - Hybrid encoder with optional time-domain and phase branches
  - Multi-resolution deep filtering for better frequency detail preservation
  - Adaptive filter order based on signal complexity
  - GAN training support with discriminator and feature matching losses
  - Quantization-aware training (QAT) for INT8 deployment
  - Knowledge distillation for lightweight DfNet4Lite variant (~1.3M params)
  - See [DeepFilterNet4 Documentation](DeepFilterNet/README.md#deepfilternet4) for details

- New DeepFilterNet Demo: *DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement*
  - Paper: https://arxiv.org/abs/2305.08227
  - Video: https://youtu.be/EO7n96YwnyE

- New Multi-Frame Filtering Paper: *Deep Multi-Frame Filtering for Hearing Aids*
  - Paper: https://arxiv.org/abs/2305.08225

- Real-time version and a LADSPA plugin
  - [Pre-compiled binary](#deep-filter), no python dependencies. Usage: `deep-filter audio-file.wav`
  - [LADSPA plugin](ladspa/) with pipewire filter-chain integration for real-time noise reduction on your mic.

- DeepFilterNet2 Paper: *DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio*
  - Paper: https://arxiv.org/abs/2205.05474
  - Samples: https://rikorose.github.io/DeepFilterNet2-Samples/
  - Demo: https://huggingface.co/spaces/hshr/DeepFilterNet2

- Original DeepFilterNet Paper: *DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering*
  - Paper: https://arxiv.org/abs/2110.05588
  - Samples: https://rikorose.github.io/DeepFilterNet-Samples/
  - Demo: https://huggingface.co/spaces/hshr/DeepFilterNet
  - Video Lecture: https://youtu.be/it90gBqkY6k

## Usage

### deep-filter

Download a pre-compiled deep-filter binary from the [release page](https://github.com/Rikorose/DeepFilterNet/releases/).
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

Install cargo via [rustup](https://rustup.rs/). Usage of a `conda` or `virtualenv` recommended.
Please read the comments and only execute the commands that you need.

Installation of python dependencies and libDF:
```bash
cd path/to/DeepFilterNet/  # cd into repository
# Recommended: Install or activate a python env
# Mandatory: Install cpu/cuda pytorch (>=1.8) dependency from pytorch.org, e.g.:
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Install build dependencies used to compile libdf and DeepFilterNet python wheels
pip install maturin poetry

#  Install remaining DeepFilterNet python dependencies
# *Option A:* Install DeepFilterNet python wheel globally within your environment. Do this if you want use
# this repos as is, and don't want to develop within this repository.
poetry -C DeepFilterNet install -E train -E eval
# *Option B:* If you want to develop within this repo, install only dependencies and work with the repository version
poetry -C DeepFilterNet install -E train -E eval --no-root
export PYTHONPATH=$PWD/DeepFilterNet # And set the python path correctly

# Build and install libdf python package required for enhance.py
maturin develop --release -m pyDF/Cargo.toml
# *Optional*: Install libdfdata python package with dataset and dataloading functionality for training
# Required build dependency: HDF5 headers (e.g. ubuntu: libhdf5-dev)
maturin develop --release -m pyDF-data/Cargo.toml
# If you have troubles with hdf5 you may try to build and link hdf5 statically:
maturin develop --release --features hdf5-static -m pyDF-data/Cargo.toml
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

See [here](https://github.com/Rikorose/DeepFilterNet/blob/main/scripts/external_usage.py) for a full example.

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
found, it will create a default config. See
[DeepFilterNet/pretrained_models/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/pretrained_models/DeepFilterNet/config.ini)
for a config file.
```py
# usage: train.py [-h] [--debug] data_config_file data_dir base_dir
python df/train.py path/to/dataset.cfg path/to/data_dir/ path/to/base_dir/
```

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
