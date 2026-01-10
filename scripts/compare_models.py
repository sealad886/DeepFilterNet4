#!/usr/bin/env python3
"""Compare all DeepFilterNet model versions on a directory of audio files.

This script processes audio files through:
- DeepFilterNet (v1)
- DeepFilterNet2
- DeepFilterNet3
- DeepFilterNet4 (MLX, using latest checkpoint)

Results are saved to separate output directories with timing and RTF metrics.

Usage:
    python compare_models.py /path/to/noisy/audio --output-dir /path/to/output
    python compare_models.py /path/to/audio --dfnet4-checkpoint /path/to/checkpoint.safetensors
    python compare_models.py /path/to/audio --dfnet4-checkpoint-dir /Users/andrew/DataDump/checkpoints
"""

import argparse
import glob
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add parent paths for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
DF_DIR = SCRIPT_DIR.parent / "DeepFilterNet"
sys.path.insert(0, str(DF_DIR))


@dataclass
class ModelResult:
    """Results from processing a single file with a model."""

    model_name: str
    input_file: str
    output_file: str
    duration_seconds: float
    processing_time: float
    rtf: float
    success: bool
    error: Optional[str] = None


@dataclass
class ComparisonResults:
    """Aggregated results from comparing all models."""

    input_dir: str
    output_dir: str
    models_tested: list = field(default_factory=list)
    results: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add_result(self, result: ModelResult):
        self.results.append(result)
        if result.model_name not in self.models_tested:
            self.models_tested.append(result.model_name)

    def compute_summary(self):
        """Compute summary statistics per model."""
        for model in self.models_tested:
            model_results = [r for r in self.results if r.model_name == model and r.success]
            if not model_results:
                continue

            total_audio = sum(r.duration_seconds for r in model_results)
            total_processing = sum(r.processing_time for r in model_results)
            avg_rtf = total_processing / total_audio if total_audio > 0 else 0

            self.summary[model] = {
                "files_processed": len(model_results),
                "total_audio_seconds": round(total_audio, 2),
                "total_processing_seconds": round(total_processing, 2),
                "average_rtf": round(avg_rtf, 4),
                "failures": len([r for r in self.results if r.model_name == model and not r.success]),
            }

    def to_dict(self):
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "models_tested": self.models_tested,
            "summary": self.summary,
            "results": [
                {
                    "model": r.model_name,
                    "input": r.input_file,
                    "output": r.output_file,
                    "duration_s": r.duration_seconds,
                    "processing_s": r.processing_time,
                    "rtf": r.rtf,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in a directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # Look for step_*.safetensors or epoch_*.safetensors
    patterns = ["step_*.safetensors", "epoch_*.safetensors"]
    checkpoints = []

    for pattern in patterns:
        checkpoints.extend(checkpoint_path.glob(pattern))

    if not checkpoints:
        # Try best.safetensors
        best = checkpoint_path / "best.safetensors"
        if best.exists():
            return str(best)
        return None

    # Extract step/epoch numbers and sort
    def extract_number(p: Path) -> int:
        name = p.stem
        # step_013000 or epoch_5
        parts = name.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return 0
        return 0

    checkpoints.sort(key=extract_number, reverse=True)
    return str(checkpoints[0])


def get_audio_files(input_path: str) -> list:
    """Get list of audio files from path (file or directory)."""
    path = Path(input_path)
    extensions = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}

    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        files = []
        for ext in extensions:
            files.extend(glob.glob(str(path / f"*{ext}")))
            files.extend(glob.glob(str(path / f"*{ext.upper()}")))
        return sorted(files)
    else:
        return []


def get_audio_duration(audio_path: str, target_sr: int = 48000) -> float:
    """Get audio duration in seconds."""
    import soundfile as sf

    info = sf.info(audio_path)
    return info.duration


def convert_audio_to_wav(
    input_path: str,
    output_path: str,
    target_sr: int = 48000,
    mono: bool = True,
) -> bool:
    """Convert any audio file to WAV format suitable for DeepFilterNet.

    Handles:
    - Format conversion (mp3, flac, ogg, opus, m4a -> wav)
    - Resampling to target sample rate
    - Stereo to mono conversion
    - Normalization to float32

    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file
        target_sr: Target sample rate (default: 48000)
        mono: Convert to mono (default: True)

    Returns:
        True if conversion succeeded, False otherwise
    """
    import numpy as np

    try:
        # Try soundfile first (handles wav, flac, ogg)
        import soundfile as sf

        try:
            audio, sr = sf.read(input_path, dtype="float32")
        except Exception:
            # Fall back to pydub for mp3, m4a, etc.
            try:
                from pydub import AudioSegment

                sound = AudioSegment.from_file(input_path)
                sr = sound.frame_rate
                samples = np.array(sound.get_array_of_samples(), dtype=np.float32)

                # Normalize to [-1, 1]
                max_val = float(2 ** (sound.sample_width * 8 - 1))
                samples = samples / max_val

                # Handle stereo
                if sound.channels == 2:
                    audio = samples.reshape(-1, 2)
                else:
                    audio = samples
            except ImportError:
                # Try librosa as last resort
                import librosa

                audio, sr = librosa.load(input_path, sr=None, mono=False)
                if audio.ndim == 2:
                    audio = audio.T  # librosa returns (channels, samples)

        # Convert to mono if needed
        if mono and audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Ensure 1D for mono
        if audio.ndim > 1 and audio.shape[1] == 1:
            audio = audio.squeeze()

        # Resample if needed
        if sr != target_sr:
            from scipy import signal

            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples).astype(np.float32)

        # Normalize to prevent clipping
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val * 0.95
        elif max_val < 0.01:
            # Very quiet audio, normalize up
            audio = audio / (max_val + 1e-8) * 0.5

        # Ensure float32
        audio = audio.astype(np.float32)

        # Save as WAV
        sf.write(output_path, audio, target_sr, subtype="FLOAT")
        return True

    except Exception as e:
        print(f"    WARNING: Audio conversion failed for {input_path}: {e}")
        return False


def prepare_audio_file(input_path: str, temp_dir: Path, target_sr: int = 48000) -> Optional[str]:
    """Prepare an audio file for processing, converting if necessary.

    Args:
        input_path: Path to input audio file
        temp_dir: Directory for temporary converted files
        target_sr: Target sample rate

    Returns:
        Path to prepared audio file (may be original or converted), or None if failed
    """
    input_path = Path(input_path)

    # Check if already a compatible WAV
    if input_path.suffix.lower() == ".wav":
        try:
            import soundfile as sf

            info = sf.info(str(input_path))
            # Check if already 48kHz mono
            if info.samplerate == target_sr and info.channels == 1:
                return str(input_path)
        except Exception:
            pass

    # Need to convert
    temp_dir.mkdir(parents=True, exist_ok=True)
    converted_path = temp_dir / f"{input_path.stem}_converted.wav"

    if convert_audio_to_wav(str(input_path), str(converted_path), target_sr, mono=True):
        return str(converted_path)

    return None


class DeepFilterNetV1V2V3Enhancer:
    """Enhancer for DeepFilterNet v1, v2, v3 (PyTorch)."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.df_state = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        from df.enhance import init_df

        self.model, self.df_state, _, _ = init_df(
            model_base_dir=self.model_name,
            log_level="ERROR",
            config_allow_defaults=True,
            device=self.device,
        )
        self._initialized = True

    def enhance(self, input_path: str, output_path: str) -> ModelResult:
        """Enhance a single audio file."""
        import torch

        from df.enhance import enhance
        from df.io import load_audio, save_audio
        from df.model import ModelParams

        self.initialize()

        duration = get_audio_duration(input_path)
        sr = ModelParams().sr

        try:
            audio, _ = load_audio(input_path, sr, "cpu")

            start_time = time.time()
            with torch.no_grad():
                enhanced = enhance(self.model, self.df_state, audio, device=self.device)
            processing_time = time.time() - start_time

            save_audio(input_path, enhanced, sr, output_dir=str(Path(output_path).parent), suffix=None, log=False)
            # Rename to exact output path
            base = Path(input_path).stem
            actual_output = Path(output_path).parent / f"{base}.wav"
            if actual_output.exists() and str(actual_output) != output_path:
                actual_output.rename(output_path)

            rtf = processing_time / duration if duration > 0 else 0

            return ModelResult(
                model_name=self.model_name,
                input_file=input_path,
                output_file=output_path,
                duration_seconds=duration,
                processing_time=processing_time,
                rtf=rtf,
                success=True,
            )

        except Exception as e:
            return ModelResult(
                model_name=self.model_name,
                input_file=input_path,
                output_file=output_path,
                duration_seconds=duration,
                processing_time=0,
                rtf=0,
                success=False,
                error=str(e),
            )


class DeepFilterNet4MLXEnhancer:
    """Enhancer for DeepFilterNet4 (MLX)."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._initialized = False

    def _detect_backbone_type(self, checkpoint_path: str) -> str:
        """Detect backbone type from checkpoint weights.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Backbone type: 'attention', 'gru', or 'mamba'
        """
        import mlx.core as mx

        weights = mx.load(checkpoint_path)
        weight_keys = set(weights.keys())

        # Check for attention-specific keys
        if any("attention_layers" in k for k in weight_keys):
            return "attention"
        # Check for GRU-specific keys
        elif any("gru_layers" in k or "gru.weight" in k for k in weight_keys):
            return "gru"
        # Default to mamba
        else:
            return "mamba"

    def initialize(self):
        if self._initialized:
            return

        from df_mlx.config import BackboneParams, ModelParams4
        from df_mlx.model import init_model
        from df_mlx.train import load_checkpoint

        # Detect backbone type from checkpoint
        backbone_type = "gru"  # Default
        if self.checkpoint_path:
            backbone_type = self._detect_backbone_type(self.checkpoint_path)
            print(f"  [DFNet4-MLX] Detected backbone: {backbone_type}")

        # Create config with correct backbone type
        config = ModelParams4()
        config.backbone = BackboneParams(backbone_type=backbone_type)

        self.model = init_model(config=config)

        if self.checkpoint_path:
            load_checkpoint(self.model, self.checkpoint_path)
            print(f"  [DFNet4-MLX] Loaded checkpoint: {Path(self.checkpoint_path).name}")
        else:
            print("  [DFNet4-MLX] WARNING: No checkpoint provided, using random weights!")

        self._initialized = True

    def enhance(self, input_path: str, output_path: str) -> ModelResult:
        """Enhance a single audio file."""
        import mlx.core as mx
        import numpy as np
        import soundfile as sf

        self.initialize()

        duration = get_audio_duration(input_path)
        target_sr = 48000

        try:
            # Load audio
            audio, sr = sf.read(input_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample if needed
            if sr != target_sr:
                from scipy import signal

                num_samples = int(len(audio) * target_sr / sr)
                audio = signal.resample(audio, num_samples).astype(np.float32)

            audio_mx = mx.array(audio)

            # Enhance
            start_time = time.time()
            enhanced = self.model.enhance(audio_mx)
            mx.eval(enhanced)
            processing_time = time.time() - start_time

            # Save
            enhanced_np = np.array(enhanced, dtype=np.float32)
            max_val = np.abs(enhanced_np).max()
            if max_val > 1.0:
                enhanced_np = enhanced_np / max_val * 0.95

            sf.write(output_path, enhanced_np, target_sr)

            rtf = processing_time / duration if duration > 0 else 0

            return ModelResult(
                model_name="DeepFilterNet4-MLX",
                input_file=input_path,
                output_file=output_path,
                duration_seconds=duration,
                processing_time=processing_time,
                rtf=rtf,
                success=True,
            )

        except Exception as e:
            import traceback

            return ModelResult(
                model_name="DeepFilterNet4-MLX",
                input_file=input_path,
                output_file=output_path,
                duration_seconds=duration,
                processing_time=0,
                rtf=0,
                success=False,
                error=f"{str(e)}\n{traceback.format_exc()}",
            )


def run_comparison(
    input_path: str,
    output_dir: str,
    models: list,
    dfnet4_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
) -> ComparisonResults:
    """Run comparison across all specified models."""
    audio_files = get_audio_files(input_path)
    if not audio_files:
        print(f"No audio files found in: {input_path}")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files")
    print(f"Models to test: {models}")
    print(f"Output directory: {output_dir}")

    # Create temp directory for converted audio
    temp_dir = Path(tempfile.mkdtemp(prefix="dfnet_compare_"))
    print(f"Temp directory for conversions: {temp_dir}")
    print()

    # Pre-convert audio files that need it
    print("Preparing audio files...")
    prepared_files = []
    for audio_file in audio_files:
        prepared = prepare_audio_file(audio_file, temp_dir)
        if prepared:
            prepared_files.append((audio_file, prepared))
            if prepared != audio_file:
                print(f"  ✓ Converted: {Path(audio_file).name}")
        else:
            print(f"  ✗ Failed to prepare: {Path(audio_file).name}")

    if not prepared_files:
        print("No audio files could be prepared!")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    print(f"\nPrepared {len(prepared_files)}/{len(audio_files)} files")
    print()

    results = ComparisonResults(input_dir=input_path, output_dir=output_dir)

    # Initialize enhancers
    enhancers = {}

    for model in models:
        if model == "DeepFilterNet4-MLX":
            enhancers[model] = DeepFilterNet4MLXEnhancer(dfnet4_checkpoint)
        else:
            enhancers[model] = DeepFilterNetV1V2V3Enhancer(model, device)

    # Process each model
    for model_name, enhancer in enhancers.items():
        print(f"\n{'=' * 60}")
        print(f"Processing with: {model_name}")
        print("=" * 60)

        model_output_dir = Path(output_dir) / model_name.replace(" ", "_")
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for i, (original_file, prepared_file) in enumerate(prepared_files):
            filename = Path(original_file).stem
            output_file = model_output_dir / f"{filename}_enhanced.wav"

            print(f"  [{i + 1}/{len(prepared_files)}] {Path(original_file).name}...", end=" ", flush=True)

            result = enhancer.enhance(prepared_file, str(output_file))
            # Store original file path in result for clarity
            result.input_file = original_file
            results.add_result(result)

            if result.success:
                print(f"✓ RTF={result.rtf:.3f}")
            else:
                print(f"✗ Error: {result.error[:50]}...")

    # Cleanup temp directory
    print("\nCleaning up temp directory...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    results.compute_summary()
    return results


def print_summary(results: ComparisonResults):
    """Print a formatted summary table."""
    print("\n")
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    if not results.summary:
        print("No results to summarize.")
        return

    # Header
    print(f"{'Model':<25} {'Files':>8} {'Audio(s)':>10} {'Time(s)':>10} {'Avg RTF':>10} {'Failures':>10}")
    print("-" * 80)

    for model, stats in results.summary.items():
        print(
            f"{model:<25} "
            f"{stats['files_processed']:>8} "
            f"{stats['total_audio_seconds']:>10.1f} "
            f"{stats['total_processing_seconds']:>10.2f} "
            f"{stats['average_rtf']:>10.4f} "
            f"{stats['failures']:>10}"
        )

    print("-" * 80)

    # Speed comparison
    if len(results.summary) > 1:
        print("\nRelative Speed (lower RTF = faster):")
        rtfs = [(m, s["average_rtf"]) for m, s in results.summary.items() if s["average_rtf"] > 0]
        if rtfs:
            rtfs.sort(key=lambda x: x[1])
            fastest = rtfs[0][1]
            for model, rtf in rtfs:
                speedup = rtf / fastest if fastest > 0 else 1
                bar = "█" * int(20 / speedup) if speedup > 0 else ""
                print(f"  {model:<25} {bar:<20} {speedup:.2f}x (RTF: {rtf:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare all DeepFilterNet model versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare all models on a directory
    python compare_models.py /path/to/noisy/audio -o /path/to/output

    # Use specific DFNet4 checkpoint
    python compare_models.py /path/to/audio --dfnet4-checkpoint /path/to/step_017000.safetensors

    # Auto-find latest checkpoint from directory
    python compare_models.py /path/to/audio --dfnet4-checkpoint-dir /Users/andrew/DataDump/checkpoints

    # Test only specific models
    python compare_models.py /path/to/audio --models DeepFilterNet3 DeepFilterNet4-MLX
        """,
    )

    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument(
        "--output-dir", "-o", default="./comparison_output", help="Output directory (default: ./comparison_output)"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=["DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3", "DeepFilterNet4-MLX"],
        help="Models to compare (default: all)",
    )
    parser.add_argument("--dfnet4-checkpoint", help="Path to DFNet4 MLX checkpoint (.safetensors)")
    parser.add_argument(
        "--dfnet4-checkpoint-dir",
        default="/Users/andrew/DataDump/checkpoints",
        help="Directory to search for latest DFNet4 checkpoint",
    )
    parser.add_argument("--device", "-D", default=None, help="Device for PyTorch models (cpu, cuda, mps)")
    parser.add_argument("--save-results", "-s", default="comparison_results.json", help="Save results to JSON file")

    args = parser.parse_args()

    # Resolve DFNet4 checkpoint
    dfnet4_checkpoint = args.dfnet4_checkpoint
    if not dfnet4_checkpoint and "DeepFilterNet4-MLX" in args.models:
        dfnet4_checkpoint = find_latest_checkpoint(args.dfnet4_checkpoint_dir)
        if dfnet4_checkpoint:
            print(f"Auto-detected DFNet4 checkpoint: {dfnet4_checkpoint}")
        else:
            print(f"WARNING: No DFNet4 checkpoint found in {args.dfnet4_checkpoint_dir}")
            print("DFNet4-MLX will use random weights (results will be meaningless)")
            response = input("Continue anyway? [y/N] ")
            if response.lower() != "y":
                sys.exit(0)

    # Run comparison
    results = run_comparison(
        input_path=args.input,
        output_dir=args.output_dir,
        models=args.models,
        dfnet4_checkpoint=dfnet4_checkpoint,
        device=args.device,
    )

    # Print summary
    print_summary(results)

    # Save results
    if args.save_results:
        results_path = Path(args.output_dir) / args.save_results
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
