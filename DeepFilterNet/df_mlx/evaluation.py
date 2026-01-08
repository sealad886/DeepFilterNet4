"""Evaluation utilities for MLX-based training and inference.

Provides metrics for assessing speech enhancement quality including:
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- STOI (Short-Time Objective Intelligibility)
- PESQ (Perceptual Evaluation of Speech Quality)
- DNSMOS (Deep Noise Suppression Mean Opinion Score)
- Composite metrics (CSIG, CBAK, COVL, SSNR)

For final evaluation/benchmarking, use pystoi and pypesq packages.
This module provides fast MLX-native implementations for training validation.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import mlx.core as mx
import numpy as np
from loguru import logger

# ============================================================================
# Core Metrics
# ============================================================================


def si_sdr(
    reference: mx.array,
    estimate: mx.array,
    eps: float = 1e-8,
) -> mx.array:
    """Compute Scale-Invariant Signal-to-Distortion Ratio.

    Args:
        reference: Clean reference signal (batch, samples) or (samples,)
        estimate: Enhanced/degraded signal (same shape)
        eps: Small constant for numerical stability

    Returns:
        SI-SDR in dB per batch item (higher is better)
    """
    # Handle dimensions
    if reference.ndim == 1:
        reference = mx.expand_dims(reference, axis=0)
        estimate = mx.expand_dims(estimate, axis=0)

    # Ensure same length
    min_len = min(reference.shape[1], estimate.shape[1])
    reference = reference[:, :min_len]
    estimate = estimate[:, :min_len]

    # Zero-mean
    reference = reference - mx.mean(reference, axis=-1, keepdims=True)
    estimate = estimate - mx.mean(estimate, axis=-1, keepdims=True)

    # Compute scale factor
    dot = mx.sum(estimate * reference, axis=-1, keepdims=True)
    s_ref = mx.sum(reference**2, axis=-1, keepdims=True) + eps
    scale = dot / s_ref

    # Target and distortion
    target = scale * reference
    distortion = estimate - target

    # SI-SDR in dB
    target_energy = mx.sum(target**2, axis=-1) + eps
    distortion_energy = mx.sum(distortion**2, axis=-1) + eps
    si_sdr_db = 10 * mx.log10(target_energy / distortion_energy)

    return mx.squeeze(si_sdr_db)


def snr(
    reference: mx.array,
    estimate: mx.array,
    eps: float = 1e-8,
) -> mx.array:
    """Compute Signal-to-Noise Ratio.

    Args:
        reference: Clean reference signal
        estimate: Enhanced/degraded signal
        eps: Small constant for numerical stability

    Returns:
        SNR in dB (higher is better)
    """
    if reference.ndim == 1:
        reference = mx.expand_dims(reference, axis=0)
        estimate = mx.expand_dims(estimate, axis=0)

    min_len = min(reference.shape[1], estimate.shape[1])
    reference = reference[:, :min_len]
    estimate = estimate[:, :min_len]

    noise = reference - estimate
    ref_energy = mx.sum(reference**2, axis=-1) + eps
    noise_energy = mx.sum(noise**2, axis=-1) + eps

    snr_db = 10 * mx.log10(ref_energy / noise_energy)
    return mx.squeeze(snr_db)


def segmental_snr(
    reference: mx.array,
    estimate: mx.array,
    frame_length: int = 512,
    hop_length: int = 256,
    eps: float = 1e-8,
) -> mx.array:
    """Compute Segmental Signal-to-Noise Ratio.

    Args:
        reference: Clean reference signal
        estimate: Enhanced/degraded signal
        frame_length: Analysis frame length
        hop_length: Hop size between frames
        eps: Small constant for numerical stability

    Returns:
        Segmental SNR in dB (higher is better)
    """
    if reference.ndim == 1:
        reference = mx.expand_dims(reference, axis=0)
        estimate = mx.expand_dims(estimate, axis=0)

    batch = reference.shape[0]
    min_len = min(reference.shape[1], estimate.shape[1])
    reference = reference[:, :min_len]
    estimate = estimate[:, :min_len]

    results = []

    for b in range(batch):
        ref_b = reference[b]
        est_b = estimate[b]
        noise_b = ref_b - est_b

        # Frame the signals
        n_frames = (min_len - frame_length) // hop_length + 1
        frame_snrs = []

        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            ref_frame = ref_b[start:end]
            noise_frame = noise_b[start:end]

            ref_energy = mx.sum(ref_frame**2)
            noise_energy = mx.sum(noise_frame**2) + eps

            # Only include frames with sufficient energy
            if ref_energy > eps:
                frame_snr = 10 * mx.log10(ref_energy / noise_energy)
                # Clip to reasonable range
                frame_snr = mx.clip(frame_snr, -10, 35)
                frame_snrs.append(frame_snr)

        if frame_snrs:
            results.append(mx.mean(mx.stack(frame_snrs)))
        else:
            results.append(mx.array(0.0))

    return mx.stack(results)


# ============================================================================
# Metric Data Structures
# ============================================================================


@dataclass
class MetricResult:
    """Result from a metric computation."""

    name: str
    enhanced: float
    noisy: Optional[float] = None
    filename: Optional[str] = None


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""

    metrics: Dict[str, List[MetricResult]] = field(default_factory=dict)

    def add(self, result: MetricResult):
        """Add a metric result."""
        if result.name not in self.metrics:
            self.metrics[result.name] = []
        self.metrics[result.name].append(result)

    def mean(self) -> Dict[str, float]:
        """Compute mean values for all metrics."""
        out = {}
        for name, results in self.metrics.items():
            enh_values = [r.enhanced for r in results]
            out[f"Enhanced {name}"] = np.mean(enh_values)

            noisy_values = [r.noisy for r in results if r.noisy is not None]
            if noisy_values:
                out[f"Noisy {name}"] = np.mean(noisy_values)

        return out

    def to_csv_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to CSV-friendly dict: {filename: {metric_name: value}}."""
        out = defaultdict(dict)
        for name, results in self.metrics.items():
            for r in results:
                fn = r.filename or ""
                out[fn][name] = r.enhanced
        return dict(out)


# ============================================================================
# Metric Classes
# ============================================================================


class Metric:
    """Base class for evaluation metrics."""

    def __init__(
        self,
        name: str,
        source_sr: Optional[int] = None,
        target_sr: Optional[int] = None,
    ):
        self.name = name
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.results: List[MetricResult] = []

    def resample_if_needed(self, x: np.ndarray) -> np.ndarray:
        """Resample signal if source/target sample rates differ."""
        if self.source_sr and self.target_sr and self.source_sr != self.target_sr:
            try:
                import librosa

                return librosa.resample(
                    x,
                    orig_sr=self.source_sr,
                    target_sr=self.target_sr,
                )
            except ImportError:
                # Simple linear interpolation fallback
                ratio = self.target_sr / self.source_sr
                new_len = int(len(x) * ratio)
                indices = np.linspace(0, len(x) - 1, new_len)
                return np.interp(indices, np.arange(len(x)), x)
        return x

    def compute(self, clean: np.ndarray, degraded: np.ndarray) -> float:
        """Compute the metric. Override in subclasses."""
        raise NotImplementedError

    def add(
        self,
        clean: np.ndarray,
        enhanced: np.ndarray,
        noisy: Optional[np.ndarray] = None,
        filename: Optional[str] = None,
    ):
        """Add a sample and compute metric."""
        clean = self.resample_if_needed(clean.squeeze())
        enhanced = self.resample_if_needed(enhanced.squeeze())

        enh_score = self.compute(clean, enhanced)

        noisy_score = None
        if noisy is not None:
            noisy = self.resample_if_needed(noisy.squeeze())
            noisy_score = self.compute(clean, noisy)

        result = MetricResult(
            name=self.name,
            enhanced=enh_score,
            noisy=noisy_score,
            filename=filename,
        )
        self.results.append(result)
        return result

    def mean(self) -> float:
        """Compute mean of all results."""
        if not self.results:
            return 0.0
        return float(np.mean([r.enhanced for r in self.results]))

    def clear(self):
        """Clear results."""
        self.results = []


class SiSDRMetric(Metric):
    """SI-SDR metric using MLX."""

    def __init__(self, source_sr: Optional[int] = None):
        super().__init__("SI-SDR", source_sr=source_sr, target_sr=source_sr)

    def compute(self, clean: np.ndarray, degraded: np.ndarray) -> float:
        clean_mx = mx.array(clean)
        degraded_mx = mx.array(degraded)
        result = si_sdr(clean_mx, degraded_mx)
        return float(result)


class STOIMetric(Metric):
    """STOI metric."""

    def __init__(self, sr: int):
        super().__init__("STOI", source_sr=sr, target_sr=10000)

    def compute(self, clean: np.ndarray, degraded: np.ndarray) -> float:
        try:
            from pystoi import stoi

            return stoi(clean, degraded, self.target_sr, extended=False)
        except ImportError:
            # Use MLX implementation
            from .stoi import stoi_numpy

            return stoi_numpy(clean, degraded, self.target_sr)


class PESQMetric(Metric):
    """PESQ metric."""

    def __init__(self, sr: int, nb: bool = False):
        if nb:
            name = "PESQ-NB"
            target_sr = 8000
            self.mode = "nb"
        else:
            name = "PESQ"
            target_sr = 16000
            self.mode = "wb"
        super().__init__(name, source_sr=sr, target_sr=target_sr)

    def compute(self, clean: np.ndarray, degraded: np.ndarray) -> float:
        try:
            from pesq import pesq

            return pesq(self.target_sr, clean, degraded, self.mode)
        except ImportError:
            logger.warning("pypesq not installed, returning 0")
            return 0.0
        except Exception as e:
            logger.warning(f"PESQ computation failed: {e}")
            return 0.0


class SNRMetric(Metric):
    """Signal-to-Noise Ratio metric."""

    def __init__(self, source_sr: Optional[int] = None):
        super().__init__("SNR", source_sr=source_sr, target_sr=source_sr)

    def compute(self, clean: np.ndarray, degraded: np.ndarray) -> float:
        clean_mx = mx.array(clean)
        degraded_mx = mx.array(degraded)
        result = snr(clean_mx, degraded_mx)
        return float(result)


class SegmentalSNRMetric(Metric):
    """Segmental SNR metric."""

    def __init__(
        self,
        source_sr: Optional[int] = None,
        frame_length: int = 512,
        hop_length: int = 256,
    ):
        super().__init__("SSNR", source_sr=source_sr, target_sr=source_sr)
        self.frame_length = frame_length
        self.hop_length = hop_length

    def compute(self, clean: np.ndarray, degraded: np.ndarray) -> float:
        clean_mx = mx.array(clean)
        degraded_mx = mx.array(degraded)
        result = segmental_snr(
            clean_mx,
            degraded_mx,
            self.frame_length,
            self.hop_length,
        )
        return float(result)


# ============================================================================
# Evaluation Pipeline
# ============================================================================


def get_metric_factory(sr: int) -> Dict[str, Callable[[], Metric]]:
    """Get factory functions for available metrics.

    Args:
        sr: Sample rate

    Returns:
        Dictionary mapping metric name to factory function
    """
    return {
        "si-sdr": lambda: SiSDRMetric(sr),
        "sisdr": lambda: SiSDRMetric(sr),
        "stoi": lambda: STOIMetric(sr),
        "pesq": lambda: PESQMetric(sr),
        "pesq-nb": lambda: PESQMetric(sr, nb=True),
        "snr": lambda: SNRMetric(sr),
        "ssnr": lambda: SegmentalSNRMetric(sr),
    }


def evaluate_batch(
    clean_batch: List[np.ndarray],
    enhanced_batch: List[np.ndarray],
    noisy_batch: Optional[List[np.ndarray]] = None,
    sr: int = 48000,
    metrics: List[str] = ["si-sdr", "stoi", "pesq"],
    filenames: Optional[List[str]] = None,
) -> EvaluationResults:
    """Evaluate a batch of enhanced audio.

    Args:
        clean_batch: List of clean reference signals
        enhanced_batch: List of enhanced signals
        noisy_batch: Optional list of noisy signals
        sr: Sample rate
        metrics: List of metric names to compute
        filenames: Optional list of filenames for results

    Returns:
        EvaluationResults containing all computed metrics
    """
    factory = get_metric_factory(sr)
    metric_objects = {name: factory[name]() for name in metrics if name in factory}

    results = EvaluationResults()

    for i, (clean, enhanced) in enumerate(zip(clean_batch, enhanced_batch)):
        noisy = noisy_batch[i] if noisy_batch else None
        filename = filenames[i] if filenames else None

        for metric in metric_objects.values():
            result = metric.add(clean, enhanced, noisy, filename)
            results.add(result)

    return results


def evaluate_single(
    clean: np.ndarray,
    enhanced: np.ndarray,
    noisy: Optional[np.ndarray] = None,
    sr: int = 48000,
    metrics: List[str] = ["si-sdr", "stoi", "pesq"],
    filename: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a single audio file.

    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
        noisy: Optional noisy signal
        sr: Sample rate
        metrics: List of metric names to compute
        filename: Optional filename for results

    Returns:
        Dictionary of metric values
    """
    results = evaluate_batch(
        [clean],
        [enhanced],
        [noisy] if noisy is not None else None,
        sr,
        metrics,
        [filename] if filename else None,
    )
    return results.mean()


# ============================================================================
# Training Validation Utilities
# ============================================================================


class ValidationMetrics:
    """Track validation metrics during training."""

    def __init__(
        self,
        sr: int = 48000,
        metrics: List[str] = ["si-sdr"],
    ):
        self.sr = sr
        self.metric_names = metrics
        self.factory = get_metric_factory(sr)
        self.reset()

    def reset(self):
        """Reset metrics for new epoch."""
        self.metrics = {name: self.factory[name]() for name in self.metric_names if name in self.factory}
        self.n_samples = 0

    def update(
        self,
        clean: Union[mx.array, np.ndarray],
        enhanced: Union[mx.array, np.ndarray],
        noisy: Optional[Union[mx.array, np.ndarray]] = None,
    ):
        """Update metrics with a batch of samples."""
        # Convert to numpy if needed
        if isinstance(clean, mx.array):
            clean = np.array(clean)
        if isinstance(enhanced, mx.array):
            enhanced = np.array(enhanced)
        if noisy is not None and isinstance(noisy, mx.array):
            noisy = np.array(noisy)

        # Handle batched input
        if clean.ndim == 1:
            clean = clean[np.newaxis, :]
            enhanced = enhanced[np.newaxis, :]
            if noisy is not None:
                noisy = noisy[np.newaxis, :]

        batch_size = clean.shape[0]

        for i in range(batch_size):
            c = clean[i]
            e = enhanced[i]
            n = noisy[i] if noisy is not None else None

            for metric in self.metrics.values():
                metric.add(c, e, n)

        self.n_samples += batch_size

    def compute(self) -> Dict[str, float]:
        """Compute mean metrics."""
        return {name: metric.mean() for name, metric in self.metrics.items()}

    def log(self, prefix: str = "val"):
        """Log current metrics."""
        means = self.compute()
        for name, value in means.items():
            logger.info(f"{prefix}/{name}: {value:.4f}")
        return means


# ============================================================================
# Quick Evaluation Functions
# ============================================================================


def quick_eval(
    clean: mx.array,
    enhanced: mx.array,
    sr: int = 48000,
) -> Dict[str, float]:
    """Quick evaluation using only MLX-native metrics.

    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
        sr: Sample rate

    Returns:
        Dictionary with SI-SDR and SNR values
    """
    return {
        "si-sdr": float(si_sdr(clean, enhanced)),
        "snr": float(snr(clean, enhanced)),
        "ssnr": float(segmental_snr(clean, enhanced)),
    }


def compare_before_after(
    clean: mx.array,
    noisy: mx.array,
    enhanced: mx.array,
    sr: int = 48000,
) -> Dict[str, Dict[str, float]]:
    """Compare metrics before and after enhancement.

    Args:
        clean: Clean reference signal
        noisy: Noisy input signal
        enhanced: Enhanced output signal
        sr: Sample rate

    Returns:
        Dictionary with 'noisy' and 'enhanced' metric values
    """
    noisy_metrics = quick_eval(clean, noisy, sr)
    enh_metrics = quick_eval(clean, enhanced, sr)

    return {
        "noisy": noisy_metrics,
        "enhanced": enh_metrics,
        "improvement": {k: enh_metrics[k] - noisy_metrics[k] for k in noisy_metrics},
    }


def log_improvement(
    clean: mx.array,
    noisy: mx.array,
    enhanced: mx.array,
    sr: int = 48000,
    prefix: str = "",
):
    """Log improvement from noisy to enhanced."""
    comparison = compare_before_after(clean, noisy, enhanced, sr)

    for metric_name in comparison["noisy"]:
        noisy_val = comparison["noisy"][metric_name]
        enh_val = comparison["enhanced"][metric_name]
        imp_val = comparison["improvement"][metric_name]
        logger.info(f"{prefix}{metric_name}: noisy={noisy_val:.2f} → enhanced={enh_val:.2f} " f"(Δ={imp_val:+.2f})")
