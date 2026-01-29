"""Visualization utilities for MLX-based speech enhancement.

Provides plotting functions for spectrograms, waveforms, loss curves,
and other diagnostic visualizations for training and inference.

Usage:
    from df_mlx.visualization import spec_figure, plot_waveform

    # Plot spectrogram
    fig = spec_figure(audio, sr=48000, from_audio=True)
    fig.savefig("spectrogram.png")

    # Plot waveform comparison
    fig = plot_waveform_comparison(noisy, enhanced, clean, sr=48000)
    fig.savefig("comparison.png")
"""

from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np

# Lazy import matplotlib to avoid startup overhead
_plt = None
_Figure = None


def _get_matplotlib():
    """Lazy import matplotlib."""
    global _plt, _Figure
    if _plt is None:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        from matplotlib.figure import Figure  # type: ignore[import-not-found]

        _plt = plt
        _Figure = Figure
    return _plt


def to_numpy(x: Union[mx.array, np.ndarray]) -> np.ndarray:
    """Convert MLX array to numpy."""
    if isinstance(x, mx.array):
        return np.array(x)
    return x


def spec_figure(
    spec: Union[mx.array, np.ndarray],
    sr: int,
    figsize: Tuple[int, int] = (15, 5),
    colorbar: bool = False,
    colorbar_format: Optional[str] = None,
    from_audio: bool = False,
    figure=None,
    return_im: bool = False,
    labels: bool = False,
    xlabels: bool = False,
    ylabels: bool = False,
    n_fft: int = 1024,
    hop: int = 256,
    **kwargs,
):
    """Create a spectrogram figure.

    Args:
        spec: Audio waveform or spectrogram tensor
        sr: Sample rate
        figsize: Figure size (width, height)
        colorbar: Whether to show colorbar
        colorbar_format: Format string for colorbar labels
        from_audio: If True, compute STFT from audio waveform
        figure: Existing figure to use
        return_im: If True, return image handle instead of figure
        labels: Add both x and y labels
        xlabels: Add x-axis label
        ylabels: Add y-axis label
        n_fft: FFT size (if from_audio=True)
        hop: Hop size (if from_audio=True)
        **kwargs: Additional arguments passed to specshow

    Returns:
        Figure or image handle
    """
    plt = _get_matplotlib()
    spec = to_numpy(spec)

    if labels or xlabels:
        kwargs.setdefault("xlabel", "Time [s]")
    if labels or ylabels:
        if kwargs.get("kHz", False):
            kwargs.setdefault("ylabel", "Frequency [kHz]")
        else:
            kwargs.setdefault("ylabel", "Frequency [Hz]")

    if from_audio:
        # Compute STFT from audio
        from scipy.signal import stft as scipy_stft

        _, _, spec = scipy_stft(spec, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        spec = np.abs(spec)

    # Convert to dB if complex
    if np.iscomplexobj(spec):
        spec = 20 * np.log10(np.abs(spec) + 1e-12)
    elif spec.ndim == 3 and spec.shape[-1] == 2:
        # Real/imag format
        spec = np.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)
        spec = 20 * np.log10(spec + 1e-12)

    kwargs.setdefault("vmax", max(0.0, np.max(spec)))

    if figure is None:
        figure = plt.figure(figsize=figsize)
        figure.set_layout_engine("tight")

    if spec.ndim > 2:
        spec = spec.squeeze()

    im = specshow(spec, sr, n_fft=n_fft, hop=hop, **kwargs)

    if colorbar:
        ckwargs = {}
        if "ax" in kwargs:
            if colorbar_format is None:
                if kwargs.get("vmin") is not None or kwargs.get("vmax") is not None:
                    colorbar_format = "%+2.0f dB"
            ckwargs = {"ax": kwargs["ax"]}
        plt.colorbar(im, format=colorbar_format, **ckwargs)

    if return_im:
        return im
    return figure


def specshow(
    spec: np.ndarray,
    sr: int,
    ax=None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    n_fft: Optional[int] = None,
    hop: Optional[int] = None,
    t: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    vmin: float = -100,
    vmax: float = 0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    kHz: bool = False,
    ticks: bool = False,
    cmap: str = "inferno",
):
    """Plot a spectrogram of shape [F, T].

    Args:
        spec: Spectrogram array [F, T]
        sr: Sample rate
        ax: Matplotlib axis (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        n_fft: FFT size for computing frequency axis
        hop: Hop size for computing time axis
        t: Time axis values (optional)
        f: Frequency axis values (optional)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        xlim: X-axis limits
        ylim: Y-axis limits
        kHz: If True, show frequency in kHz
        ticks: If False, hide axis ticks
        cmap: Colormap name

    Returns:
        Image handle from pcolormesh
    """
    plt = _get_matplotlib()

    if spec.ndim > 2:
        spec = spec.squeeze()

    if ax is not None:
        set_title = ax.set_title
        set_xlabel = ax.set_xlabel
        set_ylabel = ax.set_ylabel
        set_xlim = ax.set_xlim
        set_ylim = ax.set_ylim
        pcolormesh = ax.pcolormesh
        axis = ax.axis
    else:
        ax = plt
        set_title = plt.title
        set_xlabel = plt.xlabel
        set_ylabel = plt.ylabel
        set_xlim = plt.xlim
        set_ylim = plt.ylim
        pcolormesh = plt.pcolormesh
        axis = plt.axis

    n_fft = n_fft or (spec.shape[0] - 1) * 2
    hop = hop or n_fft // 4

    if t is None:
        t = np.arange(0, spec.shape[-1]) * hop / sr
    if f is None:
        f = np.arange(0, spec.shape[0]) * sr // 2 / (n_fft // 2)
        if kHz:
            f = f / 1000

    im = pcolormesh(t, f, spec, rasterized=True, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    if not ticks:
        axis("off")
    if title is not None:
        set_title(title)
    if xlabel is not None:
        set_xlabel(xlabel)
    if ylabel is not None:
        set_ylabel(ylabel)
    if xlim is not None:
        set_xlim(xlim)
    if ylim is not None:
        set_ylim(ylim)

    return im


def plot_waveform(
    audio: Union[mx.array, np.ndarray],
    sr: int,
    figsize: Tuple[int, int] = (15, 3),
    title: Optional[str] = None,
    ax=None,
):
    """Plot audio waveform.

    Args:
        audio: Audio waveform [T] or [1, T]
        sr: Sample rate
        figsize: Figure size
        title: Plot title
        ax: Matplotlib axis (optional)

    Returns:
        Figure or axis
    """
    plt = _get_matplotlib()
    audio = to_numpy(audio).squeeze()
    t = np.arange(len(audio)) / sr

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    ax.plot(t, audio, linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim((0, t[-1]))
    if title:
        ax.set_title(title)

    return fig if fig else ax


def plot_waveform_comparison(
    noisy: Union[mx.array, np.ndarray],
    enhanced: Union[mx.array, np.ndarray],
    clean: Optional[Union[mx.array, np.ndarray]] = None,
    sr: int = 48000,
    figsize: Tuple[int, int] = (15, 8),
):
    """Plot waveform comparison between noisy, enhanced, and clean.

    Args:
        noisy: Noisy audio waveform
        enhanced: Enhanced audio waveform
        clean: Clean reference waveform (optional)
        sr: Sample rate
        figsize: Figure size

    Returns:
        Figure
    """
    plt = _get_matplotlib()

    n_plots = 3 if clean is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    plot_waveform(noisy, sr, title="Noisy", ax=axes[0])
    plot_waveform(enhanced, sr, title="Enhanced", ax=axes[1])
    if clean is not None:
        plot_waveform(clean, sr, title="Clean", ax=axes[2])

    fig.tight_layout()
    return fig


def plot_spectrogram_comparison(
    noisy: Union[mx.array, np.ndarray],
    enhanced: Union[mx.array, np.ndarray],
    clean: Optional[Union[mx.array, np.ndarray]] = None,
    sr: int = 48000,
    figsize: Tuple[int, int] = (15, 10),
    n_fft: int = 1024,
    hop: int = 256,
):
    """Plot spectrogram comparison between noisy, enhanced, and clean.

    Args:
        noisy: Noisy audio waveform
        enhanced: Enhanced audio waveform
        clean: Clean reference waveform (optional)
        sr: Sample rate
        figsize: Figure size
        n_fft: FFT size
        hop: Hop size

    Returns:
        Figure
    """
    plt = _get_matplotlib()

    n_plots = 3 if clean is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)

    spec_figure(noisy, sr, from_audio=True, n_fft=n_fft, hop=hop, ax=axes[0])
    axes[0].set_title("Noisy")

    spec_figure(enhanced, sr, from_audio=True, n_fft=n_fft, hop=hop, ax=axes[1])
    axes[1].set_title("Enhanced")

    if clean is not None:
        spec_figure(clean, sr, from_audio=True, n_fft=n_fft, hop=hop, ax=axes[2])
        axes[2].set_title("Clean")

    fig.tight_layout()
    return fig


def plot_loss_curves(
    train_losses: Union[list, np.ndarray],
    val_losses: Optional[Union[list, np.ndarray]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Training Progress",
    log_scale: bool = True,
):
    """Plot training and validation loss curves.

    Args:
        train_losses: Training loss values per epoch
        val_losses: Validation loss values per epoch (optional)
        figsize: Figure size
        title: Plot title
        log_scale: If True, use log scale for y-axis

    Returns:
        Figure
    """
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)

    if val_losses is not None:
        ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    return fig


def plot_metrics(
    metrics: dict,
    figsize: Tuple[int, int] = (12, 4),
    title: str = "Evaluation Metrics",
):
    """Plot evaluation metrics as a bar chart.

    Args:
        metrics: Dictionary of metric name -> value
        figsize: Figure size
        title: Plot title

    Returns:
        Figure
    """
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)

    names = list(metrics.keys())
    values = list(metrics.values())

    bars = ax.bar(names, values, color="steelblue", edgecolor="navy")
    ax.set_ylabel("Score")
    ax.set_title(title)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    return fig


def plot_attention_weights(
    attention: Union[mx.array, np.ndarray],
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Attention Weights",
    cmap: str = "viridis",
):
    """Plot attention weights heatmap.

    Args:
        attention: Attention weights [T_query, T_key] or [H, T_query, T_key]
        figsize: Figure size
        title: Plot title
        cmap: Colormap name

    Returns:
        Figure
    """
    plt = _get_matplotlib()
    attention = to_numpy(attention)

    if attention.ndim == 3:
        # Multi-head attention, average over heads
        attention = attention.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention, aspect="auto", cmap=cmap)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig


def save_figure(fig, path: str, dpi: int = 150, **kwargs):
    """Save figure to file.

    Args:
        fig: Matplotlib figure
        path: Output file path
        dpi: Resolution in dots per inch
        **kwargs: Additional arguments to savefig
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    plt = _get_matplotlib()
    plt.close(fig)


# Quick test when run directly
if __name__ == "__main__":
    import mlx.core as mx  # noqa: F811

    # Test with random audio
    audio = mx.random.normal((48000,))

    print("Testing visualization utilities...")
    fig = spec_figure(np.array(audio), sr=48000, from_audio=True, labels=True)
    print("  spec_figure: OK")

    fig = plot_waveform(audio, sr=48000, title="Test Waveform")
    print("  plot_waveform: OK")

    metrics = {"SI-SDR": 15.2, "STOI": 0.89, "PESQ": 3.45}
    fig = plot_metrics(metrics)
    print("  plot_metrics: OK")

    print("✓ All visualization tests passed!")
