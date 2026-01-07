"""MLX-native datastore for DeepFilterNet training.

This module provides efficient data storage and loading for MLX training:
- Sharded .npz files for parallel loading
- Pre-computed spectral features (STFT, ERB, DF)
- Memory-efficient streaming with configurable batch sizes
- Index-based random access and shuffling
- Background thread shard saving for non-blocking I/O
- Resume support for interrupted builds

The datastore format:
    dataset_dir/
        index.json          # Metadata and shard manifest
        train_000.npz       # Sharded training data
        train_001.npz
        ...
        valid_000.npz       # Validation data
        ...
"""

import json
import random
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

# Type alias for batch dict returned by data loaders
BatchDict = Dict[str, Union[mx.array, Tuple[mx.array, mx.array]]]

# Global flag for graceful shutdown
_shutdown_requested = False
_shutdown_message_shown = False


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested, _shutdown_message_shown
    _shutdown_requested = True
    if not _shutdown_message_shown:
        _shutdown_message_shown = True
        print("\n  Interrupt received, saving pending writes...")


def _install_signal_handlers():
    """Install signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested (for use by external code)."""
    return _shutdown_requested


@dataclass
class DatastoreConfig:
    """Configuration for MLX datastore."""

    sample_rate: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96
    norm_alpha: float = 0.99
    samples_per_shard: int = 1000
    min_nb_freqs: int = 481  # FFT bins
    dtype: str = "float32"
    compress: bool = False  # Use np.savez (fast) vs np.savez_compressed (slow but smaller)


@dataclass
class ShardInfo:
    """Information about a single shard."""

    path: str
    num_samples: int
    split: str  # "train", "valid", "test"


@dataclass
class DatastoreIndex:
    """Index for the entire datastore."""

    config: DatastoreConfig
    shards: List[ShardInfo] = field(default_factory=list)
    total_samples: Dict[str, int] = field(default_factory=dict)
    files_processed: Dict[str, int] = field(default_factory=dict)  # Files processed per split

    def to_dict(self) -> dict:
        return {
            "config": {
                "sample_rate": self.config.sample_rate,
                "fft_size": self.config.fft_size,
                "hop_size": self.config.hop_size,
                "nb_erb": self.config.nb_erb,
                "nb_df": self.config.nb_df,
                "norm_alpha": self.config.norm_alpha,
                "samples_per_shard": self.config.samples_per_shard,
                "min_nb_freqs": self.config.min_nb_freqs,
                "dtype": self.config.dtype,
                "compress": self.config.compress,
            },
            "shards": [{"path": s.path, "num_samples": s.num_samples, "split": s.split} for s in self.shards],
            "total_samples": self.total_samples,
            "files_processed": self.files_processed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatastoreIndex":
        config = DatastoreConfig(**data["config"])
        shards = [ShardInfo(**s) for s in data["shards"]]
        files_processed = data.get("files_processed", {})  # Backward compat
        return cls(config=config, shards=shards, total_samples=data["total_samples"], files_processed=files_processed)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DatastoreIndex":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class MLXDatastoreWriter:
    """Write pre-computed features to MLX datastore format.

    Features:
    - Background thread for non-blocking shard saves
    - Resume support: can continue from interrupted builds
    - Progress tracking with incremental index updates
    - Graceful shutdown on Ctrl+C
    """

    def __init__(
        self,
        output_dir: Path,
        config: Optional[DatastoreConfig] = None,
        resume: bool = True,
        num_write_threads: int = 2,
    ):
        """Initialize datastore writer.

        Args:
            output_dir: Directory to write datastore to
            config: Datastore configuration
            resume: If True, resume from existing index if present
            num_write_threads: Number of background threads for writing shards
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or DatastoreConfig()

        self._current_shard: Dict[str, List[np.ndarray]] = {}
        self._current_split: str = "train"
        self._shard_counts: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
        self._sample_counts: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
        self._files_processed: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
        self._shards: List[ShardInfo] = []

        # Background writing with daemon threads for clean shutdown
        self._write_executor = ThreadPoolExecutor(
            max_workers=num_write_threads,
            thread_name_prefix="shard_writer",
        )
        self._pending_writes: List = []
        self._write_lock = threading.Lock()
        self._shutdown = False

        # Install signal handlers for graceful shutdown
        _install_signal_handlers()

        # Resume from existing index if available
        self._resumed = False
        if resume:
            self._try_resume()

    def _try_resume(self) -> None:
        """Try to resume from existing index."""
        index_path = self.output_dir / "index.json"
        if not index_path.exists():
            return

        try:
            existing_index = DatastoreIndex.load(index_path)

            # Verify config matches (allow minor differences)
            if (
                existing_index.config.sample_rate != self.config.sample_rate
                or existing_index.config.fft_size != self.config.fft_size
            ):
                print("  Warning: Config mismatch, starting fresh")
                return

            # Load existing shard info
            self._shards = existing_index.shards
            self._sample_counts = existing_index.total_samples.copy()
            self._files_processed = existing_index.files_processed.copy()

            # Count shards per split
            for shard in self._shards:
                self._shard_counts[shard.split] = max(
                    self._shard_counts[shard.split],
                    int(shard.path.split("_")[1].split(".")[0]) + 1,
                )

            self._resumed = True
            print("  Resumed from existing datastore:")
            for split in ["train", "valid", "test"]:
                samples = self._sample_counts.get(split, 0)
                files = self._files_processed.get(split, 0)
                if samples > 0 or files > 0:
                    print(f"    {split.capitalize()}: {samples:,} samples from {files:,} files")

        except Exception as e:
            print(f"  Warning: Could not resume from index: {e}")

    @property
    def resumed(self) -> bool:
        """Whether this writer resumed from an existing datastore."""
        return self._resumed

    def get_sample_count(self, split: str) -> int:
        """Get current sample count for a split."""
        return self._sample_counts.get(split, 0)

    def get_files_processed(self, split: str) -> int:
        """Get number of files already processed for a split."""
        return self._files_processed.get(split, 0)

    def increment_files_processed(self, split: str) -> None:
        """Increment the count of processed files for a split."""
        self._files_processed[split] = self._files_processed.get(split, 0) + 1

    def set_split(self, split: str) -> None:
        """Set current split (train/valid/test)."""
        if self._current_shard:
            self._flush_shard()
        self._current_split = split
        self._current_shard = {}

    def add_sample(
        self,
        spec_real: np.ndarray,
        spec_imag: np.ndarray,
        feat_erb: np.ndarray,
        feat_spec: np.ndarray,
        clean_real: np.ndarray,
        clean_imag: np.ndarray,
        snr_db: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a single sample to the current shard.

        Args:
            spec_real: Noisy spectrum real part (time, freq)
            spec_imag: Noisy spectrum imaginary part (time, freq)
            feat_erb: ERB features (time, erb_bands)
            feat_spec: DF-band features (time, df_bins, 2)
            clean_real: Clean spectrum real part (time, freq)
            clean_imag: Clean spectrum imaginary part (time, freq)
            snr_db: Signal-to-noise ratio in dB
            metadata: Optional metadata dict
        """
        if "spec_real" not in self._current_shard:
            self._current_shard = {
                "spec_real": [],
                "spec_imag": [],
                "feat_erb": [],
                "feat_spec": [],
                "clean_real": [],
                "clean_imag": [],
                "snr_db": [],
            }

        self._current_shard["spec_real"].append(spec_real.astype(np.float32))
        self._current_shard["spec_imag"].append(spec_imag.astype(np.float32))
        self._current_shard["feat_erb"].append(feat_erb.astype(np.float32))
        self._current_shard["feat_spec"].append(feat_spec.astype(np.float32))
        self._current_shard["clean_real"].append(clean_real.astype(np.float32))
        self._current_shard["clean_imag"].append(clean_imag.astype(np.float32))
        self._current_shard["snr_db"].append(np.float32(snr_db))

        if len(self._current_shard["spec_real"]) >= self.config.samples_per_shard:
            self._flush_shard(force=False)

    def _write_shard_async(
        self,
        shard_path: Path,
        shard_data: Dict[str, np.ndarray],
        shard_info: ShardInfo,
    ) -> None:
        """Write shard to disk (called from background thread)."""
        try:
            # Use compressed or uncompressed based on config
            save_fn = np.savez_compressed if self.config.compress else np.savez
            save_fn(
                shard_path,
                spec_real=shard_data["spec_real"],
                spec_imag=shard_data["spec_imag"],
                feat_erb=shard_data["feat_erb"],
                feat_spec=shard_data["feat_spec"],
                clean_real=shard_data["clean_real"],
                clean_imag=shard_data["clean_imag"],
                snr_db=shard_data["snr_db"],
            )

            # Update index incrementally
            with self._write_lock:
                self._shards.append(shard_info)
                self._save_index_incremental()

        except Exception as e:
            print(f"  Error writing shard {shard_path}: {e}")

    def _save_index_incremental(self) -> None:
        """Save index after each shard write for resume support."""
        index = DatastoreIndex(
            config=self.config,
            shards=self._shards,
            total_samples=self._sample_counts.copy(),
            files_processed=self._files_processed.copy(),
        )
        index.save(self.output_dir / "index.json")

    def _flush_shard(self, force: bool = False) -> None:
        """Queue current shard for async writing.

        Args:
            force: If True, flush even during shutdown.
        """
        if not self._current_shard or not self._current_shard.get("spec_real"):
            return

        # Only skip on shutdown if not forcing
        if self._shutdown and not force:
            print("  Shutdown in progress, skipping new shard writes")
            return

        shard_idx = self._shard_counts[self._current_split]
        shard_name = f"{self._current_split}_{shard_idx:04d}.npz"
        shard_path = self.output_dir / shard_name

        num_samples = len(self._current_shard["spec_real"])

        # Stack arrays for writing
        shard_data = {
            "spec_real": np.stack(self._current_shard["spec_real"]),
            "spec_imag": np.stack(self._current_shard["spec_imag"]),
            "feat_erb": np.stack(self._current_shard["feat_erb"]),
            "feat_spec": np.stack(self._current_shard["feat_spec"]),
            "clean_real": np.stack(self._current_shard["clean_real"]),
            "clean_imag": np.stack(self._current_shard["clean_imag"]),
            "snr_db": np.array(self._current_shard["snr_db"]),
        }

        shard_info = ShardInfo(path=shard_name, num_samples=num_samples, split=self._current_split)

        # Update counts immediately
        self._shard_counts[self._current_split] += 1
        self._sample_counts[self._current_split] += num_samples

        # Clean up completed futures to prevent memory buildup
        self._pending_writes = [f for f in self._pending_writes if not f.done()]

        # Submit async write
        future = self._write_executor.submit(self._write_shard_async, shard_path, shard_data, shard_info)
        self._pending_writes.append(future)

        # Clear current shard
        self._current_shard = {}

        print(f"  Queued shard: {shard_name} ({num_samples} samples)")

    def wait_for_writes(self, timeout: float = 30.0) -> int:
        """Wait for pending writes to complete.

        Args:
            timeout: Maximum total time to wait in seconds.

        Returns:
            Number of completed writes.
        """
        from concurrent.futures import TimeoutError as FuturesTimeoutError
        from concurrent.futures import as_completed

        completed = 0
        pending = list(self._pending_writes)
        self._pending_writes.clear()

        if not pending:
            return 0

        try:
            for future in as_completed(pending, timeout=timeout):
                try:
                    future.result()  # Raises if the write failed
                    completed += 1
                except Exception as e:
                    print(f"  Warning: shard write failed: {e}")
        except FuturesTimeoutError:
            # Some futures didn't complete in time
            for f in pending:
                if not f.done():
                    self._pending_writes.append(f)

        return completed

    def finalize(self, force: bool = False) -> DatastoreIndex | None:
        """Finalize datastore and write index.

        Args:
            force: If True, always save index (even on interrupt).

        Returns:
            DatastoreIndex if successful, None only if no data was written.
        """
        global _shutdown_requested

        # Flush any remaining samples (unless already shutting down)
        if self._current_shard and not self._shutdown:
            # Allow flush even on interrupt - we want to save work
            self._flush_shard()

        # Always wait for pending writes - this is critical for data integrity
        if self._pending_writes:
            num_pending = len(self._pending_writes)
            print(f"  Waiting for {num_pending} pending write(s)...")
            completed = self.wait_for_writes(timeout=60.0)  # Wait up to 60s
            print(f"  Completed {completed} write(s).")

        # Shutdown executor
        self._shutdown = True
        try:
            self._write_executor.shutdown(wait=True, cancel_futures=False)
        except TypeError:
            # Python < 3.9 doesn't have cancel_futures
            self._write_executor.shutdown(wait=True)

        # Always save index if we have any shards
        if not self._shards:
            print("\n  No shards written.")
            return None

        # Save final index
        index = DatastoreIndex(
            config=self.config,
            shards=self._shards,
            total_samples=self._sample_counts.copy(),
            files_processed=self._files_processed.copy(),
        )
        index.save(self.output_dir / "index.json")

        print("\nDatastore finalized:")
        for split in ["train", "valid", "test"]:
            samples = self._sample_counts.get(split, 0)
            files = self._files_processed.get(split, 0)
            if samples > 0 or files > 0:
                print(f"  {split.capitalize()}: {samples:,} samples from {files:,} files")
        if _shutdown_requested:
            print("  (Interrupted - progress saved for resume)")

        return index

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure writes complete and index is saved, even on interrupt."""
        # Always try to finalize and save work
        try:
            self.finalize(force=True)
        except Exception as e:
            print(f"  Warning: Error during finalize: {e}")

        # Don't suppress exceptions
        return False


class MLXDataLoader:
    """Efficient data loader for MLX training.

    Loads pre-computed features from sharded .npz files and yields
    batches as MLX arrays.

    Example:
        >>> loader = MLXDataLoader("dataset/", split="train", batch_size=8)
        >>> for batch in loader:
        ...     spec, feat_erb, feat_spec, target = batch
        ...     loss = train_step(spec, feat_erb, feat_spec, target)
    """

    def __init__(
        self,
        datastore_dir: Path | str,
        split: str = "train",
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.datastore_dir = Path(datastore_dir)
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_samples = max_samples

        # Load index
        self.index = DatastoreIndex.load(self.datastore_dir / "index.json")

        # Get shards for this split
        self.shards = [s for s in self.index.shards if s.split == split]
        if not self.shards:
            raise ValueError(f"No shards found for split '{split}'")

        # Build sample index: (shard_idx, sample_idx)
        self._sample_indices: List[Tuple[int, int]] = []
        for shard_idx, shard in enumerate(self.shards):
            for sample_idx in range(shard.num_samples):
                self._sample_indices.append((shard_idx, sample_idx))

        if max_samples:
            self._sample_indices = self._sample_indices[:max_samples]

        self._loaded_shard_idx: Optional[int] = None
        self._loaded_shard_data: Optional[Dict[str, np.ndarray]] = None

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self._sample_indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        return len(self._sample_indices)

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        """Load a shard into memory."""
        if self._loaded_shard_idx == shard_idx and self._loaded_shard_data is not None:
            return self._loaded_shard_data

        shard = self.shards[shard_idx]
        shard_path = self.datastore_dir / shard.path

        data = dict(np.load(shard_path))
        self._loaded_shard_idx = shard_idx
        self._loaded_shard_data = data
        return data

    def _get_sample(self, shard_idx: int, sample_idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample."""
        data = self._load_shard(shard_idx)
        return {
            "spec_real": data["spec_real"][sample_idx],
            "spec_imag": data["spec_imag"][sample_idx],
            "feat_erb": data["feat_erb"][sample_idx],
            "feat_spec": data["feat_spec"][sample_idx],
            "clean_real": data["clean_real"][sample_idx],
            "clean_imag": data["clean_imag"][sample_idx],
            "snr_db": data["snr_db"][sample_idx],
        }

    def __getitem__(self, index: int) -> BatchDict:
        """Get a specific batch by index.

        Args:
            index: Batch index (0 to len(self) - 1)

        Returns:
            Dict with keys:
            - "spec": (spec_real, spec_imag) tuple of mx.arrays
            - "feat_erb": ERB features mx.array
            - "feat_spec": DF-band features mx.array
            - "target": (clean_real, clean_imag) tuple of mx.arrays
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(f"Batch index {index} out of range [0, {len(self)})")

        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        batch_spec_real = []
        batch_spec_imag = []
        batch_feat_erb = []
        batch_feat_spec = []
        batch_clean_real = []
        batch_clean_imag = []

        for shard_idx, sample_idx in self._sample_indices[start_idx:end_idx]:
            sample = self._get_sample(shard_idx, sample_idx)
            batch_spec_real.append(sample["spec_real"])
            batch_spec_imag.append(sample["spec_imag"])
            batch_feat_erb.append(sample["feat_erb"])
            batch_feat_spec.append(sample["feat_spec"])
            batch_clean_real.append(sample["clean_real"])
            batch_clean_imag.append(sample["clean_imag"])

        return self._make_batch(
            batch_spec_real,
            batch_spec_imag,
            batch_feat_erb,
            batch_feat_spec,
            batch_clean_real,
            batch_clean_imag,
        )

    def __iter__(self) -> Iterator[BatchDict]:
        """Iterate over batches."""
        indices = self._sample_indices.copy()
        if self.shuffle:
            random.shuffle(indices)

        batch_spec_real = []
        batch_spec_imag = []
        batch_feat_erb = []
        batch_feat_spec = []
        batch_clean_real = []
        batch_clean_imag = []

        for shard_idx, sample_idx in indices:
            sample = self._get_sample(shard_idx, sample_idx)

            batch_spec_real.append(sample["spec_real"])
            batch_spec_imag.append(sample["spec_imag"])
            batch_feat_erb.append(sample["feat_erb"])
            batch_feat_spec.append(sample["feat_spec"])
            batch_clean_real.append(sample["clean_real"])
            batch_clean_imag.append(sample["clean_imag"])

            if len(batch_spec_real) >= self.batch_size:
                yield self._make_batch(
                    batch_spec_real,
                    batch_spec_imag,
                    batch_feat_erb,
                    batch_feat_spec,
                    batch_clean_real,
                    batch_clean_imag,
                )
                batch_spec_real = []
                batch_spec_imag = []
                batch_feat_erb = []
                batch_feat_spec = []
                batch_clean_real = []
                batch_clean_imag = []

        # Handle remaining samples
        if batch_spec_real and not self.drop_last:
            yield self._make_batch(
                batch_spec_real,
                batch_spec_imag,
                batch_feat_erb,
                batch_feat_spec,
                batch_clean_real,
                batch_clean_imag,
            )

    def _make_batch(
        self,
        spec_real: List[np.ndarray],
        spec_imag: List[np.ndarray],
        feat_erb: List[np.ndarray],
        feat_spec: List[np.ndarray],
        clean_real: List[np.ndarray],
        clean_imag: List[np.ndarray],
    ) -> Dict[str, mx.array | Tuple[mx.array, mx.array]]:
        """Convert lists to MLX batch tensors.

        Returns:
            Dict with keys matching train_with_data.py:
            - "spec": (spec_real, spec_imag) tuple
            - "feat_erb": ERB features array
            - "feat_spec": DF-band features array
            - "target": (clean_real, clean_imag) tuple
        """
        return {
            "spec": (
                mx.array(np.stack(spec_real)),
                mx.array(np.stack(spec_imag)),
            ),
            "feat_erb": mx.array(np.stack(feat_erb)),
            "feat_spec": mx.array(np.stack(feat_spec)),
            "target": (
                mx.array(np.stack(clean_real)),
                mx.array(np.stack(clean_imag)),
            ),
        }


class StreamingMLXDataLoader:
    """Memory-efficient streaming data loader.

    Loads one shard at a time for large datasets that don't fit in memory.
    """

    def __init__(
        self,
        datastore_dir: Path | str,
        split: str = "train",
        batch_size: int = 8,
        shuffle_shards: bool = True,
        shuffle_samples: bool = True,
    ):
        self.datastore_dir = Path(datastore_dir)
        self.split = split
        self.batch_size = batch_size
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples

        self.index = DatastoreIndex.load(self.datastore_dir / "index.json")
        self.shards = [s for s in self.index.shards if s.split == split]

    def __len__(self) -> int:
        total = sum(s.num_samples for s in self.shards)
        return total // self.batch_size

    def __getitem__(self, index: int) -> BatchDict:
        """Get a specific batch by index.

        Note: For streaming loader, this requires loading shards sequentially
        until we reach the target batch. For random access, use MLXDataLoader.

        Args:
            index: Batch index (0 to len(self) - 1)

        Returns:
            Dict with keys:
            - "spec": (spec_real, spec_imag) tuple of mx.arrays
            - "feat_erb": ERB features mx.array
            - "feat_spec": DF-band features mx.array
            - "target": (clean_real, clean_imag) tuple of mx.arrays
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(f"Batch index {index} out of range [0, {len(self)})")

        # Calculate which samples we need
        start_sample = index * self.batch_size
        end_sample = start_sample + self.batch_size

        # Find which shard(s) contain these samples
        batch_spec_real = []
        batch_spec_imag = []
        batch_feat_erb = []
        batch_feat_spec = []
        batch_clean_real = []
        batch_clean_imag = []

        current_sample = 0
        for shard in self.shards:
            shard_end = current_sample + shard.num_samples

            # Check if this shard contains any samples we need
            if shard_end > start_sample and current_sample < end_sample:
                shard_path = self.datastore_dir / shard.path
                data = dict(np.load(shard_path))

                # Calculate which samples from this shard
                local_start = max(0, start_sample - current_sample)
                local_end = min(shard.num_samples, end_sample - current_sample)

                for idx in range(local_start, local_end):
                    batch_spec_real.append(data["spec_real"][idx])
                    batch_spec_imag.append(data["spec_imag"][idx])
                    batch_feat_erb.append(data["feat_erb"][idx])
                    batch_feat_spec.append(data["feat_spec"][idx])
                    batch_clean_real.append(data["clean_real"][idx])
                    batch_clean_imag.append(data["clean_imag"][idx])

            current_sample = shard_end
            if current_sample >= end_sample:
                break

        return self._make_batch(
            batch_spec_real,
            batch_spec_imag,
            batch_feat_erb,
            batch_feat_spec,
            batch_clean_real,
            batch_clean_imag,
        )

    def __iter__(self) -> Iterator[BatchDict]:
        """Iterate over batches, streaming one shard at a time."""
        shard_order = list(range(len(self.shards)))
        if self.shuffle_shards:
            random.shuffle(shard_order)

        buffer_spec_real = []
        buffer_spec_imag = []
        buffer_feat_erb = []
        buffer_feat_spec = []
        buffer_clean_real = []
        buffer_clean_imag = []

        for shard_idx in shard_order:
            shard = self.shards[shard_idx]
            shard_path = self.datastore_dir / shard.path
            data = dict(np.load(shard_path))

            indices = list(range(shard.num_samples))
            if self.shuffle_samples:
                random.shuffle(indices)

            for idx in indices:
                buffer_spec_real.append(data["spec_real"][idx])
                buffer_spec_imag.append(data["spec_imag"][idx])
                buffer_feat_erb.append(data["feat_erb"][idx])
                buffer_feat_spec.append(data["feat_spec"][idx])
                buffer_clean_real.append(data["clean_real"][idx])
                buffer_clean_imag.append(data["clean_imag"][idx])

                if len(buffer_spec_real) >= self.batch_size:
                    yield self._make_batch(
                        buffer_spec_real[: self.batch_size],
                        buffer_spec_imag[: self.batch_size],
                        buffer_feat_erb[: self.batch_size],
                        buffer_feat_spec[: self.batch_size],
                        buffer_clean_real[: self.batch_size],
                        buffer_clean_imag[: self.batch_size],
                    )
                    buffer_spec_real = buffer_spec_real[self.batch_size :]
                    buffer_spec_imag = buffer_spec_imag[self.batch_size :]
                    buffer_feat_erb = buffer_feat_erb[self.batch_size :]
                    buffer_feat_spec = buffer_feat_spec[self.batch_size :]
                    buffer_clean_real = buffer_clean_real[self.batch_size :]
                    buffer_clean_imag = buffer_clean_imag[self.batch_size :]

    def _make_batch(
        self,
        spec_real: List[np.ndarray],
        spec_imag: List[np.ndarray],
        feat_erb: List[np.ndarray],
        feat_spec: List[np.ndarray],
        clean_real: List[np.ndarray],
        clean_imag: List[np.ndarray],
    ) -> BatchDict:
        """Convert lists to MLX batch tensors.

        Returns:
            Dict with keys matching train_with_data.py:
            - "spec": (spec_real, spec_imag) tuple
            - "feat_erb": ERB features array
            - "feat_spec": DF-band features array
            - "target": (clean_real, clean_imag) tuple
        """
        return {
            "spec": (
                mx.array(np.stack(spec_real)),
                mx.array(np.stack(spec_imag)),
            ),
            "feat_erb": mx.array(np.stack(feat_erb)),
            "feat_spec": mx.array(np.stack(feat_spec)),
            "target": (
                mx.array(np.stack(clean_real)),
                mx.array(np.stack(clean_imag)),
            ),
        }
