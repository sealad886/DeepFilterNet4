#!/usr/bin/env python3
"""Prepare audio data for DeepFilterNet training by converting to HDF5 format.

Uses torchcodec for efficient audio decoding and encoding with built-in resampling.
"""

import argparse
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Optional, cast

import h5py as h5
import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from tqdm import tqdm

from df.logger import init_logger

if TYPE_CHECKING:
    from h5py import Group


def _worker_init_fn(worker_id: int, quiet: bool = False, log_file: str | None = None) -> None:
    """Initialize logging in DataLoader worker processes."""
    if quiet:
        logger.remove()
        if log_file:
            logger.add(log_file, level="DEBUG")


def _check_file(file: str, working_dir: str) -> str:
    """Validate a file path exists (module-level for pickling in multiprocessing)."""
    file = file.strip()
    if not file:
        return ""
    # Handle both absolute and relative paths
    if os.path.isabs(file):
        path = file
    else:
        path = os.path.join(working_dir, file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path


def write_to_h5(
    file_name: str,
    data: dict,
    sr: int,
    max_freq: int = -1,
    dtype: str = "float32",
    codec: str = "pcm",
    mono: bool = False,
    compression: Optional[str] = None,
    num_workers: int = 4,
    quiet: bool = False,
    log_file: str | None = None,
):
    """Creates an HDF5 dataset based on the provided dict.

    Args:
        file_name: Full path of the HDF5 dataset file.
        data: Dictionary containing for each key (dataset group) a list of file names
            to pre-process and store.
        sr: Target sample rate for all audio.
        max_freq: Maximum frequency to consider during training (metadata only).
        dtype: Storage dtype - 'float32' or 'int16'.
        codec: Storage codec - 'pcm', 'flac', or 'vorbis'.
        mono: Whether to convert to mono.
        compression: HDF5 compression filter (e.g., 'gzip').
        num_workers: Number of worker processes for data loading.
    """
    if max_freq <= 0:
        max_freq = sr // 2

    # Create worker init function with config baked in via partial
    worker_init = partial(_worker_init_fn, quiet=quiet, log_file=log_file) if num_workers > 0 else None

    with h5.File(file_name, "a", libver="latest") as f, torch.no_grad():
        # Check for parameter mismatch with existing file
        if "sr" in f.attrs:
            existing_sr = f.attrs["sr"]
            existing_dtype = f.attrs.get("dtype", "unknown")
            existing_codec = f.attrs.get("codec", "unknown")

            mismatches = []
            if existing_sr != sr:
                mismatches.append(f"sr: {existing_sr} -> {sr}")
            if existing_dtype != dtype:
                mismatches.append(f"dtype: {existing_dtype} -> {dtype}")
            if existing_codec != codec:
                mismatches.append(f"codec: {existing_codec} -> {codec}")

            if mismatches:
                raise ValueError(
                    f"Parameter mismatch with existing HDF5 file!\n"
                    f"  Mismatches: {', '.join(mismatches)}\n"
                    f"  Delete the file to start fresh: rm {file_name}"
                )

        f.attrs["db_id"] = int(time.time())
        f.attrs["db_name"] = os.path.basename(file_name)
        f.attrs["max_freq"] = max_freq
        f.attrs["dtype"] = dtype
        f.attrs["sr"] = sr
        f.attrs["codec"] = codec

        for key, data_dict in data.items():
            working_dir = data_dict["working_dir"]

            try:
                grp: Group = f.create_group(key)
                existing_keys: set[str] = set()
            except ValueError:
                logger.info(f"Found existing group {key}")
                grp = cast("Group", f[key])
                existing_keys = set(grp.keys())
                logger.info(f"Found {len(existing_keys)} existing entries in group {key}")

            def make_ds_key(fn: str) -> str:
                """Generate HDF5 dataset key from filename (relative path with / -> _)."""
                rel_path = os.path.relpath(fn, working_dir) if working_dir else fn
                return rel_path.replace("/", "_")

            # Filter out files that already exist in the HDF5
            all_files = data_dict["files"]
            files_to_process = [fn for fn in all_files if make_ds_key(fn) not in existing_keys]
            skipped = len(all_files) - len(files_to_process)

            if skipped > 0:
                logger.info(f"Skipping {skipped} files already in HDF5, processing {len(files_to_process)} remaining")

            if not files_to_process:
                logger.info(f"All {len(all_files)} files already exist in group {key}, nothing to do.")
                continue

            dataset = PreProcessingDataset(
                sr=sr,
                file_names=files_to_process,
                dtype=dtype,
                codec=codec,
                mono=mono,
            )
            loader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_audio,
                worker_init_fn=worker_init,
            )

            n_samples = len(dataset)
            pbar = tqdm(
                enumerate(loader),
                total=n_samples,
                desc=f"Writing {key}",
                unit="files",
                file=sys.stderr,
            )
            written = 0
            for i, sample in pbar:
                if sample is None:
                    continue

                fn = sample["file_name"]
                ds_key = make_ds_key(fn)
                audio: np.ndarray = sample["data"]
                n_audio_samples = sample["n_samples"]

                if n_audio_samples < sr // 100:  # Should be at least 10 ms
                    logger.warning(f"Short audio {fn}: {audio.shape}.")

                logger.debug(f"Writing {ds_key} to {key} dataset.")

                if n_audio_samples == 0:
                    continue

                ds = grp.create_dataset(ds_key, data=audio, compression=compression)
                ds.attrs["n_samples"] = n_audio_samples
                written += 1
                del audio, sample

            pbar.close()
            logger.info(f"Group {key}: {written} written, {skipped} skipped ({len(all_files)} total).")


def collate_audio(batch):
    """Custom collate function to handle variable-length audio and errors."""
    sample = batch[0]
    if sample is None:
        return None
    return sample


class PreProcessingDataset(Dataset):
    """Dataset for preprocessing audio files using torchcodec."""

    def __init__(
        self,
        sr: int,
        file_names: list[str] | None = None,
        dtype: str = "float32",
        codec: str = "pcm",
        mono: bool = False,
    ):
        self.file_names = file_names or []
        self.sr = sr
        self.dtype = dtype.lower()
        self.codec = codec.lower()
        self.mono = mono

        if self.dtype not in ("float32", "int16"):
            raise ValueError(f"dtype must be 'float32' or 'int16', got '{dtype}'")
        if self.codec not in ("pcm", "flac", "vorbis"):
            raise ValueError(f"codec must be 'pcm', 'flac', or 'vorbis', got '{codec}'")

    def read(self, file: str) -> tuple[Tensor, int]:
        """Read and resample audio file using torchcodec.

        Returns:
            Tuple of (audio tensor [C, T], number of samples).
        """
        try:
            decoder = AudioDecoder(file, sample_rate=self.sr)
            samples = decoder.get_all_samples()
            x = samples.data  # Shape: [num_channels, num_samples], dtype: float32, range [-1, 1]

            if self.mono and x.shape[0] > 1:
                x = x.mean(dim=0, keepdim=True)

            if x.dim() == 1:
                x = x.unsqueeze(0)

            return x, x.shape[1]
        except Exception as e:
            logger.error(f"Failed to read {file}: {e}")
            raise

    def __getitem__(self, index: int) -> dict | None:
        fn = self.file_names[index]
        logger.debug(f"Reading audio file {fn}")

        try:
            x, n_samples = self.read(fn)
        except Exception as e:
            logger.error(f"Skipping {fn}: {e}")
            return None

        if x.shape[0] > 16:
            logger.warning(f"Unexpected channel count in {fn}: {x.shape}")

        # Encode to target format
        audio_data = encode(x, self.sr, self.codec, self.dtype)

        return {"file_name": fn, "data": audio_data, "n_samples": n_samples}

    def __len__(self) -> int:
        return len(self.file_names)


def encode(x: Tensor, sr: int, codec: str, dtype: str) -> np.ndarray:
    """Encode audio tensor to the specified format.

    Args:
        x: Audio tensor [C, T], float32, normalized to [-1, 1].
        sr: Sample rate.
        codec: 'pcm', 'flac', or 'vorbis'.
        dtype: 'float32' or 'int16' (only used for pcm).

    Returns:
        Numpy array containing the encoded audio data.
    """
    if codec == "vorbis":
        encoder = AudioEncoder(x, sample_rate=sr)
        encoded = encoder.to_tensor(format="ogg")
        return encoded.numpy()

    elif codec == "flac":
        encoder = AudioEncoder(x, sample_rate=sr)
        encoded = encoder.to_tensor(format="flac")
        return encoded.numpy()

    elif codec == "pcm":
        if dtype == "int16":
            # Convert from float32 [-1, 1] to int16
            x_int16 = (x * 32767).clamp(-32768, 32767).to(torch.int16)
            return x_int16.numpy()
        else:
            return x.numpy()

    else:
        raise NotImplementedError(f"Codec '{codec}' not supported.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio data for DeepFilterNet training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "type",
        type=str,
        choices=["speech", "noise", "noisy", "rir"],
        help="Dataset type.",
    )
    parser.add_argument(
        "audio_files",
        type=str,
        help="Text file containing audio file paths, one per line.",
    )
    parser.add_argument(
        "hdf5_db",
        type=str,
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--max_freq",
        type=int,
        default=-1,
        help="Maximum frequency to consider during training. "
        "Useful for upsampled signals with no high-frequency content.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="Target sample rate (audio will be resampled if needed).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="int16",
        choices=["float32", "int16"],
        help="Storage dtype for PCM codec.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="pcm",
        choices=["pcm", "flac", "vorbis"],
        help="Audio codec for storage.",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Convert multi-channel audio to mono.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="HDF5 compression filter (e.g., 'gzip').",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path. If not specified, logs to /tmp/prepare_data.log",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output (logs still written to file).",
    )
    args = parser.parse_args()

    if not args.hdf5_db.endswith(".hdf5"):
        args.hdf5_db += ".hdf5"

    log_file = args.log_file or "/tmp/prepare_data.log"
    init_logger(log_file)
    if args.quiet:
        logger.remove()
        logger.add(log_file, level="DEBUG")

    logger.info(f"Preparing {args.type} dataset: {args.audio_files} -> {args.hdf5_db}")
    logger.info(f"Settings: sr={args.sr}, dtype={args.dtype}, codec={args.codec}, mono={args.mono}")
    logger.info(f"Log file: {log_file}")

    data = {
        args.type: {"working_dir": None, "files": []},
    }

    with open(args.audio_files) as f:
        working_dir = os.path.dirname(args.audio_files)
        data[args.type]["working_dir"] = working_dir
        logger.info(f"Using working directory: {working_dir}")

        # Read all lines first to get count for progress bar
        lines = f.readlines()
        logger.info(f"Found {len(lines)} entries in file list.")

        check_file_fn = partial(_check_file, working_dir=working_dir)
        with Pool(max(args.num_workers, 1)) as p:
            results = list(
                tqdm(
                    p.imap(check_file_fn, lines, chunksize=100),
                    total=len(lines),
                    desc="Validating files",
                    unit="files",
                    file=sys.stderr,
                )
            )
            files = [r for r in results if r]

        data[args.type]["files"] = files
        logger.info(f"Validated {len(files)} audio files.")

    write_to_h5(
        file_name=args.hdf5_db,
        data=data,
        sr=args.sr,
        max_freq=args.max_freq,
        dtype=args.dtype,
        codec=args.codec.lower(),
        mono=args.mono,
        compression=args.compression,
        num_workers=args.num_workers,
        quiet=args.quiet,
        log_file=log_file,
    )

    logger.info(f"Done. Output: {args.hdf5_db}")


if __name__ == "__main__":
    main()
