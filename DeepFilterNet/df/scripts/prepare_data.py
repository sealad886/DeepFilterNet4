#!/usr/bin/env python3
"""Prepare audio data for DeepFilterNet training by converting to HDF5 format.

Uses torchcodec for efficient audio decoding and encoding with built-in resampling.
"""

import argparse
import os
import time
from multiprocessing import Pool
from typing import Optional

import h5py as h5
import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from df.logger import init_logger


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

    with h5.File(file_name, "a", libver="latest") as f, torch.no_grad():
        f.attrs["db_id"] = int(time.time())
        f.attrs["db_name"] = os.path.basename(file_name)
        f.attrs["max_freq"] = max_freq
        f.attrs["dtype"] = dtype
        f.attrs["sr"] = sr
        f.attrs["codec"] = codec

        for key, data_dict in data.items():
            try:
                grp = f.create_group(key)
            except ValueError:
                logger.info(f"Found existing group {key}")
                grp = f[key]

            dataset = PreProcessingDataset(
                sr=sr,
                file_names=data_dict["files"],
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
            )

            n_samples = len(dataset)
            for i, sample in enumerate(loader):
                if sample is None:
                    continue

                fn = os.path.relpath(sample["file_name"], data_dict["working_dir"])
                audio: np.ndarray = sample["data"]
                n_audio_samples = sample["n_samples"]

                if n_audio_samples < sr // 100:  # Should be at least 10 ms
                    logger.warning(f"Short audio {fn}: {audio.shape}.")

                progress = (i + 1) / n_samples * 100
                logger.info(f"{progress:5.1f}% | Writing {fn} to {key} dataset.")

                if n_audio_samples == 0:
                    continue

                ds_key = fn.replace("/", "_")
                if ds_key in grp:
                    logger.debug(f"Found dataset {ds_key}. Replacing.")
                    del grp[ds_key]

                ds = grp.create_dataset(ds_key, data=audio, compression=compression)
                ds.attrs["n_samples"] = n_audio_samples
                del audio, sample

            logger.info(f"Added {n_samples} samples to group {key}.")


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
    args = parser.parse_args()

    if not args.hdf5_db.endswith(".hdf5"):
        args.hdf5_db += ".hdf5"

    init_logger("/tmp/prepare_data.log")
    logger.info(f"Preparing {args.type} dataset: {args.audio_files} -> {args.hdf5_db}")
    logger.info(f"Settings: sr={args.sr}, dtype={args.dtype}, codec={args.codec}, mono={args.mono}")

    data = {
        args.type: {"working_dir": None, "files": []},
    }

    with open(args.audio_files) as f:
        working_dir = os.path.dirname(args.audio_files)
        data[args.type]["working_dir"] = working_dir
        logger.info(f"Using working directory: {working_dir}")

        def _check_file(file: str) -> str:
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

        with Pool(max(args.num_workers, 1)) as p:
            results = p.imap(_check_file, f, chunksize=100)
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
    )

    logger.info(f"Done. Output: {args.hdf5_db}")


if __name__ == "__main__":
    main()
