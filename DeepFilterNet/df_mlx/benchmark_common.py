"""Shared helpers for df_mlx benchmark entrypoints.

This module centralizes parsing/validation and dataset construction logic
used by benchmark_pipeline.py and benchmark_train_step.py.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Sequence

import numpy as np

from df_mlx.dynamic_dataset import DatasetConfig, DynamicDataset, read_file_list


def parse_csv_tokens(value: str) -> List[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_int_list(value: str) -> List[int]:
    values: List[int] = []
    for token in parse_csv_tokens(value):
        values.append(int(token))
    if not values:
        raise ValueError("At least one value is required")
    return values


def parse_float_list(value: str) -> List[float]:
    values: List[float] = []
    for token in parse_csv_tokens(value):
        values.append(float(token))
    if not values:
        raise ValueError("At least one value is required")
    return values


def parse_bool_list(value: str) -> List[bool]:
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    values: List[bool] = []
    for token in parse_csv_tokens(value):
        lowered = token.lower()
        if lowered in truthy:
            values.append(True)
        elif lowered in falsy:
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean value '{token}'. Use one of {sorted(truthy | falsy)}")
    if not values:
        raise ValueError("At least one boolean value is required")
    return values


def parse_backend_list(value: str) -> List[str]:
    valid = {"prefetch", "mlx_stream"}
    backends = parse_csv_tokens(value)
    if not backends:
        raise ValueError("At least one backend is required")
    invalid = [b for b in backends if b not in valid]
    if invalid:
        raise ValueError(f"Invalid backends: {invalid}. Valid values: {sorted(valid)}")
    return backends


def parse_split_list(value: str) -> List[str]:
    valid = {"train", "valid", "test"}
    splits = parse_csv_tokens(value)
    if not splits:
        raise ValueError("At least one split is required")
    invalid = [split for split in splits if split not in valid]
    if invalid:
        raise ValueError(f"Invalid split values: {invalid}. Valid values: {sorted(valid)}")
    return splits


def require_min(name: str, values: Sequence[int], minimum: int) -> None:
    invalid = [value for value in values if value < minimum]
    if invalid:
        raise ValueError(f"{name} must be >= {minimum}, got {invalid}")


def require_positive_float(name: str, values: Sequence[float]) -> None:
    invalid = [value for value in values if value <= 0.0]
    if invalid:
        raise ValueError(f"{name} must be > 0, got {invalid}")


def safe_percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(values, q))


def batch_size_from_batch(batch: Dict[str, Any]) -> int:
    snr = batch.get("snr")
    if snr is not None:
        return int(snr.shape[0])
    first = next(iter(batch.values()))
    return int(first.shape[0])


def load_source_lists(args: argparse.Namespace) -> Dict[str, List[str]]:
    if args.cache_dir is None and (args.speech_list is None or args.noise_list is None):
        raise ValueError("Provide either --cache-dir or both --speech-list and --noise-list")

    source_lists: Dict[str, List[str]] = {"speech": [], "noise": [], "rir": []}
    if args.speech_list:
        source_lists["speech"] = read_file_list(args.speech_list)
    if args.noise_list:
        source_lists["noise"] = read_file_list(args.noise_list)
    if args.rir_list:
        source_lists["rir"] = read_file_list(args.rir_list)
    return source_lists


def build_dataset(args: argparse.Namespace, case: Any, source_lists: Dict[str, List[str]]) -> DynamicDataset:
    config = DatasetConfig(
        cache_dir=args.cache_dir,
        speech_files=source_lists["speech"],
        noise_files=source_lists["noise"],
        rir_files=source_lists["rir"],
        sample_rate=case.sample_rate,
        segment_length=case.segment_length,
        fft_size=case.fft_size,
        hop_size=case.hop_size,
        nb_erb=case.nb_erb,
        nb_df=case.nb_df,
        seed=case.seed,
    )
    dataset = DynamicDataset(config)
    dataset.set_split(case.split)
    dataset.set_epoch(case.epoch)
    return dataset
