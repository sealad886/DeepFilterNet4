import math

import numpy as np

from df_mlx.benchmark_common import (
    batch_size_from_batch,
    get_chip_name,
    get_gpu_cores,
    get_memory_gb,
    safe_percentile,
)


def test_safe_percentile_matches_numpy_and_handles_empty() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert safe_percentile(values, 95) == float(np.percentile(values, 95))
    assert math.isnan(safe_percentile([], 95))


def test_batch_size_from_batch_prefers_snr_and_falls_back_to_first_tensor() -> None:
    with_snr = {
        "noisy_real": np.zeros((2, 3, 4), dtype=np.float32),
        "snr": np.zeros((7,), dtype=np.float32),
    }
    no_snr = {
        "feat_erb": np.zeros((5, 10, 32), dtype=np.float32),
        "feat_spec": np.zeros((5, 10, 96, 2), dtype=np.float32),
    }

    assert batch_size_from_batch(with_snr) == 7
    assert batch_size_from_batch(no_snr) == 5


def test_get_chip_name_returns_nonempty_string() -> None:
    name = get_chip_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_get_gpu_cores_returns_int() -> None:
    cores = get_gpu_cores()
    assert isinstance(cores, int)


def test_get_memory_gb_returns_int() -> None:
    mem = get_memory_gb()
    assert isinstance(mem, int)
