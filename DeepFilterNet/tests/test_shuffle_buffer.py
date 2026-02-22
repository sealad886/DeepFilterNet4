"""Tests for PrefetchDataLoader shuffle_buffer_size feature."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest


@dataclass
class _FakeConfig:
    seed: int = 42


class _FakeSample:
    def __init__(self, idx: int):
        shape = (1, 1)
        self.noisy_spec = np.complex64(idx + np.zeros(shape))
        self.clean_spec = np.complex64(idx + np.zeros(shape))
        self.feat_erb = np.float32(idx + np.zeros(shape))
        self.feat_spec = np.float32(idx + np.zeros(shape))
        self.snr = float(idx)


class _FakeDataset:
    """Minimal dataset stub that returns deterministic samples by index."""

    def __init__(self, size: int, seed: int = 42):
        self.config = _FakeConfig(seed=seed)
        self._epoch = 0
        self._size = size
        self._samples = [_FakeSample(i) for i in range(size)]

    def __len__(self) -> int:
        return self._size

    def get_sample(self, idx: int) -> Optional[_FakeSample]:
        if 0 <= idx < self._size:
            return self._samples[idx]
        return None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


def _collect_snr_order(loader) -> List[float]:
    return [float(b["snr"].tolist()[0]) for b in loader]


class TestShuffleBufferStrictOrdering:
    """shuffle_buffer_size=0 preserves strict sequential order."""

    def test_strict_order_preserved(self):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        ds = _FakeDataset(size=10, seed=42)
        loader = PrefetchDataLoader(ds, batch_size=1, num_workers=1, shuffle_buffer_size=0)
        order = _collect_snr_order(loader)
        assert order == list(range(10))

    def test_strict_order_batch_size_2(self):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        ds = _FakeDataset(size=10, seed=42)
        loader = PrefetchDataLoader(ds, batch_size=2, num_workers=1, shuffle_buffer_size=0)
        snrs = []
        for b in loader:
            snrs.extend(b["snr"].tolist())
        assert snrs == list(range(10))


class TestShuffleBufferNoDataLoss:
    """All batches are yielded regardless of shuffle_buffer_size."""

    @pytest.mark.parametrize("buf_size", [1, 3, 5, 10, 20])
    def test_all_batches_yielded(self, buf_size: int):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        ds = _FakeDataset(size=10, seed=42)
        loader = PrefetchDataLoader(ds, batch_size=1, num_workers=1, shuffle_buffer_size=buf_size)
        order = _collect_snr_order(loader)
        assert sorted(order) == list(range(10)), f"Data loss with buffer={buf_size}"

    def test_no_data_loss_batch_size_2(self):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        ds = _FakeDataset(size=10, seed=42)
        loader = PrefetchDataLoader(ds, batch_size=2, num_workers=1, shuffle_buffer_size=4)
        snrs = []
        for b in loader:
            snrs.extend(b["snr"].tolist())
        assert sorted(snrs) == list(range(10))


class TestShuffleBufferReordering:
    """With a large enough buffer, order should differ from strict mode."""

    def test_order_differs_from_strict(self):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        ds = _FakeDataset(size=20, seed=42)
        strict_loader = PrefetchDataLoader(ds, batch_size=1, num_workers=1, shuffle_buffer_size=0)
        strict_order = _collect_snr_order(strict_loader)

        ds.set_epoch(0)
        shuffled_loader = PrefetchDataLoader(ds, batch_size=1, num_workers=1, shuffle_buffer_size=10)
        shuffled_order = _collect_snr_order(shuffled_loader)

        assert sorted(shuffled_order) == sorted(strict_order)
        assert shuffled_order != strict_order, "Shuffle buffer should reorder batches"


class TestShuffleBufferDeterminism:
    """Same seed + epoch produces the same shuffle order."""

    def test_deterministic_across_runs(self):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        orders = []
        for _ in range(3):
            ds = _FakeDataset(size=15, seed=99)
            loader = PrefetchDataLoader(ds, batch_size=1, num_workers=1, shuffle_buffer_size=5)
            orders.append(_collect_snr_order(loader))

        assert orders[0] == orders[1] == orders[2]

    def test_different_epochs_differ(self):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        ds1 = _FakeDataset(size=15, seed=99)
        ds1.set_epoch(0)
        loader1 = PrefetchDataLoader(ds1, batch_size=1, num_workers=1, shuffle_buffer_size=5)
        order1 = _collect_snr_order(loader1)

        ds2 = _FakeDataset(size=15, seed=99)
        ds2.set_epoch(1)
        loader2 = PrefetchDataLoader(ds2, batch_size=1, num_workers=1, shuffle_buffer_size=5)
        order2 = _collect_snr_order(loader2)

        assert sorted(order1) == sorted(order2)
        assert order1 != order2, "Different epochs should produce different orders"


class TestShuffleBufferEdgeCases:
    """Edge cases: buffer_size=1, fewer batches than buffer, empty dataset."""

    def test_buffer_size_1_like_strict(self):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        ds = _FakeDataset(size=10, seed=42)
        loader = PrefetchDataLoader(ds, batch_size=1, num_workers=1, shuffle_buffer_size=1)
        order = _collect_snr_order(loader)
        assert order == list(range(10)), "Buffer size 1 should behave like strict ordering"

    def test_fewer_batches_than_buffer(self):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        ds = _FakeDataset(size=3, seed=42)
        loader = PrefetchDataLoader(ds, batch_size=1, num_workers=1, shuffle_buffer_size=100)
        order = _collect_snr_order(loader)
        assert sorted(order) == [0.0, 1.0, 2.0]

    def test_empty_dataset(self):
        from df_mlx.dynamic_dataset import PrefetchDataLoader

        ds = _FakeDataset(size=0, seed=42)
        loader = PrefetchDataLoader(ds, batch_size=1, num_workers=1, shuffle_buffer_size=5, strict_failures=False)
        order = _collect_snr_order(loader)
        assert order == []
