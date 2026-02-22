"""Tests for SyncMode enum and throughput vs diagnostic mode partitioning."""

import json

import pytest

from df_mlx.run_config import SYNC_MODE_EVAL_FREQUENCY, SyncMode


class TestSyncModeEnum:
    def test_values(self):
        assert SyncMode.FAST.value == "fast"
        assert SyncMode.NORMAL.value == "normal"
        assert SyncMode.DEBUG.value == "debug"
        assert SyncMode.PROFILE.value == "profile"

    def test_from_string(self):
        assert SyncMode("fast") == SyncMode.FAST
        assert SyncMode("normal") == SyncMode.NORMAL

    def test_string_comparison(self):
        """SyncMode must compare equal to plain strings (str base class)."""
        assert SyncMode.FAST == "fast"
        assert SyncMode.NORMAL != "fast"

    def test_emit_detailed_metrics_fast(self):
        assert not SyncMode.FAST.emit_detailed_metrics

    def test_emit_detailed_metrics_normal(self):
        assert SyncMode.NORMAL.emit_detailed_metrics

    def test_emit_detailed_metrics_debug(self):
        assert SyncMode.DEBUG.emit_detailed_metrics

    def test_emit_detailed_metrics_profile(self):
        assert SyncMode.PROFILE.emit_detailed_metrics

    def test_emit_snr_buckets_fast(self):
        assert not SyncMode.FAST.emit_snr_buckets

    def test_emit_snr_buckets_normal(self):
        assert SyncMode.NORMAL.emit_snr_buckets

    def test_emit_mask_stats_fast(self):
        assert not SyncMode.FAST.emit_mask_stats

    def test_emit_mask_stats_normal(self):
        assert SyncMode.NORMAL.emit_mask_stats

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            SyncMode("invalid")

    def test_all_modes_in_eval_frequency(self):
        for mode in SyncMode:
            assert mode.value in SYNC_MODE_EVAL_FREQUENCY

    def test_json_serializable(self):
        data = {"mode": SyncMode.FAST}
        result = json.dumps(data)
        assert '"fast"' in result
