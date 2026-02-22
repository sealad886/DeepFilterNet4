from argparse import Namespace

import pytest

from df_mlx.benchmark_pipeline import (
    _build_benchmark_cases,
    parse_backend_list,
    parse_bool_list,
    parse_int_list,
    parse_split_list,
)


def _matrix_args(**overrides):
    base = {
        "backends": ["prefetch", "mlx_stream"],
        "split": ["train"],
        "epoch": [0],
        "workers": [1, 2],
        "batch_size": [8],
        "batches": [100],
        "warmup_batches": [5],
        "repeats": [2],
        "sync_arrays": [True],
        "sample_rate": [48000],
        "segment_length": [5.0],
        "fft_size": [960],
        "hop_size": [480],
        "nb_erb": [32],
        "nb_df": [96],
        "seed": [42],
        "prefetch_factor": [2, 4],
        "prefetch_size": [8, 16],
    }
    base.update(overrides)
    return Namespace(**base)


def test_parse_backend_split_and_bool_lists():
    assert parse_backend_list("prefetch,mlx_stream") == ["prefetch", "mlx_stream"]
    assert parse_split_list("train,valid,test") == ["train", "valid", "test"]
    assert parse_bool_list("true,false,1,0") == [True, False, True, False]


def test_parse_bool_list_rejects_invalid_value():
    with pytest.raises(ValueError):
        parse_bool_list("true,maybe")


def test_build_benchmark_cases_expands_matrix_only_on_relevant_prefetch_axis():
    cases = _build_benchmark_cases(_matrix_args())
    # 2 backends * 2 workers * 2 backend-specific prefetch values.
    assert len(cases) == 8

    prefetch_cases = [c for c in cases if c.backend == "prefetch"]
    stream_cases = [c for c in cases if c.backend == "mlx_stream"]

    assert {c.prefetch_factor for c in prefetch_cases} == {2, 4}
    assert {c.prefetch_size for c in prefetch_cases} == {1}
    assert {c.prefetch_size for c in stream_cases} == {8, 16}
    assert {c.prefetch_factor for c in stream_cases} == {1}


def test_build_benchmark_cases_validates_positive_values():
    with pytest.raises(ValueError, match="batch-size must be >= 1"):
        _build_benchmark_cases(_matrix_args(batch_size=[0]))

    with pytest.raises(ValueError, match="segment-length must be > 0"):
        _build_benchmark_cases(_matrix_args(segment_length=[0.0]))


def test_parse_int_list_requires_values():
    with pytest.raises(ValueError):
        parse_int_list(" ,, ")
