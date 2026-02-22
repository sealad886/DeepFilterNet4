from pathlib import Path

from df_mlx import build_audio_cache, dynamic_dataset, generate_file_lists, prepare_data
from df_mlx.file_lists import read_file_list


def _write_list(path: Path, content: str) -> str:
    path.write_text(content)
    return str(path)


def test_prepare_and_dynamic_readers_preserve_tab_parsing(tmp_path: Path) -> None:
    file_list = _write_list(
        tmp_path / "with_tabs.txt",
        "# comment\n" "\n" "/a/clean.wav\t3.21\n" "/b/noise.flac\n",
    )

    expected = ["/a/clean.wav", "/b/noise.flac"]
    assert prepare_data.read_file_list(file_list) == expected
    assert dynamic_dataset.read_file_list(file_list) == expected


def test_build_audio_cache_reader_preserves_exists_filtering(tmp_path: Path, capsys) -> None:
    existing = tmp_path / "exists.wav"
    existing.write_text("x")
    missing = tmp_path / "missing.wav"
    file_list = _write_list(tmp_path / "cache_list.txt", f"{existing}\n{missing}\n")

    files = build_audio_cache.read_file_list(file_list)
    captured = capsys.readouterr()

    assert files == [str(existing)]
    assert f"Warning: File not found: {missing}" in captured.out


def test_generate_file_lists_reader_warns_on_missing_list(tmp_path: Path, capsys) -> None:
    missing_list = tmp_path / "does_not_exist.txt"
    files = generate_file_lists.read_file_list(str(missing_list))
    captured = capsys.readouterr()

    assert files == []
    assert f"Warning: File list not found: {missing_list}" in captured.out


def test_canonical_reader_supports_all_policy_combinations(tmp_path: Path) -> None:
    existing = tmp_path / "audio.wav"
    existing.write_text("x")
    file_list = _write_list(
        tmp_path / "mixed.txt",
        f"{existing}\t1.0\n" f"{tmp_path / 'missing.wav'}\n" "# ignored\n",
    )

    files = read_file_list(file_list, split_tab=True, check_exists=True)
    assert files == [str(existing)]
