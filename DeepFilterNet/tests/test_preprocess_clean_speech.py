from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_SCRIPT = REPO_ROOT / "scripts" / "datasets" / "preprocess_clean_speech.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("preprocess_clean_speech_test_module", PREPROCESS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_resumes_existing_outputs_by_default(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    base_dir = tmp_path / "raw"
    output_root = tmp_path / "preprocessed"
    file_list = tmp_path / "clean.txt"
    output_list = tmp_path / "clean.preprocessed.txt"

    source_a = base_dir / "speaker_a" / "a.wav"
    source_b = base_dir / "speaker_b" / "b.wav"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    file_list.write_text(f"{source_a}\n{source_b}\n", encoding="utf-8")

    target_a = output_root / "speaker_a" / "a.wav"
    target_b = output_root / "speaker_b" / "b.wav"
    target_a.parent.mkdir(parents=True, exist_ok=True)
    target_a.write_bytes(b"already-done")

    seen_dataset_files: list[str] = []
    seen_loader_kwargs: dict[str, object] = {}
    progress_config: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, files: list[str], sr: int):
            seen_dataset_files[:] = files
            self.files = files

        def __len__(self) -> int:
            return len(self.files)

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            seen_loader_kwargs.update(kwargs)
            self.dataset = dataset

        def __iter__(self):
            for file in self.dataset.files:
                yield [file], torch.zeros(1, 8), torch.tensor([16_000])

    class FakeParams:
        sr = 16_000

    saved_targets: list[str] = []
    fake_durations = {
        source_a.resolve(): 1.25,
        source_b.resolve(): 2.75,
    }

    def fake_save_audio(path: str, audio, sr: int, output_dir=None, suffix=None, log=False):
        saved_targets.append(path)
        Path(path).write_bytes(b"enhanced")

    class FakeProgress:
        def __init__(self, iterable=None, **kwargs):
            progress_config.update(kwargs)
            self._iterable = [] if iterable is None else iterable
            self._updated = 0.0

        def __iter__(self):
            return iter(self._iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_postfix(self, value):
            progress_config["postfix"] = value

        def set_postfix_str(self, value, refresh=True):
            progress_config["postfix"] = value
            progress_config["postfix_refresh"] = refresh

        def update(self, amount):
            self._updated += amount
            progress_config["updated"] = self._updated

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            file_list=str(file_list),
            output_root=str(output_root),
            base_dir=str(base_dir),
            output_list=str(output_list),
            model_base_dir="DeepFilterNet3",
            device=None,
            num_workers=2,
            probe_workers=None,
            probe_cache=None,
            enhance_batch_size=None,
            overwrite=False,
            allow_non_speech_paths=False,
        ),
    )
    monkeypatch.setattr(module, "resolve_ffprobe_bin", lambda: "ffprobe")
    monkeypatch.setattr(
        module,
        "probe_audio_durations",
        lambda paths, ffprobe_bin, num_workers, cache_path=None: {path: fake_durations[path] for path in paths},
    )
    monkeypatch.setattr(module, "init_df", lambda **kwargs: (object(), object(), None, None))
    monkeypatch.setattr(module, "ModelParams", lambda: FakeParams())
    monkeypatch.setattr(module, "AudioDataset", FakeDataset)
    monkeypatch.setattr(module, "DataLoader", FakeLoader)
    monkeypatch.setattr(module, "enhance", lambda model, df_state, audio, pad, device: audio + 1)
    monkeypatch.setattr(module, "resample", lambda audio, orig_sr, new_sr: audio)
    monkeypatch.setattr(module, "save_audio", fake_save_audio)
    monkeypatch.setattr(module, "tqdm", FakeProgress)

    rc = module.main()

    assert rc == 0
    assert seen_dataset_files == [str(source_b.resolve())]
    assert seen_loader_kwargs["persistent_workers"] is True
    assert seen_loader_kwargs["prefetch_factor"] == 2
    assert target_a.read_bytes() == b"already-done"
    assert target_b.exists()
    assert target_b.read_bytes() == b"enhanced"
    assert saved_targets == [str(module.build_temp_output_path(target_b))]
    output_list_lines = output_list.read_text(encoding="utf-8").splitlines()
    assert output_list_lines == [str(target_a), str(target_b)]
    assert progress_config["total"] == 2.75
    assert progress_config["initial"] == 0.0
    assert progress_config["updated"] == 2.75
    assert progress_config["bar_format"] == "{desc}: {percentage:3.0f}%|{bar}| {elapsed}{postfix}"
    postfix = str(progress_config["postfix"])
    assert "audio=2.8s/2.8s" in postfix
    assert "eta=00:00" in postfix
    assert "save=" in postfix


def test_module_prioritizes_repo_package_root_on_sys_path() -> None:
    module = _load_module()

    assert str(module.PACKAGE_ROOT) in sys.path


def test_main_fully_resumed_run_skips_ffprobe_and_backend_init(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    base_dir = tmp_path / "raw"
    output_root = tmp_path / "preprocessed"
    file_list = tmp_path / "clean.txt"
    output_list = tmp_path / "clean.preprocessed.txt"

    source_a = base_dir / "speaker_a" / "a.wav"
    source_b = base_dir / "speaker_b" / "b.wav"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    file_list.write_text(f"{source_a}\n{source_b}\n", encoding="utf-8")

    target_a = output_root / "speaker_a" / "a.wav"
    target_b = output_root / "speaker_b" / "b.wav"
    target_a.parent.mkdir(parents=True, exist_ok=True)
    target_b.parent.mkdir(parents=True, exist_ok=True)
    target_a.write_bytes(b"done-a")
    target_b.write_bytes(b"done-b")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            file_list=str(file_list),
            output_root=str(output_root),
            base_dir=str(base_dir),
            output_list=str(output_list),
            model_base_dir="DeepFilterNet3",
            device=None,
            num_workers=2,
            probe_workers=None,
            probe_cache=None,
            enhance_batch_size=None,
            overwrite=False,
            allow_non_speech_paths=False,
        ),
    )
    monkeypatch.setattr(
        module,
        "resolve_ffprobe_bin",
        lambda: (_ for _ in ()).throw(AssertionError("ffprobe should not run when resume is already complete")),
    )
    monkeypatch.setattr(
        module,
        "resolve_backend",
        lambda model_base_dir, requested_device: (_ for _ in ()).throw(
            AssertionError("backend init should not run when resume is already complete")
        ),
    )

    rc = module.main()

    assert rc == 0
    assert output_list.read_text(encoding="utf-8").splitlines() == [str(target_a), str(target_b)]


def test_main_rejects_colliding_output_paths(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    source_root_a = tmp_path / "external_a"
    source_root_b = tmp_path / "external_b"
    output_root = tmp_path / "preprocessed"
    base_dir = tmp_path / "raw"
    file_list = tmp_path / "clean.txt"
    output_list = tmp_path / "clean.preprocessed.txt"

    source_a = source_root_a / "speaker_a" / "shared.wav"
    source_b = source_root_b / "speaker_b" / "shared.wav"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    file_list.write_text(f"{source_a}\n{source_b}\n", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            file_list=str(file_list),
            output_root=str(output_root),
            base_dir=str(base_dir),
            output_list=str(output_list),
            model_base_dir="DeepFilterNet3",
            device=None,
            num_workers=0,
            probe_workers=None,
            probe_cache=None,
            enhance_batch_size=None,
            overwrite=False,
            allow_non_speech_paths=False,
        ),
    )

    try:
        module.main()
    except SystemExit as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected colliding output paths to be rejected")

    assert "Multiple source files map to the same preprocess output path" in message
    assert "shared.wav" in message


def test_resolve_backend_prefers_mlx_for_mlx_models_on_apple_silicon(monkeypatch) -> None:
    module = _load_module()
    mlx_backend = module.EnhanceBackend(name="mlx", sample_rate=48_000, enhance_audio=lambda audio: audio)

    monkeypatch.setattr(module, "running_on_apple_silicon", lambda: True)
    monkeypatch.setattr(module, "load_mlx_backend", lambda model_base_dir: mlx_backend)

    backend = module.resolve_backend("DeepFilterNet4-MLX", requested_device=None)

    assert backend is mlx_backend


def test_resolve_backend_prefers_mlx_for_dfnet3_mlx_on_apple_silicon(monkeypatch) -> None:
    module = _load_module()
    mlx_backend = module.EnhanceBackend(name="mlx", sample_rate=48_000, enhance_audio=lambda audio: audio)

    monkeypatch.setattr(module, "running_on_apple_silicon", lambda: True)
    monkeypatch.setattr(module, "load_mlx_backend", lambda model_base_dir: mlx_backend)

    backend = module.resolve_backend("DeepFilterNet3-MLX", requested_device=None)

    assert backend is mlx_backend


def test_resolve_backend_uses_torch_for_default_deepfilternet3(monkeypatch) -> None:
    module = _load_module()
    torch_backend = module.EnhanceBackend(name="torch", sample_rate=48_000, enhance_audio=lambda audio: audio)

    monkeypatch.setattr(module, "running_on_apple_silicon", lambda: True)
    monkeypatch.setattr(module, "load_torch_backend", lambda model_base_dir, requested_device: torch_backend)

    backend = module.resolve_backend("DeepFilterNet3", requested_device=None)

    assert backend is torch_backend


def test_resolve_backend_prefers_mlx_for_custom_model_dirs_on_apple_silicon(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    mlx_backend = module.EnhanceBackend(name="mlx", sample_rate=48_000, enhance_audio=lambda audio: audio)
    model_dir = tmp_path / "custom_model"
    model_dir.mkdir()

    monkeypatch.setattr(module, "running_on_apple_silicon", lambda: True)
    monkeypatch.setattr(module, "load_mlx_backend", lambda model_base_dir: mlx_backend)

    backend = module.resolve_backend(str(model_dir), requested_device=None)

    assert backend is mlx_backend


def test_resolve_backend_falls_back_to_torch_dfnet3_when_mlx_init_fails(monkeypatch) -> None:
    module = _load_module()
    torch_backend = module.EnhanceBackend(name="torch", sample_rate=48_000, enhance_audio=lambda audio: audio)
    seen: dict[str, str | None] = {}

    monkeypatch.setattr(module, "running_on_apple_silicon", lambda: True)

    def fake_load_mlx_backend(model_base_dir: str):
        raise RuntimeError(f"cannot load {model_base_dir}")

    def fake_load_torch_backend(model_base_dir: str, requested_device: str | None):
        seen["model_base_dir"] = model_base_dir
        seen["requested_device"] = requested_device
        return torch_backend

    monkeypatch.setattr(module, "load_mlx_backend", fake_load_mlx_backend)
    monkeypatch.setattr(module, "load_torch_backend", fake_load_torch_backend)

    backend = module.resolve_backend("DeepFilterNet3-MLX", requested_device=None)

    assert backend is torch_backend
    assert seen == {"model_base_dir": "DeepFilterNet3", "requested_device": None}


def test_load_mlx_backend_retries_after_resource_limit_and_clears_cache(monkeypatch) -> None:
    module = _load_module()
    clear_calls: list[str] = []
    run_calls = {"count": 0}

    class FakeMx:
        def array(self, value):
            return value

        def clear_cache(self):
            clear_calls.append("clear")

    fake_mlx_enhance = types.SimpleNamespace(
        mx=FakeMx(),
        load_model=lambda model_path, epoch: (object(), types.SimpleNamespace(sr=48_000), None, None),
    )

    def fake_run(mlx_enhance_mod, model, audio_mx, params):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")
        return torch.tensor([0.25, -0.5], dtype=torch.float32).numpy()

    gc_calls: list[str] = []
    monkeypatch.setattr(module, "_import_mlx_enhance_module", lambda: fake_mlx_enhance)
    monkeypatch.setattr(module, "_run_mlx_enhancement", fake_run)
    monkeypatch.setattr(module.gc, "collect", lambda: gc_calls.append("gc"))

    backend = module.load_mlx_backend("DeepFilterNet3-MLX")
    enhanced = backend.enhance_audio(torch.tensor([1.0, -1.0], dtype=torch.float32))

    assert torch.allclose(enhanced, torch.tensor([0.25, -0.5], dtype=torch.float32))
    assert run_calls["count"] == 2
    assert clear_calls == ["clear"]
    assert gc_calls == ["gc"]


def test_load_mlx_backend_periodically_clears_cache(monkeypatch) -> None:
    module = _load_module()
    clear_calls: list[str] = []

    class FakeMx:
        def array(self, value):
            return value

        def clear_cache(self):
            clear_calls.append("clear")

    fake_mlx_enhance = types.SimpleNamespace(
        mx=FakeMx(),
        load_model=lambda model_path, epoch: (object(), types.SimpleNamespace(sr=48_000), None, None),
    )

    monkeypatch.setattr(module, "_import_mlx_enhance_module", lambda: fake_mlx_enhance)
    monkeypatch.setattr(module, "_run_mlx_enhancement", lambda *args, **kwargs: torch.tensor([0.1]).numpy())
    monkeypatch.setattr(module, "MLX_CLEAR_CACHE_INTERVAL", 2)

    backend = module.load_mlx_backend("DeepFilterNet3-MLX")
    _ = backend.enhance_audio(torch.tensor([1.0], dtype=torch.float32))
    _ = backend.enhance_audio(torch.tensor([2.0], dtype=torch.float32))

    assert clear_calls == ["clear"]


def test_main_rejects_obvious_non_speech_sources_by_default(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    base_dir = tmp_path / "raw"
    output_root = tmp_path / "preprocessed"
    file_list = tmp_path / "clean.txt"
    output_list = tmp_path / "clean.preprocessed.txt"

    noise_source = base_dir / "musan" / "noise" / "noise.wav"
    noise_source.parent.mkdir(parents=True, exist_ok=True)
    noise_source.write_bytes(b"noise")
    file_list.write_text(f"{noise_source}\n", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            file_list=str(file_list),
            output_root=str(output_root),
            base_dir=str(base_dir),
            output_list=str(output_list),
            model_base_dir="DeepFilterNet3",
            device=None,
            num_workers=0,
            probe_workers=None,
            probe_cache=None,
            enhance_batch_size=None,
            overwrite=False,
            allow_non_speech_paths=False,
        ),
    )
    monkeypatch.setattr(module, "resolve_ffprobe_bin", lambda: "ffprobe")
    monkeypatch.setattr(
        module,
        "probe_audio_durations",
        lambda paths, ffprobe_bin, num_workers, cache_path=None: {noise_source.resolve(): 0.5},
    )

    try:
        module.main()
    except SystemExit as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected obvious non-speech input to be rejected")

    assert "Refusing to preprocess obvious non-speech inputs" in message
    assert str(noise_source.resolve()) in message


def test_save_enhanced_audio_atomically_cleans_partial_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    target = tmp_path / "out" / "sample.wav"
    temp_target = module.build_temp_output_path(target)

    def failing_save_audio(path: str, audio, sr: int, output_dir=None, suffix=None, log=False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"partial")
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "save_audio", failing_save_audio)
    monkeypatch.setattr(module, "resample", lambda audio, orig_sr, new_sr: audio)

    try:
        module.save_enhanced_audio_atomically(target, torch.zeros(1, 4), df_sr=16_000, orig_sr=16_000)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected save_enhanced_audio_atomically to raise")

    assert not temp_target.exists()
    assert not target.exists()


def test_build_temp_output_path_preserves_audio_extension(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "out" / "sample.wav"

    temp_target = module.build_temp_output_path(target)

    assert temp_target.name.startswith(".sample.partial.")
    assert temp_target.suffix == ".wav"


def test_build_progress_postfix_reports_rate_queue_and_stage_timings() -> None:
    module = _load_module()
    stats = module.PreprocessProgressStats(start_time=10.0)
    stats.enhance_count = 8
    stats.enhance_seconds = 2.4
    stats.processed_audio_seconds = 12.0
    stats.save_count = 6
    stats.save_seconds = 1.2
    stats.queue_high_water = 4

    postfix = module.build_progress_postfix(
        stats,
        inflight_saves=3,
        completed_audio_seconds=18.0,
        total_audio_seconds=30.0,
        now=14.0,
    )

    assert postfix == "audio=18.0s/30.0s, eta=00:04, rt=3.00x, save_q=3, enh=300ms, save=200ms"


def test_build_probe_postfix_reports_probe_rate_eta_audio_and_failures() -> None:
    module = _load_module()

    postfix = module.build_probe_postfix(
        completed_files=6,
        total_files=10,
        discovered_audio_seconds=12.4,
        failure_count=1,
        start_time=10.0,
        now=12.0,
    )

    assert postfix == "files=6/10, eta=00:01, probe=3.0/s, audio=12.4s, fail=1"


def test_probe_audio_durations_uses_requested_parallelism_and_updates_progress(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    sources = [tmp_path / f"clip_{idx}.wav" for idx in range(3)]
    for source in sources:
        source.write_bytes(b"wav")

    seen_executor: dict[str, int] = {}
    progress_config: dict[str, object] = {}

    class FakeProgress:
        def __init__(self, iterable=None, **kwargs):
            progress_config.update(kwargs)
            self._iterable = [] if iterable is None else iterable
            self._updated = 0

        def __iter__(self):
            return iter(self._iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_postfix_str(self, value, refresh=True):
            progress_config["postfix"] = value
            progress_config["postfix_refresh"] = refresh

        def update(self, amount):
            self._updated += amount
            progress_config["updated"] = self._updated

    class FakeExecutor:
        def __init__(self, max_workers: int):
            seen_executor["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = module.Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:  # pragma: no cover - defensive guard
                future.set_exception(exc)
            return future

    duration_map = {source: float(idx + 1) for idx, source in enumerate(sources)}

    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(module, "tqdm", FakeProgress)
    monkeypatch.setattr(module, "probe_audio_duration_seconds", lambda path, ffprobe_bin: duration_map[path])

    durations = module.probe_audio_durations(sources, "ffprobe", num_workers=7)

    assert durations == duration_map
    assert seen_executor["max_workers"] == 7
    assert progress_config["total"] == 3
    assert progress_config["unit"] == "file"
    assert progress_config["updated"] == 3
    assert "probe=" in str(progress_config["postfix"])
    assert "audio=6.0s" in str(progress_config["postfix"])


def test_probe_audio_durations_reuses_cache_for_unchanged_files(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    cached_source = tmp_path / "cached.wav"
    uncached_source = tmp_path / "uncached.wav"
    cached_source.write_bytes(b"cached")
    uncached_source.write_bytes(b"uncached")
    cache_path = tmp_path / "probe-cache.json"

    cached_entry = module._build_probe_cache_entry(cached_source, 1.5)
    module.write_probe_cache(cache_path, {cached_source: cached_entry})

    seen_probes: list[Path] = []

    class FakeExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = module.Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:  # pragma: no cover - defensive guard
                future.set_exception(exc)
            return future

    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        module,
        "probe_audio_duration_seconds",
        lambda path, ffprobe_bin: seen_probes.append(path) or 2.25,
    )

    durations = module.probe_audio_durations(
        [cached_source, uncached_source],
        "ffprobe",
        num_workers=3,
        cache_path=cache_path,
    )

    assert durations == {cached_source: 1.5, uncached_source: 2.25}
    assert seen_probes == [uncached_source]

    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache_payload["entries"][str(cached_source)]["duration_seconds"] == 1.5
    assert cache_payload["entries"][str(uncached_source)]["duration_seconds"] == 2.25


def test_probe_audio_durations_reprobes_when_file_stat_changes(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    source = tmp_path / "clip.wav"
    source.write_bytes(b"old")
    cache_path = tmp_path / "probe-cache.json"
    module.write_probe_cache(cache_path, {source: module._build_probe_cache_entry(source, 1.0)})

    source.write_bytes(b"new-content")
    seen_probes: list[Path] = []

    class FakeExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = module.Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:  # pragma: no cover - defensive guard
                future.set_exception(exc)
            return future

    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        module,
        "probe_audio_duration_seconds",
        lambda path, ffprobe_bin: seen_probes.append(path) or 3.5,
    )

    durations = module.probe_audio_durations([source], "ffprobe", num_workers=1, cache_path=cache_path)

    assert durations == {source: 3.5}
    assert seen_probes == [source]


def test_resolve_probe_cache_path_defaults_next_to_output_list(tmp_path: Path) -> None:
    module = _load_module()

    output_list = tmp_path / "clean_all.preprocessed.txt"

    cache_path = module.resolve_probe_cache_path(output_list, explicit_cache_path=None)

    assert cache_path == tmp_path / "clean_all.preprocessed.ffprobe-cache.json"


def test_choose_enhance_batch_size_defaults_to_mlx_batching() -> None:
    module = _load_module()

    assert module.choose_enhance_batch_size("mlx") == module.MLX_DEFAULT_ENHANCE_BATCH_SIZE
    assert module.choose_enhance_batch_size("torch") == 1
    assert module.choose_enhance_batch_size("mlx", override=8) == 8
    assert module.choose_enhance_batch_size("torch", override=2) == 2
    assert module.choose_enhance_batch_size("mlx", override=None) == module.MLX_DEFAULT_ENHANCE_BATCH_SIZE
    assert module.choose_enhance_batch_size("mlx", override=0) == 1


def test_enhance_audio_batch_pads_inputs_and_trims_outputs() -> None:
    module = _load_module()

    seen_shapes: list[tuple[int, ...]] = []

    def fake_enhance_audio(audio: torch.Tensor) -> torch.Tensor:
        seen_shapes.append(tuple(audio.shape))
        return audio + 2

    backend = module.EnhanceBackend(name="mlx", sample_rate=16_000, enhance_audio=fake_enhance_audio)
    audios = [torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[4.0, 5.0]])]

    enhanced, elapsed = module.enhance_audio_batch(backend, audios)

    assert seen_shapes == [(2, 3)]
    assert elapsed >= 0.0
    assert len(enhanced) == 2
    assert torch.equal(enhanced[0], torch.tensor([[3.0, 4.0, 5.0]]))
    assert torch.equal(enhanced[1], torch.tensor([[6.0, 7.0]]))
