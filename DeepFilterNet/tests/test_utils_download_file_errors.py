import pytest

from df import utils


class _DummyResponse:
    def __init__(self, status_code: int = 500, reason: str = "boom"):
        self.status_code = status_code
        self.reason = reason
        self.raw = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_file_raises_runtime_error_on_http_failure(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        return _DummyResponse(status_code=500, reason="server error")

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(RuntimeError, match="Error downloading file"):
        utils.download_file("https://example.com/model.zip", str(tmp_path), extract=False)
