import pytest

libdfdata_pkg = pytest.importorskip("libdfdata")
if not hasattr(libdfdata_pkg, "PytorchDataLoader"):
    pytest.skip("torch dataloader not available in this environment", allow_module_level=True)

PytorchDataLoader = libdfdata_pkg.PytorchDataLoader
DataLoaderTimeoutError = libdfdata_pkg.DataLoaderTimeoutError


def test_timeout_path_raises_exception_instead_of_exit():
    obj = object.__new__(PytorchDataLoader)

    class FakeLoader:
        def get_batch(self):
            raise RuntimeError("DF dataloader error: TimeoutError")

        def cleanup(self):
            return None

    obj.loader = FakeLoader()
    obj.idx = 0
    obj.cleanup_pin_memory_thread = lambda: None

    queue_like = obj._get_worker_queue_dummy()
    with pytest.raises(DataLoaderTimeoutError, match="TimeoutError"):
        queue_like.get()
