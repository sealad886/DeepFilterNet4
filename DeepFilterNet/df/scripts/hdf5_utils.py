import io

import numpy as np
import torchaudio as ta
from torch import Tensor


def load_encoded(buffer: np.ndarray, codec: str) -> Tensor:
    """Decode audio stored as encoded bytes in an HDF5 dataset."""
    # In some rare cases, torchaudio fails to fully decode vorbis resulting in a shorter signal.
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format=codec.lower())
    return wav
