"""Validated source-audio loading for the TactEar core prototype."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class AudioData:
    """Mono floating-point audio ready for downstream processing."""

    waveform: np.ndarray
    sample_rate: int


def load_audio(path: str | Path) -> AudioData:
    """Decode *path* and return validated mono floating-point audio.

    Multi-channel source audio is averaged into a single source waveform. This
    is intentionally distinct from later simulated headset stereo signals.
    """

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")

    try:
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
    except sf.LibsndfileError as exc:
        raise ValueError(f"Could not decode audio file: {audio_path}") from exc

    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError(
            f"Decoded audio must be one- or two-dimensional, got {waveform.ndim} dimensions"
        )
    if sample_rate <= 0:
        raise ValueError(f"Audio sample rate must be positive, got {sample_rate}")
    if waveform.size == 0:
        raise ValueError("Decoded audio waveform is empty")
    if not np.issubdtype(waveform.dtype, np.floating):
        waveform = waveform.astype(np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError("Decoded audio waveform contains non-finite samples")

    return AudioData(waveform=waveform, sample_rate=sample_rate)
