"""Deterministic simulated headset microphone signals for TactEar."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from tactear.audio import AudioData


DEFAULT_MIC_SPACING_M = 0.18
DEFAULT_SPEED_OF_SOUND_M_S = 343.0


@dataclass
class SimulatedStereo:
    """Simulated left/right microphone audio with explicit ground truth.

    ``injected_delay_seconds`` is the signed left-arrival-time minus
    right-arrival-time difference. A positive value means the right microphone
    receives the sound first; a negative value means the left microphone does.
    """

    left: np.ndarray
    right: np.ndarray
    sample_rate: int
    ground_truth_angle_deg: float
    injected_delay_seconds: float
    injected_delay_samples: int
    hardware_simulated: bool = True


def simulate_stereo(
    audio: AudioData,
    angle_deg: float,
    mic_spacing_m: float = DEFAULT_MIC_SPACING_M,
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S,
) -> SimulatedStereo:
    """Generate two zero-padded, equal-length simulated headset channels.

    The far-field model uses ``tdoa = (spacing / sound_speed) * sin(angle)``.
    Here TDOA is ``left arrival time - right arrival time`` and the convention
    is 0 degrees front, negative angles left, and positive angles right. Thus,
    a negative angle makes the left channel lead and a positive angle makes the
    right channel lead. Integer-sample quantization is intentional for this
    prototype; delayed tails are trimmed rather than wrapped around.
    """

    _validate_audio(audio)
    _validate_finite_real("angle_deg", angle_deg)
    _validate_finite_real("mic_spacing_m", mic_spacing_m, positive=True)
    _validate_finite_real("speed_of_sound_m_s", speed_of_sound_m_s, positive=True)

    if not -90.0 <= angle_deg <= 90.0:
        raise ValueError("angle_deg must be between -90 and 90 degrees")

    injected_delay_seconds = (
        mic_spacing_m / speed_of_sound_m_s * math.sin(math.radians(angle_deg))
    )
    injected_delay_samples = round(injected_delay_seconds * audio.sample_rate)

    max_delay_samples = round(
        mic_spacing_m / speed_of_sound_m_s * audio.sample_rate
    )
    if abs(injected_delay_samples) > max_delay_samples:
        raise ValueError("Requested delay exceeds the physical microphone limit")

    if injected_delay_samples > 0:
        left = _delay_with_zeros(audio.waveform, injected_delay_samples)
        right = audio.waveform.copy()
    elif injected_delay_samples < 0:
        left = audio.waveform.copy()
        right = _delay_with_zeros(audio.waveform, -injected_delay_samples)
    else:
        left = audio.waveform.copy()
        right = audio.waveform.copy()

    return SimulatedStereo(
        left=left,
        right=right,
        sample_rate=audio.sample_rate,
        ground_truth_angle_deg=angle_deg,
        injected_delay_seconds=injected_delay_seconds,
        injected_delay_samples=injected_delay_samples,
    )


def _delay_with_zeros(waveform: np.ndarray, delay_samples: int) -> np.ndarray:
    """Delay a channel without circular wraparound, preserving its length."""

    delayed = np.zeros_like(waveform)
    if delay_samples < waveform.size:
        delayed[delay_samples:] = waveform[: waveform.size - delay_samples]
    return delayed


def _validate_audio(audio: AudioData) -> None:
    if not isinstance(audio, AudioData):
        raise TypeError("audio must be an AudioData instance")
    if not isinstance(audio.waveform, np.ndarray):
        raise ValueError("Audio waveform must be a NumPy array")
    if audio.waveform.ndim != 1 or audio.waveform.size == 0:
        raise ValueError("Audio waveform must be a non-empty mono array")
    if not np.issubdtype(audio.waveform.dtype, np.floating):
        raise ValueError("Audio waveform must use a floating-point dtype")
    if not np.isfinite(audio.waveform).all():
        raise ValueError("Audio waveform contains non-finite samples")
    if not isinstance(audio.sample_rate, (int, np.integer)) or audio.sample_rate <= 0:
        raise ValueError("Audio sample rate must be positive")


def _validate_finite_real(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
