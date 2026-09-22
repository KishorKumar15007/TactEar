"""GCC-PHAT two-channel acoustic localization for TactEar."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from tactear.events import LocalizationResult
from tactear.simulator import DEFAULT_MIC_SPACING_M, DEFAULT_SPEED_OF_SOUND_M_S


def gcc_phat(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    max_tau: float | None = None,
) -> tuple[float, float]:
    """Estimate signed left-minus-right arrival delay with GCC-PHAT.

    ``left`` is the first cross-spectrum operand, so a positive returned lag
    means the left waveform arrives later than the right waveform. This is the
    same signed delay convention used by the stereo simulator.
    """

    left_signal, right_signal = _validate_and_prepare_signals(
        left, right, sample_rate
    )
    if max_tau is None:
        max_tau = (left_signal.size - 1) / sample_rate
    _validate_finite_real("max_tau", max_tau, positive=True)

    fft_size = 1 << (left_signal.size + right_signal.size - 1).bit_length()
    left_spectrum = np.fft.rfft(left_signal, fft_size)
    right_spectrum = np.fft.rfft(right_signal, fft_size)
    cross_power = left_spectrum * np.conj(right_spectrum)
    magnitude = np.abs(cross_power)

    phat_spectrum = np.zeros_like(cross_power)
    valid_bins = magnitude > np.finfo(magnitude.dtype).eps
    if not np.any(valid_bins):
        raise ValueError("Signals do not contain usable frequency information")
    phat_spectrum[valid_bins] = cross_power[valid_bins] / magnitude[valid_bins]

    correlation = np.fft.irfft(phat_spectrum, fft_size)
    max_shift = min(round(max_tau * sample_rate), fft_size // 2)
    if max_shift == 0:
        limited_correlation = correlation[:1]
    else:
        limited_correlation = np.concatenate(
            (correlation[-max_shift:], correlation[: max_shift + 1])
        )
    lag_samples = float(np.argmax(np.abs(limited_correlation)) - max_shift)
    return lag_samples / sample_rate, lag_samples


def localize(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    mic_spacing_m: float = DEFAULT_MIC_SPACING_M,
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S,
) -> LocalizationResult:
    """Estimate a frontal-hemisphere direction using only stereo signals.

    The physical search is constrained to the maximum delay possible for the
    supplied microphone spacing and sound speed. The returned angle follows the
    project convention: negative is LEFT, zero is FRONT, and positive is RIGHT.
    """

    _validate_finite_real("mic_spacing_m", mic_spacing_m, positive=True)
    _validate_finite_real("speed_of_sound_m_s", speed_of_sound_m_s, positive=True)
    max_tau = mic_spacing_m / speed_of_sound_m_s
    estimated_delay_seconds, estimated_delay_samples = gcc_phat(
        left, right, sample_rate, max_tau=max_tau
    )

    sine_of_angle = np.clip(
        estimated_delay_seconds * speed_of_sound_m_s / mic_spacing_m, -1.0, 1.0
    )
    estimated_angle_deg = math.degrees(math.asin(sine_of_angle))

    if estimated_angle_deg < -15.0:
        direction = "LEFT"
    elif estimated_angle_deg > 15.0:
        direction = "RIGHT"
    else:
        direction = "FRONT"

    return LocalizationResult(
        estimated_delay_seconds=estimated_delay_seconds,
        estimated_delay_samples=estimated_delay_samples,
        estimated_angle_deg=estimated_angle_deg,
        direction=direction,
    )


def _validate_and_prepare_signals(
    left: np.ndarray, right: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        raise ValueError("left and right must be NumPy arrays")
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("left and right waveforms must be one-dimensional")
    if left.size == 0 or right.size == 0:
        raise ValueError("left and right waveforms must be non-empty")
    if left.size != right.size:
        raise ValueError("left and right waveforms must have equal lengths")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    try:
        left_signal = left.astype(np.float64, copy=False)
        right_signal = right.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("left and right waveforms must contain numeric samples") from exc
    if not np.isfinite(left_signal).all() or not np.isfinite(right_signal).all():
        raise ValueError("left and right waveforms must contain finite samples")

    return _normalize_non_degenerate(left_signal), _normalize_non_degenerate(right_signal)


def _normalize_non_degenerate(signal: np.ndarray) -> np.ndarray:
    centered = signal - np.mean(signal)
    amplitude = np.max(np.abs(centered))
    if not np.isfinite(amplitude) or amplitude == 0.0:
        raise ValueError("Localization requires non-silent, non-constant signals")
    return centered / amplitude


def _validate_finite_real(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
