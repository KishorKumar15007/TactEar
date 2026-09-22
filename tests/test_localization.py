"""Controlled GCC-PHAT localization tests using simulated stereo signals."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tactear.audio import AudioData
from tactear.localization import gcc_phat, localize
from tactear.simulator import (
    DEFAULT_MIC_SPACING_M,
    DEFAULT_SPEED_OF_SOUND_M_S,
    simulate_stereo,
)


SAMPLE_RATE = 16_000
ANGLE_TOLERANCE_DEG = 5.0


def _broadband_audio() -> AudioData:
    waveform = np.random.default_rng(42).normal(size=4_096).astype(np.float32)
    return AudioData(waveform, SAMPLE_RATE)


@pytest.mark.parametrize(
    ("angle_deg", "expected_direction"),
    [
        (-60.0, "LEFT"),
        (-30.0, "LEFT"),
        (0.0, "FRONT"),
        (30.0, "RIGHT"),
        (60.0, "RIGHT"),
    ],
)
def test_localizes_simulated_angles(
    angle_deg: float, expected_direction: str
) -> None:
    simulated = simulate_stereo(_broadband_audio(), angle_deg)

    result = localize(simulated.left, simulated.right, simulated.sample_rate)

    assert result.direction == expected_direction
    assert result.estimated_angle_deg == pytest.approx(angle_deg, abs=ANGLE_TOLERANCE_DEG)
    assert result.estimated_delay_samples == simulated.injected_delay_samples


def test_gcc_phat_uses_the_simulator_delay_sign_convention() -> None:
    simulated = simulate_stereo(_broadband_audio(), 30.0)
    max_tau = DEFAULT_MIC_SPACING_M / DEFAULT_SPEED_OF_SOUND_M_S

    delay_seconds, delay_samples = gcc_phat(
        simulated.left, simulated.right, simulated.sample_rate, max_tau=max_tau
    )

    assert delay_samples == simulated.injected_delay_samples
    assert delay_seconds == pytest.approx(
        simulated.injected_delay_samples / SAMPLE_RATE
    )


def test_physical_lag_limit_restricts_gcc_phat_search() -> None:
    source = _broadband_audio().waveform
    left = np.zeros_like(source)
    left[20:] = source[:-20]
    max_tau = DEFAULT_MIC_SPACING_M / DEFAULT_SPEED_OF_SOUND_M_S

    _, delay_samples = gcc_phat(left, source, SAMPLE_RATE, max_tau=max_tau)

    assert abs(delay_samples) <= round(max_tau * SAMPLE_RATE)


def test_rejects_silent_or_incompatible_signals() -> None:
    with pytest.raises(ValueError, match="non-silent"):
        localize(np.zeros(32), np.zeros(32), SAMPLE_RATE)
    with pytest.raises(ValueError, match="equal lengths"):
        localize(np.ones(32), np.ones(31), SAMPLE_RATE)
    with pytest.raises(ValueError, match="one-dimensional"):
        localize(np.ones((2, 2)), np.ones((2, 2)), SAMPLE_RATE)


def test_rejects_non_finite_samples_and_invalid_geometry() -> None:
    source = _broadband_audio().waveform
    non_finite = source.copy()
    non_finite[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        localize(non_finite, source, SAMPLE_RATE)
    with pytest.raises(ValueError, match="sample_rate"):
        localize(source, source, 0)
    with pytest.raises(ValueError, match="mic_spacing_m"):
        localize(source, source, SAMPLE_RATE, mic_spacing_m=0.0)
    with pytest.raises(ValueError, match="speed_of_sound_m_s"):
        localize(source, source, SAMPLE_RATE, speed_of_sound_m_s=0.0)


def test_public_localize_signature_has_no_simulation_metadata() -> None:
    parameters = inspect.signature(localize).parameters

    assert "ground_truth_angle" not in parameters
    assert "injected_delay" not in parameters
    assert "expected_direction" not in parameters
