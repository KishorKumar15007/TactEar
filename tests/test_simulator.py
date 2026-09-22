"""Deterministic tests for simulated headset stereo timing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tactear.audio import AudioData
from tactear.simulator import (
    DEFAULT_MIC_SPACING_M,
    DEFAULT_SPEED_OF_SOUND_M_S,
    simulate_stereo,
)


SAMPLE_RATE = 16_000


def _broadband_audio() -> AudioData:
    noise = np.random.default_rng(7).normal(size=128).astype(np.float32)
    noise[0] = 1.0
    return AudioData(noise, SAMPLE_RATE)


def _expected_delay(angle_deg: float) -> int:
    return round(
        DEFAULT_MIC_SPACING_M
        / DEFAULT_SPEED_OF_SOUND_M_S
        * math.sin(math.radians(angle_deg))
        * SAMPLE_RATE
    )


def test_front_has_matching_channels_and_no_delay() -> None:
    audio = _broadband_audio()

    simulated = simulate_stereo(audio, 0.0)

    assert simulated.injected_delay_samples == 0
    assert simulated.hardware_simulated is True
    assert simulated.ground_truth_angle_deg == 0.0
    np.testing.assert_array_equal(simulated.left, simulated.right)
    np.testing.assert_array_equal(simulated.left, audio.waveform)


def test_left_source_makes_left_channel_lead_without_wraparound() -> None:
    audio = _broadband_audio()
    delay = abs(_expected_delay(-30.0))

    simulated = simulate_stereo(audio, -30.0)

    assert simulated.injected_delay_samples == -delay
    np.testing.assert_array_equal(simulated.left, audio.waveform)
    np.testing.assert_array_equal(simulated.right[:delay], np.zeros(delay, dtype=np.float32))
    np.testing.assert_array_equal(simulated.right[delay:], audio.waveform[:-delay])
    assert simulated.right[-1] != audio.waveform[-1]


def test_right_source_makes_right_channel_lead_without_wraparound() -> None:
    audio = _broadband_audio()
    delay = _expected_delay(30.0)

    simulated = simulate_stereo(audio, 30.0)

    assert simulated.injected_delay_samples == delay
    np.testing.assert_array_equal(simulated.right, audio.waveform)
    np.testing.assert_array_equal(simulated.left[:delay], np.zeros(delay, dtype=np.float32))
    np.testing.assert_array_equal(simulated.left[delay:], audio.waveform[:-delay])
    assert simulated.left[-1] != audio.waveform[-1]


@pytest.mark.parametrize("angle_deg", [-60.0, 60.0])
def test_large_angles_respect_physical_delay_limit(angle_deg: float) -> None:
    audio = _broadband_audio()

    simulated = simulate_stereo(audio, angle_deg)

    maximum_delay = round(
        DEFAULT_MIC_SPACING_M / DEFAULT_SPEED_OF_SOUND_M_S * SAMPLE_RATE
    )
    assert abs(simulated.injected_delay_samples) <= maximum_delay
    assert simulated.injected_delay_samples == _expected_delay(angle_deg)
    assert simulated.left.shape == simulated.right.shape == audio.waveform.shape
    assert simulated.sample_rate == SAMPLE_RATE
    assert simulated.ground_truth_angle_deg == angle_deg


@pytest.mark.parametrize("angle_deg", [-91.0, 91.0])
def test_rejects_angles_outside_frontal_hemisphere(angle_deg: float) -> None:
    with pytest.raises(ValueError, match="between -90 and 90"):
        simulate_stereo(_broadband_audio(), angle_deg)


@pytest.mark.parametrize(
    ("mic_spacing_m", "speed_of_sound_m_s"),
    [(0.0, DEFAULT_SPEED_OF_SOUND_M_S), (DEFAULT_MIC_SPACING_M, 0.0)],
)
def test_rejects_invalid_physical_constants(
    mic_spacing_m: float, speed_of_sound_m_s: float
) -> None:
    with pytest.raises(ValueError, match="positive"):
        simulate_stereo(
            _broadband_audio(),
            30.0,
            mic_spacing_m=mic_spacing_m,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )
