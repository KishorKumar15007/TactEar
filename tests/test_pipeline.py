"""Focused integration tests for the TactEar core pipeline."""

from __future__ import annotations

from datetime import datetime
import uuid

import numpy as np
import pytest

from tactear.audio import AudioData
from tactear.events import ClassificationResult, LocalizationResult
from tactear import pipeline
from tactear.simulator import SimulatedStereo


class _RecordingClassifier:
    def __init__(self) -> None:
        self.received_audio: AudioData | None = None
        self.calls = 0

    def classify(self, audio: AudioData) -> ClassificationResult:
        self.received_audio = audio
        self.calls += 1
        return ClassificationResult("Gunshot, gunfire", "gunshot", 0.73)


def _audio() -> AudioData:
    return AudioData(np.array([0.2, -0.3, 0.4, -0.1], dtype=np.float32), 16_000)


def _simulated_stereo() -> SimulatedStereo:
    return SimulatedStereo(
        left=np.array([0.0, 0.2, -0.3, 0.4], dtype=np.float32),
        right=np.array([0.2, -0.3, 0.4, -0.1], dtype=np.float32),
        sample_rate=16_000,
        ground_truth_angle_deg=-45.0,
        injected_delay_seconds=-0.000375,
        injected_delay_samples=-6,
    )


def test_run_pipeline_wires_results_without_fabricating_values(monkeypatch) -> None:
    audio = _audio()
    stereo = _simulated_stereo()
    classifier = _RecordingClassifier()
    captured: dict[str, object] = {}
    localization_result = LocalizationResult(-0.0003125, -5.0, -36.5, "LEFT")

    def fake_load_audio(path: str) -> AudioData:
        captured["audio_path"] = path
        return audio

    def fake_simulate(received_audio: AudioData, angle_deg: float) -> SimulatedStereo:
        captured["simulation_audio"] = received_audio
        captured["simulation_angle"] = angle_deg
        return stereo

    def fake_localize(
        left: np.ndarray, right: np.ndarray, sample_rate: int
    ) -> LocalizationResult:
        captured["localization_inputs"] = (left, right, sample_rate)
        return localization_result

    monkeypatch.setattr(pipeline, "load_audio", fake_load_audio)
    monkeypatch.setattr(pipeline, "simulate_stereo", fake_simulate)
    monkeypatch.setattr(pipeline, "localize", fake_localize)

    event = pipeline.run_pipeline("demo_audio/example.wav", -45.0, classifier=classifier)

    assert captured["audio_path"] == "demo_audio/example.wav"
    assert classifier.received_audio is audio
    assert captured["simulation_audio"] is audio
    assert captured["simulation_angle"] == -45.0
    assert captured["localization_inputs"] == (stereo.left, stereo.right, stereo.sample_rate)
    assert event.sound_class == "gunshot"
    assert event.raw_model_label == "Gunshot, gunfire"
    assert event.classification_confidence == 0.73
    assert event.direction == "LEFT"
    assert event.estimated_angle_deg == -36.5
    assert event.ground_truth_angle_deg == -45.0
    assert event.localization_error_deg == 8.5
    assert event.estimated_delay_seconds == -0.0003125
    assert event.hardware_simulated is True
    uuid.UUID(event.event_id)
    assert datetime.fromisoformat(event.timestamp).tzinfo is not None


def test_default_classifier_is_cached_between_pipeline_runs(monkeypatch) -> None:
    classifier = _RecordingClassifier()
    stereo = _simulated_stereo()
    created: list[_RecordingClassifier] = []

    def create_classifier() -> _RecordingClassifier:
        created.append(classifier)
        return classifier

    monkeypatch.setattr(pipeline, "_cached_classifier", None)
    monkeypatch.setattr(pipeline, "AudioClassifier", create_classifier)
    monkeypatch.setattr(pipeline, "load_audio", lambda path: _audio())
    monkeypatch.setattr(pipeline, "simulate_stereo", lambda audio, angle: stereo)
    monkeypatch.setattr(
        pipeline,
        "localize",
        lambda left, right, sample_rate: LocalizationResult(0.0, 0.0, 0.0, "FRONT"),
    )

    pipeline.run_pipeline("first.wav", 0.0)
    pipeline.run_pipeline("second.wav", 0.0)

    assert created == [classifier]
    assert classifier.calls == 2


def test_invalid_source_path_fails_before_classifier_initialization() -> None:
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline("missing-input.wav", 0.0)
