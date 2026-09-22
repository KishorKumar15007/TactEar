"""Composable core pipeline for the TactEar review prototype."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tactear.audio import load_audio
from tactear.classifier import AudioClassifier
from tactear.events import ThreatEvent
from tactear.localization import localize
from tactear.simulator import simulate_stereo


_cached_classifier: AudioClassifier | None = None


def run_pipeline(
    audio_path: str | Path,
    simulated_angle_deg: float,
    *,
    classifier: AudioClassifier | None = None,
) -> ThreatEvent:
    """Run audio loading, classification, stereo simulation, and localization.

    ``simulated_angle_deg`` is explicit ground truth for the temporary headset
    simulation only. Localization receives only its left/right waveforms and
    sample rate, then its independent estimate is compared afterward to form
    the event's localization error.
    """

    audio = load_audio(audio_path)
    classification = _get_classifier(classifier).classify(audio)
    simulated_stereo = simulate_stereo(audio, simulated_angle_deg)
    localization = localize(
        simulated_stereo.left,
        simulated_stereo.right,
        simulated_stereo.sample_rate,
    )
    ground_truth_angle_deg = simulated_stereo.ground_truth_angle_deg

    return ThreatEvent(
        event_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_audio=str(Path(audio_path)),
        sound_class=classification.tactical_label,
        raw_model_label=classification.raw_label,
        classification_confidence=classification.confidence,
        direction=localization.direction,
        estimated_angle_deg=localization.estimated_angle_deg,
        ground_truth_angle_deg=ground_truth_angle_deg,
        localization_error_deg=abs(
            localization.estimated_angle_deg - ground_truth_angle_deg
        ),
        estimated_delay_seconds=localization.estimated_delay_seconds,
        hardware_simulated=True,
    )


def _get_classifier(classifier: AudioClassifier | None) -> AudioClassifier:
    if classifier is not None:
        return classifier

    global _cached_classifier
    if _cached_classifier is None:
        _cached_classifier = AudioClassifier()
    return _cached_classifier
