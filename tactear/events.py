"""Typed contracts shared by future TactEar pipeline components."""

from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """A future classifier's raw and tactical labels with confidence."""

    raw_label: str
    tactical_label: str
    confidence: float


@dataclass
class LocalizationResult:
    """Signal-derived interaural timing and direction estimate."""

    estimated_delay_seconds: float
    estimated_delay_samples: float
    estimated_angle_deg: float
    direction: str


@dataclass
class ThreatEvent:
    """One complete TactEar core-pipeline result."""

    event_id: str
    timestamp: str
    source_audio: str
    sound_class: str
    raw_model_label: str
    classification_confidence: float
    direction: str
    estimated_angle_deg: float
    ground_truth_angle_deg: float
    localization_error_deg: float
    estimated_delay_seconds: float
    hardware_simulated: bool
