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
