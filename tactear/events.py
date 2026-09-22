"""Typed contracts shared by future TactEar pipeline components."""

from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """A future classifier's raw and tactical labels with confidence."""

    raw_label: str
    tactical_label: str
    confidence: float
