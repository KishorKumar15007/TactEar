from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class AudioFrame:
    samples: np.ndarray
    sample_rate: int
    timestamp: float
    sequence: int = 0
    source: str = "unknown"
    device: str | None = None

@dataclass
class StereoAudioFrame:
    left: np.ndarray
    right: np.ndarray
    sample_rate: int
    timestamp: float
    sequence: int = 0
    source: str = "unknown"
    device: str | None = None

@dataclass
class Orientation:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    timestamp: float = 0.0

@dataclass
class SoundEvent:
    event_id: str
    sound_class: str
    confidence: float
    azimuth_deg: float
    direction: str
    timestamp: float
    suppression: str = "preserve"

@dataclass
class SourceChunk:
    source: str
    section: str
    text: str
    score: float

@dataclass
class Advisory:
    event_id: str
    text: str
    source_chunks: list[SourceChunk] = field(default_factory=list)
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
