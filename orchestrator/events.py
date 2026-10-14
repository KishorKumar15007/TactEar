from dataclasses import dataclass
from typing import Any
@dataclass(frozen=True)
class Event: name: str; payload: Any=None
SoundDetected=SoundClassified=DirectionEstimated=ThreatAlert=ButtonPressed=RadioStarted=RadioStopped=AdvisoryRequested=AdvisoryGenerated=TTSStarted=TTSStopped=Event
