from __future__ import annotations
from threading import RLock
class AudioPriority:
    RADIO=1; CRITICAL=2; TACTICAL_TTS=3; OTHER=4
    def __init__(self, speech): self.speech,self.radio_active,self._lock=speech,False,RLock()
    def radio_started(self):
        with self._lock: self.radio_active=True; self.speech.stop()
    def radio_stopped(self):
        with self._lock: self.radio_active=False
    def speak(self,text):
        with self._lock:
            if self.radio_active: return False
            self.speech.speak(text); return True
