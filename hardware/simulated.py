from __future__ import annotations
import time, logging
import numpy as np
from tactear.types import AudioFrame, StereoAudioFrame, Orientation

log = logging.getLogger(__name__)
class SimulatedAmbientAudioSource:
    def __init__(self, sample_rate=16000, frame_size=1024): self.sample_rate, self.frame_size, self.running = sample_rate, frame_size, False
    def start(self): self.running = True
    def stop(self): self.running = False
    def read_frame(self, timeout=1.0):
        if not self.running: raise RuntimeError("ambient source not started")
        noise = np.zeros(self.frame_size, dtype=np.float32)
        return StereoAudioFrame(noise, noise.copy(), self.sample_rate, time.time())
class SimulatedBoomAudioSource:
    def __init__(self, sample_rate=16000, frame_size=1024): self.sample_rate, self.frame_size, self.running = sample_rate, frame_size, False
    def start(self): self.running = True
    def stop(self): self.running = False
    def read_frame(self, timeout=1.0): return AudioFrame(np.zeros(self.frame_size, np.float32), self.sample_rate, time.time())
class SimulatedOrientationSensor:
    def __init__(self, yaw_deg=0.0): self.yaw_deg=yaw_deg
    def start(self): pass
    def stop(self): pass
    def read_orientation(self): return Orientation(self.yaw_deg, 0, 0, time.time())
class SimulatedHapticOutput:
    def __init__(self): self.events=[]
    def trigger(self, pattern, direction="front"):
        self.events.append((pattern,direction)); log.info("[HAPTIC] %s -> %s", pattern, direction.upper())
    def stop(self): pass
class SimulatedButtonInput:
    def start(self, callback): self.callback=callback
    def stop(self): pass
    def press(self): self.callback()
