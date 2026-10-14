"""Rolling audio and temporal confirmation; no invented classification."""
from collections import deque
import numpy as np
from .audio_input import to_mono

class AudioWindow:
    def __init__(self, seconds=10):
        self.seconds=seconds
        self.reset()
    def reset(self):
        self.parts=deque(); self.count=0; self.rate=None; self.last_sequence=None
    def add(self, frame):
        if self.rate!=frame.sample_rate or (self.last_sequence is not None and frame.sequence!=self.last_sequence+1):
            self.reset()
        self.rate=frame.sample_rate
        self.last_sequence=frame.sequence
        x,_=to_mono(np.column_stack((frame.left,frame.right)))
        self.parts.append(x); self.count+=len(x)
        target=int(self.rate*self.seconds)
        while self.parts and self.count-len(self.parts[0])>=target:
            self.count-=len(self.parts.popleft())
    def ready(self): return self.rate is not None and self.count>=int(self.rate*self.seconds)
    def samples(self): return np.concatenate(self.parts)[-int(self.rate*self.seconds):].copy()

class Confirmation:
    def __init__(self, threshold=.35, consecutive=2):
        self.threshold=threshold; self.consecutive=consecutive; self.label=None; self.count=0
    def apply(self,prediction):
        best=prediction["top"][0]
        if best["score"]<self.threshold:
            self.label=None; self.count=0
            return {**prediction,"status":"UNCERTAIN","label":None}
        self.count=self.count+1 if self.label==best["label"] else 1
        self.label=best["label"]
        return {**prediction,"status":"DETECTED" if self.count>=self.consecutive else "CANDIDATE",
                "label":best["label"], "confirmations":self.count}
