from __future__ import annotations
import numpy as np, time
from tactear.types import StereoAudioFrame
def delayed_frame(azimuth_deg, sample_rate=16000, frame_size=1024, distance=.17):
    delay=int(round(np.sin(np.radians(azimuth_deg))*distance/343*sample_rate)); rng=np.random.default_rng(7); x=rng.normal(0,1,frame_size).astype(np.float32)
    right=np.roll(x,delay); return StereoAudioFrame(x,right,sample_rate,time.time())
SCENARIOS=[("gunshot",43), ("vehicle",-90), ("helicopter",180), ("siren",0), ("human_voice",-43)]
