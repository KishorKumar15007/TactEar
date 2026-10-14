from __future__ import annotations
import numpy as np
SPEED_OF_SOUND=343.0
def gcc_phat(left, right, sample_rate, microphone_distance_m=0.17):
    """Estimate microphone-relative azimuth. Positive values mean right-channel lag.

    Channel-to-side mapping must be calibrated on actual microphone hardware.
    """
    left,right=np.asarray(left,float),np.asarray(right,float)
    if sample_rate <= 0 or microphone_distance_m <= 0:
        raise ValueError("Sample rate and microphone spacing must be positive")
    if left.ndim != 1 or right.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return 0.0, 0.0
    if len(left)<8 or len(right)<8 or np.std(left)<1e-9 or np.std(right)<1e-9: return 0.0,0.0
    n=1
    while n < len(left)+len(right): n*=2
    cross=np.fft.rfft(right,n)*np.conj(np.fft.rfft(left,n)); cross/=np.maximum(np.abs(cross),1e-12)
    cc=np.fft.irfft(cross,n); max_delay=microphone_distance_m/SPEED_OF_SOUND
    max_shift=min(max(1,int(np.ceil(sample_rate*max_delay))),n//2-1)
    cc=np.concatenate((cc[-max_shift:],cc[:max_shift+1])); shift=int(np.argmax(np.abs(cc)))-max_shift
    delay=shift/sample_rate
    azimuth=float(np.degrees(np.arcsin(np.clip(delay*SPEED_OF_SOUND/microphone_distance_m,-1,1))))
    confidence=float(np.max(np.abs(cc))/(np.mean(np.abs(cc))+1e-12)); return azimuth, min(1.0,confidence/8)
def egocentric_azimuth(mic_azimuth, yaw_deg): return ((mic_azimuth-yaw_deg+180)%360)-180
def direction_sector(azimuth):
    names=["front","front-right","right","back-right","back","back-left","left","front-left"]
    return names[int(((azimuth+22.5)%360)//45)]
