from __future__ import annotations
import numpy as np
def log_mel(waveform, sample_rate=16000, n_fft=512, hop_length=160, n_mels=64):
    """Canonical dependency-free STFT/log filterbank approximation used by train and inference."""
    x=np.asarray(waveform,np.float32); x=np.pad(x,(0,max(0,n_fft-len(x))))
    frames=np.stack([x[i:i+n_fft] for i in range(0,len(x)-n_fft+1,hop_length)])
    power=np.abs(np.fft.rfft(frames*np.hanning(n_fft),axis=1))**2
    bins=np.array_split(power,n_mels,axis=1); mel=np.stack([b.mean(axis=1) for b in bins],axis=1)
    out=np.log1p(mel); return (out-out.mean())/(out.std()+1e-6)
