"""Shared, bounded decoding and signal diagnostics for files and live PCM."""
import io
import numpy as np

MAX_BYTES = 24 * 1024 * 1024
MAX_SECONDS = 30

def to_mono(samples):
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not x.size or x.shape[1] not in (1, 2) or not np.isfinite(x).all():
        raise ValueError("Expected finite mono or stereo audio")
    mean = x.mean(axis=1)
    energy = np.mean(x.astype(np.float64)**2, axis=0)
    # A silent or inverted channel must not erase the audible channel.
    if np.mean(mean.astype(np.float64)**2) < .25 * max(energy):
        return x[:, int(np.argmax(energy))].copy(), "stronger channel (phase cancellation protection)"
    return mean, "mono" if x.shape[1] == 1 else "stereo mean"

def decode_audio(data):
    import soundfile as sf
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose an audio file under 24 MiB")
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            if f.format not in {"WAV", "WAVEX", "FLAC", "OGG"}:
                raise ValueError("Supported formats: WAV, FLAC and OGG")
            if f.channels not in (1, 2) or not 8000 <= f.samplerate <= 192000:
                raise ValueError("Use mono/stereo audio at 8–192 kHz")
            if not 1 <= len(f) / f.samplerate <= MAX_SECONDS:
                raise ValueError("Clip duration must be 1–30 seconds")
            rate = f.samplerate
            samples = f.read(dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise ValueError("Cannot decode audio. Use a valid WAV, FLAC or OGG file") from exc
    mono, mix = to_mono(samples)
    if np.max(np.abs(samples)) > 1.01:
        raise ValueError("Audio exceeds PCM amplitude range ±1; export a normalized file")
    return samples, mono, rate, mix

def measurements(samples, rate):
    x = np.asarray(samples, np.float32)
    if x.ndim == 1: x = x[:, None]
    mono, mix = to_mono(x)
    rms = float(np.sqrt(np.mean(mono.astype(float)**2)))
    frame = mono[:2048]
    spectrum = np.abs(np.fft.rfft((frame-frame.mean()) * np.hanning(len(frame)))) / len(frame)
    return {"waveform": [float(a[np.argmax(np.abs(a))]) for a in np.array_split(mono, min(512, len(mono)))],
            "spectrum": [float(a.max()) for a in np.array_split(spectrum, min(96,len(spectrum)))],
            "rms": rms, "peak": float(np.abs(x).max()), "db": float(20*np.log10(max(rms,1e-9))),
            "channel_rms": np.sqrt(np.mean(x.astype(float)**2, axis=0)).tolist(),
            "sample_rate": rate, "channels": x.shape[1], "frame_ms": len(x)/rate*1000,
            "mix": mix, "clipped_fraction": float(np.mean(np.abs(x)>=.999))}

def clip_visuals(mono, rate):
    # Bounded STFT overview. Each column is a genuine time slice, not generated art.
    starts = np.linspace(0, max(0,len(mono)-512), min(240,max(1,len(mono)//256))).astype(int)
    columns = []
    for start in starts:
        frame = mono[start:start+512]
        spectrum = np.abs(np.fft.rfft(frame*np.hanning(len(frame)))) / len(frame)
        columns.append([float(max(-100,20*np.log10(max(float(a.max()),1e-9))))
                        for a in np.array_split(spectrum,64)])
    return {"columns": columns, "duration": len(mono)/rate, "nyquist": rate/2}
