"""Offline PANNs CNN14 16 kHz inference using the official AudioSet checkpoint."""
from pathlib import Path
import csv
import math
import time
import numpy as np

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "data/models/panns/cnn14_16k.pth"
# These are separate AudioSet outputs, not a forced five-way softmax.
WATCH_LABELS = [
    "Gunshot, gunfire", "Machine gun", "Fusillade", "Fire", "Crackle",
    "Alarm", "Fire alarm", "Smoke detector, smoke alarm", "Siren", "Alarm clock",
    "Engine", "Mechanisms", "Mechanical fan", "Power tool", "Drill", "Jackhammer",
    "Helicopter", "Vehicle", "Speech", "Explosion", "Fireworks", "Firecracker",
    "Silence", "Noise", "White noise", "Pink noise", "Wind", "Rain",
]
class ModelUnavailable(RuntimeError): pass

class PannsCNN14Encoder:
    name = "PANNs CNN14 16 kHz / AudioSet 527"
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
        path = Path(checkpoint)
        if not path.is_file(): raise ModelUnavailable(f"Checkpoint missing: {path}; run scripts/download_panns.py explicitly")
        try:
            import torch
            from .cnn14 import CNN14
        except ImportError as exc: raise ModelUnavailable("Install requirements-ml.txt") from exc
        labels_path = path.with_name("class_labels_indices.csv")
        with labels_path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 527 or [int(r["index"]) for r in rows] != list(range(527)):
            raise ModelUnavailable("AudioSet label order is invalid")
        self.labels = [r["display_name"] for r in rows]
        self.watched_indices = [(label, self.labels.index(label)) for label in WATCH_LABELS]
        self.device = device
        torch.set_num_threads(4)
        self.network = CNN14()
        # Official legacy checkpoint contains NumPy statistics alongside tensors.
        # Allow only these array types; never fall back to unrestricted pickle loading.
        from numpy._core.multiarray import _reconstruct
        with torch.serialization.safe_globals([
            (_reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, np.dtype,
            type(np.dtype("float64")), type(np.dtype("float32")), type(np.dtype("int64")),
        ]):
            checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
        self.network.load_state_dict(checkpoint_data["model"], strict=True)
        self.network.to(device).eval()
        with torch.inference_mode():
            output = self.network(torch.zeros(1, 16000, device=device))
        if output["clipwise_output"].shape != (1,527) or output["embedding"].shape != (1,2048):
            raise ModelUnavailable("Invalid PANNs output dimensions")

    def predict(self, waveform, sample_rate):
        import torch
        from scipy.signal import resample_poly
        started = time.perf_counter()
        x = np.asarray(waveform, dtype=np.float32)
        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError("Model requires a nonempty finite mono waveform")
        if sample_rate < 8000: raise ValueError("Sample rate must be at least 8000 Hz")
        if sample_rate != 16000:
            divisor = math.gcd(int(sample_rate),16000)
            x = resample_poly(x,16000//divisor,int(sample_rate)//divisor).astype(np.float32)
        if x.size < 16000: raise ValueError("Supply at least one second of audio")
        with torch.inference_mode():
            scores = self.network(torch.from_numpy(x.copy())[None].to(self.device))["clipwise_output"][0].cpu().numpy()
        if scores.shape != (527,) or not np.isfinite(scores).all(): raise ValueError("Invalid model output")
        ordered = np.argsort(scores)[::-1]
        return {"model": self.name, "top": [{"label":self.labels[i],"score":float(scores[i])} for i in ordered[:8]],
                "watched": [{"label":label,"score":float(scores[index])} for label, index in self.watched_indices],
                "latency_ms": (time.perf_counter()-started)*1000,
                "window_seconds": len(x)/16000, "timestamp": time.time(),
                "score_note": "Independent sigmoid scores; not calibrated certainty and do not sum to 100%."}
