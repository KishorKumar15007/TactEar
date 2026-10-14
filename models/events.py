"""Multi-label event filtering; thresholds are explicitly uncalibrated."""
import time
import uuid

GROUPS = {
    "Gunshot-like": ["Gunshot, gunfire"],
    "Rapid gunfire-like": ["Machine gun", "Fusillade"],
    "Fire / crackle": ["Fire", "Crackle"],
    "Alarm": ["Alarm", "Fire alarm", "Smoke detector, smoke alarm", "Alarm clock"],
    "Siren": ["Siren"],
    "Engine": ["Engine", "Vehicle"],
    "Rotor": ["Helicopter"],
    "Machinery": ["Mechanisms", "Mechanical fan", "Power tool", "Drill", "Jackhammer"],
    "Impact / explosion-like": ["Explosion", "Fireworks", "Firecracker"],
    "Speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking"],
    "Background": ["Noise", "White noise", "Pink noise", "Wind", "Rain"],
    "Silence": ["Silence"],
}

def category(label):
    return next((group for group, labels in GROUPS.items() if label in labels), "Other")

def severity(label):
    group = category(label)
    return "attention" if group in {"Gunshot-like", "Rapid gunfire-like", "Fire / crackle", "Impact / explosion-like"} else "warning" if group in {"Alarm","Siren"} else "normal"

class EventFilter:
    def __init__(self, threshold=.35, cooldown=8):
        self.threshold, self.cooldown = threshold, cooldown
        self.counts, self.last = {}, {}

    def apply(self, prediction, source, single=False):
        now = prediction.get("timestamp",time.time())
        rows = {r["label"]: r for r in [*prediction.get("watched",[]), *prediction["top"]]}
        eligible = sorted((r for r in rows.values() if r["score"] >= self.threshold), key=lambda r:-r["score"])
        self.counts = {r["label"]:self.counts.get(r["label"],0)+1 for r in eligible}
        events = []
        for row in eligible[:12]:
            label = row["label"]
            status = "CANDIDATE" if single or self.counts[label] < 2 else "DETECTED"
            previous = self.last.get(label)
            # Candidates remain visible immediately; promote in-place on repetition.
            if previous and now-previous[0] < self.cooldown:
                if previous[1] == "CANDIDATE" and status == "DETECTED":
                    event_id = previous[2]
                else:
                    continue
            else:
                event_id = str(uuid.uuid4())
            self.last[label] = (now,status,event_id)
            events.append({"event_id":event_id,"timestamp":now,"label":label,"score":row["score"],
                           "source":source,"latency_ms":prediction["latency_ms"],"status":status,
                           "category":category(label),"severity":severity(label),"direction":None})
        best = prediction["top"][0]
        status = "UNCERTAIN" if best["score"] < self.threshold else "CANDIDATE" if single or self.counts[best["label"]]<2 else "DETECTED"
        groups = [{"label":group,"score":max((rows[label]["score"] for label in labels if label in rows),default=0)}
                  for group,labels in GROUPS.items()]
        return {**prediction,"status":status,"label":best["label"] if status!="UNCERTAIN" else None,
                "category":category(best["label"]) if status!="UNCERTAIN" else "Unknown",
                "groups":groups,"source":source,"threshold":self.threshold,
                "filter_note":"Candidate retained immediately; repeated windows confirm persistence, not correctness. Per-label cooldown: 8 s.",
                "suppressed":max(0,len(eligible)-len(events))},events
