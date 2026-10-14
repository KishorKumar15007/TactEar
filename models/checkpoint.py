from __future__ import annotations
from pathlib import Path
import json, subprocess, time
def metadata(config, classes):
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: commit="unavailable"
    return {"version":"0.1", "timestamp":time.time(), "classes":classes, "config":config, "git_commit":commit}
def save_metadata(path, config, classes): Path(path).write_text(json.dumps(metadata(config,classes),indent=2),encoding="utf8")
