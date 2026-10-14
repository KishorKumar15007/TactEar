"""Versioned, transport-neutral messages for dashboard WebSocket bridges.

The core publishes events; a separately hosted bridge can serialize these messages
without making audio capture depend on a web server.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
from tactear.types import SoundEvent, Advisory
PROTOCOL_VERSION = 1
def message(kind: str, payload: Any | None = None) -> dict[str, Any]:
    return {"version": PROTOCOL_VERSION, "type": kind, **({"payload": asdict(payload) if hasattr(payload,"__dataclass_fields__") else payload} if payload is not None else {})}
def sound_event(event: SoundEvent) -> dict[str, Any]:
    data=asdict(event); data["class"]=data.pop("sound_class"); return {"version":1,"type":"sound_event","payload":data}
def advisory_completed(advisory: Advisory) -> dict[str, Any]: return message("advisory_completed", advisory)
