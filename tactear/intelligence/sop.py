"""Deterministic local SOP retrieval engine for TactEar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from tactear.events import ThreatEvent


@dataclass(frozen=True)
class SOPResult:
    """Typed result from the local SOP retriever."""

    sop_id: str
    title: str
    content: str


# Explicit deterministic mapping from tactical sound classes to SOP filenames.
SOP_MAP: dict[str, str] = {
    "gunshot": "gunshot.txt",
    "vehicle": "vehicle.txt",
    "siren": "siren.txt",
    "speech": "speech.txt",
    "other": "unknown.txt",
}

# Human-readable procedure titles for display and advisory grounding.
SOP_TITLES: dict[str, str] = {
    "gunshot": "Gunfire Response Procedure",
    "vehicle": "Vehicle Activity Response Procedure",
    "siren": "Emergency Signal Response Procedure",
    "speech": "Speech Detection Procedure",
    "unknown": "Unclassified Acoustic Event Procedure",
}


def get_sop_directory() -> Path:
    """Resolve the directory holding local SOP procedure files.

    Checks package-relative directory (tactear/sop) first, falling back to
    the project root directory (sop/).
    """
    package_sop_dir = Path(__file__).resolve().parent.parent / "sop"
    if package_sop_dir.is_dir():
        return package_sop_dir

    project_sop_dir = Path(__file__).resolve().parents[2] / "sop"
    if project_sop_dir.is_dir():
        return project_sop_dir

    # Fallback to package_sop_dir as default
    return package_sop_dir


def retrieve_sop(event_or_class: Union[ThreatEvent, str]) -> SOPResult:
    """Retrieve the standard operating procedure for a given event or class.

    Accepts either a ThreatEvent or a tactical class string. Normalizes the
    class deterministically and falls back safely to unknown.txt for any
    unrecognized or unexpected input. Never makes network requests.
    """
    if isinstance(event_or_class, ThreatEvent):
        raw_category = event_or_class.sound_class
    elif isinstance(event_or_class, str):
        raw_category = event_or_class
    else:
        raw_category = str(event_or_class)

    normalized_class = raw_category.strip().casefold()
    filename = SOP_MAP.get(normalized_class, "unknown.txt")

    # Map sop_id: "other" maps to "unknown"
    if normalized_class in SOP_MAP and normalized_class != "other":
        sop_id = normalized_class
    else:
        sop_id = "unknown"

    title = SOP_TITLES.get(sop_id, SOP_TITLES["unknown"])

    sop_dir = get_sop_directory()
    sop_path = sop_dir / filename

    if not sop_path.is_file():
        raise FileNotFoundError(f"SOP procedure file not found: {sop_path}")

    content = sop_path.read_text(encoding="utf-8").strip()

    return SOPResult(
        sop_id=sop_id,
        title=title,
        content=content,
    )
