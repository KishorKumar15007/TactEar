"""Local Ollama advisory adapter for TactEar Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import urllib.error
import urllib.request

from tactear.events import ThreatEvent
from tactear.intelligence.sop import SOPResult

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "tinyllama"
UNAVAILABLE_MESSAGE = (
    "Local advisory model unavailable. Classification and localization remain operational."
)


@dataclass(frozen=True)
class AdvisoryResult:
    """Typed situational advisory from the local LLM component."""

    text: str
    model: str
    available: bool = True


def get_default_model() -> str:
    """Return configured or default model name from environment."""
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def build_advisory_prompt(event: ThreatEvent, sop: SOPResult) -> str:
    """Construct a strictly grounded prompt using only event and SOP data."""
    confidence_pct = event.classification_confidence * 100.0
    uncertainty_note = ""
    if event.classification_confidence < 0.5:
        uncertainty_note = (
            "\nNote: Detection confidence is below 50%. Explicitly preserve uncertainty in the advisory."
        )

    return f"""SYSTEM:
You are the local advisory component of TactEar.

Generate a concise situational advisory using ONLY the supplied event and retrieved SOP.

Rules:
- Maximum 2 short sentences.
- Do not invent threats.
- Do not invent people.
- Do not invent weapons.
- Do not invent locations.
- Do not invent commands.
- Do not claim certainty beyond detection confidence.
- Do not contradict the supplied SOP.
- If confidence is low, express uncertainty.
- Automated detection is advisory only.

EVENT DATA:
Detected class: {event.sound_class}
Raw model label: {event.raw_model_label}
Confidence: {event.classification_confidence:.4f} ({confidence_pct:.1f}%)
Direction: {event.direction}
Estimated angle: {event.estimated_angle_deg:.1f}°
Hardware simulation: {event.hardware_simulated}{uncertainty_note}

RETRIEVED SOP:
{sop.content}

Generate advisory:"""


def check_llm_status(
    host: str = DEFAULT_OLLAMA_HOST,
    timeout_seconds: float = 1.5,
) -> tuple[bool, str]:
    """Check whether local Ollama is reachable and return availability and active model name."""
    configured_model = get_default_model()
    tags_url = f"{host.rstrip('/')}/api/tags"

    try:
        req = urllib.request.Request(tags_url, headers={"User-Agent": "TactEar/0.2"})
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            if response.status == 200:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                models = [m.get("name", "") for m in data.get("models", [])]
                if configured_model in models:
                    return True, configured_model
                if models:
                    # Return first available installed model
                    return True, models[0]
                return True, configured_model
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        pass

    return False, "unavailable"


def generate_advisory(
    event: ThreatEvent,
    sop: SOPResult,
    *,
    model: str | None = None,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout_seconds: float = 10.0,
) -> AdvisoryResult:
    """Generate a situational advisory grounded strictly in the supplied event and SOP.

    If Ollama is offline or generation fails, returns a safe failure AdvisoryResult
    without raising exceptions, ensuring core classification and localization
    remain operational.
    """
    selected_model = model or get_default_model()
    endpoint = f"{host.rstrip('/')}/api/generate"
    prompt = build_advisory_prompt(event, sop)

    payload = {
        "model": selected_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 128,
        },
    }

    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "TactEar/0.2",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            if response.status == 200:
                raw_body = response.read().decode("utf-8")
                response_json = json.loads(raw_body)
                advisory_text = response_json.get("response", "").strip()
                if advisory_text:
                    return AdvisoryResult(
                        text=advisory_text,
                        model=selected_model,
                        available=True,
                    )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.info("Local Ollama advisory unavailable (%s): %s", type(exc).__name__, exc)

    return AdvisoryResult(
        text=UNAVAILABLE_MESSAGE,
        model="unavailable",
        available=False,
    )
