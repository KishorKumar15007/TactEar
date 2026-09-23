"""Unit tests for the local Ollama advisory adapter with mocked network boundary."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from tactear.events import ThreatEvent
from tactear.intelligence.advisor import (
    UNAVAILABLE_MESSAGE,
    AdvisoryResult,
    build_advisory_prompt,
    check_llm_status,
    generate_advisory,
)
from tactear.intelligence.sop import SOPResult


def _sample_threat_event(
    sound_class: str = "gunshot",
    confidence: float = 0.912,
    direction: str = "LEFT",
    estimated_angle_deg: float = -43.0,
) -> ThreatEvent:
    return ThreatEvent(
        event_id="test-event-uuid-1234",
        timestamp="2026-09-22T12:00:00+00:00",
        source_audio="demo_audio/gunshot.wav",
        sound_class=sound_class,
        raw_model_label="Gunshot, gunfire",
        classification_confidence=confidence,
        direction=direction,
        estimated_angle_deg=estimated_angle_deg,
        ground_truth_angle_deg=-45.0,
        localization_error_deg=2.0,
        estimated_delay_seconds=-0.00035,
        hardware_simulated=True,
    )


def _sample_sop(sop_id: str = "gunshot") -> SOPResult:
    return SOPResult(
        sop_id=sop_id,
        title="Gunfire Response Procedure",
        content="Potential gunfire detected.\nMaintain cover and situational awareness.",
    )


class AdvisorPromptTests(unittest.TestCase):
    def test_event_enters_prompt(self) -> None:
        event = _sample_threat_event()
        sop = _sample_sop()

        prompt = build_advisory_prompt(event, sop)

        self.assertIn("gunshot", prompt)
        self.assertIn("Gunshot, gunfire", prompt)
        self.assertIn("0.9120", prompt)
        self.assertIn("91.2%", prompt)
        self.assertIn("LEFT", prompt)
        self.assertIn("-43.0°", prompt)
        self.assertIn("True", prompt)

    def test_sop_enters_prompt(self) -> None:
        event = _sample_threat_event()
        sop = _sample_sop()

        prompt = build_advisory_prompt(event, sop)

        self.assertIn("Potential gunfire detected.", prompt)
        self.assertIn("Maintain cover and situational awareness.", prompt)

    def test_low_confidence_triggers_uncertainty_guidance(self) -> None:
        event = _sample_threat_event(confidence=0.35)
        sop = _sample_sop()

        prompt = build_advisory_prompt(event, sop)

        self.assertIn("below 50%", prompt)
        self.assertIn("uncertainty", prompt.lower())


class AdvisorInferenceTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_returns_mocked_advisory_response(self, mock_urlopen) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_payload = {
            "response": "Possible gunfire detected to the left. Maintain cover and confirm the contact."
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        event = _sample_threat_event()
        sop = _sample_sop()

        result = generate_advisory(event, sop, model="tinyllama")

        self.assertIsInstance(result, AdvisoryResult)
        self.assertTrue(result.available)
        self.assertEqual(result.model, "tinyllama")
        self.assertEqual(
            result.text,
            "Possible gunfire detected to the left. Maintain cover and confirm the contact.",
        )

    @patch("urllib.request.urlopen")
    def test_handles_connection_error_gracefully(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        event = _sample_threat_event()
        sop = _sample_sop()

        result = generate_advisory(event, sop)

        self.assertIsInstance(result, AdvisoryResult)
        self.assertFalse(result.available)
        self.assertEqual(result.model, "unavailable")
        self.assertEqual(result.text, UNAVAILABLE_MESSAGE)

    @patch("urllib.request.urlopen")
    def test_handles_timeout_gracefully(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = TimeoutError("Request timed out")

        event = _sample_threat_event()
        sop = _sample_sop()

        result = generate_advisory(event, sop)

        self.assertFalse(result.available)
        self.assertEqual(result.model, "unavailable")
        self.assertIn("unavailable", result.text.lower())

    @patch("urllib.request.urlopen")
    def test_handles_malformed_json_gracefully(self, mock_urlopen) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"NOT VALID JSON"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        event = _sample_threat_event()
        sop = _sample_sop()

        result = generate_advisory(event, sop)

        self.assertFalse(result.available)
        self.assertEqual(result.model, "unavailable")

    @patch("urllib.request.urlopen")
    def test_check_llm_status_detects_online_and_offline(self, mock_urlopen) -> None:
        # Test offline
        mock_urlopen.side_effect = urllib.error.URLError("Down")
        online, model = check_llm_status()
        self.assertFalse(online)
        self.assertEqual(model, "unavailable")

        # Test online
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(
            {"models": [{"name": "tinyllama"}, {"name": "llama3:8b"}]}
        ).encode("utf-8")
        mock_urlopen.side_effect = None
        mock_urlopen.return_value.__enter__.return_value = mock_response

        online, model = check_llm_status()
        self.assertTrue(online)
        self.assertEqual(model, "tinyllama")


if __name__ == "__main__":
    unittest.main()
