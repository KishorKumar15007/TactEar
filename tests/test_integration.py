"""Lightweight integration tests for Phase 2 downstream intelligence flow."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from tactear.events import ThreatEvent
from tactear.intelligence.advisor import generate_advisory
from tactear.intelligence.sop import retrieve_sop


def _mock_pipeline_run(
    sound_class: str = "gunshot",
    direction: str = "LEFT",
    angle_deg: float = -45.0,
) -> ThreatEvent:
    return ThreatEvent(
        event_id="int-test-event-uuid-001",
        timestamp="2026-09-22T14:30:00+00:00",
        source_audio="demo_audio/gunshot.wav",
        sound_class=sound_class,
        raw_model_label="Gunshot, gunfire",
        classification_confidence=0.92,
        direction=direction,
        estimated_angle_deg=angle_deg + 2.0,
        ground_truth_angle_deg=angle_deg,
        localization_error_deg=2.0,
        estimated_delay_seconds=-0.00035,
        hardware_simulated=True,
    )


def map_direction_to_haptics(direction: str) -> dict[str, bool]:
    """Map authoritative event direction to simulated haptic motor indicators."""
    normalized = direction.strip().upper()
    return {
        "left_motor": normalized == "LEFT",
        "center_motor": normalized == "FRONT",
        "right_motor": normalized == "RIGHT",
    }


class Phase2DownstreamIntegrationTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_complete_downstream_flow_left_gunshot(self, mock_urlopen) -> None:
        # Mock LLM advisory response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(
            {"response": "Gunfire detected to the left. Take cover and report bearing."}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # 1. Pipeline produces ThreatEvent (mocked boundary)
        event = _mock_pipeline_run(sound_class="gunshot", direction="LEFT", angle_deg=-45.0)

        # 2. SOP Retrieval
        sop = retrieve_sop(event)
        self.assertEqual(sop.sop_id, "gunshot")
        self.assertIn("Potential gunfire detected", sop.content)

        # 3. Local Advisory Generation
        advisory = generate_advisory(event, sop, model="tinyllama")
        self.assertTrue(advisory.available)
        self.assertEqual(advisory.model, "tinyllama")
        self.assertEqual(
            advisory.text,
            "Gunfire detected to the left. Take cover and report bearing.",
        )

        # 4. Haptic Mapping from event.direction
        haptics = map_direction_to_haptics(event.direction)
        self.assertTrue(haptics["left_motor"])
        self.assertFalse(haptics["center_motor"])
        self.assertFalse(haptics["right_motor"])

    @patch("urllib.request.urlopen")
    def test_complete_downstream_flow_right_vehicle_with_offline_llm(
        self, mock_urlopen
    ) -> None:
        # Simulate Ollama offline
        mock_urlopen.side_effect = ConnectionRefusedError("Ollama port 11434 unreachable")

        # 1. ThreatEvent
        event = _mock_pipeline_run(sound_class="vehicle", direction="RIGHT", angle_deg=45.0)

        # 2. SOP Retrieval
        sop = retrieve_sop(event)
        self.assertEqual(sop.sop_id, "vehicle")
        self.assertIn("Potential vehicle activity detected", sop.content)

        # 3. Local Advisory gracefully reports offline without crashing
        advisory = generate_advisory(event, sop)
        self.assertFalse(advisory.available)
        self.assertEqual(advisory.model, "unavailable")
        self.assertIn("unavailable", advisory.text.lower())

        # 4. Haptic Mapping accurately reflects RIGHT
        haptics = map_direction_to_haptics(event.direction)
        self.assertFalse(haptics["left_motor"])
        self.assertFalse(haptics["center_motor"])
        self.assertTrue(haptics["right_motor"])

    def test_haptic_mapping_covers_front_direction(self) -> None:
        haptics = map_direction_to_haptics("FRONT")
        self.assertFalse(haptics["left_motor"])
        self.assertTrue(haptics["center_motor"])
        self.assertFalse(haptics["right_motor"])


if __name__ == "__main__":
    unittest.main()
