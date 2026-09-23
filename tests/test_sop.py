"""Unit tests for deterministic local SOP retrieval."""

from __future__ import annotations

import unittest

from tactear.events import ThreatEvent
from tactear.intelligence.sop import SOPResult, retrieve_sop


def _dummy_event(sound_class: str) -> ThreatEvent:
    return ThreatEvent(
        event_id="test-event-uuid",
        timestamp="2026-09-22T00:00:00+00:00",
        source_audio="demo_audio/test.wav",
        sound_class=sound_class,
        raw_model_label="Test label",
        classification_confidence=0.88,
        direction="LEFT",
        estimated_angle_deg=-45.0,
        ground_truth_angle_deg=-45.0,
        localization_error_deg=0.0,
        estimated_delay_seconds=-0.00037,
        hardware_simulated=True,
    )


class SOPRetrievalTests(unittest.TestCase):
    def test_gunshot_retrieval(self) -> None:
        result = retrieve_sop("gunshot")
        self.assertIsInstance(result, SOPResult)
        self.assertEqual(result.sop_id, "gunshot")
        self.assertEqual(result.title, "Gunfire Response Procedure")
        self.assertIn("Potential gunfire detected", result.content)
        self.assertIn("Maintain cover and situational awareness", result.content)

    def test_vehicle_retrieval(self) -> None:
        result = retrieve_sop("vehicle")
        self.assertEqual(result.sop_id, "vehicle")
        self.assertEqual(result.title, "Vehicle Activity Response Procedure")
        self.assertIn("Potential vehicle activity detected", result.content)

    def test_siren_retrieval(self) -> None:
        result = retrieve_sop("siren")
        self.assertEqual(result.sop_id, "siren")
        self.assertEqual(result.title, "Emergency Signal Response Procedure")
        self.assertIn("Emergency or warning signal detected", result.content)

    def test_speech_retrieval(self) -> None:
        result = retrieve_sop("speech")
        self.assertEqual(result.sop_id, "speech")
        self.assertEqual(result.title, "Speech Detection Procedure")
        self.assertIn("Human speech detected", result.content)

    def test_other_maps_to_unknown(self) -> None:
        result = retrieve_sop("other")
        self.assertEqual(result.sop_id, "unknown")
        self.assertEqual(result.title, "Unclassified Acoustic Event Procedure")
        self.assertIn("Unclassified acoustic event detected", result.content)

    def test_unexpected_class_maps_to_unknown(self) -> None:
        result = retrieve_sop("explosion_or_dragon")
        self.assertEqual(result.sop_id, "unknown")
        self.assertEqual(result.title, "Unclassified Acoustic Event Procedure")
        self.assertIn("Unclassified acoustic event detected", result.content)

    def test_case_and_whitespace_insensitivity(self) -> None:
        result = retrieve_sop("   GUNSHOT  \n")
        self.assertEqual(result.sop_id, "gunshot")
        self.assertEqual(result.title, "Gunfire Response Procedure")

    def test_threat_event_input_accepted(self) -> None:
        event = _dummy_event("gunshot")
        result = retrieve_sop(event)
        self.assertEqual(result.sop_id, "gunshot")
        self.assertEqual(result.title, "Gunfire Response Procedure")
        self.assertIn("gunfire", result.content.lower())

    def test_content_is_non_empty_and_loaded_from_disk(self) -> None:
        for sound_class in ("gunshot", "vehicle", "siren", "speech", "other", "unknown"):
            with self.subTest(sound_class=sound_class):
                result = retrieve_sop(sound_class)
                self.assertTrue(len(result.content) > 30)
                self.assertIn("TactEar Prototype Response Procedure", result.content)


if __name__ == "__main__":
    unittest.main()
