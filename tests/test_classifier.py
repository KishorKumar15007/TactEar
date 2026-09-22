"""Unit tests for TactEar's classifier policy without model downloads."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from tactear.audio import AudioData
from tactear.classifier import (
    AudioClassifier,
    map_to_tactical_label,
    select_tactical_prediction,
)


class TacticalMappingTests(unittest.TestCase):
    def test_maps_audioset_family_labels(self) -> None:
        cases = {
            "Gunshot, gunfire": "gunshot",
            "Machine gun": "gunshot",
            "Car": "vehicle",
            "Engine": "vehicle",
            "Siren": "siren",
            "Emergency vehicle": "siren",
            "Speech": "speech",
            "Bird": "other",
        }

        for raw_label, expected_label in cases.items():
            with self.subTest(raw_label=raw_label):
                self.assertEqual(map_to_tactical_label(raw_label), expected_label)

    def test_mapping_is_case_insensitive(self) -> None:
        self.assertEqual(map_to_tactical_label("mAcHiNe GuN"), "gunshot")
        self.assertEqual(map_to_tactical_label("FEMALE SPEECH"), "speech")

    def test_selects_highest_relevant_prediction_in_model_order(self) -> None:
        predictions = [("Bird", 0.92), ("Siren", 0.83), ("Engine", 0.75)]

        self.assertEqual(
            select_tactical_prediction(predictions), ("Siren", "siren", 0.83)
        )

    def test_preserves_top_prediction_when_all_are_other(self) -> None:
        predictions = [("Bird", 0.91), ("Wind", 0.62)]

        self.assertEqual(select_tactical_prediction(predictions), ("Bird", "other", 0.91))


class _FakeFeatureExtractor:
    sampling_rate = 16_000

    def __call__(self, waveform, *, sampling_rate, return_tensors):
        self.last_waveform = waveform
        self.last_sample_rate = sampling_rate
        self.last_return_tensors = return_tensors
        return {"input_values": torch.from_numpy(waveform[np.newaxis, :])}


class _FakeModel:
    class config:
        id2label = {0: "Bird", 1: "Speech", 2: "Siren"}

    def eval(self):
        return self

    def __call__(self, **inputs):
        return type("Output", (), {"logits": torch.tensor([[0.1, 0.8, 1.2]])})()


class AudioClassifierTests(unittest.TestCase):
    def test_reuses_loaded_model_and_preserves_selected_score(self) -> None:
        classifier = AudioClassifier(top_k=3)
        extractor = _FakeFeatureExtractor()
        model = _FakeModel()
        audio = AudioData(np.array([0.0, 0.1, -0.1], dtype=np.float32), 16_000)

        with (
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained", return_value=extractor
            ) as load_extractor,
            patch(
                "transformers.AutoModelForAudioClassification.from_pretrained", return_value=model
            ) as load_model,
        ):
            first = classifier.classify(audio)
            second = classifier.classify(audio)

        load_extractor.assert_called_once_with(classifier.model_id)
        load_model.assert_called_once_with(classifier.model_id)
        self.assertEqual(first.raw_label, "Siren")
        self.assertEqual(first.tactical_label, "siren")
        self.assertEqual(second.confidence, first.confidence)
        self.assertEqual(first.confidence, float(torch.sigmoid(torch.tensor(1.2))))


if __name__ == "__main__":
    unittest.main()
