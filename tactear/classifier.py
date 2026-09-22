"""Pretrained AST audio-classification adapter for TactEar."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.signal import resample_poly

from tactear.audio import AudioData
from tactear.events import ClassificationResult


MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.448-v2"


class ClassifierInitializationError(RuntimeError):
    """Raised when the pretrained classifier cannot be initialized."""


class ClassifierInferenceError(RuntimeError):
    """Raised when audio cannot be processed by the pretrained classifier."""


def map_to_tactical_label(raw_label: str) -> str:
    """Map an AudioSet-family label to one TactEar tactical category."""

    normalized = raw_label.casefold()

    if any(
        label in normalized
        for label in (
            "siren",
            "emergency vehicle",
            "police car (siren)",
            "ambulance (siren)",
            "fire engine, fire truck (siren)",
        )
    ):
        return "siren"
    if any(
        label in normalized
        for label in ("gunshot", "gunfire", "machine gun", "fusillade", "cap gun", "firecracker")
    ):
        return "gunshot"
    if any(
        label in normalized
        for label in (
            "motor vehicle",
            "vehicle",
            "car",
            "truck",
            "motorcycle",
            "engine",
            "idling",
        )
    ):
        return "vehicle"
    if any(
        label in normalized
        for label in ("speech", "conversation", "narration")
    ):
        return "speech"
    return "other"


def select_tactical_prediction(
    predictions: Sequence[tuple[str, float]],
) -> tuple[str, str, float]:
    """Choose the first tactically relevant entry from descending predictions.

    The selected confidence is the unmodified score emitted by the pretrained
    model. If no entry maps to a tactical category, the strongest model label
    is returned as ``other``.
    """

    if not predictions:
        raise ValueError("At least one model prediction is required")

    for raw_label, confidence in predictions:
        tactical_label = map_to_tactical_label(raw_label)
        if tactical_label != "other":
            return raw_label, tactical_label, confidence

    raw_label, confidence = predictions[0]
    return raw_label, "other", confidence


class AudioClassifier:
    """Lazy CPU adapter for the default pretrained AST AudioSet classifier.

    One instance loads the feature extractor and model at most once. The AST
    feature extractor supplies its required sample rate; audio is resampled for
    inference only, preserving the caller's original ``AudioData``.
    """

    model_id = MODEL_ID

    def __init__(self, top_k: int = 5) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.top_k = top_k
        self._feature_extractor = None
        self._model = None
        self._sample_rate: int | None = None

    def classify(self, audio: AudioData) -> ClassificationResult:
        """Run real pretrained inference and return the selected model score."""

        self._validate_audio(audio)
        self._ensure_loaded()
        assert self._feature_extractor is not None
        assert self._model is not None
        assert self._sample_rate is not None

        waveform = self._resample_for_model(audio.waveform, audio.sample_rate)
        try:
            inputs = self._feature_extractor(
                waveform, sampling_rate=self._sample_rate, return_tensors="pt"
            )
            import torch

            with torch.inference_mode():
                outputs = self._model(**inputs)
            probabilities = torch.sigmoid(outputs.logits[0]).detach().cpu()
            top_count = min(self.top_k, probabilities.numel())
            scores, indices = torch.topk(probabilities, top_count)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ClassifierInferenceError("Pretrained AST inference failed") from exc

        predictions = [
            (self._label_for_index(int(index)), float(score))
            for score, index in zip(scores.tolist(), indices.tolist(), strict=True)
        ]
        raw_label, tactical_label, confidence = select_tactical_prediction(predictions)
        return ClassificationResult(raw_label, tactical_label, confidence)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:
            raise ClassifierInitializationError(
                "Transformers and PyTorch are required for pretrained classification"
            ) from exc

        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            model = AutoModelForAudioClassification.from_pretrained(self.model_id)
        except Exception as exc:
            raise ClassifierInitializationError(
                f"Could not load pretrained audio model {self.model_id!r}"
            ) from exc

        sample_rate = getattr(feature_extractor, "sampling_rate", None)
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ClassifierInitializationError(
                "The pretrained feature extractor did not declare a valid sample rate"
            )

        model.eval()
        self._feature_extractor = feature_extractor
        self._model = model
        self._sample_rate = sample_rate

    def _label_for_index(self, index: int) -> str:
        assert self._model is not None
        labels = self._model.config.id2label
        raw_label = labels.get(index, labels.get(str(index)))
        if not isinstance(raw_label, str):
            raise ClassifierInferenceError(f"No model label exists for output index {index}")
        return raw_label

    def _resample_for_model(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        assert self._sample_rate is not None
        if sample_rate == self._sample_rate:
            return waveform

        ratio = Fraction(self._sample_rate, sample_rate).limit_denominator()
        return resample_poly(waveform, ratio.numerator, ratio.denominator).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _validate_audio(audio: AudioData) -> None:
        if not isinstance(audio, AudioData):
            raise TypeError("audio must be an AudioData instance")
        if audio.sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive")
        if audio.waveform.ndim != 1 or audio.waveform.size == 0:
            raise ValueError("Audio waveform must be a non-empty one-dimensional array")
        if not np.issubdtype(audio.waveform.dtype, np.floating):
            raise ValueError("Audio waveform must use a floating-point dtype")
        if not np.isfinite(audio.waveform).all():
            raise ValueError("Audio waveform contains non-finite samples")
