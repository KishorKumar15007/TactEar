"""Focused tests for the source-audio contract and WAV loader."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from tactear.audio import AudioData, load_audio
from tactear.events import ClassificationResult


class LoadAudioTests(unittest.TestCase):
    def test_loads_mono_wav_as_floating_point_audio(self) -> None:
        waveform = np.array([0.0, 0.25, -0.5, 0.75], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mono.wav"
            sf.write(path, waveform, 16_000, subtype="FLOAT")
            audio = load_audio(path)

        self.assertIsInstance(audio, AudioData)
        self.assertEqual(audio.sample_rate, 16_000)
        self.assertEqual(audio.waveform.ndim, 1)
        self.assertTrue(np.issubdtype(audio.waveform.dtype, np.floating))
        self.assertGreater(audio.waveform.size, 0)
        np.testing.assert_allclose(audio.waveform, waveform)

    def test_averages_stereo_wav_to_mono(self) -> None:
        stereo = np.array(
            [[0.2, 0.6], [-0.8, 0.4], [0.0, 1.0]], dtype=np.float32
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, stereo, 22_050, subtype="FLOAT")
            audio = load_audio(path)

        self.assertEqual(audio.sample_rate, 22_050)
        self.assertEqual(audio.waveform.ndim, 1)
        np.testing.assert_allclose(audio.waveform, stereo.mean(axis=1))

    def test_rejects_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_audio("does-not-exist.wav")

    def test_rejects_non_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-audio.wav"
            path.write_text("not a WAV file", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_audio(path)

    def test_rejects_empty_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.wav"
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16_000)

            with self.assertRaisesRegex(ValueError, "empty"):
                load_audio(path)

    def test_classification_result_contract(self) -> None:
        result = ClassificationResult("Speech", "VOICE", 0.95)

        self.assertEqual(result.raw_label, "Speech")
        self.assertEqual(result.tactical_label, "VOICE")
        self.assertEqual(result.confidence, 0.95)


if __name__ == "__main__":
    unittest.main()
