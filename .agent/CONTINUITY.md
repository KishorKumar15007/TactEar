# PROJECT STATE

## ARCHITECTURE

[FACT] Minimal local Python package with validated source-audio loading, a classification-result contract, and a replaceable pretrained AST classifier adapter.

Planned current core flow:

```text
audio
→ classifier
→ stereo simulator
→ GCC-PHAT localization
→ ThreatEvent
→ pipeline
```

## LOCKED DECISIONS

- [LOCKED] Python core with a real pretrained classifier baseline; no model training in this milestone.
- [LOCKED] Microphone hardware is simulated; GCC-PHAT must estimate direction from signals without reading simulator ground truth.
- [LOCKED] The final output is a structured `ThreatEvent`.
- [LOCKED] Another developer owns SOP retrieval, local LLM, and UI work; no cloud, database, API, or frontend is in this milestone.

## CURRENT MILESTONE

[MILESTONE] TactEar Core Audio Baseline

## VERIFIED FEATURES

[FACT] The repository-local `tactear` package imports successfully with `.venv` Python 3.14.2.
[FACT] `load_audio` decodes valid WAV files into finite, floating-point, mono `AudioData`, preserving sample rate and averaging decoded multi-channel source audio.
[FACT] Missing, unreadable, and empty audio inputs fail clearly.
[FACT] `AudioClassifier` lazily integrates `MIT/ast-finetuned-audioset-10-10-0.448-v2` behind a TactEar-owned interface and resamples only its temporary inference waveform to the feature extractor's declared sample rate.
[FACT] Tactical mapping and top-prediction selection preserve model labels and unmodified model scores; unit tests cover mapping, selection, and per-instance model reuse.
[FACT] Eleven focused audio/classifier tests passed with NumPy 2.5.3, SoundFile 0.14.0, PyTorch 2.14.0+cpu, Transformers 5.17.0, and SciPy 1.18.1.
[UNKNOWN] Real pretrained inference awaits a review WAV in `demo_audio/`.

## KNOWN LIMITATIONS

- No physical headset, microphones, IMU, or haptic motor.
- No noise suppression, shared multitask encoder, or custom trained classifier.
- No SOP retrieval, LLM, or UI.

## KNOWN BUGS

None currently verified.

## DEFERRED WORK

- Hardware integration, IMU correction, haptic drivers, noise suppression, and shared encoder.
- Custom model training and Raspberry Pi deployment.
- SOP retrieval, local LLM, and UI.

## LAST KNOWN-GOOD BASELINE

[FACT] Python 3.14.2 local virtual environment; audio/classifier unit tests, package imports, and dependency checks pass.

## ACTIVE TASK

Next: Simulated stereo headset input

## RECENT VERIFIED CHANGES

[FACT] Added the lazy pretrained AST adapter, explicit tactical mapping, safe inference-only resampling, and no-download classifier unit tests.
