# PROJECT STATE

## ARCHITECTURE

[FACT] Minimal local Python package with validated source-audio loading, a replaceable pretrained AST classifier adapter, deterministic simulated headset stereo input, signal-only GCC-PHAT localization, and a structured core pipeline.

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

[MILESTONE] TactEar Core Audio Baseline — complete

## VERIFIED FEATURES

[FACT] The repository-local `tactear` package imports successfully with `.venv` Python 3.14.2.
[FACT] `load_audio` decodes valid WAV files into finite, floating-point, mono `AudioData`, preserving sample rate and averaging decoded multi-channel source audio.
[FACT] Missing, unreadable, and empty audio inputs fail clearly.
[FACT] `AudioClassifier` lazily integrates `MIT/ast-finetuned-audioset-10-10-0.448-v2` behind a TactEar-owned interface and resamples only its temporary inference waveform to the feature extractor's declared sample rate.
[FACT] Tactical mapping and top-prediction selection preserve model labels and unmodified model scores; unit tests cover mapping, selection, and per-instance model reuse.
[FACT] Eleven focused audio/classifier tests passed with NumPy 2.5.3, SoundFile 0.14.0, PyTorch 2.14.0+cpu, Transformers 5.17.0, and SciPy 1.18.1.
[FACT] Simulated stereo generation verified: the 0 degrees front, negative-left, positive-right convention uses signed left-arrival-time minus right-arrival-time TDOA; delayed channels are zero-padded without wraparound.
[FACT] GCC-PHAT localization uses only left/right signals, sample rate, and microphone geometry; its peak search is physically constrained to 0.18 / 343.0 seconds and preserves the simulator's signed-delay convention.
[FACT] Deterministic 16 kHz validation estimated -56.48, -28.45, 0.00, 28.45, and 56.48 degrees for injected -60, -30, 0, 30, and 60 degrees respectively, with LEFT/FRONT/RIGHT labels correct.
[FACT] `run_pipeline` connects mono loading, unmodified classifier output, simulated stereo, signal-only localization, and a `ThreatEvent`; tests verify classifier reuse and post-estimation localization error calculation.
[FACT] `scripts/run_demo.py` accepts an audio path and simulated angle, calls the public pipeline, and renders returned event values without duplicating pipeline logic.
[FACT] README and HANDOFF document the public `run_pipeline(...) -> ThreatEvent` boundary for the second developer.
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

[FACT] Python 3.14.2 local virtual environment; `python -m pytest` passed all 33 audio, classifier, simulator, localization, and pipeline tests on 2026-09-22. Package import and CLI help checks also passed.
[FACT] Fresh 16 kHz five-angle localization validation: injected -60/-30/0/+30/+60 degrees estimated -56.48/-28.45/0.00/+28.45/+56.48 degrees, with LEFT/LEFT/FRONT/RIGHT/RIGHT labels.

## ACTIVE TASK

Next milestone: Advisory/UI integration by second developer

## RECENT VERIFIED CHANGES

[FACT] Completed review handoff materials: thin CLI demo runner, current README, and `HANDOFF.md` integration guidance.
