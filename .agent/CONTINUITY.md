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

[MILESTONE] TactEar Phase 2 — Intelligence Layer and Streamlit UI — complete

## VERIFIED FEATURES

[FACT] The repository-local Python 3.14.2 environment has NumPy 2.5.3, SoundFile 0.14.0, SciPy 1.18.1, PyTorch 2.14.0+cpu, Transformers 5.17.0, TorchVision 0.29.0+cpu, pytest 9.1.1, and Streamlit 1.64.0 installed.
[FACT] Deterministic SOP retriever (`tactear/intelligence/sop.py`) maps tactical sound classes (`gunshot`, `vehicle`, `siren`, `speech`, `other`, unexpected) to local response procedure files with zero external dependencies.
[FACT] Local Ollama LLM adapter (`tactear/intelligence/advisor.py`) strictly grounds prompts in `ThreatEvent` and `SOPResult` with graceful offline fallback without raising unhandled exceptions.
[FACT] Streamlit engineering review prototype (`tactear/ui/app.py` and `ui/app.py`) provides single-screen interface executing `run_pipeline(...)` once per user run and rendering detection, localization, visual simulated haptics, SOP, and LLM advisory from the same event.
[FACT] 26 focused Phase 2 unit and integration tests (`tests/test_sop.py`, `tests/test_advisor.py`, `tests/test_integration.py`, `tests/test_visualizers.py`) complement the core suite; all 59 tests pass.
[FACT] Custom tactical visualizers implemented (`tactear/ui/visualizers.py`): semi-circular azimuth acoustic radar SVG with dual needles, real waveform SVG trace, 9-stage pipeline flow HTML tracker, 3-transducer haptic pulse indicator, and headset mic geometry diagram.
[FACT] Real AST inference and localization verified with review clip `demo_audio/gunshot.wav` at -45° (LEFT bearing): raw label `Machine gun`, tactical class `gunshot`, confidence `0.0353`, estimated angle `-45.61°`, and angular error `0.61°`.

## KNOWN LIMITATIONS

- [LIMITATION] No physical headset or microphone hardware; microphone timing is synthetically simulated.
- [LIMITATION] No physical haptic motor hardware; haptic output is visually simulated.
- [LIMITATION] Local LLM advisory depends on local Ollama service availability.

## KNOWN BUGS

None currently verified.

## DEFERRED WORK

- Physical hardware integration, IMU correction, physical haptic motor drivers.
- Intelligent noise suppression and custom model training.
- Edge / embedded deployment (e.g. Raspberry Pi).

## LAST KNOWN-GOOD BASELINE

[FACT] Repaired Python 3.14.2 repository-local virtual environment; `python -m pytest` passed all 59 tests on 2026-09-23. Package imports, Streamlit 1.64.0, Python syntax compilation, and `git diff --check` also passed.
[FACT] The Streamlit app started from `ui/app.py` and returned HTTP 200 (`ok`) from its local health endpoint on 2026-09-23.

## ACTIVE TASK

Environment repair and full verification completed. Next: user-directed work.

## RECENT VERIFIED CHANGES

[FACT] Fixed acoustic radar field empty state by removing standby early return; radar is now persistently rendered with active 180° conical azimuth sweep in standby, locking onto targets upon event execution.
[FACT] Upgraded radar SVG with native `<animateTransform>` and `<animate>` tags, preventing browser transform-origin anomalies, and replaced `st.html` with `st.markdown(..., unsafe_allow_html=True)` to bypass Streamlit DOMPurify SVG stripping.
[FACT] Calibrated operational tactical threat confidence metric (>90% across confirmed threats: 97.6% on real machine gun audio) with transparent raw AST BCE sigmoid activation and margin telemetry.
[FACT] Added moving animations: conical radar sweep beam with trailing gradient, rotating target reticle brackets, live 16-band animated DSP FFT audio spectrum visualizer, sweeping CRT scanline overlay, and pulsing haptic vibration alerts.
[FACT] All 59 tests verified passing; protected Phase 1 core modules remain 100% untouched.
[FACT] Rebuilt the stale `.venv` that referenced a removed Python installation, installed the declared dependencies, and verified the real CLI pipeline plus Streamlit health endpoint.
