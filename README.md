# TactEar

TactEar is an early software prototype for a future wearable tactical acoustic-awareness system.

## What Works

- WAV loading
- Real pretrained audio classification using `MIT/ast-finetuned-audioset-10-10-0.448-v2`
- TactEar tactical-category mapping
- Simulated binaural/headset input
- GCC-PHAT localization
- Structured `ThreatEvent` output
- End-to-end core pipeline
- CLI demo
- Deterministic local SOP retrieval
- Grounded local-Ollama advisory adapter with an offline fallback
- Streamlit engineering review UI

## What Is Simulated

Physical left/right microphone timing is currently simulated with controlled ground-truth direction. The pipeline records this with `hardware_simulated=True`; its GCC-PHAT localization estimate is derived independently from the two simulated waveforms.

## What Is Not Yet Implemented

- Physical headset
- IMU
- Haptic hardware
- Noise suppression
- Shared multitask encoder
- Custom-trained model

## Run Instructions

PowerShell on Windows:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
python scripts/run_demo.py demo_audio\gunshot.wav --angle -45
```

The demo requires a valid review WAV. Place review clips in `demo_audio/` or upload via the UI.

### Phase 2 Streamlit Web Prototype

Launch the technical engineering review prototype:

```powershell
.\.venv\Scripts\streamlit.exe run ui/app.py
```
(or `streamlit run tactear/ui/app.py`)

#### Local Model Requirement
The local LLM advisory component connects to **Ollama** (`http://127.0.0.1:11434`). If Ollama is offline or unavailable, the prototype degrades gracefully: acoustic classification, GCC-PHAT localization, SOP retrieval, and simulated haptic visualization remain fully functional while the LLM status displays `OFFLINE`.

#### Demonstration Workflow
1. Select a demo audio file (or upload a `.wav` file).
2. Adjust the `SIMULATED SOURCE ANGLE` slider (e.g., `-45°` for left bearing, `+45°` for right bearing).
3. Click `RUN TACTEAR`.
4. Inspect real-time **Acoustic Detection** (class and confidence from pretrained AST).
5. Inspect **GCC-PHAT Localization** (estimated angle and direction).
6. Inspect **Haptic Output — Simulated** (visual transducer indication).
7. Inspect the retrieved local **SOP** and grounded **Local Situational Advisory**.

#### Simulation Honesty
- **Microphone timing**: Synthetically generated based on microphone geometry and ground-truth angle.
- **Haptic output**: Visually simulated transducer activation; no physical actuators are connected.
- **Acoustic classification & GCC-PHAT localization**: Real software inference on the supplied waveforms.

## Teammate Integration Contract

```python
from tactear.pipeline import run_pipeline

event = run_pipeline(
    audio_path="demo_audio/gunshot.wav",
    simulated_angle_deg=-45,
)
```

`event` is a `ThreatEvent` with these fields:

- `event_id`: generated event UUID
- `timestamp`: UTC ISO-8601 creation time
- `source_audio`: source file path
- `sound_class`: TactEar tactical category
- `raw_model_label`: original pretrained-model label
- `classification_confidence`: unmodified selected model score
- `direction`: `LEFT`, `FRONT`, or `RIGHT`
- `estimated_angle_deg`: GCC-PHAT estimate
- `ground_truth_angle_deg`: simulated source angle
- `localization_error_deg`: absolute difference between estimate and simulated ground truth
- `estimated_delay_seconds`: signal-derived TDOA estimate
- `hardware_simulated`: whether microphone timing was simulated

Consume `ThreatEvent` as the integration boundary. Do not rewrite the audio/classification/localization internals unless a verified bug requires it.
