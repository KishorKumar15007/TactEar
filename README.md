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

## What Is Simulated

Physical left/right microphone timing is currently simulated with controlled ground-truth direction. The pipeline records this with `hardware_simulated=True`; its GCC-PHAT localization estimate is derived independently from the two simulated waveforms.

## What Is Not Yet Implemented

- Physical headset
- IMU
- Haptic hardware
- Noise suppression
- Shared multitask encoder
- Custom-trained model
- SOP retrieval
- Local LLM
- UI

## Run Instructions

PowerShell on Windows:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
python scripts/run_demo.py demo_audio\gunshot.wav --angle -45
```

The demo requires a valid review WAV. The repository currently contains no review clip; place one in `demo_audio/` before running the final command.

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
