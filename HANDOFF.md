# TactEar Core Audio Handoff

## YOU RECEIVE

`ThreatEvent`

## YOUR WORK

```text
ThreatEvent
→ SOP lookup
→ local LLM advisory
→ Streamlit UI
```

Use `sound_class` and `classification_confidence` for the acoustic assessment. Use `direction` and `estimated_angle_deg` for the estimated source location. `ground_truth_angle_deg` and `localization_error_deg` are simulation-review fields, and `hardware_simulated` must remain visible until real headset hardware replaces the simulator.

Depend only on the public `run_pipeline(...) -> ThreatEvent` contract. Do not depend on or rewrite the internal classifier, simulator, or localization implementation unless a verified bug requires it.
