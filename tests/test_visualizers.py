"""Unit tests for tactical SVG and HTML visualizers."""

from __future__ import annotations

import unittest
import numpy as np

from tactear.ui.visualizers import (
    render_haptic_motors_html,
    render_mic_simulation_html,
    render_pipeline_flow_html,
    render_radar_svg,
    render_waveform_svg,
)


class VisualizerTests(unittest.TestCase):
    def test_radar_svg_standby(self) -> None:
        svg = render_radar_svg()
        self.assertIn("<svg", svg)
        self.assertIn("0° FRONT", svg)
        self.assertIn("STANDBY", svg)

    def test_radar_svg_with_angles(self) -> None:
        svg = render_radar_svg(
            estimated_angle_deg=-45.6,
            ground_truth_angle_deg=-45.0,
            direction="LEFT",
            error_deg=0.6,
        )
        self.assertIn("-45.6°", svg)
        self.assertIn("TRUTH: -45°", svg)
        self.assertIn("BEARING: <tspan fill=\"#00e5ff\" font-weight=\"700\">LEFT</tspan>", svg)
        self.assertIn("ERR: 0.6°", svg)

    def test_waveform_svg_empty_and_valid(self) -> None:
        empty = render_waveform_svg(None)
        self.assertIn("Awaiting source audio", empty)

        data = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        wf_svg = render_waveform_svg(data, sample_rate=16000, duration_s=1.0)
        self.assertIn("<svg", wf_svg)
        self.assertIn("16000 Hz", wf_svg)
        self.assertIn("1.00s", wf_svg)
        self.assertIn("<polyline", wf_svg)

    def test_pipeline_flow_html(self) -> None:
        stages = {
            "AUDIO": "COMPLETE",
            "AST AI": "COMPLETE",
            "LOCAL LLM": "OFFLINE",
        }
        flow = render_pipeline_flow_html(stages)
        self.assertIn("LIVE PIPELINE ARCHITECTURE", flow)
        self.assertIn("COMPLETE", flow)
        self.assertIn("OFFLINE", flow)

    def test_haptic_motors_html(self) -> None:
        left_haptic = render_haptic_motors_html("LEFT")
        self.assertIn("LEFT MOTOR (TRANSDUCER L)", left_haptic)
        self.assertIn("ACTIVE [PULSING]", left_haptic)

        right_haptic = render_haptic_motors_html("RIGHT")
        self.assertIn("RIGHT MOTOR (TRANSDUCER R)", right_haptic)
        self.assertIn("ACTIVE [PULSING]", right_haptic)

    def test_mic_simulation_html(self) -> None:
        mic_html = render_mic_simulation_html(
            injected_delay_s=-0.00035,
            estimated_delay_s=-0.00035,
            angle_deg=-45.0,
        )
        self.assertIn("HEADSET MIC GEOMETRY", mic_html)
        self.assertIn("MIC L (0.00m)", mic_html)
        self.assertIn("Left leads by", mic_html)


if __name__ == "__main__":
    unittest.main()
