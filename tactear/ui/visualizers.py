"""Tactical SVG and HTML visualization components for TactEar."""

from __future__ import annotations

import math
import numpy as np


def render_radar_svg(
    estimated_angle_deg: float | None = None,
    ground_truth_angle_deg: float | None = None,
    direction: str = "FRONT",
    error_deg: float | None = None,
) -> str:
    """Generate a high-contrast semi-circular tactical radar azimuth display.

    Azimuth convention:
      -90 deg is Left (bearing 270 deg)
        0 deg is Front (bearing 0 deg / North)
      +90 deg is Right (bearing 90 deg)
    Radar center is at (160, 160), radius is 130.
    """
    cx, cy = 160, 160
    r = 130

    # Calculate estimated needle endpoint
    est_needle_svg = ""
    gt_needle_svg = ""
    error_arc_svg = ""

    if estimated_angle_deg is not None:
        # Angle relative to vertical: 0 is straight up, negative is left, positive is right
        est_rad = math.radians(estimated_angle_deg)
        ex = cx + r * math.sin(est_rad)
        ey = cy - r * math.cos(est_rad)

        beam_color = "#00e5ff" if direction != "FRONT" else "#10b981"
        est_needle_svg = f"""
        <!-- Estimated Direction Beam -->
        <line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{beam_color}" stroke-width="3.5" stroke-linecap="round" filter="url(#glow)"/>
        <!-- Radar Contact Target Ping & Rotating Reticle -->
        <g transform="translate({ex:.1f}, {ey:.1f})">
            <circle cx="0" cy="0" r="14" fill="none" stroke="{beam_color}" stroke-width="1.4">
                <animate attributeName="r" values="6; 16; 6" dur="1.8s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="1; 0.2; 1" dur="1.8s" repeatCount="indefinite"/>
            </circle>
            <circle cx="0" cy="0" r="5.5" fill="{beam_color}" stroke="#ffffff" stroke-width="1.5" filter="url(#glow)"/>
            <g>
                <path d="M -9,-4 L -9,-9 L -4,-9" stroke="{beam_color}" stroke-width="1.5" fill="none"/>
                <path d="M 4,-9 L 9,-9 L 9,-4" stroke="{beam_color}" stroke-width="1.5" fill="none"/>
                <path d="M -9,4 L -9,9 L -4,9" stroke="{beam_color}" stroke-width="1.5" fill="none"/>
                <path d="M 4,9 L 9,9 L 9,4" stroke="{beam_color}" stroke-width="1.5" fill="none"/>
                <animateTransform
                    attributeName="transform"
                    type="rotate"
                    from="0 0 0"
                    to="360 0 0"
                    dur="8s"
                    repeatCount="indefinite"
                />
            </g>
        </g>
        <text x="{ex + (14 if estimated_angle_deg >= 0 else -44):.1f}" y="{ey - 10:.1f}" fill="{beam_color}" font-family="JetBrains Mono, monospace" font-size="12" font-weight="800">
            {estimated_angle_deg:+.1f}°
        </text>
        """

    if ground_truth_angle_deg is not None:
        gt_rad = math.radians(ground_truth_angle_deg)
        gx = cx + r * math.sin(gt_rad)
        gy = cy - r * math.cos(gt_rad)

        gt_needle_svg = f"""
        <!-- Ground Truth Target -->
        <line x1="{cx}" y1="{cy}" x2="{gx:.1f}" y2="{gy:.1f}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4,4"/>
        <circle cx="{gx:.1f}" cy="{gy:.1f}" r="4" fill="#f59e0b"/>
        <text x="{gx + (10 if ground_truth_angle_deg >= 0 else -40):.1f}" y="{gy + 14:.1f}" fill="#f59e0b" font-family="JetBrains Mono, monospace" font-size="9">
            TRUTH: {ground_truth_angle_deg:+.0f}°
        </text>
        """

    # Error badge
    error_text = f"ERR: {error_deg:.1f}°" if error_deg is not None else "READY"
    dir_label = direction if estimated_angle_deg is not None else "STANDBY"

    svg = f"""
<div style="background: #080d1a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; box-shadow: inset 0 0 25px rgba(0, 229, 255, 0.08); width: 100%;">
<svg viewBox="0 0 320 190" style="width: 100%; height: auto; min-height: 215px; max-height: 260px; display: block; margin: 0 auto;">
    <defs>
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        <radialGradient id="radar-grid-grad" cx="50%" cy="100%" r="100%">
            <stop offset="0%" stop-color="rgba(0, 229, 255, 0.16)"/>
            <stop offset="60%" stop-color="rgba(0, 229, 255, 0.05)"/>
            <stop offset="100%" stop-color="rgba(8, 13, 26, 0.95)"/>
        </radialGradient>
        <linearGradient id="sweep-trail" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="rgba(0, 229, 255, 0.0)"/>
            <stop offset="100%" stop-color="rgba(0, 229, 255, 0.35)"/>
        </linearGradient>
    </defs>

    <!-- Radar Background Grid -->
    <path d="M 30,160 A 130,130 0 0,1 290,160 Z" fill="url(#radar-grid-grad)" stroke="#00e5ff" stroke-width="1.8"/>
    <path d="M 62.5,160 A 97.5,97.5 0 0,1 257.5,160" fill="none" stroke="rgba(0, 229, 255, 0.4)" stroke-width="1.2" stroke-dasharray="4,4"/>
    <path d="M 95,160 A 65,65 0 0,1 225,160" fill="none" stroke="rgba(0, 229, 255, 0.4)" stroke-width="1.2" stroke-dasharray="4,4"/>
    <path d="M 127.5,160 A 32.5,32.5 0 0,1 192.5,160" fill="none" stroke="rgba(0, 229, 255, 0.55)" stroke-width="1.2"/>

    <!-- Azimuth Radial Spokes -->
    <!-- -60 deg -->
    <line x1="{cx}" y1="{cy}" x2="{cx - r * 0.866:.1f}" y2="{cy - r * 0.5:.1f}" stroke="rgba(0, 229, 255, 0.25)" stroke-width="1.2"/>
    <!-- -30 deg -->
    <line x1="{cx}" y1="{cy}" x2="{cx - r * 0.5:.1f}" y2="{cy - r * 0.866:.1f}" stroke="rgba(0, 229, 255, 0.25)" stroke-width="1.2"/>
    <!-- 0 deg Center Front -->
    <line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - r}" stroke="#00e5ff" stroke-width="2"/>
    <!-- +30 deg -->
    <line x1="{cx}" y1="{cy}" x2="{cx + r * 0.5:.1f}" y2="{cy - r * 0.866:.1f}" stroke="rgba(0, 229, 255, 0.25)" stroke-width="1.2"/>
    <!-- +60 deg -->
    <line x1="{cx}" y1="{cy}" x2="{cx + r * 0.866:.1f}" y2="{cy - r * 0.5:.1f}" stroke="rgba(0, 229, 255, 0.25)" stroke-width="1.2"/>

    <!-- Active Conical Radar Sweep Sector with Native SVG Animation -->
    <g>
        <polygon points="160,160 134,34 160,30" fill="url(#sweep-trail)"/>
        <line x1="160" y1="160" x2="160" y2="30" stroke="#00e5ff" stroke-width="2.5" stroke-linecap="round" filter="url(#glow)"/>
        <animateTransform
            attributeName="transform"
            type="rotate"
            values="-88 160 160; 88 160 160; -88 160 160"
            keyTimes="0; 0.5; 1"
            dur="3.2s"
            repeatCount="indefinite"
        />
    </g>

    <!-- Baseline Axis -->
    <line x1="20" y1="160" x2="300" y2="160" stroke="#475569" stroke-width="2"/>

    <!-- Angle Text Labels -->
    <text x="22" y="152" fill="#94a3b8" font-family="JetBrains Mono, monospace" font-size="11" font-weight="700">-90°</text>
    <text x="135" y="20" fill="#00e5ff" font-family="JetBrains Mono, monospace" font-size="12" font-weight="800">0° FRONT</text>
    <text x="268" y="152" fill="#94a3b8" font-family="JetBrains Mono, monospace" font-size="11" font-weight="700">+90°</text>
    <text x="65" y="80" fill="#64748b" font-family="JetBrains Mono, monospace" font-size="10">-45°</text>
    <text x="230" y="80" fill="#64748b" font-family="JetBrains Mono, monospace" font-size="10">+45°</text>

    {gt_needle_svg}
    {est_needle_svg}

    <!-- Center Origin Sensor Node -->
    <circle cx="{cx}" y="160" r="7" fill="#0f172a" stroke="#00e5ff" stroke-width="2.5"/>
    <circle cx="{cx}" y="160" r="3.5" fill="#00e5ff"/>

    <!-- Tactical Status Legend in Radar -->
    <rect x="10" y="165" width="130" height="20" rx="3" fill="#0f172a" stroke="#1e293b"/>
    <text x="16" y="179" fill="#94a3b8" font-family="JetBrains Mono, monospace" font-size="9">BEARING: <tspan fill="#00e5ff" font-weight="700">{dir_label}</tspan></text>

    <rect x="180" y="165" width="130" height="20" rx="3" fill="#0f172a" stroke="#1e293b"/>
    <text x="186" y="179" fill="#94a3b8" font-family="JetBrains Mono, monospace" font-size="9">ACCURACY: <tspan fill="#10b981" font-weight="700">{error_text}</tspan></text>
</svg>
</div>
"""
    return svg


def render_waveform_svg(
    waveform: np.ndarray | None,
    sample_rate: int = 16000,
    duration_s: float | None = None,
) -> str:
    """Render a clean high-tech audio waveform plot from real numpy audio samples."""
    width = 460
    height = 100
    mid_y = height / 2

    if waveform is None or len(waveform) == 0:
        return f"""
        <div style="background: #080d1a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; text-align: center; color: #64748b; font-family: JetBrains Mono, monospace; font-size: 0.8rem; min-height: 120px; display: flex; align-items: center; justify-content: center;">
            Awaiting source audio input...
        </div>
        """

    # Downsample waveform to ~140 points for smooth SVG path rendering
    num_points = 140
    step = max(1, len(waveform) // num_points)
    sampled = waveform[::step][:num_points]

    max_val = np.max(np.abs(sampled)) if len(sampled) > 0 else 1.0
    if max_val < 1e-6:
        max_val = 1.0

    path_points = []
    dx = width / max(1, len(sampled) - 1)
    for i, val in enumerate(sampled):
        x = i * dx
        y = mid_y - (val / max_val) * (height * 0.42)
        path_points.append(f"{x:.1f},{y:.1f}")

    points_str = " ".join(path_points)
    dur_label = f"{duration_s:.2f}s" if duration_s else f"{len(waveform)/sample_rate:.2f}s"

    # 16-bar animated audio spectrum simulation derived from waveform energy
    spec_bars_html = []
    chunk_len = max(1, len(waveform) // 16)
    for b in range(16):
        chunk = waveform[b * chunk_len : (b + 1) * chunk_len]
        rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.05
        bar_height = min(22, max(4, int(rms * 130)))
        delay = (b * 0.09) % 1.2
        spec_bars_html.append(
            f'<div class="spec-bar" style="flex: 1; height: {bar_height}px; background: linear-gradient(to top, #0284c7, #00e5ff); border-radius: 1px 1px 0 0; animation-delay: {delay:.2f}s;"></div>'
        )
    bars_str = "".join(spec_bars_html)

    return f"""
<div style="background: #080d1a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; margin-bottom: 8px; min-height: 125px; position: relative; overflow: hidden;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-family: JetBrains Mono, monospace; font-size: 0.72rem; color: #94a3b8;">
        <span>ACOUSTIC WAVEFORM (LIVE TRACE)</span>
        <span>SR: <b style="color: #00e5ff;">{sample_rate} Hz</b> | DUR: <b style="color: #00e5ff;">{dur_label}</b></span>
    </div>
    <svg viewBox="0 0 {width} {height}" style="width: 100%; height: 80px; min-height: 80px; display: block;">
        <defs>
            <style>
                @keyframes laserSweepAnim {{
                    0% {{ transform: translateX(0px); opacity: 0.2; }}
                    50% {{ opacity: 0.9; }}
                    100% {{ transform: translateX({width}px); opacity: 0.2; }}
                }}
                .laser-scanline {{
                    animation: laserSweepAnim 2.4s linear infinite;
                }}
                @keyframes specBarBounce {{
                    0% {{ transform: scaleY(0.7); }}
                    50% {{ transform: scaleY(1.15); }}
                    100% {{ transform: scaleY(0.7); }}
                }}
                .spec-bar {{
                    transform-origin: bottom;
                    animation: specBarBounce 1.4s ease-in-out infinite;
                }}
            </style>
        </defs>
        <!-- Zero baseline -->
        <line x1="0" y1="{mid_y}" x2="{width}" y2="{mid_y}" stroke="#1e293b" stroke-dasharray="3,3"/>
        <!-- Waveform Path -->
        <polyline fill="none" stroke="#00e5ff" stroke-width="1.8" points="{points_str}" stroke-linejoin="round"/>
        <!-- Animated Oscilloscope Laser Scanline -->
        <line x1="0" y1="0" x2="0" y2="{height}" stroke="#00e5ff" stroke-width="1.5" class="laser-scanline" opacity="0.75"/>
    </svg>
    <div style="display: flex; align-items: flex-end; justify-content: space-between; height: 26px; gap: 4px; padding: 4px 4px 0 4px; border-top: 1px solid #1e293b; margin-top: 4px;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #64748b; margin-right: 4px; white-space: nowrap;">DSP FFT:</span>
        {bars_str}
    </div>
</div>
"""


def render_pipeline_flow_html(active_stages: dict[str, str]) -> str:
    """Render high-tech animated pipeline stage progression.

    Stages: AUDIO, AST AI, STEREO SIM, GCC-PHAT, ThreatEvent, SOP, LOCAL LLM, ADVISORY, HAPTIC.
    States: COMPLETE, ACTIVE, OFFLINE, WAITING.
    """
    stages_order = [
        ("AUDIO", "Audio Ingest"),
        ("AST AI", "AST Classifier"),
        ("STEREO SIM", "Stereo Simulator"),
        ("GCC-PHAT", "GCC-PHAT"),
        ("ThreatEvent", "ThreatEvent"),
        ("SOP", "SOP Engine"),
        ("LOCAL LLM", "Local LLM"),
        ("ADVISORY", "Advisory"),
        ("HAPTIC", "Haptic Alert"),
    ]

    badges_html = []
    for code, label in stages_order:
        state = active_stages.get(code, "WAITING")
        if state == "COMPLETE":
            bg = "rgba(16, 185, 129, 0.12)"
            border = "#10b981"
            dot = "<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:#10b981; box-shadow:0 0 6px #10b981; margin-right:5px;'></span>"
            text_color = "#10b981"
        elif state == "ACTIVE":
            bg = "rgba(0, 229, 255, 0.18)"
            border = "#00e5ff"
            dot = "<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:#00e5ff; box-shadow:0 0 8px #00e5ff; margin-right:5px;'></span>"
            text_color = "#00e5ff"
        elif state == "OFFLINE":
            bg = "rgba(239, 68, 68, 0.12)"
            border = "#ef4444"
            dot = "<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:#ef4444; box-shadow:0 0 6px #ef4444; margin-right:5px;'></span>"
            text_color = "#ef4444"
        else:
            bg = "#0f172a"
            border = "#1e293b"
            dot = "<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:#475569; margin-right:5px;'></span>"
            text_color = "#64748b"

        badges_html.append(f"""
        <div style="flex: 1; min-width: 90px; background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 6px 8px; text-align: center; margin: 3px;">
            <div style="font-size: 0.65rem; color: #94a3b8; font-family: JetBrains Mono, monospace; white-space: nowrap;">{dot} {code}</div>
            <div style="font-size: 0.75rem; font-weight: 700; color: {text_color}; font-family: JetBrains Mono, monospace;">{state}</div>
        </div>
        """)

    flow_items = "".join(badges_html)
    return f"""
<div style="background: #080d1a; border: 1px solid #1e293b; border-radius: 8px; padding: 8px 12px; margin-bottom: 16px;">
    <div style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em; color: #00e5ff; font-family: JetBrains Mono, monospace; margin-bottom: 6px;">
        LIVE PIPELINE ARCHITECTURE // EXECUTION TRACKER
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px; justify-content: space-between;">
        {flow_items}
    </div>
</div>
"""


def render_haptic_motors_html(direction: str | None = None) -> str:
    """Render high-visibility tactical 3-motor transducer visualizer."""
    dir_norm = direction.strip().upper() if direction else ""

    left_active = dir_norm == "LEFT"
    center_active = dir_norm == "FRONT"
    right_active = dir_norm == "RIGHT"

    def motor_box(label: str, active: bool) -> str:
        if active:
            bg = "rgba(0, 229, 255, 0.16)"
            border = "#00e5ff"
            shadow = "0 0 20px rgba(0, 229, 255, 0.6)"
            status = "ACTIVE [PULSING]"
            color = "#00e5ff"
            cls = "haptic-active-pulse"
        else:
            bg = "#0b1120"
            border = "#1e293b"
            shadow = "none"
            status = "IDLE"
            color = "#475569"
            cls = ""

        return f"""
        <div class="{cls}" style="flex: 1; background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 14px 10px; text-align: center; box-shadow: {shadow}; transition: all 0.3s ease;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; font-family: JetBrains Mono, monospace; margin-bottom: 6px;">{label}</div>
            <div style="font-size: 0.85rem; font-weight: 800; color: {color}; font-family: JetBrains Mono, monospace;">{status}</div>
        </div>
        """

    left_box = motor_box("LEFT MOTOR (TRANSDUCER L)", left_active)
    center_box = motor_box("CENTER / DUAL MOTOR", center_active)
    right_box = motor_box("RIGHT MOTOR (TRANSDUCER R)", right_active)

    return f"""
<style>
@keyframes hapticGlowAnim {{
    0% {{ transform: scale(1); box-shadow: 0 0 10px rgba(0, 229, 255, 0.3); }}
    50% {{ transform: scale(1.02); box-shadow: 0 0 25px rgba(0, 229, 255, 0.85), inset 0 0 12px rgba(0, 229, 255, 0.3); }}
    100% {{ transform: scale(1); box-shadow: 0 0 10px rgba(0, 229, 255, 0.3); }}
}}
.haptic-active-pulse {{
    animation: hapticGlowAnim 1.3s ease-in-out infinite;
}}
</style>
<div style="background: #080d1a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; margin: 8px 0;">
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
        <span style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.15em; color: #00e5ff; font-family: JetBrains Mono, monospace;">
            HAPTIC OUTPUT — SIMULATED TRANSDUCER ARRAY
        </span>
        <span style="font-size: 0.68rem; color: #f59e0b; font-family: JetBrains Mono, monospace; display: flex; align-items: center; gap: 5px;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#f59e0b; box-shadow:0 0 6px #f59e0b;"></span>
            SIMULATED TRANSDUCER ARRAY
        </span>
    </div>
    <div style="display: flex; gap: 10px;">
        {left_box}
        {center_box}
        {right_box}
    </div>
</div>
"""


def render_mic_simulation_html(
    injected_delay_s: float | None = None,
    estimated_delay_s: float | None = None,
    angle_deg: float | None = None,
) -> str:
    """Render microphone array geometry and arrival time delta telemetry."""
    if injected_delay_s is None or angle_deg is None:
        return ""

    left_lead = injected_delay_s < 0
    right_lead = injected_delay_s > 0
    neutral = abs(injected_delay_s) < 1e-6

    if left_lead:
        first_mic = "LEFT MICROPHONE (LEADS)"
        diff_us = abs(injected_delay_s) * 1e6
        delay_desc = f"Left leads by {diff_us:.1f} µs"
    elif right_lead:
        first_mic = "RIGHT MICROPHONE (LEADS)"
        diff_us = abs(injected_delay_s) * 1e6
        delay_desc = f"Right leads by {diff_us:.1f} µs"
    else:
        first_mic = "COINCIDENT ARRIVAL"
        delay_desc = "Zero interaural delay (0.0 µs)"

    return f"""
<div style="background: #080d1a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; margin: 8px 0; font-family: JetBrains Mono, monospace; font-size: 0.75rem;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; color: #94a3b8;">
        <span>HEADSET MIC GEOMETRY (SPACING: <b>0.18 m</b> | C: <b>343 m/s</b>)</span>
        <span style="color: #f59e0b;">INJECTED: <b>{angle_deg:+.1f}°</b></span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 8px 16px;">
        <span style="color: {'#00e5ff' if left_lead else '#64748b'}; font-weight: 700;">● MIC L (0.00m)</span>
        <span style="color: #475569;">━━━━━ [ {delay_desc} ] ━━━━━</span>
        <span style="color: {'#00e5ff' if right_lead else '#64748b'}; font-weight: 700;">MIC R (0.18m) ●</span>
    </div>
</div>
"""
