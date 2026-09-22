"""Command-line review runner for the TactEar core audio pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Running ``python scripts/run_demo.py`` makes ``scripts/`` the import root.
# Add the repository root so the public TactEar package can be imported.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tactear.pipeline import run_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TactEar core audio review demo.",
        epilog="Place a valid review WAV inside demo_audio/ before running the example.",
    )
    parser.add_argument("audio_path", help="Path to a decodable review audio file")
    parser.add_argument(
        "--angle",
        required=True,
        type=float,
        help="Simulated source angle in degrees (-90 to 90; negative is LEFT)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the public pipeline once and print its runtime result."""

    args = _parse_args()
    try:
        event = run_pipeline(args.audio_path, simulated_angle_deg=args.angle)
    except Exception as exc:
        print(f"TactEar demo failed: {exc}", file=sys.stderr)
        return 1

    print("TACTEAR CORE AUDIO DEMO")
    print("-----------------------")
    print()
    print(f"Audio: {Path(event.source_audio).name}")
    print()
    print("CLASSIFICATION")
    print(f"Raw label: {event.raw_model_label}")
    print(f"Tactical class: {event.sound_class}")
    print(f"Confidence: {event.classification_confidence:.4f}")
    print()
    print("HEADSET INPUT")
    print("Hardware: SIMULATED" if event.hardware_simulated else "Hardware: PHYSICAL")
    print(f"Injected angle: {event.ground_truth_angle_deg:.2f}\N{DEGREE SIGN}")
    print()
    print("LOCALIZATION")
    print(f"Estimated angle: {event.estimated_angle_deg:.2f}\N{DEGREE SIGN}")
    print(f"Direction: {event.direction}")
    print(f"Angular error: {event.localization_error_deg:.2f}\N{DEGREE SIGN}")
    print()
    print("STATUS")
    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
