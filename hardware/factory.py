from __future__ import annotations
from dataclasses import dataclass
from tactear.capture.interfaces import AmbientAudioSource, BoomAudioSource, OrientationSensor, HapticOutput, ButtonInput
from .simulated import SimulatedAmbientAudioSource, SimulatedBoomAudioSource, SimulatedOrientationSensor, SimulatedHapticOutput, SimulatedButtonInput
from .laptop import SoundDeviceAmbientAudioSource
@dataclass
class HardwareBundle:
    ambient_audio: AmbientAudioSource
    boom_audio: BoomAudioSource
    orientation: OrientationSensor
    haptic: HapticOutput
    button: ButtonInput
def create_hardware(config):
    mode=config["hardware"].get("mode", "simulation")
    a=config["audio"]
    if mode == "simulation": return HardwareBundle(SimulatedAmbientAudioSource(a["sample_rate"],a["frame_size"]), SimulatedBoomAudioSource(a["sample_rate"],a["frame_size"]), SimulatedOrientationSensor(), SimulatedHapticOutput(), SimulatedButtonInput())
    if mode == "laptop":
        ambient=SoundDeviceAmbientAudioSource(a["sample_rate"],a["frame_size"],a.get("ambient",{}).get("device"),a.get("ambient",{}).get("channels",1))
        return HardwareBundle(ambient, SimulatedBoomAudioSource(a["sample_rate"],a["frame_size"]), SimulatedOrientationSensor(), SimulatedHapticOutput(), SimulatedButtonInput())
    if mode == "raspberry_pi":
        raise RuntimeError("Physical Raspberry Pi adapters are not implemented. Use laptop mode for real audio; simulation mode is only for tests.")
    raise ValueError(f"Unknown hardware mode: {mode}")
