"""Lazy Raspberry Pi adapters. Optional packages are never imported in simulation."""
from __future__ import annotations
from tactear.hardware.simulated import SimulatedAmbientAudioSource, SimulatedBoomAudioSource
class RaspberryPiUnavailable(RuntimeError): pass
def require_pi():
    try:
        import RPi.GPIO # noqa
    except ImportError as e: raise RaspberryPiUnavailable("Raspberry Pi dependencies missing; install RPi.GPIO, smbus2 and sounddevice on the Pi") from e
class RaspberryPiAmbientAudioSource(SimulatedAmbientAudioSource):
    def __init__(self, *a, **kw): require_pi(); super().__init__(*a, **kw)
class RaspberryPiBoomAudioSource(SimulatedBoomAudioSource):
    def __init__(self, *a, **kw): require_pi(); super().__init__(*a, **kw)
