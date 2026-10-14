"""Laptop-first TactEar command line."""
from __future__ import annotations
import argparse, logging
from tactear.hardware.laptop import list_audio_devices, WavAmbientAudioSource, SoundDeviceAmbientAudioSource, AudioUnavailable
from tactear.orchestrator.config import load_config
from tactear.models.panns import PannsCNN14Encoder, ModelUnavailable
def main():
    parser=argparse.ArgumentParser(prog="tactear",description="Offline TactEar laptop runtime")
    parser.add_argument("command",nargs="?",default="run",choices=["run","audio-devices","doctor","serve"])
    parser.add_argument("--config",default="configs/laptop.yaml");parser.add_argument("--audio-file");parser.add_argument("--device",help="sounddevice input index or name")
    args=parser.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
    if args.command=="audio-devices":
        try:
            for d in list_audio_devices(): print(f"{d['index']:>2}  {d['name']} | input={d['input_channels']} output={d['output_channels']} rate={d['sample_rate']} host={d['host_api']}")
        except AudioUnavailable as e: print(f"AUDIO UNAVAILABLE: {e}")
        return
    cfg=load_config(args.config)
    if args.audio_file:
        source=WavAmbientAudioSource(args.audio_file,cfg['audio']['frame_size']); source.start(); print(f"AUDIO FILE STREAM ACTIVE: {args.audio_file}"); frames=0
        try:
            while True:
                frame=source.read_frame(); frames+=1; rms=float((frame.left**2).mean()**.5); print(f"frame={frame.sequence} rms={rms:.5f} rate={frame.sample_rate}")
        except EOFError: print(f"EOF: processed {frames} real audio frames")
        finally: source.stop()
        return
    if args.command=="doctor":
        try: print(f"AUDIO: AVAILABLE ({len(list_audio_devices())} devices)")
        except AudioUnavailable as e: print(f"AUDIO: UNAVAILABLE ({e})")
        try:
            encoder=cfg['ml']['encoder']; PannsCNN14Encoder(encoder['checkpoint'],encoder.get('device','cpu')); print("MODEL: READY")
        except ModelUnavailable as e: print(f"MODEL: UNAVAILABLE ({e})")
        print("IMU: MANUAL HEADING | HAPTIC: SOFTWARE LOG | RADIO: NOT CONNECTED"); return
    if args.command=="serve":
        try:
            import uvicorn
            from tactear.api import create_api
            uvicorn.run(create_api(),host="127.0.0.1",port=8765)
        except ImportError: print("API UNAVAILABLE: install fastapi and uvicorn from requirements-laptop.txt")
        return
    audio=cfg['audio']; device=int(args.device) if args.device and args.device.isdigit() else args.device
    source=SoundDeviceAmbientAudioSource(audio['sample_rate'],audio['frame_size'],device if device is not None else audio.get('ambient',{}).get('device'),audio.get('ambient',{}).get('channels',1))
    try:
        source.start(); print(f"MICROPHONE STREAM ACTIVE: device={device or 'default'} rate={audio['sample_rate']} Hz. Ctrl+C stops capture.")
        while True:
            frame=source.read_frame(); rms=float((frame.left**2).mean()**.5); peak=float(abs(frame.left).max()); print(f"frame={frame.sequence} rms={rms:.5f} peak={peak:.5f} dropped={source.dropped}")
    except KeyboardInterrupt: print("Microphone stream stopped.")
    except (AudioUnavailable, OSError) as e: print(f"AUDIO ERROR: {e}")
    finally: source.stop()
if __name__=="__main__":main()
