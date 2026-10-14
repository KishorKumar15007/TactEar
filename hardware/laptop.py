"""Real laptop input adapters. Optional audio packages are imported only at use time."""
from __future__ import annotations
from pathlib import Path
import queue, time, wave
import numpy as np
from tactear.types import AudioFrame, StereoAudioFrame

class AudioUnavailable(RuntimeError): pass

def list_audio_devices() -> list[dict]:
    try:
        import sounddevice as sd
    except ImportError as e: raise AudioUnavailable("sounddevice is not installed; install optional laptop audio dependencies") from e
    hostapis=sd.query_hostapis(); result=[]
    for index, dev in enumerate(sd.query_devices()):
        result.append({"index":index,"name":dev["name"],"input_channels":dev["max_input_channels"],"output_channels":dev["max_output_channels"],"sample_rate":dev["default_samplerate"],"host_api":hostapis[dev["hostapi"]]["name"]})
    return result

class SoundDeviceAmbientAudioSource:
    """Bounded, continuous real microphone capture; drops oldest queued frames under load."""
    def __init__(self, sample_rate=16000, frame_size=1024, device=None, channels=1, queue_size=8):
        self.sample_rate,self.frame_size,self.device,self.channels=sample_rate,frame_size,device,channels; self._queue=queue.Queue(maxsize=queue_size); self._stream=None; self.sequence=0; self.dropped=0
    def start(self):
        try: import sounddevice as sd
        except ImportError as e: raise AudioUnavailable("Install sounddevice and PortAudio to capture from a microphone") from e
        self.callback_errors = 0
        self.last_status = None
        info = sd.query_devices(self.device, "input")
        self.device_name = info["name"]
        def callback(indata, frames, time_info, status):
            if status:
                self.callback_errors += 1
                self.last_status = str(status)
                self.sequence += 1  # Break contiguous inference windows after device loss.
            samples=np.asarray(indata,dtype=np.float32).copy(); self.sequence+=1
            frame=(samples,time.time(),self.sequence)
            try:self._queue.put_nowait(frame)
            except queue.Full:
                try:self._queue.get_nowait(); self._queue.put_nowait(frame); self.dropped+=1
                except queue.Empty: pass
        self._stream=sd.InputStream(device=self.device,channels=self.channels,samplerate=self.sample_rate,blocksize=self.frame_size,dtype="float32",callback=callback); self._stream.start()
    def stop(self):
        if self._stream:
            try: self._stream.stop()
            finally: self._stream.close(); self._stream=None
    def read_frame(self, timeout=1.0):
        samples,timestamp,sequence=self._queue.get(timeout=timeout)
        if samples.ndim==1 or samples.shape[1]==1:
            mono=samples.reshape(-1); return StereoAudioFrame(mono,mono.copy(),self.sample_rate,timestamp,sequence,"microphone",str(self.device if self.device is not None else "default"))
        return StereoAudioFrame(samples[:,0],samples[:,1],self.sample_rate,timestamp,sequence,"microphone",str(self.device if self.device is not None else "default"))

class WavAmbientAudioSource:
    """Streams a real local WAV through the same StereoAudioFrame contract as microphone input."""
    def __init__(self, path, frame_size=1024): self.path=Path(path); self.frame_size=frame_size; self.wav=None; self.sequence=0
    def start(self):
        self.wav=wave.open(str(self.path),"rb"); self.sample_rate=self.wav.getframerate(); self.channels=self.wav.getnchannels(); self.width=self.wav.getsampwidth()
        if self.width not in (1,2,4):
            self.stop()
            raise AudioUnavailable("Only 8/16/32-bit PCM WAV is supported by the built-in file adapter")
    def stop(self):
        if self.wav:self.wav.close();self.wav=None
    def read_frame(self, timeout=1.0):
        if not self.wav: raise AudioUnavailable("audio file source not started")
        raw=self.wav.readframes(self.frame_size)
        if not raw: raise EOFError
        dtype={1:np.uint8,2:'<i2',4:'<i4'}[self.width]; data=np.frombuffer(raw,dtype=dtype).astype(np.float32)
        if self.width==1:data=(data-128)/128
        else:data/=float(2**(8*self.width-1))
        data=data.reshape(-1,self.channels); self.sequence+=1; left=data[:,0];right=data[:,1] if self.channels>1 else left.copy()
        return StereoAudioFrame(left,right,self.sample_rate,time.time(),self.sequence,"audio_file",str(self.path))
