"""Local microphone control and bounded real-time observation stream."""
import asyncio
import queue
import time
import logging
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tactear.hardware.laptop import SoundDeviceAmbientAudioSource, list_audio_devices
from tactear.llm.rag_index import LocalRAG
from tactear.models.panns import PannsCNN14Encoder
from tactear.models.spatial import spatial_estimate
from tactear.models.detection import AudioWindow
from tactear.models.events import EventFilter
from tactear.models.audio_input import measurements, decode_audio, clip_visuals, MAX_BYTES

log = logging.getLogger(__name__)

class StartRequest(BaseModel):
    device: int | None = None
    sample_rate: int = Field(16000, ge=8000, le=192000)
    channels: int = Field(1, ge=1, le=2)
    spacing_m: float | None = Field(None, ge=.01, le=2)
    threshold: float = Field(.35, ge=.05, le=.99)
    window_seconds: int = Field(10, ge=3, le=10)

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)

def analyze_frame(frame):
    x = np.asarray(frame.left, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Invalid audio frame")
    data = measurements(np.column_stack((frame.left,frame.right)),frame.sample_rate)
    data["waveform"] = data["waveform"][::2]
    return {**data,"sequence":frame.sequence,"captured_at":frame.timestamp,
            "latency_ms":max(0,(time.time()-frame.timestamp)*1000)}

def create_api(app_core=None, source_factory=SoundDeviceAmbientAudioSource, model_factory=PannsCNN14Encoder):
    state = {"state": "READY", "mode": "laptop", "model": "UNAVAILABLE",
             "reason": "Load the local PANNs model to enable inference",
             "audio": None, "error": None, "frames": 0, "drops": 0,
             "classification": None, "spatial": None, "history": [],
             "error_count":0,"input":None,"inference":"IDLE","inferences":0,
             "buffer_seconds":0,"threshold_note":"Uncalibrated research threshold; raw scores remain visible",
             "callback_errors":0,"callback_status":None,"inference_drops":0}
    source = None
    worker = None
    inference_worker = None
    model = None
    model_lock = asyncio.Lock()
    prediction_lock = asyncio.Lock()
    upload_lock = asyncio.Lock()
    window = AudioWindow()
    confirm = EventFilter()
    inference_queue = asyncio.Queue(maxsize=1)
    stream_settings = StartRequest()
    clients = set()
    control = asyncio.Lock()
    def remember(events):
        ids = {e["event_id"] for e in events}
        state["history"] = [*reversed(events), *(e for e in state["history"] if e["event_id"] not in ids)][:100]

    async def predict(samples, rate):
        # One Torch call at a time, including file requests and cancellation.
        async with prediction_lock:
            running = asyncio.create_task(asyncio.to_thread(model.predict,samples,rate))
            try: return await asyncio.shield(running)
            except asyncio.CancelledError:
                await running
                raise
    async def publish():
        for inbox in tuple(clients):
            if inbox.full(): inbox.get_nowait()
            inbox.put_nowait({"version": 1, "type": "snapshot", "payload": dict(state)})
    async def consume():
        last_inference = 0.
        last_publish = 0.
        try:
            while True:
                try: frame = await asyncio.to_thread(source.read_frame, 0.5)
                except queue.Empty:
                    if state["audio"] and time.time()-state["audio"]["captured_at"]>2:
                        state.update(audio=None,spatial=None,classification=None,inference="NO AUDIO")
                        window.reset()
                        await publish()
                    continue
                state["audio"] = analyze_frame(frame)
                state["spatial"] = spatial_estimate(frame.left, frame.right, frame.sample_rate,
                    stream_settings.channels, stream_settings.spacing_m)
                state["frames"] += 1
                state["drops"] = source.dropped
                state["audio"]["channels"] = stream_settings.channels
                if stream_settings.channels == 1: state["audio"]["mix"] = "mono"
                state["audio"]["channel_rms"] = state["audio"]["channel_rms"][:stream_settings.channels]
                state["callback_errors"] = getattr(source,"callback_errors",0)
                state["callback_status"] = getattr(source,"last_status",None)
                window.add(frame)
                state["buffer_seconds"] = min(window.seconds,window.count/frame.sample_rate)
                if not window.ready():
                    state.update(classification=None,inference="BUFFERING")
                now = time.monotonic()
                if model is not None and window.ready() and now-last_inference>=1.:
                    if inference_queue.full():
                        inference_queue.get_nowait()
                        state["inference_drops"] += 1
                    inference_queue.put_nowait((window.samples(),frame.sample_rate,frame.timestamp))
                    last_inference=now
                if now-last_publish>=.1:
                    await publish()
                    last_publish=now
        except asyncio.CancelledError: raise
        except Exception as exc:
            state.update(state="ERROR", error=str(exc), audio=None,spatial=None,classification=None,inference="ERROR")
            state["error_count"] += 1
            log.exception("Capture failed")
            if source:
                await asyncio.to_thread(source.stop)
            await publish()
    async def infer():
        while True:
            samples, rate, captured_at = await inference_queue.get()
            try:
                state["inference"] = "RUNNING"
                if np.sqrt(np.mean(samples*samples))<1e-5:
                    confirm.counts.clear()
                    state["classification"]={"status":"SILENT","label":None,"top":[],"watched":[],
                        "timestamp":captured_at,"source":"microphone","reason":"Measured RMS below −100 dBFS; model skipped"}
                else:
                    prediction = await predict(samples,rate)
                    prediction["captured_at"] = captured_at
                    result, events = confirm.apply(prediction,"microphone")
                    state["classification"]=result
                    remember(events)
                    state["inferences"] += 1
                state["inference"] = "WAITING"
                await publish()
            except asyncio.CancelledError: raise
            except Exception as exc:
                state["classification"]={"status":"ERROR","reason":str(exc),"top":[],"watched":[]}
                state.update(inference="ERROR",error=str(exc))
                state["error_count"] += 1
                log.exception("Inference failed")
                await publish()
    async def stop():
        nonlocal source, worker, inference_worker
        if worker:
            worker.cancel()
            try: await worker
            except asyncio.CancelledError: pass
            worker = None
        if inference_worker:
            # A running torch call cannot be safely cancelled; join it before reusing the model.
            inference_worker.cancel()
            try: await inference_worker
            except asyncio.CancelledError: pass
            inference_worker=None
        if source:
            await asyncio.to_thread(source.stop)
            source = None
        window.reset()
        while not inference_queue.empty(): inference_queue.get_nowait()
        state.update(state="READY", audio=None, classification=None, spatial=None,inference="IDLE",buffer_seconds=0)
        await publish()
    @asynccontextmanager
    async def lifespan(api):
        yield
        await stop()
    api = FastAPI(title="TactEar local API", version="1", lifespan=lifespan)
    api.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    @api.get("/api/status")
    async def status(): return state
    @api.get("/api/devices")
    async def devices():
        try: return await asyncio.to_thread(list_audio_devices)
        except Exception as exc: raise HTTPException(503, str(exc)) from exc
    def samples_manifest():
        root = Path(__file__).resolve().parents[2] / "data" / "sample123"
        path = root / "manifest.json"
        return root, json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    @api.get("/api/samples")
    async def samples_list():
        _, rows = samples_manifest()
        return rows
    @api.get("/api/samples/{filename}")
    async def sample_file(filename: str):
        root, rows = samples_manifest()
        if filename not in {row["file"] for row in rows}:
            raise HTTPException(404,"Sample not found")
        path=(root/filename).resolve()
        if path.parent != root.resolve() or not path.is_file(): raise HTTPException(404,"Sample not found")
        return FileResponse(path,media_type="audio/wav")
    @api.post("/api/model/load")
    async def load_model():
        nonlocal model
        async with model_lock:
            if model is not None: return {"model":"READY","reason":state["reason"]}
            state.update(model="LOADING",reason="Loading and warming up local CNN14")
            await publish()
            try: model=await asyncio.to_thread(model_factory)
            except Exception as exc:
                state.update(model="UNAVAILABLE",reason=str(exc))
                state["error_count"] += 1
                log.exception("Model load failed")
                await publish()
                raise HTTPException(503,str(exc)) from exc
            state.update(model="READY",reason=model.name)
            await publish()
            return {"model":"READY","reason":model.name}
    @api.post("/api/audio/start")
    async def start(request: StartRequest):
        nonlocal source, worker, inference_worker, stream_settings, confirm, window
        async with control:
            await stop()
            stream_settings=request
            confirm=EventFilter(threshold=request.threshold)
            window=AudioWindow(seconds=request.window_seconds)
            try:
                source = source_factory(request.sample_rate, 1024, request.device, request.channels)
                await asyncio.to_thread(source.start)
            except Exception as exc:
                if source: await asyncio.to_thread(source.stop)
                source = None
                state.update(state="ERROR", error=str(exc))
                state["error_count"] += 1
                await publish()
                raise HTTPException(400, str(exc)) from exc
            state.update(state="STREAMING", error=None, frames=0, drops=0,inferences=0,inference_drops=0,
                         inference="BUFFERING",input={"source":"microphone","device":request.device,
                         "name":getattr(source,"device_name",str(request.device or "System default")),
                         "sample_rate":request.sample_rate,"channels":request.channels,"window_seconds":request.window_seconds})
            worker = asyncio.create_task(consume())
            inference_worker=asyncio.create_task(infer())
            await publish()
            return state
    @api.post("/api/audio/stop")
    async def stop_audio():
        async with control: await stop()
        return state
    @api.post("/api/rag/query")
    async def query(request: QueryRequest):
        root = Path(__file__).resolve().parents[2] / "data" / "sop"
        def retrieve():
            return [asdict(c) for c in LocalRAG.from_directory(root).query(request.query)]
        return await asyncio.to_thread(retrieve)
    @api.get("/api/events")
    async def events():
        return state["history"]
    @api.post("/api/events/clear")
    async def clear_events():
        state["history"] = []
        confirm.last.clear()
        await publish()
        return {"cleared":True}

    @api.post("/api/audio/file")
    async def audio_file(request: Request, name: str = "Local clip", classify: bool = True, threshold: float = .35):
        if not .05 <= threshold <= .99: raise HTTPException(422,"Threshold must be .05–.99")
        if upload_lock.locked(): raise HTTPException(429,"Another clip is being processed")
        async with upload_lock:
            data = bytearray()
            async for chunk in request.stream():
                if len(data)+len(chunk)>MAX_BYTES: raise HTTPException(413,"Maximum upload is 24 MiB")
                data.extend(chunk)
            try: samples, mono, rate, mix = await asyncio.to_thread(decode_audio,data)
            except ValueError as exc: raise HTTPException(400,str(exc)) from exc
            except ImportError as exc: raise HTTPException(503,"Install requirements-ml.txt for file decoding") from exc
            audio = await asyncio.to_thread(measurements,samples,rate)
            visuals = await asyncio.to_thread(clip_visuals,mono,rate)
            result = {"name":name[:160],"audio":{**audio,"sequence":0,"latency_ms":0},
                      "spectrogram":visuals,"mix":mix,"segments":[],"classification":None}
            if classify:
                if model is None: raise HTTPException(503,"Load the CNN14 model before running classification")
                # Cover the complete clip in <=10-second windows; never silently truncate it.
                size = rate*10
                starts = list(range(0,max(1,len(mono)-size+1),rate*5))
                if len(mono)>size and starts[-1]!=len(mono)-size: starts.append(len(mono)-size)
                try:
                    for start in starts:
                        prediction = await predict(mono[start:start+size],rate)
                        filtered, emitted = EventFilter(threshold).apply(prediction,"file: "+name[:160],single=True)
                        result["segments"].append({"offset_seconds":start/rate,**filtered})
                        remember(emitted)
                    result["classification"] = max(result["segments"],key=lambda p:p["top"][0]["score"])
                except Exception as exc:
                    log.exception("File inference failed")
                    state["error_count"] += 1
                    raise HTTPException(503,str(exc)) from exc
                await publish()
            return result
    @api.websocket("/ws")
    async def websocket(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin and origin not in ("http://localhost:5173", "http://127.0.0.1:5173"):
            await ws.close(code=1008)
            return
        await ws.accept()
        inbox = asyncio.Queue(maxsize=2)
        clients.add(inbox)
        try:
            await ws.send_json({"version": 1, "type": "snapshot", "payload": state})
            while True:
                try: packet = await asyncio.wait_for(inbox.get(), timeout=2)
                except asyncio.TimeoutError: packet = {"version": 1, "type": "snapshot", "payload": state}
                await ws.send_json(packet)
        except (WebSocketDisconnect, RuntimeError): pass
        finally: clients.discard(inbox)
    return api
