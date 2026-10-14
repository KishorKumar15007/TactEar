from __future__ import annotations
import time, uuid, logging
from tactear.models.localization import gcc_phat, egocentric_azimuth, direction_sector
from tactear.models.joint_model import SUPPRESSION_POLICY
from tactear.types import SoundEvent, Advisory
log=logging.getLogger(__name__)
class TactEarApplication:
    """Application core: depends only on contracts and injectable local services."""
    def __init__(self, hardware, model, rag, llm, priority, bus, config):
        self.hardware,self.model,self.rag,self.llm,self.priority,self.bus,self.config=hardware,model,rag,llm,priority,bus,config
        self.latest_event=None; hardware.button.start(self.request_advisory)
    def process_frame(self, frame, hint=None):
        label,confidence,_=self.model.infer((frame.left+frame.right)/2,hint)
        threshold=self.config["classification"].get("confidence_threshold",.7)
        if confidence < threshold: return None
        az,loc_conf=gcc_phat(frame.left,frame.right,frame.sample_rate,self.config["localization"]["microphone_distance_m"])
        yaw=self.hardware.orientation.read_orientation().yaw_deg
        az=egocentric_azimuth(az,yaw); event=SoundEvent(str(uuid.uuid4()),label,confidence,az,direction_sector(az),time.time(),SUPPRESSION_POLICY[label])
        self.latest_event=event; self.hardware.haptic.trigger(label,event.direction); self.bus.publish("SoundClassified",event); self.bus.publish("DirectionEstimated",event)
        log.info("Sound classified: %s confidence=%.2f direction=%s localization_confidence=%.2f",label,confidence,event.direction,loc_conf); return event
    def request_advisory(self):
        self.bus.publish("ButtonPressed")
        if not self.latest_event: return None
        query=f"{self.latest_event.sound_class} {self.latest_event.direction}"
        chunks=self.rag.query(query, self.config["rag"].get("top_k",3), self.config["rag"].get("minimum_score",.05))
        if not chunks: text=self.config["rag"]["fallback"]
        else: text=self.llm.generate("Event: "+query+"\nSources:\n"+"\n".join(c.text for c in chunks))
        advisory=Advisory(self.latest_event.event_id,text,chunks,time.time()); self.bus.publish("AdvisoryGenerated",advisory); self.priority.speak(text); return advisory
