from __future__ import annotations
from collections import defaultdict
from threading import RLock
class EventBus:
    def __init__(self): self._handlers=defaultdict(list); self._lock=RLock()
    def subscribe(self,name,handler):
        with self._lock: self._handlers[name].append(handler)
    def publish(self,name,payload=None):
        with self._lock: handlers=list(self._handlers[name])
        for handler in handlers: handler(payload)
