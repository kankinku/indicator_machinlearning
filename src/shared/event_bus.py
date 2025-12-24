from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import threading

@dataclass
class Event:
    event_type: str
    batch_id: Optional[int] = None
    policy_id: Optional[str] = None
    stage: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class EventRecorder:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventRecorder, cls).__new__(cls)
                cls._instance._events = []
                cls._instance._active = False
        return cls._instance

    def activate(self):
        self._active = True
        self.clear()

    def deactivate(self):
        self._active = False

    def is_active(self) -> bool:
        return self._active

    def record(self, event_type: str, batch_id: Optional[int] = None, policy_id: Optional[str] = None, stage: Optional[str] = None, payload: Optional[Dict[str, Any]] = None):
        if not self._active:
            return
        
        event = Event(
            event_type=event_type,
            batch_id=batch_id,
            policy_id=policy_id,
            stage=stage,
            payload=payload or {}
        )
        with self._lock:
            self._events.append(event)

    def get_events(self, event_type: Optional[str] = None) -> List[Event]:
        with self._lock:
            if event_type:
                return [e for e in self._events if e.event_type == event_type]
            return list(self._events)

    def clear(self):
        with self._lock:
            self._events = []

def record_event(event_type: str, **kwargs):
    EventRecorder().record(event_type, **kwargs)
