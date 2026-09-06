from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional
import time


@dataclass
class FrameData:
    image: bytes
    timestamp: float


class FrameBuffer:
    """Thread-safe buffer storing latest processed frames per stream."""

    def __init__(self) -> None:
        self._frames: Dict[str, FrameData] = {}
        self._lock = Lock()

    def update(self, stream_id: str, image: bytes) -> None:
        """Store most recent frame for a stream."""
        with self._lock:
            self._frames[stream_id] = FrameData(image=image, timestamp=time.time())

    def get(self, stream_id: str) -> Optional[FrameData]:
        """Retrieve latest frame for a stream, if available."""
        with self._lock:
            return self._frames.get(stream_id)


frame_buffer = FrameBuffer()
