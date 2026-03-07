import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from emem.config import SpatioTemporalMemoryConfig
from emem.store import MemoryStore
from emem.types import ObservationNode


class WorkingMemory:
    """Tier-1 working memory: in-process buffer that batches writes to
    :class:`~emem.store.MemoryStore`.

    Maintains current position, active episode, and a deque of recent
    observations.  Flushes to the store on batch size or time interval.
    """

    def __init__(self, store, config=None):
        # type: (MemoryStore, Optional[SpatioTemporalMemoryConfig]) -> None
        self.store = store
        self.config = config or SpatioTemporalMemoryConfig()

        self._buffer = deque()  # type: Deque[ObservationNode]
        self._recent = deque(maxlen=self.config.working_memory_size)  # type: Deque[ObservationNode]
        self._lock = threading.Lock()

        self.current_position = None  # type: Optional[np.ndarray]
        self.active_episode_id = None  # type: Optional[str]
        self._last_flush_time = time.time()  # type: float

    def add(self, obs):
        # type: (ObservationNode) -> None
        """Add an observation to working memory.

        Auto-flushes when batch-size or time-interval thresholds are met.

        :param obs: Observation to buffer.
        """
        with self._lock:
            if self.active_episode_id and obs.episode_id is None:
                obs.episode_id = self.active_episode_id
            obs.tier = "working"
            self._buffer.append(obs)
            self._recent.append(obs)
            self.current_position = obs.coordinates.copy()

        if self._should_flush():
            self.flush()

    def _should_flush(self):
        # type: () -> bool
        if len(self._buffer) >= self.config.flush_batch_size:
            return True
        if time.time() - self._last_flush_time >= self.config.flush_interval:
            return True
        return False

    def flush(self):
        # type: () -> int
        """Flush buffered observations to the store.

        :returns: Number of observations flushed.
        :rtype: int
        """
        with self._lock:
            if not self._buffer:
                return 0
            to_flush = list(self._buffer)
            self._buffer.clear()
            self._last_flush_time = time.time()

        # Update tier to short_term on flush
        for obs in to_flush:
            obs.tier = "short_term"
        self.store.add_observations_batch(to_flush)
        return len(to_flush)

    def get_recent(self, n=None):
        # type: (Optional[int]) -> List[ObservationNode]
        """Get recent observations from the in-memory buffer.

        :param n: Return only the last *n* observations.  ``None`` returns all.
        :returns: List of recent observations.
        :rtype: List[ObservationNode]
        """
        with self._lock:
            items = list(self._recent)
        if n is not None:
            items = items[-n:]
        return items

    def start_episode(self, name, metadata=None, parent_episode_id=None):
        # type: (str, Optional[Dict[str, Any]], Optional[str]) -> str
        ts = time.time()
        episode_id = self.store.start_episode(name, ts, metadata, parent_episode_id)
        self.active_episode_id = episode_id
        return episode_id

    def end_episode(self, gist="", gist_embedding=None):
        # type: (str, Optional[np.ndarray]) -> Optional[str]
        if not self.active_episode_id:
            return None
        self.flush()
        ep_id = self.active_episode_id
        self.store.end_episode(ep_id, time.time(), gist, gist_embedding)
        self.active_episode_id = None
        return ep_id

    @property
    def buffer_size(self):
        # type: () -> int
        return len(self._buffer)

    @property
    def recent_count(self):
        # type: () -> int
        return len(self._recent)
