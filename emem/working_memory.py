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

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[SpatioTemporalMemoryConfig] = None,
    ) -> None:
        self.store = store
        self.config = config or SpatioTemporalMemoryConfig()

        self._buffer: Deque[ObservationNode] = deque()
        self._recent: Deque[ObservationNode] = deque(maxlen=self.config.working_memory_size)
        self._lock = threading.Lock()

        self.current_position: Optional[np.ndarray] = None
        self.active_episode_id: Optional[str] = None
        self._last_flush_time: float = time.time()

    def add(self, obs: ObservationNode) -> None:
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

    def _should_flush(self) -> bool:
        if len(self._buffer) >= self.config.flush_batch_size:
            return True
        if time.time() - self._last_flush_time >= self.config.flush_interval:
            return True
        return False

    def flush(self) -> int:
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

    def get_recent(self, n: Optional[int] = None) -> List[ObservationNode]:
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

    def start_episode(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_episode_id: Optional[str] = None,
    ) -> str:
        ts = time.time()
        episode_id = self.store.start_episode(name, ts, metadata, parent_episode_id)
        self.active_episode_id = episode_id
        return episode_id

    def end_episode(
        self,
        gist: str = "",
        gist_embedding: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        if not self.active_episode_id:
            return None
        self.flush()
        ep_id = self.active_episode_id
        self.store.end_episode(ep_id, time.time(), gist, gist_embedding)
        self.active_episode_id = None
        return ep_id

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def recent_count(self) -> int:
        return len(self._recent)
