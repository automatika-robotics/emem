import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from emem.config import SpatioTemporalMemoryConfig
from emem.consolidation import ConsolidationEngine, LLMClient
from emem.embeddings import EmbeddingProvider
from emem.store import MemoryStore
from emem.tools import MemoryTools
from emem.types import ObservationNode
from emem.working_memory import WorkingMemory


class SpatioTemporalMemory:
    """High-level facade for the eMEM spatio-temporal memory system.

    Manages observation ingestion, episode lifecycle, consolidation, and
    queries through a single object.  Internal buffering, flushing, and
    consolidation happen automatically.

    Example::

        mem = SpatioTemporalMemory(db_path="/tmp/mem.db")

        mem.start_episode("kitchen_patrol")
        mem.add("Red chair near table", x=10.0, y=10.0)
        mem.add("Cat on chair", x=10.2, y=10.1)
        mem.end_episode()

        print(mem.spatial_query(x=10.0, y=10.0, radius=3.0))
        print(mem.episode_summary(last_n=1))
    """

    def __init__(
        self,
        db_path="memory.db",          # type: str
        config=None,                   # type: Optional[SpatioTemporalMemoryConfig]
        embedding_provider=None,       # type: Optional[EmbeddingProvider]
        llm_client=None,               # type: Optional[LLMClient]
        get_current_time=None,         # type: Optional[Callable[[], float]]
    ):
        if config is None:
            hnsw_path = str(Path(db_path).with_suffix(".hnsw.bin"))
            config = SpatioTemporalMemoryConfig(db_path=db_path, hnsw_path=hnsw_path)

        self._config = config
        self._get_time = get_current_time or time.time

        self._store = MemoryStore(config=config, embedding_provider=embedding_provider)
        self._wm = WorkingMemory(store=self._store, config=config)
        self._consolidation = ConsolidationEngine(
            store=self._store, config=config, llm_client=llm_client,
        )
        self._tools = MemoryTools(
            store=self._store,
            get_current_time=self._get_time,
            get_current_position=lambda: self._wm.current_position,
        )

    # ── Ingestion ─────────────────────────────────────────────────

    def add(
        self,
        text,             # type: str
        x,                # type: float
        y,                # type: float
        z=0.0,            # type: float
        timestamp=None,   # type: Optional[float]
        layer_name="default",   # type: str
        source_type="manual",   # type: str
        confidence=1.0,   # type: float
        metadata=None,    # type: Optional[Dict[str, Any]]
        embedding=None,   # type: Optional[np.ndarray]
    ):
        # type: (...) -> str
        """Add an observation.

        Observations are buffered and auto-flushed to the store based on
        batch size and time interval config.  Episode assignment is automatic
        if an episode is active.

        :param text: Observation text.
        :param x: X coordinate.
        :param y: Y coordinate.
        :param z: Z coordinate.
        :param timestamp: Observation timestamp.  Defaults to current time.
        :param layer_name: Perception layer (e.g. ``"vlm"``, ``"detections"``).
        :param source_type: Source identifier.
        :param confidence: Confidence score in ``[0, 1]``.
        :param metadata: Arbitrary key-value metadata.
        :param embedding: Pre-computed embedding vector.
        :returns: Observation ID.
        :rtype: str
        """
        obs = ObservationNode(
            text=text,
            coordinates=np.array([x, y, z]),
            timestamp=timestamp or self._get_time(),
            layer_name=layer_name,
            source_type=source_type,
            confidence=confidence,
            metadata=metadata or {},
            embedding=embedding,
        )
        self._wm.add(obs)
        return obs.id

    # ── Episodes ──────────────────────────────────────────────────

    def start_episode(self, name, metadata=None, parent_episode_id=None):
        # type: (str, Optional[Dict[str, Any]], Optional[str]) -> str
        """Start a new episode.

        All subsequent observations are automatically assigned to this episode
        until :meth:`end_episode` is called.

        :param name: Episode name.
        :param metadata: Arbitrary key-value metadata.
        :param parent_episode_id: Parent episode for hierarchical nesting.
        :returns: Episode ID.
        :rtype: str
        """
        ts = self._get_time()
        episode_id = self._store.start_episode(name, ts, metadata, parent_episode_id)
        self._wm.active_episode_id = episode_id
        return episode_id

    def end_episode(self, consolidate=True):
        # type: (bool) -> Optional[str]
        """End the active episode.

        Flushes buffered observations, generates a consolidated gist from
        all episode observations (unless *consolidate* is ``False``), and
        archives the raw observations.

        :param consolidate: Whether to generate a gist and archive observations.
        :returns: Episode ID, or ``None`` if no episode was active.
        :rtype: Optional[str]
        """
        if not self._wm.active_episode_id:
            return None

        self._wm.flush()
        ep_id = self._wm.active_episode_id
        self._wm.active_episode_id = None

        if consolidate:
            gist_id = self._consolidation.consolidate_episode(ep_id)
            gist_text = ""
            if gist_id:
                gist = self._store.get_gist(gist_id)
                if gist:
                    gist_text = gist.text
            self._store.end_episode(ep_id, self._get_time(), gist=gist_text)
        else:
            self._store.end_episode(ep_id, self._get_time())

        return ep_id

    @property
    def active_episode_id(self):
        # type: () -> Optional[str]
        return self._wm.active_episode_id

    # ── Queries (LLM tool interface) ──────────────────────────────

    def _ensure_flushed(self):
        # type: () -> None
        if self._wm.buffer_size > 0:
            self._wm.flush()

    def semantic_search(self, query, **kwargs):
        # type: (str, **Any) -> str
        self._ensure_flushed()
        return self._tools.semantic_search(query=query, **kwargs)

    def spatial_query(self, x, y, **kwargs):
        # type: (float, float, **Any) -> str
        self._ensure_flushed()
        return self._tools.spatial_query(x=x, y=y, **kwargs)

    def temporal_query(self, **kwargs):
        # type: (**Any) -> str
        self._ensure_flushed()
        return self._tools.temporal_query(**kwargs)

    def episode_summary(self, **kwargs):
        # type: (**Any) -> str
        self._ensure_flushed()
        return self._tools.episode_summary(**kwargs)

    def get_current_context(self, **kwargs):
        # type: (**Any) -> str
        self._ensure_flushed()
        return self._tools.get_current_context(**kwargs)

    def search_gists(self, query, **kwargs):
        # type: (str, **Any) -> str
        self._ensure_flushed()
        return self._tools.search_gists(query=query, **kwargs)

    # ── Consolidation ─────────────────────────────────────────────

    def consolidate_time_window(self):
        # type: () -> List[str]
        """Manually trigger time-window consolidation.

        Normally you don't need to call this -- episode consolidation happens
        automatically in :meth:`end_episode`.  Use this for non-episodic
        observations that have aged past the consolidation window.

        :returns: Created gist IDs.
        :rtype: List[str]
        """
        self._ensure_flushed()
        return self._consolidation.consolidate_time_window(reference_time=self._get_time())

    # ── Tool dispatch (for LLM integration) ───────────────────────

    def get_tool_definitions(self):
        # type: () -> List[Dict[str, Any]]
        """Return tool definitions suitable for LLM function calling.

        :returns: List of tool definition dicts.
        :rtype: List[Dict[str, Any]]
        """
        return self._tools.get_tool_definitions()

    def dispatch_tool_call(self, tool_name, arguments):
        # type: (str, Dict[str, Any]) -> str
        """Dispatch a tool call by name.  Auto-flushes before executing.

        :param tool_name: One of the six tool names.
        :param arguments: Tool arguments dict.
        :returns: Formatted result string.
        :rtype: str
        """
        self._ensure_flushed()
        return self._tools.dispatch_tool_call(tool_name, arguments)

    # ── Programmatic access ───────────────────────────────────────

    @property
    def store(self):
        # type: () -> MemoryStore
        """Direct access to the underlying :class:`~emem.store.MemoryStore`."""
        return self._store

    @property
    def current_position(self):
        # type: () -> Optional[np.ndarray]
        return self._wm.current_position

    def get_recent(self, n=None):
        # type: (Optional[int]) -> List[ObservationNode]
        """Get recent observations from the in-memory buffer.

        :param n: Return only the last *n* observations.
        :returns: List of recent observations.
        :rtype: List[ObservationNode]
        """
        return self._wm.get_recent(n)

    # ── Lifecycle ─────────────────────────────────────────────────

    def save(self):
        # type: () -> None
        self._wm.flush()
        self._store.save()

    def close(self):
        # type: () -> None
        self._wm.flush()
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
