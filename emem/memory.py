import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from emem.config import SpatioTemporalMemoryConfig
from emem.consolidation import ConsolidationEngine, LLMClient
from emem.embeddings import EmbeddingProvider
from emem.store import MemoryStore
from emem.tools import MemoryTools
from emem.types import EntityNode, ObservationNode
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
        db_path: str = "memory.db",
        config: Optional[SpatioTemporalMemoryConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_client: Optional[LLMClient] = None,
        get_current_time: Optional[Callable[[], float]] = None,
    ) -> None:
        if config is None:
            hnsw_path = str(Path(db_path).with_suffix(".hnsw.bin"))
            config = SpatioTemporalMemoryConfig(db_path=db_path, hnsw_path=hnsw_path)

        self._config = config
        self._get_time = get_current_time or time.time

        self._store = MemoryStore(config=config, embedding_provider=embedding_provider)
        self._consolidation = ConsolidationEngine(
            store=self._store, config=config, llm_client=llm_client,
        )
        self._entity_buffer: List[ObservationNode] = []
        self._entity_flush_count = 0
        self._entity_last_extract_time = time.time()
        self._wm = WorkingMemory(
            store=self._store, config=config,
            on_flush=self._on_observations_flushed,
        )
        self._tools = MemoryTools(
            store=self._store,
            get_current_time=self._get_time,
            get_current_position=lambda: self._wm.current_position,
        )

    def _on_observations_flushed(self, observations: List[ObservationNode]) -> None:
        self._entity_buffer.extend(observations)
        self._entity_flush_count += 1
        now = time.time()
        elapsed = now - self._entity_last_extract_time
        if (self._entity_flush_count >= self._config.entity_extract_flush_interval
                or elapsed >= self._config.entity_extract_time_interval):
            self._drain_entity_buffer()

    def _drain_entity_buffer(self) -> None:
        if not self._entity_buffer:
            return
        batch = self._entity_buffer[:]
        self._entity_buffer.clear()
        self._entity_flush_count = 0
        self._entity_last_extract_time = time.time()
        self._consolidation.extract_entities_from_observations(batch)

    # ── Ingestion ─────────────────────────────────────────────────

    def add(
        self,
        text: str,
        x: float,
        y: float,
        z: float = 0.0,
        timestamp: Optional[float] = None,
        layer_name: str = "default",
        source_type: str = "manual",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
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

    def add_body_state(
        self,
        text: str,
        layer_name: str,
        timestamp: Optional[float] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a body-state (interoception) observation.

        Automatically sets ``source_type="interoception"`` and uses the
        robot's current position for coordinates (falls back to the origin
        if no position has been established yet).

        :param text: Body-state description (e.g. ``"battery: 45%"``).
        :param layer_name: Interoception layer (e.g. ``"battery"``, ``"cpu_temp"``).
        :param timestamp: Observation timestamp.  Defaults to current time.
        :param confidence: Confidence score in ``[0, 1]``.
        :param metadata: Arbitrary key-value metadata.
        :returns: Observation ID.
        :rtype: str
        """
        pos = self._wm.current_position
        if pos is None:
            pos = np.array([0.0, 0.0, 0.0])
        return self.add(
            text=text,
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            timestamp=timestamp,
            layer_name=layer_name,
            source_type="interoception",
            confidence=confidence,
            metadata=metadata,
        )

    # ── Episodes ──────────────────────────────────────────────────

    def start_episode(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_episode_id: Optional[str] = None,
    ) -> str:
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

    def end_episode(self, consolidate: bool = True) -> Optional[str]:
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
        self._drain_entity_buffer()
        ep_id = self._wm.active_episode_id
        self._wm.active_episode_id = None

        if consolidate:
            gist_ids = self._consolidation.consolidate_episode(ep_id)
            gist_text = ""
            if gist_ids:
                gist_texts = []
                for gid in gist_ids:
                    g = self._store.get_gist(gid)
                    if g and g.text:
                        gist_texts.append(g.text)
                gist_text = "\n".join(gist_texts)
            self._store.end_episode(ep_id, self._get_time(), gist=gist_text)
        else:
            self._store.end_episode(ep_id, self._get_time())

        return ep_id

    @property
    def active_episode_id(self) -> Optional[str]:
        return self._wm.active_episode_id

    # ── Queries (LLM tool interface) ──────────────────────────────

    def _ensure_flushed(self) -> None:
        if self._wm.buffer_size > 0:
            self._wm.flush()

    def semantic_search(self, query: str, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.semantic_search(query=query, **kwargs)

    def spatial_query(self, x: float, y: float, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.spatial_query(x=x, y=y, **kwargs)

    def temporal_query(self, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.temporal_query(**kwargs)

    def episode_summary(self, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.episode_summary(**kwargs)

    def get_current_context(self, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.get_current_context(**kwargs)

    def search_gists(self, query: str, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.search_gists(query=query, **kwargs)

    def add_entity(
        self,
        name: str,
        x: float,
        y: float,
        z: float = 0.0,
        timestamp: Optional[float] = None,
        entity_type: Optional[str] = None,
        confidence: float = 1.0,
        layer_name: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        ts = timestamp or self._get_time()
        entity = EntityNode(
            name=name,
            coordinates=np.array([x, y, z]),
            last_seen=ts,
            first_seen=ts,
            confidence=confidence,
            entity_type=entity_type,
            layer_name=layer_name,
            metadata=metadata or {},
        )
        return self._store.add_entity(entity)

    def entity_query(self, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.entity_query(**kwargs)

    def body_status(self, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.body_status(**kwargs)

    def locate(self, concept: str, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.locate(concept=concept, **kwargs)

    def recall(self, query: str, **kwargs: Any) -> str:
        self._ensure_flushed()
        return self._tools.recall(query=query, **kwargs)

    # ── Consolidation ─────────────────────────────────────────────

    def consolidate_time_window(self) -> List[str]:
        """Manually trigger time-window consolidation.

        Normally you don't need to call this -- episode consolidation happens
        automatically in :meth:`end_episode`.  Use this for non-episodic
        observations that have aged past the consolidation window.

        :returns: Created gist IDs.
        :rtype: List[str]
        """
        self._ensure_flushed()
        return self._consolidation.consolidate_time_window(reference_time=self._get_time())

    def maintenance(self) -> int:
        """Archive long-term observations that have aged past the threshold.

        :returns: Number of observations archived.
        :rtype: int
        """
        return self._consolidation.archive_long_term(reference_time=self._get_time())

    # ── Tool dispatch (for LLM integration) ───────────────────────

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return tool definitions suitable for LLM function calling.

        :returns: List of tool definition dicts.
        :rtype: List[Dict[str, Any]]
        """
        return self._tools.get_tool_definitions()

    def dispatch_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call by name.  Auto-flushes before executing.

        :param tool_name: One of the ten tool names.
        :param arguments: Tool arguments dict.
        :returns: Formatted result string.
        :rtype: str
        """
        self._ensure_flushed()
        return self._tools.dispatch_tool_call(tool_name, arguments)

    def get_tools_for_registration(self) -> List[Tuple[Callable, Dict]]:
        """Return ``(callable, description)`` pairs for external tool registration.

        Each callable wraps :meth:`dispatch_tool_call` for a single tool.
        Compatible with EmbodiedAgents ``LLM.register_tool()``.

        :returns: List of (function, OpenAI tool description) tuples.
        :rtype: List[Tuple[Callable, Dict]]
        """
        def _make_fn(tool_name: str) -> Callable:
            def fn(**kwargs: Any) -> str:
                return self.dispatch_tool_call(tool_name, kwargs)
            fn.__name__ = tool_name
            return fn

        return [
            (_make_fn(td["function"]["name"]), td)
            for td in self.get_tool_definitions()
        ]

    # ── Programmatic access ───────────────────────────────────────

    @property
    def store(self) -> MemoryStore:
        """Direct access to the underlying :class:`~emem.store.MemoryStore`."""
        return self._store

    @property
    def current_position(self) -> Optional[np.ndarray]:
        return self._wm.current_position

    def get_recent(self, n: Optional[int] = None) -> List[ObservationNode]:
        """Get recent observations from the in-memory buffer.

        :param n: Return only the last *n* observations.
        :returns: List of recent observations.
        :rtype: List[ObservationNode]
        """
        return self._wm.get_recent(n)

    # ── Lifecycle ─────────────────────────────────────────────────

    def save(self) -> None:
        self._wm.flush()
        self._drain_entity_buffer()
        self._store.save()

    def close(self) -> None:
        self._wm.flush()
        self._drain_entity_buffer()
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
