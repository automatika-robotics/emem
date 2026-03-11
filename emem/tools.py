import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from emem.store import MemoryStore
from emem.types import EntityNode, GistNode


def _parse_relative_time(value: str, reference_time: Optional[float] = None) -> float:
    """Parse relative time strings like ``'-10m'``, ``'-1h'``, ``'-2d'``
    into absolute timestamps.

    :param value: Time string or numeric timestamp.
    :param reference_time: Reference point for relative values.
    :returns: Absolute timestamp as float.
    :rtype: float
    """
    ref = reference_time or time.time()
    match = re.match(r"^-(\d+(?:\.\d+)?)\s*([smhd])$", value.strip())
    if not match:
        return float(value)
    amount = float(match.group(1))
    unit = match.group(2)
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return ref - amount * multiplier[unit]


def _format_observation(obs: Any, include_coords: bool = True) -> str:
    parts = [f"[{obs.layer_name}]"]
    if include_coords:
        c = obs.coordinates
        parts.append(f"({c[0]:.1f},{c[1]:.1f})")
    parts.append(obs.text)
    return " ".join(parts)


def _format_memory_result(item: Any, include_coords: bool = True) -> str:
    if isinstance(item, EntityNode):
        c = item.coordinates
        type_label = item.entity_type or "object"
        return (
            f"[entity/{type_label}] "
            f"({c[0]:.1f},{c[1]:.1f}) "
            f"[seen {item.observation_count}x] {item.name}"
        )
    if isinstance(item, GistNode):
        pos = item.center_position
        return (
            f"[gist/{item.layer_name or 'cross-layer'}] "
            f"({pos[0]:.1f},{pos[1]:.1f}) "
            f"[{item.source_observation_count} obs] {item.text}"
        )
    return _format_observation(item, include_coords)


def _format_observations(observations: list, include_coords: bool = True) -> str:
    if not observations:
        return "No observations found."
    lines = []
    for i, obs in enumerate(observations, 1):
        lines.append(f"{i}. {_format_observation(obs, include_coords)}")
    return "\n".join(lines)


def _format_observations_by_layer(observations: list, include_coords: bool = True) -> str:
    """Format observations grouped by layer_name."""
    if not observations:
        return "No observations found."
    by_layer: Dict[str, list] = {}
    for obs in observations:
        by_layer.setdefault(obs.layer_name, []).append(obs)
    lines = []
    for layer_name in sorted(by_layer.keys()):
        lines.append(f"  [{layer_name}]")
        for obs in by_layer[layer_name]:
            parts = []
            if include_coords:
                c = obs.coordinates
                parts.append(f"({c[0]:.1f},{c[1]:.1f})")
            parts.append(obs.text)
            lines.append(f"    - {' '.join(parts)}")
    return "\n".join(lines)


def _format_results(results: list, include_coords: bool = True) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. {_format_memory_result(item, include_coords)}")
    return "\n".join(lines)


class MemoryTools:
    """LLM tool interface providing ten memory query tools.

    Each tool method returns a formatted string for token-efficient LLM
    consumption.
    """

    _TOOL_NAMES = frozenset({
        "semantic_search", "spatial_query", "temporal_query",
        "episode_summary", "get_current_context", "search_gists",
        "entity_query", "locate", "recall", "body_status",
    })

    def __init__(
        self,
        store: MemoryStore,
        get_current_time: Optional[Callable[[], float]] = None,
        get_current_position: Optional[Callable] = None,
    ) -> None:
        self.store = store
        self._get_time = get_current_time or time.time
        self._get_position = get_current_position

    def _resolve_time(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        return _parse_relative_time(value, self._get_time())

    def _time_range(
        self,
        time_after: Optional[str],
        time_before: Optional[str],
    ) -> Optional[Tuple[float, float]]:
        after = self._resolve_time(time_after)
        before = self._resolve_time(time_before)
        if after is None and before is None:
            return None
        return (after or 0.0, before or self._get_time())

    # ── Tool 1: Semantic Search ───────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        layer: Optional[str] = None,
        time_after: Optional[str] = None,
        time_before: Optional[str] = None,
        near_x: Optional[float] = None,
        near_y: Optional[float] = None,
        spatial_radius: Optional[float] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        spatial_center = None
        if near_x is not None and near_y is not None:
            spatial_center = np.array([near_x, near_y, 0.0])

        results = self.store.semantic_search(
            query=query,
            n_results=n_results,
            layer=layer,
            time_range=self._time_range(time_after, time_before),
            spatial_center=spatial_center,
            spatial_radius=spatial_radius,
            episode_id=episode_id,
        )
        return _format_results(results)

    # ── Tool 2: Spatial Query ─────────────────────────────────────────

    def spatial_query(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        radius: float = 2.0,
        layer: Optional[str] = None,
        time_after: Optional[str] = None,
        time_before: Optional[str] = None,
        n_results: int = 10,
        source_type: Optional[str] = None,
        exclude_source_type: Optional[str] = None,
    ) -> str:
        results = self.store.spatial_query(
            center=np.array([x, y, z]),
            radius=radius,
            layer=layer,
            time_range=self._time_range(time_after, time_before),
            n_results=n_results,
            source_type=source_type,
            exclude_source_type=exclude_source_type,
        )
        return _format_observations(results)

    # ── Tool 3: Temporal Query ────────────────────────────────────────

    def temporal_query(
        self,
        time_after: Optional[str] = None,
        time_before: Optional[str] = None,
        last_n_minutes: Optional[float] = None,
        layer: Optional[str] = None,
        near_x: Optional[float] = None,
        near_y: Optional[float] = None,
        spatial_radius: Optional[float] = None,
        order: str = "newest",
        n_results: int = 10,
        source_type: Optional[str] = None,
        exclude_source_type: Optional[str] = None,
    ) -> str:
        spatial_center = None
        if near_x is not None and near_y is not None:
            spatial_center = np.array([near_x, near_y, 0.0])

        last_n_seconds = last_n_minutes * 60 if last_n_minutes else None
        time_range = self._time_range(time_after, time_before)
        ref_time = self._get_time()

        results = self.store.temporal_query(
            time_range=time_range,
            last_n_seconds=last_n_seconds,
            layer=layer,
            spatial_center=spatial_center,
            spatial_radius=spatial_radius,
            order=order,
            n_results=n_results,
            reference_time=ref_time,
            source_type=source_type,
            exclude_source_type=exclude_source_type,
        )

        if results:
            return _format_observations(results)

        # Fallback: observations may have been consolidated (archived).
        # Search gists covering the same time window instead.
        # Resolve the effective time_after for gist overlap query.
        effective_after = time_range[0] if time_range else None
        if effective_after is None and last_n_seconds is not None:
            effective_after = ref_time - last_n_seconds
        gists = self.store.get_recent_gists(
            time_after=effective_after,
            time_before=time_range[1] if time_range else None,
            layer=layer,
            order=order,
            n_results=n_results,
        )
        if gists:
            lines = ["(Observations consolidated — showing summaries)"]
            for i, g in enumerate(gists, 1):
                pos = g.center_position
                layer_label = g.layer_name or "cross-layer"
                lines.append(
                    f"{i}. [{layer_label}] ({pos[0]:.1f},{pos[1]:.1f}) "
                    f"[{g.source_observation_count} obs] {g.text}"
                )
            return "\n".join(lines)

        return "No observations found."

    # ── Tool 4: Episode Summary ───────────────────────────────────────

    def episode_summary(
        self,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        last_n: int = 1,
    ) -> str:
        if episode_id:
            ep = self.store.get_episode(episode_id)
            if not ep:
                return f"Episode {episode_id} not found."
            episodes = [ep]
        else:
            episodes = self.store.list_episodes(task_name=task_name, last_n=last_n)

        if not episodes:
            return "No episodes found."

        lines = []
        for ep in episodes:
            status = ep.status
            gist = ep.gist or "(no summary)"
            line = f"[{ep.name}] ({status}) {gist}"
            lines.append(line)
        return "\n".join(lines)

    # ── Tool 5: Current Context ───────────────────────────────────────

    def get_current_context(
        self,
        radius: float = 3.0,
        include_recent_minutes: float = 5.0,
    ) -> str:
        parts = []

        pos = self._get_position() if self._get_position else None
        if pos is not None:
            parts.append(f"Position: ({pos[0]:.1f}, {pos[1]:.1f})")
            nearby = self.store.spatial_query(
                center=pos, radius=radius, n_results=10,
            )
            if nearby:
                parts.append(f"Nearby ({radius}m):")
                parts.append(_format_observations_by_layer(nearby, include_coords=False))

            area_gists = self.store.search_gists_by_area(center=pos, radius=radius)
            if area_gists:
                parts.append("Area summaries:")
                for g in area_gists:
                    parts.append(f"  - {g.text}")

            nearby_entities = self.store.query_entities(
                near_coordinates=pos, spatial_radius=radius, n_results=10,
            )
            if nearby_entities:
                parts.append("Nearby entities:")
                for ent in nearby_entities:
                    type_label = ent.entity_type or "object"
                    parts.append(f"  - [{type_label}] {ent.name} (seen {ent.observation_count}x)")

        recent = self.store.temporal_query(
            last_n_seconds=include_recent_minutes * 60,
            n_results=10,
            reference_time=self._get_time(),
        )
        if recent:
            parts.append(f"Recent ({include_recent_minutes}min):")
            parts.append(_format_observations_by_layer(recent, include_coords=True))

        body = self.body_status()
        if body and "No body state" not in body:
            parts.append(body)

        return "\n".join(parts) if parts else "No context available."

    # ── Tool 6: Search Gists ─────────────────────────────────────────

    def search_gists(
        self,
        query: str,
        n_results: int = 5,
        time_after: Optional[str] = None,
        time_before: Optional[str] = None,
    ) -> str:
        results = self.store.search_gists(
            query=query,
            n_results=n_results,
            time_range=self._time_range(time_after, time_before),
        )
        if not results:
            return "No gists found."

        lines = []
        for i, g in enumerate(results, 1):
            pos = g.center_position
            lines.append(
                f"{i}. [{g.layer_name or 'cross-layer'}] "
                f"({pos[0]:.1f},{pos[1]:.1f}) "
                f"[{g.source_observation_count} obs] {g.text}"
            )
        return "\n".join(lines)

    # ── Tool 7: Entity Query ───────────────────────────────────────

    def entity_query(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        near_x: Optional[float] = None,
        near_y: Optional[float] = None,
        spatial_radius: Optional[float] = None,
        last_seen_after: Optional[str] = None,
        n_results: int = 10,
    ) -> str:
        near_coordinates = None
        if near_x is not None and near_y is not None:
            near_coordinates = np.array([near_x, near_y, 0.0])

        last_seen_ts = None
        if last_seen_after is not None:
            last_seen_ts = _parse_relative_time(last_seen_after, self._get_time())

        results = self.store.query_entities(
            name=name,
            entity_type=entity_type,
            near_coordinates=near_coordinates,
            spatial_radius=spatial_radius,
            last_seen_after=last_seen_ts,
            n_results=n_results,
        )
        if not results:
            return "No entities found."
        lines = []
        for i, entity in enumerate(results, 1):
            lines.append(f"{i}. {_format_memory_result(entity)}")
        return "\n".join(lines)

    # ── Tool 8: Locate ────────────────────────────────────────────

    def _locate_coords(
        self,
        concept: str,
        n_results: int = 10,
        layer: Optional[str] = None,
        time_after: Optional[str] = None,
        time_before: Optional[str] = None,
    ) -> Optional[Tuple[np.ndarray, float, int, list]]:
        """Returns (centroid, radius, count, results) or None."""
        results = self.store.semantic_search(
            query=concept,
            n_results=n_results,
            layer=layer,
            time_range=self._time_range(time_after, time_before),
        )
        if not results:
            return None
        coords = []
        for item in results:
            if hasattr(item, "coordinates"):
                coords.append(item.coordinates)
            elif hasattr(item, "center_position"):
                coords.append(item.center_position)
        if not coords:
            return None
        coords_arr = np.array(coords)
        centroid = coords_arr.mean(axis=0)
        distances = np.linalg.norm(coords_arr - centroid, axis=1)
        radius = float(distances.max()) if len(distances) > 1 else 1.0
        return centroid, radius, len(coords), results

    def locate(
        self,
        concept: str,
        n_results: int = 10,
        layer: Optional[str] = None,
        time_after: Optional[str] = None,
        time_before: Optional[str] = None,
    ) -> str:
        result = self._locate_coords(
            concept, n_results, layer, time_after, time_before,
        )
        if result is None:
            return "Could not locate: no matching memories found."
        centroid, radius, count, results = result

        layers_seen: Dict[str, int] = {}
        for item in results:
            ln = getattr(item, "layer_name", None) or "unknown"
            layers_seen[ln] = layers_seen.get(ln, 0) + 1
        layer_summary = ", ".join(
            f"{v}x {k}" for k, v in sorted(layers_seen.items())
        )

        parts = [
            f"Location: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})",
            f"Radius: {radius:.1f}m",
            f"Based on: {count} memories ({layer_summary})",
        ]
        for item in results[:3]:
            text = getattr(item, "text", None) or getattr(item, "name", "")
            if text:
                ln = getattr(item, "layer_name", "") or ""
                parts.append(f"  [{ln}] {text[:80]}")

        return "\n".join(parts)

    # ── Tool 9: Recall ─────────────────────────────────────────────

    def recall(
        self,
        query: str,
        n_results: int = 10,
        radius_multiplier: float = 1.5,
    ) -> str:
        location = self._locate_coords(query, n_results=n_results)
        if location is None:
            return f"No memories found for: {query}"
        centroid, radius, match_count, _results = location
        search_radius = max(radius * radius_multiplier, 2.0)

        observations = self.store.spatial_query(
            center=centroid, radius=search_radius, n_results=n_results * 3,
        )
        gists = self.store.search_gists_by_area(
            center=centroid, radius=search_radius,
        )
        entities = self.store.query_entities(
            near_coordinates=centroid, spatial_radius=search_radius,
            n_results=n_results,
        )

        parts = [
            f"About '{query}' — location ({centroid[0]:.1f}, {centroid[1]:.1f}), "
            f"radius {search_radius:.1f}m:"
        ]
        if observations:
            parts.append("Observations:")
            parts.append(_format_observations_by_layer(observations))
        if gists:
            parts.append("Summaries:")
            for g in gists:
                parts.append(
                    f"  [{g.layer_name or 'cross-layer'}] {g.text}"
                )
        if entities:
            parts.append("Entities:")
            for ent in entities:
                parts.append(
                    f"  [{ent.entity_type or 'object'}] {ent.name} "
                    f"(seen {ent.observation_count}x)"
                )
        return "\n".join(parts)

    # ── Tool 10: Body Status ─────────────────────────────────────────

    def body_status(self, layers: Optional[List[str]] = None) -> str:
        """Return the latest reading from each interoception layer.

        :param layers: Optional list of layer names to restrict results.
        :returns: Formatted body status string.
        :rtype: str
        """
        latest = self.store.get_latest_by_source_type("interoception", layers)
        if not latest:
            return "No body state data available."

        now = self._get_time()
        lines = ["Body Status:"]
        for layer_name in sorted(latest.keys()):
            obs = latest[layer_name]
            age_s = now - obs.timestamp
            if age_s < 60:
                age_str = f"{age_s:.0f}s ago"
            elif age_s < 3600:
                age_str = f"{age_s / 60:.0f}min ago"
            else:
                age_str = f"{age_s / 3600:.1f}h ago"
            lines.append(f"  [{layer_name}] {obs.text} ({age_str})")
        return "\n".join(lines)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return tool definitions suitable for LLM function calling.

        :returns: List of tool definition dicts in OpenAI function-calling format.
        :rtype: List[Dict[str, Any]]
        """
        return [
            {
                "name": "semantic_search",
                "description": "Search memory by meaning. Searches both recent observations and consolidated summaries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "n_results": {"type": "integer", "default": 5},
                        "layer": {"type": "string", "description": "Filter by layer name"},
                        "time_after": {"type": "string", "description": "After this time (e.g. '-10m', '-1h')"},
                        "time_before": {"type": "string", "description": "Before this time"},
                        "near_x": {"type": "number", "description": "X coordinate to search near"},
                        "near_y": {"type": "number", "description": "Y coordinate to search near"},
                        "spatial_radius": {"type": "number", "description": "Radius in meters"},
                        "episode_id": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "spatial_query",
                "description": "Find observations within a radius of a point.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number", "default": 0.0},
                        "radius": {"type": "number", "default": 2.0},
                        "layer": {"type": "string"},
                        "time_after": {"type": "string"},
                        "time_before": {"type": "string"},
                        "n_results": {"type": "integer", "default": 10},
                    },
                    "required": ["x", "y"],
                },
            },
            {
                "name": "temporal_query",
                "description": "Find observations in a time range, chronologically ordered. Falls back to consolidated summaries if raw observations have been archived.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_after": {"type": "string"},
                        "time_before": {"type": "string"},
                        "last_n_minutes": {"type": "number"},
                        "layer": {"type": "string"},
                        "near_x": {"type": "number"},
                        "near_y": {"type": "number"},
                        "spatial_radius": {"type": "number"},
                        "order": {"type": "string", "enum": ["newest", "oldest"], "default": "newest"},
                        "n_results": {"type": "integer", "default": 10},
                    },
                },
            },
            {
                "name": "episode_summary",
                "description": "Get summary of episode(s).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "episode_id": {"type": "string"},
                        "task_name": {"type": "string"},
                        "last_n": {"type": "integer", "default": 1},
                    },
                },
            },
            {
                "name": "get_current_context",
                "description": "Get situational awareness: what's nearby and recent activity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "radius": {"type": "number", "default": 3.0},
                        "include_recent_minutes": {"type": "number", "default": 5.0},
                    },
                },
            },
            {
                "name": "search_gists",
                "description": "Search consolidated memory summaries (long-term memory).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "n_results": {"type": "integer", "default": 5},
                        "time_after": {"type": "string"},
                        "time_before": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "entity_query",
                "description": "Find known entities (objects, people, landmarks). Entities are auto-tracked across observations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name substring match"},
                        "entity_type": {"type": "string", "description": "Type filter (e.g. 'furniture')"},
                        "near_x": {"type": "number"},
                        "near_y": {"type": "number"},
                        "spatial_radius": {"type": "number"},
                        "last_seen_after": {"type": "string", "description": "e.g. '-10m'"},
                        "n_results": {"type": "integer", "default": 10},
                    },
                },
            },
            {
                "name": "locate",
                "description": "Find the spatial location of a concept (e.g. 'kitchen', 'charging station'). Returns centroid coordinates and spread radius.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string", "description": "What to locate"},
                        "n_results": {"type": "integer", "default": 10},
                        "layer": {"type": "string", "description": "Filter by layer name"},
                        "time_after": {"type": "string"},
                        "time_before": {"type": "string"},
                    },
                    "required": ["concept"],
                },
            },
            {
                "name": "recall",
                "description": "Recall everything known about a concept. Locates it spatially, then gathers cross-layer observations, gists, and entities from that area.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to recall (e.g. 'kitchen', 'the red chair')"},
                        "n_results": {"type": "integer", "default": 10},
                        "radius_multiplier": {"type": "number", "default": 1.5, "description": "Multiplier for search radius around located area"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "body_status",
                "description": "Get the latest body/internal state readings (battery, temperature, joint health, etc.).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter to specific body-state layers (e.g. ['battery', 'cpu_temp'])",
                        },
                    },
                },
            },
        ]

    def dispatch_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call by name.

        :param tool_name: One of the ten tool names.
        :param arguments: Tool arguments dict.
        :returns: Formatted result string.
        :rtype: str
        """
        if tool_name not in self._TOOL_NAMES:
            return f"Unknown tool: {tool_name}"
        return getattr(self, tool_name)(**arguments)
