import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from emem.store import MemoryStore
from emem.types import EntityNode, GistNode


### Public tool schemas ###
# OpenAI-format JSON schemas for the 10 retrieval tools exposed by
# SpatioTemporalMemory. Keyed by tool name so callers can reference individual
# schemas at class-definition time without constructing a memory instance.
# Descriptions are written for tool-calling LLMs: each includes the tool's
# purpose, what it returns, and guidance on when to pick it over sibling
# tools. All time parameters accept either relative strings ('-10m', '-1h',
# '-2d') or numeric Unix timestamps. All coordinates are in world-frame meters.

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "semantic_search": {
        "name": "semantic_search",
        "description": (
            "Use when the question asks WHAT a topic is about and no more "
            "specific tool fits. General meaning-based retrieval over "
            "observations and summaries. Do NOT use for time-focused "
            "questions (use temporal_query), for a named entity's attributes "
            "(use entity_query), for long-ago events (use search_gists), "
            "or for 'where is X' (use locate). Prefer a single targeted "
            "query over chaining multiple searches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what to find, "
                        "e.g. 'coffee mug', 'person waving', 'charging station'."
                    ),
                },
                "n_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return.",
                },
                "layer": {
                    "type": "string",
                    "description": (
                        "Only return observations from this perception layer "
                        "(e.g. 'vision', 'speech', 'detections'). Omit to "
                        "search all layers."
                    ),
                },
                "time_after": {
                    "type": "string",
                    "description": (
                        "Only observations after this time. Relative "
                        "('-10m', '-1h', '-2d') or Unix timestamp."
                    ),
                },
                "time_before": {
                    "type": "string",
                    "description": (
                        "Only observations before this time. Same format as time_after."
                    ),
                },
                "near_x": {
                    "type": "number",
                    "description": (
                        "X coordinate of a spatial filter point, in "
                        "world-frame meters. Must be paired with near_y and "
                        "spatial_radius."
                    ),
                },
                "near_y": {
                    "type": "number",
                    "description": (
                        "Y coordinate of the spatial filter point, in "
                        "world-frame meters."
                    ),
                },
                "spatial_radius": {
                    "type": "number",
                    "description": (
                        "Radius in meters around (near_x, near_y). "
                        "Observations farther than this are excluded."
                    ),
                },
                "episode_id": {
                    "type": "string",
                    "description": (
                        "Limit to a specific episode (as returned by start_episode)."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    "spatial_query": {
        "name": "spatial_query",
        "description": (
            "Use when the question asks WHAT IS NEAR a specific "
            "coordinate. Returns all observations within a radius of "
            "(x, y, z), with no semantic filter. Only useful when "
            "observations carry real world coordinates — skip for "
            "text-only or conversational memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "X coordinate of the query point, in world-frame meters.",
                },
                "y": {
                    "type": "number",
                    "description": "Y coordinate of the query point, in world-frame meters.",
                },
                "z": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Z coordinate in meters. Use 0 for 2D maps.",
                },
                "radius": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Radius in meters. Observations farther than this are excluded.",
                },
                "layer": {
                    "type": "string",
                    "description": "Only return observations from this perception layer.",
                },
                "time_after": {
                    "type": "string",
                    "description": (
                        "Only observations after this time. Relative "
                        "('-10m', '-1h', '-2d') or Unix timestamp."
                    ),
                },
                "time_before": {
                    "type": "string",
                    "description": "Only observations before this time.",
                },
                "n_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return.",
                },
            },
            "required": ["x", "y"],
        },
    },
    "temporal_query": {
        "name": "temporal_query",
        "description": (
            "Use when the question asks WHEN something happened or requests "
            "events in a specific time window or date range — including "
            "'when did X', 'what happened on <date>', 'last <duration>', "
            "'between X and Y', 'yesterday', 'last week'. Returns "
            "observations ordered chronologically. Prefer this over "
            "semantic_search whenever the question is time-focused. Time "
            "arguments accept relative ('-10m', '2 hours ago', 'yesterday') "
            "or numeric Unix timestamps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "time_after": {
                    "type": "string",
                    "description": (
                        "Start of the time window. Relative ('-10m', '-1h', "
                        "'-2d') or Unix timestamp."
                    ),
                },
                "time_before": {
                    "type": "string",
                    "description": "End of the time window. Same format.",
                },
                "last_n_minutes": {
                    "type": "number",
                    "description": (
                        "Shortcut equivalent to setting time_after to "
                        "'-<N>m' and time_before to now. Use for questions "
                        "about recent history."
                    ),
                },
                "layer": {
                    "type": "string",
                    "description": "Only return observations from this perception layer.",
                },
                "near_x": {
                    "type": "number",
                    "description": "Optional spatial filter; X in world-frame meters.",
                },
                "near_y": {
                    "type": "number",
                    "description": "Optional spatial filter; Y in world-frame meters.",
                },
                "spatial_radius": {
                    "type": "number",
                    "description": "Radius in meters around (near_x, near_y) for the spatial filter.",
                },
                "order": {
                    "type": "string",
                    "enum": ["newest", "oldest"],
                    "default": "newest",
                    "description": "Sort order. 'newest' returns most recent first.",
                },
                "n_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return.",
                },
            },
        },
    },
    "episode_summary": {
        "name": "episode_summary",
        "description": (
            "Use when the question names a past episode, task, or session "
            "— e.g. 'the kitchen patrol', 'the last session', 'session 3', "
            "'yesterday's task'. Returns the consolidated summary of those "
            "episodes. Provide at least one of episode_id, task_name, or "
            "last_n."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "episode_id": {
                    "type": "string",
                    "description": (
                        "Exact episode ID (as returned by start_episode). "
                        "Use this if you already have the ID."
                    ),
                },
                "task_name": {
                    "type": "string",
                    "description": (
                        "Episode name substring. Returns all episodes "
                        "whose name matches."
                    ),
                },
                "last_n": {
                    "type": "integer",
                    "default": 1,
                    "description": (
                        "Return the last N episodes by time. Use when the "
                        "user says 'last', 'previous', or 'most recent task'."
                    ),
                },
            },
        },
    },
    "get_current_context": {
        "name": "get_current_context",
        "description": (
            "Use when the question asks about NOW — what the robot sees "
            "right now, what is around it, or what is going on at the "
            "current moment. Returns nearby entities, area summaries, "
            "recent observations, and current body status. Requires the "
            "robot to have a live position — not useful for text-only or "
            "past-focused questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "radius": {
                    "type": "number",
                    "default": 3.0,
                    "description": "Spatial radius in meters for 'nearby'.",
                },
                "include_recent_minutes": {
                    "type": "number",
                    "default": 5.0,
                    "description": "Time window in minutes for 'recent activity'.",
                },
            },
        },
    },
    "search_gists": {
        "name": "search_gists",
        "description": (
            "Use when the question is about EVENTS FROM LONG AGO (older "
            "than the recent observation horizon) or explicitly asks for "
            "a high-level summary rather than specific events. Searches "
            "consolidated summaries only. Try semantic_search first; "
            "fall back to this only when semantic_search returns nothing "
            "and the question is long-horizon."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in long-term memory.",
                },
                "n_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return.",
                },
                "time_after": {
                    "type": "string",
                    "description": (
                        "Only gists covering a time after this. Relative "
                        "('-1h', '-1d') or Unix timestamp."
                    ),
                },
                "time_before": {
                    "type": "string",
                    "description": "Only gists covering a time before this.",
                },
            },
            "required": ["query"],
        },
    },
    "entity_query": {
        "name": "entity_query",
        "description": (
            "Use when the question names a SPECIFIC persistent entity "
            "(object, person, landmark) and asks about that entity's "
            "attributes, count, or last-known location — e.g. 'how often "
            "did I see Maria', 'is wall a common object', 'where was the "
            "red chair last seen'. Returns entity records with name, "
            "type, coordinates, and observation count. Fall back to "
            "semantic_search if the entity is not found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Substring to match against entity names, e.g. "
                        "'chair', 'alice'. Case-insensitive."
                    ),
                },
                "entity_type": {
                    "type": "string",
                    "description": (
                        "Filter by type, e.g. 'furniture', 'person', "
                        "'landmark'. Combine with name for precise lookup."
                    ),
                },
                "near_x": {
                    "type": "number",
                    "description": "Optional spatial filter; X in world-frame meters.",
                },
                "near_y": {
                    "type": "number",
                    "description": "Optional spatial filter; Y in world-frame meters.",
                },
                "spatial_radius": {
                    "type": "number",
                    "description": "Radius in meters for the spatial filter.",
                },
                "last_seen_after": {
                    "type": "string",
                    "description": (
                        "Only entities last observed after this time. "
                        "Format: '-10m', '-1h', '-2d'."
                    ),
                },
                "n_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return.",
                },
            },
        },
    },
    "locate": {
        "name": "locate",
        "description": (
            "Use when the question asks WHERE a concept is — e.g. 'where "
            "is the kitchen', 'find the charging station'. Returns a "
            "centroid (x, y, z) and a spread radius. Only useful when "
            "observations were stored with real world-frame coordinates; "
            "skip for text-only or conversational memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "What to locate. Natural-language phrase.",
                },
                "n_results": {
                    "type": "integer",
                    "default": 10,
                    "description": (
                        "How many matching observations to aggregate when "
                        "computing the centroid."
                    ),
                },
                "layer": {
                    "type": "string",
                    "description": "Only aggregate observations from this perception layer.",
                },
                "time_after": {
                    "type": "string",
                    "description": "Only consider observations after this time.",
                },
                "time_before": {
                    "type": "string",
                    "description": "Only consider observations before this time.",
                },
            },
            "required": ["concept"],
        },
    },
    "recall": {
        "name": "recall",
        "description": (
            "Use when the question asks 'tell me about X' or 'describe "
            "X' where X is a physical PLACE or entity with a consistent "
            "real-world location (e.g. 'describe the kitchen', "
            "'everything about Maria's desk'). Returns a cross-layer "
            "bundle: observations + gists + entities around that "
            "location. Only useful for robots with real coordinates; "
            "skip for text-only memory and abstract topics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to recall. Natural-language phrase.",
                },
                "n_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Target number of items per source.",
                },
                "radius_multiplier": {
                    "type": "number",
                    "default": 1.5,
                    "description": (
                        "Multiplier on the located spread radius for "
                        "gathering nearby data. Larger values cast a wider "
                        "net."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    "body_status": {
        "name": "body_status",
        "description": (
            "Use when the question is about the robot's OWN INTERNAL "
            "STATE — battery level, CPU temperature, joint health, or "
            "other body/interoception readings. Do NOT use for anything "
            "about the external environment. Returns the latest "
            "reading from each matching body-state layer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter to specific body-state layer names, e.g. "
                        "['battery', 'cpu_temp', 'joint_health']. Omit to "
                        "return all body layers."
                    ),
                },
            },
        },
    },
}


# ── Relative time parsing ─────────────────────────────────────────────
#
# The retrieval tools accept strings that describe points in the past,
# plus numeric Unix timestamps. The parser is lenient so that weak tool-
# calling LLMs do not lose step budget arguing with the exact format.
#
# Supported forms (case-insensitive, whitespace-tolerant):
#   Canonical relative:  '-10m', '-1h', '-2d', '-30s', '-1w'
#   Without leading '-': '10m', '1h', '2d', '30s', '1w'
#   Longer unit names:   '10min', '10 minutes', '2 hours', '3 days',
#                        '1 week', '30 seconds'
#   Wrapped:             'last 10 minutes', 'past 2 hours',
#                        'the last hour', '10 minutes ago', '2h ago'
#   'a'/'an' as 1:       'a minute', 'an hour', 'a day'
#   Bare unit names:     'minute', 'hour', 'day', 'week'
#   Named keywords:      'now', 'today', 'yesterday', 'last week',
#                        'last month', 'last year'
#   Numeric:             '1715000000', '1715000000.5'  (Unix timestamp)

_UNIT_SECONDS: Dict[str, int] = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
    "w": 604800,
    "wk": 604800,
    "wks": 604800,
    "week": 604800,
    "weeks": 604800,
}

_NAMED_OFFSETS: Dict[str, int] = {
    "now": 0,
    "today": 0,
    "yesterday": 86400,
    "last week": 604800,
    "past week": 604800,
    "last month": 2592000,  # ~30 days
    "past month": 2592000,
    "last year": 31536000,  # 365 days
    "past year": 31536000,
}

_WRAPPER_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:last|past|previous|within\s+the\s+last)\s+"
)
_WRAPPER_SUFFIX_RE = re.compile(r"\s+ago$")
_LEADING_MINUS_RE = re.compile(r"^-\s*")
_AN_PREFIX_RE = re.compile(r"^(?:a|an)\s+")
_NUM_UNIT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([a-z]+)$")


def _parse_relative_time(value: str, reference_time: Optional[float] = None) -> float:
    """Parse a human-friendly time expression into an absolute Unix timestamp.

    Accepts the canonical sugarcoat-style ``-10m`` form, a range of
    natural-language variants (``'10 minutes ago'``, ``'last hour'``,
    ``'yesterday'``), and numeric Unix timestamps. Returns the absolute
    timestamp the expression resolves to.

    :param value: Time string or numeric timestamp.
    :param reference_time: "Now" reference for relative values.
    :returns: Absolute Unix timestamp as float.
    :raises ValueError: if ``value`` cannot be parsed.
    """
    ref = reference_time if reference_time is not None else time.time()
    v = value.strip().lower()

    if not v:
        raise ValueError(f"Cannot parse empty time string: {value!r}")

    # Named offsets first — exact match.
    if v in _NAMED_OFFSETS:
        return ref - _NAMED_OFFSETS[v]

    # Strip conversational wrappers and any leading minus.
    stripped = _WRAPPER_PREFIX_RE.sub("", v)
    stripped = _WRAPPER_SUFFIX_RE.sub("", stripped)
    stripped = _LEADING_MINUS_RE.sub("", stripped)
    stripped = _AN_PREFIX_RE.sub("1 ", stripped)

    # <number> <unit>  (whitespace optional)
    m = _NUM_UNIT_RE.match(stripped)
    if m:
        amount = float(m.group(1))
        unit = m.group(2)
        secs = _UNIT_SECONDS.get(unit)
        if secs is not None:
            return ref - amount * secs

    # Bare unit with the wrappers stripped implies "1 <unit>"
    # (e.g. "hour", "day", "last minute").
    if stripped in _UNIT_SECONDS:
        return ref - _UNIT_SECONDS[stripped]

    # Numeric Unix timestamp fallback.
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"Cannot parse time string: {value!r}") from e


def _format_observation(obs: Any, include_coords: bool = True) -> str:
    """Format a single observation as a one-line display string."""
    parts = [f"[{obs.layer_name}]"]
    if include_coords:
        c = obs.coordinates
        parts.append(f"({c[0]:.1f},{c[1]:.1f})")
    dt = datetime.fromtimestamp(obs.timestamp, tz=timezone.utc)
    parts.append(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}]")
    parts.append(obs.text)
    return " ".join(parts)


def _format_memory_result(item: Any, include_coords: bool = True) -> str:
    """Format an entity, gist, or observation for display."""
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
    """Format a list of observations as a numbered string list."""
    if not observations:
        return "No observations found."
    lines = []
    for i, obs in enumerate(observations, 1):
        lines.append(f"{i}. {_format_observation(obs, include_coords)}")
    return "\n".join(lines)


def _format_observations_by_layer(
    observations: list, include_coords: bool = True
) -> str:
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
    """Format a heterogeneous result list as a numbered string list."""
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
        "semantic_search",
        "spatial_query",
        "temporal_query",
        "episode_summary",
        "get_current_context",
        "search_gists",
        "entity_query",
        "locate",
        "recall",
        "body_status",
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
        """Resolve a relative or absolute time string to a Unix timestamp."""
        if value is None:
            return None
        return _parse_relative_time(value, self._get_time())

    def _time_range(
        self,
        time_after: Optional[str],
        time_before: Optional[str],
    ) -> Optional[Tuple[float, float]]:
        """Resolve ``(time_after, time_before)`` strings into a numeric range."""
        after = self._resolve_time(time_after)
        before = self._resolve_time(time_before)
        if after is None and before is None:
            return None
        return (after or 0.0, before or self._get_time())

    # ── Tool 1: Semantic Search ───────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        layer: Optional[str] = None,
        time_after: Optional[str] = None,
        time_before: Optional[str] = None,
        near_x: Optional[float] = None,
        near_y: Optional[float] = None,
        spatial_radius: Optional[float] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        """Run a semantic similarity search over stored memories."""
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
            reference_time=self._get_time(),
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
        """Return observations within *radius* of ``(x, y, z)``."""
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
        """Return observations matching a temporal (and optional spatial) filter."""
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
        """Return formatted summaries for one or more episodes."""
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
            obs = self.store.get_episode_observations(ep.id)
            if obs:
                t0 = datetime.fromtimestamp(obs[0].timestamp, tz=timezone.utc)
                t1 = datetime.fromtimestamp(obs[-1].timestamp, tz=timezone.utc)
                line += (
                    f"\n  ({len(obs)} observations from "
                    f"{t0.strftime('%Y-%m-%d %H:%M:%S')} to "
                    f"{t1.strftime('%Y-%m-%d %H:%M:%S')})"
                )
            lines.append(line)
        return "\n".join(lines)

    # ── Tool 5: Current Context ───────────────────────────────────────

    def get_current_context(
        self,
        radius: float = 3.0,
        include_recent_minutes: float = 5.0,
    ) -> str:
        """Summarize the agent's current spatial and temporal context."""
        parts = []

        pos = self._get_position() if self._get_position else None
        if pos is not None:
            parts.append(f"Position: ({pos[0]:.1f}, {pos[1]:.1f})")
            nearby = self.store.spatial_query(
                center=pos,
                radius=radius,
                n_results=10,
            )
            if nearby:
                parts.append(f"Nearby ({radius}m):")
                parts.append(
                    _format_observations_by_layer(nearby, include_coords=False)
                )

            area_gists = self.store.search_gists_by_area(center=pos, radius=radius)
            if area_gists:
                parts.append("Area summaries:")
                for g in area_gists:
                    parts.append(f"  - {g.text}")

            nearby_entities = self.store.query_entities(
                near_coordinates=pos,
                spatial_radius=radius,
                n_results=10,
            )
            if nearby_entities:
                parts.append("Nearby entities:")
                for ent in nearby_entities:
                    type_label = ent.entity_type or "object"
                    parts.append(
                        f"  - [{type_label}] {ent.name} (seen {ent.observation_count}x)"
                    )

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
        n_results: int = 10,
        time_after: Optional[str] = None,
        time_before: Optional[str] = None,
    ) -> str:
        """Search consolidated gists for *query* and return formatted results."""
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
        """Query stored entities by name, type, location, or recency."""
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
            reference_time=self._get_time(),
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
        """Locate the centroid of memories matching *concept*."""
        result = self._locate_coords(
            concept,
            n_results,
            layer,
            time_after,
            time_before,
        )
        if result is None:
            return "Could not locate: no matching memories found."
        centroid, radius, count, results = result

        layers_seen: Dict[str, int] = {}
        for item in results:
            ln = getattr(item, "layer_name", None) or "unknown"
            layers_seen[ln] = layers_seen.get(ln, 0) + 1
        layer_summary = ", ".join(f"{v}x {k}" for k, v in sorted(layers_seen.items()))

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
        """Recall observations near the location associated with *query*."""
        location = self._locate_coords(query, n_results=n_results)
        if location is None:
            return f"No memories found for: {query}"
        centroid, radius, _, _results = location
        search_radius = max(radius * radius_multiplier, 2.0)

        observations = self.store.spatial_query(
            center=centroid,
            radius=search_radius,
            n_results=n_results * 3,
        )
        gists = self.store.search_gists_by_area(
            center=centroid,
            radius=search_radius,
        )
        entities = self.store.query_entities(
            near_coordinates=centroid,
            spatial_radius=search_radius,
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
                parts.append(f"  [{g.layer_name or 'cross-layer'}] {g.text}")
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
        """Return tool definitions in OpenAI function-calling format.

        Each entry has the shape
        ``{"type": "function", "function": {"name", "description", "parameters"}}``.

        :returns: List of tool definition dicts.
        :rtype: List[Dict[str, Any]]
        """
        return [{"type": "function", "function": d} for d in self._tool_schemas()]

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        """Raw tool schemas (name/description/parameters only)."""
        return list(TOOL_SCHEMAS.values())

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
