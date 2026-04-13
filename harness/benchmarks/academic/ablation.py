from dataclasses import dataclass, field
from typing import Dict, List, Optional

ALL_TOOLS = [
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
]

SPATIAL_TOOLS = {"locate", "recall", "spatial_query"}


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""

    name: str
    use_consolidation: bool = True
    use_multi_layer: bool = True
    available_tools: List[str] = field(default_factory=lambda: list(ALL_TOOLS))

    def filter_tool_definitions(self, tool_defs: List[dict]) -> List[dict]:
        """Return only tool definitions whose names are in ``available_tools``.

        :param tool_defs: Full list of tool definition dicts.
        :returns: Filtered list.
        """
        allowed = set(self.available_tools)
        return [t for t in tool_defs if t.get("function", t)["name"] in allowed]


ABLATIONS: Dict[str, AblationConfig] = {
    "full": AblationConfig("full"),
    "vector_only": AblationConfig(
        "vector_only",
        available_tools=["semantic_search"],
    ),
    "no_consolidation": AblationConfig(
        "no_consolidation",
        use_consolidation=False,
    ),
    "no_spatial": AblationConfig(
        "no_spatial",
        available_tools=[t for t in ALL_TOOLS if t not in SPATIAL_TOOLS],
    ),
    "flat_layer": AblationConfig(
        "flat_layer",
        use_multi_layer=False,
    ),
}
