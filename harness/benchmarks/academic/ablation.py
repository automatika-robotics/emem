from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

# Dataset-specific tool filters. Tools that cannot meaningfully apply to a
# dataset are excluded so the model doesn't waste a tool-selection slot
# weighing them. Applied by the runner after the ablation's own filter.
DATASET_TOOL_FILTERS: Dict[str, List[str]] = {
    "locomo": [
        "semantic_search",
        "temporal_query",
        "search_gists",
        "entity_query",
        "episode_summary",
        "get_current_context",
    ],
    # eMEM-Bench v1 exercises all 10 tools, so no filter.
    "emem-bench-v1": list(ALL_TOOLS),
    # SQA3D is a single-frame spatial QA, not a memory benchmark, but
    # included for completeness.
    "sqa3d": list(ALL_TOOLS),
}


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant.

    ``mem_config_overrides`` lets an ablation flip fields on the
    :class:`~emem.config.SpatioTemporalMemoryConfig` used for the
    sample (e.g. to disable hybrid retrieval or recency weighting
    for a comparative run). The runner merges these into any CLI-
    level overrides at sample-construction time.
    """

    name: str
    use_consolidation: bool = True
    use_multi_layer: bool = True
    available_tools: List[str] = field(default_factory=lambda: list(ALL_TOOLS))
    mem_config_overrides: Dict[str, Any] = field(default_factory=dict)

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
    # Cross-system baseline: plain RAG (semantic search only, no
    # consolidation, no layer structure). Closest fair comparison to
    # "just give the agent a vector store + timestamp filter". Intended
    # to answer the reviewer's "is this just what the LLM does with any
    # retrieval tool" question.
    "flat_rag": AblationConfig(
        "flat_rag",
        use_consolidation=False,
        use_multi_layer=False,
        available_tools=["semantic_search", "temporal_query"],
    ),
    # Generative-Agents-style memory stream: append-only, retrieved via
    # the recency-weighted semantic score. Consolidation and spatial
    # tools removed; locate/recall stay off. Enable with
    # --recency-weight 1.0 on the CLI to activate the Park-et-al.
    # scoring shape.
    "gen_agents_stream": AblationConfig(
        "gen_agents_stream",
        use_consolidation=False,
        use_multi_layer=False,
        available_tools=[
            "semantic_search",
            "temporal_query",
            "get_current_context",
        ],
    ),
    # Same pipeline with hybrid retrieval (BM25 + HNSW) turned off,
    # so the retrieval path falls back to HNSW-only. Comparing
    # ``full`` vs ``no_hybrid`` isolates the BM25 contribution.
    "no_hybrid": AblationConfig(
        "no_hybrid",
        mem_config_overrides={"use_hybrid_retrieval": False},
    ),
    # Same pipeline with the entity-merge context-similarity gate
    # disabled: two observations with the same entity name and within
    # spatial radius collapse into a single entity regardless of how
    # different their surrounding text is. Comparing ``full`` vs
    # ``no_context_merge`` isolates the contribution of context-aware
    # merging on any benchmark that touches entity_query /
    # semantic_search.
    "no_context_merge": AblationConfig(
        "no_context_merge",
        mem_config_overrides={"entity_text_similarity_threshold": 0.0},
    ),
}
