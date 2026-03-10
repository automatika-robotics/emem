from dataclasses import dataclass, field


@dataclass
class BenchmarkQuery:
    """A single evaluation query.

    :param query: Natural language question.
    :param expected_tool: Tool name the agent should select.
    :param expected_substrings: Substrings the answer should contain.
    :param description: Human-readable description of what this tests.
    """

    query: str
    expected_tool: str
    expected_substrings: list[str] = field(default_factory=list)
    description: str = ""


STANDARD_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        "What places have I visited?",
        "semantic_search",
        description="Broad semantic retrieval over place descriptions",
    ),
    BenchmarkQuery(
        "What is near position (3, 3)?",
        "spatial_query",
        description="Spatial retrieval at specific coordinates",
    ),
    BenchmarkQuery(
        "What did I see recently?",
        "temporal_query",
        description="Recent temporal retrieval",
    ),
    BenchmarkQuery(
        "What's my battery level?",
        "body_status",
        expected_substrings=["battery"],
        description="Interoception body state query",
    ),
    BenchmarkQuery(
        "Summarize my exploration",
        "episode_summary",
        description="Episode-level summary retrieval",
    ),
    BenchmarkQuery(
        "Where is the door?",
        "locate",
        description="Concept-to-location resolution",
    ),
    BenchmarkQuery(
        "Tell me everything about the hallway",
        "recall",
        description="Compound multi-layer retrieval",
    ),
    BenchmarkQuery(
        "What's the current situation?",
        "get_current_context",
        description="Nearby context including body state",
    ),
]
