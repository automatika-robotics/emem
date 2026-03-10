import statistics
from dataclasses import dataclass, field
from typing import Any

from harness.benchmarks.scenarios import BenchmarkQuery


@dataclass
class QueryResult:
    """Result of running a single benchmark query."""

    query: BenchmarkQuery
    tools_used: list[str]
    answer: str
    latency_s: float
    correct_tool: bool = False
    answer_relevant: bool = False


@dataclass
class MetricsReport:
    """Aggregated metrics from a harness run."""

    tool_selection_accuracy: float = 0.0
    answer_relevance_rate: float = 0.0
    query_latency_p50: float = 0.0
    query_latency_p95: float = 0.0
    ingestion_throughput: float = 0.0
    vlm_latency_avg: float = 0.0
    total_observations: int = 0
    total_queries: int = 0
    per_query: list[dict[str, Any]] = field(default_factory=list)


def compute_metrics(
    query_results: list[QueryResult],
    ingestion_time_s: float = 0.0,
    total_observations: int = 0,
    vlm_latencies: list[float] | None = None,
) -> MetricsReport:
    """Compute aggregate metrics from individual query results.

    :param query_results: Per-query results from the evaluation phase.
    :param ingestion_time_s: Total wall-clock time for the ingestion phase.
    :param total_observations: Number of observations + body states ingested.
    :param vlm_latencies: Per-frame VLM inference times in seconds.
    :returns: Aggregated metrics report.
    """
    report = MetricsReport(
        total_queries=len(query_results),
        total_observations=total_observations,
    )

    if ingestion_time_s > 0:
        report.ingestion_throughput = total_observations / ingestion_time_s
    if vlm_latencies:
        report.vlm_latency_avg = statistics.mean(vlm_latencies)

    if not query_results:
        return report

    # Score each query
    for qr in query_results:
        qr.correct_tool = qr.query.expected_tool in qr.tools_used
        if qr.query.expected_substrings:
            lower = qr.answer.lower()
            qr.answer_relevant = all(s.lower() in lower for s in qr.query.expected_substrings)
        else:
            qr.answer_relevant = len(qr.answer.strip()) > 0

    report.tool_selection_accuracy = sum(qr.correct_tool for qr in query_results) / len(query_results)
    report.answer_relevance_rate = sum(qr.answer_relevant for qr in query_results) / len(query_results)

    latencies = sorted(qr.latency_s for qr in query_results)
    report.query_latency_p50 = _percentile(latencies, 50)
    report.query_latency_p95 = _percentile(latencies, 95)

    report.per_query = [
        {
            "query": qr.query.query,
            "expected_tool": qr.query.expected_tool,
            "tools_used": qr.tools_used,
            "correct_tool": qr.correct_tool,
            "answer_relevant": qr.answer_relevant,
            "latency_s": round(qr.latency_s, 3),
        }
        for qr in query_results
    ]

    return report


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
