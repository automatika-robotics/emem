"""End-to-end runner tests with FakeEmbedder (no Ollama needed)."""

import pytest
import numpy as np

from harness.benchmarks.metrics import QueryResult, compute_metrics
from harness.benchmarks.scenarios import STANDARD_QUERIES, BenchmarkQuery


class TestMetrics:
    def test_compute_metrics_empty(self):
        report = compute_metrics([])
        assert report.total_queries == 0
        assert report.tool_selection_accuracy == 0.0

    def test_compute_metrics_perfect(self):
        results = [
            QueryResult(
                query=BenchmarkQuery("test", "semantic_search"),
                tools_used=["semantic_search"],
                answer="found stuff",
                latency_s=0.5,
            ),
            QueryResult(
                query=BenchmarkQuery("test2", "body_status", expected_substrings=["battery"]),
                tools_used=["body_status"],
                answer="battery: 85%",
                latency_s=0.3,
            ),
        ]
        report = compute_metrics(results)
        assert report.tool_selection_accuracy == 1.0
        assert report.answer_relevance_rate == 1.0

    def test_compute_metrics_partial(self):
        results = [
            QueryResult(
                query=BenchmarkQuery("test", "semantic_search"),
                tools_used=["spatial_query"],  # wrong tool
                answer="stuff",
                latency_s=0.5,
            ),
            QueryResult(
                query=BenchmarkQuery("test2", "body_status"),
                tools_used=["body_status"],
                answer="ok",
                latency_s=0.3,
            ),
        ]
        report = compute_metrics(results)
        assert report.tool_selection_accuracy == 0.5

    def test_latency_percentiles(self):
        results = [
            QueryResult(
                query=BenchmarkQuery(f"q{i}", "semantic_search"),
                tools_used=["semantic_search"],
                answer="ok",
                latency_s=float(i),
            )
            for i in range(1, 11)
        ]
        report = compute_metrics(results)
        assert report.query_latency_p50 > 0
        assert report.query_latency_p95 > report.query_latency_p50

    def test_ingestion_throughput(self):
        report = compute_metrics([], ingestion_time_s=10.0, total_observations=100)
        assert report.ingestion_throughput == 10.0

    def test_vlm_latency(self):
        report = compute_metrics([], vlm_latencies=[1.0, 2.0, 3.0])
        assert report.vlm_latency_avg == 2.0


class TestScenarios:
    def test_standard_queries_not_empty(self):
        assert len(STANDARD_QUERIES) > 0

    def test_all_queries_have_expected_tool(self):
        valid_tools = {
            "semantic_search", "spatial_query", "temporal_query",
            "episode_summary", "get_current_context", "search_gists",
            "entity_query", "locate", "recall", "body_status",
        }
        for q in STANDARD_QUERIES:
            assert q.expected_tool in valid_tools, f"{q.query} has invalid tool {q.expected_tool}"
