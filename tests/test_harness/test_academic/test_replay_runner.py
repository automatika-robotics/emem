import numpy as np

from harness.benchmarks.academic.ablation import ABLATIONS
from harness.benchmarks.academic.replay_runner import BenchmarkRunner, _AblatedMemory
from harness.benchmarks.academic.scorers.exact_match import ExactMatchScorer
from harness.benchmarks.academic.trajectory import (
    BenchmarkQuestion,
    BenchmarkSample,
    TrajectoryFrame,
)
from harness.agent.react_agent import AgentResult


class FakeEmbedder:
    def __init__(self, dim=32):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def embed(self, texts):
        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.RandomState(hash(text) % (2**31))
            result[i] = rng.randn(self._dim).astype(np.float32)
            result[i] /= np.linalg.norm(result[i]) + 1e-8
        return result


class FakeAgent:
    """Agent that always returns a fixed answer."""

    def __init__(self, mem, answer="chair"):
        self._answer = answer

    def run(self, query):
        return AgentResult(query=query, answer=self._answer)


class SyntheticLoader:
    """Yields synthetic BenchmarkSamples for testing."""

    def __init__(self, n_samples=2, n_frames=5, n_questions=2):
        self._n_samples = n_samples
        self._n_frames = n_frames
        self._n_questions = n_questions

    @property
    def name(self):
        return "synthetic"

    def load(self):
        for s in range(self._n_samples):
            frames = [
                TrajectoryFrame(
                    frame_id=f"f{i}",
                    position=(float(i), float(i), 0.0),
                    timestamp=float(i * 10),
                    text=f"I see a {'chair' if i % 2 == 0 else 'table'} at position {i}",
                    layer_name="description",
                )
                for i in range(self._n_frames)
            ]
            questions = [
                BenchmarkQuestion(
                    question_id=f"q{s}_{j}",
                    question="What furniture do you see?",
                    answer="chair",
                )
                for j in range(self._n_questions)
            ]
            yield BenchmarkSample(
                sample_id=f"sample_{s}",
                scene_id=f"scene_{s}",
                trajectory=frames,
                questions=questions,
            )


class TestBenchmarkRunnerSynthetic:
    def test_end_to_end(self):
        runner = BenchmarkRunner(
            loader=SyntheticLoader(n_samples=2, n_frames=3, n_questions=2),
            scorer=ExactMatchScorer(),
            ablation=ABLATIONS["full"],
            embedding_provider=FakeEmbedder(),
            agent_factory=lambda mem: FakeAgent(mem, answer="chair"),
            max_samples=2,
        )
        report = runner.run()

        assert report.dataset == "synthetic"
        assert report.ablation == "full"
        assert len(report.sample_results) == 2
        assert report.mean_score() == 100.0

    def test_wrong_answer_scores_zero(self):
        runner = BenchmarkRunner(
            loader=SyntheticLoader(n_samples=1, n_frames=3, n_questions=1),
            scorer=ExactMatchScorer(),
            ablation=ABLATIONS["full"],
            embedding_provider=FakeEmbedder(),
            agent_factory=lambda mem: FakeAgent(mem, answer="table"),
        )
        report = runner.run()
        assert report.mean_score() == 0.0

    def test_max_samples_limits_output(self):
        runner = BenchmarkRunner(
            loader=SyntheticLoader(n_samples=10),
            scorer=ExactMatchScorer(),
            ablation=ABLATIONS["full"],
            embedding_provider=FakeEmbedder(),
            agent_factory=lambda mem: FakeAgent(mem),
            max_samples=3,
        )
        report = runner.run()
        assert len(report.sample_results) == 3

    def test_report_summary(self):
        runner = BenchmarkRunner(
            loader=SyntheticLoader(n_samples=1, n_frames=2, n_questions=1),
            scorer=ExactMatchScorer(),
            ablation=ABLATIONS["full"],
            embedding_provider=FakeEmbedder(),
            agent_factory=lambda mem: FakeAgent(mem),
        )
        report = runner.run()
        summary = report.summary()
        assert summary["dataset"] == "synthetic"
        assert summary["n_questions"] == 1
        assert "score" in summary["metrics"]

    def test_ingestion_records_observations(self):
        runner = BenchmarkRunner(
            loader=SyntheticLoader(n_samples=1, n_frames=5, n_questions=1),
            scorer=ExactMatchScorer(),
            ablation=ABLATIONS["full"],
            embedding_provider=FakeEmbedder(),
            agent_factory=lambda mem: FakeAgent(mem),
        )
        report = runner.run()
        assert report.sample_results[0].n_observations == 5

    def test_agent_position_injected(self):
        frames = [
            TrajectoryFrame(
                frame_id="f0", position=(1.0, 2.0, 0.0),
                timestamp=0.0, text="A chair",
            )
        ]
        questions = [
            BenchmarkQuestion(
                question_id="q0", question="What do you see?", answer="chair"
            )
        ]
        sample = BenchmarkSample(
            sample_id="s0", scene_id="sc0",
            trajectory=frames, questions=questions,
            agent_position=(3.0, 4.0, 0.0),
            agent_situation="Standing near the door",
        )

        class SingleLoader:
            name = "test"
            def load(self):
                yield sample

        runner = BenchmarkRunner(
            loader=SingleLoader(),
            scorer=ExactMatchScorer(),
            ablation=ABLATIONS["full"],
            embedding_provider=FakeEmbedder(),
            agent_factory=lambda mem: FakeAgent(mem),
        )
        report = runner.run()
        assert report.sample_results[0].n_observations == 1


class TestAblationConfigs:
    def test_all_ablations_defined(self):
        expected = {"full", "vector_only", "no_consolidation", "no_spatial", "flat_layer"}
        assert set(ABLATIONS.keys()) == expected

    def test_vector_only_has_one_tool(self):
        abl = ABLATIONS["vector_only"]
        assert abl.available_tools == ["semantic_search"]

    def test_no_spatial_excludes_spatial_tools(self):
        abl = ABLATIONS["no_spatial"]
        assert "locate" not in abl.available_tools
        assert "recall" not in abl.available_tools
        assert "spatial_query" not in abl.available_tools
        assert "semantic_search" in abl.available_tools
        assert "temporal_query" in abl.available_tools

    def test_filter_tool_definitions(self):
        abl = ABLATIONS["vector_only"]
        tool_defs = [
            {"name": "semantic_search", "description": "search"},
            {"name": "locate", "description": "find location"},
            {"name": "recall", "description": "recall"},
        ]
        filtered = abl.filter_tool_definitions(tool_defs)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "semantic_search"

    def test_flat_layer_has_all_tools(self):
        abl = ABLATIONS["flat_layer"]
        assert len(abl.available_tools) == 10
        assert not abl.use_multi_layer

    def test_no_consolidation_config(self):
        abl = ABLATIONS["no_consolidation"]
        assert not abl.use_consolidation
        assert len(abl.available_tools) == 10


class TestAblatedMemory:
    def test_tool_restriction(self):
        class MockMem:
            def get_tool_definitions(self):
                return [
                    {"type": "function", "function": {"name": "semantic_search", "description": "s"}},
                    {"type": "function", "function": {"name": "locate", "description": "l"}},
                ]
            def dispatch_tool_call(self, name, args):
                return f"called {name}"

        abl = ABLATIONS["vector_only"]
        wrapped = _AblatedMemory(MockMem(), abl)

        defs = wrapped.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "semantic_search"

        assert "not available" in wrapped.dispatch_tool_call("locate", {})
        assert wrapped.dispatch_tool_call("semantic_search", {}) == "called semantic_search"
