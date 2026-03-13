import logging
import random
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from harness.benchmarks.metrics import MetricsReport, QueryResult, compute_metrics
from harness.benchmarks.scenarios import STANDARD_QUERIES, BenchmarkQuery

log = logging.getLogger("harness")


@dataclass
class HarnessReport:
    """Full report from a harness run."""

    metrics: MetricsReport
    n_steps: int = 0
    n_episodes: int = 0
    n_observations: int = 0
    n_body_states: int = 0
    ingestion_time_s: float = 0.0
    query_results: list[dict[str, Any]] = field(default_factory=list)


class HarnessRunner:
    """Ties together environment, VLM, memory, and agent for benchmarking.

    :param env_name: MiniGrid environment name or AI2-THOR scene (e.g. ``"FloorPlan1"``).
    :param vlm_model: Model name for VLM inference.
    :param llm_model: Model name for LLM (consolidation + agent).
    :param embed_model: Model name for embeddings.
    :param provider: Backend provider (``"ollama"`` or ``"gemini"``).
    :param ollama_url: Ollama server URL (only used when ``provider="ollama"``).
    :param gemini_api_key: Gemini API key (only used when ``provider="gemini"``).
    :param n_steps: Number of environment steps.
    :param vlm_every_n: Run VLM inference every N steps.
    :param queries: Evaluation queries (defaults to :data:`STANDARD_QUERIES`).
    :param db_path: Memory DB path (uses a temp file if ``None``).
    """

    def __init__(
        self,
        env_name: str = "MiniGrid-MultiRoom-N6-v0",
        vlm_model: str = "qwen3.5:latest",
        llm_model: str = "qwen3.5:latest",
        embed_model: str = "nomic-embed-text-v2-moe:latest",
        provider: str = "ollama",
        ollama_url: str = "http://localhost:11434",
        gemini_api_key: str | None = None,
        n_steps: int = 100,
        vlm_every_n: int = 5,
        queries: list[BenchmarkQuery] | None = None,
        db_path: str | None = None,
        headless: bool = False,
        resolution: int = 300,
        exploration_mode: str = "random",
    ):
        self._env_name = env_name
        self._vlm_model = vlm_model
        self._llm_model = llm_model
        self._embed_model = embed_model
        self._provider = provider
        self._ollama_url = ollama_url
        self._gemini_api_key = gemini_api_key
        self._n_steps = n_steps
        self._vlm_every_n = vlm_every_n
        self._queries = queries or STANDARD_QUERIES
        self._db_path = db_path
        self._headless = headless
        self._resolution = resolution
        self._exploration_mode = exploration_mode

    def run(self) -> HarnessReport:
        """Execute the full harness: ingestion then evaluation.

        :returns: Report with metrics and per-query results.
        """
        from emem import SpatioTemporalMemory

        from harness.environments.interoception import SyntheticInteroception

        log.info("Initializing %s providers...", self._provider)
        embedder, llm, vlm, agent_kwargs = self._make_providers()
        log.info("Providers ready (embed_dim=%d)", embedder.dim)

        log.info("Creating environment: %s", self._env_name)
        env = self._make_env()
        intero = SyntheticInteroception()

        db_path = self._db_path or tempfile.mktemp(suffix=".db")
        mem = SpatioTemporalMemory(
            db_path=db_path,
            embedding_provider=embedder,
            llm_client=llm,
        )
        log.info("Memory initialised at %s", db_path)

        # -- Ingestion --
        n_obs = 0
        n_body = 0
        n_ep = 0
        vlm_latencies: list[float] = []

        log.info("=== Ingestion: %d steps, VLM every %d ===", self._n_steps, self._vlm_every_n)
        t_ingest = time.monotonic()

        mem.start_episode("exploration")
        n_ep += 1
        frame, pos = env.reset()

        for step in range(self._n_steps):
            action = random.choice(env.available_actions())
            frame, pos, reward, done, info = env.step(action)

            if step % self._vlm_every_n == 0:
                log.info("[step %d/%d] VLM at pos=(%.1f, %.1f)...", step, self._n_steps, pos[0], pos[1])

                t0 = time.monotonic()
                description = vlm.describe(
                    frame,
                    "Describe what you see in this scene in 1-2 sentences.",
                )
                dt = time.monotonic() - t0
                vlm_latencies.append(dt)
                log.info("  description (%.1fs): %s", dt, description[:80])

                t0 = time.monotonic()
                place = vlm.describe(
                    frame,
                    "What type of place or room is this? Answer in one word only.",
                    think=False,
                )
                dt = time.monotonic() - t0
                vlm_latencies.append(dt)
                log.info("  place (%.1fs): %s", dt, place[:80])

                if description.strip():
                    mem.add(description, x=float(pos[0]), y=float(pos[1]), layer_name="description")
                    n_obs += 1
                if place.strip():
                    mem.add(place, x=float(pos[0]), y=float(pos[1]), layer_name="place")
                    n_obs += 1

            body = intero.step()
            for layer, text in body.items():
                mem.add_body_state(text, layer_name=layer)
                n_body += 1

            if done:
                mem.end_episode()
                mem.start_episode("exploration")
                n_ep += 1
                frame, pos = env.reset()
                intero.reset()
                log.info("[step %d/%d] Episode done, starting #%d", step, self._n_steps, n_ep)

        mem.end_episode()
        ingestion_time = time.monotonic() - t_ingest
        log.info("Ingestion complete: %d obs + %d body in %.1fs", n_obs, n_body, ingestion_time)

        # -- Evaluation --
        log.info("=== Evaluation: %d queries ===", len(self._queries))
        agent = self._make_agent(mem, agent_kwargs)
        query_results: list[QueryResult] = []

        for i, bq in enumerate(self._queries, 1):
            log.info("[query %d/%d] %s", i, len(self._queries), bq.query)
            t0 = time.monotonic()
            result = agent.run(bq.query)
            latency = time.monotonic() - t0
            log.info("  tools=%s (%.1fs): %s", result.tools_used, latency, (result.answer or "")[:80])

            query_results.append(QueryResult(
                query=bq, tools_used=result.tools_used,
                answer=result.answer, latency_s=latency,
            ))

        metrics = compute_metrics(
            query_results,
            ingestion_time_s=ingestion_time,
            total_observations=n_obs + n_body,
            vlm_latencies=vlm_latencies,
        )

        env.close()
        mem.close()

        return HarnessReport(
            metrics=metrics, n_steps=self._n_steps, n_episodes=n_ep,
            n_observations=n_obs, n_body_states=n_body,
            ingestion_time_s=ingestion_time, query_results=metrics.per_query,
        )

    def _make_env(self) -> Any:
        """Create the environment adapter based on env_name."""
        if self._env_name.startswith("FloorPlan"):
            from harness.environments.ai2thor_adapter import AI2ThorAdapter

            return AI2ThorAdapter(
                scene=self._env_name,
                headless=self._headless,
                width=self._resolution,
                height=self._resolution,
                exploration_mode=self._exploration_mode,
            )

        from harness.environments.minigrid_adapter import MiniGridAdapter

        return MiniGridAdapter(self._env_name)

    def _make_providers(self) -> tuple:
        """Create ``(embedder, llm, vlm, agent_kwargs)``."""
        if self._provider == "ollama":
            from harness.providers.ollama_embeddings import OllamaEmbeddingProvider
            from harness.providers.ollama_llm import OllamaLLMClient
            from harness.providers.ollama_vlm import OllamaVLM

            return (
                OllamaEmbeddingProvider(self._embed_model, self._ollama_url),
                OllamaLLMClient(self._llm_model, self._ollama_url),
                OllamaVLM(self._vlm_model, self._ollama_url),
                {"model": self._llm_model, "base_url": self._ollama_url},
            )

        if self._provider == "gemini":
            from harness.providers.gemini_embeddings import GeminiEmbeddingProvider
            from harness.providers.gemini_llm import GeminiLLMClient
            from harness.providers.gemini_vlm import GeminiVLM

            key = self._gemini_api_key
            return (
                GeminiEmbeddingProvider(model=self._embed_model, api_key=key),
                GeminiLLMClient(model=self._llm_model, api_key=key),
                GeminiVLM(model=self._vlm_model, api_key=key),
                {"model": self._llm_model, "api_key": key},
            )

        raise ValueError(f"Unknown provider: {self._provider!r}. Use 'ollama' or 'gemini'.")

    def _make_agent(self, mem: Any, kwargs: dict) -> Any:
        if self._provider == "gemini":
            from harness.agent.react_agent import GeminiReactAgent
            return GeminiReactAgent(mem, **kwargs)
        from harness.agent.react_agent import ReactAgent
        return ReactAgent(mem, **kwargs)
