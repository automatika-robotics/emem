from __future__ import annotations

import argparse
import json
import logging

_DEFAULTS = {
    "ollama": {
        "vlm": "qwen3.5:latest",
        "llm": "qwen3.5:latest",
        "embed": "nomic-embed-text-v2-moe:latest",
    },
    "gemini": {
        "vlm": "gemini-2.0-flash-lite",
        "llm": "gemini-2.0-flash-lite",
        "embed": "gemini-embedding-001",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="eMEM Testing Harness — simulate EMOS perception pipeline"
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of environment steps"
    )
    parser.add_argument(
        "--vlm-every", type=int, default=5, help="Run VLM every N steps"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-MultiRoom-N6-v0",
        help="Environment name: MiniGrid env (e.g. MiniGrid-MultiRoom-N6-v0) or AI2-THOR scene (e.g. FloorPlan1)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use CloudRendering for AI2-THOR (no display)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=300,
        help="Frame resolution for AI2-THOR (default: 300)",
    )
    parser.add_argument(
        "--explore",
        type=str,
        default="random",
        choices=["random", "teleport"],
        help="Exploration mode for AI2-THOR: random actions or teleport tour (default: random)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "gemini"],
        help="LLM/VLM/embedding provider backend",
    )
    parser.add_argument(
        "--vlm-model", type=str, default=None, help="VLM model (default: per provider)"
    )
    parser.add_argument(
        "--llm-model", type=str, default=None, help="LLM model (default: per provider)"
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help="Embedding model (default: per provider)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY)",
    )
    parser.add_argument("--db-path", type=str, default=None, help="Memory DB path")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    defaults = _DEFAULTS[args.provider]
    vlm_model = args.vlm_model or defaults["vlm"]
    llm_model = args.llm_model or defaults["llm"]
    embed_model = args.embed_model or defaults["embed"]

    from harness.benchmarks.runner import HarnessRunner

    runner = HarnessRunner(
        env_name=args.env,
        vlm_model=vlm_model,
        llm_model=llm_model,
        embed_model=embed_model,
        provider=args.provider,
        ollama_url=args.ollama_url,
        gemini_api_key=args.gemini_api_key,
        n_steps=args.steps,
        vlm_every_n=args.vlm_every,
        db_path=args.db_path,
        headless=args.headless,
        resolution=args.resolution,
        exploration_mode=args.explore,
    )

    report = runner.run()

    if args.json:
        out = {
            "provider": args.provider,
            "n_steps": report.n_steps,
            "n_episodes": report.n_episodes,
            "n_observations": report.n_observations,
            "n_body_states": report.n_body_states,
            "ingestion_time_s": round(report.ingestion_time_s, 2),
            "tool_selection_accuracy": round(report.metrics.tool_selection_accuracy, 3),
            "answer_relevance_rate": round(report.metrics.answer_relevance_rate, 3),
            "query_latency_p50": round(report.metrics.query_latency_p50, 3),
            "query_latency_p95": round(report.metrics.query_latency_p95, 3),
            "ingestion_throughput": round(report.metrics.ingestion_throughput, 1),
            "vlm_latency_avg": round(report.metrics.vlm_latency_avg, 3),
            "per_query": report.query_results,
        }
        print(json.dumps(out, indent=2))
    else:
        m = report.metrics
        print()
        print("=== Harness Report ===")
        print(f"Steps: {report.n_steps}")
        print(f"Episodes: {report.n_episodes}")
        print(f"VLM observations: {report.n_observations}")
        print(f"Body states: {report.n_body_states}")
        print(f"Ingestion time: {report.ingestion_time_s:.1f}s")
        print(f"Ingestion throughput: {m.ingestion_throughput:.1f} obs/s")
        print(f"VLM latency (avg): {m.vlm_latency_avg:.2f}s")
        print()
        print("=== Query Evaluation ===")
        print(f"Tool selection accuracy: {m.tool_selection_accuracy:.1%}")
        print(f"Answer relevance rate: {m.answer_relevance_rate:.1%}")
        print(f"Query latency p50: {m.query_latency_p50:.2f}s")
        print(f"Query latency p95: {m.query_latency_p95:.2f}s")
        print()
        for qr in report.query_results:
            status = "OK" if qr["correct_tool"] else "MISS"
            print(
                f"  [{status}] {qr['query']}"
                f" (expected={qr['expected_tool']}, got={qr['tools_used']}, {qr['latency_s']:.1f}s)"
            )
            answer = qr.get("answer", "")
            if answer:
                # Indent and wrap the answer for readability
                for line in answer.split("\n"):
                    print(f"        {line}")
                print()


if __name__ == "__main__":
    main()
