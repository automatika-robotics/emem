#!/usr/bin/env python3
"""Simulated robot scenario with LLM tool-calling pattern.

Demonstrates how an LLM agent would use the memory tools in a ReAct-style loop.
"""

import json
import tempfile
from pathlib import Path

from emem import SpatioTemporalMemory


class MockLLMAgent:
    """Simulates an LLM agent that uses memory tools."""

    def __init__(self, mem: SpatioTemporalMemory):
        self.mem = mem

    def think_and_act(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"User: {user_query}")
        print(f"{'='*60}")

        tool_calls = self._select_tools(user_query)
        results = []
        for tool_name, args in tool_calls:
            print(f"\n[Agent] {tool_name}({json.dumps(args, default=str)})")
            result = self.mem.dispatch_tool_call(tool_name, args)
            print(f"[Result]\n{result}")
            results.append((tool_name, result))

        response = self._synthesize(results)
        print(f"\n[Agent Response] {response}")
        return response

    def _select_tools(self, query: str):
        q = query.lower()
        calls = []
        if "where" in q or "near" in q:
            calls.append(("get_current_context", {"radius": 5.0}))
        if "remember" in q or "what" in q or "find" in q:
            calls.append(("semantic_search", {"query": query, "n_results": 3}))
        if "summary" in q or "episode" in q or "did" in q:
            calls.append(("episode_summary", {"last_n": 3}))
        if "history" in q or "long-term" in q:
            calls.append(("search_gists", {"query": query}))
        return calls or [("get_current_context", {})]

    def _synthesize(self, results):
        found = [name for name, r in results if "No " not in r and len(r) > 10]
        if found:
            return f"Found info from: {', '.join(found)}"
        return "No relevant information found."


def main():
    tmp = Path(tempfile.mkdtemp())

    with SpatioTemporalMemory(db_path=str(tmp / "agent.db")) as mem:

        # ── Office patrol ─────────────────────────────────────────
        mem.start_episode("office_patrol")
        for text, x, y in [
            ("Person sitting at desk typing on laptop", 5.0, 5.0),
            ("Empty conference room with whiteboard", 8.0, 5.0),
            ("Printer making loud noise, paper jam warning", 10.0, 3.0),
            ("Water cooler in break area", 12.0, 7.0),
            ("Fire extinguisher on wall near exit", 14.0, 5.0),
            ("Broken ceiling light flickering", 8.0, 8.0),
        ]:
            mem.add(text, x=x, y=y, layer_name="vlm", source_type="vlm")
        mem.end_episode()  # auto-consolidates

        # ── Warehouse check ───────────────────────────────────────
        mem.start_episode("warehouse_check")
        for text, x, y in [
            ("Forklift parked near loading dock", 50.0, 20.0),
            ("Stack of boxes labeled 'fragile'", 52.0, 22.0),
            ("Wet floor sign near entrance", 48.0, 18.0),
            ("Temperature display reading 18C", 55.0, 25.0),
        ]:
            mem.add(text, x=x, y=y, layer_name="vlm", source_type="vlm")
        mem.end_episode()

        print(f"\nTotal observations: {mem.store.count_observations()}")

        # ── LLM agent queries ─────────────────────────────────────
        agent = MockLLMAgent(mem)
        agent.think_and_act("Where am I and what's nearby?")
        agent.think_and_act("What did I find during my patrols?")
        agent.think_and_act("Do you remember anything about a printer?")
        agent.think_and_act("Search long-term history for office observations")

    print(f"\nDone. Storage: {tmp}")


if __name__ == "__main__":
    main()
