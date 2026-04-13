#!/usr/bin/env python3
"""Basic standalone usage of eMEM.

Demonstrates the high-level SpatioTemporalMemory API:
- Adding observations
- Episode lifecycle (auto-consolidation on end)
- Querying with the 6 built-in tools
"""

import tempfile
from pathlib import Path

from emem import SpatioTemporalMemory


def main():
    tmp = Path(tempfile.mkdtemp())
    print(f"Storage: {tmp}")

    with SpatioTemporalMemory(db_path=str(tmp / "demo.db")) as mem:
        # ── Episode 1: Kitchen patrol ─────────────────────────────
        mem.start_episode("kitchen_patrol")
        observations = [
            ("Red chair near table", 10.0, 10.0),
            ("Wooden table with plates", 10.5, 10.2),
            ("Microwave on counter", 11.0, 9.5),
            ("Coffee machine next to sink", 11.5, 9.0),
            ("Cat sleeping on chair", 10.2, 10.1),
            ("Open refrigerator", 12.0, 10.0),
            ("Dirty dishes in sink", 11.8, 9.2),
        ]
        for text, x, y in observations:
            mem.add(text, x=x, y=y, layer_name="vlm")
        mem.end_episode()  # auto-flushes, generates gist, archives observations

        # ── Episode 2: Hallway patrol ─────────────────────────────
        mem.start_episode("hallway_patrol")
        for i in range(5):
            x = 25.0 + i * 2
            mem.add(
                f"Hallway: {'door' if i % 2 == 0 else 'painting'}",
                x=x,
                y=10.0,
                layer_name="vlm",
            )
        mem.end_episode()

        # ── Queries ───────────────────────────────────────────────

        print("\n=== Spatial Query (near kitchen) ===")
        print(mem.spatial_query(x=10.0, y=10.0, radius=3.0))

        print("\n=== Episode Summaries ===")
        print(mem.episode_summary(last_n=2))

        print("\n=== Search Gists (long-term memory) ===")
        print(mem.search_gists(query="kitchen"))

        print("\n=== Current Context ===")
        print(mem.get_current_context(radius=5.0))

        # ── Tool dispatch (as LLM would call) ─────────────────────
        print("\n=== Tool Dispatch ===")
        result = mem.dispatch_tool_call("episode_summary", {"task_name": "hallway"})
        print(result)

        print("\n=== Available Tools ===")
        for td in mem.get_tool_definitions():
            print(f"  - {td['function']['name']}: {td['function']['description']}")

    print(f"\nDone. Files in: {tmp}")


if __name__ == "__main__":
    main()
