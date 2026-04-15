<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/eMEM_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/eMEM_light.png">
  <img alt="eMEM Logo" src="https://raw.githubusercontent.com/automatika-robotics/emem/main/docs/_static/eMEM_light.png" width="50%">
</picture>

<br/>

Part of the [EMOS](https://github.com/automatika-robotics/emos) ecosystem

[![PyPI version](https://img.shields.io/pypi/v/emem.svg)](https://pypi.org/project/emem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/B9ZU6qjzND)

**Embodied Memory for Situated Agents**

A hybrid graph-based spatio-temporal memory system that gives embodied agents the ability to remember *what* they observed, *where* they observed it, and *when*.

[**EMOS Documentation**](https://emos.automatikarobotics.com) | [**Discord**](https://discord.gg/B9ZU6qjzND)

</div>

<p align="center">
  <a href="#installation">Installation</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#llm-tool-interface">LLM Tools</a> &middot;
  <a href="#api-reference">API</a> &middot;
  <a href="#testing-harness">Harness</a>
</p>

---

## Why eMEM?

eMEM gives Physical AI agents a persistent, queryable sense of place and history, so they don't just react to the current frame, but **remember**, **recall**, and **reason** over everything they've observed.

Mobile robots and embodied AI agents accumulate thousands of observations per session; object detections, scene descriptions, sensor readings, but existing memory systems force a choice: **vector databases** that discard spatial structure, or **metric maps** that discard semantics. eMEM unifies both.

| Capability | eMEM |
|---|---|
| Semantic search (meaning) | HNSW approximate nearest neighbor |
| Spatial queries (where) | R-tree range and nearest-neighbor |
| Temporal queries (when) | SQLite indexed timestamps |
| Episode structure | Graph edges with hierarchical sub-tasks |
| Memory consolidation | Automatic gist generation + tiered archival |
| LLM integration | 10 ready-to-use tool definitions |

Everything runs **fully embedded**, SQLite, hnswlib, and Rtree, with zero external services. A single `.db` file and a `.hnsw.bin` file contain the entire memory state.

**eMEM** is the memory layer of the [EMOS](https://github.com/automatika-robotics/emos) (Embodied Operating System) ecosystem.
## Installation

```bash
pip install emem
```

For semantic search with automatic embedding generation:

```bash
pip install emem[embeddings]  # adds sentence-transformers
```

**Requirements:** Python >= 3.8

## Quick Start

```python
from emem import SpatioTemporalMemory

with SpatioTemporalMemory(db_path="robot_memory.db") as mem:

    # Record observations with spatial coordinates
    mem.start_episode("kitchen_patrol")
    mem.add("Red chair near the dining table", x=10.0, y=10.0)
    mem.add("Cat sleeping on the chair",       x=10.2, y=10.1)
    mem.add("Coffee machine next to the sink", x=11.5, y=9.0)
    mem.end_episode()  # auto-consolidates into a gist, archives raw observations

    # Query by meaning -- searches both recent observations and consolidated summaries
    print(mem.semantic_search("furniture"))

    # Query by location
    print(mem.spatial_query(x=10.0, y=10.0, radius=3.0))

    # Query by time
    print(mem.temporal_query(last_n_minutes=30))

    # Situational awareness: nearby objects + area summaries + recent activity
    print(mem.get_current_context())
```

## Architecture

eMEM is built around three complementary index structures that share a unified node/edge graph:

```
                    +-----------------+
                    | SpatioTemporal  |
                    |     Memory      |  <-- High-level facade
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
     +--------+--+   +------+------+  +----+-------+
     |  Working   |  |   Memory    |  | Consolida- |
     |  Memory    |  |   Tools     |  | tion       |
     |  (buffer)  |  | (10 tools)  |  | Engine     |
     +--------+---+  +------+------+  +----+-------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------+--------+
                    |   MemoryStore   |
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
        +-----+----+  +-----+-----+  +----+-----+
        |  SQLite   |  |  hnswlib  |  |  R-tree  |
        | (nodes,   |  | (vector   |  | (spatial |
        |  edges,   |  |  search)  |  |  index)  |
        |  tiers)   |  |           |  |          |
        +-----------+  +-----------+  +----------+
```

### Memory Graph

The memory is structured as a typed graph with four node types and six edge types:

**Nodes:**
- **ObservationNode** -- A single perception event: text, coordinates, timestamp, layer, confidence
- **EpisodeNode** -- A task or activity span grouping related observations
- **GistNode** -- A consolidated summary of multiple observations, with spatial extent and time range
- **EntityNode** -- A persistent tracked object/landmark with auto-merge by semantic similarity + spatial proximity

**Edges:**
- `BELONGS_TO` -- Observation &rarr; Episode
- `FOLLOWS` -- Episode &rarr; Episode (temporal sequence)
- `SUBTASK_OF` -- Episode &rarr; Episode (hierarchical nesting)
- `SUMMARIZES` -- Gist &rarr; Observation(s)
- `OBSERVED_IN` -- Entity &rarr; Observation (entity was seen in this observation)
- `COOCCURS_WITH` -- Entity &harr; Entity (entities observed together)

### Memory Tiers

Observations flow through four tiers, inspired by human memory consolidation:

```
 working  -->  short_term  -->  long_term  -->  archived
 (buffer)     (in store)      (promoted)      (text dropped,
                                               gist remains)
```

When an episode ends, its observations are automatically consolidated into a **GistNode** that preserves the semantic content, spatial center, and time span. The raw observations are archived (text and embeddings removed) to bound storage growth while keeping the gist searchable.

### Unified Search

`semantic_search` transparently queries **both** the observation and gist indexes through a single HNSW lookup. After consolidation, knowledge is not lost -- it is compressed into gists that remain fully searchable alongside recent observations.

## LLM Tool Interface

eMEM provides 10 tools designed for LLM function-calling. Each returns a token-efficient formatted string.

| Tool | Description |
|---|---|
| `semantic_search` | Search by meaning across observations and consolidated summaries |
| `spatial_query` | Find observations within a radius of a point |
| `temporal_query` | Find observations in a time range, chronologically ordered |
| `episode_summary` | Get summary of one or more episodes |
| `get_current_context` | Situational awareness: nearby objects, area summaries, recent activity |
| `search_gists` | Search only consolidated long-term memory |
| `entity_query` | Find known entities (objects, people, landmarks) by name, type, or location |
| `locate` | Resolve a concept to a spatial position (returns centroid + radius) |
| `recall` | Recall everything known about a concept (cross-layer observations, gists, entities) |
| `body_status` | Get latest body/internal state readings (battery, temperature, joint health) |

### Using with an LLM Agent

```python
from emem import SpatioTemporalMemory

mem = SpatioTemporalMemory(db_path="agent.db")

# Get tool definitions (OpenAI function-calling format)
tools = mem.get_tool_definitions()

# Dispatch a tool call from the LLM
result = mem.dispatch_tool_call("semantic_search", {"query": "red chair"})

# Or call tools directly
result = mem.semantic_search("red chair", n_results=5, layer="vlm")
```

### Relative Time Queries

Time parameters accept human-readable relative strings:

```python
mem.temporal_query(time_after="-10m")    # last 10 minutes
mem.temporal_query(time_after="-2h")     # last 2 hours
mem.semantic_search("door", time_after="-1d")  # last 24 hours
```

## API Reference

### `SpatioTemporalMemory`

The primary interface. Manages ingestion, episodes, consolidation, and queries through a single object.

```python
SpatioTemporalMemory(
    db_path="memory.db",           # SQLite database path
    config=None,                   # SpatioTemporalMemoryConfig (optional)
    embedding_provider=None,       # EmbeddingProvider (optional, enables semantic search)
    llm_client=None,               # LLMClient for gist generation (optional)
    get_current_time=None,         # Custom clock (optional)
)
```

**Ingestion:**
```python
mem.add(text, x, y, z=0.0, layer_name="default", source_type="manual",
        confidence=1.0, metadata=None, embedding=None) -> str  # returns observation ID
```

**Episodes:**
```python
mem.start_episode(name, metadata=None, parent_episode_id=None) -> str
mem.end_episode(consolidate=True) -> Optional[str]
```

**Queries** (all return formatted strings):
```python
mem.semantic_search(query, n_results=5, layer=None, ...)
mem.spatial_query(x, y, radius=2.0, layer=None, ...)
mem.temporal_query(time_after=None, last_n_minutes=None, ...)
mem.episode_summary(episode_id=None, task_name=None, last_n=1)
mem.get_current_context(radius=3.0, include_recent_minutes=5.0)
mem.search_gists(query, n_results=5, ...)
mem.entity_query(name=None, entity_type=None, near_x=None, ...)
mem.locate(concept, n_results=10, ...)
mem.recall(query, n_results=10, ...)
mem.body_status(layers=None)
mem.add_body_state(text, layer_name, x=0.0, y=0.0, z=0.0, timestamp=None, confidence=1.0, metadata=None)
```

### Embedding Providers

eMEM uses a protocol-based embedding interface. Bring your own or use the built-in providers:

```python
from emem.embeddings import SentenceTransformerProvider

mem = SpatioTemporalMemory(
    db_path="memory.db",
    embedding_provider=SentenceTransformerProvider("all-MiniLM-L6-v2"),
)
```

Custom providers implement the `EmbeddingProvider` protocol:

```python
class MyEmbedder:
    @property
    def dim(self) -> int:
        return 768

    def embed(self, texts: list[str]) -> np.ndarray:
        # Return (N, dim) float32 array
        ...
```

### Consolidation

**Automatic:** `end_episode()` consolidates all episode observations into a gist and archives them.

**Manual:** For non-episodic observations that age past the consolidation window:

```python
gist_ids = mem.consolidate_time_window()
```

Time-window consolidation uses DBSCAN to spatially cluster old observations before summarizing each cluster.

### Configuration

```python
from emem import SpatioTemporalMemoryConfig

config = SpatioTemporalMemoryConfig(
    db_path="memory.db",
    hnsw_path="memory_hnsw.bin",
    embedding_dim=384,

    # Working memory
    working_memory_size=50,        # max buffered observations
    flush_interval=2.0,            # seconds between auto-flushes
    flush_batch_size=5,            # flush after N observations

    # Consolidation
    consolidation_window=1800.0,   # 30 min before time-window consolidation
    consolidation_spatial_eps=3.0, # DBSCAN eps (meters)
    consolidation_min_samples=3,   # min cluster size

    # HNSW tuning
    hnsw_ef_construction=200,
    hnsw_m=16,
    hnsw_ef_search=50,
    hnsw_max_elements=100_000,
)
```

## Observation Layers

Observations can be tagged with a `layer_name` to separate different perception modalities:

```python
mem.add("Person detected", x=5.0, y=5.0, layer_name="detections", confidence=0.92)
mem.add("A kitchen with modern appliances", x=5.0, y=5.0, layer_name="vlm")
mem.add("Temperature: 22.5C", x=5.0, y=5.0, layer_name="sensors")

# Query specific layers
mem.semantic_search("person", layer="detections")
mem.spatial_query(x=5.0, y=5.0, radius=3.0, layer="vlm")
```

## Body State / Interoception

eMEM treats internal body state (battery, temperature, joint health) as a first-class memory dimension alongside world observations:

```python
# Record body state — pass the robot's current world-frame position so
# the reading is co-located with world observations at the same pose.
# (x/y/z default to 0 for text-only / non-embodied callers.)
mem.add_body_state("battery: 45%", layer_name="battery", x=robot_x, y=robot_y)
mem.add_body_state("72C across 4 cores", layer_name="cpu_temp", x=robot_x, y=robot_y)
mem.add_body_state("all joints nominal", layer_name="joint_health", x=robot_x, y=robot_y)

# Get latest body state
print(mem.body_status())
# Body Status:
#   [battery] battery: 45% (2min ago)
#   [cpu_temp] 72C across 4 cores (30s ago)
#   [joint_health] all joints nominal (5min ago)

# Body state is automatically included in get_current_context()
print(mem.get_current_context())

# Spatial-interoceptive associations emerge naturally:
# spatial_query near a steep ramp surfaces both terrain and battery observations
mem.spatial_query(x=10.0, y=10.0, radius=3.0)
```

Body state observations flow through the standard ObservationNode pipeline — they participate in episodes, consolidation, and all query tools. The key design decisions:

- **No new node types**: body state uses `ObservationNode` with `source_type="interoception"`
- **Position-stamped**: body observations inherit the robot's current position
- **Position-preserving**: body observations do NOT update the robot's tracked position
- **Peer layers**: `"battery"`, `"cpu_temp"` etc. are regular layer names alongside `"vlm"`, `"detections"`

## Examples

See the [`examples/`](examples/) directory:

- **[`basic_memory.py`](examples/basic_memory.py)** -- Standalone usage: adding observations, episode lifecycle, all query tools
- **[`memory_agent.py`](examples/memory_agent.py)** -- Simulated LLM agent using memory tools in a ReAct-style loop

## Testing Harness

The `harness/` directory contains a standalone test client that simulates what EMOS will do: an agent navigates a MiniGrid environment, a VLM describes each frame, synthetic body state is generated, and everything flows into eMEM for storage and querying. A ReAct agent then answers questions against the populated memory.

### Setup

```bash
# Install harness dependencies (not part of emem itself)
pip install minigrid gymnasium

# Pull Ollama models
ollama pull nomic-embed-text-v2-moe:latest
ollama pull qwen3.5:latest
```

### Running

```bash
# Quick smoke test (20 steps, VLM every 10)
python -m harness.run --steps 20 --vlm-every 10

# Full run (100 steps, VLM every 5)
python -m harness.run --steps 100 --vlm-every 5

# Keep the memory DB for later inspection
python -m harness.run --steps 50 --db-path my_test.db

# JSON output
python -m harness.run --steps 50 --json

# Verbose logging (debug level)
python -m harness.run --steps 20 -v
```

The harness logs progress to stderr as it runs: each VLM inference, episode boundaries, and each evaluation query with its result.

### What it does

Each run:

1. **Ingestion** -- navigates MiniGrid with random actions, periodically sends rendered frames to a VLM (`qwen3.5:latest`) with two prompts (scene description + place type), and feeds results into eMEM as separate observation layers. Synthetic body state (battery, CPU temp, joint health) is generated every step.

2. **Evaluation** -- a ReAct agent answers 8 benchmark queries ("What places have I visited?", "What's my battery level?", "Where is the door?", etc.) and the harness measures tool selection accuracy, answer relevance, and latency.

### Using from Python

```python
from emem import SpatioTemporalMemory
from harness.providers import OllamaEmbeddingProvider, OllamaLLMClient, OllamaVLM
from harness.environments import MiniGridAdapter, SyntheticInteroception
from harness.agent import ReactAgent

embedder = OllamaEmbeddingProvider()
llm = OllamaLLMClient()
vlm = OllamaVLM()
env = MiniGridAdapter()
intero = SyntheticInteroception()

with SpatioTemporalMemory(
    db_path="my_test.db", embedding_provider=embedder, llm_client=llm,
) as mem:
    mem.start_episode("exploration")
    frame, pos = env.reset()

    for step in range(50):
        frame, pos, reward, done, info = env.step(2)  # forward

        if step % 5 == 0:
            desc = vlm.describe(frame, "Describe what you see in 1-2 sentences.")
            mem.add(desc, x=float(pos[0]), y=float(pos[1]), layer_name="description")

        for layer, text in intero.step().items():
            mem.add_body_state(
                text, layer_name=layer, x=float(pos[0]), y=float(pos[1])
            )

        if done:
            mem.end_episode()
            mem.start_episode("exploration")
            frame, pos = env.reset()

    mem.end_episode()

    # Query directly
    print(mem.semantic_search("door"))
    print(mem.body_status())

    # Or through the ReAct agent
    agent = ReactAgent(mem)
    print(agent.run("What places have I visited?").answer)

env.close()
```

## Academic Benchmarks

The `harness/benchmarks/academic/` module provides replay-based evaluation on established academic benchmarks for quantitative comparison. Unlike the live harness above, these load pre-recorded trajectories, ingest them into eMEM, answer dataset questions via a ReAct agent, and score against ground truth.

### Supported Benchmarks

| Benchmark | Venue | Questions | Focus | Scorer |
|-----------|-------|-----------|-------|--------|
| **LoCoMo** | ACL 2024 | 1,986 conversational QA | Temporal reasoning, multi-hop recall | Token F1 + BLEU-1 |
| **eMEM-Bench** (ours) | — | 492 embodied QA across 12 AI2-THOR scenes | Spatial, temporal, entity, episodic, interoceptive recall | LLM judge (1-5) |

### Ablation Study

Five configurations isolate the contribution of each eMEM component:

| Ablation | What it removes |
|----------|----------------|
| `full` | Nothing (baseline) |
| `vector_only` | All tools except `semantic_search` |
| `no_spatial` | `spatial_query`, `locate`, `recall` tools |
| `no_consolidation` | Gist generation and archival |
| `flat_layer` | Multi-layer support (all observations use `layer_name="default"`) |

### Running Benchmarks

```bash
# LoCoMo (text-only conversational memory)
python -m harness.run_benchmark \
  --dataset locomo --data-dir ./data/locomo \
  --max-samples 3 --ablation full

# eMEM-Bench (embodied QA over AI2-THOR scenes)
python -m harness.run_benchmark \
  --dataset emem_bench --data-dir ./data/emem_bench \
  --ablation full

# Full ablation sweep with JSON output
python -m harness.run_benchmark \
  --dataset locomo --data-dir ./data/locomo \
  --ablation full,vector_only,no_spatial,no_consolidation,flat_layer \
  --json > results/locomo_ablation.json

# Use Gemini provider instead of Ollama
python -m harness.run_benchmark \
  --dataset locomo --data-dir ./data/locomo \
  --provider gemini --embed-model text-embedding-004 \
  --llm-model gemini-2.0-flash-lite
```

### CLI Options

```
--dataset         locomo | emem_bench
--data-dir        Path to dataset directory
--ablation        Comma-separated ablation names (default: full)
--max-samples     Limit number of samples evaluated
--provider        ollama | gemini (default: ollama)
--embed-model     Embedding model (default: nomic-embed-text-v2-moe:latest)
--llm-model       LLM for agent + consolidation (default: qwen3.5:latest)
--json            Output JSON report instead of formatted tables
-v                Verbose logging
```

### Data Directory Layouts

**LoCoMo:**
```
data/locomo/
  locomo.json   # array of conversations with sessions + qa_pairs
```

**eMEM-Bench:**
```
data/emem_bench/
  scenes/{scene_id}/trajectory.json
  scenes/{scene_id}/questions.json
```

## Development

```bash
git clone https://github.com/Automatika-Robotics/eMEM.git
cd eMEM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Part of the EMOS Ecosystem

eMEM is the memory layer of [EMOS](https://github.com/automatika-robotics/emos) (Embodied Operating System) — the unified orchestration layer for Physical AI. Alongside its sibling components:

- **[EmbodiedAgents](https://github.com/automatika-robotics/embodied-agents)** — Intelligence and manipulation. ML model graphs with semantic routing and adaptive reconfiguration.
- **[Kompass](https://github.com/automatika-robotics/kompass)** — Navigation. GPU-accelerated planning and control.
- **[Sugarcoat](https://github.com/automatika-robotics/sugarcoat)** — Lifecycle management. Event-driven system design for ROS 2.
- **eMEM** — Memory. Spatio-temporal recall for situated agents.

Write a recipe once. Deploy it on any robot. No code changes.

## Resources

- [EMOS Documentation](https://emos.automatikarobotics.com) — Tutorials, recipes, and usage guides
- [Discord](https://discord.gg/B9ZU6qjzND) — Community and support

## License

All rights reserved. Copyright (c) 2024 Automatika Robotics.
