<p align="center">
   <picture>
    <source media="(prefers-color-scheme: dark)" srcset="_static/eMEM_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="_static/eMEM_light.png">
    <img alt="EMOS" src="docs/_static/Emos_light.png" width="50%">
  </picture>
  <p align="center">
    <strong>Embodied Memory for Situated Agents</strong>
  </p>
  <p align="center">
    A hybrid graph-based spatio-temporal memory system that gives embodied agents<br>
    the ability to remember <em>what</em> they observed, <em>where</em> they observed it, and <em>when</em>.
  </p>
</p>

<p align="center">
  <a href="#installation">Installation</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#llm-tool-interface">LLM Tools</a> &middot;
  <a href="#api-reference">API</a>
</p>

---

## Why eMEM?

Mobile robots and embodied AI agents accumulate thousands of observations per session -- object detections, scene descriptions, sensor readings -- but existing memory systems force a choice: **vector databases** that discard spatial structure, or **metric maps** that discard semantics. eMEM unifies both.

| Capability | eMEM |
|---|---|
| Semantic search (meaning) | HNSW approximate nearest neighbor |
| Spatial queries (where) | R-tree range and nearest-neighbor |
| Temporal queries (when) | SQLite indexed timestamps |
| Episode structure | Graph edges with hierarchical sub-tasks |
| Memory consolidation | Automatic gist generation + tiered archival |
| LLM integration | 6 ready-to-use tool definitions |

Everything runs **fully embedded** -- SQLite, hnswlib, and Rtree -- with zero external services. A single `.db` file and a `.hnsw.bin` file contain the entire memory state.

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
     |  (buffer)  |  |  (6 tools)  |  | Engine     |
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

The memory is structured as a typed graph with three node types and four edge types:

**Nodes:**
- **ObservationNode** -- A single perception event: text, coordinates, timestamp, layer, confidence
- **EpisodeNode** -- A task or activity span grouping related observations
- **GistNode** -- A consolidated summary of multiple observations, with spatial extent and time range

**Edges:**
- `BELONGS_TO` -- Observation &rarr; Episode
- `FOLLOWS` -- Episode &rarr; Episode (temporal sequence)
- `SUBTASK_OF` -- Episode &rarr; Episode (hierarchical nesting)
- `SUMMARIZES` -- Gist &rarr; Observation(s)

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

eMEM provides 6 tools designed for LLM function-calling. Each returns a token-efficient formatted string.

| Tool | Description |
|---|---|
| `semantic_search` | Search by meaning across observations and consolidated summaries |
| `spatial_query` | Find observations within a radius of a point |
| `temporal_query` | Find observations in a time range, chronologically ordered |
| `episode_summary` | Get summary of one or more episodes |
| `get_current_context` | Situational awareness: nearby objects, area summaries, recent activity |
| `search_gists` | Search only consolidated long-term memory |

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

## Examples

See the [`examples/`](examples/) directory:

- **[`basic_memory.py`](examples/basic_memory.py)** -- Standalone usage: adding observations, episode lifecycle, all 6 query tools
- **[`memory_agent.py`](examples/memory_agent.py)** -- Simulated LLM agent using memory tools in a ReAct-style loop

## Development

```bash
git clone https://github.com/Automatika-Robotics/eMEM.git
cd eMEM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## License

All rights reserved. Copyright (c) 2024 Automatika Robotics.
