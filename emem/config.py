from dataclasses import dataclass


@dataclass
class SpatioTemporalMemoryConfig:
    """Configuration parameters for a SpatioTemporalMemory instance."""

    # Storage
    db_path: str = "memory.db"
    hnsw_path: str = "memory_hnsw.bin"

    # Embedding
    embedding_dim: int = 384

    # Working memory
    working_memory_size: int = 50
    flush_interval: float = 2.0  # seconds
    flush_batch_size: int = 5  # observations before auto-flush

    # Consolidation
    consolidation_window: float = 1800.0  # 30 minutes in seconds
    consolidation_spatial_eps: float = 3.0  # DBSCAN eps in meters
    consolidation_min_samples: int = 3

    # Archival
    archive_after_seconds: float = 3600.0  # seconds in long_term before archival

    # Entity extraction
    entity_extract_flush_interval: int = 10  # extract every N flushes
    entity_extract_time_interval: float = (
        60.0  # or every N seconds, whichever comes first
    )
    entity_similarity_threshold: float = 0.85
    entity_text_similarity_threshold: float = 0.65
    entity_spatial_radius: float = 5.0

    # Hybrid retrieval (BM25 + HNSW). When True, semantic_search and
    # search_gists combine a sparse BM25 lookup with the dense HNSW
    # lookup via Reciprocal Rank Fusion. FTS5 entries are always
    # populated at ingest so the flag can be toggled per query without
    # re-ingesting.
    use_hybrid_retrieval: bool = True
    rrf_k: int = 60  # RRF constant; 60 is the de facto default

    # Recency weighting
    recency_weight: float = 0.0  # alpha multiplier (0.0 = disabled)
    recency_halflife: float = 3600.0  # time constant in seconds

    # HNSW parameters
    hnsw_ef_construction: int = 200
    hnsw_m: int = 16
    hnsw_ef_search: int = 50
    hnsw_max_elements: int = 100_000
