import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hnswlib
import numpy as np

from emem.config import SpatioTemporalMemoryConfig
from emem.embeddings import EmbeddingProvider, NullEmbeddingProvider
from emem.spatial import SpatialIndex, _to_xyz
from emem.types import (
    Edge,
    EdgeType,
    EntityNode,
    EpisodeNode,
    EpisodeStatus,
    GistNode,
    ObservationNode,
    Tier,
)


class MemoryStore:
    """Hybrid graph-based spatio-temporal memory store.

    Combines SQLite (structured data + graph edges), hnswlib (vector similarity),
    and rtree (spatial range queries) into a single queryable store.
    """

    def __init__(
        self,
        config: Optional[SpatioTemporalMemoryConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.config = config or SpatioTemporalMemoryConfig()
        self.embedding_provider = embedding_provider or NullEmbeddingProvider(self.config.embedding_dim)
        self._auto_embed = not isinstance(self.embedding_provider, NullEmbeddingProvider)
        self._lock = threading.Lock()

        # SQLite
        self._db = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

        # hnswlib
        self._hnsw = hnswlib.Index(space="cosine", dim=self.embedding_provider.dim)
        self._hnsw_path = Path(self.config.hnsw_path)
        self._hnsw_id_map: Dict[int, str] = {}
        self._hnsw_str_map: Dict[str, int] = {}
        self._hnsw_counter = 0
        if self._hnsw_path.exists():
            self._hnsw.load_index(str(self._hnsw_path), max_elements=self.config.hnsw_max_elements)
            self._load_hnsw_mappings()
        else:
            self._hnsw.init_index(
                max_elements=self.config.hnsw_max_elements,
                ef_construction=self.config.hnsw_ef_construction,
                M=self.config.hnsw_m,
            )
        self._hnsw.set_ef(self.config.hnsw_ef_search)

        # Spatial index (in-memory, rebuilt on load)
        self._spatial = SpatialIndex()
        self._load_spatial_index()

    def _init_schema(self) -> None:
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL DEFAULT 0.0,
                timestamp REAL NOT NULL,
                layer_name TEXT NOT NULL DEFAULT 'default',
                source_type TEXT NOT NULL DEFAULT 'manual',
                confidence REAL NOT NULL DEFAULT 1.0,
                episode_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                tier TEXT NOT NULL DEFAULT 'short_term',
                has_embedding INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON observations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_obs_layer ON observations(layer_name);
            CREATE INDEX IF NOT EXISTS idx_obs_episode ON observations(episode_id);
            CREATE INDEX IF NOT EXISTS idx_obs_tier ON observations(tier);
            CREATE INDEX IF NOT EXISTS idx_obs_source_type ON observations(source_type);

            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT NOT NULL DEFAULT 'active',
                gist TEXT NOT NULL DEFAULT '',
                has_gist_embedding INTEGER NOT NULL DEFAULT 0,
                parent_episode_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_ep_name ON episodes(name);
            CREATE INDEX IF NOT EXISTS idx_ep_status ON episodes(status);

            CREATE TABLE IF NOT EXISTS gists (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                cx REAL NOT NULL,
                cy REAL NOT NULL,
                cz REAL NOT NULL DEFAULT 0.0,
                radius REAL NOT NULL,
                time_start REAL NOT NULL,
                time_end REAL NOT NULL,
                source_observation_count INTEGER NOT NULL,
                source_observation_ids TEXT NOT NULL DEFAULT '[]',
                layer_name TEXT,
                episode_id TEXT,
                has_embedding INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_gist_time ON gists(time_start, time_end);

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edge_type ON edges(edge_type);

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL DEFAULT 0.0,
                last_seen REAL NOT NULL,
                first_seen REAL NOT NULL,
                observation_count INTEGER NOT NULL DEFAULT 1,
                confidence REAL NOT NULL DEFAULT 1.0,
                entity_type TEXT,
                layer_name TEXT NOT NULL DEFAULT 'default',
                metadata TEXT NOT NULL DEFAULT '{}',
                has_embedding INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entity_last_seen ON entities(last_seen);

            CREATE TABLE IF NOT EXISTS hnsw_mappings (
                int_id INTEGER PRIMARY KEY,
                str_id TEXT NOT NULL,
                node_type TEXT NOT NULL DEFAULT 'observation'
            );
        """)
        self._db.commit()

    def _load_hnsw_mappings(self) -> None:
        rows = self._db.execute("SELECT int_id, str_id FROM hnsw_mappings").fetchall()
        for row in rows:
            self._hnsw_id_map[row["int_id"]] = row["str_id"]
            self._hnsw_str_map[row["str_id"]] = row["int_id"]
        if rows:
            self._hnsw_counter = max(r["int_id"] for r in rows)

    def _load_spatial_index(self) -> None:
        rows = self._db.execute(
            "SELECT id, x, y, z FROM observations WHERE tier != 'archived'"
        ).fetchall()
        for row in rows:
            self._spatial.insert(row["id"], np.array([row["x"], row["y"], row["z"]]))

        entity_rows = self._db.execute("SELECT id, x, y, z FROM entities").fetchall()
        for row in entity_rows:
            self._spatial.insert(row["id"], np.array([row["x"], row["y"], row["z"]]))

    @staticmethod
    def _spatial_filter_sql(
        center: np.ndarray,
        radius: float,
        x_col: str = "x",
        y_col: str = "y",
        z_col: str = "z",
    ) -> Tuple[str, List[float]]:
        """Return a SQL WHERE clause fragment and params for a distance filter."""
        x, y, z = _to_xyz(center)
        clause = (
            f"(({x_col} - ?) * ({x_col} - ?) + ({y_col} - ?) * ({y_col} - ?) "
            f"+ ({z_col} - ?) * ({z_col} - ?)) <= ?"
        )
        params = [x, x, y, y, z, z, radius ** 2]
        return clause, params

    def _next_hnsw_id(self) -> int:
        self._hnsw_counter += 1
        return self._hnsw_counter

    def _add_to_hnsw(self, str_id: str, embedding: np.ndarray, node_type: str = "observation") -> None:
        # Remove old entry if re-inserting
        if str_id in self._hnsw_str_map:
            self._remove_from_hnsw(str_id)

        int_id = self._next_hnsw_id()
        self._hnsw_id_map[int_id] = str_id
        self._hnsw_str_map[str_id] = int_id
        if self._hnsw.get_current_count() >= self._hnsw.get_max_elements() - 1:
            self._hnsw.resize_index(self._hnsw.get_max_elements() + self.config.hnsw_max_elements)
        self._hnsw.add_items(embedding.reshape(1, -1), np.array([int_id]))
        self._db.execute(
            "INSERT OR REPLACE INTO hnsw_mappings (int_id, str_id, node_type) VALUES (?, ?, ?)",
            (int_id, str_id, node_type),
        )

    def _remove_from_hnsw(self, str_id: str) -> None:
        int_id = self._hnsw_str_map.get(str_id)
        if int_id is not None:
            self._hnsw.mark_deleted(int_id)
            del self._hnsw_id_map[int_id]
            del self._hnsw_str_map[str_id]
            self._db.execute("DELETE FROM hnsw_mappings WHERE int_id = ?", (int_id,))

    # ── Write Operations ──────────────────────────────────────────────

    def add_observation(self, obs: ObservationNode) -> str:
        return self.add_observations_batch([obs])[0]

    def add_observations_batch(self, observations: List[ObservationNode]) -> List[str]:
        ids = []
        with self._lock:
            # Batch embed observations that need it
            needs_embed = [o for o in observations if o.embedding is None and self._auto_embed]
            if needs_embed:
                embeddings = self.embedding_provider.embed([o.text for o in needs_embed])
                for o, emb in zip(needs_embed, embeddings):
                    o.embedding = emb

            for obs in observations:
                x, y, z = _to_xyz(obs.coordinates)
                has_emb = 1 if obs.embedding is not None else 0

                self._db.execute(
                    """INSERT INTO observations
                       (id, text, x, y, z, timestamp, layer_name, source_type,
                        confidence, episode_id, metadata, tier, has_embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (obs.id, obs.text, x, y, z, obs.timestamp, obs.layer_name,
                     obs.source_type, obs.confidence, obs.episode_id,
                     json.dumps(obs.metadata), obs.tier, has_emb),
                )

                if obs.embedding is not None:
                    self._add_to_hnsw(obs.id, obs.embedding, "observation")

                self._spatial.insert(obs.id, np.array([x, y, z]))

                if obs.episode_id:
                    self._add_edge(Edge(
                        source_id=obs.id,
                        target_id=obs.episode_id,
                        edge_type=EdgeType.BELONGS_TO,
                    ))
                ids.append(obs.id)

            self._db.commit()
        return ids

    def start_episode(
        self,
        name: str,
        start_time: float,
        metadata: Optional[Dict[str, Any]] = None,
        parent_episode_id: Optional[str] = None,
    ) -> str:
        ep = EpisodeNode(
            name=name,
            start_time=start_time,
            metadata=metadata or {},
            parent_episode_id=parent_episode_id,
        )
        with self._lock:
            self._db.execute(
                """INSERT INTO episodes (id, name, start_time, end_time, status, gist,
                   has_gist_embedding, parent_episode_id, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ep.id, ep.name, ep.start_time, ep.end_time, ep.status, ep.gist,
                 0, ep.parent_episode_id, json.dumps(ep.metadata)),
            )
            if parent_episode_id:
                self._add_edge(Edge(
                    source_id=ep.id,
                    target_id=parent_episode_id,
                    edge_type=EdgeType.SUBTASK_OF,
                ))
            self._db.commit()
        return ep.id

    def end_episode(
        self,
        episode_id: str,
        end_time: float,
        gist: str = "",
        gist_embedding: Optional[np.ndarray] = None,
    ) -> None:
        with self._lock:
            has_emb = 0
            if gist_embedding is not None:
                has_emb = 1
                self._add_to_hnsw(episode_id, gist_embedding, "episode")

            self._db.execute(
                """UPDATE episodes SET end_time = ?, status = ?, gist = ?,
                   has_gist_embedding = ? WHERE id = ?""",
                (end_time, EpisodeStatus.COMPLETED.value, gist, has_emb, episode_id),
            )
            self._db.commit()

    def add_gist(self, gist: GistNode) -> str:
        with self._lock:
            cx, cy, cz = _to_xyz(gist.center_position)

            if gist.embedding is None and self._auto_embed:
                gist.embedding = self.embedding_provider.embed([gist.text])[0]

            has_emb = 1 if gist.embedding is not None else 0

            self._db.execute(
                """INSERT OR REPLACE INTO gists
                   (id, text, cx, cy, cz, radius, time_start, time_end,
                    source_observation_count, source_observation_ids,
                    layer_name, episode_id, has_embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (gist.id, gist.text, cx, cy, cz, gist.radius,
                 gist.time_start, gist.time_end, gist.source_observation_count,
                 json.dumps(gist.source_observation_ids), gist.layer_name,
                 gist.episode_id, has_emb),
            )

            if gist.embedding is not None:
                self._add_to_hnsw(gist.id, gist.embedding, "gist")

            for obs_id in gist.source_observation_ids:
                self._add_edge(Edge(
                    source_id=gist.id,
                    target_id=obs_id,
                    edge_type=EdgeType.SUMMARIZES,
                ))

            self._db.commit()
        return gist.id

    def _add_edge(self, edge: Edge) -> str:
        self._db.execute(
            "INSERT OR REPLACE INTO edges (id, source_id, target_id, edge_type, metadata) VALUES (?, ?, ?, ?, ?)",
            (edge.id, edge.source_id, edge.target_id, edge.edge_type.value, json.dumps(edge.metadata)),
        )
        return edge.id

    def update_observation_tiers(self, obs_ids: List[str], tier: str, drop_text: bool = False) -> None:
        """Batch update tier for multiple observations in a single transaction.

        :param obs_ids: Observation IDs to update.
        :param tier: Target tier value.
        :param drop_text: If ``True``, clear text and remove from HNSW/spatial indexes.
        """
        with self._lock:
            for obs_id in obs_ids:
                if drop_text:
                    self._db.execute(
                        "UPDATE observations SET tier = ?, text = '' WHERE id = ?",
                        (tier, obs_id),
                    )
                    self._remove_from_hnsw(obs_id)
                    row = self._db.execute("SELECT x, y, z FROM observations WHERE id = ?", (obs_id,)).fetchone()
                    if row:
                        self._spatial.delete(obs_id, np.array([row["x"], row["y"], row["z"]]))
                else:
                    self._db.execute(
                        "UPDATE observations SET tier = ? WHERE id = ?",
                        (tier, obs_id),
                    )
            self._db.commit()

    def update_observation_tier(self, obs_id: str, tier: str, drop_text: bool = False) -> None:
        self.update_observation_tiers([obs_id], tier, drop_text)

    def add_edge(self, edge: Edge) -> str:
        with self._lock:
            edge_id = self._add_edge(edge)
            self._db.commit()
        return edge_id

    def add_edges(self, edges: List[Edge]) -> None:
        if not edges:
            return
        with self._lock:
            for edge in edges:
                self._add_edge(edge)
            self._db.commit()

    # ── Entity Operations ────────────────────────────────────────────

    def add_entity(self, entity: EntityNode) -> str:
        with self._lock:
            x, y, z = _to_xyz(entity.coordinates)

            if entity.embedding is None and self._auto_embed:
                entity.embedding = self.embedding_provider.embed([entity.name])[0]

            has_emb = 1 if entity.embedding is not None else 0

            self._db.execute(
                """INSERT OR REPLACE INTO entities
                   (id, name, x, y, z, last_seen, first_seen,
                    observation_count, confidence, entity_type, layer_name,
                    metadata, has_embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entity.id, entity.name, x, y, z, entity.last_seen,
                 entity.first_seen, entity.observation_count, entity.confidence,
                 entity.entity_type, entity.layer_name,
                 json.dumps(entity.metadata), has_emb),
            )

            if entity.embedding is not None:
                self._add_to_hnsw(entity.id, entity.embedding, "entity")

            self._spatial.insert(entity.id, np.array([x, y, z]))
            self._db.commit()
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        row = self._db.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return self._row_to_entity(row) if row else None

    def update_entity(self, entity: EntityNode) -> None:
        with self._lock:
            # Remove old spatial entry
            old_row = self._db.execute("SELECT x, y, z FROM entities WHERE id = ?", (entity.id,)).fetchone()
            if old_row:
                self._spatial.delete(entity.id, np.array([old_row["x"], old_row["y"], old_row["z"]]))

            x, y, z = _to_xyz(entity.coordinates)

            if entity.embedding is None and self._auto_embed:
                entity.embedding = self.embedding_provider.embed([entity.name])[0]

            has_emb = 1 if entity.embedding is not None else 0

            self._db.execute(
                """UPDATE entities SET name = ?, x = ?, y = ?, z = ?,
                   last_seen = ?, first_seen = ?, observation_count = ?,
                   confidence = ?, entity_type = ?, layer_name = ?,
                   metadata = ?, has_embedding = ?
                   WHERE id = ?""",
                (entity.name, x, y, z, entity.last_seen, entity.first_seen,
                 entity.observation_count, entity.confidence, entity.entity_type,
                 entity.layer_name, json.dumps(entity.metadata), has_emb,
                 entity.id),
            )

            if entity.embedding is not None:
                self._add_to_hnsw(entity.id, entity.embedding, "entity")

            self._spatial.insert(entity.id, np.array([x, y, z]))
            self._db.commit()

    def find_matching_entity(
        self,
        name: str,
        coordinates: np.ndarray,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[EntityNode]:
        if embedding is None and self._auto_embed:
            embedding = self.embedding_provider.embed([name])[0]

        if embedding is not None and self._hnsw.get_current_count() > 0:
            live_count = len(self._hnsw_id_map)
            if live_count > 0:
                fetch_k = min(20, live_count)
                labels, distances = self._hnsw.knn_query(embedding.reshape(1, -1), k=fetch_k)

                # Resolve all candidate IDs and their types in one batch
                candidate_ids = []
                candidate_dists = []
                for label, dist in zip(labels[0], distances[0]):
                    str_id = self._hnsw_id_map.get(int(label))
                    if str_id:
                        candidate_ids.append(str_id)
                        candidate_dists.append(dist)

                if candidate_ids:
                    ph = ",".join("?" * len(candidate_ids))
                    type_rows = self._db.execute(
                        "SELECT str_id, node_type FROM hnsw_mappings WHERE str_id IN (%s)" % ph,
                        candidate_ids,
                    ).fetchall()
                    type_map = {r["str_id"]: r["node_type"] for r in type_rows}

                    coords_xyz = np.array(_to_xyz(coordinates))
                    for str_id, dist in zip(candidate_ids, candidate_dists):
                        if type_map.get(str_id) != "entity":
                            continue
                        similarity = 1.0 - dist
                        if similarity < self.config.entity_similarity_threshold:
                            continue
                        entity = self.get_entity(str_id)
                        if entity is None:
                            continue
                        dist_spatial = float(np.linalg.norm(entity.coordinates - coords_xyz))
                        if dist_spatial <= self.config.entity_spatial_radius:
                            return entity

        # Fallback: exact name match within spatial radius
        clause, dist_params = self._spatial_filter_sql(
            coordinates, self.config.entity_spatial_radius,
        )
        rows = self._db.execute(
            "SELECT * FROM entities WHERE name = ? AND " + clause,
            [name] + dist_params,
        ).fetchall()
        if rows:
            return self._row_to_entity(rows[0])
        return None

    def query_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        near_coordinates: Optional[np.ndarray] = None,
        spatial_radius: Optional[float] = None,
        last_seen_after: Optional[float] = None,
        n_results: int = 10,
    ) -> List[EntityNode]:
        query = "SELECT * FROM entities WHERE 1=1"
        params: list = []
        if name:
            query += " AND name LIKE ?"
            params.append("%%%s%%" % name)
        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)
        if near_coordinates is not None and spatial_radius is not None:
            clause, dist_params = self._spatial_filter_sql(near_coordinates, spatial_radius)
            query += " AND " + clause
            params.extend(dist_params)
        if last_seen_after is not None:
            query += " AND last_seen >= ?"
            params.append(last_seen_after)
        query += " ORDER BY last_seen DESC LIMIT ?"
        params.append(n_results)
        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def get_entity_observations(self, entity_id: str) -> List[ObservationNode]:
        edges = self.get_edges(source_id=entity_id, edge_type=EdgeType.OBSERVED_IN)
        obs_ids = [e.target_id for e in edges]
        if not obs_ids:
            return []
        placeholders = ",".join("?" * len(obs_ids))
        rows = self._db.execute(
            "SELECT * FROM observations WHERE id IN (%s) ORDER BY timestamp" % placeholders,
            obs_ids,
        ).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def get_cooccurring_entities(self, entity_id: str) -> List[EntityNode]:
        edges_out = self.get_edges(source_id=entity_id, edge_type=EdgeType.COOCCURS_WITH)
        edges_in = self.get_edges(target_id=entity_id, edge_type=EdgeType.COOCCURS_WITH)
        related_ids = set()
        for e in edges_out:
            related_ids.add(e.target_id)
        for e in edges_in:
            related_ids.add(e.source_id)
        if not related_ids:
            return []
        placeholders = ",".join("?" * len(related_ids))
        rows = self._db.execute(
            "SELECT * FROM entities WHERE id IN (%s)" % placeholders,
            list(related_ids),
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ── Read Operations ───────────────────────────────────────────────

    def _row_to_observation(self, row: sqlite3.Row) -> ObservationNode:
        return ObservationNode(
            id=row["id"],
            text=row["text"],
            coordinates=np.array([row["x"], row["y"], row["z"]]),
            timestamp=row["timestamp"],
            layer_name=row["layer_name"],
            source_type=row["source_type"],
            confidence=row["confidence"],
            episode_id=row["episode_id"],
            metadata=json.loads(row["metadata"]),
            tier=row["tier"],
        )

    def _row_to_episode(self, row: sqlite3.Row) -> EpisodeNode:
        return EpisodeNode(
            id=row["id"],
            name=row["name"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            status=row["status"],
            gist=row["gist"],
            parent_episode_id=row["parent_episode_id"],
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_gist(self, row: sqlite3.Row) -> GistNode:
        return GistNode(
            id=row["id"],
            text=row["text"],
            center_position=np.array([row["cx"], row["cy"], row["cz"]]),
            radius=row["radius"],
            time_start=row["time_start"],
            time_end=row["time_end"],
            source_observation_count=row["source_observation_count"],
            source_observation_ids=json.loads(row["source_observation_ids"]),
            layer_name=row["layer_name"],
            episode_id=row["episode_id"],
        )

    def _row_to_entity(self, row: sqlite3.Row) -> EntityNode:
        return EntityNode(
            id=row["id"],
            name=row["name"],
            coordinates=np.array([row["x"], row["y"], row["z"]]),
            last_seen=row["last_seen"],
            first_seen=row["first_seen"],
            observation_count=row["observation_count"],
            confidence=row["confidence"],
            entity_type=row["entity_type"],
            layer_name=row["layer_name"],
            metadata=json.loads(row["metadata"]),
        )

    def get_observation(self, obs_id: str) -> Optional[ObservationNode]:
        row = self._db.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        return self._row_to_observation(row) if row else None

    def get_episode(self, episode_id: str) -> Optional[EpisodeNode]:
        row = self._db.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return self._row_to_episode(row) if row else None

    def get_gist(self, gist_id: str) -> Optional[GistNode]:
        row = self._db.execute("SELECT * FROM gists WHERE id = ?", (gist_id,)).fetchone()
        return self._row_to_gist(row) if row else None

    def get_episode_observations(self, episode_id: str) -> List[ObservationNode]:
        rows = self._db.execute(
            "SELECT * FROM observations WHERE episode_id = ? ORDER BY timestamp",
            (episode_id,),
        ).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def list_episodes(
        self,
        task_name: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        last_n: Optional[int] = None,
        status: Optional[str] = None,
    ) -> List[EpisodeNode]:
        query = "SELECT * FROM episodes WHERE 1=1"
        params: list = []
        if task_name:
            query += " AND name LIKE ?"
            params.append("%%%s%%" % task_name)
        if time_range:
            query += " AND start_time >= ? AND (end_time IS NULL OR end_time <= ?)"
            params.append(time_range[0])
            params.append(time_range[1])
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY start_time DESC"
        if last_n:
            query += " LIMIT ?"
            params.append(last_n)
        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_episode(r) for r in rows]

    # ── Query Operations ──────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        layer: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        spatial_center: Optional[np.ndarray] = None,
        spatial_radius: Optional[float] = None,
        episode_id: Optional[str] = None,
    ) -> List[Union[ObservationNode, GistNode, EntityNode]]:
        """Search by meaning across observations, consolidated gists, and entities.

        :param query: Natural language query string.
        :param n_results: Maximum results to return.
        :param layer: Filter by layer name.
        :param time_range: ``(start, end)`` timestamp bounds.
        :param spatial_center: Centre point for spatial filter.
        :param spatial_radius: Radius for spatial filter.
        :param episode_id: Filter by episode.
        :returns: Ranked list of observations, gists, and/or entities.
        :rtype: List[Union[ObservationNode, GistNode, EntityNode]]
        """
        query_emb = self.embedding_provider.embed([query])[0]
        return self.semantic_search_by_vector(
            query_emb, n_results, layer, time_range, spatial_center, spatial_radius, episode_id
        )

    def semantic_search_by_vector(
        self,
        query_vector: np.ndarray,
        n_results: int = 5,
        layer: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        spatial_center: Optional[np.ndarray] = None,
        spatial_radius: Optional[float] = None,
        episode_id: Optional[str] = None,
    ) -> List[Union[ObservationNode, GistNode, EntityNode]]:
        """Search by embedding vector across observations, consolidated gists, and entities.

        :param query_vector: Embedding vector to search with.
        :param n_results: Maximum results to return.
        :param layer: Filter by layer name.
        :param time_range: ``(start, end)`` timestamp bounds.
        :param spatial_center: Centre point for spatial filter.
        :param spatial_radius: Radius for spatial filter.
        :param episode_id: Filter by episode.
        :returns: Ranked list of observations, gists, and/or entities.
        :rtype: List[Union[ObservationNode, GistNode, EntityNode]]
        """
        if self._hnsw.get_current_count() == 0:
            return []

        # Number of live (non-deleted) elements
        live_count = len(self._hnsw_id_map)
        if live_count == 0:
            return []
        fetch_k = min(n_results * 5, live_count)
        labels, distances = self._hnsw.knn_query(query_vector.reshape(1, -1), k=fetch_k)

        candidate_ids = []
        for label in labels[0]:
            str_id = self._hnsw_id_map.get(int(label))
            if str_id:
                candidate_ids.append(str_id)

        if not candidate_ids:
            return []

        # Partition candidates by node type
        placeholders = ",".join("?" * len(candidate_ids))
        type_rows = self._db.execute(
            "SELECT str_id, node_type FROM hnsw_mappings WHERE str_id IN (%s)" % placeholders,
            candidate_ids,
        ).fetchall()
        type_map = {r["str_id"]: r["node_type"] for r in type_rows}

        obs_ids: List[str] = []
        gist_ids: List[str] = []
        entity_ids: List[str] = []
        for cid in candidate_ids:
            node_type = type_map.get(cid)
            if node_type == "observation":
                obs_ids.append(cid)
            elif node_type == "gist":
                gist_ids.append(cid)
            elif node_type == "entity":
                entity_ids.append(cid)

        results: List[Union[ObservationNode, GistNode, EntityNode]] = []

        # Query observations
        if obs_ids:
            obs_ph = ",".join("?" * len(obs_ids))
            obs_sql = "SELECT * FROM observations WHERE id IN (%s)" % obs_ph
            obs_params: list = list(obs_ids)

            if layer:
                obs_sql += " AND layer_name = ?"
                obs_params.append(layer)
            if time_range:
                obs_sql += " AND timestamp >= ? AND timestamp <= ?"
                obs_params.extend(time_range)
            if episode_id:
                obs_sql += " AND episode_id = ?"
                obs_params.append(episode_id)
            if spatial_center is not None and spatial_radius is not None:
                clause, dist_params = self._spatial_filter_sql(spatial_center, spatial_radius)
                obs_sql += " AND " + clause
                obs_params.extend(dist_params)

            obs_sql += " AND tier != 'archived'"

            rows = self._db.execute(obs_sql, obs_params).fetchall()
            results.extend(self._row_to_observation(r) for r in rows)

        # Query gists
        if gist_ids:
            gist_ph = ",".join("?" * len(gist_ids))
            gist_sql = "SELECT * FROM gists WHERE id IN (%s)" % gist_ph
            gist_params: list = list(gist_ids)

            if layer:
                gist_sql += " AND layer_name = ?"
                gist_params.append(layer)
            if time_range:
                gist_sql += " AND time_start <= ? AND time_end >= ?"
                # Gist overlaps time range if gist.time_start <= range_end AND gist.time_end >= range_start
                gist_params.extend([time_range[1], time_range[0]])
            if episode_id:
                gist_sql += " AND episode_id = ?"
                gist_params.append(episode_id)
            if spatial_center is not None and spatial_radius is not None:
                clause, dist_params = self._spatial_filter_sql(
                    spatial_center, spatial_radius, x_col="cx", y_col="cy", z_col="cz",
                )
                gist_sql += " AND " + clause
                gist_params.extend(dist_params)

            rows = self._db.execute(gist_sql, gist_params).fetchall()
            results.extend(self._row_to_gist(r) for r in rows)

        # Query entities
        if entity_ids:
            ent_ph = ",".join("?" * len(entity_ids))
            ent_sql = "SELECT * FROM entities WHERE id IN (%s)" % ent_ph
            ent_params: list = list(entity_ids)

            if layer:
                ent_sql += " AND layer_name = ?"
                ent_params.append(layer)
            if time_range:
                ent_sql += " AND last_seen >= ? AND last_seen <= ?"
                ent_params.extend(time_range)
            if spatial_center is not None and spatial_radius is not None:
                clause, dist_params = self._spatial_filter_sql(spatial_center, spatial_radius)
                ent_sql += " AND " + clause
                ent_params.extend(dist_params)

            rows = self._db.execute(ent_sql, ent_params).fetchall()
            results.extend(self._row_to_entity(r) for r in rows)

        # Preserve HNSW distance ordering
        id_order = {sid: i for i, sid in enumerate(candidate_ids)}
        results.sort(key=lambda item: id_order.get(item.id, 9999))
        return results[:n_results]

    def spatial_query(
        self,
        center: np.ndarray,
        radius: float,
        layer: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        n_results: int = 10,
        source_type: Optional[str] = None,
        exclude_source_type: Optional[str] = None,
    ) -> List[ObservationNode]:
        """Find observations within a radius of a point.

        :param center: 3D coordinate array.
        :param radius: Search radius in metres.
        :param layer: Filter by layer name.
        :param time_range: ``(start, end)`` timestamp bounds.
        :param n_results: Maximum results to return.
        :returns: Observations sorted by distance from *center*.
        :rtype: List[ObservationNode]
        """
        candidate_ids = self._spatial.query_radius(center, radius)
        if not candidate_ids:
            return []

        placeholders = ",".join("?" * len(candidate_ids))
        query = "SELECT * FROM observations WHERE id IN (%s) AND tier != 'archived'" % placeholders
        params: list = list(candidate_ids)

        if layer:
            query += " AND layer_name = ?"
            params.append(layer)
        if time_range:
            query += " AND timestamp >= ? AND timestamp <= ?"
            params.extend(time_range)
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        if exclude_source_type:
            query += " AND source_type != ?"
            params.append(exclude_source_type)

        rows = self._db.execute(query, params).fetchall()
        results = [self._row_to_observation(r) for r in rows]

        # Sort by distance from center in Python (avoids f-string in SQL)
        cx, cy, cz = _to_xyz(center)
        center_pt = np.array([cx, cy, cz])
        results.sort(key=lambda o: float(np.sum((o.coordinates - center_pt) ** 2)))
        return results[:n_results]

    def spatial_nearest(self, point: np.ndarray, k: int = 5) -> List[ObservationNode]:
        """Find *k* nearest observations to *point*.

        :param point: 3D coordinate array.
        :param k: Number of nearest neighbours.
        :returns: Observations sorted by distance.
        :rtype: List[ObservationNode]
        """
        candidate_ids = self._spatial.query_nearest(point, k=k * 2)
        if not candidate_ids:
            return []
        placeholders = ",".join("?" * len(candidate_ids))
        rows = self._db.execute(
            "SELECT * FROM observations WHERE id IN (%s) AND tier != 'archived'" % placeholders,
            candidate_ids,
        ).fetchall()
        results = [self._row_to_observation(r) for r in rows]
        px, py, pz = _to_xyz(point)
        pt = np.array([px, py, pz])
        results.sort(key=lambda o: float(np.sum((o.coordinates - pt) ** 2)))
        return results[:k]

    def temporal_query(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        last_n_seconds: Optional[float] = None,
        layer: Optional[str] = None,
        spatial_center: Optional[np.ndarray] = None,
        spatial_radius: Optional[float] = None,
        order: str = "newest",
        n_results: int = 10,
        reference_time: Optional[float] = None,
        source_type: Optional[str] = None,
        exclude_source_type: Optional[str] = None,
    ) -> List[ObservationNode]:
        """Find observations in a time range.

        :param time_range: ``(start, end)`` timestamp bounds.
        :param last_n_seconds: Alternative to *time_range*; seconds before *reference_time*.
        :param layer: Filter by layer name.
        :param spatial_center: Centre point for spatial filter.
        :param spatial_radius: Radius for spatial filter.
        :param order: ``"newest"`` or ``"oldest"``.
        :param n_results: Maximum results to return.
        :param reference_time: Reference timestamp for *last_n_seconds*.
        :returns: Observations in chronological order.
        :rtype: List[ObservationNode]
        """
        query = "SELECT * FROM observations WHERE tier != 'archived'"
        params: list = []

        if time_range:
            query += " AND timestamp >= ? AND timestamp <= ?"
            params.extend(time_range)
        elif last_n_seconds and reference_time:
            query += " AND timestamp >= ?"
            params.append(reference_time - last_n_seconds)

        if layer:
            query += " AND layer_name = ?"
            params.append(layer)
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        if exclude_source_type:
            query += " AND source_type != ?"
            params.append(exclude_source_type)
        if spatial_center is not None and spatial_radius is not None:
            clause, dist_params = self._spatial_filter_sql(spatial_center, spatial_radius)
            query += " AND " + clause
            params.extend(dist_params)

        order_dir = "DESC" if order == "newest" else "ASC"
        query += " ORDER BY timestamp %s LIMIT ?" % order_dir
        params.append(n_results)

        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def get_latest_by_source_type(
        self,
        source_type: str,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, ObservationNode]:
        """Return the most recent observation per layer for a given source_type.

        :param source_type: Source type to filter on (e.g. ``"interoception"``).
        :param layers: Optional list of layer names to restrict results.
        :returns: Dict mapping layer_name to the latest ObservationNode.
        :rtype: Dict[str, ObservationNode]
        """
        query = """
            SELECT o.* FROM observations o
            INNER JOIN (
                SELECT layer_name, MAX(timestamp) AS max_ts
                FROM observations
                WHERE source_type = ?
        """
        params: list = [source_type]
        if layers:
            placeholders = ",".join("?" * len(layers))
            query += " AND layer_name IN (%s)" % placeholders
            params.extend(layers)
        query += """
                GROUP BY layer_name
            ) latest ON o.layer_name = latest.layer_name
                    AND o.timestamp = latest.max_ts
            WHERE o.source_type = ?
        """
        params.append(source_type)
        if layers:
            placeholders = ",".join("?" * len(layers))
            query += " AND o.layer_name IN (%s)" % placeholders
            params.extend(layers)

        rows = self._db.execute(query, params).fetchall()
        result: Dict[str, ObservationNode] = {}
        for r in rows:
            obs = self._row_to_observation(r)
            result[obs.layer_name] = obs
        return result

    def search_gists(
        self,
        query: str,
        n_results: int = 5,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[GistNode]:
        """Search consolidated gist summaries by meaning.

        :param query: Natural language query string.
        :param n_results: Maximum results to return.
        :param time_range: ``(start, end)`` timestamp bounds.
        :returns: Ranked list of gists.
        :rtype: List[GistNode]
        """
        query_emb = self.embedding_provider.embed([query])[0]
        if self._hnsw.get_current_count() == 0:
            return []

        fetch_k = min(n_results * 5, self._hnsw.get_current_count())
        labels, distances = self._hnsw.knn_query(query_emb.reshape(1, -1), k=fetch_k)

        candidate_ids = []
        for label in labels[0]:
            str_id = self._hnsw_id_map.get(int(label))
            if str_id:
                candidate_ids.append(str_id)

        if not candidate_ids:
            return []

        placeholders = ",".join("?" * len(candidate_ids))
        query_sql = "SELECT * FROM gists WHERE id IN (%s)" % placeholders
        params: list = list(candidate_ids)

        if time_range:
            query_sql += " AND time_start >= ? AND time_end <= ?"
            params.extend(time_range)

        rows = self._db.execute(query_sql, params).fetchall()
        results = [self._row_to_gist(r) for r in rows]
        id_order = {sid: i for i, sid in enumerate(candidate_ids)}
        results.sort(key=lambda g: id_order.get(g.id, 9999))
        return results[:n_results]

    def search_gists_by_area(
        self,
        center: np.ndarray,
        radius: float,
        n_results: int = 5,
    ) -> List[GistNode]:
        """Find gists whose centre falls within *radius* of *center*.

        :param center: 3D coordinate array.
        :param radius: Search radius in metres.
        :param n_results: Maximum results to return.
        :returns: Matching gists.
        :rtype: List[GistNode]
        """
        clause, dist_params = self._spatial_filter_sql(
            center, radius, x_col="cx", y_col="cy", z_col="cz",
        )
        rows = self._db.execute(
            "SELECT * FROM gists WHERE " + clause + " LIMIT ?",
            dist_params + [n_results],
        ).fetchall()
        return [self._row_to_gist(r) for r in rows]

    def get_recent_gists(
        self,
        time_after: Optional[float] = None,
        time_before: Optional[float] = None,
        layer: Optional[str] = None,
        order: str = "newest",
        n_results: int = 10,
    ) -> List[GistNode]:
        """Find gists overlapping a time range.

        A gist overlaps if its ``time_end >= time_after`` and
        ``time_start <= time_before``.

        :param time_after: Start of time window.
        :param time_before: End of time window.
        :param layer: Filter by layer name.
        :param order: ``"newest"`` or ``"oldest"``.
        :param n_results: Maximum results.
        :returns: Matching gists.
        :rtype: List[GistNode]
        """
        query = "SELECT * FROM gists WHERE 1=1"
        params: list = []

        if time_after is not None:
            query += " AND time_end >= ?"
            params.append(time_after)
        if time_before is not None:
            query += " AND time_start <= ?"
            params.append(time_before)
        if layer:
            query += " AND layer_name = ?"
            params.append(layer)

        order_col = "time_end" if order == "newest" else "time_start"
        order_dir = "DESC" if order == "newest" else "ASC"
        query += f" ORDER BY {order_col} {order_dir} LIMIT ?"
        params.append(n_results)

        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_gist(r) for r in rows]

    def get_observations_for_consolidation(
        self,
        older_than: float,
        tier: str = Tier.SHORT_TERM.value,
    ) -> List[ObservationNode]:
        """Fetch observations eligible for time-window consolidation.

        :param older_than: Timestamp cutoff; observations older than this are returned.
        :param tier: Only consider observations in this tier.
        :returns: Observations ordered by timestamp.
        :rtype: List[ObservationNode]
        """
        rows = self._db.execute(
            "SELECT * FROM observations WHERE tier = ? AND timestamp < ? ORDER BY timestamp",
            (tier, older_than),
        ).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> List[Edge]:
        query = "SELECT * FROM edges WHERE 1=1"
        params: list = []
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)
        if target_id:
            query += " AND target_id = ?"
            params.append(target_id)
        if edge_type:
            query += " AND edge_type = ?"
            params.append(edge_type.value)
        rows = self._db.execute(query, params).fetchall()
        return [Edge(
            id=r["id"], source_id=r["source_id"], target_id=r["target_id"],
            edge_type=EdgeType(r["edge_type"]), metadata=json.loads(r["metadata"]),
        ) for r in rows]

    def count_observations(self, tier: Optional[str] = None) -> int:
        if tier:
            return self._db.execute("SELECT COUNT(*) FROM observations WHERE tier = ?", (tier,)).fetchone()[0]
        return self._db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    def save(self) -> None:
        """Persist HNSW index to disk."""
        self._hnsw.save_index(str(self._hnsw_path))
        self._db.commit()

    def close(self) -> None:
        self.save()
        self._db.close()
