import time
from typing import Dict, List, Optional, Protocol

import numpy as np

from emem.config import SpatioTemporalMemoryConfig
from emem.store import MemoryStore
from emem.types import Edge, EdgeType, EntityNode, GistNode, ObservationNode, Tier


class LLMClient(Protocol):
    def summarize(self, texts: List[str]) -> str:
        """Summarize a list of text observations into a single gist string.

        :param texts: Observation texts to summarize.
        :returns: Consolidated summary.
        :rtype: str
        """
        ...


class ConcatenationSummarizer:
    """Fallback summarizer that joins texts with ``|``."""

    def summarize(self, texts: List[str]) -> str:
        return " | ".join(texts)

    def synthesize(self, layer_texts: Dict[str, List[str]]) -> str:
        """Synthesize texts grouped by layer into a structured summary."""
        parts = []
        for layer_name in sorted(layer_texts.keys()):
            parts.append(f"[{layer_name}] {' | '.join(layer_texts[layer_name])}")
        return " || ".join(parts)


def _common_layer(observations: List[ObservationNode], default: Optional[str] = None) -> Optional[str]:
    """Return the layer name if all observations share one, else *default*."""
    layers = set(obs.layer_name for obs in observations)
    return next(iter(layers)) if len(layers) == 1 else default


class ConsolidationEngine:
    """Handles memory consolidation: gist generation, tier promotion, and
    archival.

    Triggered by episode completion
    (:meth:`consolidate_episode`) or time-window rollover
    (:meth:`consolidate_time_window`).
    """

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[SpatioTemporalMemoryConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.store = store
        self.config = config or SpatioTemporalMemoryConfig()
        self._summarizer = llm_client or ConcatenationSummarizer()

    def consolidate_episode(self, episode_id: str) -> Optional[str]:
        """Generate a gist from all observations in an episode.

        :param episode_id: Episode to consolidate.
        :returns: Created gist ID, or ``None`` if the episode had no observations.
        :rtype: Optional[str]
        """
        observations = self.store.get_episode_observations(episode_id)
        if not observations:
            return None

        gist = self._create_gist_from_observations(observations, episode_id=episode_id)
        gist_id = self.store.add_gist(gist)

        self._extract_and_merge_entities(observations)

        self.store.update_observation_tiers(
            [obs.id for obs in observations], Tier.LONG_TERM.value, drop_text=False,
        )

        return gist_id

    def consolidate_time_window(self, reference_time: Optional[float] = None) -> List[str]:
        """Cluster old short-term observations by spatial proximity and
        generate gists.

        :param reference_time: Current timestamp.  Defaults to ``time.time()``.
        :returns: List of created gist IDs.
        :rtype: List[str]
        """
        ref_time = reference_time or time.time()
        cutoff = ref_time - self.config.consolidation_window

        candidates = self.store.get_observations_for_consolidation(older_than=cutoff)
        if not candidates:
            return []

        clusters = self._spatial_cluster(candidates)
        gist_ids = []

        for cluster_obs in clusters:
            if len(cluster_obs) < self.config.consolidation_min_samples:
                continue
            gist = self._create_gist_from_observations(cluster_obs)
            gist_id = self.store.add_gist(gist)
            gist_ids.append(gist_id)

            self._extract_and_merge_entities(cluster_obs)

            self.store.update_observation_tiers(
                [obs.id for obs in cluster_obs], Tier.LONG_TERM.value, drop_text=False,
            )

        return gist_ids

    def archive_long_term(self, reference_time: Optional[float] = None) -> int:
        """Archive long-term observations older than ``archive_after_seconds``.

        :param reference_time: Current timestamp.  Defaults to ``time.time()``.
        :returns: Number of observations archived.
        :rtype: int
        """
        ref_time = reference_time or time.time()
        cutoff = ref_time - self.config.archive_after_seconds
        candidates = self.store.get_observations_for_consolidation(
            older_than=cutoff, tier=Tier.LONG_TERM.value,
        )
        if not candidates:
            return 0
        self.store.update_observation_tiers(
            [obs.id for obs in candidates], Tier.ARCHIVED.value, drop_text=True,
        )
        return len(candidates)

    def _spatial_cluster(self, observations: List[ObservationNode]) -> List[List[ObservationNode]]:
        if len(observations) < self.config.consolidation_min_samples:
            return [observations] if observations else []

        from sklearn.cluster import DBSCAN

        coords = np.array([obs.coordinates[:2] for obs in observations])
        clustering = DBSCAN(
            eps=self.config.consolidation_spatial_eps,
            min_samples=self.config.consolidation_min_samples,
        ).fit(coords)

        clusters: Dict[int, List[ObservationNode]] = {}
        for obs, label in zip(observations, clustering.labels_):
            if label == -1:
                continue  # Noise — leave in short-term
            clusters.setdefault(label, []).append(obs)

        return list(clusters.values())

    def _create_gist_from_observations(
        self,
        observations: List[ObservationNode],
        episode_id: Optional[str] = None,
    ) -> GistNode:
        # Determine layer — use common layer if all same, else None
        layer_name = _common_layer(observations)
        is_multi_layer = layer_name is None and len(observations) > 0

        if is_multi_layer and hasattr(self._summarizer, "synthesize"):
            layer_texts: Dict[str, List[str]] = {}
            for obs in observations:
                if obs.text:
                    layer_texts.setdefault(obs.layer_name, []).append(obs.text)
            gist_text = self._summarizer.synthesize(layer_texts) if layer_texts else ""
        else:
            texts = [obs.text for obs in observations if obs.text]
            gist_text = self._summarizer.summarize(texts) if texts else ""

        coords = np.array([obs.coordinates for obs in observations])
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        radius = float(distances.max()) if len(distances) > 0 else 0.0

        timestamps = [obs.timestamp for obs in observations]

        return GistNode(
            text=gist_text,
            center_position=center,
            radius=radius,
            time_start=min(timestamps),
            time_end=max(timestamps),
            source_observation_count=len(observations),
            source_observation_ids=[obs.id for obs in observations],
            layer_name=layer_name,
            episode_id=episode_id,
        )

    def extract_entities_from_observations(self, observations: List[ObservationNode]) -> List[str]:
        """Extract entities from observations and mark them as processed.

        Safe to call multiple times — already-extracted observations are skipped.

        :param observations: Observations to extract entities from.
        :returns: List of entity IDs created or merged.
        :rtype: List[str]
        """
        entity_ids = self._extract_and_merge_entities(observations)
        return entity_ids

    def _extract_and_merge_entities(self, observations: List[ObservationNode]) -> List[str]:
        if not hasattr(self._summarizer, "extract_entities"):
            return []

        # Filter to only unprocessed observations
        unprocessed_ids = self.store.get_unextracted_obs_ids([obs.id for obs in observations])
        observations = [obs for obs in observations if obs.id in unprocessed_ids]
        if not observations:
            return []

        texts = [obs.text for obs in observations if obs.text]
        if not texts:
            return []

        raw_entities = self._summarizer.extract_entities(texts)
        if not raw_entities:
            return []

        coords = np.array([obs.coordinates for obs in observations])
        centroid = coords.mean(axis=0)
        timestamps = [obs.timestamp for obs in observations]

        entity_ids: List[str] = []
        edges: List[Edge] = []

        for raw in raw_entities:
            name = raw["name"]
            entity_type = raw.get("entity_type")
            conf = raw.get("confidence", 1.0)

            existing = self.store.find_matching_entity(name, centroid)
            if existing:
                existing.coordinates = centroid
                existing.last_seen = max(existing.last_seen, max(timestamps))
                existing.first_seen = min(existing.first_seen, min(timestamps))
                existing.observation_count += len(observations)
                existing.confidence = max(existing.confidence, conf)
                if entity_type and not existing.entity_type:
                    existing.entity_type = entity_type
                self.store.update_entity(existing)
                entity_ids.append(existing.id)
            else:
                layer_name = _common_layer(observations, default="default")
                entity = EntityNode(
                    name=name,
                    coordinates=centroid,
                    last_seen=max(timestamps),
                    first_seen=min(timestamps),
                    observation_count=len(observations),
                    confidence=conf,
                    entity_type=entity_type,
                    layer_name=layer_name,
                )
                self.store.add_entity(entity)
                entity_ids.append(entity.id)

            # OBSERVED_IN edges
            for obs in observations:
                edges.append(Edge(
                    source_id=entity_ids[-1],
                    target_id=obs.id,
                    edge_type=EdgeType.OBSERVED_IN,
                ))

        # COOCCURS_WITH edges between all entity pairs
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                edges.append(Edge(
                    source_id=entity_ids[i],
                    target_id=entity_ids[j],
                    edge_type=EdgeType.COOCCURS_WITH,
                ))

        self.store.add_edges(edges)
        self.store.mark_entities_extracted([obs.id for obs in observations])

        return entity_ids
