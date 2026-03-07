import time
from typing import Dict, List, Optional, Protocol

import numpy as np

from emem.config import SpatioTemporalMemoryConfig
from emem.store import MemoryStore
from emem.types import GistNode, ObservationNode, Tier


class LLMClient(Protocol):
    def summarize(self, texts):
        # type: (List[str]) -> str
        """Summarize a list of text observations into a single gist string.

        :param texts: Observation texts to summarize.
        :returns: Consolidated summary.
        :rtype: str
        """
        ...


class ConcatenationSummarizer:
    """Fallback summarizer that joins texts with ``|``."""

    def summarize(self, texts):
        # type: (List[str]) -> str
        return " | ".join(texts)


class ConsolidationEngine:
    """Handles memory consolidation: gist generation, tier promotion, and
    archival.

    Triggered by episode completion
    (:meth:`consolidate_episode`) or time-window rollover
    (:meth:`consolidate_time_window`).
    """

    def __init__(self, store, config=None, llm_client=None):
        # type: (MemoryStore, Optional[SpatioTemporalMemoryConfig], Optional[LLMClient]) -> None
        self.store = store
        self.config = config or SpatioTemporalMemoryConfig()
        self._summarizer = llm_client or ConcatenationSummarizer()

    def consolidate_episode(self, episode_id):
        # type: (str) -> Optional[str]
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

        self.store.update_observation_tiers(
            [obs.id for obs in observations], Tier.ARCHIVED.value, drop_text=True,
        )

        return gist_id

    def consolidate_time_window(self, reference_time=None):
        # type: (Optional[float]) -> List[str]
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

            self.store.update_observation_tiers(
                [obs.id for obs in cluster_obs], Tier.ARCHIVED.value, drop_text=True,
            )

        return gist_ids

    def _spatial_cluster(self, observations):
        # type: (List[ObservationNode]) -> List[List[ObservationNode]]
        if len(observations) < self.config.consolidation_min_samples:
            return [observations] if observations else []

        from sklearn.cluster import DBSCAN

        coords = np.array([obs.coordinates[:2] for obs in observations])
        clustering = DBSCAN(
            eps=self.config.consolidation_spatial_eps,
            min_samples=self.config.consolidation_min_samples,
        ).fit(coords)

        clusters = {}  # type: Dict[int, List[ObservationNode]]
        for obs, label in zip(observations, clustering.labels_):
            if label == -1:
                continue  # Noise — leave in short-term
            clusters.setdefault(label, []).append(obs)

        return list(clusters.values())

    def _create_gist_from_observations(self, observations, episode_id=None):
        # type: (List[ObservationNode], Optional[str]) -> GistNode
        texts = [obs.text for obs in observations if obs.text]
        gist_text = self._summarizer.summarize(texts) if texts else ""

        coords = np.array([obs.coordinates for obs in observations])
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        radius = float(distances.max()) if len(distances) > 0 else 0.0

        timestamps = [obs.timestamp for obs in observations]

        # Determine layer — use common layer if all same, else None
        layers = set(obs.layer_name for obs in observations)
        layer_name = layers.pop() if len(layers) == 1 else None

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
