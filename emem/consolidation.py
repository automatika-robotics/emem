import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

from emem.config import SpatioTemporalMemoryConfig
from emem.store import MemoryStore
from emem.types import Edge, EdgeType, EntityNode, GistNode, ObservationNode, Tier

log = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for an LLM-backed summarizer used during consolidation."""

    def summarize(self, texts: List[str]) -> str:
        """Summarize a list of text observations into a single gist string.

        :param texts: Observation texts to summarize.
        :returns: Consolidated summary.
        :rtype: str
        """
        ...

    def synthesize(self, layer_texts: Dict[str, List[str]]) -> str:
        """Synthesize observations grouped by perception layer into a
        structured summary.

        :param layer_texts: Mapping of layer name to observation texts.
        :returns: Cross-layer summary.
        :rtype: str
        """
        ...

    def extract_entities(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract named entities from observation texts.

        :param texts: Observation texts.
        :returns: List of dicts with keys ``name``, ``entity_type``, ``confidence``.
        :rtype: List[Dict[str, Any]]
        """
        ...


def _parse_entities(raw: str) -> List[Dict[str, Any]]:
    """Parse a JSON entity array from LLM output.

    Extracts the first ``[...]`` block from *raw* and returns a list of
    entity dicts with keys ``name``, ``entity_type``, ``confidence``,
    and ``observation_index`` (0-based, or ``None`` when the model did
    not supply an index).
    """
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        entities = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    out: List[Dict[str, Any]] = []
    for e in entities:
        if not (isinstance(e, dict) and e.get("name")):
            continue
        raw_idx = e.get("observation_index")
        obs_idx: Optional[int]
        if raw_idx is None:
            obs_idx = None
        else:
            try:
                obs_idx = int(raw_idx) - 1  # 1-based in prompt, 0-based internally
                if obs_idx < 0:
                    obs_idx = None
            except (TypeError, ValueError):
                obs_idx = None
        out.append({
            "name": str(e["name"]),
            "entity_type": e.get("entity_type"),
            "confidence": float(e.get("confidence") or 1.0),
            "observation_index": obs_idx,
        })
    return out


class InferenceLLMClient:
    """Wraps a model inference function into an :class:`LLMClient`.

    The function should accept a dict with at least::

        {"query": List[Dict], "temperature": float,
         "max_new_tokens": int, "stream": bool}

    and return a mapping with an ``"output"`` key containing the model's
    text response.  This matches the ``ModelClient.inference()`` interface
    in EmbodiedAgents.

    Provides ``summarize``, ``synthesize``, and ``extract_entities`` using
    the same prompts as the eMEM evaluation harness.

    Example::

        from emem.consolidation import InferenceLLMClient

        llm_client = InferenceLLMClient(ollama_client.inference)
        mem = SpatioTemporalMemory(llm_client=llm_client)

    :param inference_fn: Model inference callable.
    :param temperature: Sampling temperature for consolidation calls.
    :param max_new_tokens: Maximum tokens for consolidation calls.
    """

    def __init__(
        self,
        inference_fn: Callable,
        temperature: float = 0.3,
        max_new_tokens: int = 500,
    ):
        self._fn = inference_fn
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens

    def _chat(self, prompt: str) -> str:
        """Invoke the wrapped inference function and return cleaned text output."""
        result = self._fn({
            "query": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_new_tokens": self._max_new_tokens,
            "stream": False,
        })
        if result and result.get("output"):
            text = result["output"]
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return ""

    def summarize(self, texts: List[str]) -> str:
        """Summarize observations into a concise paragraph."""
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        return self._chat(
            "Summarize the following observations into a concise paragraph. "
            "Preserve spatial and temporal details.\n\n" + numbered
        )

    def synthesize(self, layer_texts: Dict[str, List[str]]) -> str:
        """Synthesize observations grouped by perception layer."""
        block = "\n".join(
            f"[{layer}]: {'; '.join(texts)}" for layer, texts in layer_texts.items()
        )
        return self._chat(
            "Synthesize the following observations grouped by perception layer "
            "into a coherent summary. Highlight agreements and "
            "contradictions.\n\n" + block
        )

    def extract_entities(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract named entities from observations.

        The prompt asks the model to tag each extracted entity with the
        1-based index of the observation it came from
        (``observation_index``) so the caller can link the entity to
        that specific observation. When an entity genuinely spans
        multiple observations the model is instructed to emit a
        separate record per observation.
        """
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        raw = self._chat(
            "Extract named entities (objects, places, people) from these "
            "observations. Return ONLY a JSON array where each element has "
            'keys: "name" (string), "entity_type" (string or null), '
            '"confidence" (float 0-1), "observation_index" (integer, '
            "1-based, indicating which numbered observation below this "
            "entity came from). If the same entity appears in multiple "
            "observations, emit one record per observation.\n\n" + numbered
        )
        return _parse_entities(raw)


class ConcatenationSummarizer:
    """Fallback summarizer that joins texts with ``|``."""

    def summarize(self, texts: List[str]) -> str:
        """Join *texts* with ``|`` separators."""
        return " | ".join(texts)

    def synthesize(self, layer_texts: Dict[str, List[str]]) -> str:
        """Join layer-grouped texts into a single concatenated summary string."""
        parts = []
        for layer_name in sorted(layer_texts.keys()):
            parts.append(f"[{layer_name}] {' | '.join(layer_texts[layer_name])}")
        return " || ".join(parts)

    def extract_entities(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Return an empty list — concatenation summarizer does not extract entities."""
        return []


def _common_layer(
    observations: List[ObservationNode], default: Optional[str] = None
) -> Optional[str]:
    """Return the layer name if all observations share one, else *default*."""
    layers = {obs.layer_name for obs in observations}
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

    def consolidate_episode(self, episode_id: str) -> List[str]:
        """Generate gists from observations in an episode, chunked by time.

        Observations are split into temporal chunks using
        ``consolidation_window``.  Each chunk produces its own gist, so long
        episodes get multiple focused summaries instead of one monolithic one.

        :param episode_id: Episode to consolidate.
        :returns: List of created gist IDs (empty if episode had no observations).
        :rtype: List[str]
        """
        observations = self.store.get_episode_observations(episode_id)
        if not observations:
            return []

        chunks = self._temporal_chunk(observations)
        gist_ids = []
        for chunk in chunks:
            gist = self._create_gist_from_observations(chunk, episode_id=episode_id)
            gist_id = self.store.add_gist(gist)
            gist_ids.append(gist_id)

        self._extract_and_merge_entities(observations)

        self.store.update_observation_tiers(
            [obs.id for obs in observations],
            Tier.LONG_TERM.value,
            drop_text=False,
        )

        return gist_ids

    def consolidate_time_window(
        self, reference_time: Optional[float] = None
    ) -> List[str]:
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
                [obs.id for obs in cluster_obs],
                Tier.LONG_TERM.value,
                drop_text=False,
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
            older_than=cutoff,
            tier=Tier.LONG_TERM.value,
        )
        if not candidates:
            return 0
        self.store.update_observation_tiers(
            [obs.id for obs in candidates],
            Tier.ARCHIVED.value,
            drop_text=True,
        )
        return len(candidates)

    def _temporal_chunk(
        self, observations: List[ObservationNode]
    ) -> List[List[ObservationNode]]:
        """Split observations into chunks separated by time gaps.

        A new chunk starts whenever the gap between consecutive observations
        exceeds ``consolidation_window``.  All chunks are returned regardless
        of size.

        :param observations: Observations to chunk (need not be sorted).
        :returns: List of temporally contiguous chunks.
        :rtype: List[List[ObservationNode]]
        """
        if not observations:
            return []

        sorted_obs = sorted(observations, key=lambda o: o.timestamp)
        chunks: List[List[ObservationNode]] = [[sorted_obs[0]]]

        for obs in sorted_obs[1:]:
            if (
                obs.timestamp - chunks[-1][-1].timestamp
                > self.config.consolidation_window
            ):
                chunks.append([obs])
            else:
                chunks[-1].append(obs)

        return chunks

    def _spatial_cluster(
        self, observations: List[ObservationNode]
    ) -> List[List[ObservationNode]]:
        """Cluster observations by 2D position using DBSCAN."""
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
        """Build a :class:`GistNode` summarizing *observations*."""
        # Determine layer — use common layer if all same, else None
        layer_name = _common_layer(observations)
        is_multi_layer = layer_name is None and len(observations) > 0

        if is_multi_layer:
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

    def extract_entities_from_observations(
        self, observations: List[ObservationNode]
    ) -> List[str]:
        """Extract entities from observations and mark them as processed.

        Safe to call multiple times — already-extracted observations are skipped.

        :param observations: Observations to extract entities from.
        :returns: List of entity IDs created or merged.
        :rtype: List[str]
        """
        entity_ids = self._extract_and_merge_entities(observations)
        return entity_ids

    def _extract_and_merge_entities(
        self, observations: List[ObservationNode]
    ) -> List[str]:
        """Extract entities from observations and merge them into the store.

        When the LLM supplies ``observation_index`` for each entity, the
        entity is linked only to that specific observation and its
        coordinates / timestamp come from that observation. When the
        LLM omits the index we fall back to the earlier batch-link
        behaviour (every entity linked to every observation in the
        batch, coordinates at the batch centroid) and emit a warning,
        so a non-compliant model is loud but not crashing.
        """
        # Filter to only unprocessed observations
        unprocessed_ids = self.store.get_unextracted_obs_ids([
            obs.id for obs in observations
        ])
        observations = [obs for obs in observations if obs.id in unprocessed_ids]
        if not observations:
            return []

        texts = [obs.text for obs in observations if obs.text]
        if not texts:
            return []

        raw_entities = self._summarizer.extract_entities(texts)
        if not raw_entities:
            return []

        text_to_obs: Dict[str, ObservationNode] = {
            obs.text: obs for obs in observations if obs.text
        }
        texts_order = [obs.text for obs in observations if obs.text]

        n_missing_idx = sum(
            1 for r in raw_entities if r.get("observation_index") is None
        )
        if n_missing_idx == len(raw_entities):
            log.warning(
                "entity-extractor returned no observation_index for any of "
                "%d entities; falling back to batch-link behaviour",
                len(raw_entities),
            )
        elif n_missing_idx > 0:
            log.warning(
                "entity-extractor omitted observation_index for %d/%d "
                "entities; those will fall back to batch-link",
                n_missing_idx,
                len(raw_entities),
            )

        entity_ids: List[str] = []
        per_entity_obs_idx: List[Optional[int]] = []
        edges: List[Edge] = []

        batch_centroid = np.array([obs.coordinates for obs in observations]).mean(
            axis=0
        )
        batch_timestamps = [obs.timestamp for obs in observations]

        for raw in raw_entities:
            name = raw["name"]
            entity_type = raw.get("entity_type")
            conf = raw.get("confidence", 1.0)
            obs_idx = raw.get("observation_index")

            if isinstance(obs_idx, int) and 0 <= obs_idx < len(texts_order):
                attributed_obs = text_to_obs[texts_order[obs_idx]]
                anchor_coords = attributed_obs.coordinates
                anchor_layer = attributed_obs.layer_name
                anchor_timestamp = attributed_obs.timestamp
                linked_obs_ids = [attributed_obs.id]
                contribution_count = 1
                context_embedding = attributed_obs.embedding
            else:
                # Fallback: batch-link (previous behaviour).
                attributed_obs = None
                anchor_coords = batch_centroid
                anchor_layer = _common_layer(observations, default="default")
                anchor_timestamp = max(batch_timestamps)
                linked_obs_ids = [obs.id for obs in observations]
                contribution_count = len(observations)
                context_embedding = None

            existing = self.store.find_matching_entity(
                name, anchor_coords, context_embedding=context_embedding
            )
            if existing:
                existing.coordinates = anchor_coords
                existing.last_seen = max(existing.last_seen, anchor_timestamp)
                existing.first_seen = min(
                    existing.first_seen,
                    attributed_obs.timestamp
                    if attributed_obs is not None
                    else min(batch_timestamps),
                )
                existing.observation_count += contribution_count
                existing.confidence = max(existing.confidence, conf)
                if entity_type and not existing.entity_type:
                    existing.entity_type = entity_type
                self.store.update_entity(existing)
                entity_ids.append(existing.id)
            else:
                entity = EntityNode(
                    name=name,
                    coordinates=anchor_coords,
                    last_seen=anchor_timestamp,
                    first_seen=(
                        attributed_obs.timestamp
                        if attributed_obs is not None
                        else min(batch_timestamps)
                    ),
                    observation_count=contribution_count,
                    confidence=conf,
                    entity_type=entity_type,
                    layer_name=anchor_layer,
                )
                self.store.add_entity(entity)
                entity_ids.append(entity.id)

            per_entity_obs_idx.append(obs_idx if isinstance(obs_idx, int) else None)

            # Flush OBSERVED_IN edges for this entity *before* the next
            # iteration so a subsequent candidate's context-similarity
            # probe can find the source observation.
            entity_edges = [
                Edge(
                    source_id=entity_ids[-1],
                    target_id=obs_id,
                    edge_type=EdgeType.OBSERVED_IN,
                )
                for obs_id in linked_obs_ids
            ]
            self.store.add_edges(entity_edges)

        # COOCCURS_WITH edges. With per-observation attribution, two
        # entities co-occur when they were extracted from the same
        # observation. When either entity lacks attribution we assume cooccurence
        # in batch.
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                idx_i = per_entity_obs_idx[i]
                idx_j = per_entity_obs_idx[j]
                if idx_i is not None and idx_j is not None:
                    if idx_i != idx_j:
                        continue
                edges.append(
                    Edge(
                        source_id=entity_ids[i],
                        target_id=entity_ids[j],
                        edge_type=EdgeType.COOCCURS_WITH,
                    )
                )

        self.store.add_edges(edges)
        self.store.mark_entities_extracted([obs.id for obs in observations])

        return entity_ids
