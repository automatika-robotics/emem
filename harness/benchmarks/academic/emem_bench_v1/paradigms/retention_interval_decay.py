"""Retention-interval-decay paradigm generator.

Ebbinghaus' forgetting curve (1885), the canonical demonstration that
recall probability decays with time-since-encoding. The signature
shape is a steep early drop followed by an asymptote toward chance.

For an embodied robot the analogue is multi-day / multi-week
deployment: a system that ingests an episode and is then probed
hours, days, or weeks later should retain the salient items at a
predictable, measurable rate. The schedule shape is::

    IngestPhase(scene)
    AdvanceClockPhase(1h, maintenance=True)
    ProbePhase(at_time=t+1h, "did you see X?")
    AdvanceClockPhase(1d, maintenance=True)
    ProbePhase(at_time=t+1d, "did you see X?")
    AdvanceClockPhase(1wk, maintenance=True)
    ProbePhase(at_time=t+1wk, "did you see X?")

Each ``maintenance=True`` step fires the consolidation +
archival pair, so the probe at +1d sees memory state after a day's
worth of consolidation/archival, not after raw ingestion.

Probe form: ``"Did you see the <item>?"`` with ground-truth answer
``"yes"`` (the item was genuinely encoded). The benchmark metric is
the per-delay-bucket recall rate; a passing memory architecture
should produce a monotonically-non-increasing curve across the
buckets, with the slope diagnosing how aggressive the consolidation
+ archival pipeline is.

Items are drawn from objects unique to a single room of the scene
(same machinery as pattern_completion, so each probe has clean
ground truth). Each picked item emits one candidate *per delay
bucket* — 4 items × 3 delays per scene → 12 probes per scene.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
    ParadigmGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import (
    _parse_absent_items,
    _probe_for_question,
    _sanitize_id_fragment,
)

log = logging.getLogger(__name__)

ELIGIBLE_ROOMS: Set[str] = {"Kitchen", "Bedroom", "Bathroom", "LivingRoom"}

_ROOM_PHRASE: Dict[str, str] = {
    "Kitchen": "kitchen",
    "Bedroom": "bedroom",
    "Bathroom": "bathroom",
    "LivingRoom": "living room",
}

# Delay buckets the schedule probes at. Six buckets span two regimes:
# calibration (1h, 1d, 1wk) — the deployment-relevant horizons over
# which retention should be flat-and-high — and long-horizon stress
# (1mo, 6mo, 1yr) — virtual horizons that probe the consolidation +
# archival policy's behaviour at scales the architecture was never
# directly tested at. A flat curve across all six is the design-
# intended result; a cliff at any single bucket diagnoses an
# over-aggressive archival threshold; a gradual slope suggests
# non-deterministic archival.
RETENTION_DELAYS: Tuple[Tuple[str, int], ...] = (
    ("1h", 3600),
    ("1d", 86400),
    ("1wk", 604800),
    ("1mo", 2592000),  # 30 days
    ("6mo", 15552000),  # 180 days
    ("1yr", 31536000),  # 365 days
)


_PICKER_PROMPT = """\
You are helping build a cognitive-psychology benchmark for embodied \
robots. The retention-interval-decay paradigm tests whether the \
system still recalls a perceived object after a long delay (hours, \
days, weeks).

Below is the simulator's ground-truth list of objects that appear \
in EXACTLY ONE room of the scene the robot toured — i.e., the \
answer to "did the robot see this object" is unambiguously YES \
because the simulator literally placed it there.

== Objects unique to one room of the house ==
{object_list}
{avoid_block}
Your task: pick exactly {n} objects from the list above that make \
GOOD retention probes. A good probe is one where:

  * The object name is concrete and discrete (a thing the robot's \
perception layers would emit as a clean noun), not a structural \
item (wall, floor, ceiling), abstract category ('decor', 'utensils'), \
or a duplicate of items already picked.
  * The object is memorable enough to anchor a retention test — \
distinctive items beat ubiquitous ones.

Return ONLY a JSON array of strings, no explanation, no markdown \
fences. Each string must be copied verbatim from the list above. \
Example format:
["Alarm Clock", "Painting", "Toaster"]
"""

_AVOID_BLOCK = """

You have already picked these items for retention probes in this \
dataset — DO NOT pick any of them again:
{avoid_list}
"""


def _question_id(sample_id: str, probe_object: str, delay_bucket: str) -> str:
    """Stable question id for a single (item, delay) candidate."""
    return (
        f"retdec_{_sanitize_id_fragment(sample_id)}_"
        f"{_sanitize_id_fragment(probe_object)}_"
        f"{_sanitize_id_fragment(delay_bucket)}"
    )


def _bullet_list(items: List[Tuple[str, str]]) -> str:
    """Render a (room, type) list as bullets, room-tagged."""
    if not items:
        return "  (none)"
    return "\n".join(f"  - {t}  (in {_ROOM_PHRASE[r]})" for r, t in items)


class RetentionIntervalDecayGenerator(ParadigmGenerator):
    """Emit retention-interval-decay probes for each scene.

    :param llm_chat: Pluggable ``(prompt: str) -> str`` callable.
    :param house_lookup: ``int -> house_dict`` from
        ``prior.load_dataset("procthor-10k")``.
    :param target_total: Global cap on emitted candidates.
    :param exclude_probes: Optional per-room set of probe objects to
        avoid (supplemental runs across scenes).
    """

    name = "retention_interval_decay"

    def __init__(
        self,
        llm_chat: Callable[[str], str],
        house_lookup: Callable[[int], Optional[Dict[str, Any]]],
        target_total: int = 480,
        exclude_probes: Optional[Dict[str, Set[str]]] = None,
    ):
        self._llm_chat = llm_chat
        self._house_lookup = house_lookup
        self._target_total = target_total
        self._excluded: Dict[str, Set[str]] = {
            room: {p.lower() for p in probes}
            for room, probes in (exclude_probes or {}).items()
        }

    def generate(
        self,
        scenes: List[Dict[str, Any]],
        *,
        n_per_scene: int = 24,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Produce retention-decay candidates across ``scenes``.

        ``n_per_scene`` is the *probe-question* budget — at 6 delay
        buckets per item, ``n_per_scene=24`` picks 4 items per scene.
        """
        del seed
        out: List[CandidateQuestion] = []
        items_per_scene = max(1, n_per_scene // len(RETENTION_DELAYS))
        for scene in scenes:
            if len(out) >= self._target_total:
                break
            out.extend(
                self._candidates_for_scene(scene, items_per_scene=items_per_scene)
            )
        return out[: self._target_total]

    def _candidates_for_scene(
        self, scene: Dict[str, Any], items_per_scene: int
    ) -> List[CandidateQuestion]:
        """Pick items for one scene; emit one candidate per item × delay."""
        unique_per_room = self._unique_per_room(scene)
        if not unique_per_room:
            return []

        # Flatten to a (room, type) candidate pool. Keeps ordering
        # stable so the picker prompt is deterministic across runs.
        pool: List[Tuple[str, str]] = []
        for room in sorted(unique_per_room):
            for t in unique_per_room[room]:
                pool.append((room, t))
        if not pool:
            return []

        excluded = self._excluded.setdefault("__retdec__", set())
        over_request = items_per_scene * 3
        picks = self._request_picks(pool, excluded, n=over_request)

        type_to_room = {t.lower(): r for r, t in pool}
        out: List[CandidateQuestion] = []
        kept_items = 0
        for pick in picks:
            pl = pick.strip().lower()
            if not pl or pl in excluded:
                continue
            if pl not in type_to_room:
                log.debug(
                    "retention_interval_decay: dropping hallucinated pick %r",
                    pick,
                )
                continue
            room_type = type_to_room[pl]
            for delay_bucket, delay_seconds in RETENTION_DELAYS:
                out.append(
                    self._build_candidate(
                        scene=scene,
                        room_type=room_type,
                        probe_object=pick,
                        delay_bucket=delay_bucket,
                        delay_seconds=delay_seconds,
                        ground_truth_objects=unique_per_room[room_type],
                    )
                )
            excluded.add(pl)
            kept_items += 1
            if kept_items >= items_per_scene:
                break
        return out

    def _unique_per_room(self, scene: Dict[str, Any]) -> Dict[str, List[str]]:
        """Return ``{room_type: [unique_object_types]}`` for one scene.

        Same logic as pattern_completion's helper — kept inline rather
        than imported to avoid coupling paradigms together. Empty dict
        if the scene's house can't be loaded or has no unique-to-one-
        room objects.
        """
        sample_id = scene.get("sample_id")
        dataset_idx = scene.get("procthor_dataset_index")
        if dataset_idx is None:
            log.warning("scene %s has no procthor_dataset_index; skipping", sample_id)
            return {}
        house = self._house_lookup(int(dataset_idx))
        if house is None:
            log.warning(
                "scene %s: house_lookup(%d) returned None; skipping",
                sample_id,
                dataset_idx,
            )
            return {}

        from harness.environments.procthor_utils import objects_by_room

        gt = objects_by_room(house)
        eligible = {r: gt.get(r, []) for r in ELIGIBLE_ROOMS if gt.get(r)}
        if not eligible:
            log.info("scene %s: no eligible rooms; skipping", sample_id)
            return {}

        unique: Dict[str, List[str]] = {}
        for room, types in eligible.items():
            others: Set[str] = set()
            for other_room, other_types in eligible.items():
                if other_room != room:
                    others.update(other_types)
            uniq = [t for t in dict.fromkeys(types) if t not in others]
            if uniq:
                unique[room] = uniq

        if not unique:
            log.info(
                "scene %s: no objects unique to a single room; skipping",
                sample_id,
            )
        return unique

    def _request_picks(
        self,
        pool: List[Tuple[str, str]],
        excluded: Set[str],
        n: int,
    ) -> List[str]:
        """Ask the LLM for ``n`` memorable items from the candidate pool."""
        avoid_block = (
            _AVOID_BLOCK.format(
                avoid_list="\n".join(f"  - {p}" for p in sorted(excluded))
            )
            if excluded
            else ""
        )
        prompt = _PICKER_PROMPT.format(
            object_list=_bullet_list(pool),
            avoid_block=avoid_block,
            n=n,
        )
        raw = self._llm_chat(prompt)
        items = _parse_absent_items(raw)
        if not items:
            log.warning(
                "retention_interval_decay: no items parsed; raw[:120]=%r",
                (raw or "")[:120],
            )
        return items[:n]

    def _build_candidate(
        self,
        *,
        scene: Dict[str, Any],
        room_type: str,
        probe_object: str,
        delay_bucket: str,
        delay_seconds: int,
        ground_truth_objects: List[str],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion for an (item, delay) pair."""
        sample_id = str(scene.get("sample_id", "unknown"))
        probe_lower = _probe_for_question(probe_object)
        question = f"Did you see the {probe_lower}?"
        return CandidateQuestion(
            question_id=_question_id(sample_id, probe_lower, delay_bucket),
            question=question,
            answer="yes",
            category="retention_interval_decay",
            paradigm=self.name,
            scene_ids=[sample_id],
            tools_expected=["semantic_search", "entity_query"],
            paradigm_metadata={
                "room_type": room_type,
                "probe_object": probe_lower,
                "retention_delay_bucket": delay_bucket,
                "retention_delay_seconds": delay_seconds,
                "ground_truth_objects_in_room": sorted(set(ground_truth_objects)),
            },
            generator_notes=(
                f"Auto-generated retention-decay probe. The {probe_lower} "
                f"was perceived by the agent during its tour of the "
                f"{_ROOM_PHRASE[room_type]}; this candidate probes recall "
                f"after a {delay_bucket} delay."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Pre-filter rubric for retention-decay candidates."""
        return (
            "You are pre-filtering benchmark questions before a human "
            "reviewer sees them. Your job is to catch obviously-bad "
            "questions — NOT to do the human's precision filtering.\n\n"
            "The questions are retention-decay probes asking 'Did you "
            "see the <item>?' with ground-truth answer 'yes' (the "
            "robot was confirmed to have observed the item). The "
            "benchmark measures whether the agent still recalls the "
            "item after a delay (1 hour / 1 day / 1 week).\n\n"
            "Answer NO if the probe item clearly fails ANY of these "
            "criteria:\n"
            "  * STRUCTURAL: a wall, floor, ceiling, doorway, "
            "window, light fixture.\n"
            "  * ABSTRACT CATEGORY: 'decor', 'dishware', 'appliance', "
            "'furniture', 'utensils'.\n"
            "  * NONSENSICAL: doesn't make sense as a discrete "
            "perceived object.\n\n"
            "Answer YES when the probe item is a discrete, concrete, "
            "memorable object that a robot's perception layers would "
            "name cleanly.\n\n"
            "Answer AMBIGUOUS when the item is borderline.\n\n"
            "Answer with exactly one word: yes, ambiguous, or no."
        )
