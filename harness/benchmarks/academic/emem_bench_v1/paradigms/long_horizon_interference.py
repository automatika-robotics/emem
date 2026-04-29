"""Long-horizon interference paradigm generator.

Underwood & Postman (1960) is the canonical reference for
retroactive interference (RI): new learning impairs recall of old
learning. The classic paradigm — study list A → study list B →
recall A — shows that performance on A drops as a function of how
recently and how similarly B was learned.

For an embodied robot deployed across multiple days the analogue
is: encode House Alpha on day 1 → time passes → encode House Beta
(another similar-shape house) → probe items from A. A robust memory
architecture should preserve A's *distinctive* content (items
genuinely seen in A, not in B) through both the temporal gap and
the interfering exposure.

This paradigm carries the paper's consolidation-as-quality claim.
The user observed earlier that consolidation isn't just about
storage pressure — it's about extracting invariants and discarding
redundancy so retrieval stays *actionable* under repeated similar
exposures. LHI is the diagnostic for that property.

Schedule shape::

    IngestPhase  — house A trajectory, prefix "In house Alpha: "
    AdvanceClock — gap_seconds (default 86400s = 1 day),
                   run_maintenance=True
                   ← consolidation + archival fires here; A's
                     verbatim text is consolidated into gists,
                     possibly archived depending on policy
    IngestPhase  — house B trajectory, prefix "In house Beta: "
    ProbePhase   — "Earlier you toured House Alpha. Did you see a
                    <item> in House Alpha?"

Two probe-classes per pair:

  * A-only items (unique to A's ProcTHOR object inventory) → yes
  * B-only items (unique to B's inventory) → no

A-only-no errors diagnose forgetting under consolidation pressure;
B-only-yes errors diagnose cross-episode confusion (RI proper). The
two failure modes carry different architectural implications.

Item selection is deterministic (no LLM picker): the unique-set
arithmetic over ProcTHOR ground truth fully constrains the
candidate pool, and no creative wording is needed.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
    ParadigmGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import (
    _indefinite_article,
    _probe_for_question,
    _sanitize_id_fragment,
)

log = logging.getLogger(__name__)

ELIGIBLE_ROOMS: Set[str] = {"Kitchen", "Bedroom", "Bathroom", "LivingRoom"}

ITEM_ORIGIN_A = "A_only"
ITEM_ORIGIN_B = "B_only"

DEFAULT_GAP_SECONDS = 86400  # 1 day


def _question_id(pair_id: str, probe_object: str, origin: str) -> str:
    """Stable question id for one (pair, item, origin) candidate."""
    return (
        f"lhi_{_sanitize_id_fragment(pair_id)}_"
        f"{_sanitize_id_fragment(probe_object)}_"
        f"{_sanitize_id_fragment(origin)}"
    )


def _objects_in_house(
    house: Dict[str, Any],
) -> Set[str]:
    """Return the set of ProcTHOR object types present anywhere in the house."""
    from harness.environments.procthor_utils import objects_by_room

    out: Set[str] = set()
    for types in objects_by_room(house).values():
        for t in types:
            out.add(t)
    return out


class LongHorizonInterferenceGenerator(ParadigmGenerator):
    """Emit long-horizon-interference probes across pairs of houses.

    :param house_lookup: ``int -> house_dict`` from
        ``prior.load_dataset("procthor-10k")``.
    :param target_total: Global cap on emitted candidates.
    :param gap_seconds: Virtual time gap between A and B encoding
        (with maintenance firing at the gap end). Defaults to 1 day.
    :param exclude_probes: Optional set of probe items to avoid
        (supplemental runs across pairs; keyed by ``"__lhi__"``).
    """

    name = "long_horizon_interference"

    def __init__(
        self,
        house_lookup: Callable[[int], Optional[Dict[str, Any]]],
        target_total: int = 120,
        gap_seconds: int = DEFAULT_GAP_SECONDS,
        exclude_probes: Optional[Dict[str, Set[str]]] = None,
        # Compatibility shim: cli factory passes ``llm_chat`` first
        # for every paradigm. LHI doesn't need an LLM picker.
        llm_chat: Optional[Callable[[str], str]] = None,
    ):
        del llm_chat
        self._house_lookup = house_lookup
        self._target_total = target_total
        self._gap_seconds = gap_seconds
        self._excluded: Dict[str, Set[str]] = {
            grp: {p.lower() for p in probes}
            for grp, probes in (exclude_probes or {}).items()
        }

    def generate(
        self,
        scenes: List[Dict[str, Any]],
        *,
        n_per_scene: int = 12,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Produce LHI candidates by pairing scenes sequentially.

        ``n_per_scene`` is the *probe-question* budget per pair (not
        per single scene); 6 A-only + 6 B-only = 12 by default.
        Scenes are paired sequentially (0+1, 2+3, …, 18+19), so 20
        scenes → 10 pairs.
        """
        del seed
        out: List[CandidateQuestion] = []
        per_origin = max(1, n_per_scene // 2)
        # Pair scenes sequentially; drop a stray odd scene if any.
        for i in range(0, len(scenes) - 1, 2):
            if len(out) >= self._target_total:
                break
            scene_a = scenes[i]
            scene_b = scenes[i + 1]
            out.extend(
                self._candidates_for_pair(scene_a, scene_b, per_origin=per_origin)
            )
        return out[: self._target_total]

    def scene_units(self, scenes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group scenes into pairs (LHI's natural unit)."""
        units: List[List[Dict[str, Any]]] = []
        for i in range(0, len(scenes) - 1, 2):
            units.append([scenes[i], scenes[i + 1]])
        return units

    def _candidates_for_pair(
        self,
        scene_a: Dict[str, Any],
        scene_b: Dict[str, Any],
        per_origin: int,
    ) -> List[CandidateQuestion]:
        """Pick A-only and B-only items for one pair; emit probe pairs."""
        objs_a = self._objects_for_scene(scene_a)
        objs_b = self._objects_for_scene(scene_b)
        if not objs_a or not objs_b:
            return []

        a_only = sorted(objs_a - objs_b)
        b_only = sorted(objs_b - objs_a)
        if not a_only or not b_only:
            log.info(
                "lhi pair (%s, %s): a_only=%d b_only=%d — at least one empty, skipping",
                scene_a.get("sample_id"),
                scene_b.get("sample_id"),
                len(a_only),
                len(b_only),
            )
            return []

        excluded = self._excluded.setdefault("__lhi__", set())
        out: List[CandidateQuestion] = []
        out.extend(
            self._build_origin_candidates(
                scene_a=scene_a,
                scene_b=scene_b,
                items=a_only,
                origin=ITEM_ORIGIN_A,
                expected_answer="yes",
                budget=per_origin,
                excluded=excluded,
                a_full=sorted(objs_a),
                b_full=sorted(objs_b),
            )
        )
        out.extend(
            self._build_origin_candidates(
                scene_a=scene_a,
                scene_b=scene_b,
                items=b_only,
                origin=ITEM_ORIGIN_B,
                expected_answer="no",
                budget=per_origin,
                excluded=excluded,
                a_full=sorted(objs_a),
                b_full=sorted(objs_b),
            )
        )
        return out

    def _objects_for_scene(self, scene: Dict[str, Any]) -> Set[str]:
        """Return the ProcTHOR object types in the scene's house, eligible rooms only."""
        sample_id = scene.get("sample_id")
        dataset_idx = scene.get("procthor_dataset_index")
        if dataset_idx is None:
            log.warning("scene %s has no procthor_dataset_index", sample_id)
            return set()
        house = self._house_lookup(int(dataset_idx))
        if house is None:
            log.warning("scene %s: house_lookup returned None", sample_id)
            return set()

        from harness.environments.procthor_utils import objects_by_room

        gt = objects_by_room(house)
        out: Set[str] = set()
        for room, types in gt.items():
            if room in ELIGIBLE_ROOMS:
                out.update(types)
        return out

    def _build_origin_candidates(
        self,
        *,
        scene_a: Dict[str, Any],
        scene_b: Dict[str, Any],
        items: List[str],
        origin: str,
        expected_answer: str,
        budget: int,
        excluded: Set[str],
        a_full: List[str],
        b_full: List[str],
    ) -> List[CandidateQuestion]:
        """Build up to ``budget`` candidates from ``items`` for one origin."""
        kept: List[CandidateQuestion] = []
        for item_type in items:
            pl = _probe_for_question(item_type)
            if pl in excluded:
                continue
            kept.append(
                self._build_candidate(
                    scene_a=scene_a,
                    scene_b=scene_b,
                    probe_object=item_type,
                    origin=origin,
                    expected_answer=expected_answer,
                    a_full=a_full,
                    b_full=b_full,
                )
            )
            excluded.add(pl)
            if len(kept) >= budget:
                break
        return kept

    def _build_candidate(
        self,
        *,
        scene_a: Dict[str, Any],
        scene_b: Dict[str, Any],
        probe_object: str,
        origin: str,
        expected_answer: str,
        a_full: List[str],
        b_full: List[str],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion."""
        sid_a = str(scene_a.get("sample_id", "alpha"))
        sid_b = str(scene_b.get("sample_id", "beta"))
        pair_id = f"{sid_a}+{sid_b}"
        probe_lower = _probe_for_question(probe_object)
        article = _indefinite_article(probe_lower)
        question = (
            f"Earlier you toured House Alpha. Did you see {article} "
            f"{probe_lower} in House Alpha?"
        )
        return CandidateQuestion(
            question_id=_question_id(pair_id, probe_lower, origin),
            question=question,
            answer=expected_answer,
            category="long_horizon_interference",
            paradigm=self.name,
            scene_ids=[sid_a, sid_b],
            tools_expected=["semantic_search", "entity_query", "episode_summary"],
            paradigm_metadata={
                "probe_object": probe_lower,
                "item_origin": origin,
                "house_alpha_id": sid_a,
                "house_beta_id": sid_b,
                "gap_seconds": self._gap_seconds,
                "label_to_sample_id": {"Alpha": sid_a, "Beta": sid_b},
                "ground_truth_objects_in_alpha": a_full,
                "ground_truth_objects_in_beta": b_full,
            },
            generator_notes=(
                f"Auto-generated long-horizon-interference probe. The "
                f"{probe_lower} is unique to House {'Alpha' if origin == ITEM_ORIGIN_A else 'Beta'} "
                f"of this pair; the agent encoded Alpha, then waited "
                f"{self._gap_seconds}s with maintenance, then encoded "
                f"Beta; the correct answer is {expected_answer!r}."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Pre-filter rubric for LHI candidates."""
        return (
            "You are pre-filtering benchmark questions before a human "
            "reviewer sees them. Your job is to catch obviously-bad "
            "questions — NOT to do the human's precision filtering.\n\n"
            "The questions are long-horizon-interference probes asking "
            "'Earlier you toured House Alpha. Did you see a <item> in "
            "House Alpha?' The correct answer is 'yes' if the item "
            "was unique to Alpha; 'no' if it was unique to Beta and "
            "the test is whether the agent confuses the two houses.\n\n"
            "Answer NO if the candidate clearly fails ANY of these "
            "criteria:\n"
            "  * STRUCTURAL: a wall, floor, ceiling, doorway, "
            "window, light fixture.\n"
            "  * ABSTRACT CATEGORY: 'decor', 'utensils', "
            "'furniture'.\n"
            "  * NONSENSICAL: doesn't make sense as a discrete "
            "perceived object.\n\n"
            "Answer YES when the probe item is a discrete, concrete, "
            "memorable object whose presence in one house but not "
            "the other is a clean test of cross-episode memory.\n\n"
            "Answer AMBIGUOUS when borderline.\n\n"
            "Answer with exactly one word: yes, ambiguous, or no."
        )
