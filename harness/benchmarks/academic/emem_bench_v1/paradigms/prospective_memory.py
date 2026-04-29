"""Prospective-memory paradigm generator.

Brandimonte / Einstein / McDaniel (1996) framing: prospective memory
is *remembering to do something later*, distinct from retrospective
memory of past facts. The classic event-based paradigm tells the
subject "when you see X, do Y", embeds X in an unrelated ongoing
task, and measures whether Y is executed when X appears.

For an embodied robot prospective memory is the headline capability
for instruction-following deployment over multi-day horizons:
*"when you next visit the kitchen, check if the kettle is off"* is
the canonical user instruction. Without it, the robot can describe
its past but cannot act on instructions issued before the trigger
event happened.

Schedule shape::

    IngestPhase  — instruction encode (one observation, layer
                   ``instruction``)
    AdvanceClock — small gap (60s, no maintenance)
    IngestPhase  — scene trajectory; the trigger object appears
                   naturally somewhere mid-trajectory via the
                   object_detection layer
    ProbePhase   — "what were you instructed to do when you see X?"

We're testing *cued recall of an instruction across a distractor*,
not spontaneous PM retrieval (the probe is explicit, not
self-triggered). The same memory mechanism is exercised; the
difference is operational. Paper notes will be honest about this.

Trigger items come from objects unique to one room of the scene
(reusing pattern_completion's machinery) so the trigger fires in a
known, bounded set of frames.

Action vocabulary is small and bounded so the LLM judge can match
semantically equivalent answers cleanly; we deterministically rotate
through actions across items so the dataset spans the vocabulary
evenly.

No LLM picker is involved — items come from a constructed pool and
actions are templated. Generation runs in seconds.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
    ParadigmGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import (
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

# Bounded action vocabulary. The LLM judge matches semantic
# equivalence across paraphrases (e.g. "check it" ≈ "inspect it"),
# so the dataset can be small without losing coverage.
ACTIONS: Tuple[str, ...] = (
    "check it",
    "notify me",
    "take a picture",
    "do not touch it",
    "report back",
)


def _question_id(sample_id: str, probe_object: str, action_idx: int) -> str:
    """Stable question id for one (item, action) candidate."""
    return (
        f"prosmem_{_sanitize_id_fragment(sample_id)}_"
        f"{_sanitize_id_fragment(probe_object)}_a{action_idx}"
    )


class ProspectiveMemoryGenerator(ParadigmGenerator):
    """Emit prospective-memory probes for each scene.

    :param house_lookup: ``int -> house_dict`` from
        ``prior.load_dataset("procthor-10k")``.
    :param target_total: Global cap on emitted candidates.
    :param exclude_probes: Optional set of probe items to avoid
        (supplemental runs across scenes; keyed by ``"__prosmem__"``).
    """

    name = "prospective_memory"

    def __init__(
        self,
        house_lookup: Callable[[int], Optional[Dict[str, Any]]],
        target_total: int = 120,
        exclude_probes: Optional[Dict[str, Set[str]]] = None,
        # Compatibility shim: cli factory passes ``llm_chat`` as the
        # first positional arg for every paradigm. We accept and
        # ignore it because PM doesn't need an LLM picker.
        llm_chat: Optional[Callable[[str], str]] = None,
    ):
        del llm_chat
        self._house_lookup = house_lookup
        self._target_total = target_total
        self._excluded: Dict[str, Set[str]] = {
            grp: {p.lower() for p in probes}
            for grp, probes in (exclude_probes or {}).items()
        }

    def generate(
        self,
        scenes: List[Dict[str, Any]],
        *,
        n_per_scene: int = 6,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Produce prospective-memory candidates across ``scenes``."""
        del seed
        out: List[CandidateQuestion] = []
        for scene in scenes:
            if len(out) >= self._target_total:
                break
            out.extend(self._candidates_for_scene(scene, n_per_scene=n_per_scene))
        return out[: self._target_total]

    def _candidates_for_scene(
        self, scene: Dict[str, Any], n_per_scene: int
    ) -> List[CandidateQuestion]:
        """Pick items for one scene; emit (item, rotated action) pairs."""
        unique_per_room = self._unique_per_room(scene)
        if not unique_per_room:
            return []

        # Flatten + sort so item order is deterministic per scene.
        # Hash-rotate so different scenes pick from different parts
        # of the pool (avoids every scene starting with Bathroom).
        sample_id = str(scene.get("sample_id", "unknown"))
        pool: List[Tuple[str, str]] = []
        for room in sorted(unique_per_room):
            for t in unique_per_room[room]:
                pool.append((room, t))
        if not pool:
            return []
        offset = int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % len(pool)
        pool = pool[offset:] + pool[:offset]

        excluded = self._excluded.setdefault("__prosmem__", set())
        out: List[CandidateQuestion] = []
        action_cursor = 0
        for room_type, item_type in pool:
            pl = _probe_for_question(item_type)
            if pl in excluded:
                continue
            action_idx = action_cursor % len(ACTIONS)
            action = ACTIONS[action_idx]
            out.append(
                self._build_candidate(
                    scene=scene,
                    room_type=room_type,
                    probe_object=item_type,
                    action=action,
                    action_idx=action_idx,
                    ground_truth_objects=unique_per_room[room_type],
                )
            )
            excluded.add(pl)
            action_cursor += 1
            if len(out) >= n_per_scene:
                break
        return out

    def _unique_per_room(self, scene: Dict[str, Any]) -> Dict[str, List[str]]:
        """Same machinery as pattern_completion / retention_decay.

        Returns ``{room_type: [unique_object_types]}``; empty dict
        when the scene's house can't be loaded or has no objects
        unique to a single room.
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
        return unique

    def _build_candidate(
        self,
        *,
        scene: Dict[str, Any],
        room_type: str,
        probe_object: str,
        action: str,
        action_idx: int,
        ground_truth_objects: List[str],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion for a single (item, action) probe."""
        sample_id = str(scene.get("sample_id", "unknown"))
        probe_lower = _probe_for_question(probe_object)
        instruction_text = f"Instruction: when you see the {probe_lower}, {action}."
        question = f"What were you instructed to do when you see the {probe_lower}?"
        return CandidateQuestion(
            question_id=_question_id(sample_id, probe_lower, action_idx),
            question=question,
            answer=action,
            category="prospective_memory",
            paradigm=self.name,
            scene_ids=[sample_id],
            tools_expected=["semantic_search", "search_gists", "entity_query"],
            paradigm_metadata={
                "trigger_object": probe_lower,
                "instructed_action": action,
                "action_idx": action_idx,
                "instruction_text": instruction_text,
                "room_type": room_type,
                "ground_truth_objects_in_room": sorted(set(ground_truth_objects)),
            },
            generator_notes=(
                f"Auto-generated prospective-memory probe. Instruction: "
                f"{instruction_text!r}. The {probe_lower} is in the "
                f"{_ROOM_PHRASE[room_type]} of this scene; the trigger "
                f"fires when the agent observes it during the trajectory; "
                f"the agent should recall the action {action!r}."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Pre-filter rubric for prospective-memory candidates."""
        return (
            "You are pre-filtering benchmark questions before a human "
            "reviewer sees them. Your job is to catch obviously-bad "
            "questions — NOT to do the human's precision filtering.\n\n"
            "The questions are prospective-memory probes asking 'What "
            "were you instructed to do when you see the <item>?' The "
            "agent was given an instruction before its tour saying "
            "'when you see <item>, <action>'. Ground-truth answer is "
            "the <action> phrase.\n\n"
            "Answer NO if the candidate clearly fails ANY of these "
            "criteria:\n"
            "  * STRUCTURAL TRIGGER: the trigger object is a wall, "
            "floor, ceiling, doorway, window, light fixture — not a "
            "discrete perceivable item.\n"
            "  * NONSENSICAL PAIRING: the (item, action) combination "
            "is so absurd it would never be issued as a real "
            "instruction (e.g. 'when you see the toilet, take a "
            "picture' — silly enough that a real user wouldn't say "
            "it).\n"
            "  * ABSTRACT CATEGORY: the trigger is a category-noun "
            "('decor', 'utensils') not a specific item.\n\n"
            "Answer YES when the (item, action) combination is "
            "plausible as a user instruction and the trigger is a "
            "discrete, concrete object.\n\n"
            "Answer AMBIGUOUS when the pairing is borderline.\n\n"
            "Answer with exactly one word: yes, ambiguous, or no."
        )
