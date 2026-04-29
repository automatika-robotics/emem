"""Pattern-completion paradigm generator.

For each ProcTHOR house, the generator finds object types that appear
in exactly ONE room of the scene (so "what room was X in?" has a
clean ground-truth answer) and asks an LLM to pick the items where
the room placement is NOT trivially derivable from the object name
alone — these are the cases that genuinely test memory rather than
schema completion.

Probe form::

    "What room did you see the <object> in?"

with ground-truth answer one of ``"kitchen"`` / ``"bathroom"`` /
``"bedroom"`` / ``"living room"``. A memory system that can't
retrieve the room context from a partial cue (the object) fails by
guessing the wrong room or returning "I don't know".

Pattern completion vs DRM: DRM tests confabulation of schema-typical
absent items ("did you see X" → answer "no"). Pattern completion
tests retrieval of room context from a present item ("what room had
X" → 4-way classification). Different cognitive operation.

Pattern completion vs pattern separation: pattern separation has TWO
houses and binds features to one; pattern completion has ONE house
and binds features to one of four rooms.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
    ParadigmGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import (
    _indefinite_article,
    _parse_absent_items,
    _probe_for_question,
    _sanitize_id_fragment,
    _tokenize_objecttype,
)

log = logging.getLogger(__name__)

ELIGIBLE_ROOMS: Set[str] = {"Kitchen", "Bedroom", "Bathroom", "LivingRoom"}

_ROOM_PHRASE: Dict[str, str] = {
    "Kitchen": "kitchen",
    "Bedroom": "bedroom",
    "Bathroom": "bathroom",
    "LivingRoom": "living room",
}


_GENERATION_PROMPT = """\
You are helping build a cognitive-psychology benchmark for embodied \
robots. The pattern-completion paradigm tests whether a memory \
system can retrieve a room's context when given a partial cue (one \
object that was in the room).

A robot explored a house with these four rooms: kitchen, bathroom, \
bedroom, living room. Below is the simulator's ground-truth list of \
objects that appear in EXACTLY ONE room of this house — i.e., for \
each item, the answer to "what room was this in?" is unambiguous.

== Objects unique to the {room_phrase} ==
{object_list}
{avoid_block}
Your task: pick exactly {n} objects from the list above that make \
GOOD pattern-completion probes. A good probe is one where:

  * The room placement is NOT obvious from the object name alone — \
the name should be plausibly compatible with multiple room types, \
so the respondent has to USE MEMORY to answer correctly. Pass: \
'lamp', 'painting', 'house plant', 'statue', 'vase' (could be any \
room). Fail: 'apple' / 'pot' / 'fridge' (kitchen-obvious), \
'toilet' / 'bathtub' (bathroom-obvious), 'bed' / 'pillow' \
(bedroom-obvious).
  * The object is concrete and discrete (skip structural items \
like wall / floor / ceiling).
  * The {n} picks are diverse (not five paintings or five vases).

Return ONLY a JSON array of strings, no explanation, no markdown \
fences. Each string must be copied verbatim from the list above. \
Example format:
["Lamp", "Painting", "Vase"]
"""

_AVOID_BLOCK = """\

You have already picked these items for other {room_phrase} probes \
in this dataset — DO NOT pick any of them again:
{avoid_list}
"""


def _question_id(sample_id: str, room_type: str, probe_object: str) -> str:
    """Stable question id."""
    return (
        f"patcomp_{_sanitize_id_fragment(sample_id)}_"
        f"{_sanitize_id_fragment(room_type)}_"
        f"{_sanitize_id_fragment(probe_object)}"
    )


def _bullet_list(items: List[str]) -> str:
    """Render an object list as bullets; ``(none)`` on empty."""
    if not items:
        return "  (none)"
    return "\n".join(f"  - {t}" for t in items)


class PatternCompletionGenerator(ParadigmGenerator):
    """Emit pattern-completion probes for each ProcTHOR scene.

    :param llm_chat: Pluggable ``(prompt: str) -> str`` callable.
    :param house_lookup: ``int -> house_dict`` from
        ``prior.load_dataset("procthor-10k")``.
    :param target_total: Global cap on emitted candidates.
    :param exclude_probes: Optional per-room set of probe objects to
        avoid (supplemental runs across scenes).
    """

    name = "pattern_completion"

    def __init__(
        self,
        llm_chat: Callable[[str], str],
        house_lookup: Callable[[int], Optional[Dict[str, Any]]],
        target_total: int = 120,
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
        n_per_scene: int = 6,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Produce pattern-completion candidates across ``scenes``."""
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
        """Generate candidates for one scene."""
        unique_per_room = self._unique_per_room(scene)
        if not unique_per_room:
            return []

        sample_id = str(scene.get("sample_id", "unknown"))
        ordered_rooms = sorted(unique_per_room)
        offset = int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % len(
            ordered_rooms
        )
        ordered_rooms = ordered_rooms[offset:] + ordered_rooms[:offset]
        per_room_budget = max(1, -(-n_per_scene // len(ordered_rooms)))
        over_request = per_room_budget * 3

        room_cands: Dict[str, List[CandidateQuestion]] = {
            room_type: self._candidates_for_room(
                scene=scene,
                room_type=room_type,
                uniq=unique_per_room[room_type],
                unique_per_room=unique_per_room,
                budget=per_room_budget,
                over_request=over_request,
            )
            for room_type in ordered_rooms
        }

        out: List[CandidateQuestion] = []
        max_len = max((len(v) for v in room_cands.values()), default=0)
        for i in range(max_len):
            for room_type in ordered_rooms:
                lst = room_cands[room_type]
                if i < len(lst):
                    out.append(lst[i])
                    if len(out) >= n_per_scene:
                        return out
        return out

    def _unique_per_room(self, scene: Dict[str, Any]) -> Dict[str, List[str]]:
        """Return ``{room_type: [unique_object_types]}`` for one scene.

        Empty dict if the scene's house can't be loaded, has no
        eligible rooms, or has no objects appearing in exactly one
        eligible room.
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

    def _candidates_for_room(
        self,
        *,
        scene: Dict[str, Any],
        room_type: str,
        uniq: List[str],
        unique_per_room: Dict[str, List[str]],
        budget: int,
        over_request: int,
    ) -> List[CandidateQuestion]:
        """Validate LLM picks for one room and build candidates."""
        excluded = self._excluded.setdefault(room_type, set())
        picks = self._request_picks(room_type, uniq, excluded, n=over_request)
        uniq_lower = {t.lower() for t in uniq}
        kept: List[CandidateQuestion] = []
        for pick in picks:
            pl = pick.strip().lower()
            if pl in excluded:
                continue
            if pl not in uniq_lower:
                log.debug(
                    "pattern_completion: dropping hallucinated pick %r "
                    "(not in unique list)",
                    pick,
                )
                continue
            kept.append(
                self._build_candidate(
                    scene=scene,
                    room_type=room_type,
                    probe_object=pick,
                    unique_per_room=unique_per_room,
                )
            )
            excluded.add(pl)
            if len(kept) >= budget:
                break
        return kept

    def _request_picks(
        self,
        room_type: str,
        unique_objects: List[str],
        excluded: Set[str],
        n: int,
    ) -> List[str]:
        """Ask the LLM for ``n`` non-schema-trivial picks."""
        avoid_block = (
            _AVOID_BLOCK.format(
                room_phrase=_ROOM_PHRASE[room_type],
                avoid_list="\n".join(f"  - {p}" for p in sorted(excluded)),
            )
            if excluded
            else ""
        )
        prompt = _GENERATION_PROMPT.format(
            room_phrase=_ROOM_PHRASE[room_type],
            object_list=_bullet_list(unique_objects),
            avoid_block=avoid_block,
            n=n,
        )
        raw = self._llm_chat(prompt)
        items = _parse_absent_items(raw)
        if not items:
            log.warning(
                "pattern_completion: no items parsed for room=%s; raw[:120]=%r",
                room_type,
                (raw or "")[:120],
            )
        return items[:n]

    def _build_candidate(
        self,
        scene: Dict[str, Any],
        room_type: str,
        probe_object: str,
        unique_per_room: Dict[str, List[str]],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion."""
        sample_id = str(scene.get("sample_id", "unknown"))
        probe_lower = _probe_for_question(probe_object)
        article = _indefinite_article(probe_lower)
        room_phrase = _ROOM_PHRASE[room_type]
        question = f"What room did you see {article} {probe_lower} in?"
        return CandidateQuestion(
            question_id=_question_id(sample_id, room_type, probe_lower),
            question=question,
            answer=room_phrase,
            category="pattern_completion",
            paradigm=self.name,
            scene_ids=[sample_id],
            tools_expected=["semantic_search", "entity_query"],
            paradigm_metadata={
                "room_type": room_type,
                "probe_object": probe_lower,
                "ground_truth_objects_in_room": sorted(set(unique_per_room[room_type])),
                # Keep the full unique-per-room map for review so the
                # reviewer can sanity-check that the probe object is
                # genuinely in exactly one room.
                "unique_per_room": {
                    r: sorted(set(items)) for r, items in unique_per_room.items()
                },
            },
            generator_notes=(
                f"Auto-generated pattern-completion probe. Object is "
                f"unique to the {room_phrase} of this scene; the "
                f"agent must use memory (not schema) to retrieve the "
                f"room from the cue."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Pre-filter rubric for pattern-completion candidates."""
        return (
            "You are pre-filtering benchmark questions before a human "
            "reviewer sees them. Your job is to catch obviously-bad "
            "questions — NOT to do the human's precision filtering.\n\n"
            "The questions are pattern-completion probes asking 'What "
            "room did you see <object> in?' with ground-truth answer "
            "one of: kitchen, bathroom, bedroom, living room. The "
            "respondent has just observed a single house and must "
            "remember which room contained the cue object.\n\n"
            "Answer NO if the probe object clearly fails ANY of "
            "these criteria:\n"
            "  * STRUCTURAL: a wall, floor, ceiling, doorway, "
            "window, light fixture.\n"
            "  * ABSTRACT CATEGORY: a category-noun like 'decor', "
            "'dishware', 'appliance', 'furniture', 'utensils'.\n"
            "  * SCHEMA-OBVIOUS: the object name strongly implies a "
            "specific room from common knowledge alone, making the "
            "probe trivial. Examples that MUST be answered no: "
            "'apple', 'bread', 'egg', 'tomato', 'pot', 'fridge', "
            "'oven', 'microwave', 'toaster', 'kettle' (all kitchen-"
            "obvious); 'toilet', 'bathtub', 'shower', 'sink', "
            "'showerhead' (bathroom-obvious); 'bed', 'pillow', "
            "'duvet' (bedroom-obvious); 'sofa', 'armchair' "
            "(living-room-obvious).\n"
            "  * NONSENSICAL: doesn't make sense as a discrete "
            "object.\n\n"
            "Answer YES when the probe object is a discrete, "
            "concrete item whose room placement is not trivially "
            "derivable from the name alone — i.e., the name is "
            "compatible with multiple room types and only memory "
            "can pin it down. Examples that should pass: 'lamp', "
            "'painting', 'vase', 'house plant', 'statue', 'mirror', "
            "'rug', 'cell phone', 'remote control', 'box'.\n\n"
            "Answer AMBIGUOUS when the object is borderline — "
            "weakly schema-tilted, or wording unclear.\n\n"
            "Answer with exactly one word: yes, ambiguous, or no."
        )
