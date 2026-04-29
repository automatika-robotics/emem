"""Pattern-separation paradigm generator.

For each similarity pair of ProcTHOR houses (from A7's
``scenes.jsonl``), the generator looks up both houses' real object
lists per room, finds objects that appear in ONE house's room but
NOT the other's, and asks an LLM to pick the most discriminating
ones. Each returned object becomes a probe of the form

    "You explored two houses, Alpha and Beta. In which house did you
    see a <object> in the <room>?"

with ground-truth answer either ``"Alpha"`` or ``"Beta"`` depending
on which house's ground-truth object list contains the item.

The runtime schedule (wired in :mod:`.schedule_builder`) ingests
both houses back-to-back as separate episodes — each observation
prefixed with "In house Alpha: " / "In house Beta: " so the agent
has a natural-language label to bind features to. A memory system
that can't separate near-identical contexts fails these probes by
guessing the wrong house.

Why pair on ProcTHOR's 4 room-type set: the ``ELIGIBLE_ROOMS``
constraint from DRM carries over; probes only fire for rooms that
both houses of the pair actually have.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
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

# Labels presented to the agent at ingest + probe time. Short,
# distinct, and orthographically unrelated so the agent can't
# confuse them by a one-letter slip.
HOUSE_LABEL_A = "Alpha"
HOUSE_LABEL_B = "Beta"


_GENERATION_PROMPT = """\
You are helping build a cognitive-psychology benchmark for embodied \
robots. The pattern-separation paradigm tests whether a memory system \
can keep two near-identical contexts distinct — for example, \
distinguish one house's kitchen from a visually-similar kitchen in a \
second house.

A robot explored two houses, {label_a} and {label_b}. Here are the \
simulator's ground-truth object lists for the {room_phrase} in each:

=== House {label_a}'s {room_phrase} ===
{only_in_a}

=== House {label_b}'s {room_phrase} ===
{only_in_b}

Each list above contains ONLY objects unique to that house — i.e. \
objects present in one {room_phrase} but NOT the other. Objects \
present in both are hidden so you can't confuse them.
{avoid_block}
Your task: pick exactly {n} of these distinguishing objects that \
make the sharpest pattern-separation probes. Each pick must satisfy \
ALL of the following:
  * be copied verbatim from one of the two lists above
  * be concrete and easily describable in natural language (skip \
items like "CounterTop" or "Floor" that span whole rooms)
  * be diverse across the {n} items (don't pick five near-identical \
decor pieces)
  * prefer items that are visually distinctive — a probe object a \
reviewer would agree "yes, seeing this in one house and not the \
other is a clean separation question"
  * BALANCE the picks across the two lists — roughly half from House \
{label_a}'s list and half from House {label_b}'s list. Do NOT draw \
all your picks from a single list.

Return ONLY a JSON array of strings, no explanation, no markdown \
fences. Each string must be a verbatim copy of an object from one \
of the two lists above. Example format:
["Fridge", "Toaster", "Painting"]
"""

_AVOID_BLOCK = """\

You have already picked these items for other {room_phrase} probes \
in this dataset — DO NOT pick any of them again:
{avoid_list}
"""


def _question_id(pair_id: str, room_type: str, probe_object: str) -> str:
    """Stable question id for a pattern-separation probe."""
    return (
        f"patsep_{_sanitize_id_fragment(pair_id)}_"
        f"{_sanitize_id_fragment(room_type)}_"
        f"{_sanitize_id_fragment(probe_object)}"
    )


def _contains_token(needle: str, haystack_list: List[str]) -> bool:
    """Word-token overlap check against a list of ProcTHOR object types.

    Uses the same CamelCase-aware tokenizer the DRM paradigm does so
    ``"coffee table"`` matches a ground-truth ``"DiningTable"``.
    """
    needle_tokens = _tokenize_objecttype(needle)
    if not needle_tokens:
        return False
    for item in haystack_list:
        if needle_tokens & _tokenize_objecttype(item):
            return True
    return False


class PatternSeparationGenerator(ParadigmGenerator):
    """Emit pattern-separation probes for each ProcTHOR similarity pair.

    :param llm_chat: Pluggable ``(prompt: str) -> str`` callable.
    :param house_lookup: ``int -> house_dict`` from
        ``prior.load_dataset("procthor-10k")``. The generator uses
        this to fetch both members of each similarity pair and run
        polygon-based object attribution.
    :param target_total: Global cap on emitted candidates.
    :param exclude_probes: Optional per-room set of probe objects to
        avoid (supplemental runs).
    """

    name = "pattern_separation"

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

    def scene_units(self, scenes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group scenes by ``similarity_pair_id``; emit 2-element pairs.

        Scenes without a pair id, or pair ids with fewer than 2
        members, are dropped — the paradigm is undefined on
        singletons. Pairs are returned in a sorted-pair-id order so
        the outer CLI loop is deterministic.

        The Alpha/Beta role assignment is rotated per pair by
        hashing the pair_id: alternating pairs swap which
        sample-id-first member becomes Alpha. Without this, gemma4's
        bias toward picking items from whichever list is shown first
        would produce a correct_house distribution heavily skewed
        toward Alpha across the whole dataset.
        """
        pairs: Dict[str, List[Dict[str, Any]]] = {}
        for s in scenes:
            pid = s.get("similarity_pair_id")
            if pid:
                pairs.setdefault(str(pid), []).append(s)
        out: List[List[Dict[str, Any]]] = []
        for pid in sorted(pairs):
            members = pairs[pid]
            if len(members) < 2:
                log.info(
                    "pattern_separation: skipping pair_id=%s (only %d member)",
                    pid,
                    len(members),
                )
                continue
            members = sorted(members, key=lambda s: s.get("sample_id", ""))[:2]
            # Swap Alpha/Beta on half the pairs (hash-parity) so the
            # generator-side Alpha bias averages out across the set.
            h = int(hashlib.md5(pid.encode()).hexdigest(), 16) & 1
            if h:
                members = [members[1], members[0]]
            out.append(members)
        return out

    def generate(
        self,
        scenes: List[Dict[str, Any]],
        *,
        n_per_scene: int = 30,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Produce candidates for a single pair (scenes == [house_A, house_B])."""
        del seed
        if len(scenes) != 2:
            log.warning(
                "pattern_separation.generate expected 2 scenes, got %d; "
                "did scene_units run?",
                len(scenes),
            )
            return []
        return self._candidates_for_pair(scenes[0], scenes[1], n_per_scene)

    def _candidates_for_pair(
        self,
        scene_a: Dict[str, Any],
        scene_b: Dict[str, Any],
        n_per_pair: int,
    ) -> List[CandidateQuestion]:
        """Generate candidates for a single similarity pair."""
        house_a = self._lookup_house(scene_a)
        house_b = self._lookup_house(scene_b)
        if house_a is None or house_b is None:
            return []

        from harness.environments.procthor_utils import objects_by_room

        gt_a = objects_by_room(house_a)
        gt_b = objects_by_room(house_b)
        eligible_rooms = sorted(
            r for r in ELIGIBLE_ROOMS if gt_a.get(r) and gt_b.get(r)
        )
        if not eligible_rooms:
            log.info(
                "pattern_separation: pair %s+%s has no shared eligible rooms",
                scene_a.get("sample_id"),
                scene_b.get("sample_id"),
            )
            return []

        # Rotate the starting room per pair so the "extra" slot isn't
        # always the alphabetically-first room. Matches DRM's trick.
        pair_id = str(
            scene_a.get("similarity_pair_id")
            or scene_b.get("similarity_pair_id")
            or f"{scene_a.get('sample_id')}|{scene_b.get('sample_id')}"
        )
        offset = int(hashlib.md5(pair_id.encode()).hexdigest(), 16) % len(
            eligible_rooms
        )
        ordered_rooms = eligible_rooms[offset:] + eligible_rooms[:offset]
        per_room_budget = max(1, -(-n_per_pair // len(ordered_rooms)))
        over_request = per_room_budget * 3

        room_cands: Dict[str, List[CandidateQuestion]] = {}
        for room_type in ordered_rooms:
            a_types = list(dict.fromkeys(gt_a.get(room_type, [])))
            b_types = list(dict.fromkeys(gt_b.get(room_type, [])))
            only_a = [t for t in a_types if t not in b_types]
            only_b = [t for t in b_types if t not in a_types]
            if not only_a and not only_b:
                log.info(
                    "pattern_separation: pair %s room=%s has no distinguishing objects",
                    pair_id,
                    room_type,
                )
                room_cands[room_type] = []
                continue

            excluded = self._excluded.setdefault(room_type, set())
            picks = self._request_distinguishing_picks(
                room_type, only_a, only_b, excluded, n=over_request
            )
            kept: List[CandidateQuestion] = []
            for pick in picks:
                pick_lower = pick.lower().strip()
                if pick_lower in excluded:
                    continue
                # Determine which house actually contains this pick
                # (tokens must match exactly to be considered "in").
                in_a = _contains_token(pick, only_a) and not _contains_token(
                    pick, only_b
                )
                in_b = _contains_token(pick, only_b) and not _contains_token(
                    pick, only_a
                )
                if not (in_a ^ in_b):
                    # Either both or neither — skip; not a clean probe.
                    continue
                correct_label = HOUSE_LABEL_A if in_a else HOUSE_LABEL_B
                kept.append(
                    self._build_candidate(
                        scene_a=scene_a,
                        scene_b=scene_b,
                        pair_id=pair_id,
                        room_type=room_type,
                        probe_object=pick,
                        correct_label=correct_label,
                        only_in_a=only_a,
                        only_in_b=only_b,
                    )
                )
                excluded.add(pick_lower)
                if len(kept) >= per_room_budget:
                    break
            room_cands[room_type] = kept

        # Round-robin flatten.
        out: List[CandidateQuestion] = []
        max_len = max((len(v) for v in room_cands.values()), default=0)
        for i in range(max_len):
            for room_type in ordered_rooms:
                lst = room_cands[room_type]
                if i < len(lst):
                    out.append(lst[i])
                    if len(out) >= n_per_pair:
                        return out
        return out

    def _lookup_house(self, scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        idx = scene.get("procthor_dataset_index")
        if idx is None:
            log.warning(
                "pattern_separation: scene %s missing procthor_dataset_index",
                scene.get("sample_id"),
            )
            return None
        house = self._house_lookup(int(idx))
        if house is None:
            log.warning("pattern_separation: house_lookup(%d) returned None", idx)
        return house

    def _request_distinguishing_picks(
        self,
        room_type: str,
        only_in_a: List[str],
        only_in_b: List[str],
        excluded: Set[str],
        n: int,
    ) -> List[str]:
        """Call the LLM for ``n`` distinguishing probe objects."""
        avoid_block = (
            _AVOID_BLOCK.format(
                room_phrase=_ROOM_PHRASE[room_type],
                avoid_list="\n".join(f"  - {p}" for p in sorted(excluded)),
            )
            if excluded
            else ""
        )
        prompt = _GENERATION_PROMPT.format(
            label_a=HOUSE_LABEL_A,
            label_b=HOUSE_LABEL_B,
            room_phrase=_ROOM_PHRASE[room_type],
            only_in_a=_bullet_list(only_in_a),
            only_in_b=_bullet_list(only_in_b),
            avoid_block=avoid_block,
            n=n,
        )
        raw = self._llm_chat(prompt)
        items = _parse_absent_items(raw)
        if not items:
            log.warning(
                "pattern_separation: no picks parsed; raw[:120]=%r",
                (raw or "")[:120],
            )
        return items[:n]

    def _build_candidate(
        self,
        scene_a: Dict[str, Any],
        scene_b: Dict[str, Any],
        pair_id: str,
        room_type: str,
        probe_object: str,
        correct_label: str,
        only_in_a: List[str],
        only_in_b: List[str],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion."""
        room_phrase = _ROOM_PHRASE[room_type]
        probe_lower = _probe_for_question(probe_object)
        article = _indefinite_article(probe_lower)
        question = (
            f"You explored two houses, {HOUSE_LABEL_A} and {HOUSE_LABEL_B}. "
            f"In which house did you see {article} {probe_lower} in the "
            f"{room_phrase}?"
        )
        sample_ids = [
            str(scene_a.get("sample_id", "")),
            str(scene_b.get("sample_id", "")),
        ]
        return CandidateQuestion(
            question_id=_question_id(pair_id, room_type, probe_lower),
            question=question,
            answer=correct_label,
            category="pattern_separation",
            paradigm=self.name,
            scene_ids=sample_ids,
            tools_expected=["semantic_search", "entity_query"],
            paradigm_metadata={
                "pair_id": pair_id,
                "room_type": room_type,
                "probe_object": probe_lower,
                "correct_house": correct_label,
                "label_to_sample_id": {
                    HOUSE_LABEL_A: sample_ids[0],
                    HOUSE_LABEL_B: sample_ids[1],
                },
                "only_in_alpha": list(dict.fromkeys(only_in_a)),
                "only_in_beta": list(dict.fromkeys(only_in_b)),
            },
            generator_notes=(
                f"Auto-generated pattern-separation probe for pair "
                f"{pair_id}. Target object is in exactly one of the "
                f"two houses' {room_phrase}s; the agent must bind it "
                f"to the correct house."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Gemma4 rubric for pre-filtering pattern-separation candidates.

        Calibrated against the A14b.3 human review (84 candidates,
        65% kept). The rubric is intentionally permissive — its job
        is to catch obviously-bad probes (structural items, abstract
        categories) that the generator's post-filter missed, NOT to
        do precision review. Borderline cases default to ambiguous
        so the human reviewer makes the final call.
        """
        return (
            "You are pre-filtering benchmark questions before a human "
            "reviewer sees them. Your job is to catch obviously-bad "
            "questions — NOT to do the human's precision filtering.\n\n"
            "The questions are pattern-separation probes asking 'In "
            "which house (Alpha or Beta) did you see <object> in the "
            "<room>?'. The respondent has just observed two near-"
            "identical houses and must remember which one contained "
            "which features.\n\n"
            "Answer NO if the probe object clearly fails ANY of "
            "these criteria:\n"
            "  * STRUCTURAL: a wall, floor, ceiling, doorway, "
            "window, light fixture, or other room-spanning feature.\n"
            "  * ABSTRACT CATEGORY: a category-noun like 'decor', "
            "'dishware', 'appliance', 'furniture', 'utensils'.\n"
            "  * FUNGIBLE COMMODITY: a small everyday item that "
            "almost any home has — the respondent could reasonably "
            "think both houses contain one. Examples that MUST be "
            "answered no: 'pen', 'pencil', 'paper', 'key chain', "
            "'key', 'watch', 'wallet', 'credit card', 'phone', "
            "'cell phone', 'book', 'magazine', 'plate', 'bowl', "
            "'cup', 'mug', 'fork', 'knife', 'spoon', 'cutlery', "
            "'towel', 'soap', 'tissue', 'shampoo', 'brush', 'comb', "
            "'remote control', 'house plant', 'vase', 'pillow', "
            "'blanket'.\n"
            "  * NONSENSICAL: doesn't make sense as a discrete "
            "object you'd see in the room.\n\n"
            "Answer YES when the probe object is a discrete, "
            "concrete item that is NOT a fungible commodity — "
            "something visually distinctive that a human would "
            "remember as a feature of one specific house. "
            "Examples that should pass: 'stand mixer', 'fireplace', "
            "'teddy bear', 'dog bed', 'telescope', 'duvet', "
            "'curtains', 'toaster', 'shower curtain', 'mirror', "
            "'fridge', 'sofa', 'lamp', 'bookshelf', 'TV', 'rug', "
            "'oven', 'microwave', 'bathtub', 'bed', 'desk', "
            "'dresser', 'wardrobe'.\n\n"
            "Answer AMBIGUOUS when the object is borderline — "
            "could be argued either as a fungible commodity or a "
            "distinctive item, or wording is unclear.\n\n"
            "Answer with exactly one word: yes, ambiguous, or no."
        )


def _bullet_list(items: List[str]) -> str:
    """Render an object list as bullets; ``(none)`` on empty."""
    if not items:
        return "  (none)"
    return "\n".join(f"  - {t}" for t in items)
