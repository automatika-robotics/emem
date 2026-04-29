"""Context-dependent retrieval paradigm generator.

Tulving & Thomson's encoding specificity (1973) and Godden &
Baddeley's underwater-divers experiment (1975) showed human recall
is better when retrieval context matches encoding context. The
direction of the human effect is not the target for an embodied
memory system — explicit semantic recall *should* be context-
invariant. The interesting probe is the symmetric one: spatial
queries SHOULD be context-sensitive. The agent's R-tree spatial
index should distinguish "is X in my current room" from "does X
exist somewhere in memory" — otherwise the spatial index isn't
earning its keep.

Probe form (binary):

    "Is there a <item> near your current location?"

The phrasing matches the tool surface: ``get_current_context()`` /
``spatial_query()`` answer "near my position" directly; the agent
does not have a position→room-name translation tool, so probing on
"current room" silently demanded an extra inference step the
benchmark didn't acknowledge. "Near" collapses the answer to a
single tool call.

The agent's current position is encoded as a final pre-probe
observation (layer ``agent_position``). For each chosen item we
emit TWO candidates: one *matching-context* (agent positioned at
the centroid of the item's actual room → expected ``yes``) and
one *mismatching-context* (agent positioned at the centroid of a
different room of the same house → expected ``no``).

Items are unique-to-one-room objects (same machinery as
pattern_completion / retention_decay); the room centroid comes
from each ProcTHOR room's ``floorPolygon`` vertex average.
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

CONTEXT_MATCH = "match"
CONTEXT_MISMATCH = "mismatch"


_PICKER_PROMPT = """\
You are helping build a cognitive-psychology benchmark for embodied \
robots. The context-dependent-retrieval paradigm tests whether the \
agent's spatial index can answer "is X in my current room?" \
correctly when the agent stands either in X's actual room (answer \
yes) or in a different room (answer no).

Below is the simulator's ground-truth list of objects that appear \
in EXACTLY ONE room of the scene the robot toured.

== Objects unique to one room of the house ==
{object_list}
{avoid_block}
Your task: pick exactly {n} objects from the list above that make \
GOOD context-dependent-retrieval probes. A good probe is one where:

  * The object is concrete and discrete (a thing the robot's \
perception layers would name cleanly), not a structural item \
(wall, floor, ceiling), an abstract category ('decor', \
'utensils'), or a duplicate of items already picked.
  * The object name is room-ambiguous — i.e., its name does NOT \
trivially imply the room. Pass: 'lamp', 'painting', 'house plant', \
'box', 'cell phone'. Fail: 'fridge' (kitchen-obvious), 'toilet' \
(bathroom-obvious), 'bed' (bedroom-obvious) — those make the \
context-match probe trivially answerable from the name alone, not \
from spatial memory.

Return ONLY a JSON array of strings, no explanation, no markdown \
fences. Each string must be copied verbatim from the list above. \
Example format:
["Lamp", "Painting", "Box"]
"""

_AVOID_BLOCK = """

You have already picked these items for context-dependent probes \
in this dataset — DO NOT pick any of them again:
{avoid_list}
"""


def _question_id(sample_id: str, probe_object: str, context: str) -> str:
    """Stable question id for a single (item, context) candidate."""
    return (
        f"ctxdep_{_sanitize_id_fragment(sample_id)}_"
        f"{_sanitize_id_fragment(probe_object)}_"
        f"{_sanitize_id_fragment(context)}"
    )


def _bullet_list(items: List[Tuple[str, str]]) -> str:
    """Render a (room, type) list as room-tagged bullets."""
    if not items:
        return "  (none)"
    return "\n".join(f"  - {t}  (in {_ROOM_PHRASE[r]})" for r, t in items)


def _polygon_centroid(polygon: List[Dict[str, float]]) -> Tuple[float, float]:
    """Average vertex coordinates → centroid (x, z) on the floor plane.

    Cheap approximation of a polygon centroid; for the convex
    rectangular ProcTHOR room polygons it's effectively exact.
    """
    if not polygon:
        return (0.0, 0.0)
    xs = [float(v.get("x", 0.0)) for v in polygon]
    zs = [float(v.get("z", 0.0)) for v in polygon]
    return (sum(xs) / len(xs), sum(zs) / len(zs))


class ContextDependentRetrievalGenerator(ParadigmGenerator):
    """Emit context-dependent-retrieval probes for each scene.

    :param llm_chat: Pluggable ``(prompt: str) -> str`` callable.
    :param house_lookup: ``int -> house_dict`` from
        ``prior.load_dataset("procthor-10k")``.
    :param target_total: Global cap on emitted candidates.
    :param exclude_probes: Optional set of probe items to avoid
        (supplemental runs across scenes; keyed by ``"__ctxdep__"``).
    """

    name = "context_dependent_retrieval"

    def __init__(
        self,
        llm_chat: Callable[[str], str],
        house_lookup: Callable[[int], Optional[Dict[str, Any]]],
        target_total: int = 160,
        exclude_probes: Optional[Dict[str, Set[str]]] = None,
    ):
        self._llm_chat = llm_chat
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
        n_per_scene: int = 8,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Produce context-dependent-retrieval candidates across ``scenes``.

        ``n_per_scene`` is the *probe-question* budget — at 2
        contexts per item, ``n_per_scene=8`` picks 4 items per scene.
        """
        del seed
        out: List[CandidateQuestion] = []
        items_per_scene = max(1, n_per_scene // 2)
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
        """Pick items for one scene; emit (item, match) + (item, mismatch)."""
        room_data = self._room_data(scene)
        if not room_data:
            return []

        unique_per_room = {r: info["unique_objects"] for r, info in room_data.items()}
        if not unique_per_room:
            return []

        # Flatten to (room, type) pool; sorted ordering keeps the
        # picker prompt deterministic across runs.
        pool: List[Tuple[str, str]] = []
        for room in sorted(unique_per_room):
            for t in unique_per_room[room]:
                pool.append((room, t))
        if not pool:
            return []

        excluded = self._excluded.setdefault("__ctxdep__", set())
        over_request = items_per_scene * 3
        picks = self._request_picks(pool, excluded, n=over_request)

        type_to_room = {t.lower(): r for r, t in pool}
        out: List[CandidateQuestion] = []
        kept_items = 0
        rooms_in_scene = sorted(room_data)
        for pick in picks:
            pl = pick.strip().lower()
            if not pl or pl in excluded:
                continue
            if pl not in type_to_room:
                log.debug(
                    "context_dependent_retrieval: dropping hallucinated pick %r",
                    pick,
                )
                continue
            room_a = type_to_room[pl]
            other_rooms = [r for r in rooms_in_scene if r != room_a]
            if not other_rooms:
                continue
            room_b = other_rooms[0]
            for context, room_for_pos in (
                (CONTEXT_MATCH, room_a),
                (CONTEXT_MISMATCH, room_b),
            ):
                out.append(
                    self._build_candidate(
                        scene=scene,
                        probe_object=pick,
                        item_room=room_a,
                        agent_room=room_for_pos,
                        agent_position=room_data[room_for_pos]["centroid"],
                        context=context,
                        ground_truth_objects=room_data[room_a]["unique_objects"],
                    )
                )
            excluded.add(pl)
            kept_items += 1
            if kept_items >= items_per_scene:
                break
        return out

    def _room_data(self, scene: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Compute per-room data: unique objects + centroid.

        Returns a dict keyed by room_type with::

            {"unique_objects": [str, ...], "centroid": (x, z)}

        for every eligible room that contains at least one object
        unique to it. Empty dict if the scene's house can't be
        loaded or has no eligible rooms.
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
        eligible_objects = {r: gt.get(r, []) for r in ELIGIBLE_ROOMS if gt.get(r)}
        if not eligible_objects:
            log.info("scene %s: no eligible rooms; skipping", sample_id)
            return {}

        # Find centroids for the eligible rooms by walking house["rooms"].
        centroids: Dict[str, Tuple[float, float]] = {}
        for room in house.get("rooms", []):
            rt = room.get("roomType")
            if rt in eligible_objects and rt not in centroids:
                centroids[rt] = _polygon_centroid(room.get("floorPolygon", []))

        # Compute unique-to-one-room objects per eligible room.
        unique_per_room: Dict[str, List[str]] = {}
        for room, types in eligible_objects.items():
            others: Set[str] = set()
            for other_room, other_types in eligible_objects.items():
                if other_room != room:
                    others.update(other_types)
            uniq = [t for t in dict.fromkeys(types) if t not in others]
            if uniq and room in centroids:
                unique_per_room[room] = uniq

        return {
            r: {"unique_objects": unique_per_room[r], "centroid": centroids[r]}
            for r in unique_per_room
        }

    def _request_picks(
        self,
        pool: List[Tuple[str, str]],
        excluded: Set[str],
        n: int,
    ) -> List[str]:
        """Ask the LLM for ``n`` room-ambiguous, memorable picks."""
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
                "context_dependent_retrieval: no items parsed; raw[:120]=%r",
                (raw or "")[:120],
            )
        return items[:n]

    def _build_candidate(
        self,
        *,
        scene: Dict[str, Any],
        probe_object: str,
        item_room: str,
        agent_room: str,
        agent_position: Tuple[float, float],
        context: str,
        ground_truth_objects: List[str],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion for a single (item, context) probe."""
        sample_id = str(scene.get("sample_id", "unknown"))
        probe_lower = _probe_for_question(probe_object)
        question = f"Is there a {probe_lower} near your current location?"
        answer = "yes" if context == CONTEXT_MATCH else "no"
        x, z = agent_position
        return CandidateQuestion(
            question_id=_question_id(sample_id, probe_lower, context),
            question=question,
            answer=answer,
            category="context_dependent_retrieval",
            paradigm=self.name,
            scene_ids=[sample_id],
            tools_expected=["get_current_context", "spatial_query", "locate"],
            paradigm_metadata={
                "probe_object": probe_lower,
                "room_type": item_room,
                "item_room": item_room,
                "agent_room": agent_room,
                "context": context,
                "agent_position": [float(x), 0.0, float(z)],
                "ground_truth_objects_in_room": sorted(set(ground_truth_objects)),
            },
            generator_notes=(
                f"Auto-generated context-dependent-retrieval probe. "
                f"The {probe_lower} is in the {_ROOM_PHRASE[item_room]} "
                f"of this scene; the agent is positioned in the "
                f"{_ROOM_PHRASE[agent_room]} ({context} context); the "
                f"correct answer is {answer!r}."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Pre-filter rubric for context-dependent-retrieval candidates."""
        return (
            "You are pre-filtering benchmark questions before a human "
            "reviewer sees them. Your job is to catch obviously-bad "
            "questions — NOT to do the human's precision filtering.\n\n"
            "The questions are context-dependent-retrieval probes "
            "asking 'Is there a <item> in your current room?' with "
            "ground-truth answer 'yes' (the agent is positioned in "
            "the item's actual room) or 'no' (the agent is in a "
            "different room of the same house).\n\n"
            "Answer NO if the probe item clearly fails ANY of these "
            "criteria:\n"
            "  * STRUCTURAL: a wall, floor, ceiling, doorway, "
            "window, light fixture.\n"
            "  * ABSTRACT CATEGORY: 'decor', 'dishware', 'appliance', "
            "'furniture', 'utensils'.\n"
            "  * SCHEMA-OBVIOUS: the object name strongly implies a "
            "specific room from common knowledge alone, making the "
            "context match trivially derivable from the name. "
            "Examples that MUST be no: 'fridge', 'oven', 'microwave', "
            "'toaster' (kitchen-obvious); 'toilet', 'bathtub', "
            "'shower' (bathroom-obvious); 'bed', 'pillow' "
            "(bedroom-obvious); 'sofa', 'armchair' "
            "(living-room-obvious).\n"
            "  * NONSENSICAL: doesn't make sense as a discrete "
            "perceived object.\n\n"
            "Answer YES when the probe item is a discrete, concrete, "
            "memorable object whose room is NOT trivially derivable "
            "from the name — i.e., the answer requires spatial "
            "memory, not common knowledge.\n\n"
            "Answer AMBIGUOUS when the item is borderline.\n\n"
            "Answer with exactly one word: yes, ambiguous, or no."
        )
