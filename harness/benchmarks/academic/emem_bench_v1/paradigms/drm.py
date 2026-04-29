"""DRM false-recall paradigm generator.

For each scene, computes the **ground-truth** list of object types in
each room via point-in-polygon lookup against the ProcTHOR house's
room polygons, then asks an LLM for schema-typical objects that are
NOT in that ground-truth list. Each returned object becomes a probe
``"Did you see a <object> in the <room>?"`` with ground-truth answer
``"no"``. A memory system that confabulates typical-but-absent
objects (the "DRM intrusion" phenomenon from human memory research)
fails these probes.

Ground-truth objects come from the simulator (ProcTHOR house
``objects`` table), NOT from the VLM captions a robot produced
while exploring. VLM text can miss real objects or hallucinate
absent ones, so grounding the paradigm on it would make the
"correct" answer fuzzy; ground truth makes it exact.

DRM is the first paradigm and is hand-reviewed without a pre-filter;
reviewer decisions on its output calibrate the yes/ambiguous/no
rubric that the remaining five paradigms will use.
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

log = logging.getLogger(__name__)

# Room types eligible for DRM probes. ProcTHOR-10K exposes exactly
# these four; all are schema-rich enough to support typical-object
# probes.
ELIGIBLE_ROOMS: Set[str] = {"Kitchen", "Bedroom", "Bathroom", "LivingRoom"}

# Room-type → display phrase used in the user-facing question text.
_ROOM_PHRASE: Dict[str, str] = {
    "Kitchen": "kitchen",
    "Bedroom": "bedroom",
    "Bathroom": "bathroom",
    "LivingRoom": "living room",
}

_GENERATION_PROMPT = """\
You are helping build a cognitive-psychology benchmark for embodied \
robots. The DRM false-recall paradigm probes whether a memory system \
confabulates objects that are schematically typical of a room but \
were never actually present.

A robot explored a {room_phrase} in a simulated home. The simulator's \
ground-truth list of EVERY object present in this {room_phrase} is:

{object_list}
{avoid_block}
Your task: produce exactly {n} "DRM probe objects" for this \
{room_phrase} — items that are TYPICAL of a {room_phrase} but are \
NOT in the ground-truth list above. Each item must satisfy ALL of \
the following:
  * be a single concrete object (a noun phrase of 1-3 words)
  * be strongly schema-typical of a {room_phrase}
  * MUST NOT appear in the ground-truth list above, even as a partial \
match, synonym, or superset. For example, if the list contains \
"Fridge", you may not propose "refrigerator", "mini fridge", or \
"beverage fridge". If it contains "Chair", you may not propose \
"armchair" or "rocking chair".
  * MUST NOT appear in the "already-seen" list above (if present).
  * be diverse across the {n} items (no near-duplicates, no items \
that differ only by a modifier)

If the obvious schema-typical items are exhausted, reach into less \
common but still plausible items: personal care, decor, tech, \
cleaning supplies, furniture accessories, consumables, etc. Depth \
matters more than obviousness.

Return ONLY a JSON array of strings, no explanation, no markdown \
fences. Example format: ["item one", "item two", "item three"]
"""

_AVOID_BLOCK = """\

You have already proposed these items for other {room_phrase}s in \
this dataset — DO NOT propose any of them again (also avoid \
synonyms / pluralisations):
{avoid_list}
"""


_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _parse_absent_items(raw: str) -> List[str]:
    """Extract a JSON array of strings from the LLM's raw output.

    Tolerant of markdown fences and surrounding prose: scans for the
    first ``[...]`` block and parses it. Items are stripped and
    deduplicated (case-insensitive, preserving first-seen casing).
    """
    match = _JSON_ARRAY_RE.search(raw or "")
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().strip('"').strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _sanitize_id_fragment(text: str) -> str:
    """Lowercase + replace non-alphanum with underscores, collapsed."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned or "unknown"


def _tokenize_objecttype(text: str) -> Set[str]:
    """Split a ProcTHOR objectType or natural-language phrase into tokens.

    Handles three cases in one pass: CamelCase ("DiningTable" →
    ["Dining", "Table"]), lowercase phrases ("coffee table" →
    ["coffee", "table"]), and snake/kebab-case ("dining_TABLE" →
    ["dining", "TABLE"]). Returns lowercased tokens of length ≥3.
    """
    # Insert a space between a lowercase/digit and an uppercase letter
    # so "DiningTable" becomes "Dining Table" before tokenization.
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return {t.lower() for t in re.findall(r"[A-Za-z]+", split) if len(t) >= 3}


def _probe_object_leaks(probe_object: str, ground_truth_types: List[str]) -> bool:
    """Return True if ``probe_object`` tokens overlap any ground-truth type.

    Tokenizes the probe and every ground-truth type with
    :func:`_tokenize_objecttype` (CamelCase-aware, length-3 floor,
    case-insensitive) and returns True on any shared token. Catches
    the synonyms DRM cares about: ``"coffee table"`` leaks against
    ``"DiningTable"`` (shared ``"table"``); ``"mini fridge"`` leaks
    against ``"Fridge"``.

    :param probe_object: The LLM-proposed absent object.
    :param ground_truth_types: ProcTHOR objectType values for the
        probe's room.
    :returns: True if the probe overlaps ≥1 ground-truth type.
    """
    probe_tokens = _tokenize_objecttype(probe_object)
    if not probe_tokens:
        return False
    for gt in ground_truth_types:
        if probe_tokens & _tokenize_objecttype(gt):
            return True
    return False


def _format_object_list(types: List[str]) -> str:
    """Render a ground-truth object list for the LLM prompt.

    Dedupes while preserving insertion order so the prompt is compact
    regardless of instance count (a house may have 4 ``Chair``
    instances; the list shows ``Chair`` once).
    """
    if not types:
        return "  (none)"
    seen: Set[str] = set()
    out: List[str] = []
    for t in types:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return "\n".join(f"  - {t}" for t in out)


def _indefinite_article(word: str) -> str:
    """Return ``"an"`` for words starting with a vowel, else ``"a"``.

    Operates on the first letter only; good enough for the DRM probe
    question template ("Did you see a toaster" vs "an oven"). Ignores
    silent-h / acronym corner cases that don't appear in our probe
    vocabulary.
    """
    for ch in word:
        if ch.isalpha():
            return "an" if ch.lower() in "aeiou" else "a"
    return "a"


def _probe_for_question(probe_object: str) -> str:
    """Normalise a probe object for use inside the question text.

    Lowercases everything AND splits CamelCase so raw ProcTHOR
    objectType names ("ShelvingUnit", "CoffeeMachine", "TVStand")
    become natural phrases ("shelving unit", "coffee machine",
    "tv stand"). The paradigm asks "Did you see X in the Y?" — the
    probe is never the first word of the sentence, so any
    capitalisation in the raw LLM output is wrong.

    ``"Alarm Clock"`` -> ``"alarm clock"``
    ``"ShelvingUnit"`` -> ``"shelving unit"``
    ``"TVStand"`` -> ``"tv stand"``
    ``"dental floss"`` -> ``"dental floss"`` (unchanged)
    """
    s = probe_object.strip()
    # Insert a space between lower/digit → Upper transitions.
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    # And between UPPER → Upper+lower (handles "TVStand" -> "TV Stand").
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    # Collapse any run of whitespace to a single space.
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


class DRMGenerator(ParadigmGenerator):
    """Emit DRM probe questions grounded on ProcTHOR object tables.

    :param llm_chat: Pluggable ``(prompt: str) -> str`` callable that
        produces the LLM's raw response. Tests pass a stub; production
        wires ``OllamaLLMClient._chat`` or ``GeminiLLMClient._generate``.
    :param house_lookup: Callable mapping a manifest entry's
        ``procthor_dataset_index`` to the original ProcTHOR house
        dict (with ``"rooms"`` polygons and ``"objects"`` list).
        Production injects a closure around
        ``prior.load_dataset("procthor-10k")``; tests inject a stub.
    :param target_total: Total candidates to emit across all scenes.
    """

    name = "drm"

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
        # Seed the per-room exclusion set with any previously-seen
        # probes the caller wants to avoid. Proposals matching these
        # (case-insensitive) are rejected by the post-filter, and the
        # prompt tells the LLM to avoid them up front.
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
        """Produce DRM candidates across ``scenes``.

        :param scenes: Merged scene entries from
            :func:`load_scene_entries` — each must carry a
            ``procthor_dataset_index`` so the house dict can be
            re-fetched.
        :param n_per_scene: Target candidates to emit per scene,
            split roughly evenly across the scene's eligible rooms.
        :param seed: Reserved for future use (seeded sampling).
        :returns: Candidates in deterministic scene / room order.
        """
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
        """Generate candidates for one scene, grounded on real objects.

        Visits every eligible room (in deterministic alphabetical order)
        and collects up to ``per_room_budget`` post-filtered candidates
        from each. Rooms are then interleaved round-robin so the
        per-scene cap at ``n_per_scene`` doesn't systematically skip
        the last rooms in iteration order.
        """
        dataset_idx = scene.get("procthor_dataset_index")
        if dataset_idx is None:
            log.warning(
                "scene %s has no procthor_dataset_index; skipping",
                scene.get("sample_id"),
            )
            return []
        house = self._house_lookup(int(dataset_idx))
        if house is None:
            log.warning(
                "scene %s: house_lookup(%d) returned None; skipping",
                scene.get("sample_id"),
                dataset_idx,
            )
            return []

        # Import here so harness-only tests don't need ai2thor.
        from harness.environments.procthor_utils import objects_by_room

        gt = objects_by_room(house)
        # Start from an alphabetical baseline so the iteration order
        # is reproducible regardless of set hash order, then rotate
        # by a stable per-scene offset. Without the rotation,
        # round-robin always fills the alphabetically-first rooms to
        # their extra-candidate quota before reaching later rooms, so
        # Bathroom/Bedroom end up over-represented and Kitchen/
        # LivingRoom under-represented. Hashing sample_id distributes
        # the "first slot" evenly across rooms over many scenes.
        base = sorted(r for r in ELIGIBLE_ROOMS if gt.get(r))
        if not base:
            log.info(
                "scene %s: no eligible rooms contain objects; skipping",
                scene.get("sample_id"),
            )
            return []
        offset = int(
            hashlib.md5(str(scene.get("sample_id", "")).encode()).hexdigest(), 16
        ) % len(base)
        eligible_rooms = base[offset:] + base[:offset]

        per_room_budget = max(1, -(-n_per_scene // len(eligible_rooms)))
        # Over-request 3× so the post-filter can drop leakers AND
        # gemma4's tendency to converge on the same few items leaves
        # at least ``per_room_budget`` distinct survivors.
        over_request = per_room_budget * 3

        room_candidates: Dict[str, List[CandidateQuestion]] = {}
        for room_type in eligible_rooms:
            gt_types = gt.get(room_type, [])
            excluded = self._excluded.setdefault(room_type, set())
            absent_items = self._request_absent_items(
                room_type, gt_types, excluded, n=over_request
            )
            kept: List[CandidateQuestion] = []
            for item in absent_items:
                if _probe_object_leaks(item, gt_types):
                    log.debug(
                        "DRM: dropping %r as leaking against ground-truth room=%s",
                        item,
                        room_type,
                    )
                    continue
                key = item.strip().lower()
                if key in excluded:
                    log.debug(
                        "DRM: dropping duplicate %r against prior probes for room=%s",
                        item,
                        room_type,
                    )
                    continue
                kept.append(
                    self._build_candidate(
                        scene=scene,
                        room_type=room_type,
                        probe_object=item,
                        gt_types=gt_types,
                    )
                )
                excluded.add(key)
                if len(kept) >= per_room_budget:
                    break
            room_candidates[room_type] = kept

        # Round-robin: take the i-th candidate from each room in turn.
        # Ensures all rooms are represented before any is deepened,
        # which matters when ``n_per_scene`` is not a multiple of
        # ``len(eligible_rooms)``.
        out: List[CandidateQuestion] = []
        max_len = max((len(v) for v in room_candidates.values()), default=0)
        for i in range(max_len):
            for room_type in eligible_rooms:
                lst = room_candidates[room_type]
                if i < len(lst):
                    out.append(lst[i])
                    if len(out) >= n_per_scene:
                        return out
        return out

    def _request_absent_items(
        self, room_type: str, gt_types: List[str], excluded: Set[str], n: int
    ) -> List[str]:
        """Call the LLM for ``n`` absent DRM probe objects in ``room_type``."""
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
            object_list=_format_object_list(gt_types),
            avoid_block=avoid_block,
            n=n,
        )
        raw = self._llm_chat(prompt)
        items = _parse_absent_items(raw)
        if not items:
            log.warning(
                "DRM: no items parsed from LLM for room=%s; raw[:120]=%r",
                room_type,
                (raw or "")[:120],
            )
        return items[:n]

    def _build_candidate(
        self,
        scene: Dict[str, Any],
        room_type: str,
        probe_object: str,
        gt_types: List[str],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion."""
        sample_id = str(scene.get("sample_id", "unknown"))
        room_phrase = _ROOM_PHRASE[room_type]
        probe_lower = _probe_for_question(probe_object)
        article = _indefinite_article(probe_lower)
        qid = (
            f"drm_{_sanitize_id_fragment(sample_id)}_"
            f"{_sanitize_id_fragment(room_type)}_"
            f"{_sanitize_id_fragment(probe_lower)}"
        )
        # Keep the ground-truth list compact by deduping; reviewers
        # see it in the `paradigm_metadata` block.
        unique_gt = list(dict.fromkeys(gt_types))
        return CandidateQuestion(
            question_id=qid,
            question=f"Did you see {article} {probe_lower} in the {room_phrase}?",
            answer="no",
            category="drm",
            paradigm=self.name,
            scene_ids=[sample_id],
            tools_expected=["semantic_search", "entity_query"],
            paradigm_metadata={
                "room_type": room_type,
                "probe_object": probe_lower,
                "ground_truth_objects_in_room": unique_gt,
            },
            generator_notes=(
                f"Auto-generated DRM probe. Target object is "
                f"schema-typical for {room_phrase} but not in the "
                f"simulator's ground-truth object list for this room."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Return the pre-filter rubric for DRM probes.

        Pre-filter is intentionally permissive: it catches only the
        obviously-broken candidates (structural items, abstract
        categories, non-schema-typical objects). Borderline cases
        default to ambiguous so the human reviewer makes the final
        call on fungibility / weak-confabulation-target.
        """
        return (
            "You are pre-filtering benchmark questions before a human "
            "reviewer sees them. Your job is to catch obviously-bad "
            "questions — NOT to do the human's precision filtering.\n\n"
            "The questions are DRM false-recall probes asking 'Did "
            "you see <object> in the <room>?' with ground-truth "
            "answer 'no'. The probe object is supposed to be "
            "schematically typical of <room> but absent from this "
            "specific scene. The benchmark tests whether a memory "
            "system confabulates typical-but-absent objects.\n\n"
            "Answer NO if the probe object clearly fails ANY of "
            "these criteria:\n"
            "  * STRUCTURAL: a wall, floor, ceiling, doorway, "
            "window, light fixture, or other room-spanning feature.\n"
            "  * ABSTRACT CATEGORY: a category-noun like 'decor', "
            "'dishware', 'appliance', 'furniture', 'utensils'.\n"
            "  * FUNGIBLE COMMODITY: a small everyday item that "
            "almost any home has, making the 'no' answer suspect. "
            "Examples that MUST be answered no: 'pen', 'pencil', "
            "'paper', 'key chain', 'key', 'watch', 'wallet', "
            "'credit card', 'phone', 'cell phone', 'book', "
            "'magazine', 'plate', 'bowl', 'cup', 'mug', 'fork', "
            "'knife', 'spoon', 'cutlery', 'towel', 'soap', 'tissue', "
            "'shampoo', 'brush', 'comb', 'remote control', "
            "'house plant', 'vase', 'pillow', 'blanket'.\n"
            "  * NEVER TYPICAL OF THE ROOM: a clearly out-of-place "
            "object such as 'motorcycle in bedroom', 'microscope in "
            "bathroom', 'fishing rod in kitchen'. The probe must be "
            "something a normal person would PLAUSIBLY EXPECT to "
            "find in the named room type.\n\n"
            "Answer YES when the probe object is a discrete, "
            "concrete item that is schema-typical of the room AND "
            "not a fungible commodity. Examples that should pass: "
            "'toaster' in a kitchen, 'shower curtain' in a "
            "bathroom, 'duvet' in a bedroom, 'sofa' in a living "
            "room, 'fireplace' in a living room, 'lamp' in a "
            "bedroom, 'mirror' in a bathroom, 'bookshelf' in a "
            "living room, 'oven' in a kitchen, 'bathtub' in a "
            "bathroom.\n\n"
            "Answer AMBIGUOUS when the object is borderline — "
            "could be argued either as a fungible commodity or a "
            "distinctive item, or wording is unclear.\n\n"
            "Answer with exactly one word: yes, ambiguous, or no."
        )


def make_prior_house_lookup() -> Callable[[int], Optional[Dict[str, Any]]]:
    """Build a house-lookup closure backed by ProcTHOR-10K (train split).

    Loads the dataset once (cached on disk after the first pull) and
    returns a ``idx -> house_dict`` function for
    :class:`DRMGenerator`'s ``house_lookup`` param. Out-of-range
    indices return ``None``.
    """
    import prior

    dataset = prior.load_dataset("procthor-10k")
    train = dataset["train"]
    n = len(train)

    def lookup(idx: int) -> Optional[Dict[str, Any]]:
        if idx < 0 or idx >= n:
            return None
        return train[idx]

    return lookup
