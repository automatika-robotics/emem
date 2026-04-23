"""DRM false-recall paradigm generator.

For each scene, aggregates observations by ``room_type`` and asks an
LLM for schema-typical objects that the robot did **not** see. Each
returned object becomes a probe ``"Did you see a <object> in the
<room>?"`` with ground-truth answer ``"no"``. A memory system that
confabulates typical-but-absent objects (the "DRM intrusion"
phenomenon from human memory research) fails these probes.

The generator does not apply a pre-filter to its own output. Per the
revision plan, DRM is the first paradigm and is hand-reviewed without
a pre-filter so the human-reviewer decisions calibrate the rubric
for the remaining five paradigms.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Set

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
    ParadigmGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.scene_entries import (
    group_waypoints_by_room,
)

log = logging.getLogger(__name__)

# Room types eligible for DRM probes. ProcTHOR-10K exposes exactly
# these four; all are schema-rich enough to support typical-object
# probes. Agent-perceived room labels ("outside", doorway tags) are
# excluded via this allow-list so we don't generate nonsense probes.
ELIGIBLE_ROOMS: Set[str] = {"Kitchen", "Bedroom", "Bathroom", "LivingRoom"}

# Room-type → display phrase used when we address the user ("in the
# kitchen", "in the living room"). LivingRoom has to be split.
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
were never actually observed.

A robot explored a {room_phrase}. Here are the observations it made:

== Object detections (what the detector saw) ==
{detections}

== Scene descriptions (what the VLM saw) ==
{vlm_descriptions}

Your task: produce exactly {n} "DRM probe objects" for this \
{room_phrase} — items that are TYPICAL of a {room_phrase} but were \
NOT mentioned in the observations above. Each item must satisfy ALL \
of the following:
  * be a single concrete object (a noun phrase of 1–3 words)
  * be strongly schema-typical of a {room_phrase}
  * MUST NOT appear in the observations above, even as a partial \
match, synonym, or superset. For example, if the observations \
mention "table", you may not propose "coffee table", "dining \
table", or "wooden table". If they mention "chair", you may not \
propose "armchair" or "rocking chair".
  * be diverse across the {n} items (no near-duplicates, no \
items that differ only by a modifier)

Before returning each item, silently check the observations for any \
word or noun phrase that overlaps with your candidate; if there is \
ANY overlap, discard it and try another.

Return ONLY a JSON array of strings, no explanation, no markdown \
fences. Example format: ["item one", "item two", "item three"]
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


def _room_observation_blob(aggregate: Dict[str, Any]) -> str:
    """Return a single lowercased string concatenating every observation
    string the room produced (detections + VLM descriptions + place).

    Used by :func:`_probe_object_leaks` to detect when an LLM-proposed
    absent object actually was mentioned in the observations.
    """
    parts: List[str] = []
    for key in ("detections", "vlm_descriptions", "places"):
        items = aggregate.get(key) or []
        parts.extend(str(x) for x in items)
    return " ".join(parts).lower()


def _probe_object_leaks(probe_object: str, observation_blob: str) -> bool:
    """Return True if ``probe_object`` (or any of its tokens) appears
    in ``observation_blob``.

    We split the probe object into alphanumeric word tokens of length
    ≥3 and check whether any of them is a word-boundary match in the
    observation blob. So ``"coffee table"`` leaks against a blob that
    contains either ``"coffee"`` or ``"table"``; ``"alarm clock"``
    leaks against ``"clock"`` or ``"alarm"``. Single-token shorthand
    (e.g. ``"tv"``) is handled by the length ≥3 filter — ``"tv"``
    won't false-positive against ``"television"``.
    """
    blob = observation_blob.lower()
    tokens = [t for t in re.findall(r"[a-zA-Z]+", probe_object.lower()) if len(t) >= 3]
    for tok in tokens:
        if re.search(rf"\b{re.escape(tok)}\b", blob):
            return True
    return False


class DRMGenerator(ParadigmGenerator):
    """Emit DRM probe questions from scene manifest entries.

    :param llm_chat: Pluggable ``(prompt: str) -> str`` callable that
        produces the LLM's raw response. Tests pass a stub; production
        wires ``OllamaLLMClient._chat`` or ``GeminiLLMClient._generate``.
    :param target_total: Total number of candidates to emit across
        all scenes. The plan calls for ~2× the curated target (60),
        so 120 is the default; lower for smoke tests.
    """

    name = "drm"

    def __init__(self, llm_chat: Callable[[str], str], target_total: int = 120):
        self._llm_chat = llm_chat
        self._target_total = target_total

    def generate(
        self,
        scenes: List[Dict[str, Any]],
        *,
        n_per_scene: int = 6,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Produce DRM candidates across ``scenes``.

        :param scenes: Merged scene entries from
            :func:`load_scene_entries` — each must have a ``trajectory``
            key with room-tagged waypoints.
        :param n_per_scene: Target candidates to emit per scene, split
            roughly evenly across the scene's eligible rooms.
        :param seed: Unused for now (kept for contract parity with the
            scheduled-paradigm generators that need a seeded sampler).
        :returns: Candidates in deterministic scene / room order.
        """
        del seed  # reserved
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
        trajectory = scene.get("trajectory") or []
        per_room = group_waypoints_by_room(trajectory)
        eligible = {r: agg for r, agg in per_room.items() if r in ELIGIBLE_ROOMS}
        if not eligible:
            log.info("scene %s: no eligible rooms for DRM", scene.get("sample_id"))
            return []

        # Divide the per-scene budget across eligible rooms; round up
        # so small budgets still produce at least one item per room.
        # Over-request by 2× per room to compensate for the post-filter
        # drop rate (empirically ~10% of LLM-proposed items leak).
        per_room_budget = max(1, -(-n_per_scene // len(eligible)))
        over_request = per_room_budget * 2
        out: List[CandidateQuestion] = []
        for room_type, aggregate in eligible.items():
            absent_items = self._request_absent_items(
                room_type, aggregate, n=over_request
            )
            room_blob = _room_observation_blob(aggregate)
            kept_this_room = 0
            for item in absent_items:
                if _probe_object_leaks(item, room_blob):
                    log.debug(
                        "DRM: dropping %r as leaking against room=%s observations",
                        item,
                        room_type,
                    )
                    continue
                cand = self._build_candidate(
                    scene=scene,
                    room_type=room_type,
                    probe_object=item,
                    aggregate=aggregate,
                )
                out.append(cand)
                kept_this_room += 1
                if kept_this_room >= per_room_budget or len(out) >= n_per_scene:
                    break
            if len(out) >= n_per_scene:
                return out
        return out

    def _request_absent_items(
        self, room_type: str, aggregate: Dict[str, Any], n: int
    ) -> List[str]:
        """Call the LLM for ``n`` absent DRM probe objects in ``room_type``."""
        prompt = _GENERATION_PROMPT.format(
            room_phrase=_ROOM_PHRASE[room_type],
            detections=_format_list(aggregate.get("detections") or []),
            vlm_descriptions=_format_list(aggregate.get("vlm_descriptions") or []),
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
        aggregate: Dict[str, Any],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion."""
        sample_id = str(scene.get("sample_id", "unknown"))
        room_phrase = _ROOM_PHRASE[room_type]
        qid = (
            f"drm_{_sanitize_id_fragment(sample_id)}_"
            f"{_sanitize_id_fragment(room_type)}_"
            f"{_sanitize_id_fragment(probe_object)}"
        )
        return CandidateQuestion(
            question_id=qid,
            question=f"Did you see a {probe_object} in the {room_phrase}?",
            answer="no",
            category="drm",
            paradigm=self.name,
            scene_ids=[sample_id],
            tools_expected=["semantic_search", "entity_query"],
            paradigm_metadata={
                "room_type": room_type,
                "probe_object": probe_object,
                "n_waypoints_in_room": aggregate.get("n_waypoints", 0),
                "detections_sample": (aggregate.get("detections") or [])[:3],
            },
            generator_notes=(
                f"Auto-generated DRM probe. Target object is "
                f"schema-typical for {room_phrase} but absent from "
                f"the scene's observations."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Return the pre-filter rubric.

        Not used in A14b.2 (DRM is hand-reviewed without pre-filter so
        reviewer decisions calibrate this rubric for the remaining
        five paradigms) — kept here so later paradigms can follow the
        same yes / ambiguous / no contract.
        """
        return (
            "You are judging whether a benchmark question tests DRM "
            "false-recall.\n\n"
            "A DRM probe must:\n"
            "  (1) ask whether a specific object was seen in a specific "
            "room,\n"
            "  (2) pick an object that is schematically typical of that "
            "room type but not actually present in the scene's "
            "observations, and\n"
            "  (3) have ground-truth answer 'no'.\n\n"
            "If all three hold, answer yes. If the probe object is "
            "too generic / too ambiguous / could plausibly have been "
            "there, answer ambiguous. If the question fails any "
            "criterion, answer no."
        )


def _format_list(items: List[str], cap: int = 40) -> str:
    """Format a list of strings as a bulleted block, capped at ``cap``.

    Keeps the generation prompt short even for very long trajectories
    (~30 waypoints per room would otherwise swamp the LLM's context).
    """
    if not items:
        return "  (none)"
    shown = items[:cap]
    more = f"\n  ... (+{len(items) - cap} more)" if len(items) > cap else ""
    return "\n".join(f"  - {s}" for s in shown) + more
