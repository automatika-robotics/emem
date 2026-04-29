"""Source-monitoring paradigm generator.

Source monitoring (Johnson, Hashtroudi & Lindsay, 1993) tests whether
a memory system remembers WHERE a piece of information came from, not
just the information itself. The classic finding is that humans
confuse sources — even when they remember the fact correctly, they
misattribute its origin.

For an embodied robot the analogue is multi-stream perception: a
running agent simultaneously consumes a visual-language scene
description and an object-detection feed (in our pipeline:
``vlm`` text from a VLM, ``object_detection`` text from a ProcTHOR
ground-truth inventory presented in the same comma-separated form a
real on-robot detector emits). After ingesting both streams the
agent is probed about whether a particular fact came from the scene
description or the object detector.

The probe is a two-way forced choice::

    "Did you learn about the <item> from the scene description or
    from the object detection?"

with answer one of ``"scene description"`` / ``"object detection"``.

Ground truth: the probe ``<item>`` must be uniquely attributable to
exactly ONE of the two streams in this scene's trajectory. We pick
items whose tokens appear in only one stream's corpus.

For the object-detection-only pool we can derive candidates
mechanically from the existing ground-truth object lists. For the
scene-description-only pool we ask the LLM to extract concrete
phrases mentioned in the captions that do NOT appear in the
detection feed; the answer is then validated against the corpus.
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
    _parse_absent_items,
    _probe_for_question,
    _sanitize_id_fragment,
    _tokenize_objecttype,
)

log = logging.getLogger(__name__)

SOURCE_VLM = "scene description"
SOURCE_OD = "object detection"

_VLM_LAYER_KEY = "vlm"
_OD_LAYER_KEY = "object_detection"


_VLM_PICKER_PROMPT = """\
You are helping build a cognitive-psychology benchmark for embodied \
robots. The source-monitoring paradigm tests whether the robot \
remembers WHICH sensor told it about a fact.

Below is the verbatim text of two perception streams produced as the \
robot toured one house: the SCENE DESCRIPTION stream (free-form \
captions of what the robot saw) and the OBJECT DETECTION stream \
(comma-separated object names emitted by a detector).

== Scene description stream ==
{vlm_corpus}

== Object detection stream ==
{od_corpus}
{avoid_block}
Your task: pick exactly {n} concrete things mentioned in the scene \
description stream that DO NOT appear in the object detection \
stream. These are facts the robot would have learned ONLY from the \
caption, not from the detector.

Pass: descriptive details the detector would never emit ('blue \
flooring', 'pale green walls', 'minimalist arrangement', 'wooden \
chairs', 'framed pictures'). Fail: words that ARE in the detection \
list, generic structural items ('wall', 'floor', 'ceiling'), and \
abstract qualifiers without a referent ('the room', 'the space').

Each pick must be a short noun phrase (1-3 words), copyable verbatim \
from the scene description text. Return ONLY a JSON array of \
strings, no explanation, no markdown fences. Example format:
["blue flooring", "framed pictures", "pale green walls"]
"""

_OD_PICKER_PROMPT = """\
You are helping build a cognitive-psychology benchmark for embodied \
robots. The source-monitoring paradigm tests whether the robot \
remembers WHICH sensor told it about a fact.

Below are two perception streams produced as the robot toured one \
house: the SCENE DESCRIPTION (free-form VLM captions) and the \
OBJECT DETECTION feed (ground-truth object types, comma-separated).

== Scene description stream ==
{vlm_corpus}

== Object detection stream ==
{od_corpus}
{avoid_block}
Your task: pick exactly {n} object names from the object detection \
feed that DO NOT appear in the scene description stream. These are \
items the robot would have learned ONLY from the detector, not from \
the VLM caption.

Pass: items present in the detection feed whose name (or any of its \
words) is absent from the caption text. Fail: items already \
mentioned in the caption (e.g. caption says "wooden table" and \
detector says "dining table" — the word "table" overlaps, so skip), \
overly common structural items ('wall', 'floor', 'ceiling').

Each pick must be a short object phrase copied verbatim from the \
detection list. Return ONLY a JSON array of strings, no \
explanation, no markdown fences. Example format:
["alarm clock", "dog bed", "credit card"]
"""

_AVOID_BLOCK = """

You have already picked these items for source-monitoring probes \
in this dataset — DO NOT pick any of them again:
{avoid_list}
"""


def _question_id(sample_id: str, source: str, probe_object: str) -> str:
    """Stable question id."""
    src_short = "vlm" if source == SOURCE_VLM else "od"
    return (
        f"srcmon_{_sanitize_id_fragment(sample_id)}_"
        f"{src_short}_{_sanitize_id_fragment(probe_object)}"
    )


def _join_layer_corpus(trajectory: List[Dict[str, Any]], layer_key: str) -> str:
    """Concatenate every frame's text for ``layer_key`` into one blob."""
    parts: List[str] = []
    for frame in trajectory or []:
        text = (frame.get("layers") or {}).get(layer_key) or ""
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _od_object_list(trajectory: List[Dict[str, Any]]) -> List[str]:
    """Return the union of object names from every frame's OD feed.

    Each ``object_detection`` layer text is comma-separated; we split
    and dedupe (preserving first-seen order) so the picker prompt
    sees a clean list.
    """
    seen: List[str] = []
    seen_set: Set[str] = set()
    for frame in trajectory or []:
        text = (frame.get("layers") or {}).get(_OD_LAYER_KEY) or ""
        for token in str(text).split(","):
            t = token.strip()
            if not t:
                continue
            key = t.lower()
            if key in seen_set:
                continue
            seen_set.add(key)
            seen.append(t)
    return seen


def _phrase_tokens(text: str) -> Set[str]:
    """Tokens of a VLM phrase (length-3+, lowercased, alphanumeric)."""
    return _tokenize_objecttype(text)


class SourceMonitoringGenerator(ParadigmGenerator):
    """Emit source-monitoring probes for each scene.

    :param llm_chat: Pluggable ``(prompt: str) -> str`` callable.
    :param target_total: Global cap on emitted candidates.
    :param exclude_probes: Optional per-source set of probe items to
        avoid (supplemental runs across scenes).
    """

    name = "source_monitoring"

    def __init__(
        self,
        llm_chat: Callable[[str], str],
        target_total: int = 120,
        exclude_probes: Optional[Dict[str, Set[str]]] = None,
    ):
        self._llm_chat = llm_chat
        self._target_total = target_total
        self._excluded: Dict[str, Set[str]] = {
            src: {p.lower() for p in probes}
            for src, probes in (exclude_probes or {}).items()
        }

    def generate(
        self,
        scenes: List[Dict[str, Any]],
        *,
        n_per_scene: int = 6,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Produce source-monitoring candidates across ``scenes``."""
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
        trajectory = scene.get("trajectory") or []
        if not trajectory:
            log.info("scene %s has no trajectory; skipping", scene.get("sample_id"))
            return []

        vlm_corpus = _join_layer_corpus(trajectory, _VLM_LAYER_KEY)
        od_objects = _od_object_list(trajectory)
        if not vlm_corpus or not od_objects:
            log.info(
                "scene %s missing one stream (vlm=%d, od=%d); skipping",
                scene.get("sample_id"),
                len(vlm_corpus),
                len(od_objects),
            )
            return []
        od_corpus = ", ".join(od_objects)

        vlm_tokens = _phrase_tokens(vlm_corpus)

        sample_id = str(scene.get("sample_id", "unknown"))
        sources = [SOURCE_VLM, SOURCE_OD]
        # Hash-rotate which source goes first per scene so neither
        # always burns the budget alone.
        offset = int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % len(sources)
        ordered_sources = sources[offset:] + sources[:offset]
        per_source_budget = max(1, -(-n_per_scene // len(ordered_sources)))
        over_request = per_source_budget * 3

        per_source: Dict[str, List[CandidateQuestion]] = {}
        for source in ordered_sources:
            per_source[source] = self._candidates_for_source(
                scene=scene,
                source=source,
                vlm_corpus=vlm_corpus,
                od_corpus=od_corpus,
                vlm_tokens=vlm_tokens,
                od_objects=od_objects,
                budget=per_source_budget,
                over_request=over_request,
            )

        # Round-robin across sources for balance.
        out: List[CandidateQuestion] = []
        max_len = max((len(v) for v in per_source.values()), default=0)
        for i in range(max_len):
            for source in ordered_sources:
                lst = per_source[source]
                if i < len(lst):
                    out.append(lst[i])
                    if len(out) >= n_per_scene:
                        return out
        return out

    def _candidates_for_source(
        self,
        *,
        scene: Dict[str, Any],
        source: str,
        vlm_corpus: str,
        od_corpus: str,
        vlm_tokens: Set[str],
        od_objects: List[str],
        budget: int,
        over_request: int,
    ) -> List[CandidateQuestion]:
        """Ask the LLM for picks attributed to one source, validate, build."""
        excluded = self._excluded.setdefault(source, set())
        avoid_block = (
            _AVOID_BLOCK.format(
                avoid_list="\n".join(f"  - {p}" for p in sorted(excluded))
            )
            if excluded
            else ""
        )
        if source == SOURCE_VLM:
            prompt = _VLM_PICKER_PROMPT.format(
                vlm_corpus=vlm_corpus,
                od_corpus=od_corpus,
                avoid_block=avoid_block,
                n=over_request,
            )
        else:
            prompt = _OD_PICKER_PROMPT.format(
                vlm_corpus=vlm_corpus,
                od_corpus=od_corpus,
                avoid_block=avoid_block,
                n=over_request,
            )
        raw = self._llm_chat(prompt)
        picks = _parse_absent_items(raw)
        if not picks:
            log.warning(
                "source_monitoring: no items parsed for source=%s; raw[:120]=%r",
                source,
                (raw or "")[:120],
            )

        od_tokens_by_obj = {obj: _phrase_tokens(obj) for obj in od_objects}
        od_token_union: Set[str] = set()
        for toks in od_tokens_by_obj.values():
            od_token_union.update(toks)

        kept: List[CandidateQuestion] = []
        for pick in picks[:over_request]:
            pl = pick.strip().lower()
            if not pl or pl in excluded:
                continue
            if not self._is_unique_to_source(
                source=source,
                pick=pick,
                vlm_corpus=vlm_corpus,
                vlm_tokens=vlm_tokens,
                od_objects=od_objects,
                od_token_union=od_token_union,
            ):
                log.debug(
                    "source_monitoring: dropping %r — not unique to %s",
                    pick,
                    source,
                )
                continue
            kept.append(
                self._build_candidate(
                    scene=scene,
                    source=source,
                    probe_object=pick,
                    od_objects=od_objects,
                )
            )
            excluded.add(pl)
            if len(kept) >= budget:
                break
        return kept

    @staticmethod
    def _is_unique_to_source(
        *,
        source: str,
        pick: str,
        vlm_corpus: str,
        vlm_tokens: Set[str],
        od_objects: List[str],
        od_token_union: Set[str],
    ) -> bool:
        """Return True iff ``pick`` lives in exactly the named source.

        For ``SOURCE_VLM`` picks: the pick must appear textually in
        the VLM corpus AND its tokens must NOT overlap the OD object-
        name token union.

        For ``SOURCE_OD`` picks: the pick must match an entry in the
        OD inventory (case-insensitive equality on a comma-stripped
        object name) AND its tokens must NOT overlap the VLM corpus
        token set.
        """
        ptoks = _phrase_tokens(pick)
        if not ptoks:
            return False
        if source == SOURCE_VLM:
            if pick.lower() not in vlm_corpus.lower():
                return False
            return not (ptoks & od_token_union)
        # SOURCE_OD
        od_lower = {o.lower() for o in od_objects}
        if pick.lower() not in od_lower:
            return False
        return not (ptoks & vlm_tokens)

    def _build_candidate(
        self,
        scene: Dict[str, Any],
        source: str,
        probe_object: str,
        od_objects: List[str],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion."""
        sample_id = str(scene.get("sample_id", "unknown"))
        probe_lower = _probe_for_question(probe_object)
        # ``the`` reads correctly for singular ("the cell phone"), plural
        # ("the pale green walls"), and mass nouns ("the stereo
        # equipment"); ``a/an`` only works for singular count nouns.
        question = (
            f"Did you learn about the {probe_lower} from the scene "
            "description or from the object detection?"
        )
        return CandidateQuestion(
            question_id=_question_id(sample_id, source, probe_lower),
            question=question,
            answer=source,
            category="source_monitoring",
            paradigm=self.name,
            scene_ids=[sample_id],
            tools_expected=["semantic_search", "entity_query"],
            paradigm_metadata={
                "source": source,
                "probe_object": probe_lower,
                "object_detection_inventory": od_objects,
            },
            generator_notes=(
                f"Auto-generated source-monitoring probe. The item is "
                f"unique to the {source} stream of this scene; the "
                f"agent must remember which sensor reported it."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Pre-filter rubric for source-monitoring candidates."""
        return (
            "You are pre-filtering benchmark questions before a human "
            "reviewer sees them. Your job is to catch obviously-bad "
            "questions — NOT to do the human's precision filtering.\n\n"
            "The questions are source-monitoring probes asking 'Did "
            "you learn about <item> from the scene description or "
            "from the object detection?' with ground-truth answer "
            "one of: 'scene description', 'object detection'.\n\n"
            "Answer NO if the probe item clearly fails ANY of these "
            "criteria:\n"
            "  * STRUCTURAL: a wall, floor, ceiling, doorway, "
            "window, light fixture, generic 'room' / 'space' / "
            "'area'.\n"
            "  * NONSENSICAL: doesn't make sense as a discrete "
            "perceived item.\n"
            "  * AMBIGUOUS WORDING: the phrase is so generic it "
            "could plausibly come from either stream.\n\n"
            "Answer YES when the probe item is a discrete, concrete "
            "thing whose stream attribution is unambiguous.\n\n"
            "Answer AMBIGUOUS when the item is borderline.\n\n"
            "Answer with exactly one word: yes, ambiguous, or no."
        )
