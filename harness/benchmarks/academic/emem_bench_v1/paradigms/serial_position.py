"""Serial-position paradigm generator (A14c.5).

Murdock (1962), Glanzer & Cunitz (1966): when humans encode a list
of items in sequence and recall them, performance is a U-shaped
function of position — better recall for early (primacy) and late
(recency) items, worse for the middle.

For an embodied memory system the U-curve is *not* the target — it
is a human limitation that arises from biological capacity
constraints. A robot intended for sustained operation should produce
**flat-and-high** recall across positions when the trajectory is
within its capacity envelope. A U-curve in this regime indicates
over-eager working-memory eviction or premature archival; a flat
curve indicates the architecture is in-capacity for the horizon.

The probe binning lets either reading be diagnostic; the paper notes
will be honest that we *want* flat curves for the within-capacity
regime and that long-horizon U-curves (if they emerge) fingerprint
the consolidation strategy.

Schedule shape (multi-house chained super-trajectory)::

    IngestPhase  — chain[0] trajectory  (early items live here)
    IngestPhase  — chain[1] trajectory  (buffer; no candidates here)
    IngestPhase  — chain[2] trajectory  (middle items)
    IngestPhase  — chain[3] trajectory  (buffer)
    IngestPhase  — chain[4] trajectory  (late items)
    ProbePhase   — "Did you see X early, middle, or late?"

No clock advance between houses — it's one continuous tour. No
consolidation fires during ingestion (we want raw retention pressure
to bind capacity, not artificial archival).

Items come from objects unique to a single room of one specific
chain-position house, so each item has a clean (chain, position,
room) ground truth.
"""

from __future__ import annotations

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

# Chain length and active-position layout. With CHAIN_LENGTH=5 and
# ACTIVE_POSITIONS=(0, 2, 4), houses at chain positions 1 and 3 are
# *buffer* houses — ingested into the schedule (so they contribute
# to capacity pressure) but not used as a source of candidates. This
# keeps the early/middle/late bins cleanly separated.
CHAIN_LENGTH = 5
ACTIVE_POSITIONS: Tuple[int, ...] = (0, 2, 4)
BIN_LABELS: Dict[int, str] = {0: "early", 2: "middle", 4: "late"}


def _question_id(
    chain_id: str, source_sample_id: str, probe_object: str, bin_label: str
) -> str:
    """Stable id for one (chain, item, bin) candidate."""
    return (
        f"serpos_{_sanitize_id_fragment(chain_id)}_"
        f"{_sanitize_id_fragment(source_sample_id)}_"
        f"{_sanitize_id_fragment(probe_object)}_"
        f"{_sanitize_id_fragment(bin_label)}"
    )


class SerialPositionGenerator(ParadigmGenerator):
    """Emit serial-position probes from chained-house super-trajectories.

    :param house_lookup: ``int -> house_dict`` from
        ``prior.load_dataset("procthor-10k")``.
    :param target_total: Global cap on emitted candidates.
    :param exclude_probes: Optional set of probe items to avoid
        across chains (keyed by ``"__serpos__"``).
    """

    name = "serial_position"

    def __init__(
        self,
        house_lookup: Callable[[int], Optional[Dict[str, Any]]],
        target_total: int = 120,
        exclude_probes: Optional[Dict[str, Set[str]]] = None,
        # Compatibility shim: cli factory passes ``llm_chat`` first.
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
        n_per_scene: int = 18,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Produce serial-position candidates by chaining scenes.

        ``n_per_scene`` is the *probe-question* budget per chain (not
        per single scene). At 6 items × 3 active positions, the
        default 18 fully populates a chain.
        """
        del seed
        out: List[CandidateQuestion] = []
        per_position = max(1, n_per_scene // len(ACTIVE_POSITIONS))
        # Chain scenes in non-overlapping windows of CHAIN_LENGTH.
        # Derive chain_id from the first scene's sample_id so each
        # chain has a unique identifier regardless of how the cli
        # batches calls to generate() (it dispatches one chain at a
        # time via scene_units, so a local ``enumerate`` always
        # restarts at 0 — using sample_id avoids that footgun).
        for start in range(0, len(scenes) - CHAIN_LENGTH + 1, CHAIN_LENGTH):
            if len(out) >= self._target_total:
                break
            chain = scenes[start : start + CHAIN_LENGTH]
            first_sid = str(chain[0].get("sample_id", f"unknown_{start}"))
            chain_id = f"chain_{_sanitize_id_fragment(first_sid)}"
            out.extend(
                self._candidates_for_chain(
                    chain=chain,
                    chain_id=chain_id,
                    per_position=per_position,
                )
            )
        return out[: self._target_total]

    def scene_units(self, scenes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group scenes into non-overlapping chains of CHAIN_LENGTH."""
        units: List[List[Dict[str, Any]]] = []
        for start in range(0, len(scenes) - CHAIN_LENGTH + 1, CHAIN_LENGTH):
            units.append(scenes[start : start + CHAIN_LENGTH])
        return units

    def _candidates_for_chain(
        self,
        chain: List[Dict[str, Any]],
        chain_id: str,
        per_position: int,
    ) -> List[CandidateQuestion]:
        """Emit candidates from each active position within one chain."""
        if len(chain) != CHAIN_LENGTH:
            return []
        out: List[CandidateQuestion] = []
        excluded = self._excluded.setdefault("__serpos__", set())
        for position in ACTIVE_POSITIONS:
            source_scene = chain[position]
            unique_per_room = self._unique_per_room(source_scene)
            if not unique_per_room:
                continue
            bin_label = BIN_LABELS[position]
            picks: List[Tuple[str, str]] = []
            # Round-robin across rooms to avoid one-room dominance.
            rooms_sorted = sorted(unique_per_room)
            i = 0
            while True:
                advanced = False
                for room in rooms_sorted:
                    items = unique_per_room[room]
                    if i < len(items):
                        picks.append((room, items[i]))
                        advanced = True
                if not advanced:
                    break
                i += 1
            kept = 0
            for room_type, item_type in picks:
                pl = _probe_for_question(item_type)
                if pl in excluded:
                    continue
                out.append(
                    self._build_candidate(
                        chain=chain,
                        chain_id=chain_id,
                        source_scene=source_scene,
                        position=position,
                        bin_label=bin_label,
                        room_type=room_type,
                        probe_object=item_type,
                        ground_truth_objects=unique_per_room[room_type],
                    )
                )
                excluded.add(pl)
                kept += 1
                if kept >= per_position:
                    break
        return out

    def _unique_per_room(self, scene: Dict[str, Any]) -> Dict[str, List[str]]:
        """Same machinery as pattern_completion / retention_decay."""
        sample_id = scene.get("sample_id")
        dataset_idx = scene.get("procthor_dataset_index")
        if dataset_idx is None:
            log.warning("scene %s has no procthor_dataset_index", sample_id)
            return {}
        house = self._house_lookup(int(dataset_idx))
        if house is None:
            log.warning("scene %s: house_lookup returned None", sample_id)
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
        chain: List[Dict[str, Any]],
        chain_id: str,
        source_scene: Dict[str, Any],
        position: int,
        bin_label: str,
        room_type: str,
        probe_object: str,
        ground_truth_objects: List[str],
    ) -> CandidateQuestion:
        """Format one CandidateQuestion."""
        source_sid = str(source_scene.get("sample_id", "unknown"))
        probe_lower = _probe_for_question(probe_object)
        question = (
            f"Did you see the {probe_lower} early, in the middle, or "
            "late in your tour of the houses?"
        )
        chain_sample_ids = [str(s.get("sample_id", "")) for s in chain]
        return CandidateQuestion(
            question_id=_question_id(chain_id, source_sid, probe_lower, bin_label),
            question=question,
            answer=bin_label,
            category="serial_position",
            paradigm=self.name,
            scene_ids=chain_sample_ids,
            tools_expected=["semantic_search", "entity_query", "temporal_query"],
            paradigm_metadata={
                "probe_object": probe_lower,
                "bin_label": bin_label,
                "chain_id": chain_id,
                "chain_sample_ids": chain_sample_ids,
                "source_house_id": source_sid,
                "position_in_chain": position,
                "chain_length": CHAIN_LENGTH,
                "room_type": room_type,
                "ground_truth_objects_in_room": sorted(set(ground_truth_objects)),
            },
            generator_notes=(
                f"Auto-generated serial-position probe. Item {probe_lower!r} "
                f"is unique to the {room_type} of source house "
                f"{source_sid!r}, which is at position {position} of "
                f"chain {chain_id!r} (positions 0/2/4 = early/middle/late "
                f"in a 5-house chain). The agent encoded the chain as one "
                f"continuous tour; correct answer is {bin_label!r}."
            ),
        )

    @classmethod
    def prefilter_rubric(cls) -> str:
        """Pre-filter rubric for serial-position candidates."""
        return (
            "You are pre-filtering benchmark questions before a human "
            "reviewer sees them. Your job is to catch obviously-bad "
            "questions — NOT to do the human's precision filtering.\n\n"
            "The questions are serial-position probes asking 'Did you "
            "see the <item> early, in the middle, or late in your tour "
            "of the houses?' The ground-truth answer is one of: early, "
            "middle, late.\n\n"
            "Answer NO if the candidate clearly fails ANY of these "
            "criteria:\n"
            "  * STRUCTURAL: a wall, floor, ceiling, doorway, "
            "window, light fixture.\n"
            "  * ABSTRACT CATEGORY: 'decor', 'utensils', "
            "'furniture'.\n"
            "  * NONSENSICAL: doesn't make sense as a discrete "
            "perceived object.\n\n"
            "Answer YES when the probe item is a discrete, concrete, "
            "memorable object whose recall position can meaningfully "
            "be tested.\n\n"
            "Answer AMBIGUOUS when borderline.\n\n"
            "Answer with exactly one word: yes, ambiguous, or no."
        )
