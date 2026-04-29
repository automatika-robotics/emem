"""Propagate per-item review decisions across all retention-decay delay buckets.

Retention-decay generation emits 6 candidates per (scene, item) pair
(one per delay bucket: 1h / 1d / 1wk / 1mo / 6mo / 1yr) with
identical question text. The human review pass operates on a deduped
view (one row per unique item) for efficiency. After review, this
script fans the keep / discard decision back onto all 6 delay
buckets in the original 468-candidate file, producing the full
``*.curated.jsonl`` and ``*.rejected.jsonl`` the benchmark consumes.

Run after the reviewer has produced
``retention_interval_decay.candidates.dedupe.curated.jsonl`` and
``retention_interval_decay.candidates.dedupe.rejected.jsonl``.

Usage::

    python -m harness.benchmarks.academic.emem_bench_v1.paradigms.retention_propagate \\
      --candidates data/emem-bench-v1/paradigms/retention_interval_decay.candidates.jsonl \\
      --dedupe-curated data/emem-bench-v1/paradigms/retention_interval_decay.candidates.dedupe.curated.jsonl \\
      --dedupe-rejected data/emem-bench-v1/paradigms/retention_interval_decay.candidates.dedupe.rejected.jsonl \\
      --out-curated data/emem-bench-v1/paradigms/retention_interval_decay.curated.jsonl \\
      --out-rejected data/emem-bench-v1/paradigms/retention_interval_decay.rejected.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _load(path: Path) -> List[Dict]:
    """Load a jsonl file, ignoring blank lines."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(path: Path, rows: List[Dict]) -> None:
    """Write rows to a jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _key(cand: Dict) -> Tuple[str, str]:
    """Return the (scene_id, probe_object) dedupe key for a candidate."""
    return (
        cand["scene_ids"][0],
        cand["paradigm_metadata"]["probe_object"],
    )


def propagate(
    candidates_path: Path,
    dedupe_curated_path: Path,
    dedupe_rejected_path: Path,
    out_curated_path: Path,
    out_rejected_path: Path,
) -> Tuple[int, int, int]:
    """Fan dedupe-level decisions onto every delay-bucket candidate.

    Each (scene, item) decision in the dedupe-curated / -rejected
    files is applied to all 6 corresponding rows in ``candidates_path``.
    Candidates whose (scene, item) pair appears in neither dedupe
    file are left out of both outputs (treated as un-reviewed).

    :returns: ``(n_curated, n_rejected, n_unreviewed)``.
    """
    candidates = _load(candidates_path)
    dedupe_curated = _load(dedupe_curated_path)
    dedupe_rejected = _load(dedupe_rejected_path)

    keep_keys: Set[Tuple[str, str]] = {_key(c) for c in dedupe_curated}
    reject_keys: Set[Tuple[str, str]] = {_key(c) for c in dedupe_rejected}

    overlap = keep_keys & reject_keys
    if overlap:
        raise ValueError(
            f"{len(overlap)} (scene, item) keys appear in BOTH dedupe-"
            f"curated and dedupe-rejected; review files are inconsistent. "
            f"Examples: {sorted(overlap)[:3]}"
        )

    kept: List[Dict] = []
    rejected: List[Dict] = []
    unreviewed = 0
    for c in candidates:
        k = _key(c)
        if k in keep_keys:
            kept.append(c)
        elif k in reject_keys:
            rejected.append(c)
        else:
            unreviewed += 1

    _save(out_curated_path, kept)
    _save(out_rejected_path, rejected)
    return len(kept), len(rejected), unreviewed


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--dedupe-curated", required=True, type=Path)
    parser.add_argument("--dedupe-rejected", required=True, type=Path)
    parser.add_argument("--out-curated", required=True, type=Path)
    parser.add_argument("--out-rejected", required=True, type=Path)
    args = parser.parse_args(argv)
    n_kept, n_rej, n_un = propagate(
        args.candidates,
        args.dedupe_curated,
        args.dedupe_rejected,
        args.out_curated,
        args.out_rejected,
    )
    print("propagated decisions:")
    print(f"  curated:    {n_kept}  -> {args.out_curated}")
    print(f"  rejected:   {n_rej}  -> {args.out_rejected}")
    print(f"  unreviewed: {n_un}   (left out of both outputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
