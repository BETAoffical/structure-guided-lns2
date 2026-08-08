#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.closed_loop_trace_storage import read_trace_events  # noqa: E402
from experiments.repair_collection import (  # noqa: E402
    _fingerprint,
    _read_jsonl,
    _write_json,
)


SCHEMA = "lns2.stride.structpool_runtime_equivalence.v1"
SUMMARY_FIELDS = (
    "success",
    "repairable",
    "truncated",
    "external_timeout",
    "initial_fingerprint",
    "initial_conflicts",
    "final_conflicts",
    "conflict_trajectory",
    "conflict_auc",
    "fixed_budget_conflict_auc",
    "repair_iterations",
    "final_low_level",
    "final_sum_of_costs",
    "invalid_action_count",
    "fingerprint_mismatch_count",
    "selected_size_counts",
    "selected_family_counts",
)
TRANSITION_FIELDS = (
    "decision_index",
    "before_fingerprint",
    "after_fingerprint",
    "action",
    "low_level_delta",
    "terminated",
    "truncated",
)
METRIC_FIELDS = (
    "action_valid",
    "applied_heuristic",
    "applied_pp_random_seed",
    "conflicts_before",
    "conflicts_after",
    "conflict_delta",
    "generated",
    "iteration",
    "neighborhood",
    "repair_order",
    "replan_success",
    "requested_heuristic",
    "requested_mode",
    "requested_pp_random_seed",
    "requested_random_seed",
    "requested_repair_order",
    "sum_of_costs_before",
    "sum_of_costs_after",
)


def _runtime_key(name: str) -> bool:
    return (
        name.endswith("_seconds")
        or name.endswith("_timings")
        or name
        in {
            "controller_seconds_before_repair",
            "controller_runtime",
            "elapsed_wall_seconds",
            "native_timing_schema",
            "within_wall_budget",
        }
    )


def _without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_runtime(item)
            for key, item in value.items()
            if not _runtime_key(str(key))
        }
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


def _transition_signature(row: dict[str, Any]) -> dict[str, Any]:
    signature = {name: row.get(name) for name in TRANSITION_FIELDS}
    metrics = dict(row.get("metrics") or {})
    signature["metrics"] = {name: metrics.get(name) for name in METRIC_FIELDS}
    signature["controller"] = _without_runtime(dict(row.get("controller") or {}))
    return signature


def _episode_signature(
    collection: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float]]:
    trace_path = collection / str(manifest["trace_file"])
    resolved_trace = trace_path.resolve()
    try:
        resolved_trace.relative_to(collection.resolve())
    except ValueError as error:
        raise ValueError(f"trace escapes collection root: {trace_path}") from error
    if os.name == "nt":
        trace_path = Path("\\\\?\\" + str(resolved_trace))
    events = read_trace_events(trace_path)
    transitions = [row for row in events if row.get("event") == "transition"]
    summary = dict(manifest.get("summary") or {})
    signature = {
        "summary": {name: summary.get(name) for name in SUMMARY_FIELDS},
        "transitions": [_transition_signature(row) for row in transitions],
    }
    timing = {
        "raw_ttf_seconds": float(summary.get("wall_time_to_feasible") or 0.0),
        "selection_seconds": sum(
            float(row.get("controller", {}).get("neighborhood_selection_seconds", 0.0))
            for row in transitions
        ),
        "structpool_candidate_seconds": sum(
            float(
                row.get("controller", {})
                .get("proposal", {})
                .get("structpool_candidate_seconds", 0.0)
            )
            for row in transitions
        ),
        "pp_seconds": sum(
            float(row.get("repair_wall_seconds", 0.0)) for row in transitions
        ),
    }
    return signature, timing


def _manifests(root: Path, group: str) -> tuple[Path, dict[str, dict[str, Any]]]:
    collection = root / "groups" / group / "controllers" / "v2-plus-structpool"
    rows = _read_jsonl(collection / "realized_dynamic_manifest.jsonl")
    result = {str(row["episode_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate episode IDs in {collection}")
    return collection, result


def _timing_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "episode_count": len(rows),
        **{
            f"mean_{name}": statistics.fmean(row[name] for row in rows)
            for name in (
                "raw_ttf_seconds",
                "selection_seconds",
                "structpool_candidate_seconds",
                "pp_seconds",
            )
        },
    }


def verify(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_groups = {path.name for path in (reference / "groups").iterdir() if path.is_dir()}
    candidate_groups = {path.name for path in (candidate / "groups").iterdir() if path.is_dir()}
    mismatches: list[dict[str, Any]] = []
    semantic_rows = []
    reference_timing = []
    candidate_timing = []
    transition_count = 0
    for group in sorted(reference_groups | candidate_groups):
        if group not in reference_groups or group not in candidate_groups:
            mismatches.append({"group": group, "kind": "group_set"})
            continue
        reference_collection, expected = _manifests(reference, group)
        candidate_collection, actual = _manifests(candidate, group)
        if set(expected) != set(actual):
            mismatches.append(
                {
                    "group": group,
                    "kind": "episode_set",
                    "missing": sorted(set(expected) - set(actual)),
                    "unexpected": sorted(set(actual) - set(expected)),
                }
            )
        for episode_id in sorted(set(expected) & set(actual)):
            expected_signature, expected_timing = _episode_signature(
                reference_collection, expected[episode_id]
            )
            actual_signature, actual_timing = _episode_signature(
                candidate_collection, actual[episode_id]
            )
            transition_count += len(expected_signature["transitions"])
            reference_timing.append(expected_timing)
            candidate_timing.append(actual_timing)
            semantic_rows.append(
                {
                    "group": group,
                    "episode_id": episode_id,
                    "signature_sha256": _fingerprint(expected_signature),
                }
            )
            if expected_signature != actual_signature:
                mismatches.append(
                    {"group": group, "episode_id": episode_id, "kind": "semantic"}
                )
    return {
        "schema": SCHEMA,
        "reference": reference.resolve().as_posix(),
        "candidate": candidate.resolve().as_posix(),
        "episode_count": len(semantic_rows),
        "transition_count": transition_count,
        "semantic_rows_sha256": _fingerprint(semantic_rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "exact_match": not mismatches,
        "reference_timing": _timing_summary(reference_timing),
        "candidate_timing": _timing_summary(candidate_timing),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify runtime-only StructPool changes preserve solver semantics."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = verify(Path(arguments.reference), Path(arguments.candidate))
    _write_json(Path(arguments.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
