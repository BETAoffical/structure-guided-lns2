from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import read_jsonl as _read_jsonl  # noqa: E402
POLICIES = ("official_adaptive", "proposal_dynamic", "realized_dynamic")
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
    "conflicts_before",
    "conflicts_after",
    "conflict_delta",
    "iteration",
    "neighborhood",
    "replan_success",
    "requested_mode",
    "requested_random_seed",
    "sum_of_costs_before",
    "sum_of_costs_after",
)


def _manifest(root: Path, policy: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(root / f"{policy}_manifest.jsonl")
    result = {str(row["episode_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate episode ids in {root} for {policy}")
    return result


def _trace(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = (root / str(manifest["trace_file"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"trace escapes collection root: {path}") from error
    return _read_jsonl(path)


def _transition_signature(row: dict[str, Any]) -> dict[str, Any]:
    signature = {name: row.get(name) for name in TRANSITION_FIELDS}
    metrics = dict(row.get("metrics", {}))
    signature["metrics"] = {name: metrics.get(name) for name in METRIC_FIELDS}
    controller = dict(row.get("controller", {}))
    signature["selected_candidate_id"] = controller.get("selected_candidate_id")
    return signature


def _episode_signature(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    events = _trace(root, manifest)
    transitions = [row for row in events if row.get("event") == "transition"]
    summary = dict(manifest.get("summary", {}))
    return {
        "summary": {name: summary.get(name) for name in SUMMARY_FIELDS},
        "transitions": [_transition_signature(row) for row in transitions],
    }


def _timing(root: Path, manifests: dict[str, dict[str, Any]]) -> dict[str, float]:
    wall = []
    controller = []
    features = []
    inference = []
    proposal = []
    for manifest in manifests.values():
        summary = dict(manifest.get("summary", {}))
        wall.append(float(summary.get("wall_time_to_feasible", 0.0)))
        events = _trace(root, manifest)
        transitions = [row for row in events if row.get("event") == "transition"]
        controller.append(
            sum(
                float(row.get("controller", {}).get("controller_seconds_before_repair", 0.0))
                for row in transitions
            )
        )
        features.append(
            sum(float(row.get("controller", {}).get("feature_seconds", 0.0)) for row in transitions)
        )
        inference.append(
            sum(float(row.get("controller", {}).get("inference_seconds", 0.0)) for row in transitions)
        )
        proposal.append(
            sum(
                float(row.get("controller", {}).get("proposal", {}).get("proposal_seconds", 0.0))
                for row in transitions
            )
        )
    return {
        "episode_count": float(len(manifests)),
        "mean_wall_seconds": statistics.fmean(wall) if wall else 0.0,
        "mean_controller_seconds": statistics.fmean(controller) if controller else 0.0,
        "mean_feature_seconds": statistics.fmean(features) if features else 0.0,
        "mean_inference_seconds": statistics.fmean(inference) if inference else 0.0,
        "mean_proposal_seconds": statistics.fmean(proposal) if proposal else 0.0,
    }


def compare_collections(reference: str | Path, candidate: str | Path) -> dict[str, Any]:
    reference_root = Path(reference).resolve()
    candidate_root = Path(candidate).resolve()
    mismatches: list[dict[str, Any]] = []
    policies: dict[str, Any] = {}
    for policy in POLICIES:
        reference_manifest = _manifest(reference_root, policy)
        candidate_manifest = _manifest(candidate_root, policy)
        reference_ids = set(reference_manifest)
        candidate_ids = set(candidate_manifest)
        if reference_ids != candidate_ids:
            mismatches.append(
                {
                    "policy": policy,
                    "kind": "episode_set",
                    "missing": sorted(reference_ids - candidate_ids),
                    "unexpected": sorted(candidate_ids - reference_ids),
                }
            )
        matched = 0
        transition_count = 0
        for episode_id in sorted(reference_ids & candidate_ids):
            expected = _episode_signature(reference_root, reference_manifest[episode_id])
            actual = _episode_signature(candidate_root, candidate_manifest[episode_id])
            transition_count += len(expected["transitions"])
            if actual != expected:
                mismatches.append(
                    {"policy": policy, "kind": "episode", "episode_id": episode_id}
                )
            else:
                matched += 1
        policies[policy] = {
            "episode_count": len(reference_ids),
            "matching_episode_count": matched,
            "transition_count": transition_count,
            "reference_timing": _timing(reference_root, reference_manifest),
            "candidate_timing": _timing(candidate_root, candidate_manifest),
        }
    return {
        "schema": "lns2.closed_loop_equivalence.v1",
        "reference": str(reference_root),
        "candidate": str(candidate_root),
        "exact": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "policies": policies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare scientific closed-loop traces while excluding timing fields."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    report = compare_collections(arguments.reference, arguments.candidate)
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
