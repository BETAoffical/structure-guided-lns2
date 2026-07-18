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
from experiments.closed_loop_confirmation import validate_closed_loop_trace  # noqa: E402
from experiments.closed_loop_trace_storage import read_trace_events  # noqa: E402
from experiments.repair_collection import (  # noqa: E402
    _fingerprint,
    _read_json,
    _utc_now,
    _write_json,
)

LEGACY_POLICIES = ("official_adaptive", "proposal_dynamic", "realized_dynamic")

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
    "controller_totals",
    "mean_selected_feature_outside_fraction",
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
    "requested_random_seed",
    "requested_repair_order",
    "sum_of_costs_before",
    "sum_of_costs_after",
)
CONTROLLER_TIMING_FIELDS = (
    "feature_seconds",
    "inference_seconds",
    "controller_seconds_before_repair",
)
PROPOSAL_TIMING_FIELDS = ("proposal_seconds", "state_check_seconds")


def equivalence_comparison_fingerprint() -> str:
    return _fingerprint(
        {
            "schema": "lns2.closed_loop_equivalence_comparison.v1",
            "summary_fields": SUMMARY_FIELDS,
            "transition_fields": TRANSITION_FIELDS,
            "metric_fields": METRIC_FIELDS,
            "controller_timing_fields_excluded": CONTROLLER_TIMING_FIELDS,
            "proposal_timing_fields_excluded": PROPOSAL_TIMING_FIELDS,
        }
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
    return read_trace_events(path)


def _scientific_controller(value: Any) -> dict[str, Any]:
    controller = dict(value or {})
    for name in CONTROLLER_TIMING_FIELDS:
        controller.pop(name, None)
    proposal = dict(controller.get("proposal", {}))
    for name in PROPOSAL_TIMING_FIELDS:
        proposal.pop(name, None)
    if proposal:
        controller["proposal"] = proposal
    elif "proposal" in controller:
        controller["proposal"] = {}
    return controller


def _transition_signature(row: dict[str, Any]) -> dict[str, Any]:
    signature = {name: row.get(name) for name in TRANSITION_FIELDS}
    metrics = dict(row.get("metrics", {}))
    signature["metrics"] = {name: metrics.get(name) for name in METRIC_FIELDS}
    signature["controller"] = _scientific_controller(row.get("controller", {}))
    return signature


def _episode_evidence(
    root: Path,
    manifest: dict[str, Any],
    run_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    if run_config is not None:
        validated = validate_closed_loop_trace(
            root / str(manifest["trace_file"]),
            str(run_config["run_fingerprint"]),
            expected_episode_id=str(manifest["episode_id"]),
            expected_policy=str(manifest["policy"]),
            expected_solver_seed=int(manifest["solver_seed"]),
            metric_iteration_budget=int(
                run_config["configuration"]["metric_iteration_budget"]
            ),
            collection_root=root,
        )
        events = validated["events"]
    else:
        events = _trace(root, manifest)
    transitions = [row for row in events if row.get("event") == "transition"]
    summary = dict(manifest.get("summary", {}))
    signature = {
        "summary": {name: summary.get(name) for name in SUMMARY_FIELDS},
        "transitions": [_transition_signature(row) for row in transitions],
    }
    timing = {
        "wall": float(summary.get("wall_time_to_feasible") or 0.0),
        "controller": sum(
            float(row.get("controller", {}).get("controller_seconds_before_repair", 0.0))
            for row in transitions
        ),
        "features": sum(
            float(row.get("controller", {}).get("feature_seconds", 0.0))
            for row in transitions
        ),
        "inference": sum(
            float(row.get("controller", {}).get("inference_seconds", 0.0))
            for row in transitions
        ),
        "proposal": sum(
            float(row.get("controller", {}).get("proposal", {}).get("proposal_seconds", 0.0))
            for row in transitions
        ),
    }
    return signature, timing


def _episode_signature(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    run_path = root / "run_config.json"
    run_config = _read_json(run_path) if run_path.is_file() else None
    return _episode_evidence(root, manifest, run_config)[0]


def _timing_summary(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        "episode_count": float(len(values)),
        "mean_wall_seconds": statistics.fmean(row["wall"] for row in values)
        if values
        else 0.0,
        "mean_controller_seconds": statistics.fmean(row["controller"] for row in values)
        if values
        else 0.0,
        "mean_feature_seconds": statistics.fmean(row["features"] for row in values)
        if values
        else 0.0,
        "mean_inference_seconds": statistics.fmean(row["inference"] for row in values)
        if values
        else 0.0,
        "mean_proposal_seconds": statistics.fmean(row["proposal"] for row in values)
        if values
        else 0.0,
    }


def compare_collections(
    reference: str | Path,
    candidate: str | Path,
    *,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    reference_root = Path(reference).resolve()
    candidate_root = Path(candidate).resolve()
    reference_run_path = reference_root / "run_config.json"
    candidate_run_path = candidate_root / "run_config.json"
    if reference_run_path.is_file() and candidate_run_path.is_file():
        reference_run = _read_json(reference_run_path)
        candidate_run = _read_json(candidate_run_path)
        reference_policies = tuple(
            map(str, reference_run["configuration"].get("policies", []))
        )
        candidate_policies = tuple(
            map(str, candidate_run["configuration"].get("policies", []))
        )
        if reference_policies != candidate_policies:
            raise ValueError("collections register different policy sets")
        if str(reference_run["run_fingerprint"]) != str(candidate_run["run_fingerprint"]):
            raise ValueError("collections have different scientific run fingerprints")
    elif not reference_run_path.exists() and not candidate_run_path.exists():
        reference_policies = LEGACY_POLICIES
        reference_run = None
        candidate_run = None
    else:
        raise ValueError("only one collection contains run_config.json")
    mismatches: list[dict[str, Any]] = []
    policies: dict[str, Any] = {}
    reference_manifests = {
        policy: _manifest(reference_root, policy) for policy in reference_policies
    }
    candidate_manifests = {
        policy: _manifest(candidate_root, policy) for policy in reference_policies
    }
    total_episodes = sum(
        len(set(reference_manifests[policy]) & set(candidate_manifests[policy]))
        for policy in reference_policies
    )
    completed_episodes = 0
    progress = Path(progress_path).resolve() if progress_path is not None else None
    if progress is not None:
        _write_json(
            progress,
            {
                "schema": "lns2.closed_loop_equivalence_progress.v1",
                "status": "running",
                "total_episodes": total_episodes,
                "completed_episodes": 0,
                "updated_at": _utc_now(),
            },
        )
    for policy in reference_policies:
        reference_manifest = reference_manifests[policy]
        candidate_manifest = candidate_manifests[policy]
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
        reference_timing = []
        candidate_timing = []
        for episode_id in sorted(reference_ids & candidate_ids):
            expected, expected_timing = _episode_evidence(
                reference_root, reference_manifest[episode_id], reference_run
            )
            actual, actual_timing = _episode_evidence(
                candidate_root, candidate_manifest[episode_id], candidate_run
            )
            reference_timing.append(expected_timing)
            candidate_timing.append(actual_timing)
            transition_count += len(expected["transitions"])
            if actual != expected:
                mismatches.append(
                    {"policy": policy, "kind": "episode", "episode_id": episode_id}
                )
            else:
                matched += 1
            completed_episodes += 1
            if progress is not None:
                _write_json(
                    progress,
                    {
                        "schema": "lns2.closed_loop_equivalence_progress.v1",
                        "status": "running",
                        "total_episodes": total_episodes,
                        "completed_episodes": completed_episodes,
                        "current_policy": policy,
                        "current_episode_id": episode_id,
                        "mismatch_count": len(mismatches),
                        "updated_at": _utc_now(),
                    },
                )
        policies[policy] = {
            "episode_count": len(reference_ids),
            "matching_episode_count": matched,
            "transition_count": transition_count,
            "reference_timing": _timing_summary(reference_timing),
            "candidate_timing": _timing_summary(candidate_timing),
        }
    reference_trace_bytes = sum(
        (reference_root / str(row["trace_file"])).stat().st_size
        for policy in reference_policies
        for row in reference_manifests[policy].values()
        if str(row.get("status")) in {"ok", "resumed"}
    )
    candidate_trace_bytes = sum(
        (candidate_root / str(row["trace_file"])).stat().st_size
        for policy in reference_policies
        for row in candidate_manifests[policy].values()
        if str(row.get("status")) in {"ok", "resumed"}
    )
    state_blob_bytes = sum(
        path.stat().st_size
        for path in (candidate_root / "state_blobs").glob("*.json.gz")
        if path.is_file()
    )
    candidate_total_bytes = candidate_trace_bytes + state_blob_bytes
    reduction = (
        1.0 - candidate_total_bytes / reference_trace_bytes
        if reference_trace_bytes
        else 0.0
    )
    report = {
        "schema": "lns2.closed_loop_equivalence.v1",
        "comparison_fingerprint": equivalence_comparison_fingerprint(),
        "reference": str(reference_root),
        "candidate": str(candidate_root),
        "exact": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "policies": policies,
        "storage": {
            "reference_trace_bytes": reference_trace_bytes,
            "candidate_trace_bytes": candidate_trace_bytes,
            "candidate_state_blob_bytes": state_blob_bytes,
            "candidate_total_bytes": candidate_total_bytes,
            "reduction_fraction": reduction,
        },
    }
    if progress is not None:
        _write_json(
            progress,
            {
                "schema": "lns2.closed_loop_equivalence_progress.v1",
                "status": "complete",
                "total_episodes": total_episodes,
                "completed_episodes": completed_episodes,
                "mismatch_count": len(mismatches),
                "updated_at": _utc_now(),
            },
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare scientific closed-loop traces while excluding timing fields."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    parser.add_argument("--minimum-storage-reduction", type=float, default=0.9)
    parser.add_argument("--progress")
    arguments = parser.parse_args()
    report = compare_collections(
        arguments.reference,
        arguments.candidate,
        progress_path=arguments.progress,
    )
    report["minimum_storage_reduction"] = arguments.minimum_storage_reduction
    report["storage_target_passed"] = (
        float(report["storage"]["reduction_fraction"])
        >= arguments.minimum_storage_reduction
    )
    report["passed"] = bool(report["exact"] and report["storage_target_passed"])
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
