from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.stride_lns import assign_structure_scores
from experiments.stride_topology_anchor_quality import TRIAL_SCHEMA


CONFIG_SCHEMA = "lns2.stride.topology_anchor_failure_config.v1"
REPORT_SCHEMA = "lns2.stride.topology_anchor_failure_report.v1"


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _registered(project_root: Path, artifact: dict[str, Any]) -> Path:
    path = (project_root / str(artifact["path"])).resolve()
    if sha256_file(path) != str(artifact["sha256"]):
        raise ValueError(f"topology-anchor failure input SHA differs: {artifact['path']}")
    return path


def validate_topology_anchor_failure_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected topology-anchor failure config")
    if (
        config.get("scientific_status") != "posthoc_failure_mechanism_diagnostic"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("training_allowed"))
        or bool(config.get("candidate_repair_trials_allowed"))
    ):
        raise ValueError("topology-anchor failure analysis must remain posthoc and read-only")
    if (
        config.get("diagnostic_id") != "stride-topoanchor-failure-v1"
        or config.get("candidate_generator_id") != "stride-topoanchor-v1"
        or int(config.get("expected_state_count", -1)) != 16
        or int(config.get("expected_candidate_count", -1)) != 360
        or int(config.get("expected_trial_count", -1)) != 1440
        or tuple(map(int, config.get("trial_indices") or ())) != (0, 1, 2, 3)
        or dict(config.get("label") or {}) != {
            "structure_weight": 0.02,
            "no_progress_penalty": 0.10,
        }
    ):
        raise ValueError("topology-anchor failure diagnostic identity changed")
    if set(config.get("inputs") or {}) != {
        "quality_config", "collection_report", "repair_trials",
        "coverage_candidate_rows", "quality_report",
    }:
        raise ValueError("topology-anchor failure input registry changed")


def _candidate_summaries(
    trials: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    by_index: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        by_index[int(row["trial_index"])].append(row)
    if set(by_index) != set(map(int, config["trial_indices"])):
        raise ValueError("topology-anchor failure trial indices differ")
    candidate_sets = [{str(row["candidate_id"]) for row in rows} for rows in by_index.values()]
    if any(values != candidate_sets[0] for values in candidate_sets[1:]):
        raise ValueError("topology-anchor failure candidate pool changed across seeds")
    values: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: defaultdict(list)
    )
    structure_weight = float(config["label"]["structure_weight"])
    for _index, rows in sorted(by_index.items()):
        if len({int(row["pp_seed"]) for row in rows}) != 1:
            raise ValueError("topology-anchor failure PP seeds are not paired")
        scored = [
            {"candidate_id": row["candidate_id"], "mean_post_structure": row["post_structure"]}
            for row in rows
        ]
        assign_structure_scores(scored)
        structural = {str(row["candidate_id"]): float(row["structural_score"]) for row in scored}
        for row in rows:
            candidate_id = str(row["candidate_id"])
            before = int(row["before_conflicts"])
            after = int(row["conflicts_after"])
            reduction = (before - after) / max(1, before)
            penalty = structure_weight * structural[candidate_id]
            values[candidate_id]["normalized_conflict_reduction"].append(reduction)
            values[candidate_id]["weighted_structure_penalty"].append(penalty)
            values[candidate_id]["pre_no_progress_score"].append(reduction - penalty)
            values[candidate_id]["progress"].append(float(after < before))
            values[candidate_id]["replan_success"].append(float(bool(row["replan_success"])))
            values[candidate_id]["pp_replan_seconds"].append(float(row["pp_replan_seconds"]))
            values[candidate_id]["candidate_kind"] = str(row["candidate_kind"])
            values[candidate_id]["actual_size"] = int(row["actual_size"])
            values[candidate_id]["selection_families"] = list(row["selection_families"])
    summaries = {}
    no_progress_penalty = float(config["label"]["no_progress_penalty"])
    for candidate_id, row in values.items():
        summary = {
            key: _mean(list(map(float, row[key])))
            for key in (
                "normalized_conflict_reduction", "weighted_structure_penalty",
                "pre_no_progress_score", "progress", "replan_success", "pp_replan_seconds",
            )
        }
        summary.update({
            "candidate_id": candidate_id,
            "candidate_kind": row["candidate_kind"],
            "actual_size": row["actual_size"],
            "selection_families": row["selection_families"],
            "label_score": summary["pre_no_progress_score"] - no_progress_penalty * (1.0 - summary["progress"]),
        })
        summaries[candidate_id] = summary
    return summaries


def analyze_topology_anchor_failure(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_topology_anchor_failure_config(config)
    inputs = {name: _registered(project_root, artifact) for name, artifact in config["inputs"].items()}
    collection = _read_json(inputs["collection_report"])
    quality = _read_json(inputs["quality_report"])
    trials = _read_jsonl(inputs["repair_trials"])
    if collection.get("complete") is not True or int(collection.get("error_state_count", -1)) != 0:
        raise ValueError("topology-anchor failure requires a complete zero-error collection")
    if quality.get("passed") is not False:
        raise ValueError("topology-anchor failure requires the registered failed quality report")
    if len(trials) != int(config["expected_trial_count"]):
        raise ValueError("topology-anchor failure trial count differs")
    by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        if row.get("schema") != TRIAL_SCHEMA:
            raise ValueError("unexpected topology-anchor failure trial schema")
        by_state[str(row["state_id"])].append(row)
    if len(by_state) != int(config["expected_state_count"]):
        raise ValueError("topology-anchor failure state count differs")
    state_reports = []
    anchor_candidate_count = 0
    size_rows: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for state_id in sorted(by_state):
        summaries = _candidate_summaries(by_state[state_id], config)
        base = [row for row in summaries.values() if row["candidate_kind"] in {"base", "base_and_anchor"}]
        anchor = [row for row in summaries.values() if row["candidate_kind"] == "anchor_only"]
        if not base or not anchor:
            raise ValueError(f"topology-anchor failure state lacks pool partition: {state_id}")
        anchor_candidate_count += len(anchor)
        for row in anchor:
            size_rows[int(row["actual_size"])].append(row)
        best_base = max(base, key=lambda row: (float(row["label_score"]), str(row["candidate_id"])))
        best_anchor = max(anchor, key=lambda row: (float(row["label_score"]), str(row["candidate_id"])))
        deltas = {
            name: float(best_anchor[name]) - float(best_base[name])
            for name in (
                "label_score", "normalized_conflict_reduction", "weighted_structure_penalty",
                "progress", "replan_success", "pp_replan_seconds",
            )
        }
        state_reports.append({
            "state_id": state_id,
            "layout_family": str(by_state[state_id][0]["layout_family"]),
            "best_base_candidate_id": best_base["candidate_id"],
            "best_anchor_candidate_id": best_anchor["candidate_id"],
            "best_anchor_size": best_anchor["actual_size"],
            "best_anchor_families": best_anchor["selection_families"],
            "anchor_strict_win": deltas["label_score"] > 1e-12,
            "anchor_minus_base": deltas,
        })
    if sum(len(rows) for rows in size_rows.values()) != anchor_candidate_count:
        raise ValueError("topology-anchor failure candidate accounting differs")
    group_reports = {}
    for group in sorted({row["layout_family"] for row in state_reports}):
        rows = [row for row in state_reports if row["layout_family"] == group]
        group_reports[group] = {
            "state_count": len(rows),
            "strict_win_count": sum(bool(row["anchor_strict_win"]) for row in rows),
            "mean_anchor_minus_base": {
                name: _mean([float(row["anchor_minus_base"][name]) for row in rows])
                for name in rows[0]["anchor_minus_base"]
            },
        }
    size_reports = {
        str(size): {
            "candidate_count": len(rows),
            "mean_label_score": _mean([float(row["label_score"]) for row in rows]),
            "mean_normalized_conflict_reduction": _mean([float(row["normalized_conflict_reduction"]) for row in rows]),
            "progress_rate": _mean([float(row["progress"]) for row in rows]),
            "replan_success_rate": _mean([float(row["replan_success"]) for row in rows]),
        }
        for size, rows in sorted(size_rows.items())
    }
    best_size_counts = {
        str(size): sum(int(row["best_anchor_size"]) == size for row in state_reports)
        for size in sorted(size_rows)
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "posthoc_failure_mechanism_diagnostic",
        "formal_speed_claim": False,
        "new_repair_trials_executed": False,
        "training_allowed": False,
        "state_count": len(state_reports),
        "candidate_count": sum(
            len({str(trial["candidate_id"]) for trial in by_state[row["state_id"]]})
            for row in state_reports
        ),
        "trial_count": len(trials),
        "best_anchor_size_counts": best_size_counts,
        "anchor_size_summary": size_reports,
        "topology_groups": group_reports,
        "states": state_reports,
        "interpretation": {
            "primary_failure_mechanism": "lower_realized_current_conflict_reduction_not_structure_penalty_or_pp_runtime",
            "candidate_revision": "retain_size_16_but_replace_pair_closure_with_boundary_oriented_endpoint_selection",
            "evidence_boundary": "posthoc_on_consumed_pilot_states_requires_fresh_quality_confirmation",
        },
        "next_decision": config["next_decision"],
        "inputs": {
            "config_sha256": sha256_file(config_path),
            **{f"{name}_sha256": sha256_file(path) for name, path in sorted(inputs.items())},
        },
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "topology_anchor_failure_report.json", report)
    return report


__all__ = ["analyze_topology_anchor_failure", "validate_topology_anchor_failure_config"]
