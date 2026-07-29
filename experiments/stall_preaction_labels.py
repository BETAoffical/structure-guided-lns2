from __future__ import annotations

import collections
import csv
from pathlib import Path
from typing import Any

from experiments._common import atomic_write_csv, sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.run_output_guard import prepare_run_output


STALL_PREACTION_LABEL_SCHEMA = "lns2.stall_preaction_labels.v1"


def zero_false_negative_control_gate(
    *,
    control_state_count: int,
    control_positive_count: int,
    minimum_negative_controls: int,
) -> dict[str, int | bool]:
    """Report a false-positive gate using only confirmed negative controls."""
    if (
        control_state_count < 0
        or control_positive_count < 0
        or control_positive_count > control_state_count
        or minimum_negative_controls < 1
    ):
        raise ValueError("invalid zero-false negative-control gate counts")
    negative_count = control_state_count - control_positive_count
    return {
        "control_negative_count": negative_count,
        "additional_zero_false_negative_controls_needed": max(
            0, minimum_negative_controls - negative_count
        ),
        "negative_control_count_gate_passed": (
            negative_count >= minimum_negative_controls
        ),
    }


def derive_preaction_state_labels(
    selected: dict[str, Any],
    probe: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fingerprint = str(selected.get("before_repair_fingerprint") or "")
    if (
        not fingerprint
        or str(probe.get("before_repair_fingerprint")) != fingerprint
        or str(audit.get("before_repair_fingerprint")) != fingerprint
        or str(probe.get("task_id")) != str(selected.get("task_id"))
        or int(probe.get("solver_seed", -1)) != int(selected.get("solver_seed", -2))
        or int(probe.get("decision_index", -1))
        != int(selected.get("decision_index", -2))
        or probe.get("all_candidates") is not True
        or int(probe.get("trials_per_branch", 0)) < 4
    ):
        raise ValueError("pre-action label source identity mismatch")
    aliases = probe.get("branch_aliases")
    branches = audit.get("branches")
    if not isinstance(aliases, dict) or not isinstance(branches, list):
        raise ValueError("pre-action Oracle evidence is incomplete")
    by_key = {
        str(branch.get("branch_key")): dict(branch)
        for branch in branches
        if isinstance(branch, dict) and str(branch.get("branch_key") or "")
    }
    candidate_rows = list(selected.get("candidate_rows") or ())
    rank_limit = int(selected.get("candidate_rank_limit", 0))
    if len(candidate_rows) != rank_limit or rank_limit < 2:
        raise ValueError("selected state has invalid candidate-rank evidence")

    joined: list[dict[str, Any]] = []
    seen_neighborhoods: set[str] = set()
    for candidate in candidate_rows:
        rank = int(candidate["rank"])
        alias = "rank1" if rank == 1 else f"rank_{rank}"
        branch_key = str(aliases.get(alias) or "")
        branch = by_key.get(branch_key)
        if branch is None:
            raise ValueError(f"Oracle is missing candidate rank {rank}")
        neighborhood = str(candidate.get("neighborhood_key") or "")
        if neighborhood != str(branch.get("neighborhood_key") or ""):
            raise ValueError("candidate neighborhood differs from Oracle branch")
        duplicate_neighborhood = neighborhood in seen_neighborhoods
        seen_neighborhoods.add(neighborhood)
        joined.append(
            {
                "state_key": selected["state_key"],
                "map_id": selected["map_id"],
                "map_fold": selected["map_fold"],
                "layout_mode": selected["layout_mode"],
                "agent_count": selected["agent_count"],
                "task_id": selected["task_id"],
                "solver_seed": selected["solver_seed"],
                "decision_index": selected["decision_index"],
                "before_repair_fingerprint": fingerprint,
                "cohort_role": selected["cohort_role"],
                "observational_class": selected["observational_class"],
                "candidate_id": candidate["candidate_id"],
                "candidate_rank": rank,
                "candidate_size": candidate["actual_size"],
                "candidate_score": candidate["score"],
                "neighborhood_key": neighborhood,
                "duplicate_actual_neighborhood": duplicate_neighborhood,
                "trial_count": int(branch["trial_count"]),
                "escape_count": int(branch["escape_count"]),
                "escape_fraction": float(branch["escape_fraction"]),
                "stable_escape": branch.get("stable_escape") is True,
                "stable_failure": branch.get("stable_failure") is True,
                "pp_order_sensitive": branch.get("pp_order_sensitive") is True,
                "mean_conflict_delta": float(branch["mean_conflict_delta"]),
                "mean_total_decision_seconds": float(
                    branch["mean_total_decision_seconds"]
                ),
                "outcome_counts": branch.get("outcome_counts"),
            }
        )
    unique_rows = [row for row in joined if not row["duplicate_actual_neighborhood"]]
    rank1 = next((row for row in unique_rows if row["candidate_rank"] == 1), None)
    if rank1 is None:
        raise ValueError("pre-action labels lost the unique v2 rank-1 neighborhood")
    alternatives = [row for row in unique_rows if 2 <= row["candidate_rank"] <= rank_limit]
    stable_alternatives = [row for row in alternatives if row["stable_escape"]]
    target = bool(rank1["stable_failure"] and stable_alternatives)
    winner = (
        min(
            stable_alternatives,
            key=lambda row: (
                -float(row["mean_conflict_delta"]),
                float(row["mean_total_decision_seconds"]),
                int(row["candidate_rank"]),
            ),
        )
        if target
        else None
    )
    for row in joined:
        row["rank2_8_rescue_training_eligible"] = target
        row["recommended_rescue"] = bool(
            winner is not None
            and not row["duplicate_actual_neighborhood"]
            and row["neighborhood_key"] == winner["neighborhood_key"]
        )
        row["rescue_priority_primary"] = "stable_escape"
        row["rescue_priority_secondary"] = "mean_conflict_delta"
        row["rescue_priority_tiebreak"] = "mean_total_decision_seconds"
    state_row = {
        "state_key": selected["state_key"],
        "map_id": selected["map_id"],
        "map_fold": selected["map_fold"],
        "layout_mode": selected["layout_mode"],
        "agent_count": selected["agent_count"],
        "task_id": selected["task_id"],
        "solver_seed": selected["solver_seed"],
        "decision_index": selected["decision_index"],
        "before_repair_fingerprint": fingerprint,
        "cohort_role": selected["cohort_role"],
        "observational_class": selected["observational_class"],
        "trajectory_lookahead_is_label_only": True,
        "v2_rank1_stable_failure": bool(rank1["stable_failure"]),
        "rank2_8_stable_alternative_count": len(stable_alternatives),
        "target_rescuable_selector_failure": target,
        "full_pool_oracle_classification": str(audit.get("classification") or ""),
        "recommended_rescue_rank": winner["candidate_rank"] if winner else None,
        "recommended_rescue_candidate_id": winner["candidate_id"] if winner else None,
        "candidate_label_count": len(joined),
        "unique_candidate_neighborhood_count": len(unique_rows),
    }
    return state_row, joined


def _validate_paired_seeds(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_trial: dict[int, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_trial[int(row["trial_index"])].append(row)
    mismatch_count = 0
    for trial_rows in by_trial.values():
        requested = {int(row["requested_pp_random_seed"]) for row in trial_rows}
        random_seeds = {int(row["random_seed"]) for row in trial_rows}
        if len(requested) != 1 or requested != random_seeds:
            mismatch_count += 1
    return {
        "trial_count": len(by_trial),
        "paired_seed_mismatch_count": mismatch_count,
    }


def build_stall_preaction_labels(
    cohort: str | Path,
    oracle: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    cohort_root = Path(cohort).resolve()
    oracle_root = Path(oracle).resolve()
    output_root = Path(output).resolve()
    selected_path = cohort_root / "selected_states.jsonl"
    cohort_report_path = cohort_root / "stall_preaction_cohort_report.json"
    oracle_report_path = oracle_root / "oracle_batch_report.json"
    cohort_report = _read_json(cohort_report_path)
    oracle_report = _read_json(oracle_report_path)
    if (
        cohort_report.get("complete") is not True
        or oracle_report.get("complete") is not True
        or int(oracle_report.get("error_count", -1)) != 0
    ):
        raise ValueError("pre-action label source is incomplete")
    identity = {
        "schema": STALL_PREACTION_LABEL_SCHEMA,
        "selected_states_sha256": sha256_file(selected_path),
        "cohort_report_sha256": sha256_file(cohort_report_path),
        "oracle_report_sha256": sha256_file(oracle_report_path),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    prepare_run_output(output_root, resume=resume, identity=identity)
    selected = {
        str(row["before_repair_fingerprint"]): row
        for row in _read_jsonl(selected_path)
    }
    state_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    paired_seed_mismatches = 0
    jobs = sorted((oracle_root / "jobs").iterdir())
    for job_root in jobs:
        probe = _read_json(job_root / "stalled_state_probe_report.json")
        audit = _read_json(job_root / "audit" / "stall_oracle_report.json")
        fingerprint = str(probe.get("before_repair_fingerprint") or "")
        selected_row = selected.get(fingerprint)
        if selected_row is None:
            raise ValueError("Oracle state is absent from the registered cohort")
        paired = _validate_paired_seeds(job_root / "stalled_state_trials.csv")
        paired_seed_mismatches += int(paired["paired_seed_mismatch_count"])
        state_row, labels = derive_preaction_state_labels(
            selected_row, probe, audit
        )
        state_rows.append(state_row)
        candidate_rows.extend(labels)
    covered = {str(row["before_repair_fingerprint"]) for row in state_rows}
    positives = sum(bool(row["target_rescuable_selector_failure"]) for row in state_rows)
    map_count = len({str(row["map_id"]) for row in state_rows})
    complete_coverage = len(covered) == len(selected)
    training_rows = [
        row for row in state_rows if str(row["cohort_role"]) == "enriched_training"
    ]
    control_rows = [
        row
        for row in state_rows
        if str(row["cohort_role"]) == "outcome_blind_control"
    ]
    if len(training_rows) + len(control_rows) != len(state_rows):
        raise ValueError("pre-action labels contain an unknown cohort role")
    training_positive_count = sum(
        bool(row["target_rescuable_selector_failure"]) for row in training_rows
    )
    training_negative_count = len(training_rows) - training_positive_count
    control_positive_count = sum(
        bool(row["target_rescuable_selector_failure"]) for row in control_rows
    )
    zero_false_requirement = int(
        cohort_report["minimum_zero_false_controls_for_one_percent_wilson_gate"]
    )
    control_gate = zero_false_negative_control_gate(
        control_state_count=len(control_rows),
        control_positive_count=control_positive_count,
        minimum_negative_controls=zero_false_requirement,
    )
    control_negative_count = int(control_gate["control_negative_count"])
    training_map_count = len({str(row["map_id"]) for row in training_rows})
    fold_counts: dict[str, dict[str, int]] = {}
    grouped_folds: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    grouped_cells: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in training_rows:
        grouped_folds[str(row["map_fold"])].append(row)
        grouped_cells[f"{row['layout_mode']}:{row['agent_count']}"].append(row)
    for fold, rows in sorted(grouped_folds.items()):
        positive = sum(bool(row["target_rescuable_selector_failure"]) for row in rows)
        fold_counts[fold] = {
            "state_count": len(rows),
            "positive_count": positive,
            "negative_count": len(rows) - positive,
            "map_count": len({str(row["map_id"]) for row in rows}),
        }
    cell_counts: dict[str, dict[str, int]] = {}
    for cell, rows in sorted(grouped_cells.items()):
        positive = sum(bool(row["target_rescuable_selector_failure"]) for row in rows)
        cell_counts[cell] = {
            "state_count": len(rows),
            "positive_count": positive,
            "negative_count": len(rows) - positive,
        }
    fold_gate = bool(
        len(fold_counts) == 4
        and all(
            row["positive_count"] >= 3 and row["negative_count"] >= 3
            for row in fold_counts.values()
        )
    )
    cell_gate = bool(
        len(cell_counts) == 6
        and all(
            row["positive_count"] >= 2 and row["negative_count"] >= 2
            for row in cell_counts.values()
        )
    )
    training_eligible = bool(
        complete_coverage
        and paired_seed_mismatches == 0
        and len(training_rows) >= 60
        and training_positive_count >= 20
        and training_negative_count >= 20
        and training_map_count >= 8
        and fold_gate
        and cell_gate
    )
    report = {
        "schema": STALL_PREACTION_LABEL_SCHEMA,
        "complete": True,
        "evidence_level": "paired same-state PP labels",
        "registered_state_count": len(selected),
        "labelled_state_count": len(state_rows),
        "coverage_fraction": len(state_rows) / len(selected),
        "complete_coverage": complete_coverage,
        "candidate_label_count": len(candidate_rows),
        "map_count": map_count,
        "rescuable_selector_failure_count": positives,
        "rank1_stable_failure_count": sum(
            bool(row["v2_rank1_stable_failure"]) for row in state_rows
        ),
        "paired_seed_mismatch_count": paired_seed_mismatches,
        "training_state_count": len(training_rows),
        "training_positive_count": training_positive_count,
        "training_negative_count": training_negative_count,
        "training_map_count": training_map_count,
        "training_fold_counts": fold_counts,
        "training_layout_agent_counts": cell_counts,
        "training_fold_gate_passed": fold_gate,
        "training_layout_agent_gate_passed": cell_gate,
        "control_state_count": len(control_rows),
        "control_positive_count": control_positive_count,
        "control_negative_count": control_negative_count,
        "control_reserved_from_training": True,
        "minimum_zero_false_negative_controls_for_one_percent_wilson_gate": (
            zero_false_requirement
        ),
        "additional_zero_false_negative_controls_needed": max(
            0,
            int(control_gate["additional_zero_false_negative_controls_needed"]),
        ),
        "negative_control_count_gate_passed": bool(
            control_gate["negative_control_count_gate_passed"]
        ),
        "shadow_promotion_eligible": False,
        "pp_order_contract": (
            "same PP seed and deterministic order-generation rule per candidate; "
            "exact repair-order lists cannot match across different agent sets"
        ),
        "training_eligible": training_eligible,
        "training_started": False,
        "controller_actions_changed": False,
        "deployment_promoted": False,
        "decision": (
            "paired_label_cohort_ready_for_grouped_training"
            if training_eligible
            else "smoke_or_partial_labels_only_do_not_train"
        ),
    }
    atomic_write_csv(output_root / "stall_trigger_labels.csv", state_rows)
    atomic_write_csv(output_root / "stall_rescue_candidate_labels.csv", candidate_rows)
    _write_json(output_root / "stall_preaction_label_report.json", report)
    return report


__all__ = [
    "STALL_PREACTION_LABEL_SCHEMA",
    "build_stall_preaction_labels",
    "derive_preaction_state_labels",
]
