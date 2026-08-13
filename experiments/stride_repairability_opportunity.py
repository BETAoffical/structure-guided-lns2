from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_tailswitch import _mean


CONFIG_SCHEMA = "lns2.stride.repairability_opportunity_registration.v1"
REPORT_SCHEMA = "lns2.stride.repairability_opportunity_report.v1"
CHECKPOINT_SCHEMA = "lns2.stride.repairability_opportunity_checkpoint.v1"
STATUS_SCHEMA = "lns2.stride.repairability_opportunity_status.v1"
EXPERIMENT_ID = "stride-repairability-opportunity-v1"
EXPECTED_PARENT = "c783b7fffd89b6c207507df1f36a49bcea0abc48"


def _candidate_key(state_fingerprint: str, candidate_id: str) -> tuple[str, str]:
    return str(state_fingerprint), str(candidate_id)


def paired_repairability_diagnostic(
    candidate: dict[str, Any],
    anchor: dict[str, Any],
    candidate_trials: list[dict[str, Any]],
    anchor_trials: list[dict[str, Any]],
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    candidate_by_index = {int(row["trial_index"]): row for row in candidate_trials}
    anchor_by_index = {int(row["trial_index"]): row for row in anchor_trials}
    expected_indices = set(range(16))
    if set(candidate_by_index) != expected_indices or set(anchor_by_index) != expected_indices:
        raise ValueError("paired trial coverage changed")
    for index in expected_indices:
        if int(candidate_by_index[index]["pp_seed"]) != int(
            anchor_by_index[index]["pp_seed"]
        ):
            raise ValueError(f"paired PP seed changed at trial {index}")

    quality_checks = {
        "seed_mean": float(candidate["seed_mean"])
        + epsilon
        >= float(anchor["seed_mean"]),
        "lower_half_mean": float(candidate["lower_half_mean"])
        + epsilon
        >= float(anchor["lower_half_mean"]),
        "first_fixed_half_mean": float(candidate["first_fixed_half_mean"])
        + epsilon
        >= float(anchor["first_fixed_half_mean"]),
        "second_fixed_half_mean": float(candidate["second_fixed_half_mean"])
        + epsilon
        >= float(anchor["second_fixed_half_mean"]),
        "no_progress_rate": float(candidate["no_progress_rate"])
        <= float(anchor["no_progress_rate"]) + epsilon,
    }

    wins = []
    losses = []
    for index in sorted(expected_indices):
        candidate_success = bool(candidate_by_index[index]["replan_success"])
        anchor_success = bool(anchor_by_index[index]["replan_success"])
        wins.append(candidate_success and not anchor_success)
        losses.append(anchor_success and not candidate_success)
    half_rows = []
    for half, indices in (("trial_0_7", range(8)), ("trial_8_15", range(8, 16))):
        half_wins = sum(wins[index] for index in indices)
        half_losses = sum(losses[index] for index in indices)
        half_rows.append(
            {
                "half": half,
                "paired_win_count": half_wins,
                "paired_loss_count": half_losses,
                "direction_nonnegative": half_wins >= half_losses,
            }
        )
    total_wins = sum(wins)
    total_losses = sum(losses)
    candidate_rate = float(candidate["replan_success_rate"])
    anchor_rate = float(anchor["replan_success_rate"])
    repairability_checks = {
        "strict_success_rate_gain": candidate_rate > anchor_rate + epsilon,
        "paired_wins_above_losses": total_wins > total_losses,
        "both_fixed_halves_nonnegative": all(
            row["direction_nonnegative"] for row in half_rows
        ),
    }
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_kind": str(candidate["candidate_kind"]),
        "actual_size": int(candidate["actual_size"]),
        "quality_noninferior": all(quality_checks.values()),
        "quality_checks": quality_checks,
        "repairability_dominates": all(repairability_checks.values()),
        "repairability_checks": repairability_checks,
        "qualifies": all(quality_checks.values())
        and all(repairability_checks.values()),
        "seed_mean_delta": float(candidate["seed_mean"])
        - float(anchor["seed_mean"]),
        "lower_half_mean_delta": float(candidate["lower_half_mean"])
        - float(anchor["lower_half_mean"]),
        "no_progress_rate_delta": float(candidate["no_progress_rate"])
        - float(anchor["no_progress_rate"]),
        "replan_success_rate": candidate_rate,
        "anchor_replan_success_rate": anchor_rate,
        "replan_success_rate_delta": candidate_rate - anchor_rate,
        "paired_win_count": total_wins,
        "paired_loss_count": total_losses,
        "paired_tie_count": 16 - total_wins - total_losses,
        "fixed_halves": half_rows,
        "selection_families": list(candidate.get("selection_families") or ()),
        "structpool_family_groups": list(
            candidate.get("structpool_family_groups") or ()
        ),
    }


def _best_opportunity(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    qualifying = [row for row in rows if row["qualifies"]]
    if not qualifying:
        return None
    return min(
        qualifying,
        key=lambda row: (
            -float(row["replan_success_rate_delta"]),
            -float(row["seed_mean_delta"]),
            int(row["actual_size"]),
            str(row["candidate_id"]),
        ),
    )


def _checkpoint_opportunity(
    logical: dict[str, Any],
    checkpoint: dict[str, Any],
    aggregates: dict[tuple[str, str], dict[str, Any]],
    trials: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    epsilon: float,
) -> dict[str, Any]:
    state_fingerprint = str(logical["state_fingerprint"])
    candidate_ids = list(map(str, logical["candidate_ids"]))
    anchor_id = str(
        checkpoint["candidate_pool_diagnostic"]["original_pool_anchor_candidate_id"]
    )
    selected_id = str(logical["selected_candidate_id"])
    if selected_id != str(checkpoint["selected_candidate_id"]):
        raise ValueError(f"selected candidate identity changed: {logical['case_id']}")
    if anchor_id not in candidate_ids or selected_id not in candidate_ids:
        raise ValueError(f"anchor/selected candidate left pool: {logical['case_id']}")
    anchor = aggregates[_candidate_key(state_fingerprint, anchor_id)]
    if str(anchor["candidate_kind"]) != "base":
        raise ValueError(f"V2 anchor is no longer a base candidate: {logical['case_id']}")
    anchor_trials = trials[_candidate_key(state_fingerprint, anchor_id)]
    diagnostics = []
    for candidate_id in candidate_ids:
        if candidate_id == anchor_id:
            continue
        candidate = aggregates[_candidate_key(state_fingerprint, candidate_id)]
        diagnostics.append(
            paired_repairability_diagnostic(
                candidate,
                anchor,
                trials[_candidate_key(state_fingerprint, candidate_id)],
                anchor_trials,
                epsilon=epsilon,
            )
        )
    by_pool = {
        "base": [row for row in diagnostics if row["candidate_kind"] == "base"],
        "structural": [
            row for row in diagnostics if row["candidate_kind"] == "structural"
        ],
        "all": diagnostics,
    }
    opportunities = {}
    for pool, rows in by_pool.items():
        qualifying = [row for row in rows if row["qualifies"]]
        best = _best_opportunity(rows)
        opportunities[pool] = {
            "candidate_count": len(rows),
            "qualifying_candidate_count": len(qualifying),
            "has_opportunity": bool(best),
            "best": best,
            "has_non_larger_than_anchor_opportunity": any(
                int(row["actual_size"]) <= int(anchor["actual_size"])
                for row in qualifying
            ),
            "has_non_larger_than_selected_opportunity": any(
                int(row["actual_size"])
                <= int(
                    aggregates[_candidate_key(state_fingerprint, selected_id)][
                        "actual_size"
                    ]
                )
                for row in qualifying
            ),
        }
    selected = aggregates[_candidate_key(state_fingerprint, selected_id)]
    best_all = opportunities["all"]["best"]
    if best_all is not None and float(best_all["replan_success_rate"]) > float(
        anchor["replan_success_rate"]
    ):
        best_admissible = {
            "source": str(best_all["candidate_kind"]),
            "candidate_id": str(best_all["candidate_id"]),
            "actual_size": int(best_all["actual_size"]),
            "replan_success_rate": float(best_all["replan_success_rate"]),
            "seed_mean": float(anchor["seed_mean"])
            + float(best_all["seed_mean_delta"]),
        }
    else:
        best_admissible = {
            "source": "v2_anchor",
            "candidate_id": anchor_id,
            "actual_size": int(anchor["actual_size"]),
            "replan_success_rate": float(anchor["replan_success_rate"]),
            "seed_mean": float(anchor["seed_mean"]),
        }
    selected_diagnostic = (
        {
            "is_anchor": True,
            "qualifies_against_anchor": False,
            "replan_success_rate": float(anchor["replan_success_rate"]),
            "seed_mean": float(anchor["seed_mean"]),
            "actual_size": int(anchor["actual_size"]),
        }
        if selected_id == anchor_id
        else {
            "is_anchor": False,
            "qualifies_against_anchor": bool(
                next(
                    row for row in diagnostics if row["candidate_id"] == selected_id
                )["qualifies"]
            ),
            "replan_success_rate": float(selected["replan_success_rate"]),
            "seed_mean": float(selected["seed_mean"]),
            "actual_size": int(selected["actual_size"]),
        }
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "logical_checkpoint_id": str(logical["logical_checkpoint_id"]),
        "case_id": str(logical["case_id"]),
        "state_fingerprint": state_fingerprint,
        "map_id": str(logical["map_id"]),
        "task_id": str(logical["task_id"]),
        "solver_seed": int(logical["solver_seed"]),
        "challenger": str(logical["challenger"]),
        "checkpoint_kind": str(logical["checkpoint_kind"]),
        "classification": str(logical["classification"]),
        "anchor": {
            "candidate_id": anchor_id,
            "actual_size": int(anchor["actual_size"]),
            "seed_mean": float(anchor["seed_mean"]),
            "lower_half_mean": float(anchor["lower_half_mean"]),
            "no_progress_rate": float(anchor["no_progress_rate"]),
            "replan_success_rate": float(anchor["replan_success_rate"]),
        },
        "selected_candidate_id": selected_id,
        "selected": selected_diagnostic,
        "best_admissible": best_admissible,
        "selected_is_admissible": bool(
            selected_id == anchor_id
            or selected_diagnostic["qualifies_against_anchor"]
        ),
        "selected_repairability_gap_to_best_admissible": float(
            best_admissible["replan_success_rate"]
        )
        - float(selected["replan_success_rate"]),
        "zero_hazard_anchor_uncovered": bool(
            float(anchor["replan_success_rate"]) == 0.0
            and best_all is None
        ),
        "opportunities": opportunities,
        "future_trajectory_read": False,
        "runtime_or_ttf_read": False,
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "checkpoint_count": len(rows),
        "mean_anchor_replan_success_rate": _mean(
            float(row["anchor"]["replan_success_rate"]) for row in rows
        ),
        "mean_selected_replan_success_rate": _mean(
            float(row["selected"]["replan_success_rate"]) for row in rows
        ),
        "zero_anchor_replan_success_count": sum(
            float(row["anchor"]["replan_success_rate"]) == 0.0 for row in rows
        ),
        "zero_selected_replan_success_count": sum(
            float(row["selected"]["replan_success_rate"]) == 0.0 for row in rows
        ),
        "selected_qualifies_against_anchor_count": sum(
            bool(row["selected"]["qualifies_against_anchor"]) for row in rows
        ),
        "selected_admissible_count": sum(
            bool(row["selected_is_admissible"]) for row in rows
        ),
        "selected_admissible_fraction": (
            sum(bool(row["selected_is_admissible"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "mean_best_admissible_replan_success_rate": _mean(
            float(row["best_admissible"]["replan_success_rate"]) for row in rows
        ),
        "mean_selected_repairability_gap_to_best_admissible": _mean(
            float(row["selected_repairability_gap_to_best_admissible"])
            for row in rows
        ),
        "zero_hazard_anchor_uncovered_count": sum(
            bool(row["zero_hazard_anchor_uncovered"]) for row in rows
        ),
    }
    for pool in ("base", "structural", "all"):
        output[f"{pool}_opportunity_count"] = sum(
            bool(row["opportunities"][pool]["has_opportunity"]) for row in rows
        )
        output[f"{pool}_opportunity_fraction"] = (
            output[f"{pool}_opportunity_count"] / len(rows) if rows else 0.0
        )
        output[f"{pool}_non_larger_than_anchor_opportunity_count"] = sum(
            bool(
                row["opportunities"][pool][
                    "has_non_larger_than_anchor_opportunity"
                ]
            )
            for row in rows
        )
        output[f"{pool}_non_larger_than_selected_opportunity_count"] = sum(
            bool(
                row["opportunities"][pool][
                    "has_non_larger_than_selected_opportunity"
                ]
            )
            for row in rows
        )
    return output


def _summaries(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    return [
        {**dict(zip(fields, key)), **_group_summary(group)}
        for key, group in sorted(groups.items())
    ]


def _diagnosis(critical: dict[str, Any]) -> str:
    if int(critical["zero_hazard_anchor_uncovered_count"]) > 0:
        return "candidate_pool_gap_and_ranking_both_present"
    if int(critical["selected_admissible_count"]) < int(
        critical["checkpoint_count"]
    ):
        return "ranking_is_primary_within_current_pool_no_zero_hazard_gap_observed"
    return "no_mechanism_gap_observed"


def _render_markdown(report: dict[str, Any]) -> str:
    critical = report["critical_slice"]
    lines = [
        "# STRIDE Repairability Opportunity V1",
        "",
        "This is a frozen paired-PP opportunity audit. It reads no future trajectory or runtime, runs no solver, and trains no model.",
        "",
        "## Critical adverse first-repeat slice",
        "",
        f"- Checkpoints: `{critical['checkpoint_count']}`",
        f"- V2-base opportunity: `{critical['base_opportunity_count']}/{critical['checkpoint_count']}` (`{critical['base_opportunity_fraction']:.4f}`)",
        f"- Structural opportunity: `{critical['structural_opportunity_count']}/{critical['checkpoint_count']}` (`{critical['structural_opportunity_fraction']:.4f}`)",
        f"- Mixed-pool opportunity: `{critical['all_opportunity_count']}/{critical['checkpoint_count']}` (`{critical['all_opportunity_fraction']:.4f}`)",
        f"- Frozen selected action is admissible (V2 itself or qualifying challenger): `{critical['selected_admissible_count']}/{critical['checkpoint_count']}`",
        f"- Zero-repairability V2 anchors left uncovered: `{critical['zero_hazard_anchor_uncovered_count']}`",
        f"- Mean selected repairability: `{critical['mean_selected_replan_success_rate']:.4f}`; mean best admissible repairability: `{critical['mean_best_admissible_replan_success_rate']:.4f}`",
        f"- Diagnosis: `{report['diagnosis']}`",
        "",
        "An opportunity must preserve all registered current-step quality statistics, improve paired PP path-change success, and have nonnegative direction in both fixed eight-seed halves.",
        "",
        "## Boundary",
        "",
        "Opportunity is not a long-tail-prevention, controller, or TTF claim. Candidate generation, ranking, native PP, and end-to-end evaluation remain separate.",
        "",
    ]
    return "\n".join(lines)


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_existing_paired_pp_opportunity_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != EXPECTED_PARENT
    ):
        raise ValueError("repairability-opportunity registration changed")
    expected_boundary = {
        "paired_pp_opportunity_only": True,
        "future_trajectory_read": False,
        "runtime_or_ttf_read": False,
        "model_training_allowed": False,
        "runtime_controller_change_allowed": False,
        "candidate_pool_promotion_allowed": False,
        "longtail_prevention_claim": False,
        "ttf_improvement_claim": False,
        "generalization_claim": False,
        "no_result_based_exclusion": True,
    }
    if dict(config["claim_boundary"]) != expected_boundary:
        raise ValueError("repairability-opportunity claim boundary changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    return path, root, config, inputs


def analyze_repairability_opportunity(
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    path, _root, config, inputs = load_registration(config_path)
    collection = _read_json(inputs["collection_report"])
    source_analysis = _read_json(inputs["source_analysis"])
    plateau_report = _read_json(inputs["plateau_escape_report"])
    plateau_status = _read_json(inputs["plateau_escape_status"])
    if not (
        collection.get("complete") is True
        and collection.get("integrity_passed") is True
        and int(collection.get("error_state_count", -1)) == 0
        and int(collection.get("timeout_state_count", -1)) == 0
        and source_analysis.get("integrity_passed") is True
        and plateau_report.get("integrity_passed") is True
        and plateau_status.get("complete") is True
    ):
        raise ValueError("registered source integrity changed")

    aggregate_rows = _read_jsonl(inputs["candidate_aggregates"])
    trial_rows = _read_jsonl(inputs["repair_trials"])
    logical_rows = _read_jsonl(inputs["logical_checkpoint_results"])
    checkpoint_rows = _read_jsonl(inputs["root_diagnostic_checkpoints"])
    aggregates = {
        _candidate_key(row["state_fingerprint"], row["candidate_id"]): row
        for row in aggregate_rows
    }
    trials: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trial_rows:
        trials[_candidate_key(row["state_fingerprint"], row["candidate_id"])].append(
            row
        )
    checkpoints = {
        (str(row["case_id"]), str(row["checkpoint_kind"])): row
        for row in checkpoint_rows
    }
    epsilon = float(config["quality_noninferiority"]["epsilon"])
    output_rows = [
        _checkpoint_opportunity(
            logical,
            checkpoints[(str(logical["case_id"]), str(logical["checkpoint_kind"]))],
            aggregates,
            trials,
            epsilon=epsilon,
        )
        for logical in logical_rows
    ]
    output_rows.sort(key=lambda row: str(row["logical_checkpoint_id"]))

    critical_specification = dict(config["critical_slice"])
    critical_rows = [
        row
        for row in output_rows
        if row["checkpoint_kind"] == critical_specification["checkpoint_kind"]
        and row["classification"] == critical_specification["classification"]
    ]
    critical = _group_summary(critical_rows)
    integrity_gates = {
        "distinct_state_count_matches": len(
            {row["state_fingerprint"] for row in output_rows}
        )
        == int(config["cohort"]["distinct_state_count"]),
        "logical_checkpoint_count_matches": len(output_rows)
        == int(config["cohort"]["logical_checkpoint_count"]),
        "candidate_count_matches": len(aggregate_rows)
        == int(config["cohort"]["candidate_count"]),
        "trial_count_matches": len(trial_rows)
        == int(config["cohort"]["trial_count"]),
        "all_candidates_have_16_trials": all(len(rows) == 16 for rows in trials.values()),
        "critical_slice_count_matches": len(critical_rows)
        == int(critical_specification["expected_case_count"]),
        "all_v2_anchors_are_base": all(
            aggregates[_candidate_key(row["state_fingerprint"], row["anchor"]["candidate_id"])]["candidate_kind"]
            == "base"
            for row in output_rows
        ),
        "no_result_based_exclusion": len(output_rows) == len(logical_rows),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "frozen_paired_pp_opportunity_diagnostic_no_ttf",
        "integrity_passed": all(integrity_gates.values()),
        "integrity_gates": integrity_gates,
        "new_solver_run_count": 0,
        "model_training_performed": False,
        "future_trajectory_read": False,
        "runtime_or_ttf_read": False,
        "overall": _group_summary(output_rows),
        "critical_slice": critical,
        "by_checkpoint_kind": _summaries(output_rows, ("checkpoint_kind",)),
        "by_classification_checkpoint": _summaries(
            output_rows, ("classification", "checkpoint_kind")
        ),
        "by_map_checkpoint": _summaries(
            output_rows, ("map_id", "checkpoint_kind")
        ),
        "diagnosis": _diagnosis(critical),
        "diagnosis_rule": dict(config["diagnosis_rule"]),
        "claim_boundary": dict(config["claim_boundary"]),
        "inputs": {
            "registration_sha256": sha256_file(path),
            "registered_file_sha256": {
                name: sha256_file(value) for name, value in sorted(inputs.items())
            },
        },
    }
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "repairability_opportunity_checkpoints.jsonl"
    _write_jsonl(checkpoint_path, output_rows)
    report["checkpoint_rows_sha256"] = sha256_file(checkpoint_path)
    report_path = output_path / "repairability_opportunity_report.json"
    _write_json(report_path, report)
    markdown_path = output_path / "repairability_opportunity_report.md"
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_json(
        output_path / "repairability_opportunity_status.json",
        {
            "schema": STATUS_SCHEMA,
            "complete": True,
            "integrity_passed": report["integrity_passed"],
            "diagnosis": report["diagnosis"],
            "logical_checkpoint_count": len(output_rows),
            "report_sha256": sha256_file(report_path),
            "checkpoint_rows_sha256": report["checkpoint_rows_sha256"],
            "markdown_sha256": sha256_file(markdown_path),
        },
    )
    return report


__all__ = [
    "analyze_repairability_opportunity",
    "load_registration",
    "paired_repairability_diagnostic",
]
