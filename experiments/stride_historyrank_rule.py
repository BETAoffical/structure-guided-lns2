from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_marginalpool_action_replay import stable_dominates


CONFIG_SCHEMA = "lns2.stride.historyrank_rule_registration.v1"
REPORT_SCHEMA = "lns2.stride.historyrank_rule_report.v1"
ROW_SCHEMA = "lns2.stride.historyrank_rule_row.v1"
EXPERIMENT_ID = "stride-historyrank-rule-v1"
SCIENTIFIC_STATUS = "preregistered_history_aware_exact_noop_rerank_audit"


def load_registration(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    config_path = Path(path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or config.get("experiment_id") != EXPERIMENT_ID
        or dict(config.get("cohort") or {})
        != {
            "checkpoint_kind": "first_repeat_stall",
            "required_checkpoint_count": 45,
            "retain_all_registered_checkpoints": True,
            "classification_must_not_affect_selection": True,
            "no_result_based_exclusion": True,
        }
        or dict(config.get("history_rule") or {})
        != {
            "activation": (
                "two immediately preceding selections used the same candidate as "
                "the current frozen winner and both native PP calls were exact "
                "path-level no-ops"
            ),
            "excluded_action": "the exact repeated candidate_id only",
            "fallback": (
                "highest frozen V2 score among every remaining existing candidate"
            ),
            "score_order": "descending numeric score then ascending candidate_id",
            "candidate_generation_unchanged": True,
            "pp_and_sipps_unchanged": True,
            "no_jaccard_or_family_ban": True,
        }
    ):
        raise ValueError("HistoryRank rule registration changed")
    inputs = {
        name: registered_input(root, dict(spec), label=name)
        for name, spec in dict(config["inputs"]).items()
    }
    return config_path, config, inputs


def choose_fallback(
    candidates: list[dict[str, Any]], *, repeated_candidate_id: str
) -> dict[str, Any]:
    remaining = [
        dict(row)
        for row in candidates
        if str(row["candidate_id"]) != repeated_candidate_id
    ]
    if not remaining:
        raise ValueError("HistoryRank rule has no fallback candidate")
    return min(
        remaining,
        key=lambda row: (-float(row["score"]), str(row["candidate_id"])),
    )


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "checkpoint_count": len(rows),
        "stable_improvement_count": sum(bool(row["stable_improvement"]) for row in rows),
        "stable_improvement_fraction": statistics.fmean(
            float(row["stable_improvement"]) for row in rows
        ),
        "mean_seed_improvement": _mean(rows, "seed_mean_delta"),
        "mean_no_progress_delta": _mean(rows, "no_progress_rate_delta"),
        "both_halves_positive_fraction": statistics.fmean(
            float(row["both_halves_positive"]) for row in rows
        ),
        "mean_selected_regret": _mean(rows, "selected_normalized_regret"),
        "mean_fallback_regret": _mean(rows, "fallback_normalized_regret"),
        "root_class_counts": dict(Counter(str(row["root_class"]) for row in rows)),
    }


def run_historyrank_rule(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, config, inputs = load_registration(config_path)
    collection = _read_json(inputs["collection_report"])
    analysis = _read_json(inputs["action_replay_analysis"])
    if (
        collection.get("complete") is not True
        or collection.get("integrity_passed") is not True
        or int(collection.get("state_count", -1)) != 78
        or int(collection.get("candidate_count", -1)) != 2502
        or int(collection.get("trial_count", -1)) != 40032
        or analysis.get("integrity_passed") is not True
        or int(dict(analysis.get("summary") or {}).get("checkpoint_count", -1)) != 90
    ):
        raise ValueError("HistoryRank rule input collection is not complete")

    cases = {str(row["case_id"]): row for row in _read_jsonl(inputs["root_cases"])}
    checkpoints = [
        row
        for row in _read_jsonl(inputs["root_checkpoints"])
        if row.get("checkpoint_kind") == "first_repeat_stall"
    ]
    logical = {
        str(row["logical_checkpoint_id"]): row
        for row in _read_jsonl(inputs["logical_results"])
        if row.get("checkpoint_kind") == "first_repeat_stall"
    }
    aggregates = {
        (str(row["state_fingerprint"]), str(row["candidate_id"])): row
        for row in _read_jsonl(inputs["candidate_aggregates"])
    }
    required = int(config["cohort"]["required_checkpoint_count"])
    if len(checkpoints) != required or len(logical) != required or len(cases) != required:
        raise ValueError("HistoryRank rule cohort count changed")

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint["logical_checkpoint_id"])
        result = logical[checkpoint_id]
        case = cases[str(checkpoint["case_id"])]
        repeated = str(case["repeated_candidate_id"])
        prior = list(dict(case["pp_response"])["prior_repairs"])
        if (
            dict(case["pp_response"]).get("true_pp_noop") is not True
            or len(prior) != 2
            or any(str(row["candidate_id"]) != repeated for row in prior)
            or any(int(row["changed_agent_count"]) != 0 for row in prior)
            or any(dict(row["repair_metadata"])["replan_success"] is not False for row in prior)
            or any(
                int(dict(row["repair_metadata"])["conflicts_before"])
                != int(dict(row["repair_metadata"])["conflicts_after"])
                for row in prior
            )
            or str(checkpoint["selected_candidate_id"]) != repeated
            or str(result["selected_candidate_id"]) != repeated
        ):
            raise ValueError("HistoryRank activation history changed")
        fallback = choose_fallback(
            list(checkpoint["candidate_pool"]), repeated_candidate_id=repeated
        )
        fallback_id = str(fallback["candidate_id"])
        state_key = str(checkpoint["state_fingerprint"])
        selected_quality = aggregates[(state_key, repeated)]
        fallback_quality = aggregates[(state_key, fallback_id)]
        best_mean = max(
            float(aggregates[(state_key, str(candidate_id))]["seed_mean"])
            for candidate_id in result["candidate_ids"]
        )
        before = max(1.0, float(checkpoint["before_conflicts"]))
        row = {
            "schema": ROW_SCHEMA,
            "logical_checkpoint_id": checkpoint_id,
            "case_id": str(checkpoint["case_id"]),
            "state_fingerprint": state_key,
            "map_id": str(checkpoint["map_id"]),
            "task_id": str(checkpoint["task_id"]),
            "solver_seed": int(checkpoint["solver_seed"]),
            "classification": str(checkpoint["classification"]),
            "root_class": str(result["root_class"]),
            "selected_candidate_id": repeated,
            "fallback_candidate_id": fallback_id,
            "selected_v2_score": float(
                next(
                    candidate["score"]
                    for candidate in checkpoint["candidate_pool"]
                    if str(candidate["candidate_id"]) == repeated
                )
            ),
            "fallback_v2_score": float(fallback["score"]),
            "seed_mean_delta": float(fallback_quality["seed_mean"])
            - float(selected_quality["seed_mean"]),
            "no_progress_rate_delta": float(fallback_quality["no_progress_rate"])
            - float(selected_quality["no_progress_rate"]),
            "first_half_delta": float(fallback_quality["first_fixed_half_mean"])
            - float(selected_quality["first_fixed_half_mean"]),
            "second_half_delta": float(fallback_quality["second_fixed_half_mean"])
            - float(selected_quality["second_fixed_half_mean"]),
            "both_halves_positive": (
                float(fallback_quality["first_fixed_half_mean"])
                > float(selected_quality["first_fixed_half_mean"])
                and float(fallback_quality["second_fixed_half_mean"])
                > float(selected_quality["second_fixed_half_mean"])
            ),
            "stable_improvement": stable_dominates(
                fallback_quality,
                selected_quality,
                minimum=float(config["quality_rule"]["minimum_seed_mean_advantage"]),
            ),
            "selected_normalized_regret": best_mean - float(selected_quality["seed_mean"]),
            "fallback_normalized_regret": best_mean - float(fallback_quality["seed_mean"]),
            "before_conflicts": int(checkpoint["before_conflicts"]),
            "regret_scale_reference": before,
            "runtime_or_ttf_read": False,
            "future_trajectory_read": False,
        }
        rows.append(row)

    rows.sort(key=lambda row: str(row["logical_checkpoint_id"]))
    summary = _summary(rows)
    by_map = {
        map_id: _summary([row for row in rows if row["map_id"] == map_id])
        for map_id in sorted({str(row["map_id"]) for row in rows})
    }
    gates = dict(config["gates"])
    gate_results = {
        "complete_cohort": len(rows) == required,
        "stable_improvement_fraction": float(summary["stable_improvement_fraction"])
        >= float(gates["minimum_stable_improvement_fraction"]),
        "mean_seed_improvement": float(summary["mean_seed_improvement"])
        >= float(gates["minimum_mean_seed_improvement"]),
        "mean_no_progress_delta": float(summary["mean_no_progress_delta"])
        <= float(gates["maximum_mean_no_progress_delta"]),
        "both_halves_positive_fraction": float(summary["both_halves_positive_fraction"])
        >= float(gates["minimum_both_half_positive_fraction"]),
        "every_map_mean_seed_improvement": all(
            float(group["mean_seed_improvement"])
            > float(gates["minimum_per_map_mean_seed_improvement"])
            for group in by_map.values()
        ),
        "zero_errors": int(gates["required_error_count"]) == 0,
    }
    passed = all(gate_results.values())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    row_path = output / "historyrank_rule_rows.jsonl"
    _write_jsonl(row_path, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "offline_checkpoint_local_exact_noop_rerank_audit",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "summary": summary,
        "by_map": by_map,
        "gates": gate_results,
        "history_rule_passed": passed,
        "rows_sha256": sha256_file(row_path),
        "runtime_or_ttf_read": False,
        "future_trajectory_read": False,
        "claim_boundary": dict(config["claim_boundary"]),
        "next_step": (
            "preregister fresh-state runtime semantics without TTF"
            if passed
            else "preregister stride-historyrank-v1 feature audit over the full frozen pool"
        ),
    }
    _write_json(output / "historyrank_rule_report.json", report)
    _write_json(
        output / "historyrank_rule_status.json",
        {
            "schema": "lns2.stride.historyrank_rule_status.v1",
            "complete": True,
            "integrity_passed": True,
            "history_rule_passed": passed,
            "report_sha256": sha256_file(output / "historyrank_rule_report.json"),
        },
    )
    return report
