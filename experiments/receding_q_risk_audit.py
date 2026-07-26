from __future__ import annotations

import collections
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file, standard_error as _standard_error
from experiments.receding_q_pilot import _atomic_write_csv, _winner_key
from experiments.receding_q_stability import (
    _outcome_score,
    load_validated_four_seed_stability,
)
from experiments.repair_collection import _write_json


RECEDING_Q_RISK_AUDIT_SCHEMA = "lns2.receding_q_risk_audit.v1"
RISK_LAMBDAS = (0.25, 0.50, 1.00, 2.00)
RISK_MODES = ("auc", "quality", "quality_time")


def resolve_persisted_source_path(
    value: str | Path, *, sibling_root: str | Path
) -> Path:
    """Resolve a source path persisted by either WSL or Windows."""

    path = Path(value)
    if path.is_dir():
        return path.resolve()
    text = str(value).replace("\\", "/")
    if os.name == "nt" and text.startswith("/mnt/") and len(text) >= 7:
        drive = text[5]
        converted = Path(f"{drive.upper()}:/{text[7:]}")
        if converted.is_dir():
            return converted.resolve()
    if os.name != "nt" and len(text) >= 3 and text[1:3] == ":/":
        converted = Path(f"/mnt/{text[0].lower()}/{text[3:]}")
        if converted.is_dir():
            return converted.resolve()
    sibling = Path(sibling_root).resolve() / Path(text).name
    if sibling.is_dir():
        return sibling
    raise ValueError(f"cannot resolve persisted source path: {value}")


def risk_policy_grid() -> list[dict[str, Any]]:
    result = [
        {
            "policy_id": "mean",
            "risk_mode": "mean",
            "risk_lambda": 0.0,
            "complexity": 0,
        }
    ]
    mode_order = {name: index + 1 for index, name in enumerate(RISK_MODES)}
    for mode in RISK_MODES:
        for value in RISK_LAMBDAS:
            result.append(
                {
                    "policy_id": f"{mode}_se{value:.2f}",
                    "risk_mode": mode,
                    "risk_lambda": float(value),
                    "complexity": mode_order[mode] * 10
                    + RISK_LAMBDAS.index(value)
                    + 1,
                }
            )
    return result


def risk_candidate_summary(
    rows: Iterable[dict[str, Any]],
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("cannot score an empty risk candidate")
    candidate_ids = {str(row["candidate_id"]) for row in values}
    if len(candidate_ids) != 1:
        raise ValueError("risk candidate rows contain multiple identities")

    metrics = {
        "feasible": [float(row["feasible"]) for row in values],
        "final": [float(row["final_conflict_ratio"]) for row in values],
        "auc": [float(row["normalized_step_auc"]) for row in values],
        "wall_auc": [
            float(row["normalized_wall_auc_seconds"]) for row in values
        ],
        "time": [float(row["observed_total_seconds"]) for row in values],
    }
    means = {
        name: statistics.fmean(metric_values)
        for name, metric_values in metrics.items()
    }
    errors = {
        name: _standard_error(metric_values)
        for name, metric_values in metrics.items()
    }
    mode = str(policy["risk_mode"])
    risk_lambda = float(policy["risk_lambda"])
    penalize_quality = mode in {"quality", "quality_time"}
    penalize_auc = mode in {"auc", "quality", "quality_time"}
    penalize_time = mode == "quality_time"
    return {
        "candidate_id": next(iter(candidate_ids)),
        "mean_feasible_rate": means["feasible"],
        "mean_final_conflict_ratio": means["final"],
        "mean_normalized_step_auc": means["auc"],
        "mean_normalized_wall_auc_seconds": means["wall_auc"],
        "mean_total_seconds": means["time"],
        "se_feasible_rate": errors["feasible"],
        "se_final_conflict_ratio": errors["final"],
        "se_normalized_step_auc": errors["auc"],
        "se_normalized_wall_auc_seconds": errors["wall_auc"],
        "se_total_seconds": errors["time"],
        "score_feasible_lcb": means["feasible"]
        - (risk_lambda * errors["feasible"] if penalize_quality else 0.0),
        "score_final_conflict_ratio_ucb": means["final"]
        + (risk_lambda * errors["final"] if penalize_quality else 0.0),
        "score_normalized_step_auc_ucb": means["auc"]
        + (risk_lambda * errors["auc"] if penalize_auc else 0.0),
        "score_normalized_wall_auc_seconds_ucb": means["wall_auc"]
        + (risk_lambda * errors["wall_auc"] if penalize_quality else 0.0),
        "score_total_seconds_ucb": means["time"]
        + (risk_lambda * errors["time"] if penalize_time else 0.0),
    }


def _risk_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["score_feasible_lcb"]),
        float(row["score_final_conflict_ratio_ucb"]),
        float(row["score_normalized_step_auc_ucb"]),
        float(row["score_normalized_wall_auc_seconds_ucb"]),
        float(row["score_total_seconds_ucb"]),
        str(row["candidate_id"]),
    )


def select_risk_candidate(
    rows: Iterable[dict[str, Any]],
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(dict(row))
    if len(grouped) < 2:
        raise ValueError("risk selection requires at least two candidates")
    return min(
        (
            risk_candidate_summary(values, policy=policy)
            for values in grouped.values()
        ),
        key=_risk_key,
    )


def build_risk_loo_rows(
    rows: list[dict[str, Any]],
    *,
    target_state_ids: Iterable[str],
    policies: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected_policies = list(policies or risk_policy_grid())
    target_ids = set(map(str, target_state_ids))
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if str(row["state_id"]) in target_ids:
            grouped[str(row["state_id"])].append(dict(row))
    if set(grouped) != target_ids:
        raise ValueError("risk audit does not cover all target states")

    result = []
    for state_id, state_rows in sorted(grouped.items()):
        candidate_ids = {str(row["candidate_id"]) for row in state_rows}
        keys = {
            (str(row["candidate_id"]), int(row["trial_index"]))
            for row in state_rows
        }
        expected = {
            (candidate_id, trial)
            for candidate_id in candidate_ids
            for trial in range(4)
        }
        if keys != expected:
            raise ValueError(f"risk audit four-seed coverage mismatch: {state_id}")
        for heldout in range(4):
            training = [
                row
                for row in state_rows
                if int(row["trial_index"]) != heldout
            ]
            heldout_rows = [
                row
                for row in state_rows
                if int(row["trial_index"]) == heldout
            ]
            v2_rows = [row for row in heldout_rows if bool(row["is_v2_candidate"])]
            if len(v2_rows) != 1:
                raise ValueError("risk audit held-out fold lacks one v2 candidate")
            oracle = min(heldout_rows, key=_winner_key)
            by_candidate = {
                str(row["candidate_id"]): row for row in heldout_rows
            }
            for policy in selected_policies:
                selected_summary = select_risk_candidate(
                    training, policy=policy
                )
                candidate_id = str(selected_summary["candidate_id"])
                selected = dict(by_candidate[candidate_id])
                v2 = dict(v2_rows[0])
                result.append(
                    {
                        "policy_id": str(policy["policy_id"]),
                        "risk_mode": str(policy["risk_mode"]),
                        "risk_lambda": float(policy["risk_lambda"]),
                        "policy_complexity": int(policy["complexity"]),
                        "state_id": state_id,
                        "map_id": str(selected["map_id"]),
                        "layout_mode": str(selected["layout_mode"]),
                        "agent_count": int(selected["agent_count"]),
                        "heldout_trial": heldout,
                        "selected_candidate_id": candidate_id,
                        "v2_candidate_id": str(v2["candidate_id"]),
                        "oracle_candidate_id": str(oracle["candidate_id"]),
                        "selected_vs_v2_outcome": _outcome_score(selected, v2),
                        "selected_feasible": bool(selected["feasible"]),
                        "v2_feasible": bool(v2["feasible"]),
                        "selected_minus_v2_final_conflict_ratio": float(
                            selected["final_conflict_ratio"]
                        )
                        - float(v2["final_conflict_ratio"]),
                        "selected_minus_v2_normalized_step_auc": float(
                            selected["normalized_step_auc"]
                        )
                        - float(v2["normalized_step_auc"]),
                        "selected_minus_v2_total_seconds": float(
                            selected["observed_total_seconds"]
                        )
                        - float(v2["observed_total_seconds"]),
                        "oracle_final_conflict_ratio_regret": float(
                            selected["final_conflict_ratio"]
                        )
                        - float(oracle["final_conflict_ratio"]),
                        "oracle_normalized_step_auc_regret": float(
                            selected["normalized_step_auc"]
                        )
                        - float(oracle["normalized_step_auc"]),
                        "selected_training_auc_mean": float(
                            selected_summary["mean_normalized_step_auc"]
                        ),
                        "selected_training_auc_se": float(
                            selected_summary["se_normalized_step_auc"]
                        ),
                    }
                )
    return result


def summarize_policy_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("cannot summarize empty risk-policy rows")
    policy_ids = {str(row["policy_id"]) for row in values}
    if len(policy_ids) != 1:
        raise ValueError("risk-policy summary mixes policies")
    wins = sum(int(row["selected_vs_v2_outcome"]) > 0 for row in values)
    losses = sum(int(row["selected_vs_v2_outcome"]) < 0 for row in values)
    ties = len(values) - wins - losses
    return {
        "policy_id": next(iter(policy_ids)),
        "risk_mode": str(values[0]["risk_mode"]),
        "risk_lambda": float(values[0]["risk_lambda"]),
        "policy_complexity": int(values[0]["policy_complexity"]),
        "fold_count": len(values),
        "map_count": len({str(row["map_id"]) for row in values}),
        "state_count": len({str(row["state_id"]) for row in values}),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_wins": wins - losses,
        "selected_feasible_rate": statistics.fmean(
            float(row["selected_feasible"]) for row in values
        ),
        "v2_feasible_rate": statistics.fmean(
            float(row["v2_feasible"]) for row in values
        ),
        "feasible_rate_delta": statistics.fmean(
            float(row["selected_feasible"]) - float(row["v2_feasible"])
            for row in values
        ),
        "mean_final_conflict_ratio_delta": statistics.fmean(
            float(row["selected_minus_v2_final_conflict_ratio"])
            for row in values
        ),
        "mean_normalized_step_auc_delta": statistics.fmean(
            float(row["selected_minus_v2_normalized_step_auc"])
            for row in values
        ),
        "mean_total_seconds_delta": statistics.fmean(
            float(row["selected_minus_v2_total_seconds"])
            for row in values
        ),
        "mean_oracle_normalized_step_auc_regret": statistics.fmean(
            float(row["oracle_normalized_step_auc_regret"]) for row in values
        ),
    }


def _policy_training_key(row: dict[str, Any]) -> tuple[Any, ...]:
    eligible = (
        float(row["feasible_rate_delta"]) >= 0.0
        and float(row["mean_normalized_step_auc_delta"]) <= 0.02
        and int(row["net_wins"]) >= 0
    )
    return (
        not eligible,
        -float(row["feasible_rate_delta"]),
        -int(row["net_wins"]),
        float(row["mean_normalized_step_auc_delta"]),
        float(row["mean_final_conflict_ratio_delta"]),
        float(row["mean_total_seconds_delta"]),
        int(row["policy_complexity"]),
        str(row["policy_id"]),
    )


def map_group_policy_selection(
    loo_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    maps = sorted({str(row["map_id"]) for row in loo_rows})
    policy_ids = sorted({str(row["policy_id"]) for row in loo_rows})
    if len(maps) < 2:
        raise ValueError("risk audit requires at least two map groups")
    selected_rows = []
    selection_rows = []
    for heldout_map in maps:
        training_summaries = []
        for policy_id in policy_ids:
            subset = [
                row
                for row in loo_rows
                if str(row["policy_id"]) == policy_id
                and str(row["map_id"]) != heldout_map
            ]
            training_summaries.append(summarize_policy_rows(subset))
        selected = min(training_summaries, key=_policy_training_key)
        test_rows = [
            dict(row)
            for row in loo_rows
            if str(row["policy_id"]) == str(selected["policy_id"])
            and str(row["map_id"]) == heldout_map
        ]
        selected_rows.extend(test_rows)
        selection_rows.append(
            {
                "heldout_map_id": heldout_map,
                "selected_policy_id": str(selected["policy_id"]),
                "training_map_count": int(selected["map_count"]),
                "training_state_count": int(selected["state_count"]),
                "training_net_wins": int(selected["net_wins"]),
                "training_feasible_rate_delta": float(
                    selected["feasible_rate_delta"]
                ),
                "training_normalized_step_auc_delta": float(
                    selected["mean_normalized_step_auc_delta"]
                ),
                "heldout_fold_count": len(test_rows),
            }
        )
    return selected_rows, selection_rows


def _summaries_by_agent(
    rows: list[dict[str, Any]], *, policy_id: str
) -> list[dict[str, Any]]:
    result = []
    for agent_count, values in sorted(
        (
            (agent, [row for row in rows if int(row["agent_count"]) == agent])
            for agent in {int(row["agent_count"]) for row in rows}
        )
    ):
        prepared = [{**row, "policy_id": policy_id} for row in values]
        result.append(
            {"agent_count": agent_count, **summarize_policy_rows(prepared)}
        )
    return result


def audit_receding_q_risk(
    *,
    stability: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    stability_root = Path(stability).resolve()
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("risk audit output must be empty")
    output_root.mkdir(parents=True, exist_ok=True)

    stability_config = dict(
        read_json(stability_root / "run_config.json")
    )
    source_root = resolve_persisted_source_path(
        str(stability_config["source"]),
        sibling_root=stability_root.parent,
    )
    validated = load_validated_four_seed_stability(
        stability_root,
        source_root,
    )
    merged = list(validated["merged_rows"])
    targets = dict(validated["targets"])
    policies = risk_policy_grid()
    loo_rows = build_risk_loo_rows(
        merged,
        target_state_ids=targets["target_state_ids"],
        policies=policies,
    )

    policy_summaries = [
        summarize_policy_rows(
            row for row in loo_rows if str(row["policy_id"]) == policy["policy_id"]
        )
        for policy in policies
    ]
    globally_selected = min(policy_summaries, key=_policy_training_key)
    oof_rows, map_selections = map_group_policy_selection(loo_rows)
    oof_summary = summarize_policy_rows(
        [{**row, "policy_id": "map_group_oof"} for row in oof_rows]
    )
    agent_rows = _summaries_by_agent(oof_rows, policy_id="map_group_oof")
    fixed_policy_agent_rows = []
    for policy in policies:
        policy_rows = [
            row
            for row in loo_rows
            if str(row["policy_id"]) == str(policy["policy_id"])
        ]
        fixed_policy_agent_rows.extend(
            _summaries_by_agent(
                policy_rows, policy_id=str(policy["policy_id"])
            )
        )
    six_hundred_policy_rows = [
        row
        for row in fixed_policy_agent_rows
        if int(row["agent_count"]) == 600
    ]
    passing_six_hundred_policies = [
        row
        for row in six_hundred_policy_rows
        if float(row["feasible_rate_delta"]) >= 0.0
        and float(row["mean_normalized_step_auc_delta"]) <= 0.0
        and float(row["mean_total_seconds_delta"]) <= 0.0
        and int(row["net_wins"]) >= 0
    ]
    agent_lookup = {
        int(row["agent_count"]): row for row in agent_rows
    }
    high_load = agent_lookup.get(600)
    lower_load_rows = [
        row for row in oof_rows if int(row["agent_count"]) <= 400
    ]
    lower_load_summary = summarize_policy_rows(
        [{**row, "policy_id": "map_group_oof_le400"} for row in lower_load_rows]
    )
    mean_lower_rows = [
        row
        for row in loo_rows
        if str(row["policy_id"]) == "mean"
        and int(row["agent_count"]) <= 400
    ]
    mean_lower_summary = summarize_policy_rows(mean_lower_rows)
    checks = {
        "coverage_complete": len(loo_rows)
        == len(targets["target_state_ids"]) * 4 * len(policies),
        "at_least_four_map_groups": len(
            {str(row["map_id"]) for row in loo_rows}
        )
        >= 4,
        "oof_feasible_rate_not_below_v2": float(
            oof_summary["feasible_rate_delta"]
        )
        >= 0.0,
        "oof_auc_not_below_v2": float(
            oof_summary["mean_normalized_step_auc_delta"]
        )
        <= 0.0,
        "oof_time_not_below_v2": float(
            oof_summary["mean_total_seconds_delta"]
        )
        <= 0.0,
        "oof_wins_not_below_losses": int(oof_summary["net_wins"]) >= 0,
        "six_hundred_agent_auc_not_below_v2": (
            high_load is not None
            and float(high_load["mean_normalized_step_auc_delta"]) <= 0.0
        ),
        "six_hundred_agent_time_not_below_v2": (
            high_load is not None
            and float(high_load["mean_total_seconds_delta"]) <= 0.0
        ),
        "six_hundred_agent_wins_not_below_losses": (
            high_load is not None and int(high_load["net_wins"]) >= 0
        ),
        "some_fixed_policy_passes_all_six_hundred_agent_gates": bool(
            passing_six_hundred_policies
        ),
        "lower_load_auc_retains_mean_policy_within_2pct": float(
            lower_load_summary["mean_normalized_step_auc_delta"]
        )
        <= float(mean_lower_summary["mean_normalized_step_auc_delta"]) + 0.02,
    }
    decision = (
        "risk_aware_receding_q_signal_promising_needs_fresh_validation"
        if all(checks.values())
        else "risk_penalty_insufficient_fix_pp_order"
    )
    report = {
        "schema": RECEDING_Q_RISK_AUDIT_SCHEMA,
        "decision": decision,
        "source_stability": str(stability_root),
        "source_stability_sha256": sha256_file(
            stability_root / "receding_q_stability_report.json"
        ),
        "state_count": len(targets["target_state_ids"]),
        "map_count": len({str(row["map_id"]) for row in loo_rows}),
        "policy_count": len(policies),
        "fold_count_per_policy": len(targets["target_state_ids"]) * 4,
        "risk_definition": (
            "Feasibility uses mean-lambda*standard_error; cost metrics use "
            "mean+lambda*standard_error according to the policy mode."
        ),
        "globally_selected_policy": globally_selected,
        "map_group_oof": oof_summary,
        "map_group_selections": map_selections,
        "agent_summaries": agent_rows,
        "six_hundred_fixed_policy_summaries": six_hundred_policy_rows,
        "passing_six_hundred_fixed_policy_count": len(
            passing_six_hundred_policies
        ),
        "lower_load_oof_summary": lower_load_summary,
        "lower_load_mean_policy_summary": mean_lower_summary,
        "checks": checks,
        "limitations": [
            "The audit reuses seven previously identified unstable policy_train states; it does not add independent states or solver runs.",
            "Lambda is selected inside each outer map fold using only the other maps, but the final method still requires fresh map-isolated validation.",
            "Each held-out fold is a PP seed outcome, not an independent state or map.",
            "The audit scores measured candidate outcomes directly; no feature-based model is trained.",
        ],
    }
    _atomic_write_csv(output_root / "risk_loo_rows.csv", loo_rows)
    _atomic_write_csv(
        output_root / "risk_policy_summary.csv", policy_summaries
    )
    _atomic_write_csv(
        output_root / "map_group_selection.csv", map_selections
    )
    _atomic_write_csv(output_root / "agent_summary.csv", agent_rows)
    _atomic_write_csv(
        output_root / "fixed_policy_agent_summary.csv",
        fixed_policy_agent_rows,
    )
    _write_json(output_root / "receding_q_risk_audit_report.json", report)
    (output_root / "receding_q_risk_audit_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": RECEDING_Q_RISK_AUDIT_SCHEMA,
            "status": "complete",
            "error_count": 0,
            "decision": decision,
        },
    )
    return report


def _markdown(report: dict[str, Any]) -> str:
    oof = dict(report["map_group_oof"])
    return "\n".join(
        [
            "# Receding-Q risk audit",
            "",
            f"Decision: `{report['decision']}`",
            "",
            (
                f"- States: {int(report['state_count'])}; maps: "
                f"{int(report['map_count'])}; policies: "
                f"{int(report['policy_count'])}; folds per policy: "
                f"{int(report['fold_count_per_policy'])}."
            ),
            f"- Risk definition: {report['risk_definition']}",
            (
                "- Map-group OOF vs v2: "
                f"{int(oof['wins'])} wins / {int(oof['losses'])} losses / "
                f"{int(oof['ties'])} ties; feasible delta "
                f"{float(oof['feasible_rate_delta']):+.3%}; AUC delta "
                f"{float(oof['mean_normalized_step_auc_delta']):+.4f}; "
                f"time delta {float(oof['mean_total_seconds_delta']):+.4f}s."
            ),
            "",
            "## Checks",
            "",
            *[
                f"- {name}: `{str(bool(value)).lower()}`"
                for name, value in sorted(dict(report["checks"]).items())
            ],
            "",
            "## Scientific boundary",
            "",
            *[f"- {value}" for value in report["limitations"]],
            "",
        ]
    )


__all__ = [
    "RECEDING_Q_RISK_AUDIT_SCHEMA",
    "RISK_LAMBDAS",
    "RISK_MODES",
    "audit_receding_q_risk",
    "build_risk_loo_rows",
    "map_group_policy_selection",
    "resolve_persisted_source_path",
    "risk_candidate_summary",
    "risk_policy_grid",
    "select_risk_candidate",
    "summarize_policy_rows",
]
