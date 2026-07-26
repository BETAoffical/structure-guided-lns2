from __future__ import annotations

import collections
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file, standard_error as _standard_error
from experiments.receding_q_pilot import _atomic_write_csv, _winner_key
from experiments.receding_q_risk_audit import (
    map_group_policy_selection,
    resolve_persisted_source_path,
    summarize_policy_rows,
)
from experiments.receding_q_stability import (
    _outcome_score,
    load_validated_four_seed_stability,
)
from experiments.receding_q_variance_audit import validate_four_seed_rows
from experiments.repair_collection import _write_json


RECEDING_Q_PAIRED_LABEL_AUDIT_SCHEMA = (
    "lns2.receding_q_paired_label_audit.v2"
)
PAIRED_RISK_LAMBDAS = (0.25, 0.50, 1.00, 2.00)


def paired_label_policy_grid() -> list[dict[str, Any]]:
    result = [
        {"policy_id": "mean", "method": "mean", "risk_lambda": 0.0, "complexity": 0},
        {
            "policy_id": "seed_average_rank",
            "method": "seed_average_rank",
            "risk_lambda": 0.0,
            "complexity": 1,
        },
        {
            "policy_id": "seed_pairwise_copeland",
            "method": "seed_pairwise_copeland",
            "risk_lambda": 0.0,
            "complexity": 2,
        },
    ]
    for index, value in enumerate(PAIRED_RISK_LAMBDAS):
        result.append(
            {
                "policy_id": f"paired_quality_se{value:.2f}",
                "method": "paired_quality_se",
                "risk_lambda": float(value),
                "complexity": 10 + index,
            }
        )
    return result


def _aggregate_candidate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("cannot aggregate an empty candidate")
    candidate_ids = {str(row["candidate_id"]) for row in values}
    if len(candidate_ids) != 1:
        raise ValueError("candidate aggregate mixes identities")
    return {
        "candidate_id": next(iter(candidate_ids)),
        "feasible": statistics.fmean(float(row["feasible"]) for row in values),
        "final_conflict_ratio": statistics.fmean(
            float(row["final_conflict_ratio"]) for row in values
        ),
        "normalized_step_auc": statistics.fmean(
            float(row["normalized_step_auc"]) for row in values
        ),
        "normalized_wall_auc_seconds": statistics.fmean(
            float(row["normalized_wall_auc_seconds"]) for row in values
        ),
        "observed_total_seconds": statistics.fmean(
            float(row["observed_total_seconds"]) for row in values
        ),
    }


def _aggregate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["feasible"]),
        float(row["final_conflict_ratio"]),
        float(row["normalized_step_auc"]),
        float(row["normalized_wall_auc_seconds"]),
        float(row["observed_total_seconds"]),
        str(row["candidate_id"]),
    )


def _group_candidates(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        candidate = str(row["candidate_id"])
        trial = int(row["trial_index"])
        key = (candidate, trial)
        if key in seen:
            raise ValueError("paired label selection contains a duplicate trial")
        seen.add(key)
        grouped[candidate].append(dict(row))
    if len(grouped) < 2:
        raise ValueError("paired label selection needs at least two candidates")
    trial_sets = {
        frozenset(int(row["trial_index"]) for row in candidate_rows)
        for candidate_rows in grouped.values()
    }
    if len(trial_sets) != 1:
        raise ValueError("paired label selection has unbalanced candidate trials")
    return grouped


def _mean_winner(rows: Iterable[dict[str, Any]]) -> str:
    grouped = _group_candidates(rows)
    return str(
        min(
            (_aggregate_candidate(values) for values in grouped.values()),
            key=_aggregate_key,
        )["candidate_id"]
    )


def _average_rank_winner(rows: Iterable[dict[str, Any]]) -> str:
    values = [dict(row) for row in rows]
    grouped = _group_candidates(values)
    ranks: dict[str, list[float]] = collections.defaultdict(list)
    for trial in sorted({int(row["trial_index"]) for row in values}):
        ordered = sorted(
            (row for row in values if int(row["trial_index"]) == trial),
            key=_winner_key,
        )
        start = 0
        while start < len(ordered):
            outcome_key = _winner_key(ordered[start])[:-1]
            end = start + 1
            while (
                end < len(ordered)
                and _winner_key(ordered[end])[:-1] == outcome_key
            ):
                end += 1
            average_rank = (start + end - 1) / 2.0
            for row in ordered[start:end]:
                ranks[str(row["candidate_id"])].append(average_rank)
            start = end
    aggregates = {
        candidate: _aggregate_candidate(candidate_rows)
        for candidate, candidate_rows in grouped.items()
    }
    return min(
        grouped,
        key=lambda candidate: (
            statistics.fmean(ranks[candidate]),
            _aggregate_key(aggregates[candidate]),
        ),
    )


def _copeland_winner(rows: Iterable[dict[str, Any]]) -> str:
    values = [dict(row) for row in rows]
    grouped = _group_candidates(values)
    trials = sorted({int(row["trial_index"]) for row in values})
    by_key = {
        (str(row["candidate_id"]), int(row["trial_index"])): row
        for row in values
    }
    scores = {candidate: 0 for candidate in grouped}
    seed_net = {candidate: 0 for candidate in grouped}
    for left, right in combinations(sorted(grouped), 2):
        votes = [
            _outcome_score(by_key[(left, trial)], by_key[(right, trial)])
            for trial in trials
        ]
        net = sum(votes)
        seed_net[left] += net
        seed_net[right] -= net
        if net > 0:
            scores[left] += 1
            scores[right] -= 1
        elif net < 0:
            scores[left] -= 1
            scores[right] += 1
    aggregates = {
        candidate: _aggregate_candidate(candidate_rows)
        for candidate, candidate_rows in grouped.items()
    }
    return min(
        grouped,
        key=lambda candidate: (
            -scores[candidate],
            -seed_net[candidate],
            _aggregate_key(aggregates[candidate]),
        ),
    )


def _paired_quality_winner(
    rows: Iterable[dict[str, Any]], *, risk_lambda: float
) -> str:
    values = [dict(row) for row in rows]
    grouped = _group_candidates(values)
    metrics = (
        ("feasible", True),
        ("final_conflict_ratio", False),
        ("normalized_step_auc", False),
        ("normalized_wall_auc_seconds", False),
        ("observed_total_seconds", False),
    )
    trial_means: dict[tuple[int, str], float] = {}
    for trial in sorted({int(row["trial_index"]) for row in values}):
        trial_rows = [
            row for row in values if int(row["trial_index"]) == trial
        ]
        for metric, _ in metrics:
            trial_means[(trial, metric)] = statistics.fmean(
                float(row[metric]) for row in trial_rows
            )
    scores = {}
    for candidate, candidate_rows in grouped.items():
        candidate_scores = {}
        for metric, benefit in metrics:
            centered = [
                float(row[metric])
                - trial_means[(int(row["trial_index"]), metric)]
                for row in candidate_rows
            ]
            mean = statistics.fmean(centered)
            error = _standard_error(centered)
            candidate_scores[metric] = (
                mean - risk_lambda * error
                if benefit
                else mean + risk_lambda * error
            )
        scores[candidate] = candidate_scores
    aggregates = {
        candidate: _aggregate_candidate(candidate_rows)
        for candidate, candidate_rows in grouped.items()
    }
    return min(
        grouped,
        key=lambda candidate: (
            -scores[candidate]["feasible"],
            scores[candidate]["final_conflict_ratio"],
            scores[candidate]["normalized_step_auc"],
            scores[candidate]["normalized_wall_auc_seconds"],
            scores[candidate]["observed_total_seconds"],
            _aggregate_key(aggregates[candidate]),
        ),
    )


def select_paired_label_candidate(
    rows: Iterable[dict[str, Any]], *, policy: dict[str, Any]
) -> str:
    values = [dict(row) for row in rows]
    method = str(policy["method"])
    if method == "mean":
        return _mean_winner(values)
    if method == "seed_average_rank":
        return _average_rank_winner(values)
    if method == "seed_pairwise_copeland":
        return _copeland_winner(values)
    if method == "paired_quality_se":
        return _paired_quality_winner(
            values, risk_lambda=float(policy["risk_lambda"])
        )
    raise ValueError(f"unsupported paired label method: {method}")


def build_paired_label_loo_rows(
    rows: Iterable[dict[str, Any]],
    *,
    target_state_ids: Iterable[str],
    policies: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    target_ids = set(map(str, target_state_ids))
    validate_four_seed_rows(values, target_state_ids=target_ids)
    selected_policies = list(policies or paired_label_policy_grid())
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in values:
        if str(row["state_id"]) in target_ids:
            grouped[str(row["state_id"])].append(row)
    result = []
    for state_id, state_rows in sorted(grouped.items()):
        for heldout in range(4):
            training = [
                row
                for row in state_rows
                if int(row["trial_index"]) != heldout
            ]
            test_rows = [
                row
                for row in state_rows
                if int(row["trial_index"]) == heldout
            ]
            by_candidate = {
                str(row["candidate_id"]): row for row in test_rows
            }
            v2_rows = [row for row in test_rows if bool(row["is_v2_candidate"])]
            if len(v2_rows) != 1:
                raise ValueError("held-out seed lacks exactly one v2 candidate")
            v2 = v2_rows[0]
            oracle = min(test_rows, key=_winner_key)
            for policy in selected_policies:
                candidate_id = select_paired_label_candidate(
                    training, policy=policy
                )
                selected = by_candidate[candidate_id]
                result.append(
                    {
                        "policy_id": str(policy["policy_id"]),
                        "risk_mode": str(policy["method"]),
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
                        "oracle_normalized_step_auc_regret": float(
                            selected["normalized_step_auc"]
                        )
                        - float(oracle["normalized_step_auc"]),
                    }
                )
    return result


def _policy_key(row: dict[str, Any]) -> tuple[Any, ...]:
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


def _summaries_by_agent(
    rows: list[dict[str, Any]], *, policy_id: str
) -> list[dict[str, Any]]:
    result = []
    for agent_count in sorted({int(row["agent_count"]) for row in rows}):
        subset = [
            {**row, "policy_id": policy_id}
            for row in rows
            if int(row["agent_count"]) == agent_count
        ]
        result.append(
            {"agent_count": agent_count, **summarize_policy_rows(subset)}
        )
    return result


def _paired_label_checks(
    *,
    loo_rows: list[dict[str, Any]],
    target_state_ids: list[str],
    policies: list[dict[str, Any]],
    oof_summary: dict[str, Any],
    high_load: dict[str, Any] | None,
) -> dict[str, bool]:
    return {
        "coverage_complete": len(loo_rows)
        == len(target_state_ids) * 4 * len(policies),
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
        "six_hundred_oof_auc_not_below_v2": (
            high_load is not None
            and float(high_load["mean_normalized_step_auc_delta"]) <= 0.0
        ),
        "six_hundred_oof_feasible_rate_not_below_v2": (
            high_load is not None
            and float(high_load["feasible_rate_delta"]) >= 0.0
        ),
        "six_hundred_oof_time_not_below_v2": (
            high_load is not None
            and float(high_load["mean_total_seconds_delta"]) <= 0.0
        ),
        "six_hundred_oof_wins_not_below_losses": (
            high_load is not None and int(high_load["net_wins"]) >= 0
        ),
    }


def paired_label_audit_decision(checks: dict[str, bool]) -> str:
    return (
        "paired_h3_label_transform_promising"
        if checks and all(map(bool, checks.values()))
        else "paired_h3_label_transform_not_supported_by_current_audit"
    )


def audit_receding_q_paired_labels(
    *, stability: str | Path, output: str | Path
) -> dict[str, Any]:
    stability_root = Path(stability).resolve()
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("paired label audit output is non-empty")
    output_root.mkdir(parents=True, exist_ok=True)

    config = dict(read_json(stability_root / "run_config.json"))
    source_root = resolve_persisted_source_path(
        config["source"], sibling_root=stability_root.parent
    )
    validated = load_validated_four_seed_stability(
        stability_root,
        source_root,
    )
    targets = dict(validated["targets"])
    target_state_ids = list(map(str, targets["target_state_ids"]))
    merged = list(validated["merged_rows"])
    policies = paired_label_policy_grid()
    loo_rows = build_paired_label_loo_rows(
        merged, target_state_ids=target_state_ids, policies=policies
    )
    policy_summaries = [
        summarize_policy_rows(
            [row for row in loo_rows if str(row["policy_id"]) == policy["policy_id"]]
        )
        for policy in policies
    ]
    selected_global = min(policy_summaries, key=_policy_key)
    oof_rows, map_selections = map_group_policy_selection(loo_rows)
    oof_summary = summarize_policy_rows(
        [{**row, "policy_id": "map_group_oof"} for row in oof_rows]
    )
    agent_rows = _summaries_by_agent(oof_rows, policy_id="map_group_oof")
    fixed_agent_rows = []
    for policy in policies:
        fixed_agent_rows.extend(
            _summaries_by_agent(
                [
                    row
                    for row in loo_rows
                    if str(row["policy_id"]) == str(policy["policy_id"])
                ],
                policy_id=str(policy["policy_id"]),
            )
        )
    high_load = next(
        (row for row in agent_rows if int(row["agent_count"]) == 600), None
    )
    checks = _paired_label_checks(
        loo_rows=loo_rows,
        target_state_ids=target_state_ids,
        policies=policies,
        oof_summary=oof_summary,
        high_load=high_load,
    )
    decision = paired_label_audit_decision(checks)
    report = {
        "schema": RECEDING_Q_PAIRED_LABEL_AUDIT_SCHEMA,
        "decision": decision,
        "source_stability": str(stability_root),
        "source_stability_sha256": sha256_file(
            stability_root / "receding_q_stability_report.json"
        ),
        "state_count": len(target_state_ids),
        "map_count": len({str(row["map_id"]) for row in loo_rows}),
        "policy_count": len(policies),
        "fold_count_per_policy": len(target_state_ids) * 4,
        "exploratory_globally_selected_policy": selected_global,
        "gate_policy_selection": "leave_one_map_out",
        "map_group_oof": oof_summary,
        "map_group_selections": map_selections,
        "agent_summaries": agent_rows,
        "fixed_policy_agent_summaries": fixed_agent_rows,
        "checks": checks,
        "limitations": [
            "This audit reuses seven unstable policy_train states and does not train a feature-based model.",
            "Seed-centering and rank aggregation use paired outcomes only; they do not change the Adaptive continuation teacher.",
            "Map-group selection is diagnostic because only five maps and seven states are available.",
            "Only leave-one-map-out selections contribute to the gate; global and fixed-policy summaries are exploratory.",
            "A passing transform would justify a fresh independent pilot, not controller promotion.",
        ],
    }
    _atomic_write_csv(output_root / "paired_label_loo_rows.csv", loo_rows)
    _atomic_write_csv(
        output_root / "paired_label_policy_summary.csv", policy_summaries
    )
    _atomic_write_csv(
        output_root / "paired_label_map_selection.csv", map_selections
    )
    _atomic_write_csv(output_root / "paired_label_agent_summary.csv", agent_rows)
    _atomic_write_csv(
        output_root / "paired_label_fixed_agent_summary.csv",
        fixed_agent_rows,
    )
    _write_json(
        output_root / "receding_q_paired_label_audit_report.json", report
    )
    (output_root / "receding_q_paired_label_audit_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": RECEDING_Q_PAIRED_LABEL_AUDIT_SCHEMA,
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
            "# Receding-Q paired-label audit",
            "",
            f"Decision: `{report['decision']}`",
            "",
            (
                f"- States: {int(report['state_count'])}; maps: "
                f"{int(report['map_count'])}; methods: "
                f"{int(report['policy_count'])}; folds per method: "
                f"{int(report['fold_count_per_policy'])}."
            ),
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
    "PAIRED_RISK_LAMBDAS",
    "RECEDING_Q_PAIRED_LABEL_AUDIT_SCHEMA",
    "audit_receding_q_paired_labels",
    "build_paired_label_loo_rows",
    "paired_label_audit_decision",
    "paired_label_policy_grid",
    "select_paired_label_candidate",
]
