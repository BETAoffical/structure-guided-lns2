from __future__ import annotations

import collections
import itertools
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, read_json, sha256_file
from experiments.receding_q_pilot import (
    RECEDING_Q_PILOT_SCHEMA,
    _correlation,
    _winner_key,
    analyze_receding_q_rollouts,
)
from experiments.repair_collection import _write_json


RECEDING_Q_GATE_AUDIT_SCHEMA = "lns2.receding_q_gate_audit.v1"
QUALITY_ABSOLUTE_TOLERANCE = 1e-12
TOP_K = 3
TIMING_TOLERANCES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.001, 0.0),
    (0.005, 0.0),
    (0.010, 0.0),
    (0.0, 0.01),
    (0.0, 0.05),
)


def _quality_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(row["feasible"]),
        float(row["final_conflict_ratio"]),
        float(row["normalized_step_auc"]),
        str(row["candidate_id"]),
    )


def _quality_equivalent(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    tolerance: float = QUALITY_ABSOLUTE_TOLERANCE,
) -> bool:
    return (
        bool(left["feasible"]) == bool(right["feasible"])
        and math.isclose(
            float(left["final_conflict_ratio"]),
            float(right["final_conflict_ratio"]),
            rel_tol=0.0,
            abs_tol=float(tolerance),
        )
        and math.isclose(
            float(left["normalized_step_auc"]),
            float(right["normalized_step_auc"]),
            rel_tol=0.0,
            abs_tol=float(tolerance),
        )
    )


def quality_winner_set(
    rows: Iterable[dict[str, Any]],
    *,
    tolerance: float = QUALITY_ABSOLUTE_TOLERANCE,
) -> set[str]:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("quality winner set requires candidates")
    best_feasible = any(bool(row["feasible"]) for row in values)
    values = [
        row for row in values if bool(row["feasible"]) == best_feasible
    ]
    best_final_ratio = min(float(row["final_conflict_ratio"]) for row in values)
    values = [
        row
        for row in values
        if float(row["final_conflict_ratio"])
        <= best_final_ratio + float(tolerance)
    ]
    best_step_auc = min(float(row["normalized_step_auc"]) for row in values)
    return {
        str(row["candidate_id"])
        for row in values
        if float(row["normalized_step_auc"])
        <= best_step_auc + float(tolerance)
    }


def _timing_window(best: float, absolute: float, relative: float) -> float:
    return max(float(absolute), abs(float(best)) * float(relative))


def timing_tolerant_winner_set(
    rows: Iterable[dict[str, Any]],
    *,
    absolute_seconds: float,
    relative_fraction: float,
) -> set[str]:
    values = [dict(row) for row in rows]
    quality_ids = quality_winner_set(values)
    values = [row for row in values if str(row["candidate_id"]) in quality_ids]
    best_wall_auc = min(
        float(row["normalized_wall_auc_seconds"]) for row in values
    )
    wall_window = _timing_window(
        best_wall_auc, absolute_seconds, relative_fraction
    )
    values = [
        row
        for row in values
        if float(row["normalized_wall_auc_seconds"])
        <= best_wall_auc + wall_window
    ]
    best_total = min(float(row["observed_total_seconds"]) for row in values)
    total_window = _timing_window(
        best_total, absolute_seconds, relative_fraction
    )
    return {
        str(row["candidate_id"])
        for row in values
        if float(row["observed_total_seconds"])
        <= best_total + total_window
    }


def _top_k(rows: Iterable[dict[str, Any]], *, k: int = TOP_K) -> set[str]:
    ordered = sorted((dict(row) for row in rows), key=_quality_key)
    return {str(row["candidate_id"]) for row in ordered[: int(k)]}


def _overlap(left: set[str], right: set[str]) -> float:
    denominator = max(1, min(len(left), len(right)))
    return float(len(left & right) / denominator)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _regret(
    target_rows: list[dict[str, Any]], selected_candidate: str
) -> dict[str, Any]:
    indexed = {str(row["candidate_id"]): row for row in target_rows}
    if selected_candidate not in indexed:
        raise ValueError("cross-seed candidate coverage mismatch")
    selected = indexed[selected_candidate]
    best = min(target_rows, key=_quality_key)
    best_ids = quality_winner_set(target_rows)
    return {
        "selected_quality_optimal": selected_candidate in best_ids,
        "feasibility_miss": bool(best["feasible"]) and not bool(
            selected["feasible"]
        ),
        "final_conflict_ratio_regret": max(
            0.0,
            float(selected["final_conflict_ratio"])
            - float(best["final_conflict_ratio"]),
        ),
        "normalized_step_auc_regret": max(
            0.0,
            float(selected["normalized_step_auc"])
            - float(best["normalized_step_auc"]),
        ),
    }


def analyze_quality_stability(
    rows: Iterable[dict[str, Any]],
    *,
    expected_trials: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    values = [dict(row) for row in rows]
    if expected_trials < 2:
        raise ValueError("quality stability audit requires at least two trials")
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in values:
        grouped[str(row["state_id"])].append(row)
    if not grouped:
        raise ValueError("quality stability audit requires states")

    state_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    timing_agreements: dict[str, list[float]] = collections.defaultdict(list)
    for state_id, state_values in sorted(grouped.items()):
        by_trial: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in state_values:
            by_trial[int(row["trial_index"])].append(row)
        if set(by_trial) != set(range(int(expected_trials))):
            raise ValueError(f"trial coverage mismatch: {state_id}")
        candidate_sets = [
            {str(row["candidate_id"]) for row in by_trial[trial]}
            for trial in range(int(expected_trials))
        ]
        if not candidate_sets[0] or any(
            candidates != candidate_sets[0] for candidates in candidate_sets[1:]
        ):
            raise ValueError(f"candidate coverage mismatch: {state_id}")
        if any(
            len(by_trial[trial]) != len(candidate_sets[trial])
            for trial in range(int(expected_trials))
        ):
            raise ValueError(f"duplicate candidate row: {state_id}")

        pair_exact: list[float] = []
        pair_quality_overlap: list[float] = []
        pair_quality_jaccard: list[float] = []
        pair_mutual_optimal: list[float] = []
        pair_top_k: list[float] = []
        pair_correlations: list[float] = []
        pair_outcome_equivalent: list[float] = []
        for left_trial, right_trial in itertools.combinations(
            range(int(expected_trials)), 2
        ):
            left_rows = by_trial[left_trial]
            right_rows = by_trial[right_trial]
            left_winner = min(left_rows, key=_winner_key)
            right_winner = min(right_rows, key=_winner_key)
            left_quality = quality_winner_set(left_rows)
            right_quality = quality_winner_set(right_rows)
            exact = str(left_winner["candidate_id"]) == str(
                right_winner["candidate_id"]
            )
            outcome_equivalent = _quality_equivalent(
                left_winner, right_winner
            )
            mutual_optimal = (
                str(left_winner["candidate_id"]) in right_quality
                and str(right_winner["candidate_id"]) in left_quality
            )
            candidate_order = sorted(candidate_sets[0])
            left_index = {
                str(row["candidate_id"]): row for row in left_rows
            }
            right_index = {
                str(row["candidate_id"]): row for row in right_rows
            }
            rank_correlation = _correlation(
                [
                    float(left_index[candidate]["normalized_step_auc"])
                    for candidate in candidate_order
                ],
                [
                    float(right_index[candidate]["normalized_step_auc"])
                    for candidate in candidate_order
                ],
            )
            regrets = []
            for source_winner, target in (
                (left_winner, right_rows),
                (right_winner, left_rows),
            ):
                regrets.append(
                    _regret(target, str(source_winner["candidate_id"]))
                )
            pair_row = {
                "state_id": state_id,
                "left_trial": left_trial,
                "right_trial": right_trial,
                "left_exact_winner": str(left_winner["candidate_id"]),
                "right_exact_winner": str(right_winner["candidate_id"]),
                "exact_winner_agreement": exact,
                "winner_outcome_equivalent": outcome_equivalent,
                "quality_winner_set_overlap": bool(
                    left_quality & right_quality
                ),
                "quality_winner_set_jaccard": _jaccard(
                    left_quality, right_quality
                ),
                "mutual_cross_seed_optimal": mutual_optimal,
                "top3_overlap": _overlap(
                    _top_k(left_rows), _top_k(right_rows)
                ),
                "paired_seed_rank_correlation": rank_correlation,
                "cross_seed_quality_optimal_fraction": statistics.fmean(
                    float(regret["selected_quality_optimal"])
                    for regret in regrets
                ),
                "cross_seed_feasibility_miss_count": sum(
                    int(regret["feasibility_miss"]) for regret in regrets
                ),
                "mean_final_conflict_ratio_regret": statistics.fmean(
                    float(regret["final_conflict_ratio_regret"])
                    for regret in regrets
                ),
                "mean_normalized_step_auc_regret": statistics.fmean(
                    float(regret["normalized_step_auc_regret"])
                    for regret in regrets
                ),
            }
            pair_rows.append(pair_row)
            pair_exact.append(float(exact))
            pair_quality_overlap.append(float(bool(left_quality & right_quality)))
            pair_quality_jaccard.append(
                float(pair_row["quality_winner_set_jaccard"])
            )
            pair_mutual_optimal.append(float(mutual_optimal))
            pair_top_k.append(float(pair_row["top3_overlap"]))
            pair_correlations.append(float(rank_correlation))
            pair_outcome_equivalent.append(float(outcome_equivalent))

            for absolute, relative in TIMING_TOLERANCES:
                label = f"abs_{absolute:.3f}_rel_{relative:.3f}"
                left_timing = timing_tolerant_winner_set(
                    left_rows,
                    absolute_seconds=absolute,
                    relative_fraction=relative,
                )
                right_timing = timing_tolerant_winner_set(
                    right_rows,
                    absolute_seconds=absolute,
                    relative_fraction=relative,
                )
                timing_agreements[label].append(
                    float(bool(left_timing & right_timing))
                )

        exact_agreement = all(bool(value) for value in pair_exact)
        true_quality_instability = not all(
            bool(value) for value in pair_quality_overlap
        )
        exact_winner_cross_seed_nonoptimal = not all(
            bool(value) for value in pair_mutual_optimal
        )
        timing_only_mismatch = (
            not exact_agreement and not exact_winner_cross_seed_nonoptimal
        )
        first = state_values[0]
        state_rows.append(
            {
                "state_id": state_id,
                "map_id": str(first.get("map_id", "")),
                "layout_mode": str(first.get("layout_mode", "")),
                "agent_count": int(first.get("agent_count", 0)),
                "candidate_count": len(candidate_sets[0]),
                "trial_count": int(expected_trials),
                "exact_winner_agreement": exact_agreement,
                "timing_only_winner_mismatch": timing_only_mismatch,
                "true_quality_instability": true_quality_instability,
                "exact_winner_cross_seed_nonoptimal": (
                    exact_winner_cross_seed_nonoptimal
                ),
                "winner_outcome_equivalent_fraction": statistics.fmean(
                    pair_outcome_equivalent
                ),
                "quality_winner_set_overlap_fraction": statistics.fmean(
                    pair_quality_overlap
                ),
                "mean_quality_winner_set_jaccard": statistics.fmean(
                    pair_quality_jaccard
                ),
                "mutual_cross_seed_optimal_fraction": statistics.fmean(
                    pair_mutual_optimal
                ),
                "mean_top3_overlap": statistics.fmean(pair_top_k),
                "paired_seed_rank_correlation": statistics.fmean(
                    pair_correlations
                ),
            }
        )

    true_quality_states = [
        str(row["state_id"])
        for row in state_rows
        if bool(row["true_quality_instability"])
    ]
    state_count = len(state_rows)
    mean_rank = statistics.fmean(
        float(row["paired_seed_rank_correlation"]) for row in pair_rows
    )
    mean_top3 = statistics.fmean(float(row["top3_overlap"]) for row in pair_rows)
    quality_overlap_fraction = statistics.fmean(
        float(row["quality_winner_set_overlap"]) for row in pair_rows
    )
    mean_auc_regret = statistics.fmean(
        float(row["mean_normalized_step_auc_regret"]) for row in pair_rows
    )
    feasibility_misses = sum(
        int(row["cross_seed_feasibility_miss_count"]) for row in pair_rows
    )
    base_checks = {
        "paired_seed_rank_correlation_at_least_0_50": mean_rank >= 0.50,
        "quality_winner_set_overlap_at_least_0_80": (
            quality_overlap_fraction >= 0.80
        ),
        "mean_top3_overlap_at_least_0_80": mean_top3 >= 0.80,
        "mean_normalized_step_auc_regret_at_most_0_02": (
            mean_auc_regret <= 0.02
        ),
        "cross_seed_feasibility_misses_at_most_1": feasibility_misses <= 1,
    }
    targeted_limit = max(2, int(math.ceil(0.20 * state_count)))
    if all(base_checks.values()) and not true_quality_states:
        decision = "quality_stability_gate_passed"
    elif (
        all(base_checks.values())
        and len(true_quality_states) <= targeted_limit
    ):
        decision = "targeted_seed_followup_required"
    else:
        decision = "quality_stability_insufficient"
    report = {
        "schema": RECEDING_Q_GATE_AUDIT_SCHEMA,
        "decision": decision,
        "diagnostic_only": True,
        "state_count": state_count,
        "candidate_count": len(
            {(str(row["state_id"]), str(row["candidate_id"])) for row in values}
        ),
        "rollout_count": len(values),
        "trial_count": int(expected_trials),
        "exact_winner_agreement_fraction": statistics.fmean(
            float(row["exact_winner_agreement"]) for row in state_rows
        ),
        "timing_only_winner_mismatch_state_count": sum(
            int(row["timing_only_winner_mismatch"]) for row in state_rows
        ),
        "exact_winner_cross_seed_nonoptimal_state_count": sum(
            int(row["exact_winner_cross_seed_nonoptimal"])
            for row in state_rows
        ),
        "true_quality_instability_state_count": len(true_quality_states),
        "true_quality_instability_state_ids": true_quality_states,
        "quality_winner_set_overlap_fraction": quality_overlap_fraction,
        "mean_quality_winner_set_jaccard": statistics.fmean(
            float(row["quality_winner_set_jaccard"]) for row in pair_rows
        ),
        "mutual_cross_seed_optimal_fraction": statistics.fmean(
            float(row["mutual_cross_seed_optimal"]) for row in pair_rows
        ),
        "mean_top3_overlap": mean_top3,
        "paired_seed_rank_correlation": mean_rank,
        "cross_seed_quality_optimal_fraction": statistics.fmean(
            float(row["cross_seed_quality_optimal_fraction"])
            for row in pair_rows
        ),
        "cross_seed_feasibility_miss_count": feasibility_misses,
        "mean_final_conflict_ratio_regret": statistics.fmean(
            float(row["mean_final_conflict_ratio_regret"])
            for row in pair_rows
        ),
        "mean_normalized_step_auc_regret": mean_auc_regret,
        "timing_tolerance_agreement": {
            label: statistics.fmean(results)
            for label, results in sorted(timing_agreements.items())
        },
        "checks": base_checks,
        "targeted_followup_state_limit": targeted_limit,
        "limitations": [
            "This is a fixed-H3 label audit, not a trained controller or a complete-episode comparison.",
            "Quality equivalence uses feasibility, final conflict ratio, and normalized fixed-step AUC; timing is reported separately.",
            "The continuation teacher remains official Adaptive for steps two and three.",
            "A targeted follow-up decision does not authorize model training until the added paired seeds pass the revised gate.",
        ],
    }
    return report, state_rows, pair_rows


def _render_report(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- {name}: `{str(bool(value)).lower()}`"
        for name, value in sorted(dict(report["checks"]).items())
    )
    targets = list(report["true_quality_instability_state_ids"])
    target_lines = (
        "\n".join(f"- `{state_id}`" for state_id in targets)
        if targets
        else "- none"
    )
    timing = "\n".join(
        f"- {name}: {float(value):.3%}"
        for name, value in dict(report["timing_tolerance_agreement"]).items()
    )
    return "\n".join(
        [
            "# Receding-Q quality-stability gate audit",
            "",
            f"Decision: `{report['decision']}`",
            "",
            "## Coverage",
            "",
            f"- States: {int(report['state_count'])}",
            f"- Candidates: {int(report['candidate_count'])}",
            f"- Rollouts: {int(report['rollout_count'])}",
            f"- Paired trials: {int(report['trial_count'])}",
            "",
            "## Stability",
            "",
            f"- Exact candidate-ID agreement: {float(report['exact_winner_agreement_fraction']):.3%}",
            f"- Timing-only mismatch states: {int(report['timing_only_winner_mismatch_state_count'])}",
            f"- Exact winners not mutually optimal across seeds: {int(report['exact_winner_cross_seed_nonoptimal_state_count'])}",
            f"- True quality-instability states: {int(report['true_quality_instability_state_count'])}",
            f"- Quality-winner-set overlap: {float(report['quality_winner_set_overlap_fraction']):.3%}",
            f"- Mean quality-winner-set Jaccard: {float(report['mean_quality_winner_set_jaccard']):.3%}",
            f"- Mean Top-3 overlap: {float(report['mean_top3_overlap']):.3%}",
            f"- Paired-seed rank correlation: {float(report['paired_seed_rank_correlation']):.6f}",
            f"- Mean normalized-step-AUC regret: {float(report['mean_normalized_step_auc_regret']):.6f}",
            f"- Cross-seed feasibility misses: {int(report['cross_seed_feasibility_miss_count'])}",
            "",
            "## Checks",
            "",
            checks,
            "",
            "## Timing tolerance sensitivity",
            "",
            timing,
            "",
            "## States requiring targeted paired seeds",
            "",
            target_lines,
            "",
            "This audit is diagnostic only and does not promote or replace `v2-full`.",
            "",
        ]
    )


def audit_receding_q_gate(*, source: Path, output: Path) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("receding-Q gate audit output is non-empty")
    plan = dict(read_json(source / "plan.json"))
    config = dict(read_json(source / "run_config.json"))
    if str(plan.get("schema")) != RECEDING_Q_PILOT_SCHEMA or str(
        config.get("schema")
    ) != RECEDING_Q_PILOT_SCHEMA:
        raise ValueError("gate audit requires current-schema receding-Q labels")
    rows = [
        dict(read_json(path))
        for path in sorted((source / "rollouts").glob("*.json"))
    ]
    source_report, _ = analyze_receding_q_rollouts(
        rows,
        plan=plan,
        trials=int(config["trials"]),
        horizon=int(config["horizon"]),
        smoke_only=bool(config.get("smoke_only", False)),
    )
    report, state_rows, pair_rows = analyze_quality_stability(
        rows, expected_trials=int(config["trials"])
    )
    report["source"] = {
        "path": str(source),
        "schema": str(config["schema"]),
        "plan_sha256": sha256_file(source / "plan.json"),
        "run_config_sha256": sha256_file(source / "run_config.json"),
        "source_report_sha256": sha256_file(
            source / "receding_q_pilot_report.json"
        ),
        "source_decision": str(source_report["decision"]),
        "source_exact_winner_agreement_fraction": float(
            source_report["winner_seed_agreement"]
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "state_quality_stability.csv", state_rows)
    atomic_write_csv(output / "pair_quality_stability.csv", pair_rows)
    _write_json(
        output / "targeted_seed_followup_states.json",
        {
            "schema": RECEDING_Q_GATE_AUDIT_SCHEMA,
            "state_ids": list(report["true_quality_instability_state_ids"]),
        },
    )
    _write_json(output / "receding_q_gate_audit_report.json", report)
    (output / "receding_q_gate_audit_report.md").write_text(
        _render_report(report), encoding="utf-8"
    )
    return report


__all__ = [
    "RECEDING_Q_GATE_AUDIT_SCHEMA",
    "analyze_quality_stability",
    "audit_receding_q_gate",
    "quality_winner_set",
    "timing_tolerant_winner_set",
]
