from __future__ import annotations

import collections
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

from experiments._common import read_json, sha256_file
from experiments.receding_q_pilot import _atomic_write_csv
from experiments.receding_q_risk_audit import resolve_persisted_source_path
from experiments.receding_q_stability import (
    load_receding_q_rollouts,
    merge_followup_rollouts,
)
from experiments.repair_collection import _write_json


RECEDING_Q_VARIANCE_AUDIT_SCHEMA = "lns2.receding_q_variance_audit.v1"


def _root_step(row: dict[str, Any]) -> dict[str, Any]:
    steps = list(row.get("steps", []))
    if not steps or int(steps[0].get("step", -1)) != 1:
        raise ValueError("rollout lacks a valid root repair step")
    return dict(steps[0])


def _state_changed(row: dict[str, Any]) -> float:
    step = _root_step(row)
    return float(
        str(step["after_fingerprint"]) != str(step["before_fingerprint"])
    )


MetricExtractor = Callable[[dict[str, Any]], float]
METRICS: tuple[tuple[str, str, MetricExtractor], ...] = (
    (
        "root_conflict_reduction",
        "root",
        lambda row: float(_root_step(row)["conflict_reduction"]),
    ),
    ("root_state_changed", "root", _state_changed),
    (
        "root_conflict_reduced",
        "root",
        lambda row: float(
            float(_root_step(row)["conflict_reduction"]) > 0.0
        ),
    ),
    (
        "root_pp_replan_seconds",
        "root",
        lambda row: float(_root_step(row)["pp_replan_seconds"]),
    ),
    (
        "root_iteration_wall_seconds",
        "root",
        lambda row: float(_root_step(row)["iteration_wall_seconds"]),
    ),
    (
        "h3_normalized_step_auc",
        "h3",
        lambda row: float(row["normalized_step_auc"]),
    ),
    (
        "h3_final_conflict_ratio",
        "h3",
        lambda row: float(row["final_conflict_ratio"]),
    ),
    (
        "h3_observed_total_seconds",
        "h3",
        lambda row: float(row["observed_total_seconds"]),
    ),
    ("h3_feasible", "h3", lambda row: float(bool(row["feasible"]))),
)


def decompose_balanced_candidate_seed(
    rows: Iterable[dict[str, Any]],
    *,
    value: MetricExtractor,
) -> dict[str, Any]:
    """Two-way decomposition for a balanced candidate by PP-seed matrix.

    There is one deterministic observation per cell, so the residual from the
    additive candidate-plus-seed model is reported as candidate-seed
    interaction. It must not be interpreted as an independently estimated
    random-error term.
    """

    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("variance decomposition requires observations")
    candidates = sorted({str(row["candidate_id"]) for row in values})
    trials = sorted({int(row["trial_index"]) for row in values})
    cells = {
        (str(row["candidate_id"]), int(row["trial_index"])): float(value(row))
        for row in values
    }
    expected = {
        (candidate, trial)
        for candidate in candidates
        for trial in trials
    }
    if len(cells) != len(values) or set(cells) != expected:
        raise ValueError("candidate-seed matrix is not balanced and unique")
    if len(candidates) < 2 or len(trials) < 2:
        raise ValueError("candidate-seed decomposition needs two axes")

    grand_mean = statistics.fmean(cells.values())
    candidate_means = {
        candidate: statistics.fmean(
            cells[(candidate, trial)] for trial in trials
        )
        for candidate in candidates
    }
    seed_means = {
        trial: statistics.fmean(
            cells[(candidate, trial)] for candidate in candidates
        )
        for trial in trials
    }
    candidate_ss = len(trials) * math.fsum(
        (candidate_means[candidate] - grand_mean) ** 2
        for candidate in candidates
    )
    seed_ss = len(candidates) * math.fsum(
        (seed_means[trial] - grand_mean) ** 2 for trial in trials
    )
    interaction_ss = math.fsum(
        (
            cells[(candidate, trial)]
            - candidate_means[candidate]
            - seed_means[trial]
            + grand_mean
        )
        ** 2
        for candidate in candidates
        for trial in trials
    )
    total_ss = math.fsum(
        (cell_value - grand_mean) ** 2 for cell_value in cells.values()
    )
    closure_error = abs(
        total_ss - candidate_ss - seed_ss - interaction_ss
    )
    tolerance = max(1e-12, abs(total_ss) * 1e-12)
    if closure_error > tolerance:
        raise ValueError("candidate-seed variance decomposition did not close")

    def fraction(component: float) -> float:
        return float(component / total_ss) if total_ss > 0.0 else 0.0

    return {
        "candidate_count": len(candidates),
        "seed_count": len(trials),
        "cell_count": len(cells),
        "mean": float(grand_mean),
        "total_ss": float(total_ss),
        "candidate_ss": float(candidate_ss),
        "seed_ss": float(seed_ss),
        "interaction_ss": float(interaction_ss),
        "candidate_fraction": fraction(candidate_ss),
        "seed_fraction": fraction(seed_ss),
        "interaction_fraction": fraction(interaction_ss),
        "noncandidate_fraction": fraction(seed_ss + interaction_ss),
        "closure_error": float(closure_error),
        "informative": bool(total_ss > 0.0),
    }


def validate_four_seed_rows(
    rows: Iterable[dict[str, Any]], *, target_state_ids: Iterable[str]
) -> dict[str, Any]:
    target_ids = set(map(str, target_state_ids))
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        state_id = str(row["state_id"])
        if state_id in target_ids:
            grouped[state_id].append(dict(row))
    if set(grouped) != target_ids:
        raise ValueError("variance audit does not cover every target state")

    candidate_count = 0
    for state_id, state_rows in grouped.items():
        candidates = {str(row["candidate_id"]) for row in state_rows}
        trials = {int(row["trial_index"]) for row in state_rows}
        if trials != {0, 1, 2, 3}:
            raise ValueError(f"state lacks four PP trials: {state_id}")
        expected = {
            (candidate, trial)
            for candidate in candidates
            for trial in trials
        }
        actual = {
            (str(row["candidate_id"]), int(row["trial_index"]))
            for row in state_rows
        }
        if len(actual) != len(state_rows) or actual != expected:
            raise ValueError(f"state has incomplete candidate coverage: {state_id}")
        for trial in trials:
            trial_rows = [
                row
                for row in state_rows
                if int(row["trial_index"]) == trial
            ]
            requested = {
                int(_root_step(row)["requested_pp_seed"]) for row in trial_rows
            }
            applied = {
                int(_root_step(row)["applied_pp_seed"]) for row in trial_rows
            }
            if len(requested) != 1 or requested != applied:
                raise ValueError(f"state lacks paired PP seeds: {state_id}")
        candidate_count += len(candidates)
    return {
        "state_count": len(grouped),
        "unique_state_candidate_count": candidate_count,
        "rollout_count": sum(len(values) for values in grouped.values()),
    }


def build_state_variance_rows(
    rows: Iterable[dict[str, Any]], *, target_state_ids: Iterable[str]
) -> list[dict[str, Any]]:
    target_ids = set(map(str, target_state_ids))
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if str(row["state_id"]) in target_ids:
            grouped[str(row["state_id"])].append(dict(row))

    result = []
    for state_id, state_rows in sorted(grouped.items()):
        first = state_rows[0]
        for metric_id, phase, extractor in METRICS:
            decomposition = decompose_balanced_candidate_seed(
                state_rows, value=extractor
            )
            result.append(
                {
                    "state_id": state_id,
                    "map_id": str(first["map_id"]),
                    "layout_mode": str(first["layout_mode"]),
                    "agent_count": int(first["agent_count"]),
                    "metric": metric_id,
                    "phase": phase,
                    **decomposition,
                }
            )
    return result


def summarize_variance_rows(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in values:
        groups[(str(row["metric"]), str(int(row["agent_count"])))].append(row)
        groups[(str(row["metric"]), "all")].append(row)

    result = []
    for (metric, agent_group), grouped in sorted(groups.items()):
        total_ss = math.fsum(float(row["total_ss"]) for row in grouped)
        candidate_ss = math.fsum(float(row["candidate_ss"]) for row in grouped)
        seed_ss = math.fsum(float(row["seed_ss"]) for row in grouped)
        interaction_ss = math.fsum(
            float(row["interaction_ss"]) for row in grouped
        )

        def fraction(component: float) -> float:
            return float(component / total_ss) if total_ss > 0.0 else 0.0

        result.append(
            {
                "metric": metric,
                "phase": str(grouped[0]["phase"]),
                "agent_group": agent_group,
                "state_count": len(grouped),
                "map_count": len({str(row["map_id"]) for row in grouped}),
                "informative_state_count": sum(
                    bool(row["informative"]) for row in grouped
                ),
                "cell_count": sum(int(row["cell_count"]) for row in grouped),
                "total_ss": float(total_ss),
                "candidate_ss": float(candidate_ss),
                "seed_ss": float(seed_ss),
                "interaction_ss": float(interaction_ss),
                "candidate_fraction": fraction(candidate_ss),
                "seed_fraction": fraction(seed_ss),
                "interaction_fraction": fraction(interaction_ss),
                "noncandidate_fraction": fraction(seed_ss + interaction_ss),
                "mean_state_interaction_fraction": statistics.fmean(
                    float(row["interaction_fraction"])
                    for row in grouped
                    if bool(row["informative"])
                )
                if any(bool(row["informative"]) for row in grouped)
                else 0.0,
            }
        )
    return result


def build_candidate_variability_rows(
    rows: Iterable[dict[str, Any]], *, target_state_ids: Iterable[str]
) -> list[dict[str, Any]]:
    target_ids = set(map(str, target_state_ids))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        if str(row["state_id"]) in target_ids:
            grouped[(str(row["state_id"]), str(row["candidate_id"]))].append(
                dict(row)
            )
    result = []
    for (state_id, candidate_id), candidate_rows in sorted(grouped.items()):
        if {int(row["trial_index"]) for row in candidate_rows} != {0, 1, 2, 3}:
            raise ValueError("candidate variability lacks four trials")
        first = candidate_rows[0]

        def values(extractor: MetricExtractor) -> list[float]:
            return [float(extractor(row)) for row in candidate_rows]

        root_reduction = values(METRICS[0][2])
        root_pp = values(METRICS[3][2])
        h3_auc = values(METRICS[5][2])
        h3_time = values(METRICS[7][2])
        result.append(
            {
                "state_id": state_id,
                "candidate_id": candidate_id,
                "map_id": str(first["map_id"]),
                "layout_mode": str(first["layout_mode"]),
                "agent_count": int(first["agent_count"]),
                "actual_size": int(first["actual_size"]),
                "selection_families": "+".join(
                    map(str, first.get("selection_families", []))
                ),
                "is_v2_candidate": bool(first["is_v2_candidate"]),
                "root_state_changed_rate": statistics.fmean(
                    values(_state_changed)
                ),
                "root_conflict_reduction_mean": statistics.fmean(
                    root_reduction
                ),
                "root_conflict_reduction_std": statistics.pstdev(
                    root_reduction
                ),
                "root_pp_seconds_mean": statistics.fmean(root_pp),
                "root_pp_seconds_std": statistics.pstdev(root_pp),
                "h3_auc_mean": statistics.fmean(h3_auc),
                "h3_auc_std": statistics.pstdev(h3_auc),
                "h3_total_seconds_mean": statistics.fmean(h3_time),
                "h3_total_seconds_std": statistics.pstdev(h3_time),
            }
        )
    return result


def summarize_candidate_variability(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        groups[(int(row["agent_count"]), int(row["actual_size"]))].append(
            dict(row)
        )
    result = []
    for (agent_count, actual_size), values in sorted(groups.items()):
        result.append(
            {
                "agent_count": agent_count,
                "actual_size": actual_size,
                "state_count": len({str(row["state_id"]) for row in values}),
                "candidate_count": len(values),
                "mean_root_state_changed_rate": statistics.fmean(
                    float(row["root_state_changed_rate"]) for row in values
                ),
                "mean_root_conflict_reduction": statistics.fmean(
                    float(row["root_conflict_reduction_mean"]) for row in values
                ),
                "mean_within_candidate_root_reduction_std": statistics.fmean(
                    float(row["root_conflict_reduction_std"]) for row in values
                ),
                "mean_root_pp_seconds": statistics.fmean(
                    float(row["root_pp_seconds_mean"]) for row in values
                ),
                "mean_within_candidate_root_pp_seconds_std": statistics.fmean(
                    float(row["root_pp_seconds_std"]) for row in values
                ),
                "mean_h3_auc": statistics.fmean(
                    float(row["h3_auc_mean"]) for row in values
                ),
                "mean_within_candidate_h3_auc_std": statistics.fmean(
                    float(row["h3_auc_std"]) for row in values
                ),
                "mean_h3_total_seconds": statistics.fmean(
                    float(row["h3_total_seconds_mean"]) for row in values
                ),
                "mean_within_candidate_h3_total_seconds_std": statistics.fmean(
                    float(row["h3_total_seconds_std"]) for row in values
                ),
            }
        )
    return result


def _lookup_summary(
    rows: Iterable[dict[str, Any]], metric: str, agent_group: str
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in rows
        if str(row["metric"]) == metric
        and str(row["agent_group"]) == agent_group
    ]
    if len(matches) != 1:
        raise ValueError(f"variance summary is missing {metric}/{agent_group}")
    return matches[0]


def classify_variance_source(
    summaries: Iterable[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    rows = [dict(row) for row in summaries]
    root = _lookup_summary(rows, "root_conflict_reduction", "600")
    root_changed = _lookup_summary(rows, "root_state_changed", "600")
    h3 = _lookup_summary(rows, "h3_normalized_step_auc", "600")
    root_order_fraction = max(
        float(root["noncandidate_fraction"]),
        float(root_changed["noncandidate_fraction"]),
    )
    root_interaction_fraction = max(
        float(root["interaction_fraction"]),
        float(root_changed["interaction_fraction"]),
    )
    continuation_amplification = (
        float(h3["noncandidate_fraction"]) - root_order_fraction
    )
    diagnostic = {
        "six_hundred_root_order_fraction": root_order_fraction,
        "six_hundred_root_interaction_fraction": root_interaction_fraction,
        "six_hundred_h3_noncandidate_fraction": float(
            h3["noncandidate_fraction"]
        ),
        "six_hundred_continuation_amplification": continuation_amplification,
    }
    if root_order_fraction >= 0.50:
        if root_interaction_fraction >= 0.35:
            return "probe_candidate_conditioned_pp_order", diagnostic
        return "probe_global_deterministic_pp_order", diagnostic
    if continuation_amplification >= 0.20:
        return "fix_h3_continuation_labels_before_pp_order", diagnostic
    return "candidate_signal_dominates_collect_independent_states", diagnostic


def audit_receding_q_variance(
    *, stability: str | Path, output: str | Path
) -> dict[str, Any]:
    stability_root = Path(stability).resolve()
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("variance audit output is non-empty")
    output_root.mkdir(parents=True, exist_ok=True)

    status = dict(read_json(stability_root / "status.json"))
    if str(status.get("status")) != "complete" or int(
        status.get("error_count", -1)
    ) != 0:
        raise ValueError("stability source is not complete and clean")
    config = dict(read_json(stability_root / "run_config.json"))
    source_root = resolve_persisted_source_path(
        config["source"], sibling_root=stability_root.parent
    )
    targets = dict(read_json(stability_root / "targets.json"))
    target_state_ids = list(map(str, targets["target_state_ids"]))
    source_rows = load_receding_q_rollouts(source_root)
    followup_rows = load_receding_q_rollouts(stability_root)
    merged_rows = merge_followup_rollouts(source_rows, followup_rows)
    coverage = validate_four_seed_rows(
        merged_rows, target_state_ids=target_state_ids
    )
    state_rows = build_state_variance_rows(
        merged_rows, target_state_ids=target_state_ids
    )
    summaries = summarize_variance_rows(state_rows)
    candidate_rows = build_candidate_variability_rows(
        merged_rows, target_state_ids=target_state_ids
    )
    size_rows = summarize_candidate_variability(candidate_rows)
    decision, diagnostic = classify_variance_source(summaries)
    checks = {
        "coverage_complete": int(coverage["state_count"])
        == len(target_state_ids),
        "four_paired_pp_seeds": all(
            int(row["seed_count"]) == 4 for row in state_rows
        ),
        "decomposition_closed": all(
            float(row["closure_error"])
            <= max(1e-12, abs(float(row["total_ss"])) * 1e-12)
            for row in state_rows
        ),
        "six_hundred_states_present": any(
            str(row["agent_group"]) == "600" for row in summaries
        ),
    }
    if not all(checks.values()):
        raise ValueError("variance audit integrity checks failed")
    report = {
        "schema": RECEDING_Q_VARIANCE_AUDIT_SCHEMA,
        "decision": decision,
        "source_stability": str(stability_root),
        "source_stability_sha256": sha256_file(
            stability_root / "receding_q_stability_report.json"
        ),
        **coverage,
        "map_count": len(
            {
                str(row["map_id"])
                for row in merged_rows
                if str(row["state_id"]) in set(target_state_ids)
            }
        ),
        "metric_count": len(METRICS),
        "diagnostic": diagnostic,
        "checks": checks,
        "limitations": [
            "The audit reuses seven unstable policy_train states and does not rerun the solver.",
            "There is one deterministic observation per candidate-seed cell; residual variance is candidate-seed interaction, not an independently estimated noise term.",
            "A PP seed controls stochastic repair behavior, but the saved rollout does not contain the realized agent repair-order list.",
            "The 28 seed folds are not 28 independent states; evidence remains limited to seven states on five maps.",
        ],
    }
    _atomic_write_csv(output_root / "state_metric_variance.csv", state_rows)
    _atomic_write_csv(
        output_root / "variance_summary.csv", summaries
    )
    _atomic_write_csv(
        output_root / "candidate_seed_variability.csv", candidate_rows
    )
    _atomic_write_csv(
        output_root / "size_seed_variability.csv", size_rows
    )
    _write_json(
        output_root / "receding_q_variance_audit_report.json", report
    )
    (output_root / "receding_q_variance_audit_report.md").write_text(
        _markdown(report, summaries), encoding="utf-8"
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": RECEDING_Q_VARIANCE_AUDIT_SCHEMA,
            "status": "complete",
            "error_count": 0,
            "decision": decision,
        },
    )
    return report


def _markdown(
    report: dict[str, Any], summaries: Iterable[dict[str, Any]]
) -> str:
    rows = [dict(row) for row in summaries]
    selected = [
        row
        for row in rows
        if str(row["agent_group"]) == "600"
        and str(row["metric"])
        in {
            "root_conflict_reduction",
            "root_state_changed",
            "root_pp_replan_seconds",
            "h3_normalized_step_auc",
            "h3_observed_total_seconds",
        }
    ]
    lines = [
        "# Receding-Q candidate/PP-seed variance audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        (
            f"- Coverage: {int(report['state_count'])} states, "
            f"{int(report['map_count'])} maps, "
            f"{int(report['unique_state_candidate_count'])} state-candidates, "
            f"{int(report['rollout_count'])} four-seed rollouts."
        ),
        "- Fractions are pooled sums of squares within state.",
        "",
        "## 600-agent decomposition",
        "",
        "| Metric | Candidate | PP seed | Candidate x seed |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {row['metric']} | "
            f"{float(row['candidate_fraction']):.1%} | "
            f"{float(row['seed_fraction']):.1%} | "
            f"{float(row['interaction_fraction']):.1%} |"
        )
        for row in selected
    )
    lines.extend(
        [
            "",
            "## Integrity checks",
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
    return "\n".join(lines)


__all__ = [
    "METRICS",
    "RECEDING_Q_VARIANCE_AUDIT_SCHEMA",
    "audit_receding_q_variance",
    "build_candidate_variability_rows",
    "build_state_variance_rows",
    "classify_variance_source",
    "decompose_balanced_candidate_seed",
    "summarize_candidate_variability",
    "summarize_variance_rows",
    "validate_four_seed_rows",
]
