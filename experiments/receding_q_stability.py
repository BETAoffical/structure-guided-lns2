from __future__ import annotations

import collections
import csv
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.receding_q_pilot import (
    RECEDING_Q_PILOT_SCHEMA,
    _atomic_write_csv,
    _correlation,
    _rollout_file_name,
    _rollout_flat,
    _winner_key,
    run_receding_q_rollout,
)
from experiments.repair_collection import _fingerprint, _write_json


RECEDING_Q_STABILITY_SCHEMA = "lns2.receding_q_label_stability.v1"
FOLLOWUP_TRIALS = (2, 3)


def _rollout_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["state_id"]),
        str(row["candidate_id"]),
        int(row["trial_index"]),
    )


def load_receding_q_rollouts(root: str | Path) -> list[dict[str, Any]]:
    rollout_root = Path(root).resolve() / "rollouts"
    rows = [
        dict(read_json(path))
        for path in sorted(rollout_root.glob("*.json"))
    ]
    if not rows:
        raise ValueError("receding-Q source has no rollout files")
    keys = [_rollout_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("receding-Q source contains duplicate rollout keys")
    if not all(bool(row.get("complete")) for row in rows):
        raise ValueError("receding-Q source contains incomplete rollouts")
    return rows


def identify_stability_targets(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        grouped[(str(row["state_id"]), int(row["trial_index"]))].append(
            dict(row)
        )
    state_trials: dict[str, dict[int, str]] = collections.defaultdict(dict)
    for (state_id, trial), values in grouped.items():
        state_trials[state_id][trial] = str(
            min(values, key=_winner_key)["candidate_id"]
        )
    target_states = []
    stable_states = []
    for state_id, winners in sorted(state_trials.items()):
        if set(winners) != {0, 1}:
            raise ValueError(
                f"source state lacks trial 0/1 coverage: {state_id}"
            )
        if winners[0] == winners[1]:
            stable_states.append(state_id)
        else:
            target_states.append(state_id)
    if not target_states:
        raise ValueError("receding-Q source has no unstable winner states")
    return {
        "target_state_ids": target_states,
        "stable_state_ids": stable_states,
        "target_state_count": len(target_states),
        "stable_state_count": len(stable_states),
        "source_state_count": len(state_trials),
    }


def build_followup_jobs(
    *,
    plan: dict[str, Any],
    targets: dict[str, Any],
    horizon: int,
    continuation_teacher: str,
) -> list[dict[str, Any]]:
    target_ids = set(map(str, targets["target_state_ids"]))
    states = [
        dict(state)
        for state in plan["states"]
        if str(state["state_id"]) in target_ids
    ]
    if {str(state["state_id"]) for state in states} != target_ids:
        raise ValueError("stability targets are not covered by the source plan")
    return [
        {
            "state": state,
            "arm": dict(arm),
            "trial_index": trial_index,
            "horizon": int(horizon),
            "continuation_teacher": str(continuation_teacher),
        }
        for state in states
        for arm in state["arms"]
        for trial_index in FOLLOWUP_TRIALS
    ]


def merge_followup_rollouts(
    source_rows: Iterable[dict[str, Any]],
    followup_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    source = [dict(row) for row in source_rows]
    merged = {_rollout_key(row): row for row in source}
    if len(merged) != len(source):
        raise ValueError("source rollout keys are not unique")
    for row in followup_rows:
        key = _rollout_key(row)
        if key in merged:
            raise ValueError(f"follow-up duplicates a source rollout: {key}")
        merged[key] = dict(row)
    return [merged[key] for key in sorted(merged)]


def _aggregate_candidate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("cannot aggregate an empty candidate")
    return {
        "candidate_id": str(values[0]["candidate_id"]),
        "feasible_rate": statistics.fmean(
            float(row["feasible"]) for row in values
        ),
        "mean_final_conflict_ratio": statistics.fmean(
            float(row["final_conflict_ratio"]) for row in values
        ),
        "mean_normalized_step_auc": statistics.fmean(
            float(row["normalized_step_auc"]) for row in values
        ),
        "mean_normalized_wall_auc_seconds": statistics.fmean(
            float(row["normalized_wall_auc_seconds"]) for row in values
        ),
        "mean_total_seconds": statistics.fmean(
            float(row["observed_total_seconds"]) for row in values
        ),
        "std_normalized_step_auc": (
            statistics.pstdev(
                float(row["normalized_step_auc"]) for row in values
            )
            if len(values) > 1
            else 0.0
        ),
    }


def _aggregate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["feasible_rate"]),
        float(row["mean_final_conflict_ratio"]),
        float(row["mean_normalized_step_auc"]),
        float(row["mean_normalized_wall_auc_seconds"]),
        float(row["mean_total_seconds"]),
        str(row["candidate_id"]),
    )


def _candidate_aggregates(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(dict(row))
    return sorted(
        (_aggregate_candidate(values) for values in grouped.values()),
        key=_aggregate_key,
    )


def _quality_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(row["feasible"]),
        float(row["final_conflict_ratio"]),
        float(row["normalized_step_auc"]),
        float(row["normalized_wall_auc_seconds"]),
        float(row["observed_total_seconds"]),
    )


def _outcome_score(selected: dict[str, Any], baseline: dict[str, Any]) -> int:
    selected_key = _quality_key(selected)
    baseline_key = _quality_key(baseline)
    if selected_key < baseline_key:
        return 1
    if selected_key > baseline_key:
        return -1
    return 0


def _candidate_rows(
    rows: Iterable[dict[str, Any]],
    candidate_id: str,
    trials: set[int],
) -> list[dict[str, Any]]:
    result = [
        dict(row)
        for row in rows
        if str(row["candidate_id"]) == str(candidate_id)
        and int(row["trial_index"]) in trials
    ]
    if len(result) != len(trials):
        raise ValueError(
            f"candidate lacks requested trials: {candidate_id}/{sorted(trials)}"
        )
    return result


def analyze_four_seed_stability(
    rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    targets: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    target_ids = set(map(str, targets["target_state_ids"]))
    plan_states = {
        str(state["state_id"]): dict(state) for state in plan["states"]
    }
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        state_id = str(row["state_id"])
        if state_id in target_ids:
            grouped[state_id].append(dict(row))
    if set(grouped) != target_ids:
        raise ValueError("merged stability rows do not cover all target states")

    state_diagnostics = []
    loo_rows = []
    pair_correlations = []
    half_exact = []
    half_top3_overlap = []
    operational_stable = []
    for state_id, state_rows in sorted(grouped.items()):
        state_plan = plan_states[state_id]
        candidate_ids = {
            str(arm["candidate_id"]) for arm in state_plan["arms"]
        }
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
            raise ValueError(
                f"four-seed candidate coverage mismatch: {state_id}"
            )

        candidate_order = sorted(candidate_ids)
        correlations = []
        for left in range(4):
            left_values = {
                str(row["candidate_id"]): float(row["normalized_step_auc"])
                for row in state_rows
                if int(row["trial_index"]) == left
            }
            for right in range(left + 1, 4):
                right_values = {
                    str(row["candidate_id"]): float(
                        row["normalized_step_auc"]
                    )
                    for row in state_rows
                    if int(row["trial_index"]) == right
                }
                correlations.append(
                    _correlation(
                        [left_values[value] for value in candidate_order],
                        [right_values[value] for value in candidate_order],
                    )
                )
        mean_correlation = statistics.fmean(correlations)
        pair_correlations.append(mean_correlation)

        first = _candidate_aggregates(
            row for row in state_rows if int(row["trial_index"]) in {0, 1}
        )
        second = _candidate_aggregates(
            row for row in state_rows if int(row["trial_index"]) in {2, 3}
        )
        first_winner = str(first[0]["candidate_id"])
        second_winner = str(second[0]["candidate_id"])
        exact = first_winner == second_winner
        top3_overlap = len(
            {str(row["candidate_id"]) for row in first[:3]}
            & {str(row["candidate_id"]) for row in second[:3]}
        )
        half_exact.append(float(exact))
        half_top3_overlap.append(float(top3_overlap > 0))

        cross_regrets = []
        for selected_id, trials, oracle in (
            (first_winner, {2, 3}, second[0]),
            (second_winner, {0, 1}, first[0]),
        ):
            selected = _aggregate_candidate(
                _candidate_rows(state_rows, selected_id, trials)
            )
            cross_regrets.append(
                {
                    "feasible_rate_regret": float(oracle["feasible_rate"])
                    - float(selected["feasible_rate"]),
                    "final_conflict_ratio_regret": float(
                        selected["mean_final_conflict_ratio"]
                    )
                    - float(oracle["mean_final_conflict_ratio"]),
                    "normalized_step_auc_regret": float(
                        selected["mean_normalized_step_auc"]
                    )
                    - float(oracle["mean_normalized_step_auc"]),
                }
            )
        feasible_regret = statistics.fmean(
            row["feasible_rate_regret"] for row in cross_regrets
        )
        final_regret = statistics.fmean(
            row["final_conflict_ratio_regret"] for row in cross_regrets
        )
        auc_regret = statistics.fmean(
            row["normalized_step_auc_regret"] for row in cross_regrets
        )
        stable = (
            feasible_regret <= 0.0
            and final_regret <= 0.02
            and auc_regret <= 0.02
        )
        operational_stable.append(float(stable))

        all_aggregates = _candidate_aggregates(state_rows)
        expected_winner = all_aggregates[0]
        runner_up = all_aggregates[1]
        state_diagnostics.append(
            {
                "state_id": state_id,
                "map_id": str(state_rows[0]["map_id"]),
                "layout_mode": str(state_rows[0]["layout_mode"]),
                "agent_count": int(state_rows[0]["agent_count"]),
                "candidate_count": len(candidate_ids),
                "mean_pairwise_rank_correlation": mean_correlation,
                "half_winner_exact": exact,
                "half_top3_overlap": top3_overlap,
                "first_half_winner": first_winner,
                "second_half_winner": second_winner,
                "four_seed_expected_winner": str(
                    expected_winner["candidate_id"]
                ),
                "expected_winner_auc": float(
                    expected_winner["mean_normalized_step_auc"]
                ),
                "runner_up_auc": float(
                    runner_up["mean_normalized_step_auc"]
                ),
                "expected_winner_auc_margin": float(
                    runner_up["mean_normalized_step_auc"]
                )
                - float(expected_winner["mean_normalized_step_auc"]),
                "expected_winner_auc_std": float(
                    expected_winner["std_normalized_step_auc"]
                ),
                "cross_half_feasible_rate_regret": feasible_regret,
                "cross_half_final_conflict_ratio_regret": final_regret,
                "cross_half_normalized_step_auc_regret": auc_regret,
                "operationally_stable": stable,
            }
        )

        for heldout in range(4):
            training_trials = set(range(4)) - {heldout}
            training = _candidate_aggregates(
                row
                for row in state_rows
                if int(row["trial_index"]) in training_trials
            )
            selected_id = str(training[0]["candidate_id"])
            selected = _candidate_rows(
                state_rows, selected_id, {heldout}
            )[0]
            v2 = [
                row
                for row in state_rows
                if int(row["trial_index"]) == heldout
                and bool(row["is_v2_candidate"])
            ]
            if len(v2) != 1:
                raise ValueError("four-seed state lacks one held-out v2 row")
            oracle = min(
                (
                    row
                    for row in state_rows
                    if int(row["trial_index"]) == heldout
                ),
                key=_winner_key,
            )
            loo_rows.append(
                {
                    "state_id": state_id,
                    "map_id": str(selected["map_id"]),
                    "agent_count": int(selected["agent_count"]),
                    "heldout_trial": heldout,
                    "selected_candidate_id": selected_id,
                    "v2_candidate_id": str(v2[0]["candidate_id"]),
                    "oracle_candidate_id": str(oracle["candidate_id"]),
                    "selected_vs_v2_outcome": _outcome_score(
                        selected, v2[0]
                    ),
                    "selected_feasible": bool(selected["feasible"]),
                    "v2_feasible": bool(v2[0]["feasible"]),
                    "selected_minus_v2_final_conflict_ratio": float(
                        selected["final_conflict_ratio"]
                    )
                    - float(v2[0]["final_conflict_ratio"]),
                    "selected_minus_v2_normalized_step_auc": float(
                        selected["normalized_step_auc"]
                    )
                    - float(v2[0]["normalized_step_auc"]),
                    "selected_minus_v2_total_seconds": float(
                        selected["observed_total_seconds"]
                    )
                    - float(v2[0]["observed_total_seconds"]),
                    "oracle_final_conflict_ratio_regret": float(
                        selected["final_conflict_ratio"]
                    )
                    - float(oracle["final_conflict_ratio"]),
                    "oracle_normalized_step_auc_regret": float(
                        selected["normalized_step_auc"]
                    )
                    - float(oracle["normalized_step_auc"]),
                }
            )

    loo_wins = sum(
        int(row["selected_vs_v2_outcome"]) > 0 for row in loo_rows
    )
    loo_losses = sum(
        int(row["selected_vs_v2_outcome"]) < 0 for row in loo_rows
    )
    loo_ties = len(loo_rows) - loo_wins - loo_losses
    selected_feasible_rate = statistics.fmean(
        float(row["selected_feasible"]) for row in loo_rows
    )
    v2_feasible_rate = statistics.fmean(
        float(row["v2_feasible"]) for row in loo_rows
    )
    mean_auc_delta = statistics.fmean(
        float(row["selected_minus_v2_normalized_step_auc"])
        for row in loo_rows
    )
    mean_pair_correlation = statistics.fmean(pair_correlations)
    half_top3_fraction = statistics.fmean(half_top3_overlap)
    operational_fraction = statistics.fmean(operational_stable)
    checks = {
        "followup_coverage_complete": all(
            int(row["candidate_count"]) >= 2 for row in state_diagnostics
        ),
        "mean_pairwise_rank_correlation_at_least_50pct": (
            mean_pair_correlation >= 0.50
        ),
        "half_top3_overlap_on_at_least_80pct_states": (
            half_top3_fraction >= 0.80
        ),
        "operational_stability_on_at_least_70pct_states": (
            operational_fraction >= 0.70
        ),
        "loo_feasible_rate_not_below_v2": (
            selected_feasible_rate >= v2_feasible_rate
        ),
        "loo_normalized_step_auc_not_worse_than_v2_by_2pct": (
            mean_auc_delta <= 0.02
        ),
        "loo_wins_not_below_losses": loo_wins >= loo_losses,
    }
    decision = (
        "four_seed_receding_q_labels_promising_for_model_pilot"
        if all(checks.values())
        else "four_seed_receding_q_labels_insufficient"
    )
    report = {
        "schema": RECEDING_Q_STABILITY_SCHEMA,
        "decision": decision,
        "target_state_count": len(grouped),
        "followup_trial_indices": list(FOLLOWUP_TRIALS),
        "mean_pairwise_rank_correlation": mean_pair_correlation,
        "half_winner_exact_fraction": statistics.fmean(half_exact),
        "half_top3_overlap_fraction": half_top3_fraction,
        "operational_stability_fraction": operational_fraction,
        "mean_cross_half_final_conflict_ratio_regret": statistics.fmean(
            float(row["cross_half_final_conflict_ratio_regret"])
            for row in state_diagnostics
        ),
        "mean_cross_half_normalized_step_auc_regret": statistics.fmean(
            float(row["cross_half_normalized_step_auc_regret"])
            for row in state_diagnostics
        ),
        "loo": {
            "fold_count": len(loo_rows),
            "wins": loo_wins,
            "losses": loo_losses,
            "ties": loo_ties,
            "selected_feasible_rate": selected_feasible_rate,
            "v2_feasible_rate": v2_feasible_rate,
            "feasible_rate_delta": selected_feasible_rate - v2_feasible_rate,
            "mean_normalized_step_auc_delta": mean_auc_delta,
            "mean_final_conflict_ratio_delta": statistics.fmean(
                float(row["selected_minus_v2_final_conflict_ratio"])
                for row in loo_rows
            ),
            "mean_total_seconds_delta": statistics.fmean(
                float(row["selected_minus_v2_total_seconds"])
                for row in loo_rows
            ),
            "mean_oracle_normalized_step_auc_regret": statistics.fmean(
                float(row["oracle_normalized_step_auc_regret"])
                for row in loo_rows
            ),
        },
        "checks": checks,
        "limitations": [
            "Only the states whose trial-0 and trial-1 exact winners differed receive two additional PP seeds.",
            "Exact winner identity is diagnostic; the gate uses rank correlation, Top-3 overlap, cross-half outcome regret, and leave-one-seed-out behavior.",
            "The leave-one-seed-out policy uses measured outcomes from three seeds and is an optimistic label diagnostic, not a trained deployable controller.",
            "Continuation still uses the Adaptive teacher and the horizon remains three repairs; no complete episode is evaluated.",
        ],
    }
    return report, state_diagnostics, loo_rows


def _markdown(report: dict[str, Any]) -> str:
    loo = dict(report["loo"])
    return "\n".join(
        [
            "# Receding-Q four-seed stability follow-up",
            "",
            f"Decision: `{report['decision']}`",
            "",
            (
                f"- Target states: {int(report['target_state_count'])}; "
                f"additional trials: {report['followup_trial_indices']}."
            ),
            (
                "- Mean pairwise rank correlation: "
                f"{float(report['mean_pairwise_rank_correlation']):.4f}."
            ),
            (
                "- Half-split exact winner: "
                f"{float(report['half_winner_exact_fraction']):.3%}; "
                "Top-3 overlap: "
                f"{float(report['half_top3_overlap_fraction']):.3%}; "
                "operational stability: "
                f"{float(report['operational_stability_fraction']):.3%}."
            ),
            (
                "- LOO vs v2: "
                f"{int(loo['wins'])} wins / {int(loo['losses'])} losses / "
                f"{int(loo['ties'])} ties; feasible delta "
                f"{float(loo['feasible_rate_delta']):+.3%}; step-AUC delta "
                f"{float(loo['mean_normalized_step_auc_delta']):+.4f}."
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


def run_receding_q_stability_followup(
    *,
    source: str | Path,
    output: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not bool(resume):
        raise FileExistsError("stability output is non-empty; pass resume")
    output_root.mkdir(parents=True, exist_ok=True)

    source_status = dict(read_json(source_root / "status.json"))
    if (
        str(source_status.get("status")) != "complete"
        or int(source_status.get("error_count", -1)) != 0
    ):
        raise ValueError("source receding-Q pilot is not complete and clean")
    source_plan = dict(read_json(source_root / "plan.json"))
    source_config = dict(read_json(source_root / "run_config.json"))
    if str(source_plan.get("schema")) != RECEDING_Q_PILOT_SCHEMA:
        raise ValueError("source plan is not a receding-Q pilot")
    if int(source_config["trials"]) != 2:
        raise ValueError("stability source must contain exactly two trials")
    source_rows = load_receding_q_rollouts(source_root)
    targets = identify_stability_targets(source_rows)
    jobs = build_followup_jobs(
        plan=source_plan,
        targets=targets,
        horizon=int(source_config["horizon"]),
        continuation_teacher=str(source_config["continuation_teacher"]),
    )
    config = {
        "schema": RECEDING_Q_STABILITY_SCHEMA,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "source": str(source_root),
        "source_plan_fingerprint": _fingerprint(source_plan),
        "source_rollout_fingerprint": _fingerprint(
            sorted((_rollout_key(row), row) for row in source_rows)
        ),
        "target_fingerprint": _fingerprint(targets),
        "horizon": int(source_config["horizon"]),
        "continuation_teacher": str(source_config["continuation_teacher"]),
        "followup_trial_indices": list(FOLLOWUP_TRIALS),
        "worker_count": 1,
    }
    config_path = output_root / "run_config.json"
    target_path = output_root / "targets.json"
    if config_path.is_file():
        if dict(read_json(config_path)) != config:
            raise ValueError("stability resume configuration mismatch")
        if dict(read_json(target_path)) != targets:
            raise ValueError("stability resume targets mismatch")
    else:
        _write_json(config_path, config)
        _write_json(target_path, targets)

    rollout_root = output_root / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)
    completed = []
    errors = []
    started = time.perf_counter()
    for index, job in enumerate(jobs):
        state_id = str(job["state"]["state_id"])
        candidate_id = str(job["arm"]["candidate_id"])
        trial_index = int(job["trial_index"])
        path = rollout_root / _rollout_file_name(
            state_id, candidate_id, trial_index
        )
        try:
            if bool(resume) and path.is_file():
                row = dict(read_json(path))
                if (
                    _rollout_key(row)
                    == (state_id, candidate_id, trial_index)
                    and bool(row.get("complete"))
                ):
                    completed.append(row)
                    continue
            row = run_receding_q_rollout(job)
            partial = path.with_name(path.name + ".partial")
            _write_json(partial, row)
            os.replace(partial, path)
            completed.append(row)
        except Exception as error:
            errors.append(
                {
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "trial_index": trial_index,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            break
        finally:
            elapsed = max(1e-12, time.perf_counter() - started)
            _write_json(
                output_root / "status.json",
                {
                    "schema": RECEDING_Q_STABILITY_SCHEMA,
                    "status": "error" if errors else "running",
                    "completed_rollout_count": len(completed),
                    "total_rollout_count": len(jobs),
                    "error_count": len(errors),
                    "rollouts_per_minute": 60.0 * len(completed) / elapsed,
                    "last_job_index": index,
                },
            )
    if errors:
        _write_json(output_root / "errors.json", {"errors": errors})
        raise RuntimeError(errors[0]["error"])
    (output_root / "errors.json").unlink(missing_ok=True)

    merged = merge_followup_rollouts(source_rows, completed)
    report, state_rows, loo_rows = analyze_four_seed_stability(
        merged, plan=source_plan, targets=targets
    )
    report["run_config"] = config
    report["source_state_count"] = int(targets["source_state_count"])
    report["stable_source_state_count"] = int(targets["stable_state_count"])
    report["source_rollout_count"] = len(source_rows)
    report["followup_rollout_count"] = len(completed)
    report["merged_rollout_count"] = len(merged)
    _atomic_write_csv(
        output_root / "followup_rollouts.csv",
        [_rollout_flat(row) for row in completed],
    )
    _atomic_write_csv(
        output_root / "merged_rollouts.csv",
        [_rollout_flat(row) for row in merged],
    )
    _atomic_write_csv(
        output_root / "state_stability_diagnostics.csv", state_rows
    )
    _atomic_write_csv(
        output_root / "leave_one_seed_out.csv", loo_rows
    )
    _write_json(output_root / "receding_q_stability_report.json", report)
    (output_root / "receding_q_stability_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": RECEDING_Q_STABILITY_SCHEMA,
            "status": "complete",
            "completed_rollout_count": len(completed),
            "total_rollout_count": len(jobs),
            "error_count": 0,
            "decision": str(report["decision"]),
        },
    )
    return report


__all__ = [
    "FOLLOWUP_TRIALS",
    "RECEDING_Q_STABILITY_SCHEMA",
    "analyze_four_seed_stability",
    "build_followup_jobs",
    "identify_stability_targets",
    "load_receding_q_rollouts",
    "merge_followup_rollouts",
    "run_receding_q_stability_followup",
]
