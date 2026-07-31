from __future__ import annotations

import collections
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    config_producer_fingerprint as _config_producer_fingerprint,
    producer_identity,
    read_json,
    sha256_file,
)
from experiments.repair_collection import _fingerprint, _write_json
from experiments.v3_value_pilot import (
    DEFAULT_OVERHEAD_GRID,
    PROJECT_ROOT,
    V3_VALUE_PILOT_SCHEMA,
    V3_VALUE_PILOT_PRODUCER_FILES,
    _atomic_write_csv,
    _rollout_file_name,
    _rollout_flat,
    _winner_key,
    analyze_value_rollouts,
    load_resumable_value_rollout,
    run_value_rollout,
    validate_value_rollout,
)


V3_VALUE_STABILITY_SCHEMA = "lns2.v3_value_label_stability.v2"
V3_VALUE_STABILITY_PRODUCER_FILES = (
    *V3_VALUE_PILOT_PRODUCER_FILES,
    "experiments/v3_value_stability.py",
)


def _portable_report_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result["source_pilot"] = Path(str(result["source_pilot"])).name
    return result


def _rollout_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["state_id"]),
        str(row["arm_id"]),
        int(row["trial_index"]),
    )


def _load_rollouts(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "rollouts").glob("*.json")):
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"value rollout is not an object: {path.name}")
        rows.append(dict(value))
    if not rows:
        raise ValueError("source value pilot has no rollout files")
    keys = [_rollout_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("source value pilot contains duplicate rollout keys")
    if not all(bool(row.get("complete")) for row in rows):
        raise ValueError("source value pilot contains incomplete rollouts")
    return rows


def _validate_value_matrix(
    rows: Iterable[dict[str, Any]],
    *,
    plan: dict[str, Any],
    expected_keys: set[tuple[str, str, int]],
    max_repairs: int,
    wall_clock_seconds: float,
    producer_fingerprint: str,
    followup_reasons: dict[tuple[str, str, int], str] | None = None,
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    observed = {_rollout_key(row) for row in materialized}
    if len(observed) != len(materialized) or observed != expected_keys:
        raise ValueError("value rollout matrix coverage mismatch")
    states = {str(state["state_id"]): dict(state) for state in plan["states"]}
    arms = {
        (state_id, str(arm["arm_id"])): dict(arm)
        for state_id, state in states.items()
        for arm in state["arms"]
    }
    for row in materialized:
        key = _rollout_key(row)
        state_id, arm_id, trial_index = key
        if state_id not in states or (state_id, arm_id) not in arms:
            raise ValueError("value rollout is outside its plan")
        validate_value_rollout(
            row,
            state_plan=states[state_id],
            arm_plan=arms[(state_id, arm_id)],
            max_repairs=int(max_repairs),
            wall_clock_seconds=float(wall_clock_seconds),
            expected_trial_index=trial_index,
            expected_producer_fingerprint=producer_fingerprint,
        )
        if followup_reasons is not None and str(row.get("followup_reason")) != str(
            followup_reasons[key]
        ):
            raise ValueError("value rollout follow-up reason mismatch")
    return materialized


def _load_validated_source_pilot(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    status_value = read_json(root / "status.json")
    config_value = read_json(root / "run_config.json")
    plan_value = read_json(root / "plan.json")
    if not all(
        isinstance(value, dict)
        for value in (status_value, config_value, plan_value)
    ):
        raise ValueError("source value pilot metadata is not an object")
    status = dict(status_value)
    config = dict(config_value)
    plan = dict(plan_value)
    if str(status.get("schema")) != V3_VALUE_PILOT_SCHEMA:
        raise ValueError("source value pilot status schema mismatch")
    if (
        str(status.get("status")) != "complete"
        or int(status.get("error_count", -1)) != 0
    ):
        raise ValueError("source value pilot is not complete and clean")
    if str(config.get("schema")) != V3_VALUE_PILOT_SCHEMA:
        raise ValueError("source value pilot configuration schema mismatch")
    if str(plan.get("schema")) != V3_VALUE_PILOT_SCHEMA:
        raise ValueError("source value pilot plan schema mismatch")
    if str(config.get("plan_fingerprint")) != _fingerprint(plan):
        raise ValueError("source value pilot plan fingerprint mismatch")
    producer_fingerprint = _config_producer_fingerprint(
        config, label="source value pilot"
    )
    trials = int(config.get("trials", -1))
    if trials <= 0:
        raise ValueError("source value pilot trial count is invalid")
    expected_keys = {
        (str(state["state_id"]), str(arm["arm_id"]), trial)
        for state in plan["states"]
        for arm in state["arms"]
        for trial in range(trials)
    }
    rows = _validate_value_matrix(
        _load_rollouts(root),
        plan=plan,
        expected_keys=expected_keys,
        max_repairs=int(config["max_repairs"]),
        wall_clock_seconds=float(config["wall_clock_seconds"]),
        producer_fingerprint=producer_fingerprint,
    )
    if (
        int(status.get("completed_rollout_count", -1)) != len(rows)
        or int(status.get("total_rollout_count", -1)) != len(rows)
    ):
        raise ValueError("source value pilot status count mismatch")
    return plan, config, rows


def trial_winners(
    rows: Iterable[dict[str, Any]], *, overhead: float
) -> dict[tuple[str, int], str]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        grouped[(str(row["state_id"]), int(row["trial_index"]))].append(row)
    return {
        key: str(min(values, key=lambda row: _winner_key(row, overhead))["arm_id"])
        for key, values in grouped.items()
    }


def identify_stability_targets(
    rows: Iterable[dict[str, Any]], *, overhead: float = 0.0
) -> dict[str, Any]:
    materialized = list(rows)
    winners = trial_winners(materialized, overhead=float(overhead))
    by_state: dict[str, list[str]] = collections.defaultdict(list)
    for (state_id, _trial), arm_id in winners.items():
        by_state[state_id].append(arm_id)
    unstable = sorted(
        state_id
        for state_id, arm_ids in by_state.items()
        if len(set(arm_ids)) > 1
    )
    censored_keys = sorted(
        _rollout_key(row) for row in materialized if bool(row["censored"])
    )
    censored_states = sorted({key[0] for key in censored_keys})
    return {
        "unstable_state_ids": unstable,
        "censored_rollout_keys": censored_keys,
        "censored_state_ids": censored_states,
        "target_state_ids": sorted(set(unstable) | set(censored_states)),
    }


def build_stability_jobs(
    *,
    plan: dict[str, Any],
    source_rows: Iterable[dict[str, Any]],
    total_trials: int,
    max_repairs: int,
    wall_clock_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(source_rows)
    existing_trials = sorted({int(row["trial_index"]) for row in rows})
    if existing_trials != list(range(len(existing_trials))):
        raise ValueError("source trial indices must be contiguous from zero")
    if int(total_trials) <= len(existing_trials):
        raise ValueError("total_trials must exceed the source trial count")
    targets = identify_stability_targets(rows)
    state_lookup = {
        str(state["state_id"]): dict(state) for state in plan["states"]
    }
    missing = sorted(set(targets["target_state_ids"]) - set(state_lookup))
    if missing:
        raise ValueError(f"target states missing from source plan: {missing[0]}")

    censored_keys = {
        (str(state_id), str(arm_id), int(trial_index))
        for state_id, arm_id, trial_index in targets["censored_rollout_keys"]
    }
    jobs = []
    for state_id, arm_id, trial_index in sorted(censored_keys):
        state = state_lookup[state_id]
        arm_lookup = {
            str(arm["arm_id"]): dict(arm) for arm in state["arms"]
        }
        if arm_id not in arm_lookup:
            raise ValueError(f"censored arm missing from plan: {arm_id}")
        jobs.append(
            {
                "reason": "extend_censored",
                "state": state,
                "arm": arm_lookup[arm_id],
                "trial_index": trial_index,
                "max_repairs": int(max_repairs),
                "wall_clock_seconds": float(wall_clock_seconds),
            }
        )
    for state_id in targets["target_state_ids"]:
        state = state_lookup[state_id]
        for arm in state["arms"]:
            for trial_index in range(len(existing_trials), int(total_trials)):
                jobs.append(
                    {
                        "reason": "new_seed",
                        "state": state,
                        "arm": dict(arm),
                        "trial_index": trial_index,
                        "max_repairs": int(max_repairs),
                        "wall_clock_seconds": float(wall_clock_seconds),
                    }
                )
    job_keys = [
        (
            str(job["state"]["state_id"]),
            str(job["arm"]["arm_id"]),
            int(job["trial_index"]),
        )
        for job in jobs
    ]
    if len(job_keys) != len(set(job_keys)):
        raise ValueError("stability follow-up generated duplicate jobs")
    return targets, jobs


def merge_stability_rollouts(
    source_rows: Iterable[dict[str, Any]],
    followup_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {_rollout_key(row): dict(row) for row in source_rows}
    for row in followup_rows:
        key = _rollout_key(row)
        reason = str(row.get("followup_reason", ""))
        if key in merged and reason != "extend_censored":
            raise ValueError(f"unexpected replacement rollout: {key}")
        if key in merged and not bool(merged[key]["censored"]):
            raise ValueError(f"refusing to replace uncensored rollout: {key}")
        merged[key] = dict(row)
    return [merged[key] for key in sorted(merged)]


def _stability_rows(
    rows: list[dict[str, Any]],
    *,
    target_state_ids: set[str],
    overhead: float,
) -> list[dict[str, Any]]:
    winners = trial_winners(rows, overhead=float(overhead))
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for (state_id, _trial), arm_id in winners.items():
        grouped[state_id].append(arm_id)
    result = []
    for state_id, arm_ids in sorted(grouped.items()):
        counts = collections.Counter(arm_ids)
        winner_count = max(counts.values())
        pair_count = math.comb(len(arm_ids), 2)
        matching_pairs = sum(
            math.comb(count, 2) for count in counts.values()
        )
        plurality = sorted(
            arm_id for arm_id, count in counts.items() if count == winner_count
        )
        result.append(
            {
                "state_id": state_id,
                "targeted": state_id in target_state_ids,
                "trial_count": len(arm_ids),
                "strict_winner_agreement": len(counts) == 1,
                "winner_purity": winner_count / len(arm_ids),
                "pairwise_winner_agreement": (
                    matching_pairs / pair_count if pair_count else 1.0
                ),
                "plurality_tied": len(plurality) > 1,
                "plurality_winner": ",".join(plurality),
                "winner_counts": ",".join(
                    f"{arm_id}:{counts[arm_id]}" for arm_id in sorted(counts)
                ),
            }
        )
    return result


def _arm_summary(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        grouped[(str(row["state_id"]), str(row["arm_id"]))].append(row)
    result = []
    for (state_id, arm_id), values in sorted(grouped.items()):
        result.append(
            {
                "state_id": state_id,
                "arm_id": arm_id,
                "trial_count": len(values),
                "feasible_rate": statistics.fmean(
                    float(bool(row["feasible"])) for row in values
                ),
                "censored_count": sum(
                    bool(row["censored"]) for row in values
                ),
                "median_total_seconds": statistics.median(
                    float(row["observed_total_seconds"]) for row in values
                ),
                "median_repair_iterations": statistics.median(
                    int(row["repair_iterations"]) for row in values
                ),
                "median_final_conflicts": statistics.median(
                    int(row["final_conflicts"]) for row in values
                ),
                "median_normalized_auc_seconds": statistics.median(
                    float(row["normalized_conflict_auc_seconds"])
                    for row in values
                ),
            }
        )
    return result


def analyze_stability_followup(
    *,
    source_rows: list[dict[str, Any]],
    followup_rows: list[dict[str, Any]],
    target_state_ids: Iterable[str],
    total_trials: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    target_set = set(map(str, target_state_ids))
    merged = merge_stability_rollouts(source_rows, followup_rows)
    expected_trials = {
        state_id: int(total_trials) if state_id in target_set else 2
        for state_id in {str(row["state_id"]) for row in merged}
    }
    grouped_keys: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    for row in merged:
        grouped_keys[(str(row["state_id"]), str(row["arm_id"]))].add(
            int(row["trial_index"])
        )
    trial_coverage_complete = all(
        trials == set(range(expected_trials[state_id]))
        for (state_id, _arm_id), trials in grouped_keys.items()
    )
    merged_report, _state_diagnostics, _sensitivity = analyze_value_rollouts(
        merged,
        expected_jobs=len(merged),
        smoke_only=False,
    )
    original_uncensored = statistics.fmean(
        float(not bool(row["censored"])) for row in source_rows
    )
    merged_uncensored = statistics.fmean(
        float(not bool(row["censored"])) for row in merged
    )
    extended = [
        row
        for row in followup_rows
        if str(row.get("followup_reason")) == "extend_censored"
    ]
    stability_by_overhead = {
        str(value): _stability_rows(
            merged,
            target_state_ids=target_set,
            overhead=float(value),
        )
        for value in DEFAULT_OVERHEAD_GRID
    }
    agreement = {
        key: statistics.fmean(
            float(row["strict_winner_agreement"]) for row in values
        )
        for key, values in stability_by_overhead.items()
    }
    pairwise_agreement = {
        key: statistics.fmean(
            float(row["pairwise_winner_agreement"]) for row in values
        )
        for key, values in stability_by_overhead.items()
    }
    target_stability = stability_by_overhead[str(float(0.0))]
    targeted_rows = [row for row in target_stability if bool(row["targeted"])]
    target_pairwise_agreement = statistics.fmean(
        float(row["pairwise_winner_agreement"]) for row in targeted_rows
    )
    minimum_target_purity = min(
        float(row["winner_purity"]) for row in targeted_rows
    )
    median_target_purity = statistics.median(
        float(row["winner_purity"]) for row in targeted_rows
    )
    checks = {
        "followup_coverage_complete": trial_coverage_complete,
        "all_censored_rollouts_extended": len(extended)
        == sum(bool(row["censored"]) for row in source_rows),
        "merged_uncensored_fraction_at_least_90pct": merged_uncensored >= 0.90,
        "target_pairwise_winner_agreement_at_least_60pct": (
            target_pairwise_agreement >= 0.60
        ),
        "target_winner_purity_at_least_50pct": minimum_target_purity >= 0.50,
    }
    decision = (
        "label_stability_sufficient_for_value_model_pilot"
        if all(checks.values())
        else "label_stability_insufficient_for_value_model_pilot"
    )
    report = {
        "schema": V3_VALUE_STABILITY_SCHEMA,
        "decision": decision,
        "source_rollout_count": len(source_rows),
        "followup_rollout_count": len(followup_rows),
        "merged_rollout_count": len(merged),
        "target_state_count": len(target_set),
        "extended_censored_rollout_count": len(extended),
        "resolved_censored_rollout_count": sum(
            not bool(row["censored"]) for row in extended
        ),
        "remaining_extended_censored_rollout_count": sum(
            bool(row["censored"]) for row in extended
        ),
        "original_uncensored_fraction": original_uncensored,
        "merged_uncensored_fraction": merged_uncensored,
        "strict_winner_agreement_by_overhead": agreement,
        "pairwise_winner_agreement_by_overhead": pairwise_agreement,
        "target_pairwise_winner_agreement": target_pairwise_agreement,
        "minimum_target_winner_purity": minimum_target_purity,
        "median_target_winner_purity": median_target_purity,
        "checks": checks,
        "upstream_value_report": merged_report,
        "limitations": [
            "Only previously unstable or censored states receive two additional PP seeds; stable states retain two seeds.",
            "Previously censored branches are extended to the new cap while completed branches reuse their original measurements.",
            "Continuation actions still use official Adaptive, so the labels estimate a teacher-conditioned Q rather than an on-policy v3 value.",
            "Strict all-seed agreement is diagnostic only because its probability decreases mechanically as the seed count grows; the promotion gate uses pairwise agreement.",
            "This is a label-stability gate and does not compare complete v2 or v3 episodes.",
        ],
    }
    return report, stability_by_overhead[str(float(0.0))], _arm_summary(merged)


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Variable-horizon value-label stability follow-up",
            "",
            f"Decision: `{report['decision']}`",
            "",
            (
                f"- Targeted states: {int(report['target_state_count'])}; "
                f"follow-up rollouts: {int(report['followup_rollout_count'])}."
            ),
            (
                "- Uncensored fraction: "
                f"{float(report['original_uncensored_fraction']):.3%} -> "
                f"{float(report['merged_uncensored_fraction']):.3%}."
            ),
            (
                "- Extended censored branches resolved: "
                f"{int(report['resolved_censored_rollout_count'])}/"
                f"{int(report['extended_censored_rollout_count'])}."
            ),
            (
                "- Minimum pairwise winner agreement across overheads: "
                f"{min(report['pairwise_winner_agreement_by_overhead'].values()):.3%}."
            ),
            (
                "- Pairwise winner agreement on targeted states: "
                f"{float(report['target_pairwise_winner_agreement']):.3%}."
            ),
            (
                "- Target winner purity, median/minimum: "
                f"{float(report['median_target_winner_purity']):.3%}/"
                f"{float(report['minimum_target_winner_purity']):.3%}."
            ),
            "",
            "## Checks",
            "",
            *[
                f"- {name}: `{str(bool(value)).lower()}`"
                for name, value in sorted(report["checks"].items())
            ],
            "",
            "## Boundary",
            "",
            *[f"- {value}" for value in report["limitations"]],
            "",
        ]
    )


def run_stability_followup(
    *,
    pilot: str | Path,
    output: str | Path,
    total_trials: int = 4,
    max_repairs: int = 60,
    wall_clock_seconds: float = 120.0,
    resume: bool = False,
) -> dict[str, Any]:
    if int(total_trials) <= 0 or int(max_repairs) <= 0:
        raise ValueError("total_trials and max_repairs must be positive")
    if not math.isfinite(float(wall_clock_seconds)) or float(
        wall_clock_seconds
    ) <= 0.0:
        raise ValueError("wall_clock_seconds must be finite and positive")
    pilot_root = Path(pilot).resolve()
    output_root = Path(output).resolve()
    if not (pilot_root / "status.json").is_file():
        raise FileNotFoundError("source pilot status.json is missing")
    source_plan, source_config, source_rows = _load_validated_source_pilot(
        pilot_root
    )
    targets, jobs = build_stability_jobs(
        plan=source_plan,
        source_rows=source_rows,
        total_trials=int(total_trials),
        max_repairs=int(max_repairs),
        wall_clock_seconds=float(wall_clock_seconds),
    )
    identity = producer_identity(
        project_root=PROJECT_ROOT,
        source_files=V3_VALUE_STABILITY_PRODUCER_FILES,
        native_required=True,
        package_names=("numpy", "scikit-learn"),
    )
    identity_fingerprint = _fingerprint(identity)
    for job in jobs:
        job["producer_identity_fingerprint"] = identity_fingerprint
    config = {
        "schema": V3_VALUE_STABILITY_SCHEMA,
        "producer_identity": identity,
        "producer_identity_fingerprint": identity_fingerprint,
        "source_pilot": str(pilot_root),
        "source_producer_identity_fingerprint": str(
            source_config["producer_identity_fingerprint"]
        ),
        "source_plan_fingerprint": _fingerprint(source_plan),
        "source_rollout_fingerprint": _fingerprint(
            sorted((_rollout_key(row), row) for row in source_rows)
        ),
        "total_trials": int(total_trials),
        "max_repairs": int(max_repairs),
        "wall_clock_seconds": float(wall_clock_seconds),
        "target_fingerprint": _fingerprint(targets),
    }
    output_has_files = output_root.exists() and any(output_root.iterdir())
    if output_has_files and not bool(resume):
        raise FileExistsError("stability output is non-empty; pass resume")
    config_path = output_root / "run_config.json"
    target_path = output_root / "followup_plan.json"
    if output_has_files and not (config_path.is_file() and target_path.is_file()):
        raise ValueError(
            "stability output lacks resumable config/targets; use a new output"
        )
    if output_has_files:
        if dict(read_json(config_path)) != config:
            raise ValueError("stability resume configuration mismatch")
        if dict(read_json(target_path)) != targets:
            raise ValueError("stability resume targets mismatch")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(config_path, config)
        _write_json(target_path, targets)

    rollout_root = output_root / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)
    completed = []
    errors = []
    for index, job in enumerate(jobs):
        state_id = str(job["state"]["state_id"])
        arm_id = str(job["arm"]["arm_id"])
        trial_index = int(job["trial_index"])
        path = rollout_root / _rollout_file_name(
            state_id, arm_id, trial_index
        )
        try:
            if bool(resume) and path.is_file():
                row = load_resumable_value_rollout(
                    path,
                    state_plan=dict(job["state"]),
                    arm_plan=dict(job["arm"]),
                    max_repairs=int(max_repairs),
                    wall_clock_seconds=float(wall_clock_seconds),
                    expected_trial_index=trial_index,
                    expected_producer_fingerprint=identity_fingerprint,
                )
                if row is not None:
                    if str(row.get("followup_reason")) != str(job["reason"]):
                        raise ValueError(
                            "stability rollout follow-up reason mismatch"
                        )
                    completed.append(row)
                    continue
            row = run_value_rollout(job)
            row["followup_reason"] = str(job["reason"])
            partial = path.with_name(path.name + ".partial")
            _write_json(partial, row)
            os.replace(partial, path)
            completed.append(row)
        except Exception as error:
            errors.append(
                {
                    "state_id": state_id,
                    "arm_id": arm_id,
                    "trial_index": trial_index,
                    "reason": str(job["reason"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            break
        finally:
            _write_json(
                output_root / "status.json",
                {
                    "schema": V3_VALUE_STABILITY_SCHEMA,
                    "status": "error" if errors else "running",
                    "completed_rollout_count": len(completed),
                    "total_rollout_count": len(jobs),
                    "error_count": len(errors),
                    "last_job_index": index,
                },
            )
    if errors:
        _write_json(output_root / "errors.json", {"errors": errors})
        raise RuntimeError(errors[0]["error"])
    (output_root / "errors.json").unlink(missing_ok=True)
    expected_followups = {
        (
            str(job["state"]["state_id"]),
            str(job["arm"]["arm_id"]),
            int(job["trial_index"]),
        ): str(job["reason"])
        for job in jobs
    }
    completed = _validate_value_matrix(
        completed,
        plan=source_plan,
        expected_keys=set(expected_followups),
        max_repairs=int(max_repairs),
        wall_clock_seconds=float(wall_clock_seconds),
        producer_fingerprint=identity_fingerprint,
        followup_reasons=expected_followups,
    )
    report, state_rows, arm_rows = analyze_stability_followup(
        source_rows=source_rows,
        followup_rows=completed,
        target_state_ids=targets["target_state_ids"],
        total_trials=int(total_trials),
    )
    report["run_config"] = _portable_report_config(config)
    report["analysis_implementation_sha256"] = sha256_file(
        Path(__file__).resolve()
    )
    report["targets"] = targets
    merged = merge_stability_rollouts(source_rows, completed)
    _atomic_write_csv(
        output_root / "merged_value_rollouts.csv",
        [_rollout_flat(row) for row in merged],
    )
    _atomic_write_csv(output_root / "state_stability.csv", state_rows)
    _atomic_write_csv(output_root / "arm_stability_summary.csv", arm_rows)
    _write_json(output_root / "value_stability_report.json", report)
    (output_root / "value_stability_report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_VALUE_STABILITY_SCHEMA,
            "status": "complete",
            "completed_rollout_count": len(completed),
            "total_rollout_count": len(jobs),
            "error_count": 0,
            "decision": str(report["decision"]),
        },
    )
    return report


def load_validated_value_stability(
    *, pilot: str | Path, output: str | Path
) -> dict[str, Any]:
    pilot_root = Path(pilot).resolve()
    output_root = Path(output).resolve()
    source_plan, source_config, source_rows = _load_validated_source_pilot(
        pilot_root
    )
    config_value = read_json(output_root / "run_config.json")
    targets_value = read_json(output_root / "followup_plan.json")
    status_value = read_json(output_root / "status.json")
    if not all(
        isinstance(value, dict)
        for value in (config_value, targets_value, status_value)
    ):
        raise ValueError("value stability metadata is not an object")
    config = dict(config_value)
    targets = dict(targets_value)
    status = dict(status_value)
    if str(config.get("schema")) != V3_VALUE_STABILITY_SCHEMA:
        raise ValueError("value stability configuration schema mismatch")
    if str(status.get("schema")) != V3_VALUE_STABILITY_SCHEMA:
        raise ValueError("value stability status schema mismatch")
    if (
        str(status.get("status")) != "complete"
        or int(status.get("error_count", -1)) != 0
    ):
        raise ValueError("value stability is not complete and clean")
    if Path(str(config.get("source_pilot", ""))).resolve() != pilot_root:
        raise ValueError("value stability source path mismatch")
    if str(config.get("source_plan_fingerprint")) != _fingerprint(source_plan):
        raise ValueError("value stability source plan fingerprint mismatch")
    if str(config.get("source_producer_identity_fingerprint")) != str(
        source_config["producer_identity_fingerprint"]
    ):
        raise ValueError("value stability source producer mismatch")
    if str(config.get("source_rollout_fingerprint")) != _fingerprint(
        sorted((_rollout_key(row), row) for row in source_rows)
    ):
        raise ValueError("value stability source rollout fingerprint mismatch")
    producer_fingerprint = _config_producer_fingerprint(
        config, label="value stability"
    )
    recomputed_targets, jobs = build_stability_jobs(
        plan=source_plan,
        source_rows=source_rows,
        total_trials=int(config["total_trials"]),
        max_repairs=int(config["max_repairs"]),
        wall_clock_seconds=float(config["wall_clock_seconds"]),
    )
    if (
        _fingerprint(targets) != _fingerprint(recomputed_targets)
        or str(config.get("target_fingerprint")) != _fingerprint(targets)
    ):
        raise ValueError("value stability targets mismatch")
    expected_reasons = {
        (
            str(job["state"]["state_id"]),
            str(job["arm"]["arm_id"]),
            int(job["trial_index"]),
        ): str(job["reason"])
        for job in jobs
    }
    followup_rows = _validate_value_matrix(
        _load_rollouts(output_root),
        plan=source_plan,
        expected_keys=set(expected_reasons),
        max_repairs=int(config["max_repairs"]),
        wall_clock_seconds=float(config["wall_clock_seconds"]),
        producer_fingerprint=producer_fingerprint,
        followup_reasons=expected_reasons,
    )
    if (
        int(status.get("completed_rollout_count", -1)) != len(followup_rows)
        or int(status.get("total_rollout_count", -1)) != len(followup_rows)
    ):
        raise ValueError("value stability status count mismatch")
    return {
        "source_plan": source_plan,
        "source_config": source_config,
        "source_rows": source_rows,
        "config": config,
        "targets": targets,
        "followup_rows": followup_rows,
    }


def reanalyze_stability_followup(
    *, pilot: str | Path, output: str | Path
) -> dict[str, Any]:
    output_root = Path(output).resolve()
    validated = load_validated_value_stability(pilot=pilot, output=output)
    source_rows = list(validated["source_rows"])
    followup_rows = list(validated["followup_rows"])
    config = dict(validated["config"])
    targets = dict(validated["targets"])
    report, state_rows, arm_rows = analyze_stability_followup(
        source_rows=source_rows,
        followup_rows=followup_rows,
        target_state_ids=targets["target_state_ids"],
        total_trials=int(config["total_trials"]),
    )
    report["run_config"] = _portable_report_config(config)
    report["analysis_implementation_sha256"] = sha256_file(
        Path(__file__).resolve()
    )
    report["targets"] = targets
    merged = merge_stability_rollouts(source_rows, followup_rows)
    _atomic_write_csv(
        output_root / "merged_value_rollouts.csv",
        [_rollout_flat(row) for row in merged],
    )
    _atomic_write_csv(output_root / "state_stability.csv", state_rows)
    _atomic_write_csv(output_root / "arm_stability_summary.csv", arm_rows)
    _write_json(output_root / "value_stability_report.json", report)
    (output_root / "value_stability_report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_VALUE_STABILITY_SCHEMA,
            "status": "complete",
            "completed_rollout_count": len(followup_rows),
            "total_rollout_count": len(followup_rows),
            "error_count": 0,
            "decision": str(report["decision"]),
            "reanalyzed": True,
        },
    )
    return report
