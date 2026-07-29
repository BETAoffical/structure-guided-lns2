from __future__ import annotations

import collections
import csv
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.repair_collection import _read_json, _write_json
from experiments.run_output_guard import prepare_run_output


STALL_RESCUE_ORDER_AUDIT_SCHEMA = "lns2.stall_rescue_order_audit.v1"
NO_PROGRESS_OUTCOMES = {"accepted_noop", "hard_failure"}
RANK_POLICIES = {
    "rank2_only": (2,),
    "ranks2_3": (2, 3),
    "ranks2_3_4": (2, 3, 4),
    "ranks2_to_5": tuple(range(2, 6)),
    "ranks2_to_8": tuple(range(2, 9)),
    "ranks2_to_18": tuple(range(2, 19)),
}
RECOMMENDATION_ESCAPE_TOLERANCE = 0.02


def simulate_rank_policy(
    rows: Iterable[dict[str, Any]], ranks: tuple[int, ...]
) -> dict[str, Any]:
    by_rank = {int(row["candidate_rank"]): dict(row) for row in rows}
    attempted = 0
    repair_seconds = 0.0
    for rank in ranks:
        if rank not in by_rank:
            raise ValueError(f"rescue policy is missing candidate rank {rank}")
        row = by_rank[rank]
        attempted += 1
        repair_seconds += float(row["repair_wall_seconds"])
        outcome = str(row["repair_outcome"])
        if outcome not in NO_PROGRESS_OUTCOMES:
            return {
                "escaped": True,
                "attempt_count": attempted,
                "repair_seconds": repair_seconds,
                "escape_rank": rank,
                "repair_outcome": outcome,
                "conflict_delta": float(row["conflict_delta"]),
            }
    return {
        "escaped": False,
        "attempt_count": attempted,
        "repair_seconds": repair_seconds,
        "escape_rank": None,
        "repair_outcome": "no_escape",
        "conflict_delta": 0.0,
    }


def _number_summary(values: Iterable[float]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    p95_index = min(len(numbers) - 1, max(0, int(0.95 * len(numbers)) - 1))
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p95": numbers[p95_index],
    }


def recommend_rank_policy(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("rescue policy recommendation requires summaries")
    maximum_stable = max(
        float(row["stable_escape_state_fraction"]) for row in summaries
    )
    maximum_escape = max(float(row["escape_fraction"]) for row in summaries)
    eligible = [
        row
        for row in summaries
        if float(row["stable_escape_state_fraction"]) == maximum_stable
        and float(row["escape_fraction"])
        >= maximum_escape - RECOMMENDATION_ESCAPE_TOLERANCE
    ]
    return min(
        eligible,
        key=lambda row: (
            len(row["rank_sequence"]),
            float(row["repair_seconds"]["mean"]),
        ),
    )


def audit_stall_rescue_orders(
    oracle_batch: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    batch_root = Path(oracle_batch).resolve()
    output_root = Path(output).resolve()
    source_runner = batch_root / "runner_config.json"
    identity = {
        "schema": STALL_RESCUE_ORDER_AUDIT_SCHEMA,
        "oracle_batch_runner_sha256": sha256_file(source_runner),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "rank_policies": {name: list(ranks) for name, ranks in RANK_POLICIES.items()},
    }
    prepare_run_output(output_root, resume=resume, identity=identity)
    trial_rows: list[dict[str, Any]] = []
    selector_state_count = 0
    for job_root in sorted((batch_root / "jobs").iterdir()):
        audit_path = job_root / "audit" / "stall_oracle_report.json"
        if not audit_path.is_file():
            raise ValueError(f"oracle job audit is missing: {job_root.name}")
        audit = _read_json(audit_path)
        if str(audit.get("classification")) != "selector_failure":
            continue
        selector_state_count += 1
        probe = _read_json(job_root / "stalled_state_probe_report.json")
        with (job_root / "stalled_state_trials.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            source_rows = list(csv.DictReader(handle))
        explicit = [row for row in source_rows if row["branch_mode"] == "explicit_neighborhood"]
        by_trial: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in explicit:
            by_trial[int(row["trial_index"])].append(row)
        if len(by_trial) < 4:
            raise ValueError(f"oracle job has incomplete paired trials: {job_root.name}")
        for trial_index, paired in sorted(by_trial.items()):
            for policy_name, ranks in RANK_POLICIES.items():
                result = simulate_rank_policy(paired, ranks)
                trial_rows.append(
                    {
                        "job_id": job_root.name,
                        "task_id": probe.get("task_id"),
                        "trial_index": trial_index,
                        "policy": policy_name,
                        **result,
                    }
                )
    if not selector_state_count:
        raise ValueError("oracle batch contains no selector failures")
    summaries: list[dict[str, Any]] = []
    for policy_name in RANK_POLICIES:
        rows = [row for row in trial_rows if row["policy"] == policy_name]
        per_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            per_state[str(row["job_id"])].append(row)
        stable_states = sum(
            statistics.fmean(bool(row["escaped"]) for row in state_rows) >= 0.75
            for state_rows in per_state.values()
        )
        summaries.append(
            {
                "policy": policy_name,
                "rank_sequence": list(RANK_POLICIES[policy_name]),
                "trial_count": len(rows),
                "escape_count": sum(bool(row["escaped"]) for row in rows),
                "escape_fraction": statistics.fmean(
                    bool(row["escaped"]) for row in rows
                ),
                "stable_escape_state_count": stable_states,
                "stable_escape_state_fraction": stable_states / len(per_state),
                "mean_attempt_count": statistics.fmean(
                    int(row["attempt_count"]) for row in rows
                ),
                "repair_seconds": _number_summary(
                    float(row["repair_seconds"]) for row in rows
                ),
                "mean_conflict_delta_on_escape": statistics.fmean(
                    float(row["conflict_delta"])
                    for row in rows
                    if bool(row["escaped"])
                )
                if any(bool(row["escaped"]) for row in rows)
                else 0.0,
            }
        )
    best = max(
        summaries,
        key=lambda row: (
            float(row["stable_escape_state_fraction"]),
            float(row["escape_fraction"]),
            -float(row["repair_seconds"]["mean"]),
        ),
    )
    recommended = recommend_rank_policy(summaries)
    report = {
        "schema": STALL_RESCUE_ORDER_AUDIT_SCHEMA,
        "complete": True,
        "evidence_level": "exact same-state paired PP offline audit",
        "selector_state_count": selector_state_count,
        "trial_count": len(trial_rows),
        "policies": summaries,
        "best_policy": best["policy"],
        "best_stable_escape_state_fraction": best["stable_escape_state_fraction"],
        "recommended_policy": recommended["policy"],
        "recommendation_escape_tolerance": RECOMMENDATION_ESCAPE_TOLERANCE,
        "recommendation_reason": (
            "shortest sequence with maximum stable-state coverage and trial "
            "escape within two percentage points of the maximum"
        ),
        "training_started": False,
        "controller_actions_changed": False,
        "deployment_promoted": False,
    }
    atomic_write_csv(output_root / "rescue_order_trials.csv", trial_rows)
    atomic_write_csv(output_root / "rescue_order_summary.csv", summaries)
    _write_json(output_root / "rescue_order_audit_report.json", report)
    return report


__all__ = [
    "RANK_POLICIES",
    "audit_stall_rescue_orders",
    "recommend_rank_policy",
    "simulate_rank_policy",
]
