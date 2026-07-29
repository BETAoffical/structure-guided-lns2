from __future__ import annotations

import collections
import math
import statistics
from pathlib import Path
from typing import Any

from experiments._common import atomic_write_csv, read_json, read_jsonl, sha256_file, write_json
from experiments.lns2_bottleneck import validate_manifest_trace
from experiments.run_output_guard import prepare_run_output
from experiments.stall_escape_qualification import normalize_episode_decisions
from experiments.stall_trigger_policy_audit import wilson_upper
from experiments.trace_replay import decision_rows


STALL_CONFIRMATION_RULE_AUDIT_SCHEMA = "lns2.stall_confirmation_rule_audit.v1"
ATTEMPT_THRESHOLDS = (2, 3, 4, 6, 9, 12, 15)
MINIMUM_DISTINCT_ATTEMPTS = (2, 3)
FUTURE_HORIZONS = (3, 6)


def _repair_seconds(row: dict[str, Any]) -> float:
    return float(dict(dict(row["actual_lns2"])["outcome"])["repair_seconds"])


def evaluate_no_progress_run(
    run: list[dict[str, Any]],
    recovery: dict[str, Any] | None,
    *,
    threshold: int,
    minimum_distinct_attempts: int,
    future_horizon: int,
) -> dict[str, Any] | None:
    if threshold <= 0 or minimum_distinct_attempts <= 0 or future_horizon <= 0:
        raise ValueError("stall confirmation rule values must be positive")
    if len(run) < threshold:
        return None
    prior = run[:threshold]
    distinct = len({str(row["attempt_key"]) for row in prior})
    if distinct < minimum_distinct_attempts:
        return None
    after_trigger_failures = len(run) - threshold
    if recovery is not None:
        recovery_delay = after_trigger_failures + 1
        resolution = (
            "premature_v2_self_recovery"
            if recovery_delay <= future_horizon
            else "confirmed_no_change_window"
        )
        observable = min(future_horizon, recovery_delay)
    elif after_trigger_failures >= future_horizon:
        recovery_delay = None
        resolution = "confirmed_no_change_window"
        observable = future_horizon
    else:
        recovery_delay = None
        resolution = "right_censored_unresolved"
        observable = after_trigger_failures
    return {
        "threshold": threshold,
        "minimum_distinct_attempts": minimum_distinct_attempts,
        "future_horizon": future_horizon,
        "run_length": len(run),
        "distinct_attempt_count_at_trigger": distinct,
        "post_trigger_observed_decisions": observable,
        "recovery_delay_decisions": recovery_delay,
        "resolution": resolution,
        "pretrigger_repair_seconds": math.fsum(_repair_seconds(row) for row in prior),
    }


def _runs(decisions: list[dict[str, Any]]) -> list[tuple[list[dict[str, Any]], dict[str, Any] | None]]:
    result = []
    index = 0
    while index < len(decisions):
        if not bool(decisions[index]["no_progress"]):
            index += 1
            continue
        fingerprint = str(decisions[index]["before_repair_fingerprint"])
        start = index
        while (
            index < len(decisions)
            and bool(decisions[index]["no_progress"])
            and str(decisions[index]["before_repair_fingerprint"]) == fingerprint
        ):
            index += 1
        run = decisions[start:index]
        recovery = (
            decisions[index]
            if index < len(decisions)
            and not bool(decisions[index]["no_progress"])
            and str(decisions[index]["before_repair_fingerprint"]) == fingerprint
            else None
        )
        result.append((run, recovery))
    return result


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["threshold"]),
                int(row["minimum_distinct_attempts"]),
                int(row["future_horizon"]),
            )
        ].append(row)
    summaries = []
    for key, values in sorted(grouped.items()):
        premature = sum(row["resolution"] == "premature_v2_self_recovery" for row in values)
        confirmed = sum(row["resolution"] == "confirmed_no_change_window" for row in values)
        unresolved = len(values) - premature - confirmed
        resolved = premature + confirmed
        summaries.append(
            {
                "threshold": key[0],
                "minimum_distinct_attempts": key[1],
                "future_horizon": key[2],
                "trigger_count": len(values),
                "resolved_count": resolved,
                "premature_count": premature,
                "confirmed_count": confirmed,
                "unresolved_count": unresolved,
                "premature_rate_resolved": premature / resolved if resolved else None,
                "premature_wilson_upper_95": wilson_upper(premature, resolved),
                "map_count": len({str(row["map_id"]) for row in values}),
                "episode_count": len(
                    {(str(row["task_id"]), int(row["solver_seed"])) for row in values}
                ),
                "median_pretrigger_repair_seconds": statistics.median(
                    float(row["pretrigger_repair_seconds"]) for row in values
                ),
            }
        )
    return summaries


def audit_stall_confirmation_rules(
    config: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config).resolve()
    output_root = Path(output).resolve()
    raw = dict(read_json(config_path))
    project_root = Path(__file__).resolve().parents[1]
    sources = [
        (project_root / str(value)).resolve()
        if not Path(str(value)).is_absolute()
        else Path(str(value)).resolve()
        for value in raw.get("sources") or ()
    ]
    if not sources:
        raise ValueError("stall confirmation audit requires sources")
    identity = {
        "schema": STALL_CONFIRMATION_RULE_AUDIT_SCHEMA,
        "config_sha256": sha256_file(config_path),
        "sources": list(map(str, sources)),
        "attempt_thresholds": list(ATTEMPT_THRESHOLDS),
        "minimum_distinct_attempts": list(MINIMUM_DISTINCT_ATTEMPTS),
        "future_horizons": list(FUTURE_HORIZONS),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    prepare_run_output(output_root, resume=resume, identity=identity)
    trial_rows = []
    episode_count = 0
    source_trace_count = 0
    for source in sources:
        run_config = dict(read_json(source / "run_config.json"))
        if str(run_config.get("controller")) != "v2-full":
            raise ValueError("stall confirmation audit requires v2-full sources")
        manifests = [
            dict(row)
            for row in read_jsonl(source / "realized_dynamic_manifest.jsonl")
            if str(row.get("status")) in {"ok", "resumed"}
        ]
        for manifest in manifests:
            validate_manifest_trace(
                source,
                manifest,
                run_fingerprint=str(run_config["run_fingerprint"]),
                expected_policy="realized_dynamic",
            )
            decisions, events = decision_rows(source, manifest)
            normalized = normalize_episode_decisions(decisions, events)
            episode_count += 1
            source_trace_count += 1
            for run, recovery in _runs(normalized):
                for threshold in ATTEMPT_THRESHOLDS:
                    for minimum in MINIMUM_DISTINCT_ATTEMPTS:
                        for horizon in FUTURE_HORIZONS:
                            result = evaluate_no_progress_run(
                                run,
                                recovery,
                                threshold=threshold,
                                minimum_distinct_attempts=minimum,
                                future_horizon=horizon,
                            )
                            if result is None:
                                continue
                            trial_rows.append(
                                {
                                    "source": str(source),
                                    "map_id": str(manifest["map_id"]),
                                    "layout_mode": str(manifest.get("layout_mode", "unknown")),
                                    "agent_count": int(manifest.get("agent_count", 0)),
                                    "task_id": str(manifest["task_id"]),
                                    "solver_seed": int(manifest["solver_seed"]),
                                    "state_fingerprint": str(
                                        run[0]["before_repair_fingerprint"]
                                    ),
                                    **result,
                                }
                            )
    summaries = _summary(trial_rows)
    pilot_candidates = [
        row
        for row in summaries
        if int(row["future_horizon"]) == 3
        and int(row["confirmed_count"]) >= 5
        and int(row["map_count"]) >= 3
        and int(row["premature_count"]) == 0
        and float(row["premature_wilson_upper_95"] or 1.0) <= 0.25
    ]
    selected = (
        min(
            pilot_candidates,
            key=lambda row: (
                -int(row["confirmed_count"]),
                int(row["threshold"]),
                int(row["minimum_distinct_attempts"]),
            ),
        )
        if pilot_candidates
        else None
    )
    report = {
        "schema": STALL_CONFIRMATION_RULE_AUDIT_SCHEMA,
        "complete": True,
        "evidence_level": "direct frozen-v2 trajectory rule audit",
        "episode_count": episode_count,
        "source_trace_count": source_trace_count,
        "trigger_event_row_count": len(trial_rows),
        "rule_count": len(summaries),
        "selected_pilot_rule": selected,
        "pilot_rule_supported": selected is not None,
        "one_percent_promotion_claimed": False,
        "controller_actions_changed": False,
        "training_started": False,
        "deployment_promoted": False,
        "decision": (
            "collect_exact_full_pool_labels_at_confirmed_runtime_contexts"
            if selected is not None
            else "stop_stall_recovery_and_profile_pp"
        ),
    }
    atomic_write_csv(output_root / "stall_confirmation_events.csv", trial_rows)
    atomic_write_csv(output_root / "stall_confirmation_rule_summary.csv", summaries)
    write_json(output_root / "stall_confirmation_rule_report.json", report)
    return report


__all__ = [
    "ATTEMPT_THRESHOLDS",
    "FUTURE_HORIZONS",
    "MINIMUM_DISTINCT_ATTEMPTS",
    "STALL_CONFIRMATION_RULE_AUDIT_SCHEMA",
    "audit_stall_confirmation_rules",
    "evaluate_no_progress_run",
]
