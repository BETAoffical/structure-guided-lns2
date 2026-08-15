from __future__ import annotations

import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import mean, quantile, registered_input, sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_compact_blocker_rescue_continuation import (
    COMPACT_BLOCKER_AUGMENTED_MODE,
    INITIAL_TRIALS,
    _collection_path,
    _manifest_for_item,
    compact_blocker_schedule,
    prepare_factorial_cases,
)
from experiments.trace_replay import decision_rows as replay_decision_rows


CONFIG_SCHEMA = "lns2.stride.repairability_basin_audit_registration.v1"
REPORT_SCHEMA = "lns2.stride.repairability_basin_audit_report.v1"
ROW_SCHEMA = "lns2.stride.repairability_basin_audit_row.v1"
EXPERIMENT_ID = "stride-repairability-basin-audit-v1"


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = _read_json(path)
    if str(config.get("schema")) != CONFIG_SCHEMA:
        raise ValueError("repairability basin registration schema changed")
    if str(config.get("experiment_id")) != EXPERIMENT_ID:
        raise ValueError("repairability basin experiment id changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    population = dict(config["population"])
    if str(population["arm"]) != COMPACT_BLOCKER_AUGMENTED_MODE:
        raise ValueError("repairability basin arm changed")
    if list(map(int, population["trial_indices"])) != list(INITIAL_TRIALS):
        raise ValueError("repairability basin trials changed")
    execution = dict(config["execution"])
    if int(execution["worker_count"]) != 16:
        raise ValueError("repairability basin worker count changed")
    if execution.get("solver_calls_allowed") is not False:
        raise ValueError("repairability basin audit cannot call the solver")
    return path, root, config, inputs


def _ordered_unique(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in map(int, values):
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _attempt_diagnostics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    neighborhood = list(map(int, metrics.get("neighborhood") or ()))
    rows = list(metrics.get("pp_agent_diagnostics") or ())
    external = _ordered_unique(
        agent
        for raw in rows
        for agent in dict(raw).get("external_blocker_agents") or ()
    )
    internal = _ordered_unique(
        agent
        for raw in rows
        for agent in dict(raw).get("internal_blocker_agents") or ()
    )
    conflict_pairs = {
        tuple(sorted(map(int, pair)))
        for raw in rows
        for pair in dict(raw).get("new_conflict_pairs") or ()
    }
    failed_index = int(metrics.get("pp_failed_order_index", -1))
    failed_agent = int(metrics.get("pp_failed_agent", -1))
    failed_row = next(
        (
            dict(raw)
            for raw in rows
            if int(dict(raw).get("agent_id", -1)) == failed_agent
        ),
        {},
    )
    denominator = max(len(neighborhood) - 1, 1)
    return {
        "neighborhood": neighborhood,
        "neighborhood_size": len(neighborhood),
        "replan_success": bool(metrics.get("replan_success")),
        "failure_reason": str(metrics.get("pp_failure_reason")),
        "failed_agent": failed_agent,
        "failed_order_index": failed_index,
        "failed_order_fraction": (
            float(failed_index) / float(denominator) if failed_index >= 0 else None
        ),
        "attempted_agent_count": int(metrics.get("pp_attempted_agent_count", 0)),
        "inserted_agent_count": int(metrics.get("pp_inserted_agent_count", 0)),
        "external_blockers": external,
        "external_blocker_count": len(external),
        "internal_blockers": internal,
        "internal_blocker_count": len(internal),
        "new_conflict_pair_count": len(conflict_pairs),
        "failed_agent_external_blockers": _ordered_unique(
            failed_row.get("external_blocker_agents") or ()
        ),
        "failed_agent_internal_blockers": _ordered_unique(
            failed_row.get("internal_blocker_agents") or ()
        ),
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _outcome_class(durability: Mapping[str, Any]) -> str:
    h1 = dict(durability["horizons"]["1"])
    h8 = dict(durability["horizons"]["8"])
    if not bool(h1["sustained_escape"]):
        return "immediate_unresolved"
    if not bool(h8["observed"]):
        return "right_censored_before_h8"
    if bool(h8["new_platform_formed"]):
        return "new_platform_by_h8"
    if bool(h8["sustained_escape"]):
        return "durable_through_h8"
    return "other_observed_h8_failure"


def _external_blockers(metrics: Mapping[str, Any]) -> set[int]:
    return {
        int(agent)
        for raw in metrics.get("pp_agent_diagnostics") or ()
        for agent in dict(raw).get("external_blocker_agents") or ()
    }


def _audit_job(job: Mapping[str, Any]) -> dict[str, Any]:
    source_output = Path(str(job["source_output"])).resolve()
    item = dict(job["item"])
    durability = dict(job["durability"])
    manifest = _manifest_for_item(source_output, item)
    if manifest is None or str(manifest.get("status")) != "ok":
        raise ValueError("repairability basin source manifest is incomplete")
    rows, _events = replay_decision_rows(
        _collection_path(source_output, item), dict(manifest)
    )
    if len(rows) < 2:
        raise ValueError("repairability basin trace lacks the rescue decision")
    first = dict(rows[0]["actual_metrics"].get("failure_informed_rescue") or {})
    if not bool(first.get("trigger_eligible")):
        raise ValueError("repairability basin event is not trigger eligible")
    if not bool(first.get("exact_rollback")):
        raise ValueError("repairability basin trigger did not exact-roll back")
    rescue_record = dict(
        rows[1]["actual_metrics"].get("failure_informed_rescue") or {}
    )
    planned_agents = list(map(int, first["planned_agents"]))
    rescue_metrics = dict(rows[1]["actual_metrics"])
    if sorted(planned_agents) != sorted(
        map(int, rescue_metrics.get("neighborhood") or ())
    ):
        raise ValueError("repairability basin rescue membership changed")
    if bool(rows[1]["repair_state_changed"]) != bool(
        rescue_record.get("resolved_by_rescue")
    ):
        raise ValueError("repairability basin rescue outcome changed")
    first_attempt = _attempt_diagnostics(rows[0]["actual_metrics"])
    rescue_attempt = _attempt_diagnostics(rescue_metrics)
    compact_plan = dict(first["compact_plan"])
    selected_blockers = list(map(int, first["selected_blockers"]))
    rescue_set = set(planned_agents)
    future = rows[2:9]
    future_sets = [
        set(map(int, row["actual_metrics"].get("neighborhood") or ()))
        for row in future
    ]
    blocker_retention = [
        len(set(selected_blockers) & members) / len(selected_blockers)
        for members in future_sets
        if selected_blockers
    ]
    sequence = [rescue_set, *future_sets]
    adjacent_exact_repeats = sum(
        left == right for left, right in zip(sequence, sequence[1:])
    )
    future_external = set().union(
        *(_external_blockers(row["actual_metrics"]) for row in future)
    ) if future else set()
    h8 = dict(durability["horizons"]["8"])
    new_platform_offset = h8.get("new_platform_offset")
    platform_window: list[dict[str, Any]] = []
    if new_platform_offset is not None:
        endpoint = int(new_platform_offset)
        platform_window = rows[max(1, endpoint - 2) : endpoint + 1]
        if len(platform_window) != 3:
            raise ValueError("repairability basin new-platform window changed")
    window_external = set().union(
        *(
            _external_blockers(row["actual_metrics"])
            for row in platform_window
        )
    ) if platform_window else set()
    outcome = _outcome_class(durability)
    result = {
        "schema": ROW_SCHEMA,
        "state_fingerprint": str(item["state_fingerprint"]),
        "map_id": str(item["map_id"]),
        "trial_index": int(item["trial_index"]),
        "arm": str(item["arm"]),
        "outcome_class": outcome,
        "immediate_escape": outcome != "immediate_unresolved",
        "trace_sha256": str(manifest["trace_sha256"]),
        "origin_conflicts": int(rows[0]["before_conflicts"]),
        "trial_local_compaction": bool(compact_plan["eligible"]),
        "base_size": int(compact_plan["base_size"]),
        "compact_size": int(compact_plan["actual_size"]),
        "removed_agent_count": len(compact_plan["removed_agents"]),
        "selected_blocker_count": len(selected_blockers),
        "blocker_cap_reached": len(selected_blockers) == 8,
        "rescue_size": len(planned_agents),
        "first_attempt": first_attempt,
        "rescue_attempt": rescue_attempt,
        "rescue_residual_external_blocker_presence": bool(
            rescue_attempt["external_blocker_count"]
        ),
        "rescue_repair_wall_seconds": float(
            rows[1]["actual_lns2"]["outcome"]["repair_seconds"]
        ),
        "post_rescue_decision_count_through_h8": len(future),
        "mean_selected_blocker_retention_after_rescue": (
            mean(blocker_retention) if blocker_retention else None
        ),
        "mean_candidate_jaccard_to_rescue_after_rescue": (
            mean(_jaccard(rescue_set, members) for members in future_sets)
            if future_sets
            else None
        ),
        "adjacent_exact_candidate_repeat_count_through_h8": (
            adjacent_exact_repeats
        ),
        "unique_candidate_set_count_through_h8": len(
            {tuple(sorted(members)) for members in sequence}
        ),
        "future_external_blockers": sorted(future_external),
        "future_external_blocker_count": len(future_external),
        "new_platform_window_external_blockers": sorted(window_external),
        "new_platform_window_new_blockers": sorted(window_external - rescue_set),
        "new_platform_window_new_blocker_presence": bool(
            window_external - rescue_set
        ),
        "new_platform_window_decision_count": len(platform_window),
    }
    return result


SCALAR_FEATURES = (
    "origin_conflicts",
    "base_size",
    "compact_size",
    "removed_agent_count",
    "selected_blocker_count",
    "rescue_size",
    "rescue_repair_wall_seconds",
    "post_rescue_decision_count_through_h8",
    "mean_selected_blocker_retention_after_rescue",
    "mean_candidate_jaccard_to_rescue_after_rescue",
    "adjacent_exact_candidate_repeat_count_through_h8",
    "unique_candidate_set_count_through_h8",
    "future_external_blocker_count",
)


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalar: dict[str, Any] = {}
    for field in SCALAR_FEATURES:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        scalar[field] = {
            "observed_count": len(values),
            "mean": mean(values),
            "median": quantile(values, 0.5),
        }
    for prefix in ("first_attempt", "rescue_attempt"):
        for field in (
            "neighborhood_size",
            "failed_order_fraction",
            "external_blocker_count",
            "internal_blocker_count",
            "new_conflict_pair_count",
        ):
            values = [
                float(row[prefix][field])
                for row in rows
                if row[prefix].get(field) is not None
            ]
            scalar[f"{prefix}.{field}"] = {
                "observed_count": len(values),
                "mean": mean(values),
                "median": quantile(values, 0.5),
            }
    return {
        "event_count": len(rows),
        "state_count": len({str(row["state_fingerprint"]) for row in rows}),
        "by_map": dict(collections.Counter(str(row["map_id"]) for row in rows)),
        "trial_local_compaction_rate": mean(
            bool(row["trial_local_compaction"]) for row in rows
        ),
        "blocker_cap_rate": mean(bool(row["blocker_cap_reached"]) for row in rows),
        "rescue_residual_external_blocker_presence_rate": mean(
            bool(row["rescue_residual_external_blocker_presence"]) for row in rows
        ),
        "new_platform_window_new_blocker_presence_rate": mean(
            bool(row["new_platform_window_new_blocker_presence"]) for row in rows
        ),
        "features": scalar,
    }


def run_audit(
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    path, _root, config, inputs = load_registration(config_path)
    source_status = _read_json(inputs["source_status"])
    source_report = _read_json(inputs["source_report"])
    durability_report = _read_json(inputs["durability_report"])
    durability_rows = _read_jsonl(inputs["durability_rows"])
    if not bool(source_status.get("complete")) or int(
        source_status.get("completed_jobs", -1)
    ) != 900:
        raise ValueError("repairability basin source collection is incomplete")
    if not bool(source_report.get("integrity_passed")):
        raise ValueError("repairability basin source report failed integrity")
    if not bool(durability_report.get("integrity_passed")):
        raise ValueError("repairability basin durability report failed integrity")
    selected_durability = [
        row
        for row in durability_rows
        if str(row["arm"]) == COMPACT_BLOCKER_AUGMENTED_MODE
        and bool(row["trigger_eligible"])
    ]
    required = int(config["population"]["required_trigger_event_count"])
    if len(selected_durability) != required:
        raise ValueError("repairability basin trigger count changed")
    _loaded, cases = prepare_factorial_cases(inputs["source_registration"])
    schedule = compact_blocker_schedule(cases, INITIAL_TRIALS)
    schedule_index = {
        (
            str(item["state_fingerprint"]),
            int(item["trial_index"]),
            str(item["arm"]),
        ): item
        for item in schedule
    }
    jobs = []
    for durability in selected_durability:
        key = (
            str(durability["state_fingerprint"]),
            int(durability["trial_index"]),
            str(durability["arm"]),
        )
        if key not in schedule_index:
            raise ValueError("repairability basin source schedule key changed")
        jobs.append(
            {
                "source_output": str(inputs["source_status"].parent),
                "item": schedule_index[key],
                "durability": durability,
            }
        )
    worker_count = int(workers or config["execution"]["worker_count"])
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_audit_job, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (str(row["state_fingerprint"]), int(row["trial_index"])))
    counts = collections.Counter(str(row["outcome_class"]) for row in rows)
    known = dict(config["known_before_freeze"])
    expected_counts = {
        "immediate_unresolved": int(known["immediate_unresolved_count"]),
        "durable_through_h8": int(known["immediate_escape_h8_durable_count"]),
        "new_platform_by_h8": int(
            known["immediate_escape_h8_new_platform_count"]
        ),
        "right_censored_before_h8": int(
            known["immediate_escape_h8_censored_count"]
        ),
    }
    immediate_unresolved = [
        row for row in rows if row["outcome_class"] == "immediate_unresolved"
    ]
    immediate_escape = [
        row for row in rows if row["outcome_class"] != "immediate_unresolved"
    ]
    durable = [row for row in rows if row["outcome_class"] == "durable_through_h8"]
    new_platform = [
        row for row in rows if row["outcome_class"] == "new_platform_by_h8"
    ]
    summaries = {
        "immediate_unresolved": _group_summary(immediate_unresolved),
        "immediate_escape": _group_summary(immediate_escape),
        "durable_through_h8": _group_summary(durable),
        "new_platform_by_h8": _group_summary(new_platform),
    }
    failure_presence = summaries["immediate_unresolved"][
        "rescue_residual_external_blocker_presence_rate"
    ]
    escape_presence = summaries["immediate_escape"][
        "rescue_residual_external_blocker_presence_rate"
    ]
    new_presence = summaries["new_platform_by_h8"][
        "new_platform_window_new_blocker_presence_rate"
    ]
    accumulation_rule = dict(
        config["mechanism_rules"]["accumulated_blocker_rescue_supported_if"]
    )
    signature_rule = dict(
        config["mechanism_rules"]["new_signature_rescue_supported_if"]
    )
    accumulation_supported = bool(
        len(immediate_unresolved)
        >= int(accumulation_rule["minimum_immediate_unresolved_events"])
        and failure_presence
        >= float(
            accumulation_rule[
                "minimum_unresolved_rescue_residual_blocker_presence_rate"
            ]
        )
        and failure_presence - escape_presence
        >= float(accumulation_rule["minimum_presence_rate_difference_vs_immediate_escape"])
    )
    new_signature_supported = bool(
        len(new_platform) >= int(signature_rule["minimum_new_platform_events"])
        and new_presence
        >= float(
            signature_rule[
                "minimum_new_platform_window_new_blocker_presence_rate"
            ]
        )
    )
    if accumulation_supported and new_signature_supported:
        next_protocol = "bounded_accumulated_blockers_per_platform_signature"
    elif accumulation_supported:
        next_protocol = "bounded_accumulated_blockers_same_signature_only"
    elif new_signature_supported:
        next_protocol = "bounded_one_rescue_per_new_platform_signature"
    else:
        next_protocol = "stop_stateful_membership_branch"
    integrity = {
        "source_complete": bool(source_status["complete"]),
        "source_integrity_passed": bool(source_report["integrity_passed"]),
        "durability_integrity_passed": bool(durability_report["integrity_passed"]),
        "row_count": len(rows) == required,
        "outcome_partition_exact": all(
            int(counts.get(group, 0)) == expected
            for group, expected in expected_counts.items()
        )
        and not (set(counts) - set(expected_counts)),
        "all_compact_blocker_arm": all(
            row["arm"] == COMPACT_BLOCKER_AUGMENTED_MODE for row in rows
        ),
        "trace_hash_present": all(len(str(row["trace_sha256"])) == 64 for row in rows),
        "zero_solver_calls_by_protocol": config["execution"]["solver_calls_allowed"]
        is False,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "worker_count": worker_count,
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "outcome_counts": dict(sorted(counts.items())),
        "summaries": summaries,
        "mechanism_contrasts": {
            "immediate_failure_minus_escape_residual_blocker_presence": (
                failure_presence - escape_presence
            ),
            "new_platform_window_new_blocker_presence": new_presence,
        },
        "decision": {
            "accumulated_blocker_rescue_supported": accumulation_supported,
            "new_signature_rescue_supported": new_signature_supported,
            "next_protocol": next_protocol,
            "new_pp_authorized_in_this_stage": False,
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema": CONFIG_SCHEMA,
        "registration": str(path),
        "registration_sha256": sha256_file(path),
        "selected_schedule_sha256": _fingerprint(
            [dict(job["item"]) for job in jobs]
        ),
        "worker_count": worker_count,
    }
    _write_json(output_path / "audit_run_config.json", run_config)
    _write_jsonl(output_path / "audit_rows.jsonl", rows)
    _write_json(output_path / "report.json", report)
    return report


__all__ = [
    "EXPERIMENT_ID",
    "_attempt_diagnostics",
    "_outcome_class",
    "load_registration",
    "run_audit",
]
