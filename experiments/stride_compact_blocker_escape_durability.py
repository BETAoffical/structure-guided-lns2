from __future__ import annotations

import collections
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import (
    contained_file,
    mean,
    quantile,
    registered_input,
    sha256_file,
)
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_trace_events,
)
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_compact_blocker_rescue_continuation import (
    BLOCKER_AUGMENTED_MODE,
    COMPACT_BLOCKER_AUGMENTED_MODE,
    INITIAL_TRIALS,
    _collection_path,
    _manifest_for_item,
    compact_blocker_schedule,
    prepare_factorial_cases,
)
from experiments.trace_replay import _initial_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = (
    "lns2.stride.compact_blocker_escape_durability_audit_registration.v1"
)
REPORT_SCHEMA = "lns2.stride.compact_blocker_escape_durability_audit_report.v1"
ROW_SCHEMA = "lns2.stride.compact_blocker_escape_durability_row.v1"
EXPERIMENT_ID = "stride-compact-blocker-escape-durability-v1"
ARMS = (BLOCKER_AUGMENTED_MODE, COMPACT_BLOCKER_AUGMENTED_MODE)


def load_audit_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = _read_json(path)
    if str(config.get("schema")) != CONFIG_SCHEMA:
        raise ValueError("escape durability registration schema changed")
    if str(config.get("experiment_id")) != EXPERIMENT_ID:
        raise ValueError("escape durability experiment id changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    population = dict(config["population"])
    if list(map(int, population["required_trial_indices"])) != list(INITIAL_TRIALS):
        raise ValueError("escape durability trial identity changed")
    if tuple(map(str, population["paired_arms"])) != ARMS:
        raise ValueError("escape durability arm identity changed")
    durability = dict(config["durability"])
    if list(map(int, durability["horizons"])) != [1, 3, 8]:
        raise ValueError("escape durability horizons changed")
    execution = dict(config["execution"])
    if int(execution["worker_count"]) != 16:
        raise ValueError("escape durability worker count changed")
    if execution.get("solver_calls_allowed") is not False:
        raise ValueError("escape durability audit cannot call the solver")
    return path, root, config, inputs


def _edge_set(state: Mapping[str, Any]) -> set[tuple[int, int]]:
    return {
        tuple(sorted(map(int, edge)))
        for edge in state.get("conflict_edges", ())
    }


def _transition_rows(
    collection_root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    trace_path = contained_file(
        collection_root,
        manifest.get("trace_file"),
        field="escape durability trace file",
    )
    events = read_trace_events(trace_path)
    if len(events) < 3:
        raise ValueError("escape durability trace has no transitions")
    state = _initial_state(collection_root, trace_path, events[0])
    rows: list[dict[str, Any]] = []
    for event in events[1:-1]:
        if state_fingerprint(state) != str(event.get("before_fingerprint")):
            raise ValueError("escape durability before fingerprint mismatch")
        if str(event.get("schema")) != EPISODE_SCHEMA_V2:
            raise ValueError("escape durability requires delta trace v2")
        after = apply_state_delta(state, event["state_delta"])
        after.update(apply_extras_delta(state, event["state_extras_delta"]))
        if state_fingerprint(after) != str(event.get("after_fingerprint")):
            raise ValueError("escape durability after fingerprint mismatch")
        before_repair = repair_structure_fingerprint(state)
        after_repair = repair_structure_fingerprint(after)
        metrics = dict(event["metrics"])
        exact_rollback = bool(
            metrics.get("pp_failure_reason") == "conflict_bound_exceeded"
            and metrics.get("replan_success") is False
            and metrics.get("pp_rolled_back") is True
            and before_repair == after_repair
        )
        rows.append(
            {
                "decision_index": int(event["decision_index"]),
                "before_repair_fingerprint": before_repair,
                "after_repair_fingerprint": after_repair,
                "before_edges": _edge_set(state),
                "after_edges": _edge_set(after),
                "before_conflicts": int(state["num_of_colliding_pairs"]),
                "after_conflicts": int(after["num_of_colliding_pairs"]),
                "after_feasible": bool(after["feasible"]),
                "exact_rollback": exact_rollback,
                "failure_informed_rescue": dict(
                    metrics.get("failure_informed_rescue") or {}
                ),
                "controller": dict(event["controller"]),
            }
        )
        state = after
    return rows


def _durability_metrics(
    rows: list[dict[str, Any]], horizon: int
) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("eligible escape trace lacks rescue decision")
    origin = str(rows[0]["after_repair_fingerprint"])
    if rows[0]["before_repair_fingerprint"] != origin:
        raise ValueError("eligible source decision did not exact-roll back")
    origin_edges = set(rows[0]["after_edges"])
    origin_conflicts = int(rows[0]["after_conflicts"])
    if not origin_edges or origin_conflicts <= 0:
        raise ValueError("eligible platform lacks conflicts")
    available = rows[1 : 1 + int(horizon)]
    successful = any(bool(row["after_feasible"]) for row in available)
    observed = len(available) == int(horizon) or successful
    if not observed:
        return {
            "observed": False,
            "right_censored": True,
            "decision_count": len(available),
        }
    if successful:
        stop = next(
            index
            for index, row in enumerate(available, start=1)
            if bool(row["after_feasible"])
        )
        available = available[:stop]
    rescue_escaped = bool(
        rows[1]["after_repair_fingerprint"]
        != rows[1]["before_repair_fingerprint"]
    )
    original_reentry_offset: int | None = None
    new_platform_offset: int | None = None
    streak = 0
    streak_signature: str | None = None
    edge_retention: list[float] = []
    normalized_conflicts: list[float] = []
    for offset, row in enumerate(available, start=1):
        after_signature = str(row["after_repair_fingerprint"])
        if after_signature == origin and original_reentry_offset is None:
            original_reentry_offset = offset
        if bool(row["exact_rollback"]):
            before_signature = str(row["before_repair_fingerprint"])
            if streak_signature == before_signature:
                streak += 1
            else:
                streak_signature = before_signature
                streak = 1
            if (
                streak >= 3
                and before_signature != origin
                and new_platform_offset is None
            ):
                new_platform_offset = offset
        else:
            streak = 0
            streak_signature = None
        after_edges = set(row["after_edges"])
        edge_retention.append(len(origin_edges & after_edges) / len(origin_edges))
        normalized_conflicts.append(
            float(row["after_conflicts"]) / float(origin_conflicts)
        )
    sustained = bool(
        rescue_escaped
        and original_reentry_offset is None
        and new_platform_offset is None
    )
    return {
        "observed": True,
        "right_censored": False,
        "decision_count": len(available),
        "rescue_escaped": rescue_escaped,
        "original_platform_reentered": original_reentry_offset is not None,
        "original_reentry_offset": original_reentry_offset,
        "new_platform_formed": new_platform_offset is not None,
        "new_platform_offset": new_platform_offset,
        "sustained_escape": sustained,
        "original_edge_retention_auc": mean(edge_retention),
        "original_edge_retention_at_horizon": edge_retention[-1],
        "normalized_conflict_auc": mean(normalized_conflicts),
        "strict_conflict_improvement": min(
            int(row["after_conflicts"]) for row in available
        )
        < origin_conflicts,
        "feasible_by_horizon": any(
            bool(row["after_feasible"]) for row in available
        ),
    }


def _audit_job(job: dict[str, Any]) -> dict[str, Any]:
    output = Path(str(job["source_output"])).resolve()
    item = dict(job["item"])
    manifest = _manifest_for_item(output, item)
    if manifest is None or str(manifest.get("status")) != "ok":
        raise ValueError("escape durability source manifest is incomplete")
    rows = _transition_rows(_collection_path(output, item), manifest)
    first = dict(rows[0]["failure_informed_rescue"])
    compact_plan = dict(first["compact_plan"])
    trigger_eligible = bool(first["trigger_eligible"])
    rescue_count = sum(
        bool(row["controller"].get("failure_informed_rescue_action"))
        for row in rows
    )
    if rescue_count != int(trigger_eligible):
        raise ValueError("escape durability rescue count changed")
    result = {
        "schema": ROW_SCHEMA,
        "state_fingerprint": str(item["state_fingerprint"]),
        "map_id": str(item["map_id"]),
        "trial_index": int(item["trial_index"]),
        "arm": str(item["arm"]),
        "trigger_eligible": trigger_eligible,
        "compact_eligible": bool(compact_plan["eligible"]),
        "transition_count": len(rows),
        "source_trace_sha256": str(manifest["trace_sha256"]),
        "horizons": {},
    }
    if trigger_eligible:
        result["horizons"] = {
            str(horizon): _durability_metrics(rows, int(horizon))
            for horizon in map(int, job["horizons"])
        }
    return result


def _summary(
    rows: list[dict[str, Any]], arm: str, horizon: int
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["arm"] == arm and bool(row["trigger_eligible"])
    ]
    observed = [
        dict(row["horizons"][str(horizon)])
        for row in eligible
        if bool(row["horizons"][str(horizon)]["observed"])
    ]
    return {
        "eligible_event_count": len(eligible),
        "observed_count": len(observed),
        "right_censored_count": len(eligible) - len(observed),
        "sustained_escape_rate": mean(
            bool(row["sustained_escape"]) for row in observed
        ),
        "original_platform_reentry_rate": mean(
            bool(row["original_platform_reentered"]) for row in observed
        ),
        "new_platform_rate": mean(
            bool(row["new_platform_formed"]) for row in observed
        ),
        "mean_original_edge_retention_auc": mean(
            float(row["original_edge_retention_auc"]) for row in observed
        ),
        "mean_original_edge_retention_at_horizon": mean(
            float(row["original_edge_retention_at_horizon"])
            for row in observed
        ),
        "mean_normalized_conflict_auc": mean(
            float(row["normalized_conflict_auc"]) for row in observed
        ),
        "strict_conflict_improvement_rate": mean(
            bool(row["strict_conflict_improvement"]) for row in observed
        ),
        "feasible_by_horizon_rate": mean(
            bool(row["feasible_by_horizon"]) for row in observed
        ),
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    horizon: int,
    field: str,
    replicates: int,
    allow_empty: bool = False,
) -> dict[str, Any]:
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for row in rows:
        if not bool(row["trigger_eligible"]):
            continue
        metric = dict(row["horizons"][str(horizon)])
        if not bool(metric["observed"]):
            continue
        paired[(str(row["state_fingerprint"]), int(row["trial_index"]))][
            str(row["arm"])
        ] = metric
    complete = {
        key: value for key, value in paired.items() if set(value) == set(ARMS)
    }
    states = sorted({key[0] for key in complete})
    if not states:
        if allow_empty:
            return {
                "field": field,
                "horizon": int(horizon),
                "paired_state_count": 0,
                "paired_event_count": 0,
                "point": None,
                "lower_95": None,
                "upper_95": None,
            }
        raise ValueError("escape durability contrast has no complete state")

    def difference(sampled_states: Iterable[str]) -> float:
        selected = [
            value
            for state in sampled_states
            for key, value in complete.items()
            if key[0] == state
        ]
        baseline = [float(value[ARMS[0]][field]) for value in selected]
        treatment = [float(value[ARMS[1]][field]) for value in selected]
        return mean(treatment) - mean(baseline)

    point = difference(states)
    rng = random.Random(0x45534344 + horizon + sum(map(ord, field)))
    samples = [
        difference([rng.choice(states) for _ in states])
        for _ in range(int(replicates))
    ]
    return {
        "field": field,
        "horizon": int(horizon),
        "paired_state_count": len(states),
        "paired_event_count": len(complete),
        "point": point,
        "lower_95": quantile(samples, 0.025),
        "upper_95": quantile(samples, 0.975),
    }


def run_audit(
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    path, root, config, inputs = load_audit_registration(config_path)
    source_status = _read_json(inputs["source_status"])
    source_report = _read_json(inputs["source_report"])
    source_run = _read_json(inputs["source_run_config"])
    source_schedule = _read_jsonl(inputs["source_schedule"])
    if not bool(source_status.get("complete")) or int(
        source_status.get("completed_jobs", -1)
    ) != 900:
        raise ValueError("escape durability source collection is incomplete")
    if not bool(source_report.get("integrity_passed")):
        raise ValueError("escape durability source integrity failed")
    if str(source_run.get("run_fingerprint")) != str(
        source_status.get("run_fingerprint")
    ):
        raise ValueError("escape durability source run identity changed")
    loaded, cases = prepare_factorial_cases(inputs["source_registration"])
    _source_path, _source_root, _source_config, *_remaining = loaded
    generated_schedule = compact_blocker_schedule(cases, INITIAL_TRIALS)
    if generated_schedule != source_schedule:
        raise ValueError("escape durability source schedule changed")
    compact_states = set(map(str, source_report["compact_eligible_state_fingerprints"]))
    required_states = int(config["population"]["required_compact_eligible_state_count"])
    if len(compact_states) != required_states:
        raise ValueError("escape durability compact state count changed")
    selected = [
        item
        for item in generated_schedule
        if str(item["state_fingerprint"]) in compact_states
        and str(item["arm"]) in ARMS
    ]
    expected_count = required_states * len(INITIAL_TRIALS) * len(ARMS)
    if len(selected) != expected_count:
        raise ValueError("escape durability selected schedule count changed")
    source_output = inputs["source_status"].parent
    horizons = list(map(int, config["durability"]["horizons"]))
    jobs = [
        {
            "source_output": str(source_output),
            "item": item,
            "horizons": horizons,
        }
        for item in selected
    ]
    worker_count = int(workers or config["execution"]["worker_count"])
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_audit_job, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(
        key=lambda row: (
            str(row["state_fingerprint"]),
            int(row["trial_index"]),
            ARMS.index(str(row["arm"])),
        )
    )
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for row in rows:
        pairs[(str(row["state_fingerprint"]), int(row["trial_index"]))][
            str(row["arm"])
        ] = row
    paired_trigger_count = sum(
        set(value) == set(ARMS)
        and all(bool(value[arm]["trigger_eligible"]) for arm in ARMS)
        for value in pairs.values()
    )
    required_trigger_count = int(config["population"]["required_trigger_pair_count"])
    if paired_trigger_count != required_trigger_count:
        raise ValueError("escape durability trigger pair count changed")
    replicates = int(config["contrasts"]["state_cluster_bootstrap_replicates"])
    fields = (
        "sustained_escape",
        "original_platform_reentered",
        "new_platform_formed",
        "original_edge_retention_auc",
        "normalized_conflict_auc",
        "strict_conflict_improvement",
        "feasible_by_horizon",
    )
    summaries = {
        str(horizon): {
            arm: _summary(rows, arm, horizon) for arm in ARMS
        }
        for horizon in horizons
    }
    contrasts = {
        str(horizon): {
            field: _paired_bootstrap(
                rows,
                horizon=horizon,
                field=field,
                replicates=replicates,
            )
            for field in fields
        }
        for horizon in horizons
    }
    by_map: dict[str, Any] = {}
    for map_id in sorted({str(row["map_id"]) for row in rows}):
        selected_rows = [row for row in rows if row["map_id"] == map_id]
        by_map[map_id] = {
            str(horizon): {
                "arms": {
                    arm: _summary(selected_rows, arm, horizon) for arm in ARMS
                },
                "sustained_escape_difference": _paired_bootstrap(
                    selected_rows,
                    horizon=horizon,
                    field="sustained_escape",
                    replicates=replicates,
                    allow_empty=True,
                )["point"],
            }
            for horizon in horizons
        }
    h1 = contrasts["1"]["sustained_escape"]
    h3 = contrasts["3"]["sustained_escape"]
    h8 = contrasts["8"]["sustained_escape"]
    edge_h8 = contrasts["8"]["original_edge_retention_auc"]
    map_safe = all(
        value["8"]["sustained_escape_difference"] is not None
        and float(value["8"]["sustained_escape_difference"]) >= -0.05
        for value in by_map.values()
    )
    h1_stable = float(h1["lower_95"]) > 0.0
    h3_stable = float(h3["lower_95"]) > 0.0
    h8_stable = float(h8["lower_95"]) > 0.0
    edge_stable = float(edge_h8["upper_95"]) < 0.0
    durable = h3_stable and h8_stable and edge_stable and map_safe
    if durable:
        classification = "durable_structural_escape_signal"
    elif h1_stable and (not h3_stable or not h8_stable):
        classification = "transient_escape_signal"
    elif h1_stable and h3_stable and h8_stable:
        classification = "durable_fingerprint_escape_without_edge_clearance"
    else:
        classification = "inconclusive_escape_durability"
    integrity = {
        "source_complete": bool(source_status["complete"]),
        "source_integrity_passed": bool(source_report["integrity_passed"]),
        "source_schedule_exact": generated_schedule == source_schedule,
        "selected_row_count": len(rows) == expected_count,
        "paired_arm_complete": len(pairs) == required_states * len(INITIAL_TRIALS)
        and all(set(value) == set(ARMS) for value in pairs.values()),
        "trigger_pair_count": paired_trigger_count == required_trigger_count,
        "trace_hash_present": all(len(str(row["source_trace_sha256"])) == 64 for row in rows),
        "zero_solver_calls_by_protocol": config["execution"]["solver_calls_allowed"]
        is False,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "source_run_fingerprint": str(source_run["run_fingerprint"]),
        "worker_count": worker_count,
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "compact_eligible_state_count": len(compact_states),
        "paired_trigger_event_count": paired_trigger_count,
        "summaries": summaries,
        "contrasts": contrasts,
        "by_map": by_map,
        "decision": {
            "classification": classification,
            "horizon_1_sustained_escape_stable": h1_stable,
            "horizon_3_sustained_escape_stable": h3_stable,
            "horizon_8_sustained_escape_stable": h8_stable,
            "horizon_8_edge_clearance_stable": edge_stable,
            "per_map_horizon_8_safe": map_safe,
            "durable_signal": durable,
            "future_recovery_research_warranted": durable,
            "failed_rescue_wall_gate_reopened": False,
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": CONFIG_SCHEMA,
        "registration": str(path),
        "registration_sha256": sha256_file(path),
        "source_run_fingerprint": str(source_run["run_fingerprint"]),
        "selected_schedule_sha256": _fingerprint(selected),
        "worker_count": worker_count,
    }
    _write_json(output / "audit_run_config.json", identity)
    _write_jsonl(output / "audit_rows.jsonl", rows)
    _write_json(output / "report.json", report)
    return report


__all__ = [
    "ARMS",
    "EXPERIMENT_ID",
    "load_audit_registration",
    "run_audit",
]
