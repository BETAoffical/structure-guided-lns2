from __future__ import annotations

import collections
import csv
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    atomic_write_csv as _atomic_write_csv,
    producer_identity,
    read_json,
    sha256_file,
    strict_nonnegative_int as _strict_nonnegative_int,
)
from experiments.repair_aware import REPAIR_OUTCOMES, classify_repair_outcome
from experiments.repair_collection import (
    _fingerprint,
    _low_level_delta,
    _plain,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)
from experiments.stall_guard import repair_structure_fingerprint
from experiments.trace_replay import replay_prefix
from experiments.v3_s3_collection import (
    _paired_repair_action,
    _paired_seed,
    _source_replay_job,
    validate_v3_s3_collection_source,
)


V3_VALUE_PILOT_SCHEMA = "lns2.v3_value_label_pilot.v3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
V3_VALUE_PILOT_PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/closed_loop_trace_storage.py",
    "experiments/compact_controller_model.py",
    "experiments/context_audit.py",
    "experiments/feature_schema_v2.py",
    "experiments/feature_schema_v3.py",
    "experiments/neighborhood_candidates.py",
    "experiments/neighborhood_features.py",
    "experiments/online_feature_engine.py",
    "experiments/parallel_runtime.py",
    "experiments/repair_aware.py",
    "experiments/repair_aware_training.py",
    "experiments/repair_collection.py",
    "experiments/stall_guard.py",
    "experiments/state_analysis.py",
    "experiments/trace_replay.py",
    "experiments/v3_controller.py",
    "experiments/v3_s3.py",
    "experiments/v3_s3_collection.py",
    "experiments/v3_value_pilot.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/BasicLNS.h",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)
DEFAULT_OVERHEAD_GRID = (0.0, 0.05, 0.10, 0.15)
ARM_PRIORITY = (
    "v2_full",
    "model_s3",
    "oracle_s3_efficiency",
    "oracle_s3_quality_time",
    "oracle_h1_efficiency",
)


def _source_string(
    value: Any,
    *,
    field: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"value-pilot source {field} must be {qualifier}")
    return value


def _source_object_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"value-pilot source {field} must be a list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"value-pilot source {field}[{index}] must be an object"
            )
        result.append(dict(item))
    return result


def _source_agent_ids(value: Any, *, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            type(agent) is not int or agent < 0
            for agent in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"value-pilot source {field} must be a non-empty list of unique "
            "non-negative integers"
        )
    return list(value)


def _source_first_steps(
    row: dict[str, Any],
    *,
    field: str,
    require_executed: bool,
) -> list[dict[str, Any]]:
    result = []
    for index, step in enumerate(
        _source_object_list(row.get("steps"), field=f"{field}.steps")
    ):
        step_number = step.get("step")
        if not _strict_nonnegative_int(step_number):
            raise ValueError(
                f"value-pilot source {field}.steps[{index}].step "
                "must be a non-negative integer"
            )
        if require_executed:
            executed = step.get("executed")
            if type(executed) is not bool:
                raise ValueError(
                    f"value-pilot source {field}.steps[{index}].executed "
                    "must be boolean"
                )
            if step_number == 1 and executed:
                result.append(step)
        elif step_number == 1:
            result.append(step)
    return result


def _candidate_from_sequence(
    payload: dict[str, Any], sequence_id: str
) -> dict[str, Any]:
    expected_sequence_id = _source_string(
        sequence_id,
        field="requested sequence_id",
    )
    matches = []
    for index, row in enumerate(
        _source_object_list(payload.get("trials"), field="trials")
    ):
        stored_sequence_id = _source_string(
            row.get("sequence_id"),
            field=f"trials[{index}].sequence_id",
        )
        if stored_sequence_id == expected_sequence_id:
            matches.append(row)
    if not matches:
        raise ValueError(f"state lacks S3 sequence: {sequence_id}")
    identities = set()
    template_keys = set()
    for index, row in enumerate(matches):
        steps = _source_first_steps(
            row,
            field=f"sequence {expected_sequence_id} trial {index}",
            require_executed=True,
        )
        if len(steps) != 1:
            raise ValueError(f"S3 sequence has invalid first-step coverage: {sequence_id}")
        step = steps[0]
        candidate_id = _source_string(
            step.get("candidate_id"),
            field=f"sequence {expected_sequence_id} candidate_id",
        )
        agents = _source_agent_ids(
            step.get("agents"),
            field=f"sequence {expected_sequence_id} agents",
        )
        identities.add(
            (
                candidate_id,
                tuple(sorted(agents)),
            )
        )
        templates = _source_object_list(
            row.get("templates"),
            field=f"sequence {expected_sequence_id} templates",
        )
        if not templates:
            raise ValueError(
                f"S3 sequence has no templates: {expected_sequence_id}"
            )
        template_keys.add(
            _source_string(
                templates[0].get("template_key"),
                field=f"sequence {expected_sequence_id} template_key",
            )
        )
    if len(identities) != 1 or len(template_keys) != 1:
        raise ValueError(f"S3 sequence first action is not deterministic: {sequence_id}")
    candidate_id, agents = identities.pop()
    return {
        "candidate_id": candidate_id,
        "agents": list(agents),
        "template_key": template_keys.pop(),
        "sequence_id": str(sequence_id),
    }


def _candidate_from_template(
    payload: dict[str, Any], template_key: str
) -> dict[str, Any]:
    expected_template_key = _source_string(
        template_key,
        field="requested template_key",
    )
    candidates = {}
    for index, row in enumerate(
        _source_object_list(payload.get("trials"), field="trials")
    ):
        templates = _source_object_list(
            row.get("templates"),
            field=f"trials[{index}].templates",
        )
        if not templates:
            raise ValueError("value-pilot source trial has no templates")
        stored_template_key = _source_string(
            templates[0].get("template_key"),
            field=f"trials[{index}].template_key",
        )
        if stored_template_key != expected_template_key:
            continue
        steps = _source_first_steps(
            row,
            field=f"template {expected_template_key} trial {index}",
            require_executed=True,
        )
        if len(steps) != 1:
            continue
        step = steps[0]
        candidate_id = _source_string(
            step.get("candidate_id"),
            field=f"template {expected_template_key} candidate_id",
        )
        agents = _source_agent_ids(
            step.get("agents"),
            field=f"template {expected_template_key} agents",
        )
        key = (
            candidate_id,
            tuple(sorted(agents)),
        )
        candidates[key] = {
            "candidate_id": key[0],
            "agents": list(key[1]),
            "template_key": expected_template_key,
            "sequence_id": "",
        }
    if len(candidates) != 1:
        raise ValueError(
            f"state template does not map to one first action: {template_key}"
        )
    return next(iter(candidates.values()))


def _v2_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for index, row in enumerate(
        _source_object_list(
            payload.get("external_baselines"),
            field="external_baselines",
        )
    ):
        controller = _source_string(
            row.get("controller"),
            field=f"external_baselines[{index}].controller",
        )
        if controller == "v2-full":
            rows.append(row)
    identities = set()
    for index, row in enumerate(rows):
        steps = _source_first_steps(
            row,
            field=f"v2 baseline {index}",
            require_executed=False,
        )
        if len(steps) != 1:
            raise ValueError("v2 baseline has invalid first-step coverage")
        step = steps[0]
        action_value = step.get("action")
        if not isinstance(action_value, dict):
            raise ValueError("v2 baseline first action is not an object")
        action = dict(action_value)
        candidate_id = _source_string(
            step.get("candidate_id"),
            field="v2 baseline candidate_id",
        )
        agents = _source_agent_ids(
            action.get("agents"),
            field="v2 baseline agents",
        )
        identities.add(
            (
                candidate_id,
                tuple(sorted(agents)),
            )
        )
    if len(identities) != 1:
        raise ValueError("v2 baseline first action is not deterministic")
    candidate_id, agents = identities.pop()
    return {
        "candidate_id": candidate_id,
        "agents": list(agents),
        "template_key": "",
        "sequence_id": "",
    }


def _shared_initial_selection_seconds(payload: dict[str, Any]) -> float:
    values = []
    for index, row in enumerate(
        _source_object_list(payload.get("trials"), field="trials")
    ):
        steps = _source_first_steps(
            row,
            field=f"trials[{index}]",
            require_executed=True,
        )
        for step in steps:
            selection_seconds = step.get("selection_seconds")
            if (
                isinstance(selection_seconds, bool)
                or not isinstance(selection_seconds, (int, float))
                or not math.isfinite(float(selection_seconds))
                or float(selection_seconds) < 0.0
            ):
                raise ValueError(
                    "state has invalid initial candidate selection time"
                )
            values.append(float(selection_seconds))
    if not values:
        raise ValueError("state has no measured initial candidate selection time")
    return statistics.median(values)


def build_state_arms(
    payload: dict[str, Any], oracle_row: dict[str, Any]
) -> list[dict[str, Any]]:
    requested = [
        ("v2_full", _v2_candidate(payload)),
        (
            "model_s3",
            _candidate_from_sequence(payload, str(oracle_row["model_sequence_id"])),
        ),
        (
            "oracle_s3_efficiency",
            _candidate_from_sequence(
                payload,
                str(oracle_row["oracle_s3_efficiency_sequence_id"]),
            ),
        ),
        (
            "oracle_s3_quality_time",
            _candidate_from_sequence(
                payload,
                str(oracle_row["oracle_s3_quality_time_sequence_id"]),
            ),
        ),
        (
            "oracle_h1_efficiency",
            _candidate_from_template(
                payload,
                str(oracle_row["oracle_h1_efficiency_first_template"]),
            ),
        ),
    ]
    by_agents: dict[tuple[int, ...], dict[str, Any]] = {}
    for arm_id, candidate in requested:
        agents = tuple(sorted(map(int, candidate["agents"])))
        existing = by_agents.get(agents)
        if existing is None:
            by_agents[agents] = {
                "arm_id": arm_id,
                "aliases": [arm_id],
                **candidate,
            }
        else:
            existing["aliases"].append(arm_id)
    order = {name: index for index, name in enumerate(ARM_PRIORITY)}
    result = sorted(
        by_agents.values(),
        key=lambda row: (
            min(order[name] for name in row["aliases"]),
            tuple(row["agents"]),
        ),
    )
    for row in result:
        row["aliases"] = sorted(row["aliases"], key=order.get)
        row["arm_id"] = row["aliases"][0]
    return result


def _balanced_state_sample(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if int(count) <= 0:
        raise ValueError("state_count must be positive")
    cells: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    for row in rows:
        cells[(str(row["layout_mode"]), int(row["agent_count"]))].append(row)
    for values in cells.values():
        values.sort(
            key=lambda row: (
                -int(row["initial_conflicts"]),
                _fingerprint(str(row["state_id"])),
            )
        )
    selected = []
    offset = 0
    while len(selected) < int(count):
        added = False
        for cell in sorted(cells):
            values = cells[cell]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) >= int(count):
                    break
        if not added:
            break
        offset += 1
    return selected


def _unique_rows_by_state_id(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} contains a non-object row")
        row = dict(raw)
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError(f"{label} contains an invalid state_id")
        if state_id in result:
            raise ValueError(f"{label} contains duplicate state_id: {state_id}")
        result[state_id] = row
    return result


def build_value_pilot_plan(
    *,
    source: str | Path,
    oracle_state_comparison: str | Path,
    state_count: int,
    split: str = "policy_train",
) -> dict[str, Any]:
    if not _strict_nonnegative_int(state_count) or state_count <= 0:
        raise ValueError("value-pilot state_count must be a positive integer")
    if not isinstance(split, str) or not split:
        raise ValueError("value-pilot split must be a non-empty string")
    source_root = Path(source).resolve()
    source_validation = validate_v3_s3_collection_source(
        source_root / "collection"
    )
    oracle_path = Path(oracle_state_comparison).resolve()
    with oracle_path.open(encoding="utf-8", newline="") as stream:
        all_oracle_rows = _unique_rows_by_state_id(
            csv.DictReader(stream),
            label="value-pilot oracle comparison",
        )
    oracle_rows = {}
    for state_id, row in all_oracle_rows.items():
        row_split = _source_string(
            row.get("split"),
            field=f"oracle row {state_id} split",
        )
        if row_split == split:
            oracle_rows[state_id] = row
    all_decisions = _unique_rows_by_state_id(
        _read_jsonl(source_root / "collection" / "state_selection.jsonl"),
        label="value-pilot state selection",
    )
    decisions = {}
    for state_id, row in all_decisions.items():
        row_split = _source_string(
            row.get("split"),
            field=f"state selection {state_id} split",
        )
        if row_split == split:
            decisions[state_id] = row
    candidates = []
    state_files = sorted(
        (source_root / "collection" / "states" / split).glob("*.json")
    )
    seen_state_files: dict[str, Path] = {}
    for path in state_files:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"value-pilot state file is not an object: {path}")
        payload = dict(value)
        state_id = payload.get("state_id")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError(f"value-pilot state file has invalid state_id: {path}")
        if state_id in seen_state_files:
            raise ValueError(
                "value-pilot state files contain duplicate state_id: "
                f"{state_id}"
            )
        seen_state_files[state_id] = path
        if state_id not in oracle_rows or state_id not in decisions:
            continue
        arms = build_state_arms(payload, oracle_rows[state_id])
        if len(arms) < 2:
            continue
        trials = _source_object_list(payload.get("trials"), field="trials")
        if not trials:
            raise ValueError("value-pilot state has no trials")
        trajectory = trials[0].get("conflict_trajectory")
        if (
            not isinstance(trajectory, list)
            or not trajectory
            or any(not _strict_nonnegative_int(value) for value in trajectory)
        ):
            raise ValueError(
                "value-pilot source conflict_trajectory must be a non-empty "
                "list of non-negative integers"
            )
        initial_conflicts = trajectory[0]
        if initial_conflicts <= 0:
            continue
        decision = dict(decisions[state_id])
        agent_count = decision.get("agent_count")
        if (
            not _strict_nonnegative_int(agent_count)
            or agent_count <= 0
        ):
            raise ValueError(
                f"value-pilot state selection {state_id} agent_count "
                "must be a positive integer"
            )
        for field in (
            "map_id",
            "layout_mode",
            "source_stratum",
            "before_fingerprint",
            "before_repair_fingerprint",
        ):
            _source_string(
                decision.get(field),
                field=f"state selection {state_id} {field}",
            )
        if any(
            agent >= agent_count
            for arm in arms
            for agent in arm["agents"]
        ):
            raise ValueError(
                f"value-pilot state {state_id} has an out-of-range agent"
            )
        candidates.append(
            {
                "state_id": state_id,
                "state_file": str(path),
                "split": split,
                "map_id": decision["map_id"],
                "layout_mode": decision["layout_mode"],
                "agent_count": agent_count,
                "source_stratum": decision["source_stratum"],
                "initial_conflicts": initial_conflicts,
                "before_fingerprint": decision["before_fingerprint"],
                "before_repair_fingerprint": decision[
                    "before_repair_fingerprint"
                ],
                "shared_initial_selection_seconds": (
                    _shared_initial_selection_seconds(payload)
                ),
                "decision": decision,
                "arms": arms,
            }
        )
    selected = _balanced_state_sample(candidates, state_count)
    if len(selected) != state_count:
        raise ValueError(
            f"value pilot could select only {len(selected)}/{state_count} states"
        )
    return {
        "schema": V3_VALUE_PILOT_SCHEMA,
        "source": str(source_root),
        "source_collection_run_fingerprint": source_validation[
            "run_fingerprint"
        ],
        "source_collection_producer_identity_fingerprint": source_validation[
            "producer_identity_fingerprint"
        ],
        "source_sequence_trials_sha256": sha256_file(
            source_root / "collection" / "sequence_trials.jsonl"
        ),
        "oracle_state_comparison": str(oracle_path),
        "oracle_state_comparison_sha256": sha256_file(oracle_path),
        "split": split,
        "requested_state_count": state_count,
        "selected_state_count": len(selected),
        "states": selected,
    }


def _rollout_file_name(state_id: str, arm_id: str, trial_index: int) -> str:
    digest = _fingerprint(
        {
            "state_id": state_id,
            "arm_id": arm_id,
            "trial_index": int(trial_index),
        }
    )[:20]
    return f"{digest}.json"


def _extend_replay_repair_budget(
    replay: dict[str, Any], *, prefix_length: int, max_repairs: int
) -> dict[str, Any]:
    prepared = dict(replay)
    environment = dict(prepared["environment"])
    environment["max_repair_iterations"] = max(
        int(environment.get("max_repair_iterations", 0)),
        int(prefix_length) + int(max_repairs),
    )
    prepared["environment"] = environment
    return prepared


def _agent_ids(
    value: Any,
    *,
    agent_count: int,
    allow_empty: bool = False,
) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(
        isinstance(agent, bool)
        or not isinstance(agent, int)
        or agent < 0
        or agent >= int(agent_count)
        for agent in value
    ):
        return None
    if len(set(value)) != len(value) or (not allow_empty and not value):
        return None
    return list(value)


def _terminal_flags(
    transition: dict[str, Any], state: dict[str, Any]
) -> tuple[bool, bool, bool]:
    if not isinstance(transition.get("terminated"), bool) or not isinstance(
        transition.get("truncated"), bool
    ):
        raise RuntimeError("value pilot native step omitted boolean terminal flags")
    if not isinstance(state.get("done"), bool) or not isinstance(
        state.get("feasible"), bool
    ):
        raise RuntimeError("value pilot observation omitted boolean terminal flags")
    terminated = transition["terminated"]
    truncated = transition["truncated"]
    done = state["done"]
    feasible = state["feasible"]
    if terminated != feasible or truncated != (done and not feasible):
        raise RuntimeError("value pilot native terminal flags disagree with observation")
    if done != (terminated or truncated):
        raise RuntimeError("value pilot native done flag is inconsistent")
    return terminated, truncated, done


def _native_repair_evidence(
    metrics: dict[str, Any],
    *,
    expected_seed: int,
    expected_agents: list[int] | None,
    agent_count: int,
) -> tuple[list[int], list[int], int]:
    if metrics.get("step_applied") is not True:
        raise RuntimeError("value pilot native step ended before applying a repair")
    if not isinstance(metrics.get("replan_success"), bool):
        raise RuntimeError("value pilot native replan-success evidence is not boolean")
    requested = metrics.get("requested_pp_random_seed")
    if not _strict_nonnegative_int(requested) or requested != expected_seed:
        raise RuntimeError("value pilot requested PP seed mismatch")
    neighborhood = _agent_ids(metrics.get("neighborhood"), agent_count=agent_count)
    repair_order = _agent_ids(
        metrics.get("repair_order"),
        agent_count=agent_count,
        allow_empty=True,
    )
    if neighborhood is None or repair_order is None:
        raise RuntimeError("value pilot native neighborhood evidence is invalid")
    if expected_agents is not None and (
        len(neighborhood) != len(expected_agents)
        or set(neighborhood) != set(expected_agents)
    ):
        raise RuntimeError("value pilot first neighborhood differs from action")
    if repair_order and set(repair_order) != set(neighborhood):
        raise RuntimeError("value pilot repair order differs from neighborhood")
    applied = metrics.get("applied_pp_random_seed")
    if isinstance(applied, bool) or not isinstance(applied, int):
        raise RuntimeError("value pilot applied PP seed is invalid")
    if applied != (expected_seed if repair_order else -1):
        raise RuntimeError("value pilot applied PP seed mismatch")
    return neighborhood, repair_order, applied


def run_value_rollout(job: dict[str, Any]) -> dict[str, Any]:
    state_row = dict(job["state"])
    arm = dict(job["arm"])
    trial_index = int(job["trial_index"])
    producer_fingerprint = str(job["producer_identity_fingerprint"])
    if not producer_fingerprint:
        raise ValueError("value rollout producer identity fingerprint is empty")
    decision = dict(state_row["decision"])
    replay, _ = _source_replay_job(decision)
    replay = _extend_replay_repair_budget(
        replay,
        prefix_length=len(decision["prefix_actions"]),
        max_repairs=int(job["max_repairs"]),
    )
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    if state_fingerprint(state) != str(state_row["before_fingerprint"]):
        raise RuntimeError("value pilot replay fingerprint mismatch")
    initial_repair = repair_structure_fingerprint(state)
    if initial_repair != str(state_row["before_repair_fingerprint"]):
        raise RuntimeError("value pilot repair fingerprint mismatch")

    initial_conflicts = int(state["num_of_colliding_pairs"])
    trajectory = [initial_conflicts]
    steps = []
    rollout_started = time.perf_counter()
    total_pp_seconds = 0.0
    total_low_level = collections.Counter()
    stop_reason = "repair_limit"
    max_repairs = int(job["max_repairs"])
    if bool(state.get("done")):
        raise RuntimeError("value pilot source state is already terminal")
    for offset in range(max_repairs):
        before = state
        before_repair = repair_structure_fingerprint(before)
        conflicts_before = int(before["num_of_colliding_pairs"])
        seed = _paired_seed(initial_repair, trial_index, offset + 1)
        if offset == 0:
            action = _paired_repair_action(
                "explicit_neighborhood",
                agents=arm["agents"],
                random_seed=seed,
            )
            route = "explicit_first_action"
        else:
            action = _paired_repair_action("official", random_seed=seed)
            route = "official_adaptive_continuation"
        started = time.perf_counter()
        transition = _plain(environment.step(action))
        repair_seconds = time.perf_counter() - started
        state = dict(transition["observation"])
        metrics = dict(transition["metrics"])
        neighborhood, repair_order, applied_seed = _native_repair_evidence(
            metrics,
            expected_seed=seed,
            expected_agents=(list(map(int, arm["agents"])) if offset == 0 else None),
            agent_count=int(state_row["agent_count"]),
        )
        terminated, truncated, done = _terminal_flags(transition, state)
        conflicts_after = int(state["num_of_colliding_pairs"])
        after_repair = repair_structure_fingerprint(state)
        outcome = classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics.get("replan_success")),
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=terminated,
        )
        if not repair_order and (
            metrics["replan_success"] or outcome != "hard_failure"
        ):
            raise RuntimeError(
                "value pilot empty repair order is only valid for a hard failure"
            )
        low_level = _low_level_delta(before, state)
        for name in ("generated", "expanded", "reopened", "runs"):
            total_low_level[name] += int(low_level.get(name, 0))
        pp_seconds = max(
            0.0, float(metrics.get("pp_replan_seconds", repair_seconds))
        )
        total_pp_seconds += pp_seconds
        trajectory.append(conflicts_after)
        steps.append(
            {
                "step": offset + 1,
                "route": route,
                "action": action,
                "agents": (
                    list(map(int, action["agents"]))
                    if offset == 0
                    else neighborhood
                ),
                "repair_order": repair_order,
                "requested_pp_seed": seed,
                "applied_pp_seed": applied_seed,
                "step_applied": True,
                "terminated": terminated,
                "truncated": truncated,
                "repair_outcome": outcome,
                "conflicts_before": conflicts_before,
                "conflicts_after": conflicts_after,
                "conflict_reduction": conflicts_before - conflicts_after,
                "repair_seconds": repair_seconds,
                "pp_replan_seconds": pp_seconds,
                "low_level": {
                    name: int(low_level.get(name, 0))
                    for name in ("generated", "expanded", "reopened", "runs")
                },
                "after_done": done,
                "after_feasible": terminated,
                "replan_success": bool(metrics.get("replan_success")),
                "before_fingerprint": state_fingerprint(before),
                "after_fingerprint": state_fingerprint(state),
                "before_repair_fingerprint": before_repair,
                "after_repair_fingerprint": after_repair,
            }
        )
        if terminated:
            stop_reason = "feasible"
            break
        if done:
            stop_reason = "environment_terminal"
            break
        if (
            offset + 1 < max_repairs
            and time.perf_counter() - rollout_started
            >= float(job["wall_clock_seconds"])
        ):
            stop_reason = "wall_clock_limit"
            break
    rollout_wall = time.perf_counter() - rollout_started
    conflict_auc = math.fsum(
        float(step["conflicts_before"]) * float(step["repair_seconds"])
        for step in steps
    )
    shared_selection = float(state_row["shared_initial_selection_seconds"])
    result = {
        "schema": V3_VALUE_PILOT_SCHEMA,
        "producer_identity_fingerprint": producer_fingerprint,
        "state_id": str(state_row["state_id"]),
        "split": str(state_row["split"]),
        "map_id": str(state_row["map_id"]),
        "layout_mode": str(state_row["layout_mode"]),
        "agent_count": int(state_row["agent_count"]),
        "source_stratum": str(state_row["source_stratum"]),
        "arm_id": str(arm["arm_id"]),
        "arm_aliases": list(arm["aliases"]),
        "candidate_id": str(arm["candidate_id"]),
        "agents": list(map(int, arm["agents"])),
        "actual_size": len(arm["agents"]),
        "template_key": str(arm.get("template_key", "")),
        "trial_index": trial_index,
        "initial_fingerprint": str(state_row["before_fingerprint"]),
        "initial_repair_fingerprint": initial_repair,
        "final_fingerprint": state_fingerprint(state),
        "final_repair_fingerprint": repair_structure_fingerprint(state),
        "initial_conflicts": initial_conflicts,
        "final_conflicts": int(state["num_of_colliding_pairs"]),
        "conflict_trajectory": trajectory,
        "conflict_reduction": initial_conflicts
        - int(state["num_of_colliding_pairs"]),
        "steps": steps,
        "repair_iterations": len(steps),
        "continuation_iterations": max(0, len(steps) - 1),
        "feasible": bool(state.get("feasible")),
        "censored": not bool(state.get("feasible")),
        "stop_reason": stop_reason,
        "shared_initial_selection_seconds": shared_selection,
        "rollout_wall_seconds": rollout_wall,
        "observed_total_seconds": shared_selection + rollout_wall,
        "pp_replan_seconds": total_pp_seconds,
        "conflict_auc_seconds": conflict_auc,
        "normalized_conflict_auc_seconds": conflict_auc
        / max(1, initial_conflicts),
        "low_level": {
            name: int(total_low_level[name])
            for name in ("generated", "expanded", "reopened", "runs")
        },
        "complete": True,
    }
    validate_value_rollout(
        result,
        state_plan=state_row,
        arm_plan=arm,
        max_repairs=int(job["max_repairs"]),
        wall_clock_seconds=float(job["wall_clock_seconds"]),
        expected_trial_index=trial_index,
        expected_producer_fingerprint=producer_fingerprint,
    )
    return result


def _finite_float(value: Any, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"value rollout {field} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"value rollout {field} must be {qualifier}")
    return result


def _float_matches(actual: Any, expected: float, *, field: str) -> None:
    actual_value = _finite_float(actual, field=field)
    expected_value = _finite_float(expected, field=f"expected {field}")
    tolerance = max(1e-9, 1e-7 * max(abs(expected_value), 1.0))
    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=1e-7,
        abs_tol=tolerance,
    ):
        raise ValueError(f"value rollout {field} mismatch")


def _stored_fingerprint(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"value rollout {field} must be a non-empty string")
    return value


def _stored_string(
    value: Any,
    *,
    field: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"value rollout {field} must be {qualifier}")
    return value


def _stored_string_list(
    value: Any,
    *,
    field: str,
    allow_empty: bool = False,
) -> list[str]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        qualifier = (
            "a list of strings"
            if allow_empty
            else "a non-empty list of non-empty strings"
        )
        raise ValueError(f"value rollout {field} must be {qualifier}")
    return list(value)


def validate_value_rollout(
    row: dict[str, Any],
    *,
    state_plan: dict[str, Any],
    arm_plan: dict[str, Any],
    max_repairs: int,
    wall_clock_seconds: float,
    expected_trial_index: int | None = None,
    expected_producer_fingerprint: str | None = None,
) -> None:
    """Validate a completed value rollout against its immutable job plan."""

    if _stored_string(row.get("schema"), field="schema") != V3_VALUE_PILOT_SCHEMA:
        raise ValueError("value rollout schema mismatch")
    if row.get("complete") is not True:
        raise ValueError("value rollout is incomplete")
    wall_limit = _finite_float(
        wall_clock_seconds,
        field="wall_clock_seconds limit",
        nonnegative=True,
    )
    if (
        not _strict_nonnegative_int(max_repairs)
        or max_repairs <= 0
        or wall_limit <= 0.0
    ):
        raise ValueError("value rollout limits must be positive")
    if expected_producer_fingerprint is not None:
        if (
            not isinstance(expected_producer_fingerprint, str)
            or not expected_producer_fingerprint
        ):
            raise ValueError(
                "value rollout expected producer fingerprint is invalid"
            )
        producer_fingerprint = _stored_fingerprint(
            row.get("producer_identity_fingerprint"),
            field="producer identity fingerprint",
        )
        if producer_fingerprint != expected_producer_fingerprint:
            raise ValueError("value rollout producer identity mismatch")

    state_id = _stored_string(state_plan["state_id"], field="planned state_id")
    arm_id = _stored_string(arm_plan["arm_id"], field="planned arm_id")
    trial_index = row.get("trial_index")
    if not _strict_nonnegative_int(trial_index):
        raise ValueError("value rollout trial index is invalid")
    if _stored_string(row.get("state_id"), field="state_id") != state_id:
        raise ValueError("value rollout state mismatch")
    if _stored_string(row.get("arm_id"), field="arm_id") != arm_id:
        raise ValueError("value rollout arm mismatch")
    if expected_trial_index is not None:
        if (
            not _strict_nonnegative_int(expected_trial_index)
            or trial_index != expected_trial_index
        ):
            raise ValueError("value rollout trial mismatch")
    for field in ("split", "map_id", "layout_mode", "source_stratum"):
        expected_value = _stored_string(
            state_plan[field],
            field=f"planned {field}",
        )
        if _stored_string(row.get(field), field=field) != expected_value:
            raise ValueError(f"value rollout {field} mismatch")
    expected_agent_count = state_plan.get("agent_count")
    if (
        not _strict_nonnegative_int(expected_agent_count)
        or expected_agent_count <= 0
    ):
        raise ValueError("value rollout planned agent_count is invalid")
    if not _strict_nonnegative_int(row.get("agent_count")) or row.get(
        "agent_count"
    ) != expected_agent_count:
        raise ValueError("value rollout agent_count mismatch")
    expected_aliases = _stored_string_list(
        arm_plan["aliases"],
        field="planned arm aliases",
    )
    if (
        _stored_string_list(row.get("arm_aliases"), field="arm aliases")
        != expected_aliases
    ):
        raise ValueError("value rollout arm aliases mismatch")
    expected_candidate_id = _stored_string(
        arm_plan["candidate_id"],
        field="planned candidate_id",
    )
    if (
        _stored_string(row.get("candidate_id"), field="candidate_id")
        != expected_candidate_id
    ):
        raise ValueError("value rollout candidate mismatch")
    expected_agents = _agent_ids(
        arm_plan.get("agents"),
        agent_count=expected_agent_count,
    )
    if expected_agents is None:
        raise ValueError("value rollout planned neighborhood is invalid")
    stored_root_agents = _agent_ids(
        row.get("agents"), agent_count=expected_agent_count
    )
    if stored_root_agents != expected_agents:
        raise ValueError("value rollout neighborhood mismatch")
    if not _strict_nonnegative_int(row.get("actual_size")) or row.get(
        "actual_size"
    ) != len(expected_agents):
        raise ValueError("value rollout actual size mismatch")
    expected_template_key = _stored_string(
        arm_plan.get("template_key", ""),
        field="planned template_key",
        allow_empty=True,
    )
    if (
        _stored_string(
            row.get("template_key"),
            field="template_key",
            allow_empty=True,
        )
        != expected_template_key
    ):
        raise ValueError("value rollout template mismatch")

    initial_full = _stored_fingerprint(
        state_plan["before_fingerprint"],
        field="planned initial fingerprint",
    )
    initial_repair = _stored_fingerprint(
        state_plan["before_repair_fingerprint"],
        field="planned initial repair fingerprint",
    )
    if not _strict_nonnegative_int(state_plan.get("initial_conflicts")):
        raise ValueError("value rollout planned initial conflicts are invalid")
    initial_conflicts = state_plan["initial_conflicts"]
    if _stored_fingerprint(
        row.get("initial_fingerprint"), field="initial fingerprint"
    ) != initial_full:
        raise ValueError("value rollout initial fingerprint mismatch")
    if _stored_fingerprint(
        row.get("initial_repair_fingerprint"),
        field="initial repair fingerprint",
    ) != initial_repair:
        raise ValueError("value rollout initial repair fingerprint mismatch")
    if not _strict_nonnegative_int(row.get("initial_conflicts")) or row.get(
        "initial_conflicts"
    ) != initial_conflicts:
        raise ValueError("value rollout initial conflicts mismatch")

    raw_steps = row.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("value rollout has no repair steps")
    if len(raw_steps) > max_repairs:
        raise ValueError("value rollout exceeds its repair limit")
    steps = []
    for value in raw_steps:
        if not isinstance(value, dict):
            raise ValueError("value rollout step is not an object")
        steps.append(dict(value))

    previous_full = initial_full
    previous_repair = initial_repair
    trajectory = [initial_conflicts]
    repair_seconds_values: list[float] = []
    pp_seconds_values: list[float] = []
    low_level_totals = collections.Counter()
    conflict_auc = 0.0
    for step_index, step in enumerate(steps, start=1):
        if not _strict_nonnegative_int(step.get("step")) or step.get(
            "step"
        ) != step_index:
            raise ValueError("value rollout step indexes are not contiguous")
        if _stored_fingerprint(
            step.get("before_fingerprint"),
            field=f"steps[{step_index}].before fingerprint",
        ) != previous_full:
            raise ValueError("value rollout fingerprint chain mismatch")
        if _stored_fingerprint(
            step.get("before_repair_fingerprint"),
            field=f"steps[{step_index}].before repair fingerprint",
        ) != previous_repair:
            raise ValueError("value rollout repair fingerprint chain mismatch")
        previous_full = _stored_fingerprint(
            step.get("after_fingerprint"),
            field=f"steps[{step_index}].after fingerprint",
        )
        previous_repair = _stored_fingerprint(
            step.get("after_repair_fingerprint"),
            field=f"steps[{step_index}].after repair fingerprint",
        )

        conflicts_before = step.get("conflicts_before")
        conflicts_after = step.get("conflicts_after")
        if (
            not _strict_nonnegative_int(conflicts_before)
            or not _strict_nonnegative_int(conflicts_after)
            or conflicts_before != trajectory[-1]
        ):
            raise ValueError("value rollout conflict trajectory is discontinuous")
        reduction = step.get("conflict_reduction")
        if (
            isinstance(reduction, bool)
            or not isinstance(reduction, int)
            or reduction
            != (
            conflicts_before - conflicts_after
            )
        ):
            raise ValueError("value rollout conflict reduction mismatch")
        trajectory.append(conflicts_after)
        if not isinstance(step.get("after_done"), bool) or not isinstance(
            step.get("after_feasible"), bool
        ):
            raise ValueError("value rollout step lacks terminal evidence")
        after_done = bool(step["after_done"])
        after_feasible = bool(step["after_feasible"])
        if step.get("step_applied") is not True:
            raise ValueError("value rollout contains a non-applied repair")
        if not isinstance(step.get("terminated"), bool) or not isinstance(
            step.get("truncated"), bool
        ):
            raise ValueError("value rollout step lacks native terminal flags")
        terminated = step["terminated"]
        truncated = step["truncated"]
        if terminated != after_feasible or truncated != (
            after_done and not after_feasible
        ):
            raise ValueError("value rollout native terminal evidence mismatch")
        if after_done != (terminated or truncated):
            raise ValueError("value rollout done/terminal evidence mismatch")
        if after_feasible != (conflicts_after == 0):
            raise ValueError("value rollout step feasibility mismatch")
        if after_feasible and not after_done:
            raise ValueError("value rollout feasible step is not terminal")
        if step_index < len(steps) and after_done:
            raise ValueError("value rollout continues after a terminal step")
        step_agents = _agent_ids(
            step.get("agents"), agent_count=expected_agent_count
        )
        repair_order = _agent_ids(
            step.get("repair_order"),
            agent_count=expected_agent_count,
            allow_empty=True,
        )
        if step_agents is None or repair_order is None:
            raise ValueError("value rollout step neighborhood evidence is invalid")
        replan_success = step.get("replan_success")
        if not isinstance(replan_success, bool):
            raise ValueError("value rollout step lacks replan-success evidence")
        outcome = _stored_string(
            step.get("repair_outcome"),
            field=f"steps[{step_index}].repair_outcome",
        )
        if outcome not in REPAIR_OUTCOMES:
            raise ValueError("value rollout repair outcome is invalid")
        expected_outcome = classify_repair_outcome(
            before_fingerprint=_stored_fingerprint(
                step.get("before_repair_fingerprint"),
                field=f"steps[{step_index}].before repair fingerprint",
            ),
            after_fingerprint=previous_repair,
            replan_success=replan_success,
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=after_feasible,
        )
        if outcome != expected_outcome:
            raise ValueError("value rollout repair outcome mismatch")
        if repair_order:
            if set(repair_order) != set(step_agents):
                raise ValueError("value rollout repair order differs from neighborhood")
        elif replan_success or outcome != "hard_failure":
            raise ValueError(
                "value rollout empty repair order is only valid for a hard failure"
            )

        expected_seed = _paired_seed(initial_repair, trial_index, step_index)
        action = step.get("action")
        if not isinstance(action, dict):
            raise ValueError("value rollout step action is missing")
        if (
            not _strict_nonnegative_int(action.get("random_seed"))
            or action.get("random_seed") != expected_seed
            or not _strict_nonnegative_int(action.get("pp_random_seed"))
            or action.get("pp_random_seed") != expected_seed
        ):
            raise ValueError("value rollout PP seed mismatch")
        if (
            not _strict_nonnegative_int(step.get("requested_pp_seed"))
            or step.get("requested_pp_seed") != expected_seed
        ):
            raise ValueError("value rollout requested PP seed mismatch")
        applied_pp_seed = step.get("applied_pp_seed")
        if type(applied_pp_seed) is not int or applied_pp_seed != (
            expected_seed if repair_order else -1
        ):
            raise ValueError("value rollout applied PP seed mismatch")
        if step_index == 1:
            if (
                _stored_string(
                    step.get("route"),
                    field=f"steps[{step_index}].route",
                )
                != "explicit_first_action"
                or _stored_string(
                    action.get("mode"),
                    field=f"steps[{step_index}].action.mode",
                )
                != "explicit_neighborhood"
                or _agent_ids(
                    action.get("agents"),
                    agent_count=expected_agent_count,
                )
                != expected_agents
                or step_agents != expected_agents
            ):
                raise ValueError("value rollout first action mismatch")
        elif (
            _stored_string(
                step.get("route"),
                field=f"steps[{step_index}].route",
            )
            != "official_adaptive_continuation"
            or _stored_string(
                action.get("mode"),
                field=f"steps[{step_index}].action.mode",
            )
            != "official"
        ):
            raise ValueError("value rollout continuation action mismatch")

        repair_seconds = _finite_float(
            step.get("repair_seconds"),
            field=f"steps[{step_index}].repair_seconds",
            nonnegative=True,
        )
        pp_seconds = _finite_float(
            step.get("pp_replan_seconds"),
            field=f"steps[{step_index}].pp_replan_seconds",
            nonnegative=True,
        )
        repair_seconds_values.append(repair_seconds)
        pp_seconds_values.append(pp_seconds)
        conflict_auc += float(conflicts_before) * repair_seconds
        low_level = step.get("low_level")
        if not isinstance(low_level, dict):
            raise ValueError("value rollout step low-level metrics are missing")
        for name in ("generated", "expanded", "reopened", "runs"):
            value = low_level.get(name)
            if not _strict_nonnegative_int(value):
                raise ValueError("value rollout low-level metric is invalid")
            low_level_totals[name] += value

    stored_trajectory = row.get("conflict_trajectory")
    if (
        not isinstance(stored_trajectory, list)
        or any(not _strict_nonnegative_int(value) for value in stored_trajectory)
        or trajectory != stored_trajectory
    ):
        raise ValueError("value rollout stored conflict trajectory mismatch")
    if _stored_fingerprint(
        row.get("final_fingerprint"), field="final fingerprint"
    ) != previous_full:
        raise ValueError("value rollout final fingerprint mismatch")
    if _stored_fingerprint(
        row.get("final_repair_fingerprint"), field="final repair fingerprint"
    ) != previous_repair:
        raise ValueError("value rollout final repair fingerprint mismatch")
    final_conflicts = trajectory[-1]
    if not _strict_nonnegative_int(row.get("final_conflicts")) or row.get(
        "final_conflicts"
    ) != final_conflicts:
        raise ValueError("value rollout final conflicts mismatch")
    total_reduction = row.get("conflict_reduction")
    if (
        isinstance(total_reduction, bool)
        or not isinstance(total_reduction, int)
        or total_reduction != initial_conflicts - final_conflicts
    ):
        raise ValueError("value rollout total conflict reduction mismatch")
    if not _strict_nonnegative_int(row.get("repair_iterations")) or row.get(
        "repair_iterations"
    ) != len(steps):
        raise ValueError("value rollout repair iteration count mismatch")
    if not _strict_nonnegative_int(row.get("continuation_iterations")) or row.get(
        "continuation_iterations"
    ) != max(0, len(steps) - 1):
        raise ValueError("value rollout continuation count mismatch")

    feasible = final_conflicts == 0
    if not isinstance(row.get("feasible"), bool) or not isinstance(
        row.get("censored"), bool
    ):
        raise ValueError("value rollout final labels must be booleans")
    if bool(row.get("feasible")) != feasible:
        raise ValueError("value rollout final feasibility mismatch")
    if bool(row.get("censored")) == feasible:
        raise ValueError("value rollout censoring mismatch")
    stop_reason = _stored_string(row.get("stop_reason"), field="stop_reason")
    terminal = bool(steps[-1]["after_done"])
    if stop_reason == "feasible":
        valid_stop = feasible and terminal
    elif stop_reason == "environment_terminal":
        valid_stop = terminal and not feasible
    elif stop_reason == "wall_clock_limit":
        valid_stop = not terminal and len(steps) < max_repairs
    elif stop_reason == "repair_limit":
        valid_stop = not terminal and len(steps) == max_repairs
    else:
        valid_stop = False
    if not valid_stop:
        raise ValueError("value rollout stop reason mismatch")

    shared_selection = _finite_float(
        state_plan["shared_initial_selection_seconds"],
        field="planned shared selection time",
        nonnegative=True,
    )
    _float_matches(
        row.get("shared_initial_selection_seconds"),
        shared_selection,
        field="shared_initial_selection_seconds",
    )
    rollout_wall = _finite_float(
        row.get("rollout_wall_seconds"),
        field="rollout_wall_seconds",
        nonnegative=True,
    )
    repair_total = math.fsum(repair_seconds_values)
    tolerance = max(1e-9, 1e-7 * max(rollout_wall, 1.0))
    if repair_total > rollout_wall + tolerance:
        raise ValueError("value rollout repair time exceeds rollout wall time")
    if stop_reason == "wall_clock_limit" and rollout_wall + tolerance < wall_limit:
        raise ValueError("value rollout stopped before its wall-clock limit")
    _float_matches(
        row.get("observed_total_seconds"),
        shared_selection + rollout_wall,
        field="observed_total_seconds",
    )
    _float_matches(
        row.get("pp_replan_seconds"),
        math.fsum(pp_seconds_values),
        field="pp_replan_seconds",
    )
    _float_matches(
        row.get("conflict_auc_seconds"),
        conflict_auc,
        field="conflict_auc_seconds",
    )
    _float_matches(
        row.get("normalized_conflict_auc_seconds"),
        conflict_auc / max(1, initial_conflicts),
        field="normalized_conflict_auc_seconds",
    )
    low_level = row.get("low_level")
    if not isinstance(low_level, dict):
        raise ValueError("value rollout low-level summary is missing")
    for name in ("generated", "expanded", "reopened", "runs"):
        if not _strict_nonnegative_int(low_level.get(name)) or low_level.get(
            name
        ) != int(low_level_totals[name]):
            raise ValueError(f"value rollout low-level {name} mismatch")


def load_resumable_value_rollout(
    path: Path,
    *,
    state_plan: dict[str, Any],
    arm_plan: dict[str, Any],
    max_repairs: int,
    wall_clock_seconds: float,
    expected_trial_index: int,
    expected_producer_fingerprint: str,
) -> dict[str, Any] | None:
    """Return a valid completed row; only explicit incomplete rows may rerun."""

    if not path.is_file():
        return None
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("completed value rollout is not an object")
    row = dict(value)
    if row.get("complete") is True:
        validate_value_rollout(
            row,
            state_plan=state_plan,
            arm_plan=arm_plan,
            max_repairs=max_repairs,
            wall_clock_seconds=wall_clock_seconds,
            expected_trial_index=expected_trial_index,
            expected_producer_fingerprint=expected_producer_fingerprint,
        )
        return row
    if row.get("complete") is False:
        return None
    raise ValueError("existing value rollout is neither complete nor resumable")


def _rollout_flat(row: dict[str, Any]) -> dict[str, Any]:
    low_level = dict(row["low_level"])
    return {
        name: value
        for name, value in row.items()
        if name not in {"steps", "low_level", "arm_aliases", "agents", "conflict_trajectory"}
    } | {
        "arm_aliases": ",".join(map(str, row["arm_aliases"])),
        "agents": ",".join(map(str, row["agents"])),
        "conflict_trajectory": ",".join(map(str, row["conflict_trajectory"])),
        **{f"low_level_{name}": value for name, value in low_level.items()},
    }


def _winner_key(row: dict[str, Any], overhead: float) -> tuple[Any, ...]:
    adjusted = float(row["observed_total_seconds"]) + float(overhead) * int(
        row["continuation_iterations"]
    )
    return (
        not bool(row["feasible"]),
        (
            adjusted
            if bool(row["feasible"])
            else float(row["final_conflicts"])
            / max(1.0, float(row["initial_conflicts"]))
        ),
        float(row["normalized_conflict_auc_seconds"]),
        adjusted,
        str(row["arm_id"]),
    )


def analyze_value_rollouts(
    rows: list[dict[str, Any]],
    *,
    expected_jobs: int,
    smoke_only: bool,
    overhead_grid: Iterable[float] = DEFAULT_OVERHEAD_GRID,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) != int(expected_jobs):
        raise ValueError(
            f"value pilot rollout coverage mismatch: {len(rows)}/{expected_jobs}"
        )
    keys = [
        (str(row["state_id"]), str(row["arm_id"]), int(row["trial_index"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("value pilot contains duplicate rollout keys")
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)

    sensitivity_rows = []
    agreement_by_overhead = {}
    for overhead in map(float, overhead_grid):
        winners: dict[tuple[str, int], str] = {}
        for state_id, state_rows in sorted(grouped.items()):
            trials = sorted({int(row["trial_index"]) for row in state_rows})
            for trial in trials:
                subset = [
                    row
                    for row in state_rows
                    if int(row["trial_index"]) == trial
                ]
                winner = min(subset, key=lambda row: _winner_key(row, overhead))
                winners[(state_id, trial)] = str(winner["arm_id"])
                sensitivity_rows.append(
                    {
                        "state_id": state_id,
                        "trial_index": trial,
                        "assumed_continuation_selection_seconds": overhead,
                        "winner_arm_id": str(winner["arm_id"]),
                        "winner_feasible": bool(winner["feasible"]),
                        "winner_final_conflicts": int(winner["final_conflicts"]),
                        "winner_adjusted_seconds": float(
                            winner["observed_total_seconds"]
                        )
                        + overhead * int(winner["continuation_iterations"]),
                    }
                )
        state_agreement = []
        for state_id in sorted(grouped):
            values = [
                arm
                for (candidate_state, _trial), arm in winners.items()
                if candidate_state == state_id
            ]
            state_agreement.append(len(set(values)) == 1)
        agreement_by_overhead[str(overhead)] = statistics.fmean(
            map(float, state_agreement)
        )

    state_diagnostics = []
    for state_id, state_rows in sorted(grouped.items()):
        arm_ids = sorted({str(row["arm_id"]) for row in state_rows})
        feasible_values = {bool(row["feasible"]) for row in state_rows}
        final_ratios = [
            float(row["final_conflicts"])
            / max(1.0, float(row["initial_conflicts"]))
            for row in state_rows
        ]
        auc_values = [
            float(row["normalized_conflict_auc_seconds"]) for row in state_rows
        ]
        feasible_times = [
            float(row["observed_total_seconds"])
            for row in state_rows
            if bool(row["feasible"])
        ]
        time_spread = (
            (max(feasible_times) - min(feasible_times))
            / max(1e-12, min(feasible_times))
            if len(feasible_times) >= 2
            else 0.0
        )
        auc_spread = (
            (max(auc_values) - min(auc_values))
            / max(1e-12, min(auc_values))
            if len(auc_values) >= 2
            else 0.0
        )
        action_sensitive = (
            len(feasible_values) > 1
            or max(final_ratios) - min(final_ratios) >= 0.05
            or time_spread >= 0.10
            or auc_spread >= 0.10
        )
        state_diagnostics.append(
            {
                "state_id": state_id,
                "map_id": str(state_rows[0]["map_id"]),
                "layout_mode": str(state_rows[0]["layout_mode"]),
                "agent_count": int(state_rows[0]["agent_count"]),
                "arm_count": len(arm_ids),
                "trial_count": len(
                    {int(row["trial_index"]) for row in state_rows}
                ),
                "action_sensitive": action_sensitive,
                "feasible_outcome_differs": len(feasible_values) > 1,
                "final_conflict_ratio_spread": max(final_ratios)
                - min(final_ratios),
                "feasible_time_spread_fraction": time_spread,
                "auc_spread_fraction": auc_spread,
            }
        )
    action_sensitive_fraction = statistics.fmean(
        float(row["action_sensitive"]) for row in state_diagnostics
    )
    uncensored_fraction = statistics.fmean(
        float(not bool(row["censored"])) for row in rows
    )
    minimum_agreement = min(agreement_by_overhead.values())
    checks = {
        "coverage_complete": len(rows) == int(expected_jobs),
        "at_least_two_actions_per_state": all(
            int(row["arm_count"]) >= 2 for row in state_diagnostics
        ),
        "action_sensitive_state_fraction_at_least_50pct": (
            action_sensitive_fraction >= 0.50
        ),
        "uncensored_branch_fraction_at_least_50pct": (
            uncensored_fraction >= 0.50
        ),
        "winner_seed_agreement_at_least_50pct": minimum_agreement >= 0.50,
    }
    decision = (
        "smoke_completed_not_scientific"
        if smoke_only
        else (
            "cost_to_go_labels_promising"
            if all(checks.values())
            else "cost_to_go_labels_insufficient"
        )
    )
    report = {
        "schema": V3_VALUE_PILOT_SCHEMA,
        "decision": decision,
        "smoke_only": bool(smoke_only),
        "rollout_count": len(rows),
        "state_count": len(grouped),
        "action_sensitive_state_fraction": action_sensitive_fraction,
        "uncensored_branch_fraction": uncensored_fraction,
        "winner_seed_agreement_by_overhead": agreement_by_overhead,
        "checks": checks,
        "limitations": [
            "The continuation teacher is official Adaptive, so labels estimate Q under that teacher rather than an optimal or on-policy v3 continuation.",
            "Candidate arms were selected from existing v2, model-S3, and retrospective Oracle diagnostics; this pilot tests label signal and is not promotion evidence.",
            "Replay and prefix reconstruction time is excluded from cost-to-go labels.",
            "Censored branches require survival-aware handling before model training.",
        ],
    }
    return report, state_diagnostics, sensitivity_rows


def _report_markdown(report: dict[str, Any]) -> str:
    checks = dict(report["checks"])
    return "\n".join(
        [
            "# Variable-horizon cost-to-go label pilot",
            "",
            f"Decision: `{report['decision']}`",
            "",
            (
                f"- States: {int(report['state_count'])}; rollouts: "
                f"{int(report['rollout_count'])}."
            ),
            (
                "- Action-sensitive state fraction: "
                f"{float(report['action_sensitive_state_fraction']):.3%}."
            ),
            (
                "- Uncensored branch fraction: "
                f"{float(report['uncensored_branch_fraction']):.3%}."
            ),
            "",
            "## Checks",
            "",
            *[
                f"- {name}: `{str(bool(value)).lower()}`"
                for name, value in sorted(checks.items())
            ],
            "",
            "## Boundary",
            "",
            *[f"- {value}" for value in report["limitations"]],
            "",
        ]
    )


def run_value_label_pilot(
    *,
    source: str | Path,
    oracle_state_comparison: str | Path,
    output: str | Path,
    state_count: int,
    trials: int = 2,
    max_repairs: int = 30,
    wall_clock_seconds: float = 60.0,
    split: str = "policy_train",
    smoke_only: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if not _strict_nonnegative_int(state_count) or state_count <= 0:
        raise ValueError("state_count must be a positive integer")
    if not _strict_nonnegative_int(trials) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if not _strict_nonnegative_int(max_repairs) or max_repairs <= 0:
        raise ValueError("max_repairs must be a positive integer")
    if (
        isinstance(wall_clock_seconds, bool)
        or not isinstance(wall_clock_seconds, (int, float))
        or not math.isfinite(float(wall_clock_seconds))
        or float(wall_clock_seconds) <= 0.0
    ):
        raise ValueError("wall_clock_seconds must be a positive finite number")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    if type(smoke_only) is not bool:
        raise ValueError("smoke_only must be boolean")
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    output_root = Path(output).resolve()
    output_has_files = output_root.exists() and any(output_root.iterdir())
    if output_has_files and not bool(resume):
        raise FileExistsError("value pilot output is non-empty; pass resume")
    plan_path = output_root / "plan.json"
    config_path = output_root / "run_config.json"
    if output_has_files and not (plan_path.is_file() and config_path.is_file()):
        raise ValueError(
            "value pilot output lacks a resumable plan/config; use a new output"
        )
    identity = producer_identity(
        project_root=PROJECT_ROOT,
        source_files=V3_VALUE_PILOT_PRODUCER_FILES,
        native_required=True,
        optional_package_names=("numpy", "scikit-learn"),
    )
    identity_fingerprint = _fingerprint(identity)
    requested_plan = build_value_pilot_plan(
        source=source,
        oracle_state_comparison=oracle_state_comparison,
        state_count=int(state_count),
        split=split,
    )
    config = {
        "schema": V3_VALUE_PILOT_SCHEMA,
        "producer_identity": identity,
        "producer_identity_fingerprint": identity_fingerprint,
        "state_count": int(state_count),
        "trials": int(trials),
        "max_repairs": int(max_repairs),
        "wall_clock_seconds": float(wall_clock_seconds),
        "split": str(split),
        "smoke_only": bool(smoke_only),
        "plan_fingerprint": _fingerprint(requested_plan),
    }
    if output_has_files:
        existing_plan_value = read_json(plan_path)
        existing_config_value = read_json(config_path)
        if not isinstance(existing_plan_value, dict):
            raise ValueError("value pilot resume plan is not an object")
        if not isinstance(existing_config_value, dict):
            raise ValueError("value pilot resume configuration is not an object")
        existing_plan = dict(existing_plan_value)
        existing_config = dict(existing_config_value)
        if _fingerprint(existing_plan) != _fingerprint(requested_plan):
            raise ValueError("value pilot resume plan fingerprint mismatch")
        if _fingerprint(existing_config) != _fingerprint(config):
            raise ValueError("value pilot resume configuration mismatch")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(plan_path, requested_plan)
        _write_json(config_path, config)

    rollout_root = output_root / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for state in requested_plan["states"]:
        for arm in state["arms"]:
            for trial_index in range(int(trials)):
                jobs.append(
                    {
                        "state": state,
                        "arm": arm,
                        "trial_index": trial_index,
                        "max_repairs": int(max_repairs),
                        "wall_clock_seconds": float(wall_clock_seconds),
                        "producer_identity_fingerprint": identity_fingerprint,
                    }
                )
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
                    completed.append(row)
                    continue
            row = run_value_rollout(job)
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
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            break
        finally:
            _write_json(
                output_root / "status.json",
                {
                    "schema": V3_VALUE_PILOT_SCHEMA,
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
    report, state_diagnostics, sensitivity = analyze_value_rollouts(
        completed,
        expected_jobs=len(jobs),
        smoke_only=bool(smoke_only),
    )
    report["run_config"] = config
    report["plan_coverage"] = {
        "selected_by_layout": dict(
            collections.Counter(
                str(row["layout_mode"]) for row in requested_plan["states"]
            )
        ),
        "selected_by_agent_count": dict(
            collections.Counter(
                int(row["agent_count"]) for row in requested_plan["states"]
            )
        ),
        "unique_map_count": len(
            {str(row["map_id"]) for row in requested_plan["states"]}
        ),
    }
    _atomic_write_csv(
        output_root / "value_rollouts.csv",
        [_rollout_flat(row) for row in completed],
    )
    _atomic_write_csv(
        output_root / "state_label_diagnostics.csv", state_diagnostics
    )
    _atomic_write_csv(
        output_root / "selection_overhead_sensitivity.csv", sensitivity
    )
    _write_json(output_root / "value_label_pilot_report.json", report)
    (output_root / "value_label_pilot_report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_VALUE_PILOT_SCHEMA,
            "status": "complete",
            "completed_rollout_count": len(completed),
            "total_rollout_count": len(jobs),
            "error_count": 0,
            "decision": str(report["decision"]),
        },
    )
    return report
