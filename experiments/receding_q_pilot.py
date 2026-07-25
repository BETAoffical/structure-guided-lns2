from __future__ import annotations

import collections
import concurrent.futures
import csv
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.feature_schema_v3 import V3_FEATURE_NAMES
from experiments.repair_aware import classify_repair_outcome
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
from experiments.v3_s3 import S3_TEMPORAL_FEATURE_NAMES
from experiments.v3_s3_collection import (
    _full_candidate_rows,
    _paired_repair_action,
    _paired_seed,
    _source_replay_job,
)
from experiments.v3_value_pilot import _extend_replay_repair_budget


RECEDING_Q_PILOT_SCHEMA = "lns2.receding_q_label_pilot.v1"
RECEDING_Q_FEATURE_NAMES = (*V3_FEATURE_NAMES, *S3_TEMPORAL_FEATURE_NAMES)
SUPPORTED_CONTINUATION_TEACHERS = ("official_adaptive",)


def _atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({name for row in materialized for name in row})
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def _qualification_index(collection_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((collection_root / "qualification").glob("*.json")):
        payload = dict(read_json(path))
        if not bool(payload.get("complete")):
            continue
        state_id = str(dict(payload["decision"])["state_id"])
        if state_id in result:
            raise ValueError(f"duplicate qualification state: {state_id}")
        result[state_id] = path
    if not result:
        raise ValueError("source collection has no complete qualification files")
    return result


def _v2_first_candidates(collection_root: Path) -> dict[str, str]:
    grouped: dict[str, set[str]] = collections.defaultdict(set)
    for row in _read_jsonl(collection_root / "external_baselines.jsonl"):
        if str(row.get("controller")) != "v2-full":
            continue
        steps = sorted(
            (dict(step) for step in row.get("steps", ())),
            key=lambda step: int(step["step"]),
        )
        if not steps:
            raise ValueError("v2 baseline has no first repair")
        grouped[str(row["state_id"])].add(str(steps[0]["candidate_id"]))
    result = {}
    for state_id, candidate_ids in grouped.items():
        if len(candidate_ids) != 1:
            raise ValueError(
                f"v2 first candidate changes across paired seeds: {state_id}"
            )
        result[state_id] = next(iter(candidate_ids))
    return result


def actual_candidate_arms(
    qualification: dict[str, Any],
    *,
    feature_names: tuple[str, ...] = RECEDING_Q_FEATURE_NAMES,
) -> list[dict[str, Any]]:
    decision = dict(qualification["decision"])
    candidates = [dict(value) for value in qualification["candidates"]]
    rows = [dict(value) for value in qualification["candidate_rows"]]
    if len(candidates) != len(rows):
        raise ValueError("qualification candidate/feature coverage mismatch")

    temporal = {
        str(name): float(value)
        for name, value in dict(decision["temporal_context"]).items()
    }
    result = []
    seen_ids: set[str] = set()
    seen_agents: set[tuple[int, ...]] = set()
    for candidate, row in zip(candidates, rows):
        candidate_id = str(candidate["candidate_id"])
        if str(row["candidate_id"]) != candidate_id:
            raise ValueError("qualification candidate/feature order mismatch")
        agents = tuple(sorted(map(int, candidate["agents"])))
        if candidate_id in seen_ids or agents in seen_agents:
            raise ValueError("qualification contains duplicate actual neighborhoods")
        seen_ids.add(candidate_id)
        seen_agents.add(agents)

        profile = {
            str(name): float(value)
            for name, value in dict(
                dict(row["features"])["realized_dynamic"]
            ).items()
        }
        source = {**profile, **temporal}
        missing = set(feature_names) - set(source)
        if missing:
            raise ValueError(
                f"qualification lacks receding-Q features: {sorted(missing)}"
            )
        values = [float(source[name]) for name in feature_names]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("qualification has non-finite receding-Q features")
        result.append(
            {
                "candidate_id": candidate_id,
                "agents": list(agents),
                "actual_size": len(agents),
                "selection_families": sorted(
                    map(str, candidate.get("selection_families", ()))
                ),
                "feature_values": values,
            }
        )
    return sorted(result, key=lambda row: str(row["candidate_id"]))


def _balanced_state_sample(
    rows: list[dict[str, Any]],
    count: int,
    *,
    smoke_only: bool,
) -> list[dict[str, Any]]:
    if int(count) <= 0:
        raise ValueError("state_count must be positive")
    if smoke_only:
        return sorted(
            rows,
            key=lambda row: (
                int(row["agent_count"]),
                int(row["initial_conflicts"]),
                str(row["map_id"]),
                _fingerprint(str(row["state_id"])),
            ),
        )[: int(count)]

    cells: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    for row in rows:
        cells[(str(row["layout_mode"]), int(row["agent_count"]))].append(row)
    for values in cells.values():
        values.sort(
            key=lambda row: (
                str(row["source_stratum"]),
                -int(row["initial_conflicts"]),
                _fingerprint(str(row["state_id"])),
            )
        )
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < int(count):
        added = False
        for cell in sorted(cells):
            values = cells[cell]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) == int(count):
                    break
        if not added:
            break
        offset += 1
    return selected


def build_receding_q_plan(
    *,
    source: str | Path,
    state_count: int,
    split: str,
    smoke_only: bool,
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    collection_root = source_root / "collection"
    selection_path = collection_root / "state_selection.jsonl"
    decisions = [
        dict(row)
        for row in _read_jsonl(selection_path)
        if str(row["split"]) == str(split)
    ]
    qualifications = _qualification_index(collection_root)
    v2_candidates = _v2_first_candidates(collection_root)
    candidates = []
    for decision in decisions:
        state_id = str(decision["state_id"])
        qualification_path = qualifications.get(state_id)
        if qualification_path is None:
            raise ValueError(f"selected state lacks qualification: {state_id}")
        qualification = dict(read_json(qualification_path))
        arms = actual_candidate_arms(qualification)
        if len(arms) < 2:
            continue
        initial_conflicts = int(decision["before_conflicts"])
        if initial_conflicts <= 0:
            continue
        v2_candidate_id = v2_candidates.get(state_id)
        if v2_candidate_id not in {
            str(arm["candidate_id"]) for arm in arms
        }:
            raise ValueError(f"state lacks its v2 first candidate: {state_id}")
        candidates.append(
            {
                "state_id": state_id,
                "split": str(decision["split"]),
                "map_id": str(decision["map_id"]),
                "layout_mode": str(decision["layout_mode"]),
                "agent_count": int(decision["agent_count"]),
                "source_stratum": str(decision["source_stratum"]),
                "initial_conflicts": initial_conflicts,
                "before_fingerprint": str(decision["before_fingerprint"]),
                "before_repair_fingerprint": str(
                    decision["before_repair_fingerprint"]
                ),
                "root_selection_seconds": max(
                    0.0, float(dict(qualification["timing"])["full_pool_seconds"])
                ),
                "v2_candidate_id": str(v2_candidate_id),
                "decision": decision,
                "qualification_sha256": sha256_file(qualification_path),
                "arms": arms,
            }
        )
    selected = _balanced_state_sample(
        candidates, int(state_count), smoke_only=bool(smoke_only)
    )
    if len(selected) != int(state_count):
        raise ValueError(
            f"receding-Q pilot selected only {len(selected)}/{state_count} states"
        )
    return {
        "schema": RECEDING_Q_PILOT_SCHEMA,
        "source": str(source_root),
        "source_state_selection_sha256": sha256_file(selection_path),
        "source_external_baselines_sha256": sha256_file(
            collection_root / "external_baselines.jsonl"
        ),
        "split": str(split),
        "feature_names": list(RECEDING_Q_FEATURE_NAMES),
        "requested_state_count": int(state_count),
        "selected_state_count": len(selected),
        "states": selected,
    }


def _rollout_file_name(
    state_id: str, candidate_id: str, trial_index: int
) -> str:
    digest = _fingerprint(
        {
            "state_id": str(state_id),
            "candidate_id": str(candidate_id),
            "trial_index": int(trial_index),
        }
    )[:20]
    return f"{digest}.json"


def _validate_pp_seed(metrics: dict[str, Any], expected_seed: int) -> None:
    requested = int(metrics.get("requested_pp_random_seed", -1))
    if requested != int(expected_seed):
        raise RuntimeError(
            f"requested PP seed mismatch: {requested}/{int(expected_seed)}"
        )
    repair_order = list(metrics.get("repair_order", ()))
    applied = int(metrics.get("applied_pp_random_seed", -1))
    if repair_order and applied != int(expected_seed):
        raise RuntimeError(
            f"applied PP seed mismatch: {applied}/{int(expected_seed)}"
        )


def _fixed_horizon_labels(
    *,
    initial_conflicts: int,
    trajectory: list[int],
    horizon: int,
    feasible: bool,
    selection_wall_seconds: float,
    repair_wall_seconds: float,
    conflict_wall_auc_seconds: float,
) -> dict[str, Any]:
    if int(initial_conflicts) <= 0 or int(horizon) <= 0:
        raise ValueError("fixed-horizon labels require positive inputs")
    conflicts = list(map(int, trajectory))
    if not conflicts or conflicts[0] != int(initial_conflicts):
        raise ValueError("fixed-horizon trajectory has invalid initial state")
    if any(value < 0 for value in conflicts):
        raise ValueError("fixed-horizon trajectory has negative conflicts")
    conflicts = conflicts[: int(horizon) + 1]
    while len(conflicts) < int(horizon) + 1:
        conflicts.append(conflicts[-1])
    normalized_step_auc = math.fsum(
        (left + right) / 2.0
        for left, right in zip(conflicts[:-1], conflicts[1:])
    ) / (float(horizon) * float(initial_conflicts))
    total_seconds = max(0.0, float(selection_wall_seconds)) + max(
        0.0, float(repair_wall_seconds)
    )
    wall_auc = max(0.0, float(conflict_wall_auc_seconds))
    return {
        "horizon": int(horizon),
        "padded_steps": max(0, int(horizon) + 1 - len(trajectory)),
        "final_conflicts": int(conflicts[-1]),
        "final_conflict_ratio": float(conflicts[-1])
        / float(initial_conflicts),
        "normalized_step_auc": float(normalized_step_auc),
        "feasible": bool(feasible),
        "no_progress": min(conflicts) >= int(initial_conflicts),
        "observed_total_seconds": total_seconds,
        "log_total_seconds": math.log1p(total_seconds),
        "normalized_wall_auc_seconds": wall_auc / float(initial_conflicts),
    }


def run_receding_q_rollout(job: dict[str, Any]) -> dict[str, Any]:
    state_row = dict(job["state"])
    arm = dict(job["arm"])
    trial_index = int(job["trial_index"])
    horizon = int(job["horizon"])
    teacher = str(job["continuation_teacher"])
    if teacher not in SUPPORTED_CONTINUATION_TEACHERS:
        raise ValueError(f"unsupported continuation teacher: {teacher}")

    decision = dict(state_row["decision"])
    replay, configuration = _source_replay_job(decision)
    replay = _extend_replay_repair_budget(
        replay,
        prefix_length=len(decision["prefix_actions"]),
        max_repairs=horizon,
    )
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    if state_fingerprint(state) != str(state_row["before_fingerprint"]):
        raise RuntimeError("receding-Q replay fingerprint mismatch")
    if repair_structure_fingerprint(state) != str(
        state_row["before_repair_fingerprint"]
    ):
        raise RuntimeError("receding-Q repair fingerprint mismatch")

    initial_repair = repair_structure_fingerprint(state)
    initial_conflicts = int(state["num_of_colliding_pairs"])
    if initial_conflicts != int(state_row["initial_conflicts"]):
        raise RuntimeError("receding-Q initial conflict mismatch")
    trajectory = [initial_conflicts]
    steps = []
    total_repair_wall = 0.0
    total_selection_wall = 0.0
    conflict_wall_auc = 0.0
    total_pp_seconds = 0.0
    low_level_total = collections.Counter()
    stop_reason = "horizon_complete"

    for offset in range(horizon):
        if bool(state.get("feasible")):
            stop_reason = "feasible"
            break
        if bool(state.get("done")):
            stop_reason = "environment_terminal"
            break
        before = state
        before_full = state_fingerprint(before)
        before_repair = repair_structure_fingerprint(before)
        conflicts_before = int(before["num_of_colliding_pairs"])
        seed = _paired_seed(initial_repair, trial_index, offset + 1)
        if offset == 0:
            selection_seconds = float(state_row["root_selection_seconds"])
            shadow_candidate_count = len(state_row.get("arms", ()))
            action = _paired_repair_action(
                "explicit_neighborhood",
                agents=arm["agents"],
                random_seed=seed,
            )
            route = "explicit_root_candidate"
        else:
            before_shadow_full = state_fingerprint(before)
            shadow_candidates, _shadow_rows, shadow_timing = _full_candidate_rows(
                environment,
                before,
                {
                    **decision,
                    "decision_index": int(decision["decision_index"]) + offset,
                },
                dict(configuration["proposal"]),
            )
            if state_fingerprint(before) != before_shadow_full:
                raise RuntimeError(
                    "receding-Q shadow candidate generation changed state"
                )
            selection_seconds = max(
                0.0, float(shadow_timing["full_pool_seconds"])
            )
            shadow_candidate_count = len(shadow_candidates)
            action = _paired_repair_action("official", random_seed=seed)
            route = teacher

        started = time.perf_counter()
        transition = _plain(environment.step(action))
        solver_step_wall = time.perf_counter() - started
        state = dict(transition["observation"])
        metrics = dict(transition["metrics"])
        _validate_pp_seed(metrics, seed)
        conflicts_after = int(state["num_of_colliding_pairs"])
        after_full = state_fingerprint(state)
        after_repair = repair_structure_fingerprint(state)
        outcome = classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics.get("replan_success")),
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=bool(state.get("feasible")),
        )
        low_level = _low_level_delta(before, state)
        for name in ("generated", "expanded", "reopened", "runs"):
            low_level_total[name] += int(low_level.get(name, 0))
        pp_seconds = max(
            0.0, float(metrics.get("pp_replan_seconds", solver_step_wall))
        )
        iteration_wall = time.perf_counter() - started
        total_selection_wall += selection_seconds
        total_repair_wall += iteration_wall
        conflict_wall_auc += float(conflicts_before) * (
            selection_seconds + iteration_wall
        )
        total_pp_seconds += pp_seconds
        trajectory.append(conflicts_after)
        steps.append(
            {
                "step": offset + 1,
                "route": route,
                "candidate_id": (
                    str(arm["candidate_id"])
                    if offset == 0
                    else "official_adaptive"
                ),
                "agents": (
                    list(map(int, arm["agents"]))
                    if offset == 0
                    else list(map(int, metrics.get("neighborhood", ())))
                ),
                "action": action,
                "requested_pp_seed": seed,
                "applied_pp_seed": int(
                    metrics.get("applied_pp_random_seed", -1)
                ),
                "selection_seconds": selection_seconds,
                "shadow_candidate_count": shadow_candidate_count,
                "solver_step_wall_seconds": solver_step_wall,
                "iteration_wall_seconds": iteration_wall,
                "pp_replan_seconds": pp_seconds,
                "conflicts_before": conflicts_before,
                "conflicts_after": conflicts_after,
                "conflict_reduction": conflicts_before - conflicts_after,
                "repair_outcome": outcome,
                "before_fingerprint": before_full,
                "after_fingerprint": after_full,
                "before_repair_fingerprint": before_repair,
                "after_repair_fingerprint": after_repair,
            }
        )
        if bool(state.get("feasible")):
            stop_reason = "feasible"
            break
        if bool(state.get("done")):
            stop_reason = "environment_terminal"
            break

    labels = _fixed_horizon_labels(
        initial_conflicts=initial_conflicts,
        trajectory=trajectory,
        horizon=horizon,
        feasible=bool(state.get("feasible")),
        selection_wall_seconds=total_selection_wall,
        repair_wall_seconds=total_repair_wall,
        conflict_wall_auc_seconds=conflict_wall_auc,
    )
    return {
        "schema": RECEDING_Q_PILOT_SCHEMA,
        "complete": True,
        "state_id": str(state_row["state_id"]),
        "split": str(state_row["split"]),
        "map_id": str(state_row["map_id"]),
        "layout_mode": str(state_row["layout_mode"]),
        "agent_count": int(state_row["agent_count"]),
        "source_stratum": str(state_row["source_stratum"]),
        "candidate_id": str(arm["candidate_id"]),
        "agents": list(map(int, arm["agents"])),
        "actual_size": int(arm["actual_size"]),
        "selection_families": list(map(str, arm["selection_families"])),
        "feature_names": list(RECEDING_Q_FEATURE_NAMES),
        "feature_values": list(map(float, arm["feature_values"])),
        "is_v2_candidate": str(arm["candidate_id"])
        == str(state_row["v2_candidate_id"]),
        "trial_index": trial_index,
        "continuation_teacher": teacher,
        "initial_fingerprint": str(state_row["before_fingerprint"]),
        "initial_repair_fingerprint": initial_repair,
        "final_fingerprint": state_fingerprint(state),
        "final_repair_fingerprint": repair_structure_fingerprint(state),
        "initial_conflicts": initial_conflicts,
        "conflict_trajectory": trajectory,
        "steps": steps,
        "executed_steps": len(steps),
        "stop_reason": stop_reason,
        "root_selection_seconds": float(state_row["root_selection_seconds"]),
        "selection_wall_seconds": total_selection_wall,
        "repair_wall_seconds": total_repair_wall,
        "pp_replan_seconds": total_pp_seconds,
        "low_level": {
            name: int(low_level_total[name])
            for name in ("generated", "expanded", "reopened", "runs")
        },
        **labels,
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        rank = (offset + end - 1) / 2.0
        for index in order[offset:end]:
            result[index] = rank
        offset = end
    return result


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 1.0
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    numerator = math.fsum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank)
    )
    left_scale = math.sqrt(
        math.fsum((value - left_mean) ** 2 for value in left_rank)
    )
    right_scale = math.sqrt(
        math.fsum((value - right_mean) ** 2 for value in right_rank)
    )
    if left_scale <= 0.0 or right_scale <= 0.0:
        return 1.0 if left_rank == right_rank else 0.0
    return numerator / (left_scale * right_scale)


def _winner_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(row["feasible"]),
        float(row["final_conflict_ratio"]),
        float(row["normalized_step_auc"]),
        float(row["normalized_wall_auc_seconds"]),
        float(row["observed_total_seconds"]),
        str(row["candidate_id"]),
    )


def analyze_receding_q_rollouts(
    rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    trials: int,
    horizon: int,
    smoke_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = sum(len(state["arms"]) for state in plan["states"]) * int(trials)
    if len(rows) != expected:
        raise ValueError(
            f"receding-Q rollout coverage mismatch: {len(rows)}/{expected}"
        )
    keys = [
        (str(row["state_id"]), str(row["candidate_id"]), int(row["trial_index"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("receding-Q rollout contains duplicate keys")

    expected_by_state = {
        str(state["state_id"]): {
            str(arm["candidate_id"]) for arm in state["arms"]
        }
        for state in plan["states"]
    }
    plan_states = {
        str(state["state_id"]): dict(state) for state in plan["states"]
    }
    plan_arms = {
        (str(state["state_id"]), str(arm["candidate_id"])): dict(arm)
        for state in plan["states"]
        for arm in state["arms"]
    }
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        state_id = str(row["state_id"])
        candidate_id = str(row["candidate_id"])
        state_plan = plan_states.get(state_id)
        arm_plan = plan_arms.get((state_id, candidate_id))
        if state_plan is None or arm_plan is None:
            raise ValueError("receding-Q rollout is not present in the plan")
        grouped[state_id].append(row)
        if str(row["initial_fingerprint"]) != str(
            state_plan["before_fingerprint"]
        ):
            raise ValueError("receding-Q rollout initial fingerprint mismatch")
        if list(row["feature_names"]) != list(plan["feature_names"]):
            raise ValueError("receding-Q rollout feature schema mismatch")
        if list(map(float, row["feature_values"])) != list(
            map(float, arm_plan["feature_values"])
        ):
            raise ValueError("receding-Q rollout feature vector mismatch")
        if list(map(int, row["agents"])) != list(map(int, arm_plan["agents"])):
            raise ValueError("receding-Q rollout neighborhood mismatch")
        if int(row["executed_steps"]) > int(horizon):
            raise ValueError("receding-Q rollout exceeded its horizon")
        steps = [dict(step) for step in row["steps"]]
        if not steps:
            raise ValueError("receding-Q rollout has no root repair")
        if (
            str(steps[0]["route"]) != "explicit_root_candidate"
            or str(steps[0]["candidate_id"]) != candidate_id
            or list(map(int, steps[0]["agents"]))
            != list(map(int, arm_plan["agents"]))
        ):
            raise ValueError("receding-Q root action mismatch")
        for step in steps[1:]:
            if (
                str(step["route"]) != str(row["continuation_teacher"])
                or int(step["shadow_candidate_count"]) <= 0
            ):
                raise ValueError("receding-Q continuation replanning mismatch")
        previous = str(row["initial_fingerprint"])
        trajectory = [int(row["initial_conflicts"])]
        for step in steps:
            if str(step["before_fingerprint"]) != previous:
                raise ValueError("receding-Q step fingerprint chain mismatch")
            previous = str(step["after_fingerprint"])
            if int(step["conflicts_before"]) != trajectory[-1]:
                raise ValueError("receding-Q conflict trajectory is discontinuous")
            trajectory.append(int(step["conflicts_after"]))
            expected_seed = _paired_seed(
                str(row["initial_repair_fingerprint"]),
                int(row["trial_index"]),
                int(step["step"]),
            )
            if int(step["requested_pp_seed"]) != expected_seed:
                raise ValueError("receding-Q paired PP seed mismatch")
        if trajectory != list(map(int, row["conflict_trajectory"])):
            raise ValueError("receding-Q stored conflict trajectory mismatch")

    state_rows = []
    pair_correlations = []
    winner_agreements = []
    action_sensitive = []
    for state_id, state_rollouts in sorted(grouped.items()):
        candidates = {str(row["candidate_id"]) for row in state_rollouts}
        if candidates != expected_by_state[state_id]:
            raise ValueError(f"receding-Q candidate coverage mismatch: {state_id}")
        for candidate_id in candidates:
            candidate_trials = {
                int(row["trial_index"])
                for row in state_rollouts
                if str(row["candidate_id"]) == candidate_id
            }
            if candidate_trials != set(range(int(trials))):
                raise ValueError(
                    f"receding-Q trial coverage mismatch: {state_id}/{candidate_id}"
                )

        trial_winners = []
        trial_scores: dict[int, dict[str, float]] = {}
        for trial_index in range(int(trials)):
            subset = [
                row
                for row in state_rollouts
                if int(row["trial_index"]) == trial_index
            ]
            winner = min(subset, key=_winner_key)
            trial_winners.append(str(winner["candidate_id"]))
            trial_scores[trial_index] = {
                str(row["candidate_id"]): float(row["normalized_step_auc"])
                for row in subset
            }
        winner_agreement = len(set(trial_winners)) == 1
        if int(trials) >= 2:
            correlations = []
            candidate_order = sorted(candidates)
            for left in range(int(trials)):
                for right in range(left + 1, int(trials)):
                    correlations.append(
                        _correlation(
                            [
                                trial_scores[left][candidate]
                                for candidate in candidate_order
                            ],
                            [
                                trial_scores[right][candidate]
                                for candidate in candidate_order
                            ],
                        )
                    )
            pair_correlation = statistics.fmean(correlations)
            pair_correlations.append(pair_correlation)
            winner_agreements.append(float(winner_agreement))
        else:
            pair_correlation = 1.0

        final_ratios = [
            float(row["final_conflict_ratio"]) for row in state_rollouts
        ]
        step_aucs = [
            float(row["normalized_step_auc"]) for row in state_rollouts
        ]
        sensitive = (
            max(final_ratios) - min(final_ratios) >= 0.02
            or max(step_aucs) - min(step_aucs) >= 0.02
            or len({bool(row["feasible"]) for row in state_rollouts}) > 1
        )
        action_sensitive.append(float(sensitive))
        state_rows.append(
            {
                "state_id": state_id,
                "map_id": str(state_rollouts[0]["map_id"]),
                "layout_mode": str(state_rollouts[0]["layout_mode"]),
                "agent_count": int(state_rollouts[0]["agent_count"]),
                "candidate_count": len(candidates),
                "trial_count": int(trials),
                "action_sensitive": sensitive,
                "paired_seed_rank_correlation": pair_correlation,
                "winner_seed_agreement": winner_agreement,
                "feasible_rollout_fraction": statistics.fmean(
                    float(row["feasible"]) for row in state_rollouts
                ),
                "mean_final_conflict_ratio": statistics.fmean(final_ratios),
                "mean_normalized_step_auc": statistics.fmean(step_aucs),
            }
        )

    state_count = len(grouped)
    map_count = len({str(row["map_id"]) for row in rows})
    action_sensitive_fraction = statistics.fmean(action_sensitive)
    paired_rank_correlation = (
        statistics.fmean(pair_correlations) if pair_correlations else 1.0
    )
    winner_seed_agreement = (
        statistics.fmean(winner_agreements) if winner_agreements else 1.0
    )
    checks = {
        "coverage_complete": len(rows) == expected,
        "all_actual_candidates_covered": all(
            int(row["candidate_count"]) >= 2 for row in state_rows
        ),
        "at_least_12_states": state_count >= 12,
        "at_least_4_maps": map_count >= 4,
        "feature_schema_exact": all(
            list(row["feature_names"]) == list(plan["feature_names"])
            for row in rows
        ),
        "action_sensitive_state_fraction_at_least_50pct": (
            action_sensitive_fraction >= 0.50
        ),
        "paired_seed_rank_correlation_at_least_50pct": (
            paired_rank_correlation >= 0.50
        ),
        "winner_seed_agreement_at_least_50pct": winner_seed_agreement >= 0.50,
    }
    decision = (
        "smoke_completed_not_scientific"
        if smoke_only
        else (
            "fresh_receding_q_labels_promising"
            if all(checks.values())
            else "fresh_receding_q_labels_insufficient"
        )
    )
    report = {
        "schema": RECEDING_Q_PILOT_SCHEMA,
        "decision": decision,
        "smoke_only": bool(smoke_only),
        "state_count": state_count,
        "map_count": map_count,
        "rollout_count": len(rows),
        "candidate_count": len(
            {
                (str(row["state_id"]), str(row["candidate_id"]))
                for row in rows
            }
        ),
        "horizon": int(horizon),
        "continuation_teacher": str(rows[0]["continuation_teacher"]),
        "action_sensitive_state_fraction": action_sensitive_fraction,
        "paired_seed_rank_correlation": paired_rank_correlation,
        "winner_seed_agreement": winner_seed_agreement,
        "checks": checks,
        "limitations": [
            "The first action is every actual neighborhood generated from requested sizes 4/8/16; actual size can be smaller when the generator cannot fill the request.",
            "At each continuation state the complete actual candidate pool and features are regenerated to measure receding-controller selection cost, then official Adaptive supplies the independent teacher action.",
            "These labels estimate a fixed-horizon Q under an Adaptive continuation teacher, not an optimal or on-policy receding-Q controller.",
            "The fixed H=3 label avoids variable-horizon censoring but cannot establish complete-episode performance.",
            "Replay and prefix reconstruction time is excluded; root and continuation full-pool costs plus complete post-step orchestration are included. Root full-pool time is the source qualification measurement shared by every candidate in a state.",
            "This pilot is a label-stability gate and cannot promote or replace v2.",
        ],
    }
    return report, state_rows


def _rollout_flat(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "agents",
        "conflict_trajectory",
        "feature_names",
        "feature_values",
        "low_level",
        "selection_families",
        "steps",
    }
    return {
        name: value for name, value in row.items() if name not in excluded
    } | {
        "agents": ",".join(map(str, row["agents"])),
        "selection_families": ",".join(
            map(str, row["selection_families"])
        ),
        "conflict_trajectory": ",".join(
            map(str, row["conflict_trajectory"])
        ),
        **{
            f"low_level_{name}": int(value)
            for name, value in dict(row["low_level"]).items()
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Fresh receding-Q label pilot",
            "",
            f"Decision: `{report['decision']}`",
            "",
            (
                f"- States: {int(report['state_count'])}; maps: "
                f"{int(report['map_count'])}; actual candidates: "
                f"{int(report['candidate_count'])}; rollouts: "
                f"{int(report['rollout_count'])}."
            ),
            (
                f"- Horizon: {int(report['horizon'])}; continuation teacher: "
                f"`{report['continuation_teacher']}`."
            ),
            (
                "- Action-sensitive states: "
                f"{float(report['action_sensitive_state_fraction']):.3%}."
            ),
            (
                "- Paired-seed candidate rank correlation: "
                f"{float(report['paired_seed_rank_correlation']):.4f}; "
                "exact winner agreement: "
                f"{float(report['winner_seed_agreement']):.3%}."
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


def run_receding_q_label_pilot(
    *,
    source: str | Path,
    output: str | Path,
    state_count: int,
    trials: int = 2,
    horizon: int = 3,
    continuation_teacher: str = "official_adaptive",
    split: str = "policy_train",
    workers: int = 1,
    smoke_only: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if int(trials) <= 0 or int(horizon) <= 0 or int(workers) <= 0:
        raise ValueError("trials, horizon, and workers must be positive")
    if not bool(smoke_only) and int(trials) < 2:
        raise ValueError("scientific receding-Q pilots require at least two trials")
    if continuation_teacher not in SUPPORTED_CONTINUATION_TEACHERS:
        raise ValueError(
            f"unsupported continuation teacher: {continuation_teacher}"
        )
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not bool(resume):
        raise FileExistsError("receding-Q output is non-empty; pass resume")
    output_root.mkdir(parents=True, exist_ok=True)

    requested_plan = build_receding_q_plan(
        source=source,
        state_count=int(state_count),
        split=str(split),
        smoke_only=bool(smoke_only),
    )
    config = {
        "schema": RECEDING_Q_PILOT_SCHEMA,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "source": str(Path(source).resolve()),
        "state_count": int(state_count),
        "trials": int(trials),
        "horizon": int(horizon),
        "continuation_teacher": str(continuation_teacher),
        "split": str(split),
        "workers": int(workers),
        "smoke_only": bool(smoke_only),
        "plan_fingerprint": _fingerprint(requested_plan),
    }
    plan_path = output_root / "plan.json"
    config_path = output_root / "run_config.json"
    if plan_path.is_file():
        existing_plan = dict(read_json(plan_path))
        existing_config = dict(read_json(config_path))
        if _fingerprint(existing_plan) != _fingerprint(requested_plan):
            raise ValueError("receding-Q resume plan fingerprint mismatch")
        if existing_config != config:
            raise ValueError("receding-Q resume configuration mismatch")
    else:
        _write_json(plan_path, requested_plan)
        _write_json(config_path, config)

    rollout_root = output_root / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "state": state,
            "arm": arm,
            "trial_index": trial_index,
            "horizon": int(horizon),
            "continuation_teacher": str(continuation_teacher),
        }
        for state in requested_plan["states"]
        for arm in state["arms"]
        for trial_index in range(int(trials))
    ]
    completed: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    for job in jobs:
        state_id = str(job["state"]["state_id"])
        candidate_id = str(job["arm"]["candidate_id"])
        trial_index = int(job["trial_index"])
        path = rollout_root / _rollout_file_name(
            state_id, candidate_id, trial_index
        )
        if bool(resume) and path.is_file():
            row = dict(read_json(path))
            if (
                str(row.get("state_id")) == state_id
                and str(row.get("candidate_id")) == candidate_id
                and int(row.get("trial_index", -1)) == trial_index
                and bool(row.get("complete"))
            ):
                completed.append(row)
                continue
        pending.append((job, path))

    errors = []
    started = time.perf_counter()

    def record(row: dict[str, Any], path: Path) -> None:
        partial = path.with_name(path.name + ".partial")
        _write_json(partial, row)
        os.replace(partial, path)
        completed.append(row)
        elapsed = max(1e-12, time.perf_counter() - started)
        _write_json(
            output_root / "status.json",
            {
                "schema": RECEDING_Q_PILOT_SCHEMA,
                "status": "running",
                "completed_rollout_count": len(completed),
                "total_rollout_count": len(jobs),
                "error_count": 0,
                "rollouts_per_minute": 60.0 * len(completed) / elapsed,
            },
        )

    try:
        if int(workers) == 1:
            for job, path in pending:
                record(run_receding_q_rollout(job), path)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(workers)
            ) as executor:
                futures = {
                    executor.submit(run_receding_q_rollout, job): path
                    for job, path in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    record(future.result(), futures[future])
    except Exception as error:
        errors.append({"error": f"{type(error).__name__}: {error}"})

    if errors:
        _write_json(output_root / "errors.json", {"errors": errors})
        _write_json(
            output_root / "status.json",
            {
                "schema": RECEDING_Q_PILOT_SCHEMA,
                "status": "error",
                "completed_rollout_count": len(completed),
                "total_rollout_count": len(jobs),
                "error_count": len(errors),
            },
        )
        raise RuntimeError(errors[0]["error"])
    (output_root / "errors.json").unlink(missing_ok=True)
    completed.sort(
        key=lambda row: (
            str(row["state_id"]),
            str(row["candidate_id"]),
            int(row["trial_index"]),
        )
    )
    report, state_rows = analyze_receding_q_rollouts(
        completed,
        plan=requested_plan,
        trials=int(trials),
        horizon=int(horizon),
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
        output_root / "receding_q_rollouts.csv",
        [_rollout_flat(row) for row in completed],
    )
    _atomic_write_csv(
        output_root / "state_label_diagnostics.csv", state_rows
    )
    _write_json(output_root / "receding_q_pilot_report.json", report)
    (output_root / "receding_q_pilot_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": RECEDING_Q_PILOT_SCHEMA,
            "status": "complete",
            "completed_rollout_count": len(completed),
            "total_rollout_count": len(jobs),
            "error_count": 0,
            "decision": str(report["decision"]),
        },
    )
    return report


__all__ = [
    "RECEDING_Q_FEATURE_NAMES",
    "RECEDING_Q_PILOT_SCHEMA",
    "actual_candidate_arms",
    "analyze_receding_q_rollouts",
    "build_receding_q_plan",
    "run_receding_q_label_pilot",
    "run_receding_q_rollout",
]
