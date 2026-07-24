from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
)
from experiments.repair_collection import _make_environment, _plain, state_fingerprint
from experiments.stall_guard import repair_structure_fingerprint


TRACE_REPLAY_CONTRACT = "lns2.trace_replay.pp-seeded-neighborhood.v2"


def recorded_replay_action(event: dict[str, Any]) -> dict[str, Any]:
    """Return an action that reproduces the recorded transition, not its policy.

    Source ``official`` actions are relative to the environment's configured
    destroy strategy.  Replaying them in another environment changes their
    meaning.  The trace already contains the neighborhood and PP order that
    were actually used, so offline replay uses the dedicated native replay
    mode.  That mode also permits a legitimate recorded random no-op whose
    neighborhood did not touch a conflict.
    """

    metrics = event.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("source transition is missing replay metrics")
    neighborhood = metrics.get("neighborhood")
    repair_order = metrics.get("repair_order")
    if not isinstance(neighborhood, list) or not isinstance(repair_order, list):
        raise ValueError("source transition lacks a recorded neighborhood or PP order")
    if not neighborhood:
        raise ValueError("source transition has an empty recorded neighborhood")
    source_action = event.get("action")
    if not isinstance(source_action, dict):
        raise ValueError("source transition is missing its recorded action")
    requested_pp_seed = int(metrics.get("requested_pp_random_seed", -1))
    action_pp_seed = int(source_action.get("pp_random_seed", -1))
    applied_pp_seed = int(metrics.get("applied_pp_random_seed", -1))
    if requested_pp_seed != action_pp_seed:
        raise ValueError("source transition requested PP seed does not match its action")
    if repair_order and applied_pp_seed < 0:
        raise ValueError(
            "source transition ran PP without a deterministic pp_random_seed"
        )
    if repair_order and applied_pp_seed != requested_pp_seed:
        raise ValueError("source transition applied a different PP seed")
    action: dict[str, Any] = {
        "mode": "replay_neighborhood",
        "agents": list(map(int, neighborhood)),
        "repair_order": list(map(int, repair_order)),
    }
    if applied_pp_seed >= 0:
        action["pp_random_seed"] = applied_pp_seed
    return action


def _initial_state(
    collection_root: Path, trace_path: Path, event: dict[str, Any]
) -> dict[str, Any]:
    if str(event.get("schema")) != EPISODE_SCHEMA_V2:
        state = event.get("state")
        if not isinstance(state, dict):
            raise ValueError("source trace is missing its initial state")
        return dict(state)
    state = read_state_blob(
        resolve_state_blob(trace_path, str(event["state_blob"]), collection_root)
    )
    extras = event.get("state_extras")
    if not isinstance(extras, dict):
        raise ValueError("source trace has invalid initial extras")
    state.update(extras)
    return state


def decision_rows(
    collection_root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_path = collection_root / str(manifest["trace_file"])
    events = read_trace_events(trace_path)
    state = _initial_state(collection_root, trace_path, events[0])
    prefix: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for event in events[1:-1]:
        controller = event.get("controller")
        if not isinstance(controller, dict):
            raise ValueError("source transition is missing controller data")
        route = str(controller.get("route", ""))
        if route not in {"model", "official_adaptive"}:
            raise ValueError("source transition is missing a valid route")
        before_fingerprint = state_fingerprint(state)
        before_repair_fingerprint = repair_structure_fingerprint(state)
        if before_fingerprint != str(event.get("before_fingerprint")):
            raise ValueError("source before fingerprint mismatch")
        if str(event.get("schema")) == EPISODE_SCHEMA_V2:
            after = apply_state_delta(state, event["state_delta"])
            after.update(apply_extras_delta(state, event["state_extras_delta"]))
        else:
            after = dict(event["after"])
        after_repair_fingerprint = repair_structure_fingerprint(after)
        actual_metrics = dict(event["metrics"])
        replay_action = recorded_replay_action(event)
        controller_seconds = float(
            controller.get("controller_seconds_before_repair", 0.0)
        )
        repair_seconds = float(event.get("repair_wall_seconds", 0.0))
        rows.append(
            {
                "decision_index": int(event["decision_index"]),
                "route": route,
                "before_fingerprint": before_fingerprint,
                "after_fingerprint": str(event["after_fingerprint"]),
                "before_repair_fingerprint": before_repair_fingerprint,
                "after_repair_fingerprint": after_repair_fingerprint,
                "repair_state_changed": before_repair_fingerprint
                != after_repair_fingerprint,
                "prefix_actions": [dict(action) for action in prefix],
                "replay_action": replay_action,
                "actual_action": dict(event["action"]),
                "actual_metrics": actual_metrics,
                "before_conflicts": int(state["num_of_colliding_pairs"]),
                "actual_lns2": {
                    "source": "main-trace",
                    "action": dict(event["action"]),
                    "metrics": actual_metrics,
                    "after_fingerprint": str(event["after_fingerprint"]),
                    "outcome": {
                        "conflicts_before": int(state["num_of_colliding_pairs"]),
                        "conflicts_after": int(after["num_of_colliding_pairs"]),
                        "conflict_delta": int(state["num_of_colliding_pairs"])
                        - int(after["num_of_colliding_pairs"]),
                        "success": bool(after["feasible"]),
                        "sum_of_costs_delta": int(after["sum_of_costs"])
                        - int(state["sum_of_costs"]),
                        "low_level_delta": dict(event.get("low_level_delta") or {}),
                        "controller_seconds": controller_seconds,
                        "repair_seconds": repair_seconds,
                        "total_decision_seconds": float(
                            controller.get(
                                "total_decision_seconds",
                                controller_seconds + repair_seconds,
                            )
                        ),
                    },
                },
            }
        )
        prefix.append(replay_action)
        state = after
    return rows, events


def replay_prefix(
    job: dict[str, Any], actions: Iterable[dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    destroy_strategy = str(job.get("replay_destroy_strategy", "Adaptive"))
    environment = _make_environment(
        job["dataset_root"], job["row"], job["environment"], destroy_strategy
    )
    state = _plain(environment.reset(seed=int(job["solver_seed"])))
    for action in actions:
        if bool(state["done"]):
            raise RuntimeError("prefix terminated before target state")
        state = _plain(environment.step(dict(action)))["observation"]
    return environment, state
