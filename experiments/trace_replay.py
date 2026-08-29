from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import contained_file, sha256_file
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
)
from experiments.repair_collection import _make_environment, _plain, state_fingerprint
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


TRACE_REPLAY_CONTRACT = "lns2.trace_replay.pp-seeded-neighborhood.v2"
TARGET_STATE_RESTORE_CONTRACT = "lns2.trace_replay.target-path-restore.v1"
CHECKPOINT_BLOB_RESTORE_CONTRACT = "lns2.trace_replay.checkpoint-blob-restore.v1"


def _required_text(value: Mapping[str, Any], field: str) -> str:
    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"checkpoint {field} must be a non-empty string")
    return raw


def _required_sha256(value: Mapping[str, Any], field: str) -> str:
    digest = _required_text(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"checkpoint {field} must be a lowercase SHA-256 digest")
    return digest


def _required_integer(value: Mapping[str, Any], field: str) -> int:
    raw = value.get(field)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"checkpoint {field} must be an integer")
    return int(raw)


def target_state_from_checkpoint_blob(
    collection_root: Path,
    checkpoint: Mapping[str, Any],
    *,
    expected_map_id: str,
    expected_task_id: str,
    expected_agent_count: int,
) -> tuple[dict[str, Any], Path]:
    """Load and authenticate one frozen repair state for a current job.

    The manifest row is intentionally supplied by the caller.  Its canonical
    identity digest binds that row into the episode override/run fingerprint;
    this generic loader authenticates the contained state blob and rejects a
    checkpoint copied onto a different map, task, or agent-count job.
    """

    if checkpoint.get("source_kind") != "checkpoint_blob_v1":
        raise ValueError("checkpoint source_kind must be checkpoint_blob_v1")
    _required_text(checkpoint, "checkpoint_id")
    _required_sha256(checkpoint, "checkpoint_identity_sha256")

    checkpoint_map_id = _required_text(checkpoint, "map_id")
    checkpoint_task_id = _required_text(checkpoint, "task_id")
    checkpoint_agent_count = _required_integer(checkpoint, "agent_count")
    if checkpoint_map_id != str(expected_map_id):
        raise ValueError("checkpoint map_id does not match the current job")
    if checkpoint_task_id != str(expected_task_id):
        raise ValueError("checkpoint task_id does not match the current job")
    if checkpoint_agent_count != int(expected_agent_count):
        raise ValueError("checkpoint agent_count does not match the current job")

    state_path = contained_file(
        collection_root,
        checkpoint.get("state_blob"),
        field="checkpoint state_blob",
    )
    expected_blob_sha256 = _required_sha256(checkpoint, "state_blob_sha256")
    if sha256_file(state_path) != expected_blob_sha256:
        raise ValueError("checkpoint state blob SHA-256 changed")

    state = read_state_blob(state_path)
    expected_fingerprint = _required_sha256(checkpoint, "expected_fingerprint")
    if state_fingerprint(state) != expected_fingerprint:
        raise ValueError("checkpoint state fingerprint changed")
    expected_repair_fingerprint = _required_sha256(
        checkpoint, "repair_structure_fingerprint"
    )
    if repair_structure_fingerprint(state) != expected_repair_fingerprint:
        raise ValueError("checkpoint repair fingerprint changed")
    expected_conflicts = _required_integer(checkpoint, "expected_conflicts")
    if int(state["num_of_colliding_pairs"]) != expected_conflicts:
        raise ValueError("checkpoint conflict count changed")

    agents = state.get("agents")
    if not isinstance(agents, list) or len(agents) != checkpoint_agent_count:
        raise ValueError("checkpoint state agent_count changed")
    return state, state_path


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
    legacy_unseeded_explicit = (
        repair_order
        and requested_pp_seed < 0
        and action_pp_seed < 0
        and applied_pp_seed < 0
        and str(source_action.get("mode")) == "explicit_neighborhood"
    )
    if legacy_unseeded_explicit:
        source_agents = source_action.get("agents")
        source_random_seed = int(source_action.get("random_seed", -1))
        requested_random_seed = int(metrics.get("requested_random_seed", -1))
        if (
            not isinstance(source_agents, list)
            or list(map(int, source_agents)) != list(map(int, neighborhood))
            or sorted(map(int, repair_order)) != sorted(map(int, neighborhood))
            or source_random_seed < 0
            or source_random_seed != requested_random_seed
        ):
            raise ValueError(
                "legacy unseeded explicit transition lacks deterministic source evidence"
            )
        # Historical v2 explicit actions seeded the process RNG before PP and
        # then let PP generate the recorded order. Replaying the exact source
        # action reproduces both that shuffle and the following low-level RNG
        # position. Supplying the recorded order directly would skip the
        # shuffle and therefore change the low-level search stream.
        return {
            "mode": "explicit_neighborhood",
            "agents": list(map(int, source_agents)),
            "random_seed": source_random_seed,
        }
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
    trace_path = contained_file(
        collection_root,
        manifest.get("trace_file"),
        field="trace_file",
    )
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


def result_blind_decision_rows(
    collection_root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return pre-action states whose recorded prefix is deterministic.

    This deliberately omits the source action outcome and after-state metrics.
    A decision itself remains eligible when its incoming prefix is replayable;
    if that decision used an older unseeded PP action, later decisions are
    excluded because their prefixes cannot be reproduced exactly.
    """

    trace_path = contained_file(
        collection_root,
        manifest.get("trace_file"),
        field="trace_file",
    )
    events = read_trace_events(trace_path)
    state = _initial_state(collection_root, trace_path, events[0])
    prefix: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for event in events[1:-1]:
        before_fingerprint = state_fingerprint(state)
        if before_fingerprint != str(event.get("before_fingerprint")):
            raise ValueError("source before fingerprint mismatch")
        controller = event.get("controller")
        if not isinstance(controller, dict) or str(controller.get("route", "")) not in {
            "model",
            "official_adaptive",
        }:
            raise ValueError("source transition is missing a valid route")
        rows.append(
            {
                "decision_index": int(event["decision_index"]),
                "before_fingerprint": before_fingerprint,
                "before_conflicts": int(state["num_of_colliding_pairs"]),
                "prefix_actions": [dict(action) for action in prefix],
            }
        )
        try:
            replay_action = recorded_replay_action(event)
        except ValueError:
            break
        if str(event.get("schema")) == EPISODE_SCHEMA_V2:
            after = apply_state_delta(state, event["state_delta"])
            after.update(apply_extras_delta(state, event["state_extras_delta"]))
        else:
            after = dict(event["after"])
        prefix.append(replay_action)
        state = after
    return rows, events


def state_before_decision(
    initial_state: dict[str, Any],
    transition_events: Iterable[dict[str, Any]],
    *,
    decision_index: int,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Reconstruct one recorded pre-action state without reading its outcome.

    Only deltas from decisions strictly before ``decision_index`` are applied.
    The target transition's action, metrics, repair outcome, and after-state
    delta are deliberately not inspected.
    """

    target = int(decision_index)
    state = dict(initial_state)
    for event in transition_events:
        current = int(event["decision_index"])
        before_fingerprint = state_fingerprint(state)
        if before_fingerprint != str(event.get("before_fingerprint")):
            raise ValueError("source before fingerprint mismatch")
        if current == target:
            if (
                expected_fingerprint is not None
                and before_fingerprint != str(expected_fingerprint)
            ):
                raise ValueError("selected target fingerprint mismatch")
            return state
        if current > target:
            break
        if str(event.get("schema")) == EPISODE_SCHEMA_V2:
            after = apply_state_delta(state, event["state_delta"])
            after.update(apply_extras_delta(state, event["state_extras_delta"]))
        else:
            after = dict(event["after"])
        state = after
    raise ValueError(f"source trace does not contain decision {target}")


def target_state_from_trace(
    collection_root: Path,
    manifest: dict[str, Any],
    *,
    decision_index: int,
    expected_fingerprint: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load the exact stored state immediately before a selected decision."""

    trace_path = contained_file(
        collection_root,
        manifest.get("trace_file"),
        field="trace_file",
    )
    events = read_trace_events(trace_path)
    if len(events) < 2:
        raise ValueError("source trace has no decision transitions")
    initial = _initial_state(collection_root, trace_path, events[0])
    state = state_before_decision(
        initial,
        events[1:-1],
        decision_index=decision_index,
        expected_fingerprint=expected_fingerprint,
    )
    return state, trace_path


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


def restore_repair_state(
    job: dict[str, Any], source_state: dict[str, Any], *, seed: int
) -> tuple[Any, dict[str, Any]]:
    """Create an independent native branch from recorded repair paths.

    ``reset_paths`` intentionally resets counters and the wall clock, so full
    state fingerprints need not match.  The repair-structure fingerprint must
    match exactly before a branch is allowed to run.
    """

    agents = sorted(source_state.get("agents", []), key=lambda row: int(row["id"]))
    if [int(agent["id"]) for agent in agents] != list(range(len(agents))):
        raise ValueError("source repair state has non-contiguous agent ids")
    paths = [list(map(int, agent.get("path", []))) for agent in agents]
    if not paths or any(not path for path in paths):
        raise ValueError("source repair state has an empty agent path")
    environment = _make_environment(
        job["dataset_root"],
        job["row"],
        job["environment"],
        str(job.get("replay_destroy_strategy", "Adaptive")),
    )
    restored = _plain(environment.reset_paths(paths, seed=int(seed)))
    expected = repair_structure_fingerprint(source_state)
    if repair_structure_fingerprint(restored) != expected:
        raise RuntimeError("restored native repair structure differs from source")
    return environment, restored
