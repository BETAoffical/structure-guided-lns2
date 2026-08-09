from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from experiments._common import producer_identity, registered_input, sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_closurepool_longtail import reconstruct_trace
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import restore_repair_state, target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


PREFLIGHT_CONFIG_SCHEMA = "lns2.stride.maze_tail_action_replay_preflight_config.v1"
PREFLIGHT_REPORT_SCHEMA = "lns2.stride.maze_tail_action_replay_preflight_report.v1"
SELECTION_SCHEMA = "lns2.stride.maze_tail_action_replay_selection.v1"
REPLAY_CONFIG_SCHEMA = "lns2.stride.maze_tail_action_replay_config.v1"
STATE_SCHEMA = "lns2.stride.maze_tail_action_replay_state.v1"
TRIAL_SCHEMA = "lns2.stride.maze_tail_action_replay_trial.v1"
STATUS_SCHEMA = "lns2.stride.maze_tail_action_replay_status.v1"
REPORT_SCHEMA = "lns2.stride.maze_tail_action_replay_report.v1"
EXPERIMENT_ID = "stride-maze-tail-action-replay-v1"
CONTROLLERS = ("v2-full", "v2-plus-structpool", "v2-plus-slotpool")
CHALLENGERS = CONTROLLERS[1:]
PREFLIGHT_REPORT_FILENAME = "action_replay_preflight_report.json"
SELECTION_FILENAME = "state_selection.jsonl"
REPORT_FILENAME = "action_replay_report.json"

PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/closed_loop_trace_storage.py",
    "experiments/repair_collection.py",
    "experiments/stride_closurepool_longtail.py",
    "experiments/stride_collection.py",
    "experiments/stride_maze_tail_action_replay.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _registered(root: Path, specification: dict[str, Any], *, label: str) -> Path:
    return registered_input(root, specification, label=label)


def validate_preflight_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != PREFLIGHT_CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_all_comparison_first_divergence_extraction_no_repair_trials"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("Maze tail action-replay preflight identity changed")
    if tuple(map(str, config.get("controllers") or ())) != CONTROLLERS:
        raise ValueError("Maze tail action-replay controllers changed")
    expected = dict(config.get("expected_source") or {})
    if expected != {
        "paired_key_count": 33,
        "comparison_count": 66,
        "tail_evidence_passed": True,
    }:
        raise ValueError("Maze tail action-replay source scope changed")
    selection = dict(config.get("selection_contract") or {})
    if selection != {
        "all_66_comparisons_required": True,
        "all_and_only_first_realized_neighborhood_divergences": True,
        "identical_prefix_state_and_paired_pp_seed_required": True,
        "first_challenger_action_must_be_structural": True,
        "v2_action_must_exist_in_challenger_base_pool": True,
        "outcome_based_selection_forbidden": True,
        "no_divergence_rows_retained_in_preflight_report": True,
    }:
        raise ValueError("Maze tail first-divergence selection contract changed")
    replay = dict(config.get("replay_contract") or {})
    if (
        tuple(map(int, replay.get("trial_indices") or ())) != tuple(range(16))
        or replay.get("strictly_paired_pp_seed_per_state_and_index") is not True
        or replay.get("primary_outcome")
        != "current_step_conflict_reduction_normalized_by_before_conflicts"
        or replay.get("runtime_used_in_label") is not False
        or replay.get("future_trajectory_read") is not False
    ):
        raise ValueError("Maze tail action-replay trial contract changed")
    gates = dict(config.get("preflight_gates") or {})
    required_gates = {
        "minimum_divergence_comparison_count": 24,
        "minimum_tail_divergence_count": 12,
        "minimum_severe_tail_divergence_count": 4,
        "minimum_tail_divergence_map_count": 2,
        "minimum_tail_divergence_task_count": 6,
        "minimum_tail_divergence_solver_seed_count": 2,
        "minimum_tail_divergence_per_challenger": 3,
    }
    if gates != required_gates:
        raise ValueError("Maze tail action-replay preflight gates changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "state_condition_diagnostic_only": True,
        "model_training_allowed": False,
        "threshold_tuning_allowed": False,
        "formal_ttf_claim": False,
        "default_promotion_allowed": False,
        "predictor_design_before_replay_passes": False,
    }:
        raise ValueError("Maze tail action-replay claim boundary changed")
    if set(config.get("inputs") or {}) != {
        "state_collection_report",
        "v2_manifest",
        "structpool_manifest",
        "slotpool_manifest",
    }:
        raise ValueError("Maze tail action-replay input registry changed")


def _manifest_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = (str(row["task_id"]), int(row["solver_seed"]))
        if key in result:
            raise ValueError(f"duplicate source manifest row: {key}")
        if row.get("status") != "ok" or row.get("error") is not None:
            raise ValueError(f"non-ok source manifest row: {key}")
        result[key] = row
    return result


def _action_candidate(event: dict[str, Any]) -> dict[str, Any]:
    controller = dict(event.get("controller") or {})
    selected_id = str(controller.get("selected_candidate_id") or "")
    pool = list(controller.get("candidate_pool") or ())
    matches = [row for row in pool if str(row.get("candidate_id")) == selected_id]
    if len(matches) != 1:
        raise ValueError("selected action is absent from candidate pool")
    candidate = dict(matches[0])
    metrics = dict(event.get("metrics") or {})
    raw_agents = metrics.get("neighborhood")
    if not isinstance(raw_agents, list) or not raw_agents:
        raw_agents = dict(event.get("action") or {}).get("agents")
    agents = sorted(map(int, raw_agents or ()))
    if not agents or len(agents) != len(set(agents)):
        raise ValueError("selected action neighborhood is empty or duplicated")
    if sorted(map(int, candidate.get("agents") or ())) != agents:
        raise ValueError("selected candidate and realized neighborhood differ")
    return {
        "candidate_id": selected_id,
        "agents": agents,
        "actual_size": len(agents),
        "selection_families": sorted(map(str, candidate.get("selection_families") or ())),
        "structural_groups": sorted(
            map(str, candidate.get("structpool_family_groups") or ())
        ),
    }


def _pp_seed(event: dict[str, Any]) -> int:
    value = int(dict(event.get("action") or {}).get("pp_random_seed", -1))
    if value < 0:
        raise ValueError("transition has no paired PP seed")
    return value


def _first_divergence(
    v2_trace: dict[str, Any], challenger_trace: dict[str, Any]
) -> dict[str, Any] | None:
    v2_states = list(v2_trace["states"])
    challenger_states = list(challenger_trace["states"])
    v2_transitions = list(v2_trace["transitions"])
    challenger_transitions = list(challenger_trace["transitions"])
    if state_fingerprint(v2_states[0]) != state_fingerprint(challenger_states[0]):
        raise ValueError("three-controller initial states differ")
    shared = min(len(v2_transitions), len(challenger_transitions))
    for index in range(shared):
        before = v2_states[index]
        other_before = challenger_states[index]
        before_fingerprint = state_fingerprint(before)
        if before_fingerprint != state_fingerprint(other_before):
            raise ValueError("state diverged before first different action")
        v2_event = v2_transitions[index]
        challenger_event = challenger_transitions[index]
        v2_action = _action_candidate(v2_event)
        challenger_action = _action_candidate(challenger_event)
        if v2_action["agents"] == challenger_action["agents"]:
            if (
                _pp_seed(v2_event) != _pp_seed(challenger_event)
                or state_fingerprint(v2_states[index + 1])
                != state_fingerprint(challenger_states[index + 1])
            ):
                raise ValueError("identical prefix action did not replay identically")
            continue
        if not challenger_action["structural_groups"]:
            raise ValueError("first challenger divergence is not structural")
        pool = list(dict(challenger_event.get("controller") or {}).get("candidate_pool") or ())
        base_ids = {
            str(row.get("candidate_id"))
            for row in pool
            if not row.get("structpool_family_groups")
        }
        if v2_action["candidate_id"] not in base_ids:
            raise ValueError("V2 action is absent from challenger base pool")
        proposal = dict(dict(challenger_event.get("controller") or {}).get("proposal") or {})
        anchor_id = proposal.get("slotpool_v2_anchor_candidate_id")
        if anchor_id is not None and str(anchor_id) != v2_action["candidate_id"]:
            raise ValueError("logged SlotPool anchor differs from exact V2 action")
        if _pp_seed(v2_event) != _pp_seed(challenger_event):
            raise ValueError("source divergence PP seeds are not paired")
        return {
            "decision_index": index,
            "before_fingerprint": before_fingerprint,
            "before_conflicts": int(before["num_of_colliding_pairs"]),
            "source_paired_pp_seed": _pp_seed(v2_event),
            "v2_action": {"role": "v2_action", **v2_action},
            "challenger_action": {"role": "challenger_action", **challenger_action},
        }
    return None


def prepare_action_replay(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = _read_json(config_path)
    validate_preflight_config(config)
    inputs = {
        name: _registered(root, dict(spec), label="Maze tail action replay")
        for name, spec in dict(config["inputs"]).items()
    }
    source_report = _read_json(inputs["state_collection_report"])
    if (
        source_report.get("integrity_passed") is not True
        or source_report.get("tail_evidence_passed") is not True
        or int(source_report.get("paired_key_count", 0)) != 33
        or int(source_report.get("comparison_count", 0)) != 66
        or source_report.get("training_allowed") is not False
    ):
        raise ValueError("Maze tail source report is not eligible for action replay")
    manifest_paths = {
        "v2-full": inputs["v2_manifest"],
        "v2-plus-structpool": inputs["structpool_manifest"],
        "v2-plus-slotpool": inputs["slotpool_manifest"],
    }
    manifests = {name: _manifest_index(path) for name, path in manifest_paths.items()}
    if any(len(rows) != 33 for rows in manifests.values()):
        raise ValueError("Maze tail action-replay source coverage changed")
    tail_rows = list(source_report.get("tail_comparisons") or ())
    tail_index = {
        (str(row["task_id"]), int(row["solver_seed"]), str(row["challenger"])): row
        for row in tail_rows
    }
    if len(tail_index) != 66:
        raise ValueError("Maze tail comparison index changed")
    selections: list[dict[str, Any]] = []
    no_divergence: list[dict[str, Any]] = []
    errors: list[str] = []
    processed = 0
    trace_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def trace(controller: str, key: tuple[str, int]) -> dict[str, Any]:
        cache_key = (controller, *key)
        if cache_key not in trace_cache:
            trace_cache[cache_key] = reconstruct_trace(
                manifest_paths[controller].parent, manifests[controller][key]
            )
        return trace_cache[cache_key]

    for task_id, solver_seed, challenger in sorted(tail_index):
        processed += 1
        key = (task_id, solver_seed)
        comparison = tail_index[(task_id, solver_seed, challenger)]
        try:
            divergence = _first_divergence(trace("v2-full", key), trace(challenger, key))
            common = {
                "task_id": task_id,
                "solver_seed": solver_seed,
                "challenger": challenger,
                "map_id": str(comparison["map_id"]),
                "agent_count": int(comparison["agent_count"]),
                "initial_conflicts": int(comparison["initial_conflicts"]),
                "conflict_band": str(comparison["conflict_band"]),
                "task_variant_family": str(comparison["task_variant_family"]),
                "tail_category": str(comparison["tail_category"]),
            }
            if divergence is None:
                no_divergence.append(common)
                continue
            state_id = (
                f"{task_id}::solver_seed_{solver_seed:04d}::{challenger}"
                f"::decision_{int(divergence['decision_index']):04d}"
            )
            selections.append(
                {
                    "schema": SELECTION_SCHEMA,
                    "state_id": state_id,
                    **common,
                    **divergence,
                    "v2_source_root": str(manifest_paths["v2-full"].parent),
                    "challenger_source_root": str(manifest_paths[challenger].parent),
                    "v2_manifest": manifests["v2-full"][key],
                    "challenger_manifest": manifests[challenger][key],
                    "source_trace_sha256": {
                        "v2": str(manifests["v2-full"][key]["trace_sha256"]),
                        "challenger": str(manifests[challenger][key]["trace_sha256"]),
                    },
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{task_id}/seed-{solver_seed}/{challenger}: {error}")
    selections.sort(key=lambda row: str(row["state_id"]))
    tail = [row for row in selections if row["tail_category"] in {"adverse", "severe"}]
    severe = [row for row in selections if row["tail_category"] == "severe"]
    gates = dict(config["preflight_gates"])
    checks = {
        "all_comparisons_processed": processed == 66 and processed == len(tail_index),
        "zero_extraction_errors": not errors,
        "minimum_divergence_comparison_count": len(selections)
        >= int(gates["minimum_divergence_comparison_count"]),
        "minimum_tail_divergence_count": len(tail)
        >= int(gates["minimum_tail_divergence_count"]),
        "minimum_severe_tail_divergence_count": len(severe)
        >= int(gates["minimum_severe_tail_divergence_count"]),
        "minimum_tail_divergence_map_count": len({row["map_id"] for row in tail})
        >= int(gates["minimum_tail_divergence_map_count"]),
        "minimum_tail_divergence_task_count": len({row["task_id"] for row in tail})
        >= int(gates["minimum_tail_divergence_task_count"]),
        "minimum_tail_divergence_solver_seed_count": len(
            {int(row["solver_seed"]) for row in tail}
        )
        >= int(gates["minimum_tail_divergence_solver_seed_count"]),
        "minimum_tail_divergence_per_challenger": all(
            sum(row["challenger"] == challenger for row in tail)
            >= int(gates["minimum_tail_divergence_per_challenger"])
            for challenger in CHALLENGERS
        ),
        "selection_not_filtered_by_replay_outcome": True,
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / SELECTION_FILENAME
    _write_jsonl(selection_path, selections)
    report = {
        "schema": PREFLIGHT_REPORT_SCHEMA,
        "scientific_status": "first_divergence_actions_frozen_before_repair_trials",
        "experiment_id": EXPERIMENT_ID,
        "preflight_passed": all(checks.values()),
        "errors": errors,
        "comparison_count": processed,
        "divergence_count": len(selections),
        "no_divergence_count": len(no_divergence),
        "tail_divergence_count": len(tail),
        "severe_tail_divergence_count": len(severe),
        "divergence_count_by_challenger": dict(
            sorted(Counter(row["challenger"] for row in selections).items())
        ),
        "tail_divergence_count_by_challenger": dict(
            sorted(Counter(row["challenger"] for row in tail).items())
        ),
        "tail_divergence_map_count": len({row["map_id"] for row in tail}),
        "tail_divergence_task_count": len({row["task_id"] for row in tail}),
        "tail_divergence_solver_seed_count": len(
            {int(row["solver_seed"]) for row in tail}
        ),
        "no_divergence_rows": no_divergence,
        "integrity_gates": checks,
        "claim_boundary": dict(config["claim_boundary"]),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "state_collection_report_sha256": sha256_file(
                inputs["state_collection_report"]
            ),
            "manifest_sha256": {
                controller: sha256_file(path)
                for controller, path in manifest_paths.items()
            },
            "state_selection_sha256": sha256_file(selection_path),
        },
        "next_step": (
            "freeze_selection_hash_and_collect_sixteen_paired_pp_seeds"
            if all(checks.values())
            else "stop_before_repair_trials_and_register_second_independent_block"
        ),
        "training_allowed": False,
    }
    _write_json(output / PREFLIGHT_REPORT_FILENAME, report)
    return report


def validate_replay_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != REPLAY_CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_first_divergence_sixteen_seed_paired_action_replay"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("Maze tail action-replay identity changed")
    if tuple(map(int, config.get("trial_indices") or ())) != tuple(range(16)):
        raise ValueError("Maze tail action-replay trial indices changed")
    if (
        config.get("strictly_paired_pp_seeds_within_state_and_index") is not True
        or config.get("candidate_roles") != ["v2_action", "challenger_action"]
        or config.get("primary_outcome")
        != "current_step_conflict_reduction_normalized_by_before_conflicts"
        or config.get("runtime_used_in_label") is not False
        or config.get("future_trajectory_read") is not False
        or config.get("outcome_based_state_filtering") is not False
    ):
        raise ValueError("Maze tail action-replay evidence contract changed")
    robust = dict(config.get("robust_order_rule") or {})
    if robust != {
        "minimum_paired_win_fraction": 0.75,
        "minimum_absolute_mean_effect": 0.02,
        "require_both_half_mean_directions": True,
        "require_winner_no_progress_rate_not_worse": True,
        "first_fixed_half": list(range(8)),
        "second_fixed_half": list(range(8, 16)),
        "tie_epsilon": 1e-12,
    }:
        raise ValueError("Maze tail action-replay robust-order rule changed")
    gates = dict(config.get("analysis_gates") or {})
    if gates != {
        "minimum_robustly_ordered_fraction": 0.70,
        "minimum_robustly_ordered_tail_count": 8,
        "minimum_robust_tail_map_count": 2,
        "minimum_robust_tail_task_count": 4,
        "minimum_robust_tail_solver_seed_count": 2,
        "minimum_robust_tail_per_challenger": 2,
    }:
        raise ValueError("Maze tail action-replay analysis gates changed")
    if set(config.get("inputs") or {}) != {
        "preflight_report",
        "state_selection",
    }:
        raise ValueError("Maze tail action-replay frozen input registry changed")
    if int(config.get("workers", 0)) != 1 or float(
        config.get("state_timeout_seconds", 0.0)
    ) != 1800.0:
        raise ValueError("Maze tail action-replay execution contract changed")


def _load_replay_config(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], Path, Path, dict[str, Any], list[dict[str, Any]]]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    validate_replay_config(config)
    preflight_path = _registered(
        root, dict(config["inputs"]["preflight_report"]), label="Maze tail replay"
    )
    selection_path = _registered(
        root, dict(config["inputs"]["state_selection"]), label="Maze tail replay"
    )
    preflight = _read_json(preflight_path)
    selections = _read_jsonl(selection_path)
    if (
        preflight.get("schema") != PREFLIGHT_REPORT_SCHEMA
        or preflight.get("preflight_passed") is not True
        or preflight.get("training_allowed") is not False
        or int(preflight.get("divergence_count", -1)) != len(selections)
        or str(preflight["inputs"]["state_selection_sha256"])
        != sha256_file(selection_path)
        or int(config.get("expected_state_count", -1)) != len(selections)
    ):
        raise ValueError("Maze tail action-replay frozen preflight changed")
    if len({str(row.get("state_id")) for row in selections}) != len(selections):
        raise ValueError("Maze tail action-replay state IDs are not unique")
    return path, root, config, preflight_path, selection_path, preflight, selections


def _replay_job(selection: dict[str, Any]) -> dict[str, Any]:
    source_root = Path(str(selection["v2_source_root"])).resolve()
    run = _read_json(source_root / "run_config.json")
    dataset_root = Path(str(run["dataset"])).resolve()
    matches = [
        row
        for row in _load_dataset_rows(
            dataset_root, [str(selection["v2_manifest"]["split"])]
        )
        if str(row["task_id"]) == str(selection["task_id"])
    ]
    if len(matches) != 1:
        raise ValueError("Maze tail replay task must resolve exactly once")
    environment = dict(run["configuration"]["environment"])
    environment["max_repair_iterations"] = max(
        int(environment.get("max_repair_iterations", 0)),
        int(selection["decision_index"]) + 1,
    )
    return {
        "dataset_root": str(dataset_root),
        "row": matches[0],
        "environment": environment,
        "solver_seed": int(selection["solver_seed"]),
        "replay_destroy_strategy": "Adaptive",
    }


def _state_artifact_valid(
    payload: dict[str, Any], *, state_id: str, run_fingerprint: str
) -> bool:
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("state_id") != state_id
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("complete") is not True
        or payload.get("error") is not None
    ):
        return False
    selection = payload.get("selection")
    actions = payload.get("actions")
    trials = payload.get("trials")
    if not isinstance(selection, dict) or not isinstance(actions, list) or not isinstance(
        trials, list
    ):
        return False
    if [row.get("role") for row in actions] != ["v2_action", "challenger_action"]:
        return False
    candidate_ids = [str(row.get("candidate_id")) for row in actions]
    if len(set(candidate_ids)) != 2 or len(trials) != 32:
        return False
    by_pair = Counter(
        (str(row.get("candidate_id")), int(row.get("trial_index", -1)))
        for row in trials
    )
    if set(by_pair) != {
        (candidate_id, trial_index)
        for candidate_id in candidate_ids
        for trial_index in range(16)
    } or any(count != 1 for count in by_pair.values()):
        return False
    for trial_index in range(16):
        if len(
            {
                int(row.get("pp_seed", -1))
                for row in trials
                if int(row.get("trial_index", -1)) == trial_index
            }
        ) != 1:
            return False
    before = int(selection.get("before_conflicts", -1))
    for row in trials:
        after = row.get("conflicts_after")
        if (
            row.get("schema") != TRIAL_SCHEMA
            or row.get("state_id") != state_id
            or not isinstance(after, int)
            or after < 0
            or not math.isfinite(float(row.get("normalized_conflict_reduction", math.nan)))
            or not math.isclose(
                float(row["normalized_conflict_reduction"]),
                (before - after) / max(1, before),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            return False
    return True


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    selection = dict(job["selection"])
    output_path = Path(str(job["output_path"]))
    state_id = str(selection["state_id"])
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        payload = _read_json(output_path)
        if _state_artifact_valid(
            payload, state_id=state_id, run_fingerprint=run_fingerprint
        ):
            return {
                "job_id": state_id,
                "state_id": state_id,
                "status": "resumed",
                "candidate_count": 2,
                "trial_count": 32,
                "error_count": 0,
            }
        raise ValueError("Maze tail action-replay resume artifact is invalid")
    source_root = Path(str(selection["v2_source_root"])).resolve()
    state, trace_path = target_state_from_trace(
        source_root,
        dict(selection["v2_manifest"]),
        decision_index=int(selection["decision_index"]),
        expected_fingerprint=str(selection["before_fingerprint"]),
    )
    if (
        state_fingerprint(state) != str(selection["before_fingerprint"])
        or int(state["num_of_colliding_pairs"]) != int(selection["before_conflicts"])
    ):
        raise RuntimeError("Maze tail replay reconstructed state changed")
    replay = _replay_job(selection)
    repair_fingerprint = repair_structure_fingerprint(state)
    restore_seed = repairability_restore_seed(repair_fingerprint)
    known_agents = {int(agent["id"]) for agent in state["agents"]}
    actions = [dict(selection["v2_action"]), dict(selection["challenger_action"])]
    if any(
        not set(map(int, action["agents"])) <= known_agents or not action["agents"]
        for action in actions
    ):
        raise RuntimeError("Maze tail replay contains an illegal action")
    trials: list[dict[str, Any]] = []
    before_conflicts = int(selection["before_conflicts"])
    for action in actions:
        agents = list(map(int, action["agents"]))
        for trial_index in map(int, job["trial_indices"]):
            environment, restored = restore_repair_state(replay, state, seed=restore_seed)
            if repair_structure_fingerprint(restored) != repair_fingerprint:
                raise RuntimeError("Maze tail replay branch restore changed")
            before_low = dict(restored.get("low_level") or {})
            pp_seed = repairability_pp_seed(repair_fingerprint, trial_index)
            result = _plain(environment.step(_paired_action(agents, pp_seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            after_conflicts = int(after["num_of_colliding_pairs"])
            after_low = dict(after.get("low_level") or {})
            trials.append(
                {
                    "schema": TRIAL_SCHEMA,
                    "state_id": state_id,
                    "candidate_role": str(action["role"]),
                    "candidate_id": str(action["candidate_id"]),
                    "trial_index": trial_index,
                    "pp_seed": pp_seed,
                    "before_conflicts": before_conflicts,
                    "conflicts_after": after_conflicts,
                    "normalized_conflict_reduction": (
                        before_conflicts - after_conflicts
                    )
                    / max(1, before_conflicts),
                    "progress": after_conflicts < before_conflicts,
                    "feasible": bool(after.get("feasible")),
                    "replan_success": bool(metrics["replan_success"]),
                    "low_level_generated": int(after_low.get("generated", 0))
                    - int(before_low.get("generated", 0)),
                    "low_level_expanded": int(after_low.get("expanded", 0))
                    - int(before_low.get("expanded", 0)),
                    "pp_replan_seconds_descriptive_only": float(
                        metrics.get("pp_replan_seconds", 0.0)
                    ),
                    "after_repair_fingerprint": repair_structure_fingerprint(after),
                }
            )
    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "error": None,
        "state_id": state_id,
        "selection": selection,
        "source_trace_file": str(trace_path),
        "before_fingerprint": str(selection["before_fingerprint"]),
        "before_repair_fingerprint": repair_fingerprint,
        "restore_seed": restore_seed,
        "actions": actions,
        "trials": trials,
    }
    if not _state_artifact_valid(
        payload, state_id=state_id, run_fingerprint=run_fingerprint
    ):
        raise RuntimeError("Maze tail action-replay state artifact failed validation")
    _write_json(output_path, payload)
    return {
        "job_id": state_id,
        "state_id": state_id,
        "status": "ok",
        "candidate_count": 2,
        "trial_count": 32,
        "error_count": 0,
    }


def _mean(values: list[float]) -> float:
    return float(fmean(values)) if values else 0.0


def robust_action_order(
    v2_scores: list[float],
    challenger_scores: list[float],
    v2_progress: list[bool],
    challenger_progress: list[bool],
    rule: dict[str, Any],
) -> dict[str, Any]:
    if not (
        len(v2_scores)
        == len(challenger_scores)
        == len(v2_progress)
        == len(challenger_progress)
        == 16
    ):
        raise ValueError("robust Maze action order requires exactly 16 paired trials")
    epsilon = float(rule["tie_epsilon"])
    differences = [right - left for left, right in zip(v2_scores, challenger_scores)]
    mean_effect = _mean(differences)
    first = _mean(differences[:8])
    second = _mean(differences[8:])
    challenger_wins = sum(value > epsilon for value in differences)
    v2_wins = sum(value < -epsilon for value in differences)
    ties = 16 - challenger_wins - v2_wins
    required_wins = math.ceil(
        float(rule["minimum_paired_win_fraction"]) * len(differences)
    )
    minimum_effect = float(rule["minimum_absolute_mean_effect"])
    v2_no_progress = sum(not value for value in v2_progress) / 16.0
    challenger_no_progress = sum(not value for value in challenger_progress) / 16.0
    challenger_robust = bool(
        challenger_wins >= required_wins
        and mean_effect >= minimum_effect
        and first > 0.0
        and second > 0.0
        and challenger_no_progress <= v2_no_progress
    )
    v2_robust = bool(
        v2_wins >= required_wins
        and mean_effect <= -minimum_effect
        and first < 0.0
        and second < 0.0
        and v2_no_progress <= challenger_no_progress
    )
    winner = (
        "challenger_action"
        if challenger_robust
        else "v2_action" if v2_robust else "uncertain"
    )
    return {
        "robust_winner": winner,
        "challenger_minus_v2_mean_effect": mean_effect,
        "first_half_mean_effect": first,
        "second_half_mean_effect": second,
        "challenger_win_count": challenger_wins,
        "v2_win_count": v2_wins,
        "tie_count": ties,
        "v2_no_progress_rate": v2_no_progress,
        "challenger_no_progress_rate": challenger_no_progress,
        "required_win_count": required_wins,
    }


def analyze_action_replay(
    config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    (
        config_path,
        _root,
        config,
        preflight_path,
        selection_path,
        preflight,
        selections,
    ) = _load_replay_config(config_path)
    collection = Path(collection).resolve()
    run = _read_json(collection / "run_config.json")
    run_fingerprint = str(run["run_fingerprint"])
    expected = {str(row["state_id"]): row for row in selections}
    states: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted((collection / "states").glob("*.json")):
        payload = _read_json(path)
        state_id = str(payload.get("state_id"))
        if state_id not in expected:
            errors.append(f"unexpected state artifact: {state_id}")
            continue
        if not _state_artifact_valid(
            payload, state_id=state_id, run_fingerprint=run_fingerprint
        ):
            errors.append(f"invalid state artifact: {state_id}")
            continue
        states.append(payload)
    observed_ids = {str(state["state_id"]) for state in states}
    if observed_ids != set(expected):
        errors.append("Maze tail action-replay state coverage is incomplete")
    rule = dict(config["robust_order_rule"])
    state_reports: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda row: str(row["state_id"])):
        selection = dict(state["selection"])
        trials = sorted(
            list(state["trials"]),
            key=lambda row: (str(row["candidate_role"]), int(row["trial_index"])),
        )
        all_trials.extend(trials)
        v2 = sorted(
            [row for row in trials if row["candidate_role"] == "v2_action"],
            key=lambda row: int(row["trial_index"]),
        )
        challenger = sorted(
            [row for row in trials if row["candidate_role"] == "challenger_action"],
            key=lambda row: int(row["trial_index"]),
        )
        order = robust_action_order(
            [float(row["normalized_conflict_reduction"]) for row in v2],
            [float(row["normalized_conflict_reduction"]) for row in challenger],
            [bool(row["progress"]) for row in v2],
            [bool(row["progress"]) for row in challenger],
            rule,
        )
        state_reports.append(
            {
                "state_id": str(state["state_id"]),
                "map_id": str(selection["map_id"]),
                "task_id": str(selection["task_id"]),
                "solver_seed": int(selection["solver_seed"]),
                "challenger": str(selection["challenger"]),
                "tail_category": str(selection["tail_category"]),
                "decision_index": int(selection["decision_index"]),
                "before_conflicts": int(selection["before_conflicts"]),
                "v2_action": dict(selection["v2_action"]),
                "challenger_action": dict(selection["challenger_action"]),
                **order,
            }
        )
    robust = [row for row in state_reports if row["robust_winner"] != "uncertain"]
    robust_tail = [
        row
        for row in robust
        if row["tail_category"] in {"adverse", "severe"}
    ]
    gates = dict(config["analysis_gates"])
    integrity = {
        "preflight_passed": preflight.get("preflight_passed") is True,
        "exact_frozen_state_coverage": observed_ids == set(expected),
        "two_actions_per_state": all(len(state["actions"]) == 2 for state in states),
        "sixteen_trials_per_action": len(all_trials) == len(expected) * 2 * 16,
        "strictly_paired_pp_seeds": all(
            len(
                {
                    int(row["pp_seed"])
                    for row in state["trials"]
                    if int(row["trial_index"]) == trial_index
                }
            )
            == 1
            for state in states
            for trial_index in range(16)
        ),
        "zero_collection_errors": not errors,
        "runtime_excluded_from_label": config.get("runtime_used_in_label") is False,
        "future_trajectory_not_read": config.get("future_trajectory_read") is False,
        "outcome_based_state_filtering_forbidden": config.get(
            "outcome_based_state_filtering"
        )
        is False,
    }
    robust_fraction = len(robust) / max(1, len(state_reports))
    readiness = {
        "minimum_robustly_ordered_fraction": robust_fraction
        >= float(gates["minimum_robustly_ordered_fraction"]),
        "minimum_robustly_ordered_tail_count": len(robust_tail)
        >= int(gates["minimum_robustly_ordered_tail_count"]),
        "minimum_robust_tail_map_count": len({row["map_id"] for row in robust_tail})
        >= int(gates["minimum_robust_tail_map_count"]),
        "minimum_robust_tail_task_count": len({row["task_id"] for row in robust_tail})
        >= int(gates["minimum_robust_tail_task_count"]),
        "minimum_robust_tail_solver_seed_count": len(
            {row["solver_seed"] for row in robust_tail}
        )
        >= int(gates["minimum_robust_tail_solver_seed_count"]),
        "minimum_robust_tail_per_challenger": all(
            sum(row["challenger"] == challenger for row in robust_tail)
            >= int(gates["minimum_robust_tail_per_challenger"])
            for challenger in CHALLENGERS
        ),
    }
    integrity_passed = all(integrity.values())
    action_stability_passed = integrity_passed and all(readiness.values())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trial_path = output / "action_replay_trials.jsonl"
    _write_jsonl(trial_path, all_trials)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "paired_first_divergence_action_replay_complete",
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": integrity_passed,
        "action_stability_passed": action_stability_passed,
        "predictor_design_allowed": action_stability_passed,
        "model_training_allowed": False,
        "formal_ttf_claim": False,
        "state_count": len(state_reports),
        "candidate_count": len(state_reports) * 2,
        "trial_count": len(all_trials),
        "robustly_ordered_count": len(robust),
        "robustly_ordered_fraction": robust_fraction,
        "robustly_ordered_tail_count": len(robust_tail),
        "robust_winner_counts": dict(
            sorted(Counter(row["robust_winner"] for row in state_reports).items())
        ),
        "robust_tail_winner_counts": dict(
            sorted(Counter(row["robust_winner"] for row in robust_tail).items())
        ),
        "integrity_gates": integrity,
        "action_stability_gates": readiness,
        "errors": errors,
        "states": state_reports,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "preflight_report_sha256": sha256_file(preflight_path),
            "state_selection_sha256": sha256_file(selection_path),
            "run_config_sha256": sha256_file(collection / "run_config.json"),
            "action_replay_trials_sha256": sha256_file(trial_path),
            "state_artifact_sha256": {
                path.name: sha256_file(path)
                for path in sorted((collection / "states").glob("*.json"))
            },
        },
        "next_step": (
            "design_outcome_blind_residual_hazard_predictor_without_training"
            if action_stability_passed
            else "stop_predictor_design_and_register_second_independent_task_seed_block"
        ),
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


def collect_action_replay(
    config_path: str | Path, output: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    (
        config_path,
        root,
        config,
        preflight_path,
        selection_path,
        _preflight,
        selections,
    ) = _load_replay_config(config_path)
    output = Path(output).resolve()
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    identity = {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "preflight_report_sha256": sha256_file(preflight_path),
        "state_selection_sha256": sha256_file(selection_path),
        "selected_state_ids": [str(row["state_id"]) for row in selections],
        "trial_indices": list(map(int, config["trial_indices"])),
        "producer_identity": producer,
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("Maze tail action-replay output belongs to another run")
        if not resume:
            raise ValueError("Maze tail action-replay output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "states").mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_jsonl(output / SELECTION_FILENAME, selections)
    jobs = []
    for selection in selections:
        key = _fingerprint(
            {
                "state_id": selection["state_id"],
                "before_fingerprint": selection["before_fingerprint"],
            }
        )[:20]
        jobs.append(
            {
                "job_id": str(selection["state_id"]),
                "state_id": str(selection["state_id"]),
                "row": dict(selection["v2_manifest"]),
                "solver_seed": int(selection["solver_seed"]),
                "selection": selection,
                "trial_indices": list(map(int, config["trial_indices"])),
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    results = _run_jobs(
        _collect_state,
        jobs,
        int(config["workers"]),
        phase="stride-maze-tail-action-replay",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["state_timeout_seconds"]),
    )
    _write_jsonl(output / "collection_manifest.jsonl", results)
    completed = sum(row.get("status") in {"ok", "resumed"} for row in results)
    error_count = len(results) - completed
    status = {
        "schema": STATUS_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "total_state_count": len(selections),
        "completed_state_count": completed,
        "completed_candidate_count": completed * 2,
        "completed_trial_count": completed * 32,
        "error_state_count": error_count,
        "complete": completed == len(selections) and error_count == 0,
    }
    _write_json(output / "collection_status.json", status)
    if status["complete"]:
        report = analyze_action_replay(config_path, output, output)
        status["report_sha256"] = sha256_file(output / REPORT_FILENAME)
        _write_json(output / "collection_status.json", status)
        return report
    return status


__all__ = [
    "EXPERIMENT_ID",
    "PREFLIGHT_REPORT_FILENAME",
    "REPORT_FILENAME",
    "analyze_action_replay",
    "collect_action_replay",
    "prepare_action_replay",
    "robust_action_order",
    "validate_preflight_config",
    "validate_replay_config",
]
