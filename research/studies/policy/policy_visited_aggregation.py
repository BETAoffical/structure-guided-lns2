from __future__ import annotations

import collections
import hashlib
import itertools
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    state_storage_id as _state_storage_id,
    trial_job_id as _trial_job_id,
)
from experiments.closed_loop_confirmation import (
    _closed_loop_episode_worker,
    _sha256,
    controller_implementation_fingerprint,
    generate_online_candidates,
    load_frozen_policy_bundle,
    validate_closed_loop_trace,
)
from research.studies.representation.local_representation_audit import analyze_state
from research.studies.neighborhood.realized_neighborhood_probe import evaluation_seed
from research.studies.neighborhood.realized_neighborhood_ranking_audit import (
    _aggregate_outcomes,
    _feature_profiles_from_shared,
    _label_rows,
    candidate_feature_cache,
    state_dynamic_features,
)
from research.studies.neighborhood.realized_ranking_confirmation import _seed_isolation
from experiments.repair_collection import (
    SCHEMA_VERSION,
    CollectionLockError,
    _CollectionRunLock,
    _dataset_fingerprint,
    _fingerprint,
    _load_dataset_rows,
    _low_level_delta,
    _make_environment,
    _plain,
    _qualification_worker,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)


POLICY_VISITED_SCHEMA = "lns2.policy_visited_aggregation.v1"
POLICY_VISITED_STATE_SCHEMA = "lns2.policy_visited_state.v1"
POLICY_VISITED_OUTCOME_SCHEMA = "lns2.policy_visited_outcome.v1"
POLICY_VISITED_INDEX_SCHEMA = "lns2.policy_visited_ranking_index.v1"
ALLOWED_PROFILES = ("proposal_dynamic", "realized_dynamic")
SOURCE_POLICIES = ("official_adaptive", "realized_dynamic")
QUALIFICATION_MODES = (
    "strict_layout_coverage",
    "natural_distribution_development",
    "natural_distribution_confirmation",
)
IMPLEMENTATION_FILES = (
    "experiments/_common.py",
    "research/studies/policy/policy_visited_aggregation.py",
    "experiments/closed_loop_confirmation.py",
    "research/studies/representation/local_representation_audit.py",
    "research/studies/neighborhood/realized_neighborhood_probe.py",
    "research/studies/neighborhood/realized_neighborhood_ranking_audit.py",
    "experiments/repair_collection.py",
)


def _number_summary(values: Iterable[float | int]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": len(numbers),
        "min": numbers[0],
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "p90": numbers[int(math.floor(0.9 * (len(numbers) - 1)))],
        "max": numbers[-1],
    }


def _implementation_fingerprint(project_root: Path) -> dict[str, Any]:
    files = {
        relative: _sha256(project_root / relative)
        for relative in IMPLEMENTATION_FILES
        if (project_root / relative).is_file()
    }
    controller = controller_implementation_fingerprint(project_root)
    return {
        "sha256": _fingerprint({"files": files, "controller": controller}),
        "files": files,
        "controller": controller,
    }


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported policy-visited collection config")
    splits = tuple(map(str, config.get("splits", [])))
    if splits != ("policy_train", "policy_validation"):
        raise ValueError("policy-visited splits must be policy_train and policy_validation")
    seeds = tuple(map(int, config.get("solver_seeds", [])))
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("solver seeds must be unique non-negative integers")
    configured = dict(config.get("source_policies", {}))
    if set(configured) != set(splits):
        raise ValueError("source policies must be specified for both splits")
    for split, policies in configured.items():
        values = tuple(map(str, policies))
        if (
            not values
            or len(values) != len(set(values))
            or any(value not in SOURCE_POLICIES for value in values)
            or "realized_dynamic" not in values
        ):
            raise ValueError(f"invalid source policies for {split}")
    if int(config.get("evaluation_trials", 0)) != 4:
        raise ValueError("policy-visited collection requires four evaluation trials")
    selection = dict(config.get("state_selection", {}))
    if (
        int(selection.get("max_states_per_episode", 0)) != 3
        or str(selection.get("method")) != "early_middle_late"
    ):
        raise ValueError("policy-visited state selection must be early/middle/late with max 3")
    proposal = dict(config.get("proposal", {}))
    if (
        int(proposal.get("max_seed_agents", 0)) != 4
        or list(map(str, proposal.get("heuristics", [])))
        != ["target", "collision", "random"]
        or list(map(int, proposal.get("neighborhood_sizes", []))) != [4, 8, 16]
        or int(proposal.get("trials", 0)) != 8
        or int(proposal.get("candidates_per_family", 0)) != 2
    ):
        raise ValueError("online proposal configuration differs from the registered controller")
    for name in (
        "episode_process_timeout_seconds",
        "proposal_process_timeout_seconds",
        "trial_process_timeout_seconds",
    ):
        if float(config.get(name, 0.0)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if int(config.get("workers", 0)) <= 0:
        raise ValueError("workers must be positive")
    qualification = dict(config.get("qualification", {}))
    mode = str(qualification.get("mode", "strict_layout_coverage"))
    if mode not in QUALIFICATION_MODES:
        raise ValueError(f"unsupported qualification mode: {mode}")
    if mode == "natural_distribution_development" and str(
        config.get("study_role")
    ) != "development":
        raise ValueError("natural distribution development requires study_role=development")
    for field in (
        "minimum_nonzero_by_split",
        "minimum_nonzero_per_layout",
        "minimum_active_maps",
    ):
        values = dict(qualification.get(field, {}))
        if set(values) != set(splits) or any(int(value) < 0 for value in values.values()):
            raise ValueError(f"qualification {field} must cover both splits")


def policy_visited_dataset_design(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    settings = dict(config["dataset_design"])
    expected_tasks = set(map(str, settings["task_variants"]))
    expected_layouts = {
        str(split): {str(name): int(count) for name, count in values.items()}
        for split, values in dict(settings["layout_counts"]).items()
    }
    tasks_per_map = int(settings["tasks_per_map"])
    errors: list[str] = []
    allowed_splits = set(map(str, config["splits"]))
    if {str(row.get("split")) for row in rows} != allowed_splits:
        errors.append("dataset split set differs from the registered design")
    by_split_map: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_split_map[(str(row["split"]), str(row["map_id"]))].append(row)
    layout_counts: dict[str, collections.Counter[str]] = {
        split: collections.Counter() for split in allowed_splits
    }
    for (split, map_id), tasks in sorted(by_split_map.items()):
        layouts = {str(row.get("layout_mode")) for row in tasks}
        if len(layouts) != 1:
            errors.append(f"{split}/{map_id}: inconsistent layout")
            continue
        layout_counts[split][next(iter(layouts))] += 1
        if len(tasks) != tasks_per_map or {
            str(row.get("task_variant")) for row in tasks
        } != expected_tasks:
            errors.append(f"{split}/{map_id}: incomplete four-task pairing")
        if len({int(row["map_seed"]) for row in tasks}) != 1:
            errors.append(f"{split}/{map_id}: inconsistent map seed")
        if len({int(row["task_seed"]) for row in tasks}) != len(tasks):
            errors.append(f"{split}/{map_id}: repeated task seed")
    for split, expected in expected_layouts.items():
        if dict(sorted(layout_counts[split].items())) != expected:
            errors.append(f"{split}: layout replication differs from registration")
    expected_maps = sum(sum(values.values()) for values in expected_layouts.values())
    expected_rows = expected_maps * tasks_per_map
    if len(rows) != expected_rows or len(by_split_map) != expected_maps:
        errors.append("dataset dimensions differ from registration")
    return {
        "passed": not errors,
        "errors": errors,
        "map_count": len(by_split_map),
        "task_count": len(rows),
        "layout_counts": {
            split: dict(sorted(values.items()))
            for split, values in sorted(layout_counts.items())
        },
    }


def policy_visited_qualification_report(
    dataset_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
    config: dict[str, Any],
    design: dict[str, Any],
    isolation: dict[str, Any],
    *,
    formal: bool,
) -> dict[str, Any]:
    seeds = tuple(map(int, config["solver_seeds"]))
    valid = [row for row in qualification_rows if str(row.get("status")) == "ok"]
    errors = [row for row in qualification_rows if str(row.get("status")) != "ok"]
    nonzero = [row for row in valid if int(row.get("initial_conflicts", 0)) > 0]
    by_split = collections.Counter(str(row["split"]) for row in nonzero)
    by_split_layout = collections.Counter(
        (str(row["split"]), str(row["layout_mode"])) for row in nonzero
    )
    active_maps = {
        split: sorted(
            {str(row["map_id"]) for row in nonzero if str(row["split"]) == split}
        )
        for split in map(str, config["splits"])
    }
    fingerprints_by_seed = {
        seed: tuple(
            str(row["state_fingerprint"])
            for row in sorted(
                (item for item in valid if int(item["solver_seed"]) == seed),
                key=lambda item: (str(item["split"]), str(item["task_id"])),
            )
        )
        for seed in seeds
    }
    duplicate_streams = [
        [left, right]
        for left, right in itertools.combinations(seeds, 2)
        if fingerprints_by_seed[left] == fingerprints_by_seed[right]
    ]
    thresholds = dict(config["qualification"])
    mode = str(thresholds.get("mode", "strict_layout_coverage"))
    if mode not in QUALIFICATION_MODES:
        raise ValueError(f"unsupported qualification mode: {mode}")
    study_role = str(config.get("study_role", "legacy_confirmation"))
    gates = {
        "dataset_design": bool(design["passed"]),
        "seed_isolation": bool(isolation["passed"]),
        "all_resets_valid": len(valid) == len(dataset_rows) * len(seeds) and not errors,
        "distinct_solver_seed_trajectories": not duplicate_streams,
    }
    for split in map(str, config["splits"]):
        gates[f"minimum_nonzero_{split}"] = by_split[split] >= int(
            thresholds["minimum_nonzero_by_split"][split]
        )
        gates[f"minimum_active_maps_{split}"] = len(active_maps[split]) >= int(
            thresholds["minimum_active_maps"][split]
        )
        layouts = dict(config["dataset_design"]["layout_counts"])[split]
        gates[f"minimum_layout_coverage_{split}"] = all(
            by_split_layout[(split, layout)]
            >= int(thresholds["minimum_nonzero_per_layout"][split])
            for layout in layouts
        )
    passed = all(gates.values()) if formal else not errors
    if passed:
        decision = (
            "eligible_for_development_collection"
            if mode == "natural_distribution_development"
            else "eligible_for_independent_confirmation"
            if mode == "natural_distribution_confirmation"
            else "eligible_for_policy_visited_collection"
        )
    else:
        decision = (
            "inconclusive_development_sample_do_not_resample"
            if mode == "natural_distribution_development"
            else "inconclusive_confirmation_sample_do_not_resample"
            if mode == "natural_distribution_confirmation"
            else "inconclusive_sample_do_not_resample"
        )
    conflict_summaries = {}
    repairable_rates = {}
    for split in map(str, config["splits"]):
        for layout in sorted(config["dataset_design"]["layout_counts"][split]):
            key = f"{split}/{layout}"
            values = [
                int(row.get("initial_conflicts", 0))
                for row in valid
                if str(row["split"]) == split and str(row["layout_mode"]) == layout
            ]
            conflict_summaries[key] = _number_summary(values)
            repairable_rates[key] = (
                sum(value > 0 for value in values) / len(values) if values else 0.0
            )
    return {
        "schema": POLICY_VISITED_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": formal,
        "qualification_mode": mode,
        "study_role": study_role,
        "confirmation_evidence": bool(
            passed
            and study_role != "development"
            and mode != "natural_distribution_development"
        ),
        "passed": passed,
        "decision": decision,
        "gates": gates,
        "expected_reset_count": len(dataset_rows) * len(seeds),
        "valid_count": len(valid),
        "nonzero_state_count": len(nonzero),
        "initial_feasible_count": len(valid) - len(nonzero),
        "nonzero_by_split": dict(sorted(by_split.items())),
        "nonzero_by_split_layout": {
            f"{split}/{layout}": by_split_layout[(split, layout)]
            for split in map(str, config["splits"])
            for layout in sorted(config["dataset_design"]["layout_counts"][split])
        },
        "repairable_rate_by_split_layout": repairable_rates,
        "initial_conflicts_by_split_layout": conflict_summaries,
        "active_maps": active_maps,
        "duplicate_solver_seed_trajectories": duplicate_streams,
        "repairable_task_seeds": sorted(
            [[str(row["task_id"]), int(row["solver_seed"])] for row in nonzero]
        ),
        "errors": errors,
        "dataset_design": design,
        "seed_isolation": isolation,
    }


def select_policy_states(decision_count: int, maximum: int = 3) -> list[int]:
    if decision_count < 0 or maximum <= 0:
        raise ValueError("decision count must be non-negative and maximum positive")
    if decision_count <= maximum:
        return list(range(decision_count))
    if maximum == 1:
        return [0]
    if maximum == 2:
        return [0, decision_count - 1]
    indices = [0, (decision_count - 1) // 2, decision_count - 1]
    if maximum > 3:
        indices.extend(
            round(index * (decision_count - 1) / (maximum - 1))
            for index in range(1, maximum - 1)
        )
    return sorted(set(indices))[:maximum]


_CANDIDATE_CORE_KEYS = (
    "candidate_id",
    "agents",
    "actual_size",
    "selection_families",
    "proposal_count_by_family",
    "proposal_seeds",
    "seed_agents",
)


def candidate_core(candidate: dict[str, Any]) -> dict[str, Any]:
    value = {key: _plain(candidate[key]) for key in _CANDIDATE_CORE_KEYS}
    value["agents"] = sorted(map(int, value["agents"]))
    value["proposal_seeds"] = sorted(map(int, value["proposal_seeds"]))
    value["seed_agents"] = sorted(map(int, value["seed_agents"]))
    value["selection_families"] = sorted(map(str, value["selection_families"]))
    value["proposal_count_by_family"] = {
        str(key): int(count)
        for key, count in sorted(dict(value["proposal_count_by_family"]).items())
    }
    return value


def _source_state_rows(
    output_root: Path,
    source_manifest: list[dict[str, Any]],
    run_fingerprint: str,
    maximum_states: int,
    metric_iteration_budget: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in source_manifest:
        if str(manifest.get("policy")) != "realized_dynamic":
            continue
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            raise ValueError("source manifest contains an unsuccessful learned episode")
        validated = validate_closed_loop_trace(
            output_root / str(manifest["trace_file"]),
            run_fingerprint,
            expected_episode_id=str(manifest["episode_id"]),
            expected_policy="realized_dynamic",
            expected_solver_seed=int(manifest["solver_seed"]),
            metric_iteration_budget=metric_iteration_budget,
        )
        events = list(validated["events"])
        transitions = [event for event in events if event.get("event") == "transition"]
        selected = set(select_policy_states(len(transitions), maximum_states))
        state = events[0]["state"]
        prefix: list[dict[str, Any]] = []
        for decision_index, transition in enumerate(transitions):
            if decision_index in selected:
                state_id = (
                    f"{manifest['episode_id']}__decision_{decision_index:04d}"
                )
                if state_id in seen:
                    raise ValueError(f"duplicate policy-visited state: {state_id}")
                seen.add(state_id)
                controller = transition.get("controller")
                if not isinstance(controller, dict):
                    raise ValueError("learned source transition is missing controller data")
                source_candidates = sorted(
                    [candidate_core(value) for value in controller.get("candidate_pool", [])],
                    key=lambda value: str(value["candidate_id"]),
                )
                if not source_candidates:
                    raise ValueError("learned source transition has no candidate pool")
                rows.append(
                    {
                        "schema": POLICY_VISITED_STATE_SCHEMA,
                        "schema_version": SCHEMA_VERSION,
                        "run_fingerprint": run_fingerprint,
                        "state_id": state_id,
                        "episode_id": str(manifest["episode_id"]),
                        "decision_index": decision_index,
                        "decision_count": len(transitions),
                        "decision_fraction": (
                            decision_index / (len(transitions) - 1)
                            if len(transitions) > 1
                            else 0.0
                        ),
                        "stage": (
                            "early"
                            if decision_index == 0
                            else "late"
                            if decision_index == len(transitions) - 1
                            else "middle"
                        ),
                        "state_fingerprint": state_fingerprint(state),
                        "prefix_actions": _plain(prefix),
                        "source_candidates": source_candidates,
                        "source_selected_candidate_id": str(
                            controller["selected_candidate_id"]
                        ),
                        "state": _plain(state),
                        "split": str(manifest["split"]),
                        "map_id": str(manifest["map_id"]),
                        "task_id": str(manifest["task_id"]),
                        "layout_mode": str(manifest["layout_mode"]),
                        "task_variant": str(manifest["task_variant"]),
                        "agent_count": int(manifest["agent_count"]),
                        "solver_seed": int(manifest["solver_seed"]),
                    }
                )
            prefix.append(_plain(transition["action"]))
            state = transition["after"]
    return sorted(rows, key=lambda value: str(value["state_id"]))


def _proposal_worker(job: dict[str, Any]) -> dict[str, Any]:
    state_row = job["state_row"]
    output_root = Path(job["output_root"])
    state_id = str(state_row["state_id"])
    root = output_root / "proposals" / str(state_row["split"]) / _state_storage_id(state_id)
    metadata_path = root / "metadata.json"
    candidate_path = root / "candidate_state.json"
    if job["resume"] and metadata_path.is_file() and candidate_path.is_file():
        metadata = _read_json(metadata_path)
        if (
            str(metadata.get("run_fingerprint")) == job["run_fingerprint"]
            and bool(metadata.get("complete"))
        ):
            return {**metadata, "status": "resumed"}
    try:
        environment = _make_environment(
            job["dataset_root"], job["row"], job["environment"], "Adaptive"
        )
        state = _plain(environment.reset(seed=int(state_row["solver_seed"])))
        for action in state_row["prefix_actions"]:
            if state["done"]:
                raise RuntimeError("source prefix terminated before selected state")
            state = _plain(environment.step(action))["observation"]
        expected = str(state_row["state_fingerprint"])
        if state_fingerprint(state) != expected:
            raise RuntimeError("policy-visited prefix replay fingerprint mismatch")
        candidates, proposal_metrics = generate_online_candidates(
            environment,
            state,
            task_id=str(state_row["task_id"]),
            solver_seed=int(state_row["solver_seed"]),
            decision_index=int(state_row["decision_index"]),
            proposal_config=job["proposal"],
        )
        actual = sorted(
            [candidate_core(value) for value in candidates],
            key=lambda value: str(value["candidate_id"]),
        )
        if actual != list(state_row["source_candidates"]):
            raise RuntimeError("regenerated candidate pool differs from the source trace")
        candidate_row = {
            **state_row,
            "source_candidates": None,
            "candidates": actual,
            "candidate_count": len(actual),
            "candidate_pool_fingerprint": _fingerprint(actual),
            "proposal_metrics": proposal_metrics,
        }
        _write_json(candidate_path, candidate_row)
        metadata = {
            "schema": POLICY_VISITED_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": job["run_fingerprint"],
            "state_id": state_id,
            "split": str(state_row["split"]),
            "map_id": str(state_row["map_id"]),
            "task_id": str(state_row["task_id"]),
            "candidate_count": len(actual),
            "candidate_file": candidate_path.relative_to(output_root).as_posix(),
            "metadata_file": metadata_path.relative_to(output_root).as_posix(),
            "fingerprint_match": True,
            "candidate_pool_match": True,
            "complete": True,
            "status": "ok",
            "error": None,
        }
        _write_json(metadata_path, metadata)
        return metadata
    except Exception as error:
        return {
            "schema": POLICY_VISITED_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": job["run_fingerprint"],
            "state_id": state_id,
            "split": str(state_row["split"]),
            "map_id": str(state_row["map_id"]),
            "task_id": str(state_row["task_id"]),
            "candidate_count": 0,
            "candidate_file": None,
            "metadata_file": metadata_path.relative_to(output_root).as_posix(),
            "fingerprint_match": False,
            "candidate_pool_match": False,
            "complete": False,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def _trial_result_path(output_root: Path, job: dict[str, Any]) -> Path:
    return (
        output_root
        / "explicit_trials"
        / str(job["state_row"]["split"])
        / _state_storage_id(str(job["state_id"]))
        / str(job["candidate_id"])
        / f"trial_{int(job['evaluation_trial_index']):04d}.json"
    )


def _evaluation_trial_worker(job: dict[str, Any]) -> dict[str, Any]:
    state_row = job["state_row"]
    candidate = job["candidate"]
    state_id = str(job["state_id"])
    candidate_id = str(job["candidate_id"])
    trial_index = int(job["evaluation_trial_index"])
    agents = sorted(map(int, candidate["agents"]))
    proposal_seeds = sorted(map(int, candidate["proposal_seeds"]))
    seed = evaluation_seed(state_id, candidate_id, trial_index, proposal_seeds)
    common = {
        "schema": POLICY_VISITED_OUTCOME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": job["run_fingerprint"],
        "job_id": job["job_id"],
        "state_id": state_id,
        "candidate_id": candidate_id,
        "evaluation_trial_index": trial_index,
        "split": str(state_row["split"]),
        "map_id": str(state_row["map_id"]),
        "task_id": str(state_row["task_id"]),
    }
    try:
        environment = _make_environment(
            job["dataset_root"], job["row"], job["environment"], "Adaptive"
        )
        before = _plain(environment.reset(seed=int(state_row["solver_seed"])))
        for action in state_row["prefix_actions"]:
            if before["done"]:
                raise RuntimeError("evaluation prefix terminated before selected state")
            before = _plain(environment.step(action))["observation"]
        if state_fingerprint(before) != str(state_row["state_fingerprint"]):
            raise RuntimeError("evaluation prefix replay fingerprint mismatch")
        result = _plain(
            environment.step(
                {
                    "mode": "explicit_neighborhood",
                    "agents": agents,
                    "random_seed": seed,
                }
            )
        )
        metrics = result["metrics"]
        actual = sorted(map(int, metrics.get("neighborhood", [])))
        if not bool(metrics.get("action_valid")) or actual != agents:
            raise RuntimeError("explicit policy-visited candidate was rejected or changed")
        after = result["observation"]
        conflicts_before = int(before["num_of_colliding_pairs"])
        conflicts_after = int(after["num_of_colliding_pairs"])
        low_level = _low_level_delta(before, after)
        outcome = {
            **common,
            "agents": agents,
            "selection_families": list(candidate["selection_families"]),
            "proposal_seeds": proposal_seeds,
            "evaluation_seed": seed,
            "evaluation_seed_disjoint": seed not in set(proposal_seeds),
            "actual_neighborhood": actual,
            "action_valid": True,
            "solved": bool(after["feasible"]),
            "conflicts_before": conflicts_before,
            "conflicts_after": conflicts_after,
            "conflict_auc": (conflicts_before + conflicts_after) / 2.0,
            "sum_of_costs_after": int(after["sum_of_costs"]),
            "generated": int(low_level["generated"]),
            "runtime": float(metrics["step_runtime"]),
        }
        return {
            **common,
            "status": "ok",
            "complete": True,
            "outcome_count": 1,
            "error": None,
            "outcome": outcome,
        }
    except Exception as error:
        return {
            **common,
            "status": "error",
            "complete": False,
            "outcome_count": 0,
            "error": f"{type(error).__name__}: {error}",
            "outcome": None,
        }


def _normalize_trial_result(
    result: dict[str, Any],
    job: dict[str, Any],
    output_root: Path,
    run_fingerprint: str,
) -> dict[str, Any]:
    normalized = {
        **result,
        "schema": POLICY_VISITED_OUTCOME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "job_id": job["job_id"],
        "state_id": job["state_id"],
        "candidate_id": job["candidate_id"],
        "evaluation_trial_index": int(job["evaluation_trial_index"]),
    }
    normalized["complete"] = str(normalized.get("status")) in {"ok", "resumed"}
    normalized["outcome_count"] = int(bool(normalized["complete"]))
    path = _trial_result_path(output_root, job)
    normalized["result_file"] = path.relative_to(output_root).as_posix()
    _write_json(path, normalized)
    return normalized


def _collect_trial_results(
    jobs: list[dict[str, Any]],
    output_root: Path,
    run_fingerprint: str,
    workers: int,
    timeout_seconds: float,
    resume: bool,
) -> list[dict[str, Any]]:
    resumed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        path = _trial_result_path(output_root, job)
        if resume and path.is_file():
            value = _read_json(path)
            if (
                str(value.get("run_fingerprint")) == run_fingerprint
                and bool(value.get("complete"))
                and str(value.get("status")) in {"ok", "resumed"}
            ):
                resumed.append({**value, "status": "resumed"})
                continue
        pending.append(job)
    indexed_jobs = {str(job["job_id"]): job for job in pending}
    completed: list[dict[str, Any]] = []

    def record(result: dict[str, Any]) -> None:
        job = indexed_jobs[str(result["job_id"])]
        completed.append(
            _normalize_trial_result(result, job, output_root, run_fingerprint)
        )

    if pending:
        _run_jobs(
            _evaluation_trial_worker,
            pending,
            workers,
            phase="policy-visited-evaluation-trial",
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            timeout_seconds=timeout_seconds,
            on_result=record,
        )
    rows = sorted(
        resumed + completed,
        key=lambda row: (
            str(row["state_id"]),
            str(row["candidate_id"]),
            int(row["evaluation_trial_index"]),
        ),
    )
    _write_jsonl(output_root / "evaluation_trial_manifest.jsonl", rows)
    return rows


def _aggregate_trial_results(
    output_root: Path,
    state_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    expected_trials: int,
    run_fingerprint: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trial_rows:
        grouped[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    manifests = []
    for state_row in state_rows:
        state_id = str(state_row["state_id"])
        outcomes = []
        errors = []
        for candidate in state_row["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            values = grouped.get((state_id, candidate_id), [])
            indices = sorted(int(row["evaluation_trial_index"]) for row in values)
            if indices != list(range(expected_trials)):
                errors.append(
                    {
                        "state_id": state_id,
                        "candidate_id": candidate_id,
                        "error": "missing or duplicate evaluation trials",
                        "trial_indices": indices,
                    }
                )
                continue
            for value in values:
                if str(value.get("status")) not in {"ok", "resumed"} or not value.get(
                    "outcome"
                ):
                    errors.append(
                        {
                            "state_id": state_id,
                            "candidate_id": candidate_id,
                            "evaluation_trial_index": value.get(
                                "evaluation_trial_index"
                            ),
                            "status": value.get("status"),
                            "error": value.get("error"),
                        }
                    )
                else:
                    outcomes.append(value["outcome"])
        root = (
            output_root
            / "explicit"
            / str(state_row["split"])
            / _state_storage_id(state_id)
        )
        outcomes_path = root / "outcomes.jsonl"
        errors_path = root / "errors.jsonl"
        metadata_path = root / "metadata.json"
        _write_jsonl(outcomes_path, outcomes)
        _write_jsonl(errors_path, errors)
        manifest = {
            "schema": POLICY_VISITED_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "state_id": state_id,
            "episode_id": state_row["episode_id"],
            "split": state_row["split"],
            "map_id": state_row["map_id"],
            "task_id": state_row["task_id"],
            "state_count": 1,
            "candidate_count": len(state_row["candidates"]),
            "outcome_count": len(outcomes),
            "error_count": len(errors),
            "outcomes_file": outcomes_path.relative_to(output_root).as_posix(),
            "errors_file": errors_path.relative_to(output_root).as_posix(),
            "metadata_file": metadata_path.relative_to(output_root).as_posix(),
            "complete": not errors,
            "status": "ok" if not errors else "error",
        }
        _write_json(metadata_path, manifest)
        manifests.append(manifest)
    manifests.sort(key=lambda row: str(row["state_id"]))
    _write_jsonl(output_root / "collection_manifest.jsonl", manifests)
    return manifests


def build_policy_visited_index(
    collection: str | Path,
    *,
    expected_trials: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(collection).resolve()
    run_config = _read_json(root / "run_config.json")
    candidates = _read_jsonl(root / "candidates.jsonl")
    manifests = _read_jsonl(root / "collection_manifest.jsonl")
    manifest_by_state = {str(row["state_id"]): row for row in manifests}
    if len(manifest_by_state) != len(manifests):
        raise ValueError("policy-visited collection has duplicate state manifests")
    if {str(row["state_id"]) for row in candidates} != set(manifest_by_state):
        raise ValueError("candidate rows and state manifests differ")
    outcomes_by_key: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    total_outcomes = 0
    for manifest in manifests:
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            raise ValueError("policy-visited manifest contains an unsuccessful state")
        if _read_jsonl(root / str(manifest["errors_file"])):
            raise ValueError("policy-visited state contains evaluation errors")
        outcomes = _read_jsonl(root / str(manifest["outcomes_file"]))
        total_outcomes += len(outcomes)
        for outcome in outcomes:
            outcomes_by_key[
                (str(outcome["state_id"]), str(outcome["candidate_id"]))
            ].append(outcome)
    index: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for source in candidates:
        state = source["state"]
        analysis = analyze_state(state)
        dynamic = state_dynamic_features(state, analysis)
        feature_cache = candidate_feature_cache(state, analysis)
        for candidate in source["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            key = (str(source["state_id"]), candidate_id)
            if key in keys:
                raise ValueError(f"duplicate policy-visited candidate: {key}")
            keys.add(key)
            trials = outcomes_by_key.get(key, [])
            if len(trials) != expected_trials:
                raise ValueError(
                    f"candidate {candidate_id} has {len(trials)} trials, expected {expected_trials}"
                )
            indices = sorted(int(row["evaluation_trial_index"]) for row in trials)
            if indices != list(range(expected_trials)):
                raise ValueError(f"candidate {candidate_id} has invalid trial indices")
            seeds = [int(row["evaluation_seed"]) for row in trials]
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"candidate {candidate_id} repeats an evaluation seed")
            agents = sorted(map(int, candidate["agents"]))
            if int(candidate.get("actual_size", len(agents))) != len(agents):
                raise ValueError(f"candidate {candidate_id} has an invalid actual size")
            for outcome in trials:
                if not bool(outcome.get("action_valid")):
                    raise ValueError(f"candidate {candidate_id} contains an invalid action")
                if not bool(outcome.get("evaluation_seed_disjoint")):
                    raise ValueError(f"candidate {candidate_id} reuses a proposal seed")
                if sorted(map(int, outcome.get("actual_neighborhood", []))) != agents:
                    raise ValueError(f"candidate {candidate_id} changed its explicit neighborhood")
                if sorted(map(int, outcome.get("agents", []))) != agents:
                    raise ValueError(f"candidate {candidate_id} outcome agents differ")
                if int(outcome.get("conflicts_before", -1)) != int(
                    state["num_of_colliding_pairs"]
                ):
                    raise ValueError(f"candidate {candidate_id} has the wrong source conflicts")
            all_profiles = _feature_profiles_from_shared(
                state,
                analysis,
                candidate,
                dynamic=dynamic,
                feature_cache=feature_cache,
            )
            profiles = {name: all_profiles[name] for name in ALLOWED_PROFILES}
            index.append(
                {
                    "schema": POLICY_VISITED_INDEX_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "state_id": str(source["state_id"]),
                    "candidate_id": candidate_id,
                    "candidate_key": candidate_id,
                    "map_id": str(source["map_id"]),
                    "task_id": str(source["task_id"]),
                    "split": str(source["split"]),
                    "layout_mode": str(source["layout_mode"]),
                    "task_variant": str(source["task_variant"]),
                    "agent_count": int(source["agent_count"]),
                    "decision_index": int(source["decision_index"]),
                    "decision_count": int(source["decision_count"]),
                    "decision_fraction": float(source["decision_fraction"]),
                    "stage": str(source["stage"]),
                    "actual_size": len(agents),
                    "agents": agents,
                    "selection_families": list(candidate["selection_families"]),
                    "seed_agents": list(candidate["seed_agents"]),
                    "trial_count": len(trials),
                    "features": profiles,
                    "outcome": _aggregate_outcomes(trials),
                    "neighborhood_sha256": hashlib.sha256(
                        ",".join(map(str, agents)).encode("ascii")
                    ).hexdigest(),
                }
            )
    if set(outcomes_by_key) != keys:
        raise ValueError("policy-visited collection contains orphan outcomes")
    _label_rows(index)
    index.sort(key=lambda row: (str(row["state_id"]), str(row["candidate_id"])))
    split_counts = collections.Counter(str(row["split"]) for row in candidates)
    checks = {
        "passed": True,
        "schema": POLICY_VISITED_INDEX_SCHEMA,
        "source_run_fingerprint": str(run_config["run_fingerprint"]),
        "state_count": len(candidates),
        "candidate_count": len(index),
        "outcome_count": total_outcomes,
        "trials_per_candidate": expected_trials,
        "map_count": len({str(row["map_id"]) for row in candidates}),
        "states_by_split": dict(sorted(split_counts.items())),
        "feature_profiles": list(ALLOWED_PROFILES),
        "forbidden_split_rows": sum(
            str(row["split"]) not in {"policy_train", "policy_validation"}
            for row in candidates
        ),
    }
    return index, checks


def _incremental_manifest(path: Path, key: str):
    rows = _read_jsonl(path) if path.is_file() else []
    values = {str(row[key]): row for row in rows}

    def record(result: dict[str, Any]) -> None:
        values[str(result[key])] = result
        _write_jsonl(path, [values[name] for name in sorted(values)])

    return record


def _selected_rows(
    rows: list[dict[str, Any]], task_ids: list[str] | None
) -> list[dict[str, Any]]:
    if task_ids is None:
        return rows
    requested = list(dict.fromkeys(map(str, task_ids)))
    indexed = {str(row["task_id"]): row for row in rows}
    missing = sorted(set(requested) - set(indexed))
    if missing:
        raise ValueError(f"unknown task ids: {missing}")
    return [indexed[value] for value in requested]


def run_policy_visited_collection(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    phases = {"qualify", "source", "propose", "evaluate", "all"}
    if phase not in phases:
        raise ValueError("phase must be qualify, source, propose, evaluate, or all")
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_config(config)
    all_rows = _load_dataset_rows(dataset_root, list(map(str, config["splits"])))
    rows = _selected_rows(all_rows, task_ids)
    formal = task_ids is None and bool(config.get("formal", True))
    design = policy_visited_dataset_design(all_rows, config)
    isolation = _seed_isolation(
        all_rows, list(config.get("reference_datasets", [])), project_root
    )
    frozen_root = Path(str(config["frozen_models"]))
    if not frozen_root.is_absolute():
        frozen_root = project_root / frozen_root
    bundle = load_frozen_policy_bundle(frozen_root, dict(config["model_registration"]))
    implementation = _implementation_fingerprint(project_root)
    dataset_fp = _dataset_fingerprint(dataset_root)
    effective = {**config, "task_ids_override": task_ids}
    config_fp = _fingerprint(effective)
    run_fp = _fingerprint(
        {
            "dataset_fingerprint": dataset_fp,
            "configuration_fingerprint": config_fp,
            "frozen_models": bundle.manifest,
            "implementation": implementation,
        }
    )
    effective_workers = int(workers or config["workers"])
    solver_seeds = tuple(map(int, config["solver_seeds"]))
    learned_sources = len(rows) * len(solver_seeds)
    source_episode_count = sum(
        len(config["source_policies"][str(row["split"])]) * len(solver_seeds)
        for row in rows
    )
    maximum_states = learned_sources * int(
        config["state_selection"]["max_states_per_episode"]
    )
    maximum_candidates = maximum_states * len(config["proposal"]["heuristics"]) * len(
        config["proposal"]["neighborhood_sizes"]
    ) * int(config["proposal"]["candidates_per_family"])
    estimate = {
        "task_count": len(rows),
        "qualification_reset_count": len(rows) * len(solver_seeds),
        "frozen_ranker_source_episode_count": learned_sources,
        "source_policy_episode_count": source_episode_count,
        "maximum_state_count": maximum_states,
        "maximum_candidate_count": maximum_candidates,
        "maximum_outcome_count": maximum_candidates
        * int(config["evaluation_trials"]),
        "workers": effective_workers,
    }
    if dry_run:
        return {
            "schema": POLICY_VISITED_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "formal": formal,
            "run_fingerprint": run_fp,
            "dataset_design": design,
            "seed_isolation": isolation,
            "frozen_models": bundle.manifest,
            "implementation": implementation,
            "estimate": estimate,
        }
    run_config = {
        "schema": POLICY_VISITED_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_fp,
        "configuration": effective,
        "configuration_fingerprint": config_fp,
        "run_fingerprint": run_fp,
        "formal": formal,
        "dataset_design": design,
        "seed_isolation": isolation,
        "frozen_models": bundle.manifest,
        "implementation": implementation,
    }
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fp:
            raise ValueError("output contains a different policy-visited run")
        if not resume:
            raise ValueError("output already exists; pass resume to continue")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, run_config)
    sequence = (
        ("qualify", "source", "propose", "evaluate") if phase == "all" else (phase,)
    )
    summary: dict[str, Any] = {
        "schema": POLICY_VISITED_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": formal,
        "run_fingerprint": run_fp,
        "estimate": estimate,
    }
    dataset_index = {str(row["task_id"]): row for row in rows}
    for current in sequence:
        if current == "qualify":
            jobs = [
                {
                    "row": row,
                    "solver_seed": seed,
                    "dataset_root": str(dataset_root),
                    "environment": config["environment"],
                }
                for row in rows
                for seed in solver_seeds
            ]
            manifest_path = output_root / "qualification_manifest.jsonl"
            record = _incremental_manifest(manifest_path, "task_id_solver_seed")
            for job in jobs:
                job["task_id_solver_seed"] = (
                    f"{job['row']['task_id']}__seed_{int(job['solver_seed']):04d}"
                )

            def qualification_record(result: dict[str, Any]) -> None:
                record(
                    {
                        **result,
                        "task_id_solver_seed": (
                            f"{result['task_id']}__seed_{int(result['solver_seed']):04d}"
                        ),
                    }
                )

            with _CollectionRunLock(output_root, run_fp, "policy-visited-qualification"):
                results = _run_jobs(
                    _qualification_worker,
                    jobs,
                    effective_workers,
                    phase="policy-visited-qualification",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["episode_process_timeout_seconds"]),
                    on_result=qualification_record,
                )
            normalized = [
                {
                    **row,
                    "task_id_solver_seed": (
                        f"{row['task_id']}__seed_{int(row['solver_seed']):04d}"
                    ),
                }
                for row in results
            ]
            _write_jsonl(manifest_path, sorted(normalized, key=lambda row: row["task_id_solver_seed"]))
            report = policy_visited_qualification_report(
                rows, normalized, config, design, isolation, formal=formal
            )
            _write_json(output_root / "qualification_report.json", report)
            summary["qualification"] = report
            if not report["passed"] and phase == "all":
                break
        elif current == "source":
            qualification = policy_visited_qualification_report(
                rows,
                _read_jsonl(output_root / "qualification_manifest.jsonl"),
                config,
                design,
                isolation,
                formal=formal,
            )
            if not qualification["passed"]:
                raise ValueError("qualification failed; source collection is forbidden")
            jobs = []
            for row in rows:
                for seed in solver_seeds:
                    for policy in config["source_policies"][str(row["split"])]:
                        jobs.append(
                            {
                                "row": row,
                                "policy": str(policy),
                                "solver_seed": seed,
                                "dataset_root": str(dataset_root),
                                "environment": config["environment"],
                                "proposal": config["proposal"],
                                "max_decisions": int(config["max_decisions"]),
                                "metric_iteration_budget": int(
                                    config["metric_iteration_budget"]
                                ),
                                "wall_time_budget_seconds": float(
                                    config["wall_time_budget_seconds"]
                                ),
                                "frozen_models": str(frozen_root.resolve()),
                                "model_registration": config["model_registration"],
                                "output_root": str(output_root),
                                "run_fingerprint": run_fp,
                                "resume": resume,
                            }
                        )
            manifest_path = output_root / "source_manifest.jsonl"
            record = _incremental_manifest(manifest_path, "episode_id")
            with _CollectionRunLock(output_root, run_fp, "policy-visited-source"):
                results = _run_jobs(
                    _closed_loop_episode_worker,
                    jobs,
                    effective_workers,
                    phase="policy-visited-source",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["episode_process_timeout_seconds"]),
                    on_result=record,
                )
            _write_jsonl(
                manifest_path, sorted(results, key=lambda row: str(row["episode_id"]))
            )
            summary["source"] = {
                "episode_count": len(results),
                "success_count": sum(
                    bool(row.get("summary", {}).get("success")) for row in results
                ),
                "error_count": sum(
                    str(row.get("status")) not in {"ok", "resumed"}
                    for row in results
                ),
            }
            if summary["source"]["error_count"] and phase == "all":
                break
        elif current == "propose":
            source_manifest = _read_jsonl(output_root / "source_manifest.jsonl")
            state_rows = _source_state_rows(
                output_root,
                source_manifest,
                run_fp,
                int(config["state_selection"]["max_states_per_episode"]),
                int(config["metric_iteration_budget"]),
            )
            _write_jsonl(output_root / "selected_states.jsonl", state_rows)
            jobs = [
                {
                    "state_row": state_row,
                    "state_id": str(state_row["state_id"]),
                    "solver_seed": int(state_row["solver_seed"]),
                    "row": dataset_index[str(state_row["task_id"])],
                    "dataset_root": str(dataset_root),
                    "environment": config["environment"],
                    "proposal": config["proposal"],
                    "output_root": str(output_root),
                    "run_fingerprint": run_fp,
                    "resume": resume,
                }
                for state_row in state_rows
            ]
            manifest_path = output_root / "proposal_manifest.jsonl"
            record = _incremental_manifest(manifest_path, "state_id")
            with _CollectionRunLock(output_root, run_fp, "policy-visited-proposal"):
                results = _run_jobs(
                    _proposal_worker,
                    jobs,
                    effective_workers,
                    phase="policy-visited-proposal",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["proposal_process_timeout_seconds"]),
                    on_result=record,
                )
            _write_jsonl(
                manifest_path, sorted(results, key=lambda row: str(row["state_id"]))
            )
            if any(str(row.get("status")) not in {"ok", "resumed"} for row in results):
                summary["proposal"] = {
                    "state_count": len(results),
                    "error_count": sum(
                        str(row.get("status")) not in {"ok", "resumed"}
                        for row in results
                    ),
                }
                if phase == "all":
                    break
            candidate_rows = [
                _read_json(output_root / str(row["candidate_file"]))
                for row in results
                if str(row.get("status")) in {"ok", "resumed"}
            ]
            candidate_rows.sort(key=lambda row: str(row["state_id"]))
            _write_jsonl(output_root / "candidates.jsonl", candidate_rows)
            summary["proposal"] = {
                "state_count": len(candidate_rows),
                "candidate_count": sum(
                    int(row["candidate_count"]) for row in candidate_rows
                ),
                "error_count": len(results) - len(candidate_rows),
                "stages": dict(
                    sorted(collections.Counter(row["stage"] for row in candidate_rows).items())
                ),
            }
        elif current == "evaluate":
            state_rows = _read_jsonl(output_root / "candidates.jsonl")
            jobs = []
            for state_row in state_rows:
                for candidate in state_row["candidates"]:
                    for trial_index in range(int(config["evaluation_trials"])):
                        state_id = str(state_row["state_id"])
                        candidate_id = str(candidate["candidate_id"])
                        jobs.append(
                            {
                                "job_id": _trial_job_id(
                                    state_id, candidate_id, trial_index
                                ),
                                "state_id": state_id,
                                "candidate_id": candidate_id,
                                "evaluation_trial_index": trial_index,
                                "state_row": state_row,
                                "candidate": candidate,
                                "row": dataset_index[str(state_row["task_id"])],
                                "dataset_root": str(dataset_root),
                                "environment": config["environment"],
                                "run_fingerprint": run_fp,
                            }
                        )
            with _CollectionRunLock(output_root, run_fp, "policy-visited-evaluation"):
                trial_rows = _collect_trial_results(
                    jobs,
                    output_root,
                    run_fp,
                    effective_workers,
                    float(config["trial_process_timeout_seconds"]),
                    resume,
                )
            manifests = _aggregate_trial_results(
                output_root,
                state_rows,
                trial_rows,
                int(config["evaluation_trials"]),
                run_fp,
            )
            summary["evaluation"] = {
                "state_count": len(manifests),
                "candidate_count": sum(int(row["candidate_count"]) for row in manifests),
                "trial_count": len(trial_rows),
                "outcome_count": sum(int(row["outcome_count"]) for row in manifests),
                "error_count": sum(int(row["error_count"]) for row in manifests),
                "timeout_count": sum(
                    str(row.get("status")) == "timeout" for row in trial_rows
                ),
            }
    _write_json(output_root / "collection_summary.json", summary)
    return summary


__all__ = [
    "ALLOWED_PROFILES",
    "CollectionLockError",
    "POLICY_VISITED_SCHEMA",
    "build_policy_visited_index",
    "candidate_core",
    "policy_visited_dataset_design",
    "policy_visited_qualification_report",
    "run_policy_visited_collection",
    "select_policy_states",
]
