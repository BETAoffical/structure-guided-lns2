from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import summarize_initial_state_complexity
from experiments.stride_collection import (
    FULL_POOL_PROPOSAL,
    STRIDE_SOURCE_POLICIES,
    _paired_action,
    _replay_job,
    _validate_native_repair,
    load_stride_selection,
)
from experiments.stride_lns import (
    FROZEN_FEATURE_DIMENSION,
    FROZEN_FEATURE_SCHEMA_ID,
    STRIDE_TRIAL_SCHEMA,
    post_structure_metrics,
)
from experiments.stride_repairability import validate_repairability_label_config
from experiments.trace_replay import (
    TARGET_STATE_RESTORE_CONTRACT,
    restore_repair_state,
    target_state_from_trace,
)
from lns2_selector.runtime.artifact_validation import (
    candidate_records as validate_candidate_records,
    repair_trial_semantics_valid,
    strict_integer,
    trial_product_matches,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import generate_online_candidates
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


COLLECTION_SCHEMA = "lns2.stride.repairability_collection.v2"
STATE_SCHEMA = "lns2.stride.repairability_state.v2"
BOUNDARY_AUGMENTATION = {
    "enabled": True,
    "generator_id": "stride-topoboundary-v1",
    "neighborhood_size": 16,
    "core_budget": 4,
    "maximum_added_candidates": 2,
    "runtime_id": "stride-boundary-static-cache-v1",
    "static_grid_cache": True,
}
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_lns.py",
    "experiments/stride_repairability.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/artifact_validation.py",
    "lns2_selector/runtime/online_selection.py",
    "lns2_selector/runtime/topology_candidates.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def repairability_pp_seed(state_repair_fingerprint: str, trial_index: int) -> int:
    if trial_index not in range(16):
        raise ValueError("repairability PP trial index must be in 0..15")
    return int(
        _fingerprint(
            {
                "namespace": "stride-repairability-paired-pp-v1",
                "repair_state": str(state_repair_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def repairability_restore_seed(state_repair_fingerprint: str) -> int:
    return int(
        _fingerprint(
            {
                "namespace": TARGET_STATE_RESTORE_CONTRACT,
                "repair_state": str(state_repair_fingerprint),
            }
        )[:16],
        16,
    ) % (2**31)


def _source_target_state(
    decision: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    source_root = Path(str(decision["source_root"])).resolve()
    source_policy = str(decision["source_policy"])
    matching_policies = [
        manifest_name
        for _policy_key, (manifest_name, policy_name) in STRIDE_SOURCE_POLICIES.items()
        if policy_name == source_policy
    ]
    if len(matching_policies) != 1:
        raise ValueError(f"unsupported repairability source policy: {source_policy}")
    manifests = [
        row
        for row in _read_jsonl(source_root / matching_policies[0])
        if str(row.get("episode_id")) == str(decision["episode_id"])
    ]
    if len(manifests) != 1:
        raise ValueError("selected repairability episode must resolve exactly once")
    manifest = manifests[0]
    if (
        str(manifest.get("status")) != "ok"
        or str(manifest.get("task_id")) != str(decision["task_id"])
        or int(manifest.get("solver_seed", -1)) != int(decision["solver_seed"])
    ):
        raise ValueError("selected repairability episode provenance changed")
    state, trace_path = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(decision["decision_index"]),
        expected_fingerprint=str(decision["before_fingerprint"]),
    )
    if int(state["num_of_colliding_pairs"]) != int(decision["before_conflicts"]):
        raise ValueError("selected repairability conflict count changed")
    return state, manifest, trace_path


def _candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(candidate["candidate_id"]),
        tuple(sorted(map(int, candidate["agents"]))),
    )


def _artifact_valid(
    payload: dict[str, Any],
    *,
    run_fingerprint: str,
    state_id: str,
    trial_indices: tuple[int, ...],
    decision: dict[str, Any] | None = None,
) -> bool:
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("state_id") != state_id
        or payload.get("complete") is not True
    ):
        return False
    embedded_decision = payload.get("decision")
    if not isinstance(embedded_decision, dict):
        return False
    if decision is not None and embedded_decision != decision:
        return False
    expected_decision = decision if decision is not None else embedded_decision
    if expected_decision.get("state_id") != state_id:
        return False
    candidates = payload.get("candidates")
    trials = payload.get("trials")
    indexed_candidates = validate_candidate_records(candidates)
    if indexed_candidates is None or not isinstance(trials, list):
        return False
    candidate_ids = list(indexed_candidates)
    before_fingerprint = expected_decision.get("before_fingerprint")
    before_repair = payload.get("before_repair_fingerprint")
    before_conflicts = payload.get("before_conflicts")
    state_restore = payload.get("state_restore")
    if (
        not isinstance(before_fingerprint, str)
        or payload.get("before_fingerprint") != before_fingerprint
        or expected_decision.get("before_conflicts") != before_conflicts
        or not strict_integer(before_conflicts, minimum=1)
        or not isinstance(before_repair, str)
        or not before_repair
        or not isinstance(state_restore, dict)
        or state_restore.get("contract") != TARGET_STATE_RESTORE_CONTRACT
        or state_restore.get("restore_seed")
        != repairability_restore_seed(before_repair)
        or state_restore.get("repair_structure_fingerprint") != before_repair
    ):
        return False
    required_names = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    if not trial_product_matches(
        trials, candidate_ids=candidate_ids, trial_indices=trial_indices
    ):
        return False
    metadata = {
        "layout_family": str(expected_decision.get("layout_mode", "unknown")),
        "map_id": expected_decision.get("map_id"),
        "task_id": expected_decision.get("task_id"),
        "split": expected_decision.get("research_split"),
        "source_split": expected_decision.get("split"),
        "source_policy": expected_decision.get("source_policy"),
        "decision_stage": expected_decision.get("decision_stage"),
        "solver_seed": expected_decision.get("solver_seed"),
        "agent_count": expected_decision.get("agent_count"),
    }
    features_by_candidate: dict[str, dict[str, Any]] = {}
    for row in trials:
        candidate_id = str(row["candidate_id"])
        candidate = indexed_candidates[candidate_id]
        trial_index = int(row["trial_index"])
        candidate_metadata = {
            **metadata,
            "candidate_kind": candidate.get("candidate_kind"),
            "actual_size": candidate.get("actual_size"),
            "selection_families": candidate.get("selection_families"),
            "agents": candidate.get("agents"),
        }
        if not repair_trial_semantics_valid(
            row,
            schema=STRIDE_TRIAL_SCHEMA,
            state_id=state_id,
            candidate_id=candidate_id,
            trial_index=trial_index,
            pp_seed=repairability_pp_seed(before_repair, trial_index),
            before_conflicts=before_conflicts,
            before_fingerprint=before_fingerprint,
            before_repair_fingerprint=before_repair,
            feature_schema_id=FROZEN_FEATURE_SCHEMA_ID,
            required_feature_names=required_names,
            expected_metadata=candidate_metadata,
        ):
            return False
        features = dict(row["features"])
        previous = features_by_candidate.setdefault(candidate_id, features)
        if previous != features:
            return False
    boundary_count = sum(
        candidate.get("candidate_kind") == "boundary_only"
        for candidate in indexed_candidates.values()
    )
    return (
        payload.get("boundary_candidate_count") == boundary_count
        and payload.get("base_candidate_count") == len(candidates) - boundary_count
    )


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    trial_indices = tuple(map(int, job["trial_indices"]))
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _artifact_valid(
            existing,
            run_fingerprint=run_fingerprint,
            state_id=str(decision["state_id"]),
            trial_indices=trial_indices,
            decision=decision,
        ):
            return {
                "state_id": str(decision["state_id"]),
                "state_file": str(output_path),
                "status": "resumed",
                "state_count": 1,
                "outcome_count": len(existing["trials"]),
                "error_count": 0,
                "candidate_count": len(existing["candidates"]),
                "boundary_candidate_count": int(existing["boundary_candidate_count"]),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"completed repairability artifact is invalid: {output_path}")

    replay = _replay_job(decision)
    state, source_manifest, source_trace_path = _source_target_state(decision)
    initial_fingerprint = state_fingerprint(state)
    if initial_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError(
            f"repairability replay mismatch for {decision['state_id']}"
        )
    initial_repair_fingerprint = repair_structure_fingerprint(state)
    restore_seed = repairability_restore_seed(initial_repair_fingerprint)
    environment, restored_state = restore_repair_state(
        replay, state, seed=restore_seed
    )
    restored_fingerprint = state_fingerprint(restored_state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    if before_conflicts <= 0 or bool(state.get("done")):
        raise ValueError("repairability state must be active and conflicting")

    base_proposal = {**dict(replay["proposal"]), **FULL_POOL_PROPOSAL}
    base_proposal.pop("topology_boundary", None)
    base_candidates, base_generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=base_proposal,
        state_hash=initial_fingerprint,
        verify_full_state=False,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    feature_engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={
            "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
        },
        dense_output=False,
    )
    augmented_proposal = {
        **base_proposal,
        "topology_boundary": dict(BOUNDARY_AUGMENTATION),
    }
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=augmented_proposal,
        state_hash=initial_fingerprint,
        verify_full_state=False,
        proposal_backend="optimized",
        shadow_validation=False,
        topology_static_grid=feature_engine.static_grid,
        topology_no_progress_streak=0,
        topology_remaining_wall_seconds=None,
    )
    if (
        state_fingerprint(state) != initial_fingerprint
        or repair_structure_fingerprint(_plain(environment.get_state()))
        != initial_repair_fingerprint
    ):
        raise RuntimeError("repairability candidate generation changed the state")
    base_by_id = {
        str(candidate["candidate_id"]): _candidate_signature(candidate)
        for candidate in base_candidates
    }
    augmented_by_id = {
        str(candidate["candidate_id"]): _candidate_signature(candidate)
        for candidate in candidates
    }
    if not set(base_by_id) <= set(augmented_by_id) or any(
        base_by_id[candidate_id] != augmented_by_id[candidate_id]
        for candidate_id in base_by_id
    ):
        raise RuntimeError("repairability augmentation changed the frozen base pool")
    boundary_ids = set(augmented_by_id) - set(base_by_id)
    if len(boundary_ids) > 2 or len(candidates) > 20:
        raise RuntimeError("repairability augmented pool exceeds its cap")

    feature_rows, feature_metrics = feature_engine.realized_rows(
        candidates, state_hash=initial_fingerprint
    )
    features_by_candidate = {
        str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
        for row in feature_rows
    }
    if set(features_by_candidate) != set(augmented_by_id) or any(
        len(values) != FROZEN_FEATURE_DIMENSION
        for values in features_by_candidate.values()
    ):
        raise RuntimeError("repairability candidate features are incomplete")

    trials: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        agents = sorted(map(int, candidate["agents"]))
        candidate_kind = "boundary_only" if candidate_id in boundary_ids else "base"
        candidate_record = {
            **candidate,
            "candidate_kind": candidate_kind,
            "actual_size": len(agents),
            "agents": agents,
            "selection_families": sorted(
                map(str, candidate.get("selection_families") or ())
            ),
        }
        candidate_records.append(candidate_record)
        for trial_index in trial_indices:
            branch_environment, branch_state = restore_repair_state(
                replay, state, seed=restore_seed
            )
            if (
                repair_structure_fingerprint(branch_state)
                != initial_repair_fingerprint
            ):
                raise RuntimeError("repairability paired branch restore changed")
            seed = repairability_pp_seed(initial_repair_fingerprint, trial_index)
            result = _plain(branch_environment.step(_paired_action(agents, seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair_fingerprint = repair_structure_fingerprint(after)
            repair_outcome = classify_repair_outcome(
                before_fingerprint=initial_repair_fingerprint,
                after_fingerprint=after_repair_fingerprint,
                replan_success=bool(metrics["replan_success"]),
                conflicts_before=before_conflicts,
                conflicts_after=conflicts_after,
                feasible=bool(after.get("feasible")),
            )
            trials.append(
                {
                    "schema": STRIDE_TRIAL_SCHEMA,
                    "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
                    "state_id": str(decision["state_id"]),
                    "candidate_id": candidate_id,
                    "candidate_kind": candidate_kind,
                    "actual_size": len(agents),
                    "selection_families": candidate_record["selection_families"],
                    "agents": agents,
                    "layout_family": str(decision.get("layout_mode", "unknown")),
                    "map_id": str(decision["map_id"]),
                    "task_id": str(decision["task_id"]),
                    "split": str(decision["research_split"]),
                    "source_split": str(decision["split"]),
                    "source_policy": str(decision["source_policy"]),
                    "decision_stage": str(decision["decision_stage"]),
                    "solver_seed": int(decision["solver_seed"]),
                    "agent_count": int(decision["agent_count"]),
                    "before_conflicts": before_conflicts,
                    "before_fingerprint": initial_fingerprint,
                    "before_repair_fingerprint": initial_repair_fingerprint,
                    "features": features_by_candidate[candidate_id],
                    "trial_index": trial_index,
                    "pp_seed": seed,
                    "feasible": bool(after.get("feasible")),
                    "replan_success": bool(metrics["replan_success"]),
                    "repair_outcome": repair_outcome,
                    "conflicts_after": conflicts_after,
                    "after_fingerprint": state_fingerprint(after),
                    "after_repair_fingerprint": after_repair_fingerprint,
                    "post_structure": post_structure_metrics(after),
                    "native_step_seconds": float(metrics["native_step_seconds"]),
                    "pp_replan_seconds": float(
                        metrics.get("pp_replan_seconds", 0.0)
                    ),
                }
            )

    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": str(decision["state_id"]),
        "decision": decision,
        "before_fingerprint": initial_fingerprint,
        "before_repair_fingerprint": initial_repair_fingerprint,
        "before_conflicts": before_conflicts,
        "state_restore": {
            "contract": TARGET_STATE_RESTORE_CONTRACT,
            "restore_seed": restore_seed,
            "source_trace_file": str(source_manifest["trace_file"]),
            "source_trace_path": str(source_trace_path),
            "source_full_fingerprint": initial_fingerprint,
            "restored_full_fingerprint": restored_fingerprint,
            "repair_structure_fingerprint": initial_repair_fingerprint,
        },
        "state_summary": summarize_initial_state_complexity(state),
        "base_candidate_generation": base_generation,
        "augmented_candidate_generation": generation,
        "base_candidate_count": len(base_candidates),
        "boundary_candidate_count": len(boundary_ids),
        "feature_metrics": feature_metrics,
        "candidates": candidate_records,
        "trials": trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": str(decision["state_id"]),
        "state_file": str(output_path),
        "status": "ok",
        "state_count": 1,
        "outcome_count": len(trials),
        "error_count": 0,
        "candidate_count": len(candidates),
        "boundary_candidate_count": len(boundary_ids),
        "trial_count": len(trials),
    }


def collect_repairability_trials(
    *,
    config_path: str | Path,
    selection_path: str | Path,
    output: str | Path,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("repairability workers must be positive")
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_repairability_label_config(config)
    if dict(config["candidate_pool"]) != {
        "base_families": ["target", "collision", "random"],
        "base_sizes": [4, 8, 16],
        "augmentation": "stride-topoboundary-v1",
        "boundary_size": 16,
        "boundary_core_budget": 4,
        "maximum_boundary_candidates": 2,
        "maximum_total_candidates": 20,
    }:
        raise ValueError("repairability collection candidate pool changed")
    selection_path = Path(selection_path).resolve()
    selected = load_stride_selection(selection_path)
    map_splits: dict[str, set[str]] = {}
    for row in selected:
        research_split = str(row.get("research_split", ""))
        if research_split not in {"train", "validation"}:
            raise ValueError(
                "repairability selection requires an explicit research_split"
            )
        map_splits.setdefault(str(row["map_id"]), set()).add(research_split)
    leaking_maps = sorted(
        map_id for map_id, splits in map_splits.items() if len(splits) != 1
    )
    if leaking_maps:
        raise ValueError(f"repairability selection leaks maps: {leaking_maps}")
    trial_indices = tuple(map(int, config["trial_indices"]))
    project_root = Path(__file__).resolve().parents[1]
    producer = producer_identity(
        project_root=project_root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    source_run_configs = {
        str(Path(str(row["source_root"])).resolve()): sha256_file(
            Path(str(row["source_root"])).resolve() / "run_config.json"
        )
        for row in selected
    }
    identity = {
        "schema": COLLECTION_SCHEMA,
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selected_state_ids": [str(row["state_id"]) for row in selected],
        "label_config_sha256": sha256_file(config_path),
        "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
        "feature_dimension": FROZEN_FEATURE_DIMENSION,
        "base_proposal": FULL_POOL_PROPOSAL,
        "boundary_augmentation": BOUNDARY_AUGMENTATION,
        "target_state_restore_contract": TARGET_STATE_RESTORE_CONTRACT,
        "pp_trial_indices": list(trial_indices),
        "source_run_config_sha256": dict(sorted(source_run_configs.items())),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    output = Path(output).resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("repairability output belongs to another collection")
        if not resume:
            raise ValueError("repairability output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_jsonl(output / "state_selection.jsonl", selected)
    jobs = []
    for row in selected:
        key = _fingerprint(
            {"state_id": row["state_id"], "before": row["before_fingerprint"]}
        )[:20]
        jobs.append(
            {
                "job_id": str(row["state_id"]),
                "row": row,
                "state_id": str(row["state_id"]),
                "solver_seed": int(row["solver_seed"]),
                "decision": row,
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "trial_indices": list(trial_indices),
                "resume": bool(resume),
            }
        )
    observed: list[dict[str, Any]] = []

    def update_status(result: dict[str, Any]) -> None:
        observed.append(result)
        failures = [
            row for row in observed if row.get("status") in {"error", "timeout"}
        ]
        _write_json(
            output / "collection_status.json",
            {
                "schema": COLLECTION_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "error_state_count": len(failures),
                "status": "running",
                "errors": [
                    {
                        "state_id": str(row.get("state_id", row.get("job_id"))),
                        "error": str(row.get("error")),
                    }
                    for row in failures
                ],
            },
        )

    _write_json(
        output / "collection_status.json",
        {
            "schema": COLLECTION_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "requested_state_count": len(jobs),
            "completed_state_count": 0,
            "error_state_count": 0,
            "status": "running",
            "errors": [],
        },
    )
    try:
        observed = _run_jobs(
            _collect_state,
            jobs,
            workers,
            phase="stride-repairability",
            output_root=output,
            run_fingerprint=run_fingerprint,
            on_result=update_status,
        )
    except BaseException as error:
        current = _read_json(output / "collection_status.json")
        _write_json(
            output / "collection_status.json",
            {
                **current,
                "status": (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
                ),
                "runner_error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    results = [row for row in observed if row.get("status") in {"ok", "resumed"}]
    errors = [
        {
            "state_id": str(row.get("state_id", row.get("job_id"))),
            "error": str(row.get("error")),
        }
        for row in observed
        if row.get("status") in {"error", "timeout"}
    ]
    all_trials: list[dict[str, Any]] = []
    candidate_counts: Counter[int] = Counter()
    boundary_counts: Counter[int] = Counter()
    selected_by_id = {str(row["state_id"]): row for row in selected}
    for result in sorted(results, key=lambda row: str(row["state_id"])):
        payload = _read_json(Path(str(result["state_file"])))
        if not _artifact_valid(
            payload,
            run_fingerprint=run_fingerprint,
            state_id=str(result["state_id"]),
            trial_indices=trial_indices,
            decision=selected_by_id.get(str(result["state_id"])),
        ):
            errors.append(
                {"state_id": str(result["state_id"]), "error": "invalid state artifact"}
            )
            continue
        candidate_counts[len(payload["candidates"])] += 1
        boundary_counts[int(payload["boundary_candidate_count"])] += 1
        all_trials.extend(payload["trials"])
    complete = len(results) == len(jobs) and not errors
    if complete:
        _write_jsonl(output / "repair_trials.jsonl", all_trials)
    report = {
        "schema": COLLECTION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "selection_sha256": identity["selection_sha256"],
        "requested_state_count": len(jobs),
        "completed_state_count": len(results),
        "new_state_count": sum(row["status"] == "ok" for row in results),
        "resumed_state_count": sum(row["status"] == "resumed" for row in results),
        "error_state_count": len(errors),
        "errors": errors,
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "boundary_candidate_count_distribution": {
            str(key): value for key, value in sorted(boundary_counts.items())
        },
        "trial_count": len(all_trials),
        "expected_trial_count": sum(
            key * value * len(trial_indices)
            for key, value in candidate_counts.items()
        ),
        "complete": complete,
    }
    _write_json(output / "collection_report.json", report)
    _write_json(
        output / "collection_status.json",
        {**report, "status": "complete" if complete else "error"},
    )
    return report


__all__ = [
    "collect_repairability_trials",
    "repairability_pp_seed",
    "repairability_restore_seed",
]
