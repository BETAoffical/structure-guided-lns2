from __future__ import annotations

import collections
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import resource
except ImportError:  # pragma: no cover - collection is WSL-only
    resource = None  # type: ignore[assignment]

from experiments._common import (
    closed_loop_producer_identity,
    contained_file,
    registered_input,
    sha256_file,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_closurepool_longtail import reconstruct_trace
from experiments.stride_collection import _paired_action
from experiments.stride_maze_tail_state_collection import (
    _fused_controller_kwargs,
    load_maze_tail_state_collection_config,
)
from experiments.stride_multivalue import (
    WorkerPreflightMeasurement,
    choose_dynamic_worker_count,
    jaccard_similarity,
    multihorizon_targets,
)
from experiments.stride_repairability_collection import (
    repairability_restore_seed,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.paretopool import (
    PARETOPOOL_BUDGETS,
    generate_paretopool_candidates,
)
from lns2_selector.runtime.temporal_state import (
    TemporalHistoryContext,
    temporal_history_context,
    temporal_state_identity,
)


CONFIG_SCHEMA = "lns2.stride.multivalue_collection_registration.v1"
STATE_MANIFEST_SCHEMA = "lns2.stride.multivalue_state_manifest.v1"
PREPARATION_REPORT_SCHEMA = "lns2.stride.multivalue_preparation_report.v1"
PREFLIGHT_REPORT_SCHEMA = "lns2.stride.multivalue_worker_preflight.v1"
ROLLOUT_STATE_SCHEMA = "lns2.stride.multivalue_rollout_state.v1"
ROLLOUT_PARTIAL_SCHEMA = "lns2.stride.multivalue_rollout_partial.v1"
COLLECTION_STATUS_SCHEMA = "lns2.stride.multivalue_collection_status.v1"
LABEL_ROW_SCHEMA = "lns2.stride.multivalue_label_row.v1"


def _registered(root: Path, specification: Mapping[str, Any]) -> Path:
    return registered_input(root, dict(specification), label="MultiValue collection")


def load_multivalue_collection_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_outcome_enriched_label_stability_pilot"
        or config.get("experiment_id") != "stride-multivalue-pilot-v1"
    ):
        raise ValueError("MultiValue pilot identity changed")
    inputs = {
        name: _registered(root, specification)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "root_checkpoints",
        "paretopool_gap_report",
        "state_collection_config",
        "runtime_config",
        "qualification_report",
    }:
        raise ValueError("MultiValue pilot input registry changed")
    rollout = dict(config.get("rollout") or {})
    if (
        tuple(map(int, rollout.get("horizons") or ())) != (1, 8, 32, 128)
        or tuple(map(str, rollout.get("teachers") or ()))
        != ("v2-full", "official-adaptive")
        or int(rollout.get("initial_paired_seed_count", -1)) != 8
        or int(rollout.get("extended_paired_seed_count", -1)) != 16
        or int(rollout.get("maximum_repair_decisions", -1)) != 128
        or float(rollout.get("wall_time_fuse_seconds", -1)) != 300.0
        or float(rollout.get("process_timeout_seconds", -1)) != 360.0
        or int(rollout.get("state_attempt_timeout_seconds", -1)) != 1800
        or int(rollout.get("maximum_state_attempts", -1)) != 4
        or int(rollout.get("no_progress_attempt_limit", -1)) != 2
        or rollout.get("right_censoring_is_valid") is not True
    ):
        raise ValueError("MultiValue pilot rollout or recovery contract changed")
    if tuple(map(int, config["paretopool"]["candidate_budgets"])) != PARETOPOOL_BUDGETS:
        raise ValueError("MultiValue pilot candidate budgets changed")
    return path, root, config, inputs


def _transition_neighborhood(transition: Mapping[str, Any]) -> tuple[int, ...]:
    metrics = transition.get("metrics")
    raw = metrics.get("neighborhood") if isinstance(metrics, Mapping) else None
    if not isinstance(raw, list) or not raw:
        action = transition.get("action")
        raw = action.get("agents") if isinstance(action, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("MultiValue history transition has no neighborhood")
    result = tuple(sorted(set(map(int, raw))))
    if len(result) != len(raw):
        raise ValueError("MultiValue history neighborhood contains duplicates")
    return result


def _edge_set(state: Mapping[str, Any]) -> set[tuple[int, int]]:
    return {tuple(sorted(map(int, edge))) for edge in state["conflict_edges"]}


def history_before_decision(
    states: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
    *,
    decision_index: int,
    history_limit: int = 8,
) -> TemporalHistoryContext:
    target = int(decision_index)
    if target < 0 or target >= len(states) or target > len(transitions):
        raise ValueError("MultiValue target decision is outside its source trace")
    prefix_transitions = list(transitions[:target])
    prefix_states = list(states[: target + 1])
    neighborhoods = [_transition_neighborhood(row) for row in prefix_transitions]
    exact_repeat = []
    max_jaccard = []
    for index, neighborhood in enumerate(neighborhoods):
        prior = neighborhoods[:index]
        exact_repeat.append(neighborhood in prior)
        max_jaccard.append(
            max(
                (jaccard_similarity(neighborhood, row) for row in prior),
                default=0.0,
            )
        )
    current_edges = _edge_set(prefix_states[-1])
    previous_edges = _edge_set(prefix_states[-2]) if len(prefix_states) > 1 else set()
    older_edges = set().union(*map(_edge_set, prefix_states[:-1]))
    streak = 1
    for state in reversed(prefix_states[:-1]):
        if _edge_set(state) != current_edges:
            break
        streak += 1
    repair_counts: collections.Counter[int] = collections.Counter(
        agent for neighborhood in neighborhoods for agent in neighborhood
    )
    candidate_ids = []
    pp_history = []
    for index, transition in enumerate(prefix_transitions):
        controller = transition.get("controller")
        candidate_ids.append(
            str(controller.get("selected_candidate_id", ""))
            if isinstance(controller, Mapping)
            else ""
        )
        metrics = dict(transition.get("metrics") or {})
        changed = repair_structure_fingerprint(prefix_states[index]) != (
            repair_structure_fingerprint(prefix_states[index + 1])
        )
        pp_history.append(
            (bool(metrics.get("replan_success")), not changed, changed)
        )
    return temporal_history_context(
        recent_neighborhoods=neighborhoods[-history_limit:],
        recent_neighborhood_exact_repeat=exact_repeat[-history_limit:],
        recent_neighborhood_max_jaccard=max_jaccard[-history_limit:],
        persistent_conflict_edges=current_edges & previous_edges,
        new_conflict_edges=current_edges - previous_edges,
        disappeared_conflict_edges=previous_edges - current_edges,
        reappeared_conflict_edges=current_edges & (older_edges - previous_edges),
        conflict_signature_streak=streak,
        agent_repair_counts=repair_counts,
        recent_candidate_ids=candidate_ids[-history_limit:],
        recent_pp_history=pp_history[-history_limit:],
    )


def _base_anchor(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    base = [
        dict(candidate)
        for candidate in checkpoint["candidate_pool"]
        if not candidate.get("structpool_family_groups")
    ]
    if not base or any(candidate.get("score") is None for candidate in base):
        raise ValueError("MultiValue checkpoint has no scored V2 base pool")
    anchor = max(
        base,
        key=lambda candidate: (
            float(candidate["score"]), str(candidate["candidate_id"])
        ),
    )
    anchor["multivalue_role"] = "v2_anchor"
    anchor["paretopool_budget_membership"] = []
    return anchor


def select_pilot_occurrences(
    checkpoints: Sequence[dict[str, Any]], *, per_map: int
) -> list[dict[str, Any]]:
    by_map_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for checkpoint in checkpoints:
        by_map_task.setdefault(str(checkpoint["map_id"]), {}).setdefault(
            str(checkpoint["task_id"]), []
        ).append(dict(checkpoint))
    selected: list[dict[str, Any]] = []
    for map_id, tasks in sorted(by_map_task.items()):
        ordered = {
            task_id: sorted(
                rows,
                key=lambda row: _fingerprint(
                    {
                        "purpose": "multivalue-pilot-outcome-blind-selection",
                        "map_id": map_id,
                        "task_id": task_id,
                        "case_id": str(row["case_id"]),
                        "decision_index": int(row["decision_index"]),
                    }
                ),
            )
            for task_id, rows in tasks.items()
        }
        map_rows: list[dict[str, Any]] = []
        depth = 0
        while len(map_rows) < per_map:
            added = False
            for task_id in sorted(ordered):
                if depth < len(ordered[task_id]):
                    map_rows.append(ordered[task_id][depth])
                    added = True
                    if len(map_rows) == per_map:
                        break
            if not added:
                raise ValueError(f"MultiValue map has too few pilot states: {map_id}")
            depth += 1
        selected.extend(map_rows)
    return sorted(
        selected,
        key=lambda row: (
            str(row["map_id"]),
            str(row["task_id"]),
            str(row["case_id"]),
            int(row["decision_index"]),
        ),
    )


def _source_collection(root: Path, checkpoint: Mapping[str, Any]) -> Path:
    return (
        root
        / "build"
        / "stride-tailswitch-v1"
        / "states"
        / _fingerprint({"state_id": str(checkpoint["state_id"])})[:20]
        / str(checkpoint["treatment_policy"])
    )


def _source_trace(
    root: Path, checkpoint: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = _source_collection(root, checkpoint)
    manifests = _read_jsonl(source / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError("MultiValue source trace manifest is invalid")
    manifest = dict(manifests[0])
    trace = reconstruct_trace(source, manifest)
    return source, manifest, trace


def _candidate_union(
    state: dict[str, Any],
    analysis: Any,
    anchor: dict[str, Any],
    history: TemporalHistoryContext,
    *,
    maximum_neighborhood_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {str(anchor["candidate_id"]): dict(anchor)}
    attempts: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, Any] = {}
    for budget in PARETOPOOL_BUDGETS:
        result = generate_paretopool_candidates(
            state,
            analysis,
            v2_anchor_agents=anchor["agents"],
            history=history,
            maximum_candidates=budget,
            maximum_neighborhood_size=maximum_neighborhood_size,
        )
        counts[str(budget)] = {
            "raw_candidate_count": result.raw_candidate_count,
            "selected_candidate_count": len(result.candidates),
            "history_candidate_count": result.history_candidate_count,
        }
        attempts[str(budget)] = result.attempts
        for candidate in result.candidates:
            identity = str(candidate["candidate_id"])
            existing = by_id.get(identity)
            if existing is None:
                existing = {**candidate, "multivalue_role": "paretopool_challenger"}
                existing["paretopool_budget_membership"] = []
                by_id[identity] = existing
            elif existing["agents"] != candidate["agents"]:
                raise ValueError("MultiValue candidate identity collision")
            existing["paretopool_budget_membership"].append(budget)
    candidates = [by_id[str(anchor["candidate_id"])] ] + sorted(
        [
            candidate
            for identity, candidate in by_id.items()
            if identity != str(anchor["candidate_id"])
        ],
        key=lambda candidate: str(candidate["candidate_id"]),
    )
    return candidates, {"budget_counts": counts, "attempts": attempts}


def prepare_multivalue_pilot(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config, inputs = load_multivalue_collection_config(config_path)
    checkpoints = _read_jsonl(inputs["root_checkpoints"])
    selected = select_pilot_occurrences(
        checkpoints, per_map=int(config["cohort"]["per_map_occurrences"])
    )
    expected = int(config["cohort"]["state_occurrence_count"])
    if len(selected) != expected:
        raise ValueError("MultiValue pilot state count changed")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for checkpoint in selected:
        state_path = contained_file(
            inputs["root_checkpoints"].parent,
            checkpoint["state_blob"],
            field="MultiValue state blob",
        )
        if sha256_file(state_path) != str(checkpoint["state_blob_sha256"]):
            raise ValueError("MultiValue checkpoint state blob hash changed")
        state = read_state_blob(state_path)
        if state_fingerprint(state) != str(checkpoint["state_fingerprint"]):
            raise ValueError("MultiValue checkpoint state fingerprint changed")
        source, manifest, trace = _source_trace(root, checkpoint)
        target = int(checkpoint["decision_index"])
        if target >= len(trace["states"]):
            raise ValueError("MultiValue source trace ended before target")
        source_state = trace["states"][target]
        if state_fingerprint(source_state) != state_fingerprint(state):
            raise ValueError("MultiValue source trace and state blob differ")
        history = history_before_decision(
            trace["states"], trace["transitions"], decision_index=target
        )
        temporal = temporal_state_identity(
            episode_id=str(manifest["episode_id"]),
            decision_index=target,
            state_fingerprint=str(checkpoint["state_fingerprint"]),
            history=history,
        )
        cache = TopologyAnalysisCache(state, backend="native")
        if cache.analysis is None:
            raise RuntimeError("MultiValue native topology analysis is missing")
        anchor = _base_anchor(checkpoint)
        candidates, pool_diagnostic = _candidate_union(
            state,
            cache.analysis,
            anchor,
            history,
            maximum_neighborhood_size=int(
                config["paretopool"]["maximum_neighborhood_size"]
            ),
        )
        engine = OnlineFeatureEngine(
            state,
            backend="native",
            required_features={
                "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
            },
        )
        if cache.last_native_prepared is not None:
            engine.prepare(state, prepared_native_analysis=cache.last_native_prepared)
        feature_rows, feature_metrics = engine.realized_rows(
            candidates, state_hash=str(checkpoint["state_fingerprint"])
        )
        features = {
            str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
            for row in feature_rows
        }
        enriched = []
        for position, candidate in enumerate(candidates):
            identity = str(candidate["candidate_id"])
            values = features[identity]
            if set(values) != set(PROFILE_FEATURE_NAMES["realized_dynamic"]):
                raise RuntimeError("MultiValue feature schema changed")
            enriched.append(
                {
                    **candidate,
                    "candidate_position": position,
                    "feature_schema": "lns2.realized_features.v2",
                    "feature_count": len(values),
                    "features": values,
                    "feature_sha256": _fingerprint(values),
                }
            )
        rows.append(
            {
                "schema": STATE_MANIFEST_SCHEMA,
                "state_occurrence_id": temporal["state_occurrence_id"],
                "temporal_identity": temporal,
                "case_id": str(checkpoint["case_id"]),
                "map_id": str(checkpoint["map_id"]),
                "task_id": str(checkpoint["task_id"]),
                "solver_seed": int(checkpoint["solver_seed"]),
                "state_id": str(checkpoint["state_id"]),
                "decision_index": target,
                "before_conflicts": int(checkpoint["conflict_pair_count"]),
                "state_blob": str(state_path),
                "state_blob_sha256": str(checkpoint["state_blob_sha256"]),
                "source_collection": str(source),
                "source_manifest": manifest,
                "source_trace_sha256": str(manifest["trace_sha256"]),
                "v2_anchor_candidate_id": str(anchor["candidate_id"]),
                "candidate_count": len(enriched),
                "candidates": enriched,
                "candidate_pool_sha256": _fingerprint(
                    [
                        {
                            "candidate_id": row["candidate_id"],
                            "agents": row["agents"],
                            "budgets": row["paretopool_budget_membership"],
                            "feature_sha256": row["feature_sha256"],
                        }
                        for row in enriched
                    ]
                ),
                "pool_diagnostic": pool_diagnostic,
                "feature_metrics": feature_metrics,
                "future_outcome_used": False,
            }
        )
    rows.sort(key=lambda row: str(row["state_occurrence_id"]))
    manifest_path = output / "state_manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    candidate_count = sum(int(row["candidate_count"]) for row in rows)
    initial_trials = int(config["rollout"]["initial_paired_seed_count"])
    teacher_count = len(config["rollout"]["teachers"])
    report = {
        "schema": PREPARATION_REPORT_SCHEMA,
        "scientific_status": "prepared_outcome_enriched_label_stability_pilot",
        "state_occurrence_count": len(rows),
        "unique_state_fingerprint_count": len(
            {str(row["temporal_identity"]["state_fingerprint"]) for row in rows}
        ),
        "map_counts": dict(collections.Counter(row["map_id"] for row in rows)),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "candidate_count": candidate_count,
        "anchor_count": len(rows),
        "challenger_count": candidate_count - len(rows),
        "initial_trial_count": candidate_count * initial_trials * teacher_count,
        "maximum_extended_trial_count": candidate_count
        * int(config["rollout"]["extended_paired_seed_count"])
        * teacher_count,
        "state_manifest": str(manifest_path),
        "state_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(path),
        "input_sha256": {
            name: sha256_file(value) for name, value in sorted(inputs.items())
        },
        "integrity": {
            "state_count": len(rows) == expected,
            "map_balance": set(collections.Counter(row["map_id"] for row in rows).values())
            == {int(config["cohort"]["per_map_occurrences"])},
            "all_candidate_features_124": all(
                int(candidate["feature_count"]) == 124
                for row in rows
                for candidate in row["candidates"]
            ),
            "one_anchor_per_state": all(
                sum(
                    candidate["multivalue_role"] == "v2_anchor"
                    for candidate in row["candidates"]
                )
                == 1
                for row in rows
            ),
            "future_outcome_not_used": all(
                row["future_outcome_used"] is False for row in rows
            ),
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    report["integrity_passed"] = all(report["integrity"].values())
    _write_json(output / "preparation_report.json", report)
    return report


def _producer(root: Path) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_multivalue.py",
            "experiments/stride_multivalue_collection.py",
            "lns2_selector/runtime/paretopool.py",
            "lns2_selector/runtime/temporal_state.py",
        ),
        native_required=True,
    )


def _rollout_runtime(
    config: dict[str, Any], inputs: dict[str, Path], output: Path
) -> Path:
    source = _read_json(inputs["runtime_config"])
    rollout = dict(config["rollout"])
    runtime = {
        **source,
        "max_decisions": int(rollout["maximum_repair_decisions"]),
        "metric_iteration_budget": int(rollout["maximum_repair_decisions"]),
        "wall_time_budget_seconds": float(rollout["wall_time_fuse_seconds"]),
        "episode_process_timeout_seconds": float(rollout["process_timeout_seconds"]),
        "multivalue_parent_runtime_sha256": sha256_file(inputs["runtime_config"]),
    }
    destination = output / "rollout_runtime_config.json"
    if destination.is_file():
        if _read_json(destination) != runtime:
            raise ValueError("MultiValue rollout runtime identity changed")
    else:
        _write_json(destination, runtime)
    return destination


def _teacher_kwargs(
    root: Path, parent: dict[str, Any], teacher: str
) -> tuple[str, dict[str, Any]]:
    kwargs = _fused_controller_kwargs(root, parent, "v2-full")
    if teacher == "v2-full":
        return "realized_dynamic", kwargs
    if teacher == "official-adaptive":
        return "official_adaptive", {**kwargs, "controller": "official_adaptive"}
    raise ValueError(f"unknown MultiValue teacher: {teacher}")


def _paired_pp_seed(state_occurrence_id: str, trial_index: int) -> int:
    return int(
        _fingerprint(
            {
                "purpose": "multivalue-paired-forced-action",
                "state_occurrence_id": str(state_occurrence_id),
                "trial_index": int(trial_index),
            }
        )[:8],
        16,
    ) & 0x7FFFFFFF


def _episode_job_id(teacher: str, candidate_id: str, trial_index: int) -> str:
    return _fingerprint(
        {
            "teacher": str(teacher),
            "candidate_id": str(candidate_id),
            "trial_index": int(trial_index),
        }
    )


def _episode_collection(
    state_root: Path,
    *,
    teacher: str,
    candidate_position: int,
    trial_index: int,
) -> Path:
    return (
        state_root
        / "rollouts"
        / str(teacher)
        / f"candidate_{int(candidate_position):03d}"
        / f"trial_{int(trial_index):02d}"
    )


def _episode_override(
    state_record: dict[str, Any],
    candidate: dict[str, Any],
    *,
    trial_index: int,
    pp_seed: int,
) -> dict[str, Any]:
    state = read_state_blob(Path(str(state_record["state_blob"])))
    before_repair = repair_structure_fingerprint(state)
    return {
        "schema": "lns2.stride.multivalue_episode_override.v1",
        "state_id": str(state_record["state_id"]),
        "initial_restore": {
            "collection_root": str(state_record["source_collection"]),
            "manifest": dict(state_record["source_manifest"]),
            "decision_index": int(state_record["decision_index"]),
            "expected_fingerprint": str(
                state_record["temporal_identity"]["state_fingerprint"]
            ),
            "repair_structure_fingerprint": before_repair,
            "expected_conflicts": int(state_record["before_conflicts"]),
            "restore_seed": repairability_restore_seed(before_repair),
        },
        "forced_first_action": _paired_action(
            list(map(int, candidate["agents"])), pp_seed
        ),
        "forced_candidate_id": str(candidate["candidate_id"]),
        "forced_candidate_role": str(candidate["multivalue_role"]),
        "forced_selection_families": list(
            map(str, candidate.get("selection_families") or ())
        ),
        "multivalue_trial_index": int(trial_index),
    }


def _single_rollout(
    job: dict[str, Any],
    state_record: dict[str, Any],
    candidate: dict[str, Any],
    *,
    teacher: str,
    trial_index: int,
) -> dict[str, Any]:
    state_root = Path(str(job["state_root"]))
    collection = _episode_collection(
        state_root,
        teacher=teacher,
        candidate_position=int(candidate["candidate_position"]),
        trial_index=trial_index,
    )
    phase, kwargs = _teacher_kwargs(
        Path(str(job["root"])), dict(job["parent"]), teacher
    )
    key = (str(state_record["task_id"]), int(state_record["solver_seed"]))
    pp_seed = _paired_pp_seed(str(state_record["state_occurrence_id"]), trial_index)
    override = _episode_override(
        state_record, candidate, trial_index=trial_index, pp_seed=pp_seed
    )
    run_closed_loop_collection(
        Path(str(job["dataset"])),
        Path(str(job["runtime"])),
        collection,
        phase=phase,
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        cohort_job_keys={
            (str(task_id), int(seed))
            for task_id, seed in job["cohort_job_keys"]
        },
        job_keys={key},
        qualification_source=Path(str(job["qualification"])),
        episode_overrides={key: override},
        use_global_collection_lock=False,
        **kwargs,
    )
    manifest_path = collection / f"{phase}_manifest.jsonl"
    manifests = [
        row
        for row in _read_jsonl(manifest_path)
        if (str(row["task_id"]), int(row["solver_seed"])) == key
    ]
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise RuntimeError("MultiValue rollout did not produce one valid episode")
    manifest = dict(manifests[0])
    summary = dict(manifest.get("summary") or {})
    controller_totals = dict(summary.get("controller_totals") or {})
    if int(controller_totals.get("forced_first_action_count", -1)) != 1:
        raise RuntimeError("MultiValue rollout did not force exactly one action")
    if str(summary.get("stop_reason")) not in {
        "success",
        "repair_limit",
        "wall_timeout",
    }:
        raise RuntimeError("MultiValue rollout has an invalid stopping reason")
    return {
        "episode_job_id": _episode_job_id(
            teacher, str(candidate["candidate_id"]), trial_index
        ),
        "teacher": teacher,
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_position": int(candidate["candidate_position"]),
        "trial_index": int(trial_index),
        "pp_seed": pp_seed,
        "collection": str(collection),
        "manifest": manifest,
        "trace_sha256": str(manifest["trace_sha256"]),
        "status": "ok",
    }


def _partial_payload(
    job: dict[str, Any], episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": ROLLOUT_PARTIAL_SCHEMA,
        "run_fingerprint": str(job["run_fingerprint"]),
        "state_occurrence_id": str(job["state_record"]["state_occurrence_id"]),
        "state_record_sha256": _fingerprint(job["state_record"]),
        "complete": False,
        "completed_episode_job_ids": sorted(
            str(row["episode_job_id"]) for row in episodes
        ),
        "episodes": sorted(episodes, key=lambda row: str(row["episode_job_id"])),
    }


def _collect_rollout_state(job: dict[str, Any]) -> dict[str, Any]:
    state_record = dict(job["state_record"])
    output_path = Path(str(job["output_path"]))
    partial_path = output_path.with_name(output_path.name + ".partial")
    expected = {
        _episode_job_id(
            teacher, str(candidate["candidate_id"]), trial_index
        )
        for teacher in job["teachers"]
        for candidate in state_record["candidates"]
        for trial_index in job["trial_indices"]
    }
    if output_path.is_file():
        payload = _read_json(output_path)
        if (
            payload.get("schema") == ROLLOUT_STATE_SCHEMA
            and payload.get("run_fingerprint") == job["run_fingerprint"]
            and payload.get("complete") is True
            and set(payload.get("completed_episode_job_ids") or ()) == expected
        ):
            return {
                "state_occurrence_id": str(state_record["state_occurrence_id"]),
                "status": "resumed",
                "episode_count": len(expected),
                "candidate_count": len(state_record["candidates"]),
            }
        raise ValueError("invalid completed MultiValue state artifact")
    episodes: list[dict[str, Any]] = []
    if partial_path.is_file():
        partial = _read_json(partial_path)
        if (
            partial.get("schema") != ROLLOUT_PARTIAL_SCHEMA
            or partial.get("run_fingerprint") != job["run_fingerprint"]
            or partial.get("state_record_sha256") != _fingerprint(state_record)
        ):
            raise ValueError("invalid MultiValue partial state artifact")
        episodes = list(partial["episodes"])
    completed = {str(row["episode_job_id"]) for row in episodes}
    for teacher in job["teachers"]:
        for candidate in state_record["candidates"]:
            for trial_index in job["trial_indices"]:
                identity = _episode_job_id(
                    teacher, str(candidate["candidate_id"]), trial_index
                )
                if identity in completed:
                    continue
                row = _single_rollout(
                    job,
                    state_record,
                    dict(candidate),
                    teacher=str(teacher),
                    trial_index=int(trial_index),
                )
                episodes.append(row)
                completed.add(identity)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(partial_path, _partial_payload(job, episodes))
    payload = {
        **_partial_payload(job, episodes),
        "schema": ROLLOUT_STATE_SCHEMA,
        "complete": True,
    }
    if set(payload["completed_episode_job_ids"]) != expected:
        raise RuntimeError("MultiValue state episode matrix is incomplete")
    _write_json(output_path, payload)
    partial_path.unlink(missing_ok=True)
    return {
        "state_occurrence_id": str(state_record["state_occurrence_id"]),
        "status": "ok",
        "episode_count": len(expected),
        "candidate_count": len(state_record["candidates"]),
    }


def _failed_rollout_state(
    job: dict[str, Any], status: str, message: str
) -> dict[str, Any]:
    return {
        "state_occurrence_id": str(job["state_record"]["state_occurrence_id"]),
        "status": status,
        "error": message,
        "episode_count": 0,
        "candidate_count": 0,
    }


def _qualification(
    root: Path,
    config: dict[str, Any],
    inputs: dict[str, Path],
    parent: dict[str, Any],
    state_rows: list[dict[str, Any]],
    output: Path,
    runtime: Path,
) -> Path:
    destination = output / "qualification"
    keys = {
        (str(row["task_id"]), int(row["solver_seed"])) for row in state_rows
    }
    run_closed_loop_collection(
        (root / str(parent["cohort"]["dataset"])).resolve(),
        runtime,
        destination,
        phase="qualify",
        workers=min(8, len(keys)),
        resume=destination.joinpath("run_config.json").is_file(),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_source=inputs["qualification_report"].parent,
        **_fused_controller_kwargs(root, parent, "v2-full"),
    )
    return destination


def _memory_available_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
        return int(values["MemAvailable"])


def _preflight_episode(job: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    state_record = dict(job["state_record"])
    candidate = next(
        row
        for row in state_record["candidates"]
        if row["multivalue_role"] == "v2_anchor"
    )
    row = _single_rollout(
        job,
        state_record,
        dict(candidate),
        teacher="v2-full",
        trial_index=0,
    )
    if resource is None:
        raise RuntimeError("MultiValue worker preflight requires WSL resource usage")
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    child_peak = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024
    return {
        "state_occurrence_id": str(state_record["state_occurrence_id"]),
        "status": "ok",
        "episode_job_id": row["episode_job_id"],
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak + child_peak,
    }


def run_worker_preflight(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config, inputs = load_multivalue_collection_config(config_path)
    output = Path(output).resolve()
    state_rows = _read_jsonl(output / "state_manifest.jsonl")
    if len(state_rows) != int(config["cohort"]["state_occurrence_count"]):
        raise ValueError("MultiValue preflight requires prepared state manifest")
    _parent_path, parent_root, parent = load_maze_tail_state_collection_config(
        inputs["state_collection_config"]
    )
    if parent_root != root:
        raise ValueError("MultiValue parent root changed")
    runtime = _rollout_runtime(config, inputs, output)
    qualification = _qualification(
        root, config, inputs, parent, state_rows, output, runtime
    )
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    producer = _producer(root)
    cohort_job_keys = sorted(
        {
            (str(row["task_id"]), int(row["solver_seed"])) for row in state_rows
        }
    )
    measurements: list[WorkerPreflightMeasurement] = []
    detail: list[dict[str, Any]] = []
    for requested in map(int, config["dynamic_workers"]["heavy_candidates"]):
        actual = min(requested, len(state_rows))
        preflight_root = output / "worker_preflight" / f"workers_{requested:02d}"
        jobs = [
            {
                "job_id": str(row["state_occurrence_id"]),
                "root": str(root),
                "parent": parent,
                "dataset": str(dataset),
                "runtime": str(runtime),
                "qualification": str(qualification),
                "cohort_job_keys": cohort_job_keys,
                "state_root": str(
                    preflight_root / "states" / str(row["state_occurrence_id"])
                ),
                "state_record": row,
            }
            for row in state_rows
        ]
        started = time.monotonic()
        results = _run_jobs(
            _preflight_episode,
            jobs,
            actual,
            phase=f"stride-multivalue-worker-preflight-{requested}",
            output_root=preflight_root,
            run_fingerprint=_fingerprint(
                {
                    "config": sha256_file(path),
                    "states": sha256_file(output / "state_manifest.jsonl"),
                    "workers": requested,
                    "producer": producer,
                }
            ),
            timeout_seconds=float(config["rollout"]["process_timeout_seconds"]),
            failure_result=_failed_rollout_state,
        )
        elapsed = time.monotonic() - started
        errors = sum(row.get("status") != "ok" for row in results)
        peak = sum(
            sorted(
                (int(row.get("peak_rss_bytes", 0)) for row in results),
                reverse=True,
            )[:actual]
        )
        measurement = WorkerPreflightMeasurement(
            requested,
            len(results) / elapsed if elapsed > 0 else 0.0,
            peak,
            errors,
        )
        measurements.append(measurement)
        detail.append(
            {
                "requested_workers": requested,
                "actual_workers": actual,
                "episode_count": len(results),
                "wall_seconds": elapsed,
                "throughput_per_second": measurement.throughput_per_second,
                "conservative_peak_rss_bytes": peak,
                "error_count": errors,
            }
        )
    cpu_count = int(os.cpu_count() or 1)
    available_memory = _memory_available_bytes()
    selected = choose_dynamic_worker_count(
        measurements,
        logical_cpu_count=cpu_count,
        available_memory_bytes=available_memory,
        maximum_workers=int(config["dynamic_workers"]["maximum_workers"]),
        reserved_logical_cpus=int(
            config["dynamic_workers"]["reserved_logical_cpus"]
        ),
        memory_fraction=float(
            config["dynamic_workers"]["available_memory_fraction"]
        ),
    )
    report = {
        "schema": PREFLIGHT_REPORT_SCHEMA,
        "scientific_status": "completed_outcome_blind_worker_preflight",
        "measurements": detail,
        "selected_worker_count": selected,
        "logical_cpu_count": cpu_count,
        "available_memory_bytes": available_memory,
        "producer": producer,
        "state_manifest_sha256": sha256_file(output / "state_manifest.jsonl"),
        "runtime_sha256": sha256_file(runtime),
        "all_errors_zero": all(row["error_count"] == 0 for row in detail),
        "outcomes_used_for_worker_selection": False,
        "claim_boundary": dict(config["claim_boundary"]),
    }
    _write_json(output / "worker_preflight_report.json", report)
    return report


def _collection_status(
    output: Path,
    *,
    run_fingerprint: str,
    phase: str,
    results: Mapping[str, dict[str, Any]],
    attempt_history: list[dict[str, Any]],
    current_attempt: int,
    maximum_attempts: int,
    status: str,
) -> None:
    rows = list(results.values())
    _write_json(
        output / f"{phase}_collection_status.json",
        {
            "schema": COLLECTION_STATUS_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "phase": phase,
            "status": status,
            "completed_state_count": sum(
                row.get("status") in {"ok", "resumed"} for row in rows
            ),
            "completed_episode_count": sum(
                int(row.get("episode_count", 0))
                for row in rows
                if row.get("status") in {"ok", "resumed"}
            ),
            "error_state_count": sum(row.get("status") == "error" for row in rows),
            "timeout_state_count": sum(
                row.get("status") == "timeout" for row in rows
            ),
            "active_jobs": [],
            "current_attempt": current_attempt,
            "maximum_state_attempts": maximum_attempts,
            "attempt_failure_count": len(attempt_history),
            "attempt_history": attempt_history,
            "errors": [
                row
                for row in rows
                if row.get("status") in {"error", "timeout"}
            ],
        },
    )


def run_multivalue_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "initial",
) -> dict[str, Any]:
    path, root, config, inputs = load_multivalue_collection_config(config_path)
    if phase not in {"initial", "extension"}:
        raise ValueError("MultiValue collection phase must be initial or extension")
    output = Path(output).resolve()
    state_rows = _read_jsonl(output / "state_manifest.jsonl")
    preflight = _read_json(output / "worker_preflight_report.json")
    if (
        preflight.get("schema") != PREFLIGHT_REPORT_SCHEMA
        or preflight.get("all_errors_zero") is not True
        or preflight.get("outcomes_used_for_worker_selection") is not False
        or preflight.get("state_manifest_sha256")
        != sha256_file(output / "state_manifest.jsonl")
    ):
        raise ValueError("MultiValue collection requires a passed worker preflight")
    if phase == "initial":
        trial_indices = tuple(range(8))
        selected_rows = state_rows
    else:
        initial_report = _read_json(output / "initial_stability_report.json")
        extension_ids = set(map(str, initial_report.get("extension_state_ids") or ()))
        if not extension_ids:
            return {
                "schema": COLLECTION_STATUS_SCHEMA,
                "phase": phase,
                "status": "not_required",
                "completed_state_count": 0,
                "completed_episode_count": 0,
            }
        selected_rows = [
            row for row in state_rows if str(row["state_occurrence_id"]) in extension_ids
        ]
        if {str(row["state_occurrence_id"]) for row in selected_rows} != extension_ids:
            raise ValueError("MultiValue extension state identity changed")
        trial_indices = tuple(range(8, 16))
    _parent_path, parent_root, parent = load_maze_tail_state_collection_config(
        inputs["state_collection_config"]
    )
    if parent_root != root:
        raise ValueError("MultiValue parent root changed")
    runtime = _rollout_runtime(config, inputs, output)
    qualification = _qualification(
        root, config, inputs, parent, state_rows, output, runtime
    )
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    producer = _producer(root)
    cohort_job_keys = sorted(
        {
            (str(row["task_id"]), int(row["solver_seed"])) for row in state_rows
        }
    )
    run_fingerprint = _fingerprint(
        {
            "config": sha256_file(path),
            "state_manifest": sha256_file(output / "state_manifest.jsonl"),
            "runtime": sha256_file(runtime),
            "producer": producer,
            "phase": phase,
            "trial_indices": trial_indices,
            "selected_state_ids": [
                str(row["state_occurrence_id"]) for row in selected_rows
            ],
        }
    )
    jobs = [
        {
            "job_id": str(row["state_occurrence_id"]),
            "root": str(root),
            "parent": parent,
            "dataset": str(dataset),
            "runtime": str(runtime),
            "qualification": str(qualification),
            "cohort_job_keys": cohort_job_keys,
            "state_root": str(
                output / "states" / str(row["state_occurrence_id"])
            ),
            "state_record": row,
            "teachers": list(config["rollout"]["teachers"]),
            "trial_indices": list(trial_indices),
            "output_path": str(
                output
                / "rollout_states"
                / phase
                / f"{str(row['state_occurrence_id'])}.json"
            ),
            "run_fingerprint": run_fingerprint,
        }
        for row in selected_rows
    ]
    worker_count = min(int(preflight["selected_worker_count"]), len(jobs))
    maximum_attempts = int(config["rollout"]["maximum_state_attempts"])
    attempt_timeout = float(config["rollout"]["state_attempt_timeout_seconds"])
    no_progress_limit = int(config["rollout"]["no_progress_attempt_limit"])
    observed: dict[str, dict[str, Any]] = {}
    attempt_history: list[dict[str, Any]] = []
    last_progress: dict[str, int] = {}
    no_progress: dict[str, int] = {}
    pending = jobs
    current_attempt = 0
    _collection_status(
        output,
        run_fingerprint=run_fingerprint,
        phase=phase,
        results=observed,
        attempt_history=attempt_history,
        current_attempt=current_attempt,
        maximum_attempts=maximum_attempts,
        status="running",
    )
    for current_attempt in range(1, maximum_attempts + 1):
        attempt_results: dict[str, dict[str, Any]] = {}

        def record(result: dict[str, Any]) -> None:
            state_id = str(result["state_occurrence_id"])
            attempt_results[state_id] = result
            if result.get("status") in {"ok", "resumed"}:
                observed[state_id] = result
            _collection_status(
                output,
                run_fingerprint=run_fingerprint,
                phase=phase,
                results=observed,
                attempt_history=attempt_history,
                current_attempt=current_attempt,
                maximum_attempts=maximum_attempts,
                status="running",
            )

        _run_jobs(
            _collect_rollout_state,
            pending,
            worker_count,
            phase=f"stride-multivalue-{phase}-attempt-{current_attempt}",
            output_root=output,
            run_fingerprint=run_fingerprint,
            timeout_seconds=attempt_timeout,
            on_result=record,
            failure_result=_failed_rollout_state,
        )
        retry = []
        for job in pending:
            state_id = str(job["state_record"]["state_occurrence_id"])
            result = attempt_results[state_id]
            if result.get("status") in {"ok", "resumed"}:
                continue
            partial_path = Path(str(job["output_path"])).with_name(
                Path(str(job["output_path"])).name + ".partial"
            )
            completed = (
                len(_read_json(partial_path).get("completed_episode_job_ids") or ())
                if partial_path.is_file()
                else 0
            )
            previous = last_progress.get(state_id, 0)
            no_progress[state_id] = (
                no_progress.get(state_id, 0) + 1 if completed <= previous else 0
            )
            last_progress[state_id] = completed
            attempt_history.append(
                {
                    "attempt": current_attempt,
                    "state_occurrence_id": state_id,
                    "status": str(result["status"]),
                    "error": str(result.get("error")),
                    "completed_episode_checkpoint_count": completed,
                    "consecutive_no_progress_attempts": no_progress[state_id],
                }
            )
            if current_attempt < maximum_attempts and no_progress[state_id] < no_progress_limit:
                retry.append(job)
            else:
                observed[state_id] = result
        pending = retry
        if not pending:
            break
    complete = [
        row for row in observed.values() if row.get("status") in {"ok", "resumed"}
    ]
    final_status = "complete" if len(complete) == len(jobs) else "failed"
    _collection_status(
        output,
        run_fingerprint=run_fingerprint,
        phase=phase,
        results=observed,
        attempt_history=attempt_history,
        current_attempt=current_attempt,
        maximum_attempts=maximum_attempts,
        status=final_status,
    )
    return _read_json(output / f"{phase}_collection_status.json")


def _candidate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    for horizon in (1, 8, 32, 128):
        values = [dict(row["targets"]["horizons"][str(horizon)]) for row in rows]
        auc = [
            float(value["normalized_conflict_auc"])
            for value in values
            if value["normalized_conflict_auc"] is not None
        ]
        residual = [
            float(value["original_conflict_edge_residual_rate"])
            for value in values
            if value["original_conflict_edge_residual_rate"] is not None
        ]
        new_edges = [
            float(value["new_conflict_edge_rate"])
            for value in values
            if value["new_conflict_edge_rate"] is not None
        ]
        horizons[str(horizon)] = {
            "trial_count": len(values),
            "feasible_probability": sum(
                bool(value["feasible_event_observed"]) for value in values
            )
            / len(values),
            "right_censored_fraction": sum(
                bool(value["right_censored"]) for value in values
            )
            / len(values),
            "mean_normalized_conflict_auc": sum(auc) / len(auc) if auc else None,
            "mean_original_edge_residual_rate": (
                sum(residual) / len(residual) if residual else None
            ),
            "mean_new_conflict_edge_rate": (
                sum(new_edges) / len(new_edges) if new_edges else None
            ),
        }
    return {"horizons": horizons}


def _value_order_key(candidate_id: str, summary: Mapping[str, Any]) -> tuple[Any, ...]:
    horizons = summary["horizons"]

    def loss(horizon: int, name: str) -> float:
        value = horizons[str(horizon)][name]
        return float(value) if value is not None else float("inf")

    return (
        -float(horizons["128"]["feasible_probability"]),
        float(horizons["128"]["right_censored_fraction"]),
        loss(128, "mean_normalized_conflict_auc"),
        loss(32, "mean_normalized_conflict_auc"),
        loss(128, "mean_original_edge_residual_rate"),
        loss(128, "mean_new_conflict_edge_rate"),
        str(candidate_id),
    )


def _rank_correlation(left: list[str], right: list[str]) -> float:
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("MultiValue rank lists differ")
    first = {candidate: index for index, candidate in enumerate(left)}
    second = {candidate: index for index, candidate in enumerate(right)}
    count = len(left)
    squared = sum((first[candidate] - second[candidate]) ** 2 for candidate in left)
    return 1.0 - (6.0 * squared) / (count * (count * count - 1))


def _direction_against_anchor(
    candidate_id: str,
    candidate_summary: Mapping[str, Any],
    anchor_id: str,
    anchor_summary: Mapping[str, Any],
) -> int:
    candidate_key = _value_order_key(candidate_id, candidate_summary)
    anchor_key = _value_order_key(anchor_id, anchor_summary)
    if candidate_key < anchor_key:
        return 1
    if candidate_key > anchor_key:
        return -1
    return 0


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _stability_group(
    state_rows: Sequence[dict[str, Any]],
    direction_rows: Sequence[dict[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    teacher_rows = [
        teacher
        for state in state_rows
        for teacher in state["teachers"]
    ]
    top3 = _mean(float(row["top3_overlap"]) for row in teacher_rows)
    correlation = _mean(
        float(row["paired_rank_correlation"]) for row in teacher_rows
    )
    regrets = [
        float(row["cross_seed_normalized_regret"])
        for row in teacher_rows
        if row["cross_seed_normalized_regret"] is not None
    ]
    regret = _mean(regrets)
    direction_agreement = _mean(
        float(row["directions_agree"]) for row in direction_rows
    )
    checks = {
        "seed_half_top3_overlap": top3 is not None
        and top3 >= float(gates["seed_half_top3_overlap_minimum"]),
        "paired_rank_correlation": correlation is not None
        and correlation >= float(gates["paired_rank_correlation_minimum"]),
        "cross_teacher_direction_agreement": direction_agreement is not None
        and direction_agreement
        >= float(gates["cross_teacher_direction_agreement_minimum"]),
        "cross_seed_normalized_regret": regret is not None
        and regret <= float(gates["cross_seed_normalized_regret_maximum"]),
    }
    return {
        "state_count": len(state_rows),
        "teacher_state_count": len(teacher_rows),
        "candidate_direction_count": len(direction_rows),
        "comparable_regret_count": len(regrets),
        "mean_seed_half_top3_overlap": top3,
        "mean_paired_rank_correlation": correlation,
        "cross_teacher_direction_agreement": direction_agreement,
        "mean_cross_seed_normalized_regret": regret,
        "checks": checks,
        "passed": all(checks.values()),
    }


def analyze_multivalue_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    include_extension: bool = False,
) -> dict[str, Any]:
    _path, _root, config, _inputs = load_multivalue_collection_config(config_path)
    output = Path(output).resolve()
    states = _read_jsonl(output / "state_manifest.jsonl")
    state_by_id = {str(row["state_occurrence_id"]): row for row in states}
    phases = ["initial", "extension"] if include_extension else ["initial"]
    episode_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for phase in phases:
        phase_root = output / "rollout_states" / phase
        for state_id, state in state_by_id.items():
            state_path = phase_root / f"{state_id}.json"
            if not state_path.is_file():
                continue
            payload = _read_json(state_path)
            if payload.get("schema") != ROLLOUT_STATE_SCHEMA or payload.get("complete") is not True:
                raise ValueError("MultiValue rollout state artifact is invalid")
            for episode in payload["episodes"]:
                episode_rows.append((state, dict(episode)))
    labels: list[dict[str, Any]] = []
    for state, episode in episode_rows:
        candidate = next(
            row
            for row in state["candidates"]
            if str(row["candidate_id"]) == str(episode["candidate_id"])
        )
        collection = Path(str(episode["collection"]))
        manifest = dict(episode["manifest"])
        trace = reconstruct_trace(collection, manifest)
        summary = dict(manifest["summary"])
        controller_totals = dict(summary.get("controller_totals") or {})
        if not trace["transitions"]:
            raise ValueError("MultiValue rollout has no forced transition")
        if state_fingerprint(trace["states"][0]) != str(
            state["temporal_identity"]["state_fingerprint"]
        ):
            raise ValueError("MultiValue rollout initial state fingerprint changed")
        if len(_edge_set(trace["states"][0])) != int(state["before_conflicts"]):
            raise ValueError("MultiValue rollout initial conflict count changed")
        if _transition_neighborhood(trace["transitions"][0]) != tuple(
            sorted(map(int, candidate["agents"]))
        ):
            raise ValueError("MultiValue rollout forced neighborhood changed")
        if int(controller_totals.get("forced_first_action_count", -1)) != 1:
            raise ValueError("MultiValue rollout forced action count changed")
        if int(summary.get("invalid_action_count", -1)) != 0:
            raise ValueError("MultiValue rollout contains an invalid action")
        if int(summary.get("fingerprint_mismatch_count", -1)) != 0:
            raise ValueError("MultiValue rollout contains a fingerprint mismatch")
        targets = multihorizon_targets(
            trace["states"],
            trace["transitions"],
            stop_reason=str(summary["stop_reason"]),
        )
        expected_seed = _paired_pp_seed(
            str(state["state_occurrence_id"]), int(episode["trial_index"])
        )
        if int(episode["pp_seed"]) != expected_seed:
            raise ValueError("MultiValue paired PP seed changed")
        labels.append(
            {
                "schema": LABEL_ROW_SCHEMA,
                "state_occurrence_id": str(state["state_occurrence_id"]),
                "state_fingerprint": str(
                    state["temporal_identity"]["state_fingerprint"]
                ),
                "history_context_sha256": str(
                    state["temporal_identity"]["history_context_sha256"]
                ),
                "map_id": str(state["map_id"]),
                "task_id": str(state["task_id"]),
                "teacher": str(episode["teacher"]),
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_role": str(candidate["multivalue_role"]),
                "trial_index": int(episode["trial_index"]),
                "pp_seed": expected_seed,
                "feature_schema": str(candidate["feature_schema"]),
                "feature_sha256": str(candidate["feature_sha256"]),
                "features": dict(candidate["features"]),
                "targets": targets,
                "trace_sha256": str(episode["trace_sha256"]),
                "runtime_is_not_a_label": True,
                "ttf_is_not_a_label": True,
            }
        )
    labels.sort(
        key=lambda row: (
            str(row["state_occurrence_id"]),
            str(row["teacher"]),
            str(row["candidate_id"]),
            int(row["trial_index"]),
        )
    )
    _write_jsonl(output / "multihorizon_labels.jsonl", labels)
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in labels:
        by_group[
            (
                str(row["state_occurrence_id"]),
                str(row["teacher"]),
                str(row["candidate_id"]),
            )
        ].append(row)
    state_stability: list[dict[str, Any]] = []
    extension_ids: set[str] = set()
    full_summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for state_id, state in sorted(state_by_id.items()):
        teacher_rows = []
        for teacher in config["rollout"]["teachers"]:
            available_trials = sorted(
                {
                    int(row["trial_index"])
                    for (candidate_state, candidate_teacher, _candidate), rows in by_group.items()
                    if candidate_state == state_id and candidate_teacher == teacher
                    for row in rows
                }
            )
            if include_extension and max(available_trials, default=-1) >= 15:
                halves = (set(range(8)), set(range(8, 16)))
            else:
                halves = (set(range(4)), set(range(4, 8)))
            expected_trials = set().union(*halves)
            for candidate in state["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                rows = [
                    row
                    for row in by_group[(state_id, str(teacher), candidate_id)]
                    if int(row["trial_index"]) in expected_trials
                ]
                if len(rows) != len(expected_trials):
                    raise ValueError("MultiValue full seed matrix is incomplete")
                full_summaries[(state_id, str(teacher), candidate_id)] = (
                    _candidate_summary(rows)
                )
            rankings = []
            summaries_by_half = []
            for half in halves:
                summaries = {}
                for candidate in state["candidates"]:
                    candidate_id = str(candidate["candidate_id"])
                    rows = [
                        row
                        for row in by_group[(state_id, str(teacher), candidate_id)]
                        if int(row["trial_index"]) in half
                    ]
                    if len(rows) != len(half):
                        raise ValueError("MultiValue seed-half matrix is incomplete")
                    summaries[candidate_id] = _candidate_summary(rows)
                ranking = sorted(
                    summaries,
                    key=lambda candidate_id: _value_order_key(
                        candidate_id, summaries[candidate_id]
                    ),
                )
                rankings.append(ranking)
                summaries_by_half.append(summaries)
            top_count = min(3, len(rankings[0]))
            overlap = len(set(rankings[0][:top_count]) & set(rankings[1][:top_count])) / top_count
            correlation = _rank_correlation(rankings[0], rankings[1])
            best_other = rankings[1][0]
            chosen = rankings[0][0]
            chosen_auc = summaries_by_half[1][chosen]["horizons"]["128"][
                "mean_normalized_conflict_auc"
            ]
            best_auc = summaries_by_half[1][best_other]["horizons"]["128"][
                "mean_normalized_conflict_auc"
            ]
            regret = (
                max(0.0, float(chosen_auc) - float(best_auc))
                if chosen_auc is not None and best_auc is not None
                else None
            )
            if overlap < float(
                config["stability_gates"]["seed_half_top3_overlap_minimum"]
            ):
                extension_ids.add(state_id)
            teacher_rows.append(
                {
                    "teacher": teacher,
                    "top3_overlap": overlap,
                    "paired_rank_correlation": correlation,
                    "cross_seed_normalized_regret": regret,
                    "first_half_top3": rankings[0][:top_count],
                    "second_half_top3": rankings[1][:top_count],
                }
            )
        state_stability.append(
            {
                "state_occurrence_id": state_id,
                "map_id": str(state["map_id"]),
                "teachers": teacher_rows,
                "requires_seed_extension": state_id in extension_ids,
            }
        )
    direction_stability: list[dict[str, Any]] = []
    for state_id, state in sorted(state_by_id.items()):
        anchor_id = str(state["v2_anchor_candidate_id"])
        for candidate in state["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id == anchor_id:
                continue
            directions = {}
            for teacher in config["rollout"]["teachers"]:
                teacher = str(teacher)
                directions[teacher] = _direction_against_anchor(
                    candidate_id,
                    full_summaries[(state_id, teacher, candidate_id)],
                    anchor_id,
                    full_summaries[(state_id, teacher, anchor_id)],
                )
            direction_stability.append(
                {
                    "state_occurrence_id": state_id,
                    "map_id": str(state["map_id"]),
                    "candidate_id": candidate_id,
                    "directions": directions,
                    "directions_agree": len(set(directions.values())) == 1,
                }
            )
    gates = dict(config["stability_gates"])
    map_groups = {}
    for map_id in sorted({str(row["map_id"]) for row in state_stability}):
        map_groups[map_id] = _stability_group(
            [row for row in state_stability if str(row["map_id"]) == map_id],
            [row for row in direction_stability if str(row["map_id"]) == map_id],
            gates,
        )
    overall = _stability_group(state_stability, direction_stability, gates)
    initial_seed_count = int(config["rollout"]["initial_paired_seed_count"])
    teacher_count = len(config["rollout"]["teachers"])
    expected_label_count = sum(
        int(state["candidate_count"]) * teacher_count * initial_seed_count
        for state in states
    )
    if include_extension:
        registered_extension_ids = set(
            map(
                str,
                _read_json(output / "initial_stability_report.json").get(
                    "extension_state_ids"
                )
                or (),
            )
        )
        expected_label_count += sum(
            int(state["candidate_count"])
            * teacher_count
            * (
                int(config["rollout"]["extended_paired_seed_count"])
                - initial_seed_count
            )
            for state in states
            if str(state["state_occurrence_id"]) in registered_extension_ids
        )
    identity_count = len(
        {
            (
                str(row["state_occurrence_id"]),
                str(row["teacher"]),
                str(row["candidate_id"]),
                int(row["trial_index"]),
            )
            for row in labels
        }
    )
    integrity = {
        "expected_label_count": len(labels) == expected_label_count,
        "unique_label_identity": identity_count == len(labels),
        "all_candidate_features_124": all(
            int(row["feature_count"]) == 124
            for state in states
            for row in state["candidates"]
        ),
        "all_stop_reasons_valid": all(
            str(row["targets"]["stop_reason"])
            in {"success", "repair_limit", "wall_timeout"}
            for row in labels
        ),
        "runtime_and_ttf_not_labels": all(
            row["runtime_is_not_a_label"] is True
            and row["ttf_is_not_a_label"] is True
            for row in labels
        ),
    }
    integrity_passed = all(integrity.values())
    stability_passed = (
        include_extension
        and integrity_passed
        and overall["passed"]
        and all(group["passed"] for group in map_groups.values())
    )
    report_name = (
        "extended_stability_report.json"
        if include_extension
        else "initial_stability_report.json"
    )
    report = {
        "schema": "lns2.stride.multivalue_stability_report.v1",
        "scientific_status": "completed_label_stability_analysis",
        "include_extension": include_extension,
        "state_occurrence_count": len(states),
        "label_count": len(labels),
        "extension_state_ids": sorted(extension_ids),
        "extension_state_count": len(extension_ids),
        "state_stability": state_stability,
        "cross_teacher_direction_stability": direction_stability,
        "overall_stability": overall,
        "map_group_stability": map_groups,
        "stability_gates": gates,
        "label_stability_passed": stability_passed,
        "model_training_allowed": stability_passed,
        "cross_teacher_direction_analysis_pending": False,
        "expected_label_count": expected_label_count,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "claim_boundary": dict(config["claim_boundary"]),
        "artifact_sha256": {
            "state_manifest": sha256_file(output / "state_manifest.jsonl"),
            "labels": sha256_file(output / "multihorizon_labels.jsonl"),
        },
    }
    _write_json(output / report_name, report)
    return report


__all__ = [
    "analyze_multivalue_collection",
    "history_before_decision",
    "load_multivalue_collection_config",
    "prepare_multivalue_pilot",
    "run_multivalue_collection",
    "run_worker_preflight",
    "select_pilot_occurrences",
]
