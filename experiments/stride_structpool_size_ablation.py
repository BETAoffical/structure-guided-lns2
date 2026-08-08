from __future__ import annotations

import collections
import os
import statistics
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
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
from experiments.stride_collection import (
    _paired_action,
    _replay_job,
    _validate_native_repair,
    load_stride_selection,
)
from experiments.stride_repairability_collection import (
    _source_target_state,
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_robustaction_label_collection import (
    _aggregate_candidate,
    state_artifact_tree_sha256,
)
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT, restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import score_online_candidates
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome
from lns2_selector.runtime.topology_candidates import (
    _jaccard as _candidate_jaccard,
    generate_structpool_candidate_grid,
)


CONFIG_SCHEMA = "lns2.stride.structpool_size_ablation_config.v1"
GRID_RUN_SCHEMA = "lns2.stride.structpool_size_grid_run.v1"
GRID_STATE_SCHEMA = "lns2.stride.structpool_size_grid_state.v1"
GRID_REPORT_SCHEMA = "lns2.stride.structpool_size_grid_report.v1"
LABEL_RUN_SCHEMA = "lns2.stride.structpool_size_label_run.v1"
LABEL_STATE_SCHEMA = "lns2.stride.structpool_size_label_state.v1"
TRIAL_SCHEMA = "lns2.stride.structpool_size_label_trial.v1"
AGGREGATE_SCHEMA = "lns2.stride.structpool_size_current_step_label.v1"
LABEL_REPORT_SCHEMA = "lns2.stride.structpool_size_label_report.v1"
LABEL_AUDIT_SCHEMA = "lns2.stride.structpool_size_label_audit.v1"
ANALYSIS_SCHEMA = "lns2.stride.structpool_size_ablation_report.v1"
TRIAL_INDICES = tuple(range(16))
FORBIDDEN_FIELDS = {
    "cost_to_go",
    "future_repair_rounds",
    "future_trajectory",
    "native_step_seconds",
    "pp_replan_seconds",
    "receding_q",
    "repair_runtime",
    "time_to_feasible",
    "ttf",
}
PRODUCER_FILES = (
    "experiments/online_feature_engine.py",
    "experiments/stride_structpool_size_ablation.py",
    "lns2_selector/runtime/topology_candidates.py",
    "src/online_features.cpp",
    "src/python_bindings.cpp",
)


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"registered size-ablation input is missing: {path}")
    observed = sha256_file(path)
    if observed != str(specification["sha256"]):
        raise ValueError(
            f"registered size-ablation input changed: {path}: "
            f"expected {specification['sha256']}, got {observed}"
        )
    return path


def _forbidden_hits(value: Any) -> set[str]:
    hits: set[str] = set()
    if isinstance(value, dict):
        hits.update(FORBIDDEN_FIELDS & set(map(str, value)))
        for nested in value.values():
            hits.update(_forbidden_hits(nested))
    elif isinstance(value, list):
        for nested in value:
            hits.update(_forbidden_hits(nested))
    return hits


def validate_size_ablation_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != "stride-structpool-size-ablation-v1"
        or config.get("implementation_id") != "stride-structpool-lazy-v1"
        or config.get("scientific_status")
        != "preregistered_outcome_blind_four_size_grid_before_new_repairs"
        or config.get("pre_registration_commit") != "911b3b7"
    ):
        raise ValueError("StructPool size-ablation identity changed")
    cohort = dict(config.get("cohort") or {})
    if (
        int(cohort.get("selection_state_count", -1)) != 320
        or int(cohort.get("active_structpool_state_count", -1)) != 98
        or dict(cohort.get("active_states_per_source_policy") or {})
        != {"official_adaptive": 51, "v2-full": 47}
        or int(cohort.get("active_map_count", -1)) != 16
    ):
        raise ValueError("StructPool size-ablation cohort changed")
    grid = dict(config.get("candidate_grid") or {})
    if (
        list(map(int, grid.get("allowed_sizes") or ())) != [8, 16, 24, 32]
        or int(grid.get("maximum_raw_drafts_per_state", -1)) != 24
        or grid.get("v2_anchor_pool") != "base_candidates_only"
        or any(
            grid.get(name) is not True
            for name in (
                "deduplicate_exact_agent_sets",
                "retain_all_family_size_provenance",
                "bypass_six_candidate_reduction",
                "anchor_selection_outcome_blind",
            )
        )
    ):
        raise ValueError("StructPool four-size grid contract changed")
    paired = dict(config.get("paired_labels") or {})
    if (
        tuple(map(int, paired.get("trial_indices") or ())) != TRIAL_INDICES
        or paired.get("pp_seed_pairing")
        != "same_repair_fingerprint_and_trial_index"
        or paired.get("target") != "normalized_current_step_conflict_reduction"
        or paired.get("reuse_exact_existing_trials") is not True
        or any(
            paired.get(name) is not False
            for name in (
                "runtime_fields_allowed",
                "future_fields_allowed",
                "ttf_allowed",
                "cost_to_go_allowed",
            )
        )
    ):
        raise ValueError("StructPool paired label contract changed")
    if dict(config.get("analysis") or {}).get("formal_ttf_claim") is not False:
        raise ValueError("size ablation cannot make a TTF claim")
    if project_root is not None:
        for specification in dict(config.get("inputs") or {}).values():
            _registered(project_root.resolve(), dict(specification))
        root = (
            project_root / str(config["preflight_state_artifact_root"])
        ).resolve()
        if state_artifact_tree_sha256(root) != str(
            config["preflight_state_artifact_tree_sha256"]
        ):
            raise ValueError("preflight state artifact tree changed")


def _active_preflight_states(
    *, config: dict[str, Any], project_root: Path, decisions: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], Path, dict[str, Any]]]:
    by_id = {str(row["state_id"]): row for row in decisions}
    root = (project_root / str(config["preflight_state_artifact_root"])).resolve()
    active: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.json"), key=lambda value: value.name):
        payload = _read_json(path)
        state_id = str(payload.get("state_id", ""))
        decision = by_id.get(state_id)
        if decision is None or state_id in seen:
            raise ValueError(f"invalid preflight state identity: {path}")
        seen.add(state_id)
        if bool(payload.get("high_stress_gate_passed")):
            if payload.get("topology_analysis_executed") is not True:
                raise ValueError(f"active state lacks topology analysis: {state_id}")
            active.append((decision, path, payload))
    cohort = dict(config["cohort"])
    policy_counts = collections.Counter(
        str(decision["source_policy"]) for decision, _, _ in active
    )
    if (
        seen != set(by_id)
        or len(active) != int(cohort["active_structpool_state_count"])
        or policy_counts != collections.Counter(
            cohort["active_states_per_source_policy"]
        )
        or len({str(decision["map_id"]) for decision, _, _ in active})
        != int(cohort["active_map_count"])
    ):
        raise ValueError("active StructPool state product changed")
    long_tail = dict(cohort["known_maze_long_tail_excluded"])
    if any(
        str(decision["map_id"]) == str(long_tail["map_id"])
        and int(decision["agent_count"]) == int(long_tail["agent_count"])
        and int(decision["solver_seed"]) == int(long_tail["solver_seed"])
        for decision, _, _ in active
    ):
        raise ValueError("known Maze long-tail state entered size ablation")
    return active


def _outcome_blind_base_anchor(
    preflight: dict[str, Any], model: Any
) -> dict[str, Any]:
    base = [
        row for row in preflight["candidates"]
        if str(row.get("candidate_kind")) == "base"
    ]
    if not base:
        raise ValueError("preflight state has no base candidates")
    rows = [
        {
            "candidate_id": str(row["candidate_id"]),
            "candidate_key": str(row["candidate_id"]),
            "features": {"realized_dynamic": dict(row["features"])},
        }
        for row in base
    ]
    selected, scores, margin = score_online_candidates(rows, model)
    anchor = base[selected]
    return {
        "candidate_id": str(anchor["candidate_id"]),
        "agents": list(map(int, anchor["agents"])),
        "actual_size": int(anchor["actual_size"]),
        "v2_score": float(scores[selected]),
        "v2_margin": float(margin),
        "candidate_count": len(base),
        "candidate_repair_outcomes_read": False,
    }


def _extract_grid_state(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    preflight = _read_json(Path(str(job["preflight_path"])))
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if (
            existing.get("schema") == GRID_STATE_SCHEMA
            and existing.get("run_fingerprint") == run_fingerprint
            and existing.get("complete") is True
            and existing.get("state_id") == decision["state_id"]
            and not _forbidden_hits(existing)
        ):
            return {
                "state_id": str(decision["state_id"]),
                "state_file": str(output_path),
                "status": "resumed",
                "candidate_count": len(existing["candidates"]),
                "raw_draft_count": int(existing["raw_draft_count"]),
            }
        raise ValueError(f"invalid existing size-grid state: {output_path}")

    state, _manifest, _trace = _source_target_state(decision)
    before = state_fingerprint(state)
    before_repair = repair_structure_fingerprint(state)
    if (
        before != str(decision["before_fingerprint"])
        or before != str(preflight["before_fingerprint"])
        or before_repair != str(preflight["before_repair_fingerprint"])
        or int(state["num_of_colliding_pairs"]) != int(decision["before_conflicts"])
    ):
        raise RuntimeError("size-grid source state differs from preflight")
    cache = TopologyAnalysisCache(state, backend="native")
    if cache.analysis is None:
        raise RuntimeError("native topology analysis is missing")
    grid = generate_structpool_candidate_grid(state, cache.analysis)
    engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={
            "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
        },
    )
    if cache.last_native_prepared is not None:
        engine.prepare(state, prepared_native_analysis=cache.last_native_prepared)
    feature_rows, feature_metrics = engine.realized_rows(grid, state_hash=before)
    features_by_id = {
        str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
        for row in feature_rows
    }
    anchor = dict(job["anchor"])
    candidates = []
    for candidate in grid:
        candidate_id = str(candidate["candidate_id"])
        features = features_by_id[candidate_id]
        if set(features) != set(PROFILE_FEATURE_NAMES["realized_dynamic"]):
            raise RuntimeError("size-grid candidate feature dimension changed")
        candidates.append(
            {
                **candidate,
                "candidate_kind": "structpool-grid",
                "features": features,
                "v2_anchor_jaccard": _candidate_jaccard(
                    candidate["agents"], anchor["agents"]
                ),
            }
        )
    raw_draft_count = sum(
        int(row["structpool_grid_duplicate_provenance_count"])
        for row in candidates
    )
    if raw_draft_count > 24 or not candidates:
        raise RuntimeError("size-grid raw candidate product is invalid")
    payload = {
        "schema": GRID_STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": str(decision["state_id"]),
        "decision": decision,
        "before_fingerprint": before,
        "before_repair_fingerprint": before_repair,
        "before_conflicts": int(decision["before_conflicts"]),
        "preflight_state_file": str(job["preflight_path"]),
        "preflight_state_sha256": sha256_file(Path(str(job["preflight_path"]))),
        "v2_base_anchor": anchor,
        "raw_draft_count": raw_draft_count,
        "unique_candidate_count": len(candidates),
        "candidate_signature": _fingerprint(
            [
                {
                    "candidate_id": row["candidate_id"],
                    "agents": row["agents"],
                    "selection_families": row["selection_families"],
                }
                for row in candidates
            ]
        ),
        "feature_signature": _fingerprint(
            [
                {"candidate_id": row["candidate_id"], "features": row["features"]}
                for row in candidates
            ]
        ),
        "candidates": candidates,
        "feature_metrics": feature_metrics,
        "candidate_repair_trials_executed": False,
        "controller_actions_executed": False,
        "runtime_or_ttf_read": False,
        "future_trajectory_read": False,
    }
    if _forbidden_hits(payload):
        raise RuntimeError("forbidden outcome field entered size grid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": str(decision["state_id"]),
        "state_file": str(output_path),
        "status": "ok",
        "candidate_count": len(candidates),
        "raw_draft_count": raw_draft_count,
    }


def extract_size_grid(
    *, config_path: str | Path, output: str | Path, workers: int = 2,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_size_ablation_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    decisions = load_stride_selection(inputs["selection"])
    active = _active_preflight_states(
        config=config, project_root=project_root, decisions=decisions
    )
    bundle = load_controller_bundle(inputs["frozen_v2_manifest"].parent)
    if bundle.manifest.get("default_controller") != "v2-full":
        raise ValueError("size grid requires frozen v2-full")
    model = bundle.main_models["realized_dynamic"]
    producer = producer_identity(
        project_root=project_root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    identity = {
        "schema": GRID_RUN_SCHEMA,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "active_state_ids": [str(row[0]["state_id"]) for row in active],
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    output = Path(output).resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("size-grid output belongs to another run")
        if not resume:
            raise ValueError("size-grid output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    jobs = []
    for decision, preflight_path, preflight in active:
        key = _fingerprint(
            {"state_id": decision["state_id"], "before": decision["before_fingerprint"]}
        )[:20]
        jobs.append(
            {
                "job_id": str(decision["state_id"]),
                "decision": decision,
                "preflight_path": str(preflight_path),
                "anchor": _outcome_blind_base_anchor(preflight, model),
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    status_path = output / "grid_status.json"
    observed: list[dict[str, Any]] = []

    def update_status(result: dict[str, Any]) -> None:
        observed.append(result)
        failures = [row for row in observed if row.get("status") in {"error", "timeout"}]
        _write_json(
            status_path,
            {
                "schema": GRID_REPORT_SCHEMA,
                "status": "running",
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "completed_candidate_count": sum(
                    int(row.get("candidate_count", 0)) for row in observed
                ),
                "error_state_count": sum(row.get("status") == "error" for row in failures),
                "timeout_state_count": sum(row.get("status") == "timeout" for row in failures),
                "errors": failures,
            },
        )

    _write_json(
        status_path,
        {
            "schema": GRID_REPORT_SCHEMA,
            "status": "running",
            "requested_state_count": len(jobs),
            "completed_state_count": 0,
            "completed_candidate_count": 0,
            "error_state_count": 0,
            "timeout_state_count": 0,
            "errors": [],
        },
    )
    observed = _run_jobs(
        _extract_grid_state,
        jobs,
        workers,
        phase="stride-structpool-size-grid",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_state_timeout_seconds"]),
        on_result=update_status,
    )
    failures = [row for row in observed if row.get("status") in {"error", "timeout"}]
    successes = [row for row in observed if row.get("status") in {"ok", "resumed"}]
    manifests = []
    for result in sorted(successes, key=lambda row: str(row["state_id"])):
        payload = _read_json(Path(str(result["state_file"])))
        manifests.append(
            {
                "state_id": str(result["state_id"]),
                "map_id": str(payload["decision"]["map_id"]),
                "task_id": str(payload["decision"]["task_id"]),
                "source_policy": str(payload["decision"]["source_policy"]),
                "layout_mode": str(payload["decision"]["layout_mode"]),
                "agent_count": int(payload["decision"]["agent_count"]),
                "before_conflicts": int(payload["before_conflicts"]),
                "raw_draft_count": int(payload["raw_draft_count"]),
                "candidate_count": int(payload["unique_candidate_count"]),
                "candidate_signature": str(payload["candidate_signature"]),
                "feature_signature": str(payload["feature_signature"]),
                "state_file": str(result["state_file"]),
                "state_file_sha256": sha256_file(Path(str(result["state_file"]))),
            }
        )
    passed = (
        len(manifests) == int(config["cohort"]["active_structpool_state_count"])
        and not failures
        and all(0 < row["candidate_count"] <= row["raw_draft_count"] <= 24 for row in manifests)
    )
    artifacts: dict[str, Any] = {}
    if passed:
        manifest_path = output / "grid_manifest.jsonl"
        _write_jsonl(manifest_path, manifests)
        artifacts = {
            "grid_manifest_sha256": sha256_file(manifest_path),
            "state_artifact_tree_sha256": state_artifact_tree_sha256(output / "states"),
        }
    report = {
        "schema": GRID_REPORT_SCHEMA,
        "scientific_status": "outcome_blind_four_size_grid_complete" if passed else "failed",
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(jobs),
        "completed_state_count": len(manifests),
        "raw_draft_count": sum(row["raw_draft_count"] for row in manifests),
        "unique_candidate_count": sum(row["candidate_count"] for row in manifests),
        "exact_duplicate_count": sum(
            row["raw_draft_count"] - row["candidate_count"] for row in manifests
        ),
        "error_state_count": sum(row.get("status") == "error" for row in failures),
        "timeout_state_count": sum(row.get("status") == "timeout" for row in failures),
        "candidate_repair_trials_executed": False,
        "runtime_or_ttf_read": False,
        "future_trajectory_read": False,
        "known_maze_long_tail_excluded": True,
        "passed": passed,
        "errors": failures,
        "artifacts": artifacts,
    }
    _write_json(output / "grid_report.json", report)
    _write_json(status_path, {**report, "status": "complete" if passed else "failed"})
    return report


def _copy_reused_trial(
    row: dict[str, Any], *, candidate_kind: str
) -> dict[str, Any]:
    return {
        "schema": TRIAL_SCHEMA,
        "state_id": str(row["state_id"]),
        "candidate_id": str(row["candidate_id"]),
        "candidate_kind": candidate_kind,
        "trial_index": int(row["trial_index"]),
        "pp_seed": int(row["pp_seed"]),
        "before_conflicts": int(row["before_conflicts"]),
        "before_fingerprint": str(row["before_fingerprint"]),
        "before_repair_fingerprint": str(row["before_repair_fingerprint"]),
        "conflicts_after": int(row["conflicts_after"]),
        "normalized_conflict_reduction": float(row["normalized_conflict_reduction"]),
        "replan_success": bool(row["replan_success"]),
        "feasible": bool(row["feasible"]),
        "repair_outcome": str(row["repair_outcome"]),
        "after_repair_fingerprint": str(row["after_repair_fingerprint"]),
        "trial_source": "reused_exact_robustaction_v1",
    }


def _validate_label_matrix(
    *,
    trials: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    expected_candidates: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    aggregate_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in aggregates:
        key = (str(row["state_id"]), str(row["candidate_id"]))
        if key in aggregate_by_key:
            errors.append(f"duplicate aggregate: {key[0]} {key[1]}")
            continue
        aggregate_by_key[key] = row
    expected_keys = set(expected_candidates)
    aggregate_keys = set(aggregate_by_key)
    for state_id, candidate_id in sorted(expected_keys - aggregate_keys):
        errors.append(f"missing aggregate: {state_id} {candidate_id}")
    for state_id, candidate_id in sorted(aggregate_keys - expected_keys):
        errors.append(f"unexpected aggregate: {state_id} {candidate_id}")

    trials_by_key: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    state_trial_seeds: dict[tuple[str, int], set[int]] = collections.defaultdict(set)
    for row in trials:
        key = (str(row["state_id"]), str(row["candidate_id"]))
        trials_by_key[key].append(row)
        state_trial_seeds[(key[0], int(row["trial_index"]))].add(
            int(row["pp_seed"])
        )
    for state_id, candidate_id in sorted(set(trials_by_key) - expected_keys):
        errors.append(f"unexpected trial candidate: {state_id} {candidate_id}")

    expected_indices = set(TRIAL_INDICES)
    for key in sorted(expected_keys):
        expected = expected_candidates[key]
        aggregate = aggregate_by_key.get(key)
        if aggregate is None:
            continue
        features = dict(aggregate.get("features") or {})
        agents = tuple(int(agent) for agent in aggregate.get("agents") or [])
        expected_agents = tuple(int(agent) for agent in expected["agents"])
        if agents != expected_agents:
            errors.append(f"candidate agents changed: {key[0]} {key[1]}")
        if len(agents) != len(set(agents)) or any(
            agent < 0 or agent >= int(aggregate["agent_count"]) for agent in agents
        ):
            errors.append(f"candidate agents are not a legal native action: {key[0]} {key[1]}")
        if int(aggregate["actual_size"]) != len(agents):
            errors.append(f"candidate actual size mismatch: {key[0]} {key[1]}")
        if len(features) != 124:
            errors.append(f"candidate feature dimension is not 124: {key[0]} {key[1]}")
        if features != dict(expected["features"]):
            errors.append(f"candidate features changed: {key[0]} {key[1]}")
        rows = trials_by_key.get(key, [])
        indices = [int(row["trial_index"]) for row in rows]
        if (
            len(rows) != 16
            or set(indices) != expected_indices
            or len(indices) != len(set(indices))
        ):
            errors.append(f"candidate trial matrix is not exactly 0-15: {key[0]} {key[1]}")
            continue
        before_values = {
            (
                int(row["before_conflicts"]),
                str(row["before_fingerprint"]),
                str(row["before_repair_fingerprint"]),
            )
            for row in rows
        }
        if len(before_values) != 1:
            errors.append(f"candidate before-state identity changed: {key[0]} {key[1]}")
            continue
        before_conflicts, _, before_repair = next(iter(before_values))
        if before_conflicts != int(expected["before_conflicts"]):
            errors.append(f"candidate before conflicts changed: {key[0]} {key[1]}")
        expected_before = str(expected.get("before_fingerprint") or "")
        expected_before_repair = str(expected.get("before_repair_fingerprint") or "")
        if expected_before and next(iter(before_values))[1] != expected_before:
            errors.append(f"candidate before fingerprint changed: {key[0]} {key[1]}")
        if expected_before_repair and before_repair != expected_before_repair:
            errors.append(
                f"candidate before repair fingerprint changed: {key[0]} {key[1]}"
            )
        for row in rows:
            if str(row.get("schema")) != TRIAL_SCHEMA:
                errors.append(f"candidate trial schema mismatch: {key[0]} {key[1]}")
                break
            trial_index = int(row["trial_index"])
            expected_seed = repairability_pp_seed(before_repair, trial_index)
            if int(row["pp_seed"]) != expected_seed:
                errors.append(f"candidate PP seed formula mismatch: {key[0]} {key[1]}")
                break

    for (state_id, trial_index), seeds in sorted(state_trial_seeds.items()):
        if len(seeds) != 1:
            errors.append(
                f"state/trial PP seeds are not paired: {state_id} {trial_index}"
            )
    return {
        "passed": not errors,
        "errors": errors,
        "expected_candidate_count": len(expected_keys),
        "observed_candidate_count": len(aggregate_keys),
        "observed_trial_count": len(trials),
        "feature_dimension": 124,
        "trial_indices": list(TRIAL_INDICES),
        "paired_state_trial_count": len(state_trial_seeds),
    }


def _collect_size_label_state(job: dict[str, Any]) -> dict[str, Any]:
    grid_path = Path(str(job["grid_path"]))
    grid = _read_json(grid_path)
    decision = dict(grid["decision"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if (
            existing.get("schema") == LABEL_STATE_SCHEMA
            and existing.get("run_fingerprint") == run_fingerprint
            and existing.get("complete") is True
            and not _forbidden_hits(existing)
        ):
            return {
                "state_id": str(grid["state_id"]),
                "state_file": str(output_path),
                "status": "resumed",
                "candidate_count": len(existing["candidate_aggregates"]),
                "trial_count": len(existing["trials"]),
                "reused_trial_count": int(existing["reused_trial_count"]),
                "new_trial_count": int(existing["new_trial_count"]),
            }
        raise ValueError(f"invalid completed size-label state: {output_path}")

    replay = _replay_job(decision)
    state, source_manifest, source_trace_path = _source_target_state(decision)
    before = state_fingerprint(state)
    before_repair = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    if (
        before != str(grid["before_fingerprint"])
        or before_repair != str(grid["before_repair_fingerprint"])
        or before_conflicts != int(grid["before_conflicts"])
    ):
        raise RuntimeError("size-label source state differs from grid")
    restore_seed = repairability_restore_seed(before_repair)
    reused_by_candidate = {
        str(candidate_id): list(rows)
        for candidate_id, rows in dict(job.get("reused_trials") or {}).items()
    }
    trials: list[dict[str, Any]] = []
    aggregates = []
    reused_count = 0
    new_count = 0
    for candidate in grid["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        agents = list(map(int, candidate["agents"]))
        existing = sorted(
            reused_by_candidate.get(candidate_id, []),
            key=lambda row: int(row["trial_index"]),
        )
        candidate_trials: list[dict[str, Any]] = []
        if len(existing) == 16 and {
            int(row["trial_index"]) for row in existing
        } == set(TRIAL_INDICES):
            for row in existing:
                trial = _copy_reused_trial(
                    row, candidate_kind="structpool-grid"
                )
                if (
                    trial["before_fingerprint"] != before
                    or trial["before_repair_fingerprint"] != before_repair
                    or trial["pp_seed"]
                    != repairability_pp_seed(before_repair, trial["trial_index"])
                ):
                    raise RuntimeError("reused size-label trial identity changed")
                candidate_trials.append(trial)
            reused_count += 16
        else:
            for trial_index in TRIAL_INDICES:
                branch_environment, branch_state = restore_repair_state(
                    replay, state, seed=restore_seed
                )
                if repair_structure_fingerprint(branch_state) != before_repair:
                    raise RuntimeError("size-label branch restore changed")
                pp_seed = repairability_pp_seed(before_repair, trial_index)
                result = _plain(
                    branch_environment.step(_paired_action(agents, pp_seed))
                )
                after, metrics = _validate_native_repair(
                    result, expected_agents=agents, expected_seed=pp_seed
                )
                conflicts_after = int(after["num_of_colliding_pairs"])
                after_repair = repair_structure_fingerprint(after)
                candidate_trials.append(
                    {
                        "schema": TRIAL_SCHEMA,
                        "state_id": str(grid["state_id"]),
                        "candidate_id": candidate_id,
                        "candidate_kind": "structpool-grid",
                        "trial_index": trial_index,
                        "pp_seed": pp_seed,
                        "before_conflicts": before_conflicts,
                        "before_fingerprint": before,
                        "before_repair_fingerprint": before_repair,
                        "conflicts_after": conflicts_after,
                        "normalized_conflict_reduction": (
                            before_conflicts - conflicts_after
                        ) / max(1, before_conflicts),
                        "replan_success": bool(metrics["replan_success"]),
                        "feasible": bool(after.get("feasible")),
                        "repair_outcome": classify_repair_outcome(
                            before_fingerprint=before_repair,
                            after_fingerprint=after_repair,
                            replan_success=bool(metrics["replan_success"]),
                            conflicts_before=before_conflicts,
                            conflicts_after=conflicts_after,
                            feasible=bool(after.get("feasible")),
                        ),
                        "after_repair_fingerprint": after_repair,
                        "trial_source": "new_size_grid_repair",
                    }
                )
            new_count += 16
        trials.extend(candidate_trials)
        aggregate = _aggregate_candidate(candidate, candidate_trials)
        aggregate.update(
            {
                "schema": AGGREGATE_SCHEMA,
                "no_progress_rate": 1.0 - float(aggregate["progress_rate"]),
                "repair_success_rate": float(aggregate["replan_success_rate"]),
                "v2_anchor_jaccard": float(candidate["v2_anchor_jaccard"]),
                "structpool_support_count_by_family": dict(
                    candidate["structpool_support_count_by_family"]
                ),
                "structpool_support_ratio_by_family": dict(
                    candidate["structpool_support_ratio_by_family"]
                ),
                "structpool_nominal_size_by_family": dict(
                    candidate["structpool_nominal_size_by_family"]
                ),
                "structpool_grid_pure_family": bool(
                    candidate["structpool_grid_pure_family"]
                ),
                "structpool_grid_duplicate_provenance_count": int(
                    candidate["structpool_grid_duplicate_provenance_count"]
                ),
            }
        )
        aggregates.append(aggregate)
    payload = {
        "schema": LABEL_STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": str(grid["state_id"]),
        "decision": decision,
        "before_fingerprint": before,
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "grid_state_file": str(grid_path),
        "grid_state_sha256": sha256_file(grid_path),
        "state_restore": {
            "contract": TARGET_STATE_RESTORE_CONTRACT,
            "restore_seed": restore_seed,
            "source_trace_file": str(source_manifest["trace_file"]),
            "source_trace_path": str(source_trace_path),
        },
        "trials": trials,
        "candidate_aggregates": aggregates,
        "reused_trial_count": reused_count,
        "new_trial_count": new_count,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
    }
    expected_count = len(grid["candidates"]) * 16
    if (
        len(trials) != expected_count
        or reused_count + new_count != expected_count
        or _forbidden_hits(payload)
    ):
        raise RuntimeError("size-label state product is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": str(grid["state_id"]),
        "state_file": str(output_path),
        "status": "ok",
        "candidate_count": len(aggregates),
        "trial_count": len(trials),
        "reused_trial_count": reused_count,
        "new_trial_count": new_count,
    }


def collect_size_labels(
    *, config_path: str | Path, grid: str | Path, output: str | Path,
    workers: int = 2, resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_size_ablation_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    grid = Path(grid).resolve()
    grid_report = _read_json(grid / "grid_report.json")
    if grid_report.get("passed") is not True:
        raise ValueError("size labels require a complete outcome-blind grid")
    manifests = _read_jsonl(grid / "grid_manifest.jsonl")
    if len(manifests) != int(config["cohort"]["active_structpool_state_count"]):
        raise ValueError("size-grid manifest count changed")
    existing_trials: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _read_jsonl(inputs["existing_trials"]):
        existing_trials[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    existing_aggregates = {
        (str(row["state_id"]), str(row["candidate_id"])): row
        for row in _read_jsonl(inputs["existing_aggregates"])
    }
    identity = {
        "schema": LABEL_RUN_SCHEMA,
        "config_sha256": sha256_file(config_path),
        "grid_report_sha256": sha256_file(grid / "grid_report.json"),
        "grid_manifest_sha256": sha256_file(grid / "grid_manifest.jsonl"),
        "grid_state_tree_sha256": state_artifact_tree_sha256(grid / "states"),
        "trial_indices": list(TRIAL_INDICES),
        "producer": producer_identity(
            project_root=project_root,
            source_files=PRODUCER_FILES,
            native_required=True,
            package_names=("numpy",),
        ),
    }
    run_fingerprint = _fingerprint(identity)
    output = Path(output).resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("size-label output belongs to another run")
        if not resume:
            raise ValueError("size-label output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    jobs = []
    for manifest in manifests:
        state_id = str(manifest["state_id"])
        grid_state = _read_json(Path(str(manifest["state_file"])))
        reuse: dict[str, list[dict[str, Any]]] = {}
        for candidate in grid_state["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            key = (state_id, candidate_id)
            aggregate = existing_aggregates.get(key)
            if aggregate is not None and list(map(int, aggregate["agents"])) == list(
                map(int, candidate["agents"])
            ):
                reuse[candidate_id] = existing_trials.get(key, [])
        key = _fingerprint({"state_id": state_id, "grid": manifest["candidate_signature"]})[:20]
        jobs.append(
            {
                "job_id": state_id,
                "grid_path": str(manifest["state_file"]),
                "reused_trials": reuse,
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    status_path = output / "collection_status.json"
    observed: list[dict[str, Any]] = []

    def update_status(result: dict[str, Any]) -> None:
        observed.append(result)
        failures = [row for row in observed if row.get("status") in {"error", "timeout"}]
        _write_json(
            status_path,
            {
                "schema": LABEL_REPORT_SCHEMA,
                "status": "running",
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "completed_candidate_count": sum(int(row.get("candidate_count", 0)) for row in observed),
                "completed_trial_count": sum(int(row.get("trial_count", 0)) for row in observed),
                "reused_trial_count": sum(int(row.get("reused_trial_count", 0)) for row in observed),
                "new_trial_count": sum(int(row.get("new_trial_count", 0)) for row in observed),
                "error_state_count": sum(row.get("status") == "error" for row in failures),
                "timeout_state_count": sum(row.get("status") == "timeout" for row in failures),
                "errors": failures,
            },
        )

    _write_json(
        status_path,
        {
            "schema": LABEL_REPORT_SCHEMA,
            "status": "running",
            "requested_state_count": len(jobs),
            "completed_state_count": 0,
            "completed_candidate_count": 0,
            "completed_trial_count": 0,
            "reused_trial_count": 0,
            "new_trial_count": 0,
            "error_state_count": 0,
            "timeout_state_count": 0,
            "errors": [],
        },
    )
    observed = _run_jobs(
        _collect_size_label_state,
        jobs,
        workers,
        phase="stride-structpool-size-labels",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_state_timeout_seconds"]),
        on_result=update_status,
    )
    failures = [row for row in observed if row.get("status") in {"error", "timeout"}]
    successes = [row for row in observed if row.get("status") in {"ok", "resumed"}]
    all_trials: list[dict[str, Any]] = []
    all_aggregates: list[dict[str, Any]] = []
    state_rows = []
    expected_candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest in manifests:
        grid_state = _read_json(Path(str(manifest["state_file"])))
        state_id = str(grid_state["state_id"])
        for candidate in grid_state["candidates"]:
            expected_candidates[(state_id, str(candidate["candidate_id"]))] = {
                "agents": list(candidate["agents"]),
                "features": dict(candidate["features"]),
                "before_conflicts": int(grid_state["before_conflicts"]),
                "before_fingerprint": str(grid_state["before_fingerprint"]),
                "before_repair_fingerprint": str(
                    grid_state["before_repair_fingerprint"]
                ),
            }
    for result in sorted(successes, key=lambda row: str(row["state_id"])):
        payload = _read_json(Path(str(result["state_file"])))
        decision = dict(payload["decision"])
        all_trials.extend(payload["trials"])
        for aggregate in payload["candidate_aggregates"]:
            all_aggregates.append(
                {
                    **aggregate,
                    "state_id": str(payload["state_id"]),
                    "map_id": str(decision["map_id"]),
                    "task_id": str(decision["task_id"]),
                    "episode_id": str(decision["episode_id"]),
                    "source_policy": str(decision["source_policy"]),
                    "solver_seed": int(decision["solver_seed"]),
                    "decision_index": int(decision["decision_index"]),
                    "layout_mode": str(decision["layout_mode"]),
                    "agent_count": int(decision["agent_count"]),
                    "agent_band": str(decision["agent_band"]),
                    "conflict_band": str(decision["conflict_band"]),
                    "before_conflicts": int(payload["before_conflicts"]),
                }
            )
        state_rows.append(
            {
                "state_id": str(payload["state_id"]),
                "map_id": str(decision["map_id"]),
                "candidate_count": len(payload["candidate_aggregates"]),
                "trial_count": len(payload["trials"]),
                "reused_trial_count": int(payload["reused_trial_count"]),
                "new_trial_count": int(payload["new_trial_count"]),
                "state_file": str(result["state_file"]),
                "state_file_sha256": sha256_file(Path(str(result["state_file"]))),
            }
        )
    candidate_count = len(all_aggregates)
    trial_count = len(all_trials)
    paired = collections.defaultdict(set)
    distinct = collections.defaultdict(set)
    for row in all_trials:
        paired[(str(row["state_id"]), int(row["trial_index"]))].add(int(row["pp_seed"]))
        distinct[str(row["state_id"])].add(int(row["pp_seed"]))
    matrix_validation = _validate_label_matrix(
        trials=all_trials,
        aggregates=all_aggregates,
        expected_candidates=expected_candidates,
    )
    passed = bool(
        len(state_rows) == len(manifests)
        and trial_count == candidate_count * 16
        and not failures
        and not _forbidden_hits([all_trials, all_aggregates])
        and all(len(values) == 1 for values in paired.values())
        and all(len(values) == 16 for values in distinct.values())
        and matrix_validation["passed"]
    )
    artifacts: dict[str, Any] = {}
    if passed:
        trials_path = output / "repair_trials.jsonl"
        aggregates_path = output / "candidate_aggregates.jsonl"
        manifest_path = output / "state_manifest.jsonl"
        _write_jsonl(trials_path, all_trials)
        _write_jsonl(aggregates_path, all_aggregates)
        _write_jsonl(manifest_path, state_rows)
        artifacts = {
            "repair_trials_sha256": sha256_file(trials_path),
            "candidate_aggregates_sha256": sha256_file(aggregates_path),
            "state_manifest_sha256": sha256_file(manifest_path),
            "state_artifact_tree_sha256": state_artifact_tree_sha256(output / "states"),
        }
    report = {
        "schema": LABEL_REPORT_SCHEMA,
        "scientific_status": "paired_current_step_four_size_labels_complete" if passed else "failed",
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(jobs),
        "completed_state_count": len(state_rows),
        "candidate_count": candidate_count,
        "trial_count": trial_count,
        "reused_trial_count": sum(row["reused_trial_count"] for row in state_rows),
        "new_trial_count": sum(row["new_trial_count"] for row in state_rows),
        "error_state_count": sum(row.get("status") == "error" for row in failures),
        "timeout_state_count": sum(row.get("status") == "timeout" for row in failures),
        "paired_pp_seed_integrity": passed,
        "label_matrix_validation": matrix_validation,
        "runtime_or_future_fields_stored": False,
        "formal_ttf_claim": False,
        "passed": passed,
        "errors": failures,
        "artifacts": artifacts,
    }
    _write_json(output / "collection_report.json", report)
    _write_json(status_path, {**report, "status": "complete" if passed else "failed"})
    return report


def audit_size_labels(
    *,
    config_path: str | Path,
    grid: str | Path,
    labels: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Audit an immutable completed label run with the current verifier.

    The collection run fingerprint intentionally binds producer source hashes.  This
    audit therefore never rewrites or resumes a run produced by an older source
    snapshot; it verifies that run in place and writes current-code evidence to a
    separate output directory.
    """

    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_size_ablation_config(config, project_root=project_root)
    grid = Path(grid).resolve()
    labels = Path(labels).resolve()
    output = Path(output).resolve()
    if output == labels or labels in output.parents:
        raise ValueError("size-label audit output must be separate from label artifacts")

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    grid_report = _read_json(grid / "grid_report.json")
    grid_manifests = _read_jsonl(grid / "grid_manifest.jsonl")
    collection_report = _read_json(labels / "collection_report.json")
    collection_status = _read_json(labels / "collection_status.json")
    original_run = _read_json(labels / "run_config.json")
    trials_path = labels / "repair_trials.jsonl"
    aggregates_path = labels / "candidate_aggregates.jsonl"
    state_manifest_path = labels / "state_manifest.jsonl"
    trials = _read_jsonl(trials_path)
    aggregates = _read_jsonl(aggregates_path)
    state_manifest = _read_jsonl(state_manifest_path)

    expected_state_count = int(config["cohort"]["active_structpool_state_count"])
    require(grid_report.get("passed") is True, "size grid is not complete")
    require(
        len(grid_manifests) == expected_state_count,
        "size-grid state count changed",
    )
    require(collection_report.get("passed") is True, "label collection did not pass")
    require(collection_status.get("status") == "complete", "label status is not complete")
    require(not collection_report.get("errors"), "label collection contains errors")
    require(
        int(collection_report.get("error_state_count", -1)) == 0,
        "label collection contains error states",
    )
    require(
        int(collection_report.get("timeout_state_count", -1)) == 0,
        "label collection contains timeout states",
    )
    require(
        collection_report.get("run_fingerprint") == original_run.get("run_fingerprint"),
        "label report and run fingerprint differ",
    )
    require(
        collection_report.get("runtime_or_future_fields_stored") is False,
        "label report permits runtime or future fields",
    )
    require(
        collection_report.get("formal_ttf_claim") is False,
        "label report makes an invalid TTF claim",
    )

    expected_candidates: dict[tuple[str, str], dict[str, Any]] = {}
    grid_state_by_id: dict[str, dict[str, Any]] = {}
    for manifest in grid_manifests:
        state_file = Path(str(manifest["state_file"]))
        require(state_file.is_file(), f"missing grid state: {state_file}")
        if not state_file.is_file():
            continue
        require(
            sha256_file(state_file) == str(manifest["state_file_sha256"]),
            f"grid state SHA-256 changed: {manifest['state_id']}",
        )
        state = _read_json(state_file)
        state_id = str(manifest["state_id"])
        grid_state_by_id[state_id] = state
        require(str(state.get("state_id")) == state_id, f"grid state id changed: {state_id}")
        require(
            int(state.get("unique_candidate_count", -1))
            == int(manifest["candidate_count"]),
            f"grid candidate count changed: {state_id}",
        )
        require(not _forbidden_hits(state), f"grid state has forbidden fields: {state_id}")
        for candidate in state.get("candidates") or []:
            key = (state_id, str(candidate["candidate_id"]))
            require(key not in expected_candidates, f"duplicate grid candidate: {key}")
            expected_candidates[key] = {
                "agents": list(candidate["agents"]),
                "features": dict(candidate["features"]),
                "before_conflicts": int(state["before_conflicts"]),
                "before_fingerprint": str(state["before_fingerprint"]),
                "before_repair_fingerprint": str(state["before_repair_fingerprint"]),
            }

    matrix_validation = _validate_label_matrix(
        trials=trials,
        aggregates=aggregates,
        expected_candidates=expected_candidates,
    )
    errors.extend(str(message) for message in matrix_validation["errors"])
    forbidden_hits = sorted(_forbidden_hits([trials, aggregates]))
    require(not forbidden_hits, f"label artifacts contain forbidden fields: {forbidden_hits}")

    manifest_by_state: dict[str, dict[str, Any]] = {}
    global_trials_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    global_aggregates_by_state: dict[str, dict[str, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for row in trials:
        global_trials_by_state[str(row["state_id"])].append(row)
    for row in aggregates:
        global_aggregates_by_state[str(row["state_id"])][str(row["candidate_id"])] = row

    for row in state_manifest:
        state_id = str(row["state_id"])
        require(state_id not in manifest_by_state, f"duplicate label state manifest: {state_id}")
        manifest_by_state[state_id] = row
        state_file = Path(str(row["state_file"]))
        require(state_file.is_file(), f"missing label state: {state_id}")
        if not state_file.is_file():
            continue
        require(
            sha256_file(state_file) == str(row["state_file_sha256"]),
            f"label state SHA-256 changed: {state_id}",
        )
        payload = _read_json(state_file)
        grid_state = grid_state_by_id.get(state_id)
        require(payload.get("schema") == LABEL_STATE_SCHEMA, f"label state schema changed: {state_id}")
        require(payload.get("complete") is True, f"label state is incomplete: {state_id}")
        require(str(payload.get("state_id")) == state_id, f"label state id changed: {state_id}")
        require(
            payload.get("run_fingerprint") == original_run.get("run_fingerprint"),
            f"label state run fingerprint changed: {state_id}",
        )
        require(not _forbidden_hits(payload), f"label state has forbidden fields: {state_id}")
        if grid_state is not None:
            require(
                str(payload.get("before_fingerprint"))
                == str(grid_state["before_fingerprint"]),
                f"label state fingerprint changed: {state_id}",
            )
            require(
                str(payload.get("before_repair_fingerprint"))
                == str(grid_state["before_repair_fingerprint"]),
                f"label repair fingerprint changed: {state_id}",
            )
            require(
                int(payload.get("before_conflicts", -1))
                == int(grid_state["before_conflicts"]),
                f"label before conflicts changed: {state_id}",
            )
            require(
                sha256_file(Path(str(payload["grid_state_file"])))
                == str(payload["grid_state_sha256"]),
                f"label grid reference SHA-256 changed: {state_id}",
            )
        require(
            list(payload.get("trials") or []) == global_trials_by_state.get(state_id, []),
            f"global and per-state trials differ: {state_id}",
        )
        global_candidates = global_aggregates_by_state.get(state_id, {})
        for candidate in payload.get("candidate_aggregates") or []:
            observed = global_candidates.get(str(candidate["candidate_id"]))
            require(observed is not None, f"missing global aggregate: {state_id}")
            if observed is not None:
                require(
                    all(observed.get(key) == value for key, value in candidate.items()),
                    f"global and per-state aggregate differ: {state_id} {candidate['candidate_id']}",
                )

    expected_states = set(grid_state_by_id)
    require(set(manifest_by_state) == expected_states, "label state manifest coverage changed")
    require(len(state_manifest) == expected_state_count, "label state manifest count changed")
    require(len(aggregates) == len(expected_candidates), "global candidate count changed")
    require(len(trials) == len(expected_candidates) * len(TRIAL_INDICES), "global trial count changed")

    reused_count = sum(
        str(row.get("trial_source")) == "reused_exact_robustaction_v1" for row in trials
    )
    new_count = sum(str(row.get("trial_source")) == "new_size_grid_repair" for row in trials)
    require(reused_count + new_count == len(trials), "unexpected trial source exists")
    require(
        reused_count == int(collection_report.get("reused_trial_count", -1)),
        "reused trial count changed",
    )
    require(
        new_count == int(collection_report.get("new_trial_count", -1)),
        "new trial count changed",
    )

    artifact_paths = {
        "repair_trials_sha256": trials_path,
        "candidate_aggregates_sha256": aggregates_path,
        "state_manifest_sha256": state_manifest_path,
    }
    observed_hashes = {name: sha256_file(path) for name, path in artifact_paths.items()}
    observed_hashes["state_artifact_tree_sha256"] = state_artifact_tree_sha256(
        labels / "states"
    )
    for name, observed in observed_hashes.items():
        require(
            str((collection_report.get("artifacts") or {}).get(name)) == observed,
            f"label artifact SHA-256 changed: {name}",
        )

    report = {
        "schema": LABEL_AUDIT_SCHEMA,
        "scientific_status": "immutable_size_labels_verified_by_current_code"
        if not errors
        else "failed",
        "source_run_fingerprint": original_run.get("run_fingerprint"),
        "source_producer": original_run.get("producer"),
        "auditor": producer_identity(
            project_root=project_root,
            source_files=PRODUCER_FILES,
            native_required=True,
            package_names=("numpy",),
        ),
        "state_count": len(state_manifest),
        "candidate_count": len(aggregates),
        "trial_count": len(trials),
        "reused_trial_count": reused_count,
        "new_trial_count": new_count,
        "matrix_validation": matrix_validation,
        "forbidden_fields": forbidden_hits,
        "source_artifacts": observed_hashes,
        "source_artifacts_modified": False,
        "formal_ttf_claim": False,
        "passed": not errors,
        "errors": errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "audit_report.json", report)
    return report


def _family_variant(family: str) -> tuple[str, int, str]:
    prefix, raw_size = family.rsplit(":", 1)
    mapping = {
        "structpool-bottleneck-crossing": ("bottleneck_crossing", "bottleneck_crossing"),
        "structpool-conflict-component": ("conflict_component", "conflict_component"),
        "structpool-boundary-articulation": ("topology_boundary_articulation", "topology_boundary"),
        "structpool-boundary-low_degree": ("topology_boundary_low_degree", "topology_boundary"),
        "structpool-spatiotemporal-hotspot": ("spatiotemporal_hotspot", "spatiotemporal_hotspot"),
        "structpool-path-overlap": ("path_overlap", "path_overlap"),
    }
    if prefix not in mapping:
        raise ValueError(f"unknown StructPool size-grid family: {family}")
    variant, group = mapping[prefix]
    return variant, int(raw_size), group


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        **{
            f"mean_{name}": statistics.fmean(float(row[name]) for row in rows)
            for name in (
                "seed_mean",
                "lower_half_mean",
                "no_progress_rate",
                "seed_standard_deviation",
                "repair_success_rate",
                "feasible_rate",
            )
        },
    } if rows else {"count": 0}


def _best(rows: list[dict[str, Any]], key: str = "seed_mean") -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            -float(row[key]),
            float(row["no_progress_rate"]),
            -float(row["lower_half_mean"]),
            str(row["candidate_id"]),
        ),
    )


def _fixed_half_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique = {str(row["candidate_id"]): row for row in rows}
    candidates = list(unique.values())
    first_best = _best(candidates, key="first_fixed_half_mean")
    second_best = _best(candidates, key="second_fixed_half_mean")
    first_top3 = {
        str(row["candidate_id"])
        for row in sorted(
            candidates,
            key=lambda row: (
                -float(row["first_fixed_half_mean"]),
                str(row["candidate_id"]),
            ),
        )[:3]
    }
    second_top3 = {
        str(row["candidate_id"])
        for row in sorted(
            candidates,
            key=lambda row: (
                -float(row["second_fixed_half_mean"]),
                str(row["candidate_id"]),
            ),
        )[:3]
    }
    return {
        "first_winner_candidate_id": str(first_best["candidate_id"]),
        "second_winner_candidate_id": str(second_best["candidate_id"]),
        "exact_winner_agreement": (
            first_best["candidate_id"] == second_best["candidate_id"]
        ),
        "top3_overlap": len(first_top3 & second_top3) / max(
            1, len(first_top3 | second_top3)
        ),
    }


def _grouped_family_size_quality(
    rows: list[dict[str, Any]], dimensions: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        for dimension in dimensions:
            grouped[
                (
                    dimension,
                    str(row[dimension]),
                    str(row["family"]),
                    int(row["nominal_size"]),
                )
            ].append(row)
    return [
        {
            "dimension": dimension,
            "group_value": value,
            "family": family,
            "nominal_size": size,
            **_summary(group_rows),
        }
        for (dimension, value, family, size), group_rows in sorted(grouped.items())
    ]


def _best_size_counts_by_context(
    rows: list[dict[str, Any]], dimensions: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in dimensions:
        values: dict[str, Any] = {}
        for value in sorted({str(row[dimension]) for row in rows}):
            families: dict[str, Any] = {}
            subset = [row for row in rows if str(row[dimension]) == value]
            for family in sorted({str(row["family"]) for row in subset}):
                families[family] = dict(
                    sorted(
                        collections.Counter(
                            int(row["best_size"])
                            for row in subset
                            if str(row["family"]) == family
                        ).items()
                    )
                )
            values[value] = families
        result[dimension] = values
    return result


def analyze_size_ablation(
    *, config_path: str | Path, labels: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_size_ablation_config(config, project_root=project_root)
    labels = Path(labels).resolve()
    collection = _read_json(labels / "collection_report.json")
    if collection.get("passed") is not True:
        raise ValueError("size ablation requires complete labels")
    aggregates = _read_jsonl(labels / "candidate_aggregates.jsonl")
    expanded = []
    for row in aggregates:
        component_size = float(row["features"].get("state.largest_component", 0.0))
        component_band = (
            "le8" if component_size <= 8 else
            "9_16" if component_size <= 16 else
            "17_32" if component_size <= 32 else "gt32"
        )
        for family in row["selection_families"]:
            variant, size, group = _family_variant(str(family))
            support = int(row["structpool_support_count_by_family"][family])
            support_band = (
                "le8" if support <= 8 else
                "9_16" if support <= 16 else
                "17_32" if support <= 32 else "gt32"
            )
            expanded.append(
                {
                    **row,
                    "family": variant,
                    "family_group": group,
                    "nominal_size": size,
                    "support_count": support,
                    "support_band": support_band,
                    "component_band": component_band,
                    "agent_band": str(row["agent_band"]),
                    "conflict_band": str(row["conflict_band"]),
                    "pure_or_mixed": (
                        "pure" if row["structpool_grid_pure_family"] else "mixed"
                    ),
                }
            )
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_state_family: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in expanded:
        by_state[str(row["state_id"])].append(row)
        by_state_family[(str(row["state_id"]), str(row["family"]))].append(row)
    fixed = dict(config["analysis"]["current_fixed_sizes"])
    state_rows = []
    family_state_rows = []
    for state_id, rows in sorted(by_state.items()):
        unique = {str(row["candidate_id"]): row for row in rows}
        best = _best(list(unique.values()))
        current = {
            str(row["candidate_id"]): row
            for row in rows
            if int(row["nominal_size"]) == int(fixed[row["family_group"]])
        }
        current_best = _best(list(current.values()))
        half = _fixed_half_consistency(list(unique.values()))
        state_rows.append(
            {
                "state_id": state_id,
                "map_id": str(best["map_id"]),
                "layout_mode": str(best["layout_mode"]),
                "best_candidate_id": str(best["candidate_id"]),
                "current_candidate_id": str(current_best["candidate_id"]),
                "normalized_regret": float(best["seed_mean"]) - float(current_best["seed_mean"]),
                "fixed_half_exact_winner_agreement": bool(
                    half["exact_winner_agreement"]
                ),
                "fixed_half_top3_overlap": float(half["top3_overlap"]),
            }
        )
    for (state_id, family), rows in sorted(by_state_family.items()):
        unique = {str(row["candidate_id"]): row for row in rows}
        best = _best(list(unique.values()))
        group = str(rows[0]["family_group"])
        current = [
            row for row in rows
            if int(row["nominal_size"]) == int(fixed[group])
        ]
        if not current:
            raise ValueError(f"fixed family size is absent: {state_id} {family}")
        current_best = _best(current)
        half = _fixed_half_consistency(list(unique.values()))
        family_state_rows.append(
            {
                "state_id": state_id,
                "family": family,
                "best_size": int(best["nominal_size"]),
                "best_candidate_id": str(best["candidate_id"]),
                "current_size": int(fixed[group]),
                "current_candidate_id": str(current_best["candidate_id"]),
                "normalized_regret": float(best["seed_mean"]) - float(current_best["seed_mean"]),
                "support_count": int(best["support_count"]),
                "support_band": str(best["support_band"]),
                "component_band": str(best["component_band"]),
                "agent_band": str(best["agent_band"]),
                "conflict_band": str(best["conflict_band"]),
                "pure_or_mixed": str(best["pure_or_mixed"]),
                "map_id": str(best["map_id"]),
                "layout_mode": str(best["layout_mode"]),
                "fixed_half_exact_winner_agreement": bool(
                    half["exact_winner_agreement"]
                ),
                "fixed_half_top3_overlap": float(half["top3_overlap"]),
            }
        )
    family_size = {}
    for key, group_rows in sorted(
        collections.defaultdict(list, {
            key: [row for row in expanded if (row["family"], row["nominal_size"]) == key]
            for key in {(row["family"], row["nominal_size"]) for row in expanded}
        }).items()
    ):
        family_size[f"{key[0]}:{key[1]}"] = _summary(group_rows)
    best_size_counts = {
        family: dict(sorted(collections.Counter(
            int(row["best_size"]) for row in family_state_rows if row["family"] == family
        ).items()))
        for family in sorted({str(row["family"]) for row in family_state_rows})
    }
    context_dimensions = [
        name
        for name in config["analysis"]["report_by"]
        if name not in {"family", "size"}
    ]
    grouped_quality = _grouped_family_size_quality(expanded, context_dimensions)
    contextual_best_sizes = _best_size_counts_by_context(
        family_state_rows, context_dimensions
    )
    report = {
        "schema": ANALYSIS_SCHEMA,
        "scientific_status": "completed_current_step_family_size_ablation",
        "state_count": len(state_rows),
        "candidate_count": len(aggregates),
        "expanded_family_size_row_count": len(expanded),
        "family_size_quality": family_size,
        "current_fixed_global_regret": {
            "mean": statistics.fmean(row["normalized_regret"] for row in state_rows),
            "maximum": max(row["normalized_regret"] for row in state_rows),
            "positive_fraction": statistics.fmean(row["normalized_regret"] > 1e-12 for row in state_rows),
        },
        "current_fixed_family_regret": {
            family: {
                "mean": statistics.fmean(
                    row["normalized_regret"] for row in family_state_rows if row["family"] == family
                ),
                "maximum": max(
                    row["normalized_regret"] for row in family_state_rows if row["family"] == family
                ),
            }
            for family in sorted({str(row["family"]) for row in family_state_rows})
        },
        "fixed_half_consistency": {
            "exact_winner_agreement_rate": statistics.fmean(
                row["fixed_half_exact_winner_agreement"] for row in state_rows
            ),
            "mean_top3_overlap": statistics.fmean(
                row["fixed_half_top3_overlap"] for row in state_rows
            ),
        },
        "family_fixed_half_consistency": {
            "exact_winner_agreement_rate": statistics.fmean(
                row["fixed_half_exact_winner_agreement"]
                for row in family_state_rows
            ),
            "mean_top3_overlap": statistics.fmean(
                row["fixed_half_top3_overlap"] for row in family_state_rows
            ),
        },
        "best_size_counts_by_family": best_size_counts,
        "best_size_counts_by_context": contextual_best_sizes,
        "grouped_family_size_quality_row_count": len(grouped_quality),
        "uniform_best_size_exists": all(
            len([size for size, count in counts.items() if count]) == 1
            for counts in best_size_counts.values()
        ),
        "pure_family_quality": _summary([row for row in expanded if row["pure_or_mixed"] == "pure"]),
        "mixed_family_quality": _summary([row for row in expanded if row["pure_or_mixed"] == "mixed"]),
        "formal_ttf_claim": False,
        "known_maze_long_tail_included": False,
        "next_decision": "design_stride_scalepool_v1_from_support_conditioned_size_results",
        "artifacts": {},
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expanded_path = output / "expanded_family_size_rows.jsonl"
    state_path = output / "state_regret.jsonl"
    family_state_path = output / "family_state_regret.jsonl"
    grouped_quality_path = output / "grouped_family_size_quality.jsonl"
    _write_jsonl(expanded_path, expanded)
    _write_jsonl(state_path, state_rows)
    _write_jsonl(family_state_path, family_state_rows)
    _write_jsonl(grouped_quality_path, grouped_quality)
    report["artifacts"] = {
        "expanded_family_size_rows_sha256": sha256_file(expanded_path),
        "state_regret_sha256": sha256_file(state_path),
        "family_state_regret_sha256": sha256_file(family_state_path),
        "grouped_family_size_quality_sha256": sha256_file(grouped_quality_path),
        "candidate_aggregates_sha256": sha256_file(labels / "candidate_aggregates.jsonl"),
        "repair_trials_sha256": sha256_file(labels / "repair_trials.jsonl"),
    }
    _write_json(output / "size_ablation_report.json", report)
    return report


__all__ = [
    "ANALYSIS_SCHEMA",
    "CONFIG_SCHEMA",
    "GRID_REPORT_SCHEMA",
    "LABEL_REPORT_SCHEMA",
    "analyze_size_ablation",
    "collect_size_labels",
    "extract_size_grid",
    "validate_size_ablation_config",
]
