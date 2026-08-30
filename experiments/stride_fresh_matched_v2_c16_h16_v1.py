from __future__ import annotations

import hashlib
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, registered_input, sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
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
    FULL_POOL_PROPOSAL,
    _paired_action,
    _replay_job,
    _validate_native_repair,
)
from experiments.stride_repairability_collection import (
    _source_target_state,
    repairability_restore_seed,
)
from experiments.trace_replay import result_blind_decision_rows, restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import (
    generate_online_candidates,
    score_online_candidates,
)
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_subset,
)


CONFIG_SCHEMA = "lns2.stride.fresh_matched_v2_c16_h16_config.v1"
PLAN_SCHEMA = "lns2.stride.fresh_matched_v2_c16_h16_plan.v1"
PREFLIGHT_EPISODE_SCHEMA = (
    "lns2.stride.fresh_matched_v2_c16_h16_preflight_episode.v1"
)
SELECTION_SCHEMA = "lns2.stride.fresh_matched_v2_c16_h16_selection.v1"
H1_STATE_SCHEMA = "lns2.stride.fresh_matched_v2_c16_h16_h1_state.v1"
H1_TRIAL_SCHEMA = "lns2.stride.fresh_matched_v2_c16_h16_h1_trial.v1"
H1_REPORT_SCHEMA = "lns2.stride.fresh_matched_v2_c16_h16_h1_report.v1"
EXPERIMENT_ID = "stride_fresh_matched_v2_c16_h16_v1"
PROFILE = "realized_dynamic"
ARMS = ("v2_anchor", "component16", "hotspot16")
TRIAL_INDICES = tuple(range(16))
FIRST_HALF = tuple(range(8))
SECOND_HALF = tuple(range(8, 16))
STRUCTURAL_FAMILIES = {
    "component16": "structpool-conflict-component:16",
    "hotspot16": "structpool-spatiotemporal-hotspot:16",
}


def _registered(root: Path, specification: dict[str, Any], label: str) -> Path:
    return registered_input(root, specification, label=label)


def _strict_int(value: Any, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError("expected a strict integer")
    return int(value)


def _source_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("fresh matched source registry is empty")
    return {str(key): dict(value) for key, value in sources.items()}


def validate_config(config: dict[str, Any], *, project_root: Path | None = None) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_fresh_exact_task_solver_state_h1_opportunity_collection"
        or config.get("freshness") != "fresh_exact_task_solver_state_cohort"
        or config.get("research_split") != "fresh_matched_development"
    ):
        raise ValueError("fresh matched experiment identity changed")

    bundle = dict(config.get("controller_bundle") or {})
    if bundle.get("controller") != "v2-full" or set(bundle) != {
        "controller",
        "manifest",
    }:
        raise ValueError("fresh matched V2 bundle contract changed")

    sources = _source_specs(config)
    if set(sources) != {"movingai_ood", "balanced_wall_clock"}:
        raise ValueError("fresh matched source datasets changed")
    task_ids: set[str] = set()
    map_ids: set[str] = set()
    family_maps: dict[str, set[str]] = defaultdict(set)
    total_tasks = 0
    for source_id, source in sources.items():
        required = {
            "dataset_root",
            "manifest",
            "runtime",
            "source_split",
            "dataset_design",
            "tasks",
        }
        if set(source) != required:
            raise ValueError(f"fresh matched source fields changed: {source_id}")
        design = dict(source["dataset_design"])
        if source_id == "balanced_wall_clock":
            if design != {
                "mode": "balanced_wall_clock",
                "map_count": 6,
                "instance_count": 24,
                "source_counts": {"movingai": 24},
                "layout_counts": {
                    "game": 4,
                    "maze": 4,
                    "random": 4,
                    "room": 4,
                    "warehouse": 8,
                },
                "historical_map_ids": [],
            }:
                raise ValueError("fresh matched balanced dataset design changed")
        else:
            expected_maps = {
                "random-32-32-10": ("random", (100, 200)),
                "random-64-64-10": ("random", (400, 600)),
                "random-64-64-20": ("random", (400, 600)),
                "maze-32-32-4": ("maze", (100, 200)),
                "maze-128-128-1": ("maze", (400, 600)),
                "maze-128-128-10": ("maze", (400, 600)),
                "room-64-64-8": ("room", (400, 600)),
                "room-64-64-16": ("room", (400, 600)),
                "warehouse-10-20-10-2-2": ("warehouse", (300, 500)),
                "warehouse-20-40-10-2-2": ("warehouse", (400, 600)),
                "den312d": ("game", (200, 300)),
                "lak303d": ("game", (400, 600)),
            }
            observed_maps = {
                str(row["map_id"]): (
                    str(row["layout_family"]),
                    tuple(map(int, row["agent_counts"])),
                )
                for row in design.get("maps", [])
            }
            if (
                set(design) != {
                    "mode",
                    "map_count",
                    "task_count",
                    "scenario_indices",
                    "layout_family_counts",
                    "maps",
                    "historical_map_ids",
                }
                or design.get("mode") != "movingai_ood"
                or int(design.get("map_count", -1)) != 12
                or int(design.get("task_count", -1)) != 48
                or list(design.get("scenario_indices") or ()) != [4, 5]
                or dict(design.get("layout_family_counts") or {})
                != {"random": 3, "maze": 3, "room": 2, "warehouse": 2, "game": 2}
                or observed_maps != expected_maps
                or set(map(str, design.get("historical_map_ids") or ()))
                != {
                    "random-32-32-20",
                    "maze-32-32-2",
                    "room-32-32-4",
                    "warehouse-10-20-10-2-1",
                    "warehouse-20-40-10-2-1",
                    "den520d",
                }
            ):
                raise ValueError("fresh matched MovingAI dataset design changed")
        tasks = source["tasks"]
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"fresh matched source has no tasks: {source_id}")
        for task_value in tasks:
            task = dict(task_value)
            if set(task) != {
                "task_id",
                "task_sha256",
                "map_id",
                "map_family",
                "map_sha256",
            }:
                raise ValueError("fresh matched task identity fields changed")
            task_id = str(task["task_id"])
            map_id = str(task["map_id"])
            family = str(task["map_family"])
            if task_id in task_ids:
                raise ValueError(f"duplicate fresh matched task: {task_id}")
            if family not in {"game", "maze", "random", "room", "warehouse"}:
                raise ValueError(f"unknown fresh matched map family: {family}")
            task_ids.add(task_id)
            map_ids.add(map_id)
            family_maps[family].add(map_id)
            total_tasks += 1
    if total_tasks != 24 or len(map_ids) != 12:
        raise ValueError("fresh matched task or map count changed")
    if {family: len(maps) for family, maps in family_maps.items()} != {
        "game": 2,
        "maze": 2,
        "random": 3,
        "room": 2,
        "warehouse": 3,
    }:
        raise ValueError("fresh matched family map quotas changed")

    source = dict(config.get("source_collection") or {})
    if source != {
        "policy": "v2-full",
        "manifest_policy": "realized_dynamic",
        "solver_seeds": [41, 42],
        "expected_episode_count": 48,
        "max_decisions": 12,
        "workers": 16,
        "environment_time_limit_seconds": 200.0,
        "wall_time_budget_seconds": 200.0,
        "episode_process_timeout_seconds": 240.0,
        "deterministic_pp_replay": True,
        "fresh_reset_qualification_required": True,
    }:
        raise ValueError("fresh matched source collection contract changed")

    selection = dict(config.get("state_selection") or {})
    if selection != {
        "target_state_count": 96,
        "states_per_episode": 2,
        "eligible_before_conflicts_minimum": 1,
        "structural_required_actual_size": 16,
        "required_pairwise_distinct_arms": True,
        "rank_rule": "sha256_preaction_identity_ascending",
        "failure_action": "STATE_SUPPLY_FAIL_before_h1",
        "target_outcome_fields_read": False,
        "source_episode_outcome_used_to_filter": False,
    }:
        raise ValueError("fresh matched state-selection contract changed")

    h1 = dict(config.get("h1") or {})
    if h1 != {
        "arms": list(ARMS),
        "trial_indices": list(TRIAL_INDICES),
        "first_fixed_half": list(FIRST_HALF),
        "second_fixed_half": list(SECOND_HALF),
        "logical_trial_count": 4608,
        "workers": 16,
        "per_action_time_limit_seconds": 5.0,
        "per_state_process_fuse_seconds": 420.0,
        "same_state_trial_seed_across_arms": True,
        "runtime_or_pp_seconds_used_in_label": False,
    }:
        raise ValueError("fresh matched H1 execution contract changed")

    label = dict(config.get("h1_label") or {})
    if label != {
        "per_seed_score": "normalized_current_step_conflict_reduction",
        "minimum_strict_paired_wins": 12,
        "minimum_mean_delta": 0.02,
        "require_positive_first_half_mean_delta": True,
        "require_positive_second_half_mean_delta": True,
        "require_no_progress_rate_noninferiority": True,
        "require_rollback_rate_noninferiority": True,
        "require_time_limit_rate_noninferiority": True,
        "tie_epsilon": 1e-12,
        "duplicates_or_unavailable_label": None,
    }:
        raise ValueError("fresh matched H1 label contract changed")

    gates = dict(config.get("h1_gates") or {})
    if (
        int(gates.get("expected_state_count", -1)) != 96
        or int(gates.get("minimum_opportunity_state_count", -1)) != 20
        or int(gates.get("minimum_opportunity_map_count", -1)) != 8
        or int(gates.get("minimum_opportunity_family_count", -1)) != 4
        or gates.get("each_fold_requires_opportunity_and_nonopportunity") is not True
        or gates.get("valid_candidate_labels_require_both_classes") is not True
        or gates.get("all_integrity_gates_required") is not True
    ):
        raise ValueError("fresh matched H1 gates changed")
    folds = dict(gates.get("map_folds") or {})
    if set(folds) != {"fold0", "fold1", "fold2", "fold3"}:
        raise ValueError("fresh matched map folds changed")
    flattened = [str(map_id) for fold in folds.values() for map_id in fold]
    if len(flattened) != 12 or set(flattened) != map_ids:
        raise ValueError("fresh matched map folds do not cover the cohort once")

    h8 = dict(config.get("h8_future_contract") or {})
    if h8 != {
        "registered_but_not_executed_in_h1_stage": True,
        "requires_global_h1_pass": True,
        "run_all_96_states_and_all_canonical_arms": True,
        "trial_indices": list(TRIAL_INDICES),
        "total_horizon_including_h1": 8,
        "continuation_steps": 7,
        "continuation_controller": "fresh_exact_base_only_v2",
        "same_state_trial_step_seed_across_arms": True,
        "positive_only_continuation_forbidden": True,
    }:
        raise ValueError("fresh matched future H8 contract changed")

    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "conditional_on_three_distinct_active_cohort": True,
        "h1_opportunity_only": True,
        "training_allowed": False,
        "h8_allowed_only_on_h1_pass": True,
        "ttf_or_speed_claim_allowed": False,
        "fresh_map_claim_allowed": False,
        "no_outcome_based_state_or_arm_deletion": True,
    }:
        raise ValueError("fresh matched claim boundary changed")

    if project_root is not None:
        root = project_root.resolve()
        _registered(root, dict(bundle["manifest"]), "fresh matched controller manifest")
        for source_id, source_spec in sources.items():
            manifest = _registered(
                root, dict(source_spec["manifest"]), f"{source_id} dataset manifest"
            )
            _registered(root, dict(source_spec["runtime"]), f"{source_id} runtime")
            dataset_root = (root / str(source_spec["dataset_root"])).resolve()
            if manifest.parent != dataset_root:
                raise ValueError(f"fresh matched dataset root differs: {source_id}")
            manifest_rows = {
                str(row["task_id"]): row for row in _read_jsonl(manifest)
            }
            for task in source_spec["tasks"]:
                row = manifest_rows.get(str(task["task_id"]))
                if row is None:
                    raise ValueError(f"fresh matched task is absent: {task['task_id']}")
                task_path = contained_file(dataset_root, row["task_file"], field="task")
                map_path = contained_file(dataset_root, row["map_file"], field="map")
                if sha256_file(task_path) != str(task["task_sha256"]):
                    raise ValueError(f"fresh matched task changed: {task['task_id']}")
                if sha256_file(map_path) != str(task["map_sha256"]):
                    raise ValueError(f"fresh matched map changed: {task['map_id']}")
                if (
                    str(row["map_id"]) != str(task["map_id"])
                    or str(row["layout_mode"]) != str(task["map_family"])
                    or str(row["split"]) != str(source_spec["source_split"])
                ):
                    raise ValueError(f"fresh matched task metadata changed: {task['task_id']}")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    validate_config(config, project_root=root)
    return path, root, config


def source_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = list(map(int, config["source_collection"]["solver_seeds"]))
    rows = []
    for source_id, source in sorted(_source_specs(config).items()):
        for task in source["tasks"]:
            for seed in seeds:
                rows.append(
                    {
                        "source_id": source_id,
                        "task_id": str(task["task_id"]),
                        "map_id": str(task["map_id"]),
                        "map_family": str(task["map_family"]),
                        "solver_seed": seed,
                    }
                )
    rows.sort(key=lambda row: (row["source_id"], row["task_id"], row["solver_seed"]))
    if len(rows) != 48:
        raise ValueError("fresh matched source schedule is not 48 episodes")
    return rows


def matched_pp_seed(state_occurrence_id: str, trial_index: int, step_index: int = 0) -> int:
    if not state_occurrence_id or trial_index not in TRIAL_INDICES or step_index < 0:
        raise ValueError("invalid fresh matched PP seed identity")
    digest = _fingerprint(
        {
            "namespace": "stride-fresh-matched-v2-c16-h16-paired-pp-v1",
            "state_occurrence_id": str(state_occurrence_id),
            "trial_index": int(trial_index),
            "step_index": int(step_index),
        }
    )
    return int(digest[:16], 16) % (2**31)


def _preaction_rank(row: dict[str, Any]) -> str:
    return _fingerprint(
        {
            "namespace": "stride-fresh-matched-v2-c16-h16-state-rank-v1",
            "source_id": str(row["source_id"]),
            "episode_id": str(row["episode_id"]),
            "task_id": str(row["task_id"]),
            "solver_seed": int(row["solver_seed"]),
            "decision_index": int(row["decision_index"]),
            "before_fingerprint": str(row["before_fingerprint"]),
        }
    )


def select_preflight_states(
    rows: Iterable[dict[str, Any]], *, states_per_episode: int = 2
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_rows = [dict(row) for row in rows]
    for row in all_rows:
        grouped[(str(row["source_id"]), str(row["episode_id"]))].append(row)
    selected: list[dict[str, Any]] = []
    failed = []
    for key, values in sorted(grouped.items()):
        eligible = [row for row in values if bool(row.get("eligible"))]
        eligible.sort(key=lambda row: (_preaction_rank(row), int(row["decision_index"])))
        if len(eligible) < states_per_episode:
            failed.append({"source_id": key[0], "episode_id": key[1], "eligible": len(eligible)})
            continue
        for row in eligible[:states_per_episode]:
            identity = {
                "source_id": str(row["source_id"]),
                "episode_id": str(row["episode_id"]),
                "decision_index": int(row["decision_index"]),
                "before_fingerprint": str(row["before_fingerprint"]),
            }
            selected.append(
                {
                    **row,
                    "schema": SELECTION_SCHEMA,
                    "state_occurrence_id": "fresh-matched-" + _fingerprint(identity)[:24],
                    "selection_rank_sha256": _preaction_rank(row),
                    "target_outcome_fields_read": False,
                }
            )
    selected.sort(key=lambda row: str(row["state_occurrence_id"]))
    report = {
        "preaction_state_count": len(all_rows),
        "eligible_preaction_state_count": sum(bool(row.get("eligible")) for row in all_rows),
        "eligible_preaction_fraction": (
            sum(bool(row.get("eligible")) for row in all_rows) / len(all_rows)
            if all_rows
            else 0.0
        ),
        "episode_count": len(grouped),
        "failed_episode_count": len(failed),
        "failed_episodes": failed,
        "selected_state_count": len(selected),
        "status": "ok" if not failed else "STATE_SUPPLY_FAIL",
    }
    return selected, report


def build_plan(config_path: str | Path) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    schedule = source_schedule(config)
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "source_episode_count": len(schedule),
        "source_workers": int(config["source_collection"]["workers"]),
        "target_state_count": int(config["state_selection"]["target_state_count"]),
        "states_per_episode": int(config["state_selection"]["states_per_episode"]),
        "arms": list(ARMS),
        "trial_count_per_arm": len(TRIAL_INDICES),
        "logical_h1_trial_count": 96 * len(ARMS) * len(TRIAL_INDICES),
        "h1_workers": int(config["h1"]["workers"]),
        "h1_per_action_time_limit_seconds": float(
            config["h1"]["per_action_time_limit_seconds"]
        ),
        "h8_execution_registered": True,
        "h8_executed_by_this_stage": False,
        "ttf_or_training_run": False,
        "source_schedule_sha256": _fingerprint(schedule),
        "source_schedule": schedule,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, path)


def _source_runtime_config(
    base: dict[str, Any], source_spec: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    contract = dict(config["source_collection"])
    runtime = {
        **base,
        "split": str(source_spec["source_split"]),
        "solver_seeds": list(map(int, contract["solver_seeds"])),
        # The legacy closed-loop validator requires both registered policy
        # names. Execution below explicitly runs only qualification and V2.
        "policies": ["official_adaptive", PROFILE],
        "environment": {
            **dict(base["environment"]),
            "time_limit": float(contract["environment_time_limit_seconds"]),
            "unlimited_time": False,
        },
        "max_decisions": int(contract["max_decisions"]),
        "metric_iteration_budget": int(contract["max_decisions"]),
        "wall_time_budget_seconds": float(contract["wall_time_budget_seconds"]),
        "episode_process_timeout_seconds": float(
            contract["episode_process_timeout_seconds"]
        ),
        "workers": int(contract["workers"]),
        "deterministic_pp_replay": True,
        "dataset_design": dict(source_spec["dataset_design"]),
    }
    return runtime


def _require_source_design_pass(summary: dict[str, Any], source_id: str) -> None:
    design = dict(summary.get("dataset_design") or {})
    if design.get("passed") is not True:
        raise ValueError(
            f"fresh matched source dataset design failed: {source_id}: "
            f"{design.get('errors')}"
        )


def run_source_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Collect the 48 fresh V2 episodes, or perform a solver-free dry run."""

    path, root, config = load_config(config_path)
    output_root = Path(output).resolve()
    bundle_manifest = _registered(
        root, dict(config["controller_bundle"]["manifest"]), "fresh matched bundle"
    )
    summaries: dict[str, Any] = {}
    for source_id, source in sorted(_source_specs(config).items()):
        dataset_root = (root / str(source["dataset_root"])).resolve()
        collection_dataset_root = dataset_root.parent
        runtime_path = _registered(
            root, dict(source["runtime"]), f"fresh matched {source_id} runtime"
        )
        runtime = _source_runtime_config(_read_json(runtime_path), source, config)
        task_ids = [str(row["task_id"]) for row in source["tasks"]]
        keys = {
            (task_id, int(seed))
            for task_id in task_ids
            for seed in config["source_collection"]["solver_seeds"]
        }
        source_output = output_root / "source" / source_id
        if dry_run:
            with tempfile.TemporaryDirectory(prefix="stride-fresh-matched-") as temporary:
                derived_path = Path(temporary) / "runtime.json"
                _write_json(derived_path, runtime)
                calls = []
                for phase in ("qualify", PROFILE):
                    call = run_closed_loop_collection(
                            collection_dataset_root,
                            derived_path,
                            source_output,
                            phase=phase,
                            workers=16,
                            resume=False,
                            dry_run=True,
                            task_ids=task_ids,
                            controller="v2-full",
                            feature_backend="native",
                            controller_bundle=bundle_manifest.parent,
                            controller_runtime="optimized",
                            verification_profile="deployment",
                            job_keys=keys,
                            cohort_job_keys=keys,
                            wall_time_budget_seconds=200.0,
                            episode_process_timeout_seconds=240.0,
                            environment_time_limit_seconds=200.0,
                            qualification_process_timeout_seconds=240.0,
                            stopping_rule="historical",
                            qualification_source=None,
                            deterministic_pp_replay=True,
                        )
                    _require_source_design_pass(call, source_id)
                    calls.append(call)
                summary = {"qualification": calls[0], PROFILE: calls[1]}
        else:
            derived_path = source_output / "fresh_runtime_config.json"
            source_output.mkdir(parents=True, exist_ok=True)
            if derived_path.is_file() and _read_json(derived_path) != runtime:
                raise ValueError(f"fresh matched derived runtime changed: {source_id}")
            _write_json(derived_path, runtime)
            common = {
                "workers": 16,
                "task_ids": task_ids,
                "controller": "v2-full",
                "feature_backend": "native",
                "controller_bundle": bundle_manifest.parent,
                "controller_runtime": "optimized",
                "verification_profile": "deployment",
                "job_keys": keys,
                "cohort_job_keys": keys,
                "wall_time_budget_seconds": 200.0,
                "episode_process_timeout_seconds": 240.0,
                "environment_time_limit_seconds": 200.0,
                "qualification_process_timeout_seconds": 240.0,
                "stopping_rule": "historical",
                "qualification_source": None,
                "deterministic_pp_replay": True,
            }
            preview = run_closed_loop_collection(
                collection_dataset_root,
                derived_path,
                source_output,
                phase="qualify",
                resume=False,
                dry_run=True,
                **common,
            )
            _require_source_design_pass(preview, source_id)
            qualification = run_closed_loop_collection(
                collection_dataset_root,
                derived_path,
                source_output,
                phase="qualify",
                resume=resume,
                dry_run=False,
                **common,
            )
            policy = run_closed_loop_collection(
                collection_dataset_root,
                derived_path,
                source_output,
                phase=PROFILE,
                resume=True,
                dry_run=False,
                **common,
            )
            summary = {"qualification": qualification, PROFILE: policy}
        summaries[source_id] = summary
    report = {
        "schema": "lns2.stride.fresh_matched_v2_c16_h16_source.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "dry_run": bool(dry_run),
        "fresh_reset_qualification_reused": False,
        "source_root_identity_isolated": True,
        "expected_episode_count": 48,
        "workers": 16,
        "summaries": summaries,
    }
    if not dry_run:
        manifest_hashes = {}
        observed_keys = set()
        for source_id, source in sorted(_source_specs(config).items()):
            manifest_path = (
                output_root / "source" / source_id / "realized_dynamic_manifest.jsonl"
            )
            rows = _read_jsonl(manifest_path)
            expected_keys = {
                (source_id, str(task["task_id"]), int(seed))
                for task in source["tasks"]
                for seed in config["source_collection"]["solver_seeds"]
            }
            source_keys = {
                (
                    source_id,
                    str(row.get("task_id")),
                    int(row.get("solver_seed", -1)),
                )
                for row in rows
            }
            if source_keys != expected_keys or len(rows) != len(expected_keys):
                raise RuntimeError(f"fresh matched source product changed: {source_id}")
            if any(
                str(row.get("status")) != "ok" or not row.get("trace_file")
                for row in rows
            ):
                raise RuntimeError(f"fresh matched source product is incomplete: {source_id}")
            for row in rows:
                trace_path = contained_file(
                    manifest_path.parent,
                    row.get("trace_file"),
                    field="fresh matched trace",
                )
                if sha256_file(trace_path) != str(row.get("trace_sha256")):
                    raise RuntimeError(
                        f"fresh matched source trace changed: {source_id}:"
                        f"{row.get('episode_id')}"
                    )
            observed_keys.update(source_keys)
            manifest_hashes[source_id] = sha256_file(manifest_path)
        if len(observed_keys) != 48:
            raise RuntimeError("fresh matched source product is not exactly 48 episodes")
        report["source_manifest_sha256"] = manifest_hashes
        report["observed_episode_count"] = len(observed_keys)
        report["complete"] = True
        _atomic_json(output_root / "source_collection_report.json", report)
    return report


def _task_registry(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (source_id, str(task["task_id"])): dict(task)
        for source_id, source in _source_specs(config).items()
        for task in source["tasks"]
    }


def load_source_preaction_rows(
    config: dict[str, Any], output: str | Path
) -> list[dict[str, Any]]:
    """Read only replayable prefixes; never expose the target transition outcome."""

    output_root = Path(output).resolve()
    registry = _task_registry(config)
    expected = {
        (row["source_id"], row["task_id"], int(row["solver_seed"]))
        for row in source_schedule(config)
    }
    observed: set[tuple[str, str, int]] = set()
    decisions: list[dict[str, Any]] = []
    for source_id, source in sorted(_source_specs(config).items()):
        source_root = output_root / "source" / source_id
        manifest_path = source_root / "realized_dynamic_manifest.jsonl"
        if not manifest_path.is_file():
            raise ValueError(f"fresh matched source manifest is missing: {source_id}")
        for manifest in _read_jsonl(manifest_path):
            key = (
                source_id,
                str(manifest.get("task_id")),
                int(manifest.get("solver_seed", -1)),
            )
            if key not in expected or key in observed:
                raise ValueError(f"fresh matched source episode identity changed: {key}")
            if str(manifest.get("status")) != "ok" or not manifest.get("trace_file"):
                raise ValueError(f"fresh matched source episode is incomplete: {key}")
            trace_path = contained_file(
                source_root, manifest.get("trace_file"), field="fresh matched trace"
            )
            trace_sha256 = sha256_file(trace_path)
            if trace_sha256 != str(manifest.get("trace_sha256")):
                raise ValueError(f"fresh matched source trace changed: {key}")
            observed.add(key)
            task = registry[(source_id, key[1])]
            rows, _events = result_blind_decision_rows(source_root, manifest)
            for row in rows:
                decision_index = int(row["decision_index"])
                if decision_index >= int(config["source_collection"]["max_decisions"]):
                    continue
                identity = {
                    "source_id": source_id,
                    "episode_id": str(manifest["episode_id"]),
                    "task_id": key[1],
                    "solver_seed": key[2],
                    "decision_index": decision_index,
                    "before_fingerprint": str(row["before_fingerprint"]),
                }
                decisions.append(
                    {
                        **row,
                        **identity,
                        "state_id": "fresh-preaction-" + _fingerprint(identity)[:24],
                        "source_root": str(source_root),
                        "source_policy": "v2-full",
                        "split": str(source["source_split"]),
                        "research_split": str(config["research_split"]),
                        "map_id": str(task["map_id"]),
                        "map_family": str(task["map_family"]),
                        "layout_mode": str(task["map_family"]),
                        "agent_count": int(manifest.get("agent_count", 0)),
                        "source_trace_sha256": trace_sha256,
                        "target_outcome_fields_read": False,
                    }
                )
    if observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"fresh matched source product is incomplete: {missing[:3]}")
    if len(observed) != 48:
        raise ValueError("fresh matched source product is not exactly 48 episodes")
    return decisions


def _candidate_record(
    candidate: dict[str, Any], features: dict[str, Any], *, role: str
) -> dict[str, Any]:
    agents = sorted(map(int, candidate["agents"]))
    return {
        "role": role,
        "candidate_id": str(candidate["candidate_id"]),
        "agents": agents,
        "actual_size": len(agents),
        "selection_families": sorted(
            map(str, candidate.get("selection_families") or ())
        ),
        "features": dict(features),
    }


def _preflight_decision(
    decision: dict[str, Any], *, bundle_root: str
) -> dict[str, Any]:
    """Generate and rank all three arms without reading a target outcome."""

    source_state, source_manifest, source_trace_path = _source_target_state(decision)
    observed_trace_sha256 = sha256_file(source_trace_path)
    if (
        observed_trace_sha256 != str(source_manifest.get("trace_sha256"))
        or observed_trace_sha256 != str(decision["source_trace_sha256"])
    ):
        raise RuntimeError("fresh matched preflight source trace changed")
    before_fingerprint = state_fingerprint(source_state)
    if before_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError("fresh matched preflight source fingerprint changed")
    before_conflicts = int(source_state["num_of_colliding_pairs"])
    before_repair = repair_structure_fingerprint(source_state)
    replay = _replay_job(decision)
    restore_seed = repairability_restore_seed(before_repair)
    environment, restored = restore_repair_state(
        replay, source_state, seed=restore_seed
    )
    if repair_structure_fingerprint(restored) != before_repair:
        raise RuntimeError("fresh matched preflight restore changed repair structure")

    proposal = {**dict(replay["proposal"]), **FULL_POOL_PROPOSAL}
    for name in ("topology_boundary", "structpool", "hybridstructpool"):
        proposal.pop(name, None)
    base, generation = generate_online_candidates(
        environment,
        source_state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=proposal,
        state_hash=before_fingerprint,
        verify_full_state=False,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    engine = OnlineFeatureEngine(
        source_state,
        backend="native",
        required_features={PROFILE: PROFILE_FEATURE_NAMES[PROFILE]},
        dense_output=False,
    )
    base_rows, base_feature_metrics = engine.realized_rows(
        base, state_hash=before_fingerprint
    )
    model = load_controller_bundle(Path(bundle_root)).main_models[PROFILE]
    anchor_index, anchor_scores, anchor_margin = score_online_candidates(
        base_rows, model
    )
    anchor = dict(base[anchor_index])

    cache = TopologyAnalysisCache(source_state, backend="native")
    if cache.analysis is None:
        raise RuntimeError("fresh matched topology analysis is missing")
    structural = generate_structpool_candidate_subset(
        source_state,
        cache.analysis,
        family_sizes={
            "conflict_component": [16],
            "spatiotemporal_hotspot": [16],
        },
    )
    by_arm: dict[str, dict[str, Any] | None] = {}
    for arm, family in STRUCTURAL_FAMILIES.items():
        matches = [
            row
            for row in structural
            if family in set(map(str, row.get("selection_families") or ()))
        ]
        by_arm[arm] = dict(matches[0]) if len(matches) == 1 else None

    candidates_for_features = [
        candidate for candidate in (anchor, by_arm["component16"], by_arm["hotspot16"])
        if candidate is not None
    ]
    selected_rows, selected_feature_metrics = engine.realized_rows(
        candidates_for_features, state_hash=before_fingerprint
    )
    features = [dict(row["features"][PROFILE]) for row in selected_rows]
    feature_cursor = iter(features)
    arms: dict[str, Any] = {
        "v2_anchor": _candidate_record(
            anchor, next(feature_cursor), role="v2_anchor"
        )
    }
    for arm in ("component16", "hotspot16"):
        candidate = by_arm[arm]
        arms[arm] = (
            _candidate_record(candidate, next(feature_cursor), role=arm)
            if candidate is not None
            else None
        )

    after_environment = repair_structure_fingerprint(
        _plain(environment.get_state())
    )
    if after_environment != before_repair:
        raise RuntimeError("fresh matched candidate generation mutated repair state")
    reasons: list[str] = []
    anchor_agents = list(arms["v2_anchor"]["agents"])
    agent_count = len(source_state["agents"])
    if not anchor_agents or len(anchor_agents) != len(set(anchor_agents)) or any(
        agent < 0 or agent >= agent_count for agent in anchor_agents
    ):
        reasons.append("invalid_v2_anchor")
    for arm in ("component16", "hotspot16"):
        candidate = arms[arm]
        if candidate is None:
            reasons.append(f"{arm}_unavailable")
        elif int(candidate["actual_size"]) != 16:
            reasons.append(f"{arm}_not_size16")
    active_sets = [
        tuple(arms[arm]["agents"]) for arm in ARMS if arms.get(arm) is not None
    ]
    if len(active_sets) != 3 or len(set(active_sets)) != 3:
        reasons.append("arm_agent_sets_not_pairwise_distinct")
    if before_conflicts <= 0 or bool(source_state.get("done")):
        reasons.append("inactive_or_conflict_free")
    return {
        **decision,
        "schema": PREFLIGHT_EPISODE_SCHEMA,
        "before_conflicts": before_conflicts,
        "before_repair_fingerprint": before_repair,
        "restore_seed": restore_seed,
        "source_trace_file": str(source_manifest["trace_file"]),
        "source_trace_path": str(source_trace_path),
        "source_trace_sha256": observed_trace_sha256,
        "base_generation": _plain(generation),
        "base_feature_metrics": _plain(base_feature_metrics),
        "selected_feature_metrics": _plain(selected_feature_metrics),
        "v2_anchor_score": float(anchor_scores[anchor_index]),
        "v2_anchor_margin": float(anchor_margin),
        "v2_anchor_pool_count": len(base),
        "arms": arms,
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "repair_fingerprint_preserved": True,
        "target_outcome_fields_read": False,
        "candidate_repair_actions_executed": False,
    }


def _preflight_episode(job: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(str(job["output_path"]))
    identity = str(job["identity"])
    if bool(job["resume"]) and output_path.is_file():
        payload = _read_json(output_path)
        if (
            payload.get("schema") == PREFLIGHT_EPISODE_SCHEMA
            and payload.get("identity") == identity
            and payload.get("complete") is True
        ):
            return {
                "status": "resumed",
                "episode_key": str(job["episode_key"]),
                "output_path": str(output_path),
                "state_count": len(payload["rows"]),
            }
        raise ValueError(f"invalid fresh matched preflight artifact: {output_path}")
    rows = [
        _preflight_decision(dict(decision), bundle_root=str(job["bundle_root"]))
        for decision in job["decisions"]
    ]
    payload = {
        "schema": PREFLIGHT_EPISODE_SCHEMA,
        "identity": identity,
        "complete": True,
        "episode_key": str(job["episode_key"]),
        "rows": rows,
        "target_outcome_fields_read": False,
        "candidate_repair_actions_executed": False,
    }
    _atomic_json(output_path, payload)
    return {
        "status": "ok",
        "episode_key": str(job["episode_key"]),
        "output_path": str(output_path),
        "state_count": len(rows),
    }


def _preflight_failure_result(
    job: dict[str, Any], status: str, message: str
) -> dict[str, Any]:
    return {
        "status": status,
        "error": message,
        "job_id": str(job["job_id"]),
        "episode_key": str(job["episode_key"]),
        "output_path": str(job["output_path"]),
        "state_count": 0,
    }


def run_preflight(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output_root = Path(output).resolve()
    source_report_path = output_root / "source_collection_report.json"
    if not source_report_path.is_file():
        raise ValueError("fresh matched preflight requires the source trust report")
    source_report = _read_json(source_report_path)
    if (
        source_report.get("complete") is not True
        or source_report.get("dry_run") is not False
        or source_report.get("config_sha256") != sha256_file(path)
        or int(source_report.get("observed_episode_count", -1)) != 48
    ):
        raise ValueError("fresh matched source trust report is incompatible")
    source_manifest_hashes = {
        source_id: sha256_file(
            output_root / "source" / source_id / "realized_dynamic_manifest.jsonl"
        )
        for source_id in _source_specs(config)
    }
    if source_manifest_hashes != dict(source_report.get("source_manifest_sha256") or {}):
        raise ValueError("fresh matched source manifest changed after collection")
    decisions = load_source_preaction_rows(config, output_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        grouped[(str(decision["source_id"]), str(decision["episode_id"]))].append(
            decision
        )
    if len(grouped) != 48:
        report = {
            "schema": "lns2.stride.fresh_matched_v2_c16_h16_preflight_report.v1",
            "status": "STATE_SUPPLY_FAIL",
            "reason": "source_episode_has_no_replayable_preaction_rows",
            "source_episode_count": len(grouped),
            "expected_source_episode_count": 48,
        }
        if not dry_run:
            _atomic_json(output_root / "preflight_report.json", report)
        return report
    bundle_manifest = _registered(
        root, dict(config["controller_bundle"]["manifest"]), "fresh matched bundle"
    )
    identity = _fingerprint(
        {
            "config_sha256": sha256_file(path),
            "decisions": [
                {
                    "source_id": row["source_id"],
                    "episode_id": row["episode_id"],
                    "decision_index": row["decision_index"],
                    "before_fingerprint": row["before_fingerprint"],
                }
                for row in decisions
            ],
        }
    )
    jobs = []
    for (source_id, episode_id), episode_rows in sorted(grouped.items()):
        episode_key = f"{source_id}:{episode_id}"
        jobs.append(
            {
                "job_id": episode_key,
                "episode_key": episode_key,
                "identity": identity,
                "bundle_root": str(bundle_manifest.parent),
                "decisions": sorted(
                    episode_rows, key=lambda row: int(row["decision_index"])
                ),
                "output_path": str(
                    output_root
                    / "preflight_episodes"
                    / (hashlib.sha256(episode_key.encode("utf-8")).hexdigest() + ".json")
                ),
                "resume": bool(resume),
            }
        )
    if dry_run:
        return {
            "schema": "lns2.stride.fresh_matched_v2_c16_h16_preflight_report.v1",
            "status": "dry_run",
            "episode_job_count": len(jobs),
            "workers": 16,
            "candidate_repair_actions_executed": False,
        }
    results = _run_jobs(
        _preflight_episode,
        jobs,
        16,
        phase="fresh-matched-preflight",
        output_root=output_root / "preflight_progress",
        run_fingerprint=identity,
        timeout_seconds=420.0,
        failure_result=_preflight_failure_result,
    )
    errors = [row for row in results if str(row.get("status")) not in {"ok", "resumed"}]
    if errors:
        raise RuntimeError(f"fresh matched preflight workers failed: {errors[:2]}")
    preflight_rows = [
        row
        for result in results
        for row in _read_json(Path(str(result["output_path"]))) ["rows"]
    ]
    selected, selection = select_preflight_states(
        preflight_rows,
        states_per_episode=int(config["state_selection"]["states_per_episode"]),
    )
    selection["schema"] = "lns2.stride.fresh_matched_v2_c16_h16_preflight_report.v1"
    selection["conditional_cohort"] = "three_distinct_active_v2_c16_h16"
    selection["anchor_actual_size_distribution"] = dict(
        sorted(
            Counter(
                int(row["arms"]["v2_anchor"]["actual_size"])
                for row in preflight_rows
                if row.get("arms", {}).get("v2_anchor") is not None
            ).items()
        )
    )
    selection["candidate_repair_actions_executed"] = False
    _write_jsonl(output_root / "preflight_rows.jsonl", preflight_rows)
    _write_jsonl(output_root / "selected_states.jsonl", selected)
    selection["config_sha256"] = sha256_file(path)
    selection["source_collection_report_sha256"] = sha256_file(source_report_path)
    selection["source_manifest_sha256"] = source_manifest_hashes
    selection["preflight_rows_sha256"] = sha256_file(
        output_root / "preflight_rows.jsonl"
    )
    selection["selected_states_sha256"] = sha256_file(
        output_root / "selected_states.jsonl"
    )
    _atomic_json(output_root / "preflight_report.json", selection)
    if selection["status"] == "ok" and len(selected) != 96:
        raise RuntimeError("fresh matched Q0 did not freeze exactly 96 states")
    return selection


def _h1_trial_row_valid(row: dict[str, Any], state_occurrence_id: str) -> bool:
    arm = str(row.get("arm"))
    trial_index = int(row.get("trial_index", -1))
    if arm not in ARMS or trial_index not in TRIAL_INDICES:
        return False
    expected_seed = matched_pp_seed(state_occurrence_id, trial_index, 0)
    pp_seed = int(row.get("pp_seed", -1))
    repair_order_count = int(row.get("repair_order_count", -1))
    expected_applied = expected_seed if repair_order_count > 0 else -1
    if (
        row.get("schema") != H1_TRIAL_SCHEMA
        or row.get("state_occurrence_id") != state_occurrence_id
        or row.get("trial_identity")
        != _fingerprint(
            {
                "state_occurrence_id": state_occurrence_id,
                "arm": arm,
                "trial_index": trial_index,
            }
        )
        or pp_seed != expected_seed
        or int(row.get("requested_random_seed", -1)) != expected_seed
        or int(row.get("requested_pp_random_seed", -1)) != expected_seed
        or int(row.get("applied_pp_random_seed", -2)) != expected_applied
        or row.get("action_valid") is not True
        or row.get("generated") is not True
        or row.get("fresh_independent_environment_restore") is not True
        or row.get("integrity_ok") is not True
        or int(row.get("before_conflicts", -1)) <= 0
        or int(row.get("before_sum_of_costs", -1)) < 0
        or int(row.get("conflicts_after", -1)) < 0
        or int(row.get("after_sum_of_costs", -1)) < 0
        or not isinstance(row.get("before_repair_fingerprint"), str)
        or not row.get("before_repair_fingerprint")
        or not isinstance(row.get("after_repair_fingerprint"), str)
        or not row.get("after_repair_fingerprint")
        or type(row.get("replan_success")) is not bool
    ):
        return False
    success = bool(row.get("replan_success"))
    expected_reduction = (
        int(row["before_conflicts"]) - int(row["conflicts_after"])
    ) / max(1, int(row["before_conflicts"]))
    if not math.isclose(
        float(row.get("normalized_conflict_reduction", math.nan)),
        expected_reduction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    if success and int(row["conflicts_after"]) > int(row["before_conflicts"]):
        return False
    if not success and not (
        row.get("rollback") is True
        and row.get("atomic_rollback") is True
        and row.get("before_repair_fingerprint")
        == row.get("after_repair_fingerprint")
        and int(row["before_conflicts"]) == int(row["conflicts_after"])
        and int(row["before_sum_of_costs"]) == int(row["after_sum_of_costs"])
    ):
        return False
    is_time_limit = str(row.get("failure_reason")) == "time_limit"
    if bool(row.get("time_limit")) != (is_time_limit and not success):
        return False
    if bool(row.get("no_progress")) != (
        bool(row.get("time_limit"))
        or int(row["conflicts_after"]) >= int(row["before_conflicts"])
    ):
        return False
    return not bool(row.get("time_limit")) or (
        row.get("rollback") is True and row.get("no_progress") is True
    )


def _h1_state_artifact_valid(
    payload: dict[str, Any], *, identity: str, state_occurrence_id: str
) -> bool:
    if (
        payload.get("schema") != H1_STATE_SCHEMA
        or payload.get("identity") != identity
        or payload.get("state_occurrence_id") != state_occurrence_id
        or payload.get("complete") is not True
    ):
        return False
    trials = payload.get("trials")
    if not isinstance(trials, list) or len(trials) != len(ARMS) * len(TRIAL_INDICES):
        return False
    before_repair = payload.get("before_repair_fingerprint")
    before_conflicts = int(payload.get("before_conflicts", -1))
    before_soc = int(payload.get("before_sum_of_costs", -1))
    if (
        not isinstance(before_repair, str)
        or not before_repair
        or before_conflicts <= 0
        or before_soc < 0
    ):
        return False
    product = Counter(
        (str(row.get("arm")), int(row.get("trial_index", -1))) for row in trials
    )
    expected = {(arm, index) for arm in ARMS for index in TRIAL_INDICES}
    return (
        set(product) == expected
        and all(count == 1 for count in product.values())
        and all(_h1_trial_row_valid(dict(row), state_occurrence_id) for row in trials)
        and all(
            row.get("before_repair_fingerprint") == before_repair
            and int(row.get("before_conflicts", -1)) == before_conflicts
            and int(row.get("before_sum_of_costs", -1)) == before_soc
            for row in trials
        )
    )


def _h1_partial_artifact_valid(
    payload: dict[str, Any],
    *,
    identity: str,
    state_occurrence_id: str,
    state_row: dict[str, Any],
    before_repair_fingerprint: str,
    before_conflicts: int,
    before_sum_of_costs: int,
) -> bool:
    if (
        payload.get("schema") != H1_STATE_SCHEMA
        or payload.get("identity") != identity
        or payload.get("state_occurrence_id") != state_occurrence_id
        or payload.get("state_row") != state_row
        or payload.get("complete") is not False
        or payload.get("before_repair_fingerprint") != before_repair_fingerprint
        or int(payload.get("before_conflicts", -1)) != before_conflicts
        or int(payload.get("before_sum_of_costs", -1)) != before_sum_of_costs
    ):
        return False
    trials = payload.get("trials")
    if not isinstance(trials, list) or len(trials) > len(ARMS) * len(TRIAL_INDICES):
        return False
    product = Counter(
        (str(row.get("arm")), int(row.get("trial_index", -1))) for row in trials
    )
    expected = {(arm, index) for arm in ARMS for index in TRIAL_INDICES}
    return (
        set(product) <= expected
        and all(count == 1 for count in product.values())
        and all(_h1_trial_row_valid(dict(row), state_occurrence_id) for row in trials)
        and all(
            row.get("before_repair_fingerprint") == before_repair_fingerprint
            and int(row.get("before_conflicts", -1)) == before_conflicts
            and int(row.get("before_sum_of_costs", -1)) == before_sum_of_costs
            for row in trials
        )
    )


def _h1_state_worker(job: dict[str, Any]) -> dict[str, Any]:
    state_row = dict(job["state_row"])
    state_id = str(state_row["state_occurrence_id"])
    output_path = Path(str(job["output_path"]))
    partial_path = output_path.with_name(output_path.name + ".partial")
    identity = str(job["identity"])
    if bool(job["resume"]) and output_path.is_file():
        payload = _read_json(output_path)
        if _h1_state_artifact_valid(
            payload, identity=identity, state_occurrence_id=state_id
        ):
            return {
                "status": "resumed",
                "state_occurrence_id": state_id,
                "output_path": str(output_path),
                "trial_count": len(payload["trials"]),
            }
        raise ValueError(f"invalid fresh matched H1 artifact: {output_path}")
    if output_path.exists():
        raise ValueError(f"fresh matched H1 output exists; pass --resume: {output_path}")
    if partial_path.exists() and not bool(job["resume"]):
        raise ValueError(
            f"fresh matched H1 partial exists; pass --resume: {partial_path}"
        )

    source_state, source_manifest, source_trace_path = _source_target_state(state_row)
    observed_trace_sha256 = sha256_file(source_trace_path)
    if (
        observed_trace_sha256 != str(source_manifest.get("trace_sha256"))
        or observed_trace_sha256 != str(state_row["source_trace_sha256"])
    ):
        raise RuntimeError("fresh matched H1 source trace changed")
    if state_fingerprint(source_state) != str(state_row["before_fingerprint"]):
        raise RuntimeError("fresh matched H1 source fingerprint changed")
    before_repair = repair_structure_fingerprint(source_state)
    if before_repair != str(state_row["before_repair_fingerprint"]):
        raise RuntimeError("fresh matched H1 repair fingerprint changed")
    before_conflicts = int(source_state["num_of_colliding_pairs"])
    before_soc = int(source_state["sum_of_costs"])
    if before_conflicts <= 0:
        raise RuntimeError("fresh matched H1 state is no longer active")
    replay = _replay_job(state_row)
    restore_seed = repairability_restore_seed(before_repair)
    arms = dict(state_row["arms"])
    agent_sets = []
    for arm in ARMS:
        candidate = arms.get(arm)
        if not isinstance(candidate, dict):
            raise RuntimeError(f"fresh matched H1 arm unavailable after Q0: {arm}")
        agents = sorted(map(int, candidate["agents"]))
        if not agents or len(agents) != len(set(agents)):
            raise RuntimeError(f"fresh matched H1 arm is illegal: {arm}")
        if arm != "v2_anchor" and len(agents) != 16:
            raise RuntimeError(f"fresh matched structural arm is not size16: {arm}")
        agent_sets.append(tuple(agents))
    if len(set(agent_sets)) != 3:
        raise RuntimeError("fresh matched H1 arm sets lost pairwise distinctness")

    time_limit = float(job["per_action_time_limit_seconds"])
    partial_payload = {
        "schema": H1_STATE_SCHEMA,
        "identity": identity,
        "complete": False,
        "state_occurrence_id": state_id,
        "state_row": state_row,
        "source_trace_file": str(source_manifest["trace_file"]),
        "source_trace_path": str(source_trace_path),
        "source_trace_sha256": observed_trace_sha256,
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "before_sum_of_costs": before_soc,
        "restore_seed": restore_seed,
        "trials": [],
        "runtime_used_in_label": False,
    }
    if bool(job["resume"]) and partial_path.is_file():
        partial_payload = _read_json(partial_path)
        if not _h1_partial_artifact_valid(
            partial_payload,
            identity=identity,
            state_occurrence_id=state_id,
            state_row=state_row,
            before_repair_fingerprint=before_repair,
            before_conflicts=before_conflicts,
            before_sum_of_costs=before_soc,
        ):
            raise ValueError(f"invalid fresh matched H1 partial: {partial_path}")
    trials = list(partial_payload["trials"])
    completed = {
        (str(row["arm"]), int(row["trial_index"])) for row in trials
    }
    for arm in ARMS:
        candidate = dict(arms[arm])
        agents = sorted(map(int, candidate["agents"]))
        for trial_index in TRIAL_INDICES:
            if (arm, trial_index) in completed:
                continue
            environment, branch = restore_repair_state(
                replay, source_state, seed=restore_seed
            )
            if (
                repair_structure_fingerprint(branch) != before_repair
                or int(branch["num_of_colliding_pairs"]) != before_conflicts
                or int(branch["sum_of_costs"]) != before_soc
            ):
                raise RuntimeError("fresh matched H1 branch restore changed")
            pp_seed = matched_pp_seed(state_id, trial_index, 0)
            result = _plain(
                environment.step_with_time_limit(
                    _paired_action(agents, pp_seed), time_limit
                )
            )
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            after_repair = repair_structure_fingerprint(after)
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_soc = int(after["sum_of_costs"])
            replan_success = bool(metrics["replan_success"])
            rolled_back = bool(metrics.get("pp_rolled_back", False))
            failure_reason = str(metrics.get("pp_failure_reason", ""))
            atomic_rollback = (
                rolled_back
                and after_repair == before_repair
                and conflicts_after == before_conflicts
                and after_soc == before_soc
            )
            if not replan_success and not atomic_rollback:
                raise RuntimeError("fresh matched timed PP failure was not atomic")
            if replan_success and conflicts_after > before_conflicts:
                raise RuntimeError("fresh matched successful PP increased conflicts")
            requested_seed = metrics.get("requested_pp_random_seed")
            applied_seed = metrics.get("applied_pp_random_seed")
            repair_order = metrics.get("repair_order")
            if not isinstance(repair_order, list):
                raise RuntimeError("fresh matched timed PP omitted repair order")
            if (
                int(metrics.get("requested_random_seed", -1)) != pp_seed
                or metrics.get("action_valid") is not True
                or metrics.get("generated") is not True
            ):
                raise RuntimeError("fresh matched timed PP action identity changed")
            time_limit_rollback = (
                failure_reason == "time_limit" and not replan_success and atomic_rollback
            )
            trials.append(
                {
                    "schema": H1_TRIAL_SCHEMA,
                    "trial_identity": _fingerprint(
                        {
                            "state_occurrence_id": state_id,
                            "arm": arm,
                            "trial_index": trial_index,
                        }
                    ),
                    "state_occurrence_id": state_id,
                    "arm": arm,
                    "candidate_id": str(candidate["candidate_id"]),
                    "agents": agents,
                    "actual_size": len(agents),
                    "trial_index": trial_index,
                    "step_index": 0,
                    "restore_seed": restore_seed,
                    "fresh_independent_environment_restore": True,
                    "pp_seed": pp_seed,
                    "requested_random_seed": int(metrics["requested_random_seed"]),
                    "requested_pp_random_seed": int(requested_seed),
                    "applied_pp_random_seed": int(applied_seed),
                    "repair_order_count": len(repair_order),
                    "before_conflicts": before_conflicts,
                    "conflicts_after": conflicts_after,
                    "before_sum_of_costs": before_soc,
                    "after_sum_of_costs": after_soc,
                    "normalized_conflict_reduction": (
                        before_conflicts - conflicts_after
                    ) / max(1, before_conflicts),
                    "replan_success": replan_success,
                    "rollback": rolled_back,
                    "atomic_rollback": atomic_rollback if not replan_success else False,
                    "failure_reason": failure_reason,
                    "time_limit": time_limit_rollback,
                    "no_progress": (
                        time_limit_rollback or conflicts_after >= before_conflicts
                    ),
                    "before_repair_fingerprint": before_repair,
                    "after_repair_fingerprint": after_repair,
                    "native_step_seconds": float(
                        metrics.get("native_step_seconds", 0.0)
                    ),
                    "pp_seconds": float(
                        metrics.get(
                            "pp_replan_seconds",
                            metrics.get("native_replan_seconds", 0.0),
                        )
                    ),
                    "requested_pp_time_limit_seconds": time_limit,
                    "runtime_used_in_label": False,
                    "integrity_ok": True,
                }
            )
            partial_payload["trials"] = trials
            _atomic_json(partial_path, partial_payload)
    payload = {
        **partial_payload,
        "complete": True,
        "trials": trials,
    }
    if not _h1_state_artifact_valid(
        payload, identity=identity, state_occurrence_id=state_id
    ):
        raise RuntimeError("fresh matched H1 state artifact is invalid")
    _atomic_json(output_path, payload)
    partial_path.unlink(missing_ok=True)
    return {
        "status": "ok",
        "state_occurrence_id": state_id,
        "output_path": str(output_path),
        "trial_count": len(trials),
    }


def _h1_failure_result(
    job: dict[str, Any], status: str, message: str
) -> dict[str, Any]:
    state_id = str(job["state_row"]["state_occurrence_id"])
    return {
        "status": status,
        "error": message,
        "job_id": str(job["job_id"]),
        "state_occurrence_id": state_id,
        "output_path": str(job["output_path"]),
        "trial_count": 0,
    }


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else math.nan


def classify_h1_challenger(
    anchor_trials: Iterable[dict[str, Any]],
    challenger_trials: Iterable[dict[str, Any]],
    *,
    distinct_from_anchor: bool = True,
    tie_epsilon: float = 1e-12,
) -> dict[str, Any]:
    anchor = {int(row["trial_index"]): dict(row) for row in anchor_trials}
    challenger = {int(row["trial_index"]): dict(row) for row in challenger_trials}
    complete = set(anchor) == set(TRIAL_INDICES) == set(challenger)
    if not complete:
        return {
            "label": None,
            "label_reason": "incomplete_paired_trials",
            "complete_16_paired_trials": False,
        }
    deltas = []
    seed_matched = True
    for index in TRIAL_INDICES:
        left = anchor[index]
        right = challenger[index]
        seed_matched = seed_matched and int(left["pp_seed"]) == int(right["pp_seed"])
        deltas.append(
            float(right["normalized_conflict_reduction"])
            - float(left["normalized_conflict_reduction"])
        )
    if not distinct_from_anchor:
        return {
            "label": None,
            "label_reason": "duplicate_of_anchor",
            "complete_16_paired_trials": True,
            "paired_seed_identity": seed_matched,
        }
    strict_wins = sum(delta > tie_epsilon for delta in deltas)
    mean_delta = statistics.fmean(deltas)
    first_mean = statistics.fmean(deltas[:8])
    second_mean = statistics.fmean(deltas[8:])
    anchor_rows = [anchor[index] for index in TRIAL_INDICES]
    challenger_rows = [challenger[index] for index in TRIAL_INDICES]
    anchor_no_progress = _rate(anchor_rows, "no_progress")
    challenger_no_progress = _rate(challenger_rows, "no_progress")
    anchor_rollback = _rate(anchor_rows, "rollback")
    challenger_rollback = _rate(challenger_rows, "rollback")
    anchor_time_limit = _rate(anchor_rows, "time_limit")
    challenger_time_limit = _rate(challenger_rows, "time_limit")
    conditions = {
        "complete_16_paired_trials": complete,
        "paired_seed_identity": seed_matched,
        "minimum_12_strict_wins": strict_wins >= 12,
        "minimum_mean_delta_0_02": mean_delta >= 0.02,
        "positive_first_half_mean": first_mean > 0.0,
        "positive_second_half_mean": second_mean > 0.0,
        "no_progress_rate_noninferior": challenger_no_progress <= anchor_no_progress,
        "rollback_rate_noninferior": challenger_rollback <= anchor_rollback,
        "time_limit_rate_noninferior": challenger_time_limit <= anchor_time_limit,
    }
    return {
        "label": bool(all(conditions.values())),
        "label_reason": "frozen_h1_rule",
        **conditions,
        "strict_paired_wins": strict_wins,
        "mean_delta": mean_delta,
        "first_half_mean_delta": first_mean,
        "second_half_mean_delta": second_mean,
        "anchor_no_progress_rate": anchor_no_progress,
        "challenger_no_progress_rate": challenger_no_progress,
        "anchor_rollback_rate": anchor_rollback,
        "challenger_rollback_rate": challenger_rollback,
        "anchor_time_limit_rate": anchor_time_limit,
        "challenger_time_limit_rate": challenger_time_limit,
        "pp_seconds_report_only": {
            "anchor": sum(float(row.get("pp_seconds", 0.0)) for row in anchor_rows),
            "challenger": sum(
                float(row.get("pp_seconds", 0.0)) for row in challenger_rows
            ),
        },
    }


def analyze_h1_payloads(
    config: dict[str, Any],
    payloads: Iterable[dict[str, Any]],
    *,
    error_count: int = 0,
) -> dict[str, Any]:
    states = list(payloads)
    state_rows = []
    integrity_ok = error_count == 0
    all_trial_ids: list[str] = []
    for payload in states:
        row = dict(payload["state_row"])
        trials = list(payload.get("trials") or ())
        state_id = str(row["state_occurrence_id"])
        if not _h1_state_artifact_valid(
            payload, identity=str(payload.get("identity")), state_occurrence_id=state_id
        ):
            integrity_ok = False
        all_trial_ids.extend(str(trial.get("trial_identity")) for trial in trials)
        by_arm = {
            arm: [trial for trial in trials if str(trial.get("arm")) == arm]
            for arm in ARMS
        }
        anchor_agents = tuple(row["arms"]["v2_anchor"]["agents"])
        labels = {}
        for arm in ("component16", "hotspot16"):
            challenger_agents = tuple(row["arms"][arm]["agents"])
            labels[arm] = classify_h1_challenger(
                by_arm["v2_anchor"],
                by_arm[arm],
                distinct_from_anchor=challenger_agents != anchor_agents,
                tie_epsilon=float(config["h1_label"]["tie_epsilon"]),
            )
        state_rows.append(
            {
                "state_occurrence_id": state_id,
                "source_id": str(row["source_id"]),
                "map_id": str(row["map_id"]),
                "map_family": str(row["map_family"]),
                "task_id": str(row["task_id"]),
                "solver_seed": int(row["solver_seed"]),
                "anchor_actual_size": int(row["arms"]["v2_anchor"]["actual_size"]),
                "candidate_labels": labels,
                "opportunity": any(value.get("label") is True for value in labels.values()),
            }
        )
    if len(all_trial_ids) != len(set(all_trial_ids)):
        integrity_ok = False
    expected_trial_count = int(config["h1"]["logical_trial_count"])
    observed_trial_count = sum(len(payload.get("trials") or ()) for payload in states)
    if observed_trial_count != expected_trial_count:
        integrity_ok = False
    opportunities = [row for row in state_rows if row["opportunity"]]
    opportunity_maps = {row["map_id"] for row in opportunities}
    opportunity_families = {row["map_family"] for row in opportunities}
    by_map_opportunity = Counter(row["map_id"] for row in opportunities)
    by_map_nonopportunity = Counter(
        row["map_id"] for row in state_rows if not row["opportunity"]
    )
    fold_gates = {}
    for fold, map_ids in config["h1_gates"]["map_folds"].items():
        has_opportunity = any(
            by_map_opportunity[str(map_id)] for map_id in map_ids
        )
        has_nonopportunity = any(
            by_map_nonopportunity[str(map_id)] for map_id in map_ids
        )
        fold_gates[str(fold)] = {
            "has_opportunity": has_opportunity,
            "has_nonopportunity": has_nonopportunity,
            "positive_valid": has_opportunity,
            "negative_valid": has_nonopportunity,
        }
    labels = [
        value["label"]
        for row in state_rows
        for value in row["candidate_labels"].values()
        if value.get("label") is not None
    ]
    gates = {
        "exact_96_states": len(states) == 96,
        "exact_4608_unique_trials": (
            observed_trial_count == expected_trial_count
            and len(all_trial_ids) == len(set(all_trial_ids))
        ),
        "all_integrity_gates": integrity_ok,
        "minimum_20_opportunity_states": len(opportunities) >= 20,
        "minimum_8_opportunity_maps": len(opportunity_maps) >= 8,
        "minimum_4_opportunity_families": len(opportunity_families) >= 4,
        "each_fold_has_opportunity_and_nonopportunity": all(
            value["has_opportunity"] and value["has_nonopportunity"]
            for value in fold_gates.values()
        ),
        "valid_candidate_labels_have_both_classes": (
            any(value is True for value in labels)
            and any(value is False for value in labels)
        ),
    }
    passed = all(gates.values())
    return {
        "schema": H1_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "complete": integrity_ok and len(states) == 96,
        "passed": passed,
        "conditional_on_three_distinct_active_cohort": True,
        "state_count": len(states),
        "observed_trial_count": observed_trial_count,
        "expected_trial_count": expected_trial_count,
        "opportunity_state_count": len(opportunities),
        "opportunity_rate_conditional": len(opportunities) / 96.0,
        "opportunity_map_count": len(opportunity_maps),
        "opportunity_family_count": len(opportunity_families),
        "anchor_actual_size_distribution": dict(
            sorted(Counter(row["anchor_actual_size"] for row in state_rows).items())
        ),
        "folds": fold_gates,
        "gates": gates,
        "state_results": state_rows,
        "h8_authorized": passed,
        "training_authorized": False,
        "ttf_or_speed_claim_authorized": False,
        "runtime_or_pp_seconds_used_in_label": False,
    }


def run_h1(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    output_root = Path(output).resolve()
    selection_path = output_root / "selected_states.jsonl"
    preflight_report_path = output_root / "preflight_report.json"
    if not selection_path.is_file() or not preflight_report_path.is_file():
        raise ValueError("fresh matched H1 requires completed Q0 preflight")
    preflight_report = _read_json(preflight_report_path)
    selected = _read_jsonl(selection_path)
    source_report_path = output_root / "source_collection_report.json"
    preflight_rows_path = output_root / "preflight_rows.jsonl"
    if (
        str(preflight_report.get("status")) != "ok"
        or len(selected) != 96
        or preflight_report.get("config_sha256") != sha256_file(path)
        or preflight_report.get("selected_states_sha256")
        != sha256_file(selection_path)
        or not source_report_path.is_file()
        or preflight_report.get("source_collection_report_sha256")
        != sha256_file(source_report_path)
        or not preflight_rows_path.is_file()
        or preflight_report.get("preflight_rows_sha256")
        != sha256_file(preflight_rows_path)
    ):
        raise ValueError("fresh matched Q0 did not freeze the exact 96-state cohort")
    source_report = _read_json(source_report_path)
    current_source_hashes = {
        source_id: sha256_file(
            output_root / "source" / source_id / "realized_dynamic_manifest.jsonl"
        )
        for source_id in _source_specs(config)
    }
    if (
        source_report.get("config_sha256") != sha256_file(path)
        or source_report.get("source_manifest_sha256") != current_source_hashes
        or preflight_report.get("source_manifest_sha256") != current_source_hashes
    ):
        raise ValueError("fresh matched source/preflight trust chain changed")
    ids = [str(row["state_occurrence_id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("fresh matched selected state identities are not unique")
    identity = _fingerprint(
        {
            "config_sha256": sha256_file(path),
            "selection_sha256": sha256_file(selection_path),
            "h1": config["h1"],
        }
    )
    jobs = [
        {
            "job_id": str(row["state_occurrence_id"]),
            "state_row": row,
            "identity": identity,
            "per_action_time_limit_seconds": float(
                config["h1"]["per_action_time_limit_seconds"]
            ),
            "output_path": str(output_root / "h1_states" / f"{row['state_occurrence_id']}.json"),
            "resume": bool(resume),
        }
        for row in selected
    ]
    if dry_run:
        return {
            "schema": H1_REPORT_SCHEMA,
            "status": "dry_run",
            "state_job_count": len(jobs),
            "logical_trial_count": len(jobs) * len(ARMS) * len(TRIAL_INDICES),
            "workers": 16,
            "per_state_process_fuse_seconds": float(
                config["h1"]["per_state_process_fuse_seconds"]
            ),
            "h8_executed": False,
            "training_executed": False,
        }
    results = _run_jobs(
        _h1_state_worker,
        jobs,
        16,
        phase="fresh-matched-h1",
        output_root=output_root / "h1_progress",
        run_fingerprint=identity,
        timeout_seconds=float(config["h1"]["per_state_process_fuse_seconds"]),
        failure_result=_h1_failure_result,
    )
    successful = [
        row for row in results if str(row.get("status")) in {"ok", "resumed"}
    ]
    payloads = [_read_json(Path(str(row["output_path"]))) for row in successful]
    invalid_payload_count = sum(
        not _h1_state_artifact_valid(
            payload,
            identity=identity,
            state_occurrence_id=str(payload.get("state_occurrence_id")),
        )
        for payload in payloads
    )
    report = analyze_h1_payloads(
        config,
        payloads,
        error_count=len(results) - len(successful) + invalid_payload_count,
    )
    report["run_identity"] = identity
    report["config_sha256"] = sha256_file(path)
    report["source_collection_report_sha256"] = sha256_file(source_report_path)
    report["preflight_report_sha256"] = sha256_file(preflight_report_path)
    report["preflight_rows_sha256"] = sha256_file(preflight_rows_path)
    report["selected_states_sha256"] = sha256_file(selection_path)
    report["worker_results"] = results
    _atomic_json(output_root / "h1_report.json", report)
    _write_jsonl(
        output_root / "h1_state_results.jsonl", report["state_results"]
    )
    return report
