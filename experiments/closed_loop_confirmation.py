from __future__ import annotations

import collections
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    NATIVE_SEMANTICS_SCHEMA,
    episode_id as _episode_id,
    select_rows_by_task_id as _selected_rows,
    sha256_file as _sha256,
)
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V1,
    TRACE_FORMAT_DELTA_GZIP_V2,
    TRACE_FORMAT_FULL_V1,
    TRACE_FORMATS,
    encode_finish_event,
    encode_initial_event,
    encode_transition_event,
    open_trace_text,
    partial_trace_path,
    storage_fingerprint,
    trace_file_metadata,
    trace_suffix,
)
from experiments.compact_controller_model import (
    compact_runtime_model,
    load_controller_bundle,
)
from experiments.feature_schema_v2 import (
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    PROFILE_FEATURE_NAMES,
)
from experiments.state_analysis import (
    analyze_static_grid,
)
from experiments.neighborhood_candidates import (
    _seed_isolation,
    conflict_density,
    conflict_severity,
    no_pruning_metrics,
)
from experiments.online_feature_engine import (
    FEATURE_BACKENDS,
    OnlineFeatureEngine,
    _native_vector_function,
)
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
    POLICY_DESTROY_STRATEGIES,
    _qualification_worker,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.v3_s3 import (
    V3_S3_BUNDLE_SCHEMA,
    V3_S3_FEATURE_SCHEMA_ID,
    V3_S3_FEATURE_SCHEMA_SHA256,
    V3S3Bundle,
    load_v3_s3_bundle,
    s3_temporal_context,
)
from lns2_selector.compatibility.metrics import fixed_budget_conflict_auc
from lns2_selector.controllers.v2 import PairwiseV2Selector
from lns2_selector.controllers.v3_s3 import V3S3Selector
from lns2_selector.evaluation.trace_validation import (
    ClosedLoopTraceError,
    REPAIR_TIMING_SCHEMA,
    REPAIR_TIMING_SCHEMA_V1,
    REPAIR_TIMING_SCHEMA_V2,
    REPAIR_TIMING_SCHEMAS,
    _valid_episode_trace,
    native_repair_timing_schema as _native_repair_timing_schema,
    validate_closed_loop_trace,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.metrics import wall_clock_conflict_auc
from lns2_selector.runtime.contracts import CONTROLLER_IDS, SelectionRequest
from lns2_selector.runtime.online_selection import (
    ClosedLoopExecutionError,
    feature_range_diagnostic,
    generate_online_candidates,
    online_candidate_rows,
    pp_replay_random_seed,
    proposal_random_seed,
    proposal_random_seeds,
    repair_random_seed,
    score_online_candidates,
)
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome
from lns2_selector.solver.native import load_native_module
from lns2_selector.training.policy_bundle import (
    PortablePairwiseModel,
    export_portable_policy_bundle,
    load_frozen_policy_bundle,
    verify_portable_policy_bundle,
)


CLOSED_LOOP_SCHEMA = "lns2.closed_loop_confirmation.v1"
EPISODE_SCHEMA = EPISODE_SCHEMA_V1
FIXED_POLICIES = ("fixed_target", "fixed_collision", "fixed_random")
POLICIES = ("official_adaptive", "proposal_dynamic", "realized_dynamic")
SUPPORTED_POLICIES = ("official_adaptive", *FIXED_POLICIES, "proposal_dynamic", "realized_dynamic")
LEARNED_POLICIES = ("proposal_dynamic", "realized_dynamic")
CONTROLLER_MODES = CONTROLLER_IDS
CONTROLLER_RUNTIMES = ("reference", "optimized", "auto")
VERIFICATION_PROFILES = ("audit", "deployment")
STOPPING_RULES = ("historical", "wall-clock", "wall-clock-fixed-metric")
WALL_CLOCK_SAFETY_MAX_DECISIONS = 100_000
DEFAULT_CONTROLLER_BUNDLE = "artifacts/initlns-closed-loop-controller-v2"
DEFAULT_V3_S3_BUNDLE = (
    "build/initlns-v3-s3-mixed-load-pilot-v5-adaptive/controller"
)
CONTROLLER_IMPLEMENTATION_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/compact_controller_model.py",
    "experiments/context_audit.py",
    "experiments/feature_schema_v2.py",
    "experiments/state_analysis.py",
    "experiments/neighborhood_candidates.py",
    "experiments/neighborhood_features.py",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/v3_s3.py",
    "lns2_selector/compatibility/metrics.py",
    "lns2_selector/controllers/v2.py",
    "lns2_selector/controllers/v3_s3.py",
    "lns2_selector/runtime/contracts.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/metrics.py",
    "lns2_selector/runtime/online_selection.py",
    "lns2_selector/runtime/portable_scalar.py",
    "lns2_selector/runtime/repair_outcomes.py",
    "lns2_selector/compatibility/controller_diagnostics.py",
    "lns2_selector/evaluation/trace_validation.py",
    "lns2_selector/training/policy_bundle.py",
    "lns2_selector/solver/native.py",
    "src/python_bindings.cpp",
    "src/jsonl_observer.cpp",
    "src/online_features.cpp",
    "src/online_features.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def controller_implementation_fingerprint(project_root: Path) -> dict[str, Any]:
    files = {
        relative: _sha256(project_root / relative)
        for relative in CONTROLLER_IMPLEMENTATION_FILES
    }
    native_module = None
    try:
        lns2_env = load_native_module()

        native_path = Path(str(lns2_env.__file__)).resolve()
        semantics_schema = str(
            getattr(lns2_env, "native_semantics_schema", "")
        )
        if semantics_schema != NATIVE_SEMANTICS_SCHEMA:
            raise RuntimeError(
                "loaded lns2_env has unsupported native semantics schema: "
                f"{semantics_schema or 'missing'}"
            )
        native_module = {
            "path": native_path.name,
            "sha256": _sha256(native_path),
            "repair_timing_schema": str(
                getattr(lns2_env, "repair_timing_schema", "")
            ),
            "native_semantics_schema": semantics_schema,
        }
    except ImportError:
        pass
    return {
        "sha256": _fingerprint({"files": files, "native_module": native_module}),
        "files": files,
        "native_module": native_module,
    }


def _controller_bundle_path(
    project_root: Path, controller_bundle: str | Path | None
) -> Path:
    value = Path(str(controller_bundle or DEFAULT_CONTROLLER_BUNDLE))
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def resolve_controller_mode(
    project_root: Path,
    controller: str | None,
    controller_bundle: str | Path | None = None,
) -> tuple[str, Path, dict[str, Any] | None]:
    bundle_path = _controller_bundle_path(project_root, controller_bundle)
    loaded = None
    if (bundle_path / "controller_manifest.json").is_file():
        loaded = load_controller_bundle(bundle_path)
    if controller is None:
        mode = (
            str(loaded.manifest.get("default_controller", "official_adaptive"))
            if loaded is not None
            else "official_adaptive"
        )
    else:
        mode = str(controller)
    if mode not in CONTROLLER_MODES:
        raise ValueError(f"unsupported controller mode: {mode}")
    return mode, bundle_path, loaded.manifest if loaded is not None else None


def _number_summary(values: Iterable[float | int]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(numbers),
        "min": numbers[0],
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "max": numbers[-1],
    }


def closed_loop_dataset_design(
    rows: list[dict[str, Any]],
    split: str,
    registered: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = dict(registered or {})
    mode = str(settings.get("mode", "structured"))
    if mode == "movingai_ood":
        return movingai_ood_dataset_design(rows, split, settings)
    if mode == "balanced_wall_clock":
        errors: list[str] = []
        if any(str(row.get("split")) != split for row in rows):
            errors.append("dataset contains an unexpected split")
        task_ids = [str(row.get("task_id", "")) for row in rows]
        if not all(task_ids) or len(task_ids) != len(set(task_ids)):
            errors.append("dataset task IDs are empty or repeated")
        by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            by_map[str(row["map_id"])].append(row)
        source_counts = collections.Counter(
            str(row.get("source_group", "generated")) for row in rows
        )
        expected_sources = {
            str(name): int(count)
            for name, count in dict(settings.get("source_counts", {})).items()
        }
        if dict(sorted(source_counts.items())) != dict(sorted(expected_sources.items())):
            errors.append("dataset source counts differ from registration")
        if len(rows) != int(settings.get("instance_count", -1)) or len(by_map) != int(
            settings.get("map_count", -1)
        ):
            errors.append("dataset dimensions differ from registration")
        layout_counts = collections.Counter(str(row["layout_mode"]) for row in rows)
        expected_layouts = {
            str(name): int(count)
            for name, count in dict(settings.get("layout_counts", {})).items()
        }
        if expected_layouts and dict(sorted(layout_counts.items())) != dict(
            sorted(expected_layouts.items())
        ):
            errors.append("dataset layout counts differ from registration")
        return {
            "passed": not errors,
            "errors": errors,
            "mode": mode,
            "map_count": len(by_map),
            "task_count": len(rows),
            "source_counts": dict(sorted(source_counts.items())),
            "layout_counts": dict(sorted(layout_counts.items())),
        }
    expected_tasks = set(
        map(
            str,
            settings.get(
                "task_variants",
                ("balanced_80", "balanced_100", "bottleneck_80", "bottleneck_100"),
            ),
        )
    )
    tasks_per_map = int(settings.get("tasks_per_map", len(expected_tasks)))
    errors: list[str] = []
    if any(str(row.get("split")) != split for row in rows):
        errors.append("dataset contains a non-closed-loop split")
    by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_map[str(row["map_id"])].append(row)
    layout_counts: collections.Counter[str] = collections.Counter()
    for map_id, tasks in sorted(by_map.items()):
        layouts = {str(row.get("layout_mode")) for row in tasks}
        if len(layouts) != 1:
            errors.append(f"{map_id}: inconsistent layout")
            continue
        layout_counts[next(iter(layouts))] += 1
        if (
            {str(row.get("task_variant")) for row in tasks} != expected_tasks
            or len(tasks) != tasks_per_map
        ):
            errors.append(f"{map_id}: incomplete registered task pairing")
        if len({int(row["map_seed"]) for row in tasks}) != 1:
            errors.append(f"{map_id}: inconsistent map seed")
        if len({int(row["task_seed"]) for row in tasks}) != tasks_per_map:
            errors.append(f"{map_id}: repeated task seed")
    expected_layouts = dict(
        settings.get(
            "layout_counts",
            {
                "regular_beltway": 2,
                "compartmentalized": 2,
                "dead_end_aisles": 2,
            },
        )
    )
    expected_layouts = {str(name): int(count) for name, count in expected_layouts.items()}
    if dict(sorted(layout_counts.items())) != expected_layouts:
        errors.append("layout replication does not match the registered design")
    expected_maps = int(settings.get("map_count", sum(expected_layouts.values())))
    expected_rows = expected_maps * tasks_per_map
    if len(rows) != expected_rows or len(by_map) != expected_maps:
        errors.append("dataset dimensions do not match the registered design")
    return {
        "passed": not errors,
        "errors": errors,
        "map_count": len(by_map),
        "task_count": len(rows),
        "layout_counts": dict(sorted(layout_counts.items())),
    }


def movingai_ood_dataset_design(
    rows: list[dict[str, Any]], split: str, settings: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    if any(str(row.get("split")) != split for row in rows):
        errors.append("dataset contains a non-OOD split")
    registered_maps = {
        str(row["map_id"]): {
            "layout_family": str(row["layout_family"]),
            "agent_counts": set(map(int, row["agent_counts"])),
        }
        for row in settings.get("maps", [])
    }
    by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_map[str(row["map_id"])].append(row)
    if set(by_map) != set(registered_maps):
        errors.append("MovingAI map IDs differ from the registration")
    family_counts: collections.Counter[str] = collections.Counter()
    scenario_indices = set(map(int, settings.get("scenario_indices", [4, 5])))
    for map_id, tasks in sorted(by_map.items()):
        registered = registered_maps.get(map_id)
        if registered is None:
            continue
        layouts = {str(row.get("layout_mode")) for row in tasks}
        if layouts != {registered["layout_family"]}:
            errors.append(f"{map_id}: layout family differs from registration")
        family_counts[registered["layout_family"]] += 1
        observed = {
            (int(str(row.get("scenario_type", "")).rsplit("_", 1)[-1]), int(row["agent_count"]))
            for row in tasks
        }
        expected = {
            (scenario, agents)
            for scenario in scenario_indices
            for agents in registered["agent_counts"]
        }
        if observed != expected or len(tasks) != len(expected):
            errors.append(f"{map_id}: scenario/agent pairing differs from registration")
    expected_families = {
        str(name): int(count)
        for name, count in dict(settings.get("layout_family_counts", {})).items()
    }
    if dict(sorted(family_counts.items())) != dict(sorted(expected_families.items())):
        errors.append("MovingAI layout-family replication differs from registration")
    expected_tasks = int(settings.get("task_count", 0))
    if len(rows) != expected_tasks or len(by_map) != int(settings.get("map_count", 0)):
        errors.append("MovingAI dataset dimensions differ from registration")
    return {
        "passed": not errors,
        "errors": errors,
        "mode": "movingai_ood",
        "map_count": len(by_map),
        "task_count": len(rows),
        "layout_counts": dict(sorted(family_counts.items())),
    }


def configured_solver_seeds(config: dict[str, Any]) -> tuple[int, ...]:
    values = config.get("solver_seeds")
    seeds = (
        tuple(map(int, values))
        if values is not None
        else (int(config.get("solver_seed", 0)),)
    )
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("solver seeds must be unique non-negative integers")
    return seeds


def _qualification_reuse_fingerprint(run_config: dict[str, Any]) -> str:
    """Fingerprint the state-reset inputs that qualification depends on."""

    configuration = dict(run_config.get("configuration") or {})
    return _fingerprint(
        {
            "dataset_fingerprint": str(run_config.get("dataset_fingerprint", "")),
            "split": str(configuration.get("split", "")),
            "solver_seeds": list(configured_solver_seeds(configuration)),
            "environment": dict(configuration.get("environment") or {}),
            "seed_isolation": dict(run_config.get("seed_isolation") or {}),
            "controller_implementation": dict(
                run_config.get("controller_implementation") or {}
            ),
        }
    )


def configured_policies(config: dict[str, Any]) -> tuple[str, ...]:
    policies = tuple(map(str, config.get("policies", POLICIES)))
    if (
        not policies
        or len(policies) != len(set(policies))
        or any(policy not in SUPPORTED_POLICIES for policy in policies)
    ):
        raise ValueError("closed-loop policies are invalid or repeated")
    if "official_adaptive" not in policies or "realized_dynamic" not in policies:
        raise ValueError("closed-loop confirmation requires Adaptive and realized_dynamic")
    return policies


def closed_loop_qualification_report(
    rows: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    config: dict[str, Any],
    design: dict[str, Any],
    isolation: dict[str, Any],
    *,
    formal: bool,
    expected_job_keys: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    solver_seeds = configured_solver_seeds(config)
    available_job_keys = {
        (str(row["task_id"]), int(seed)) for row in rows for seed in solver_seeds
    }
    expected_keys = (
        {(str(task_id), int(seed)) for task_id, seed in expected_job_keys}
        if expected_job_keys is not None
        else available_job_keys
    )
    if not expected_keys or not expected_keys <= available_job_keys:
        raise ValueError("qualification expected-job cohort is empty or invalid")
    expected_solver_seeds = sorted({seed for _task_id, seed in expected_keys})
    indexed = {
        (str(row["task_id"]), int(row.get("solver_seed", solver_seeds[0]))): row
        for row in qualification
    }
    errors = [
        str(row.get("error"))
        for row in qualification
        if str(row.get("status")) != "ok"
    ]
    cohort = []
    thresholds = dict(config["severity_thresholds"])
    for source in rows:
        for solver_seed in solver_seeds:
            if (str(source["task_id"]), solver_seed) not in expected_keys:
                continue
            result = indexed.get((str(source["task_id"]), solver_seed))
            if result is None or str(result.get("status")) != "ok":
                continue
            conflicts = int(result["initial_conflicts"])
            agents = int(source["agent_count"])
            density = conflict_density(conflicts, agents)
            initial_complete = bool(result.get("initial_complete", False))
            reported_initial_feasible = bool(
                result.get("initial_feasible", False)
            )
            expected_initial_feasible = initial_complete and conflicts == 0
            initial_state_consistent = (
                reported_initial_feasible == expected_initial_feasible
            )
            initial_feasible = expected_initial_feasible
            cohort.append(
                {
                    "map_id": str(source["map_id"]),
                    "task_id": str(source["task_id"]),
                    "solver_seed": solver_seed,
                    "layout_mode": str(source["layout_mode"]),
                    "task_variant": str(source["task_variant"]),
                    "agent_count": agents,
                    "initial_conflicts": conflicts,
                    "initial_feasible": initial_feasible,
                    "reported_initial_feasible": reported_initial_feasible,
                    "initial_state_consistent": initial_state_consistent,
                    "initial_complete": initial_complete,
                    "conflict_density": density,
                    "conflict_severity": conflict_severity(density, thresholds),
                    "state_fingerprint": str(result["state_fingerprint"]),
                }
            )
    nonzero = [row for row in cohort if int(row["initial_conflicts"]) > 0]
    by_layout = collections.Counter(str(row["layout_mode"]) for row in nonzero)
    by_agent_band = collections.Counter(
        "low_mid" if int(row["agent_count"]) <= 200 else "high"
        for row in nonzero
    )
    active_maps = sorted({str(row["map_id"]) for row in nonzero})
    by_solver_seed = collections.Counter(int(row["solver_seed"]) for row in nonzero)
    fingerprints_by_seed = {
        seed: tuple(
            str(row["state_fingerprint"])
            for row in sorted(
                (item for item in cohort if int(item["solver_seed"]) == seed),
                key=lambda item: str(item["task_id"]),
            )
        )
        for seed in expected_solver_seeds
    }
    duplicate_seed_streams = [
        [left, right]
        for left, right in itertools.combinations(expected_solver_seeds, 2)
        if fingerprints_by_seed[left] == fingerprints_by_seed[right]
    ]
    incomplete_resets = [
        [str(row["task_id"]), int(row["solver_seed"])]
        for row in cohort
        if not bool(row["initial_complete"])
    ]
    inconsistent_initial_states = [
        [str(row["task_id"]), int(row["solver_seed"])]
        for row in cohort
        if not bool(row["initial_state_consistent"])
    ]
    settings = dict(config["qualification"])
    qualification_mode = str(settings.get("mode", "structured"))
    enforce_registered_thresholds = formal or bool(
        settings.get("enforce_registered_thresholds", False)
    )
    minimum_by_agent_band = {
        str(name): int(value)
        for name, value in dict(
            settings.get("minimum_nonzero_states_per_agent_band", {})
        ).items()
    }
    if (
        not set(minimum_by_agent_band) <= {"low_mid", "high"}
        or any(value < 0 for value in minimum_by_agent_band.values())
    ):
        raise ValueError(
            "qualification agent-band thresholds must be non-negative "
            "low_mid/high counts"
        )
    if formal and qualification_mode == "movingai_ood":
        required_families = set(map(str, settings["required_layout_families"]))
        active_families = {
            str(row["layout_mode"]) for row in nonzero
        }
        sample_gates = {
            "minimum_nonzero_states": len(nonzero)
            >= int(settings["minimum_nonzero_states"]),
            "minimum_active_maps": len(active_maps)
            >= int(settings["minimum_active_maps"]),
            "required_layout_families_active": required_families.issubset(active_families),
        }
    else:
        sample_gates = (
        {
            "minimum_nonzero_states": len(nonzero) >= int(settings["minimum_nonzero_states"]),
            "minimum_nonzero_per_layout": all(
                by_layout.get(layout, 0) >= int(settings["minimum_nonzero_states_per_layout"])
                for layout in ("regular_beltway", "compartmentalized", "dead_end_aisles")
            ),
            "minimum_active_maps": len(active_maps) >= int(settings["minimum_active_maps"]),
            "minimum_nonzero_per_solver_seed": all(
                by_solver_seed.get(seed, 0)
                >= int(settings.get("minimum_nonzero_states_per_solver_seed", 0))
                for seed in solver_seeds
            ),
            "minimum_nonzero_per_agent_band": all(
                by_agent_band.get(name, 0) >= minimum
                for name, minimum in minimum_by_agent_band.items()
            ),
        }
        if enforce_registered_thresholds
        else {
            "minimum_nonzero_states": bool(nonzero),
            "minimum_nonzero_per_layout": True,
            "minimum_active_maps": True,
            "minimum_nonzero_per_solver_seed": True,
            "minimum_nonzero_per_agent_band": True,
        }
        )
    gates = {
        "dataset_design": (
            bool(design["passed"]) if enforce_registered_thresholds else True
        ),
        "seed_isolation": bool(isolation["passed"]),
        "all_resets_valid": (
            len(cohort) == len(expected_keys)
            and not errors
            and not incomplete_resets
            and not inconsistent_initial_states
        ),
        "distinct_solver_seed_trajectories": not duplicate_seed_streams,
        **sample_gates,
    }
    grouped = {}
    for field in (
        "layout_mode",
        "task_variant",
        "agent_count",
        "conflict_severity",
        "solver_seed",
    ):
        groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in cohort:
            groups[str(row[field])].append(row)
        grouped[field] = {
            name: {
                "task_count": len(group),
                "initial_feasible_count": sum(
                    bool(item["initial_feasible"]) for item in group
                ),
                "conflicts": _number_summary(item["initial_conflicts"] for item in group),
                "conflict_density": _number_summary(item["conflict_density"] for item in group),
            }
            for name, group in sorted(groups.items())
        }
    passed = all(gates.values())
    return {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": formal,
        "passed": passed,
        "decision": "eligible_for_closed_loop" if passed else "inconclusive_do_not_resample",
        "gates": gates,
        "valid_count": len(cohort),
        "expected_reset_count": len(expected_keys),
        "solver_seeds": expected_solver_seeds,
        "registered_solver_seeds": list(solver_seeds),
        "initial_feasible_count": sum(
            bool(item["initial_feasible"]) for item in cohort
        ),
        "incomplete_reset_count": len(incomplete_resets),
        "incomplete_reset_job_keys": incomplete_resets,
        "inconsistent_initial_state_count": len(inconsistent_initial_states),
        "inconsistent_initial_state_job_keys": inconsistent_initial_states,
        "nonzero_state_count": len(nonzero),
        "nonzero_by_layout": dict(sorted(by_layout.items())),
        "nonzero_by_solver_seed": {
            str(seed): by_solver_seed.get(seed, 0) for seed in solver_seeds
        },
        "nonzero_by_agent_band": {
            name: by_agent_band.get(name, 0) for name in ("low_mid", "high")
        },
        "duplicate_solver_seed_trajectories": duplicate_seed_streams,
        "active_map_count": len(active_maps),
        "active_maps": active_maps,
        "repairable_task_ids": sorted({str(row["task_id"]) for row in nonzero}),
        "repairable_episode_keys": sorted(
            [str(row["task_id"]), int(row["solver_seed"])] for row in nonzero
        ),
        "zero_conflict_task_ids": sorted(
            {str(row["task_id"]) for row in cohort if int(row["initial_conflicts"]) == 0}
        ),
        "severity_thresholds": thresholds,
        "natural_distribution": {
            "conflicts": _number_summary(row["initial_conflicts"] for row in cohort),
            "conflict_density": _number_summary(row["conflict_density"] for row in cohort),
            "severity_counts": dict(
                sorted(collections.Counter(str(row["conflict_severity"]) for row in cohort).items())
            ),
            "grouped": grouped,
            "tasks": sorted(
                cohort, key=lambda row: (str(row["task_id"]), int(row["solver_seed"]))
            ),
        },
        "errors": errors,
        "dataset_design": design,
        "seed_isolation": isolation,
    }


def _emit(stream: Any, row: dict[str, Any]) -> float:
    started = time.perf_counter()
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    return time.perf_counter() - started


def _closed_loop_episode_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    policy = str(job["policy"])
    solver_seed = int(job["solver_seed"])
    episode_id = _episode_id(row, solver_seed, policy)
    output_root = Path(job["output_root"])
    trace_format = str(job.get("trace_format", TRACE_FORMAT_DELTA_GZIP_V2))
    if trace_format not in TRACE_FORMATS:
        raise ValueError(f"unsupported trace format: {trace_format}")
    storage_fp = str(job.get("storage_fingerprint", storage_fingerprint(trace_format)))
    if storage_fp != storage_fingerprint(trace_format):
        raise ValueError("trace storage fingerprint does not match the selected format")
    trace_path = (
        output_root
        / "episodes"
        / str(row["split"])
        / policy
        / f"{episode_id}{trace_suffix(trace_format)}"
    )
    partial_path = partial_trace_path(trace_path)
    relative_trace = trace_path.relative_to(output_root).as_posix()
    if job["resume"]:
        validated = _valid_episode_trace(
            trace_path,
            job["run_fingerprint"],
            expected_episode_id=episode_id,
            expected_policy=policy,
            expected_solver_seed=solver_seed,
            metric_iteration_budget=(
                int(job["metric_iteration_budget"])
                if job.get("metric_iteration_budget") is not None
                else None
            ),
        )
        if validated is not None:
            metadata = trace_file_metadata(trace_path)
            result = {
                "schema_version": SCHEMA_VERSION,
                "schema": CLOSED_LOOP_SCHEMA,
                "episode_id": episode_id,
                "split": row["split"],
                "map_id": row["map_id"],
                "task_id": row["task_id"],
                "layout_mode": row["layout_mode"],
                "task_variant": row.get("task_variant"),
                "agent_count": int(row["agent_count"]),
                "solver_seed": solver_seed,
                "policy": policy,
                "trace_file": relative_trace,
                "trace_format": trace_format,
                "storage_fingerprint": storage_fp,
                **metadata,
                "trace_event_count": int(validated["event_count"]),
                "initial_state_ref": validated.get("initial_state_ref"),
                "status": "resumed",
                "summary": validated["summary"],
                "error": None,
            }
            previous = job.get("existing_manifest_row")
            if (
                isinstance(previous, dict)
                and str(previous.get("trace_sha256"))
                == str(result.get("trace_sha256"))
                and isinstance(previous.get("episode_finalization_timings"), dict)
            ):
                result["episode_finalization_timings"] = dict(
                    previous["episode_finalization_timings"]
                )
                return result
            if not bool(job.get("require_finalization_timings", False)):
                return result
    bundle = None
    pairwise_selector: PairwiseV2Selector | None = None
    controller_mode = str(job.get("controller", "official_adaptive"))
    if controller_mode not in CONTROLLER_MODES:
        raise ValueError(f"unsupported controller mode: {controller_mode}")
    feature_backend = str(job.get("feature_backend", "auto"))
    requested_controller_runtime = str(job.get("controller_runtime", "reference"))
    if requested_controller_runtime not in CONTROLLER_RUNTIMES:
        raise ValueError(
            f"unsupported controller runtime: {requested_controller_runtime}"
        )
    verification_profile = str(job.get("verification_profile", "audit"))
    if verification_profile not in VERIFICATION_PROFILES:
        raise ValueError(f"unsupported verification profile: {verification_profile}")
    v3_s3_bundle: V3S3Bundle | None = None
    if controller_mode == "v3-s3":
        raw_v3_s3_bundle = job.get("v3_s3_bundle")
        if raw_v3_s3_bundle is None:
            raise ValueError("v3-s3 requires a frozen v3-S3 bundle")
        v3_s3_bundle = load_v3_s3_bundle(raw_v3_s3_bundle)
        if str(v3_s3_bundle.manifest.get("schema")) != V3_S3_BUNDLE_SCHEMA:
            raise ValueError("v3-s3 requires a sequence-aware controller bundle")
    runtime_models: dict[str, Any] = {}
    runtime_ranges: dict[str, dict[str, tuple[float, float]]] = {}
    shadow_models: dict[str, Any] = {}
    if policy in LEARNED_POLICIES:
        bundle = load_frozen_policy_bundle(job["frozen_models"], job["model_registration"])
        if controller_mode == "official_adaptive":
            runtime_models = bundle.models
            runtime_ranges = bundle.ranges
        else:
            if bool(job.get("feature_shadow_validation", False)):
                shadow_models = bundle.models
            controller_path = Path(str(job["controller_bundle"]))
            if (controller_path / "controller_manifest.json").is_file():
                compact_bundle = load_controller_bundle(controller_path)
                runtime_models = compact_bundle.main_models
                runtime_ranges = compact_bundle.main_ranges
            else:
                runtime_models = {
                    name: compact_runtime_model(model)
                    for name, model in bundle.models.items()
                }
                runtime_ranges = {
                    name: {
                        feature: bundle.ranges[name][feature]
                        for feature in model.base_feature_names
                    }
                    for name, model in runtime_models.items()
                }
        if controller_mode in {"v2-full", "mixed-full-v2"}:
            pairwise_selector = PairwiseV2Selector(controller_mode, runtime_models)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.unlink(missing_ok=True)
    started_wall = time.perf_counter()
    try:
        destroy_strategy = POLICY_DESTROY_STRATEGIES.get(policy, "Adaptive")
        environment_started = time.perf_counter()
        environment = _make_environment(
            job["dataset_root"], row, job["environment"], destroy_strategy
        )
        optimized_runtime_available = bool(
            callable(getattr(environment, "propose_batch_compact", None))
            and feature_backend in {"auto", "native"}
            and _native_vector_function() is not None
        )
        if (
            requested_controller_runtime == "optimized"
            and policy in LEARNED_POLICIES
            and controller_mode != "official_adaptive"
            and not optimized_runtime_available
        ):
            raise RuntimeError(
                "optimized controller runtime requires compact proposals and dense native features"
            )
        controller_runtime = (
            "optimized"
            if policy in LEARNED_POLICIES
            and controller_mode != "official_adaptive"
            and (
                requested_controller_runtime == "optimized"
                or (
                    requested_controller_runtime == "auto"
                    and optimized_runtime_available
                )
            )
            else "reference"
        )
        environment_construct_seconds = time.perf_counter() - environment_started
        initial_state_ref: str | None = None
        with open_trace_text(partial_path, "w") as stream:
            reset_started = time.perf_counter()
            state = _plain(environment.reset(seed=solver_seed))
            reset_wall_seconds = time.perf_counter() - reset_started
            reset_timing_getter = getattr(environment, "get_last_reset_timings", None)
            reset_timings = (
                _plain(reset_timing_getter())
                if callable(reset_timing_getter)
                else {"reset_total_seconds": reset_wall_seconds}
            )
            initial_fingerprint_started = time.perf_counter()
            initial_fingerprint = state_fingerprint(state)
            # The post-step fingerprint is also the next iteration's before
            # fingerprint.  Keep the exact value instead of rescanning every
            # agent path at the top of the next loop.  This changes neither the
            # fingerprint definition nor any controller/random-seed semantics.
            current_state_fingerprint = initial_fingerprint
            initial_fingerprint_seconds = (
                time.perf_counter() - initial_fingerprint_started
            )
            initial_state_elapsed_seconds = time.perf_counter() - started_wall
            conflicts = [int(state["num_of_colliding_pairs"])]
            transition_elapsed_seconds: list[float] = []
            transition_trace_write_seconds: list[float] = []
            budget_final_conflicts = conflicts[0]
            budget_final_sum_of_costs = int(state["sum_of_costs"])
            budget_final_low_level = dict(state["low_level"])
            repair_iterations_within_budget = 0
            initial_event = {
                "schema": EPISODE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": job["run_fingerprint"],
                "event": "initial",
                "episode_id": episode_id,
                "policy": policy,
                "solver_seed": solver_seed,
                "state_fingerprint": initial_fingerprint,
                "state": state,
            }
            if trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
                initial_event, initial_state_ref = encode_initial_event(
                    initial_event, state, output_root
                )
            initial_trace_write_seconds = _emit(stream, initial_event)
            controller_totals = collections.Counter()
            selected_sizes: collections.Counter[int] = collections.Counter()
            selected_families: collections.Counter[str] = collections.Counter()
            invalid_actions = 0
            fingerprint_mismatches = 0
            external_timeout = False
            total_repair_wall_seconds = 0.0
            max_decisions = int(job.get("max_decisions") or 0)
            stopping_rule = str(job.get("stopping_rule", "historical"))
            if stopping_rule not in STOPPING_RULES:
                raise ValueError(f"unsupported stopping rule: {stopping_rule}")
            safety_max_decisions = int(
                job.get("safety_max_decisions", WALL_CLOCK_SAFETY_MAX_DECISIONS)
            )
            wall_budget = float(job["wall_time_budget_seconds"])
            static_grid = (
                analyze_static_grid(state)
                if policy in LEARNED_POLICIES
                and controller_mode == "official_adaptive"
                else None
            )
            required_model_features = (
                set(runtime_models[policy].base_feature_names)
                if policy in LEARNED_POLICIES and v3_s3_bundle is None
                else set()
            )
            if v3_s3_bundle is not None and policy == "realized_dynamic":
                required_model_features.update(
                    set(v3_s3_bundle.required_feature_names)
                    & set(PROFILE_FEATURE_NAMES["realized_dynamic"])
                )
            def make_feature_engine(current_state: dict[str, Any]) -> OnlineFeatureEngine:
                return OnlineFeatureEngine(
                    current_state,
                    backend=feature_backend,
                    shadow_validation=bool(job.get("feature_shadow_validation", False)),
                    required_features={
                        policy: required_model_features,
                    },
                    dense_output=controller_runtime == "optimized",
                )

            feature_engine = (
                make_feature_engine(state)
                if policy in LEARNED_POLICIES
                and controller_mode != "official_adaptive"
                else None
            )
            pending_changed_agents: set[int] = set()
            previous_route: str | None = None
            v3_s3_selector = (
                V3S3Selector(v3_s3_bundle)
                if v3_s3_bundle is not None
                and controller_mode == "v3-s3"
                and policy == "realized_dynamic"
                else None
            )
            v3_s3_state = (
                v3_s3_selector.state if v3_s3_selector is not None else None
            )
            v3_s3_history: list[dict[str, Any]] = []
            stateful_cache: dict[str, Any] | None = None
            controller_stalled = False
            while not bool(state["done"]) and (
                max_decisions <= 0 or len(conflicts) - 1 < max_decisions
            ):
                if len(conflicts) - 1 >= safety_max_decisions:
                    raise ClosedLoopExecutionError(
                        "safety_iteration_limit",
                        "wall-clock execution reached its diagnostic safety limit",
                    )
                if time.perf_counter() - started_wall >= wall_budget:
                    external_timeout = True
                    break
                iteration_started = time.perf_counter()
                before = state
                before_fingerprint_started = time.perf_counter()
                before_hash = current_state_fingerprint
                if v3_s3_state is not None:
                    before_repair_hash = repair_structure_fingerprint(before)
                else:
                    before_repair_hash = before_hash
                before_fingerprint_seconds = (
                    time.perf_counter() - before_fingerprint_started
                )
                decision_index = len(conflicts) - 1
                controller: dict[str, Any] = {}
                route = "model" if policy in LEARNED_POLICIES else "official_adaptive"
                route_started = time.perf_counter()
                pre_step_orchestration_seconds = route_started - iteration_started
                if route == "official_adaptive":
                    action = {"mode": "official"}
                    controller_seconds_before_repair = time.perf_counter() - route_started
                    controller.update(
                        {
                            "controller_mode": controller_mode,
                            "controller_runtime": controller_runtime,
                            "verification_profile": verification_profile,
                            "route": route,
                            "route_conflicts": int(state["num_of_colliding_pairs"]),
                            "route_conflict_threshold": None,
                            "controller_seconds_before_repair": controller_seconds_before_repair,
                            "candidate_generation_seconds": 0.0,
                            "state_check_seconds": 0.0,
                            "state_check_fingerprint_seconds": 0.0,
                            "state_analysis_seconds": 0.0,
                            "proposal_feature_seconds": 0.0,
                            "realized_feature_seconds": 0.0,
                            "ranking_inference_seconds": 0.0,
                            "selection_residual_seconds": controller_seconds_before_repair,
                        }
                    )
                else:
                    proposal_started = time.perf_counter()
                    effective_proposal = dict(job["proposal"])
                    if v3_s3_state is not None:
                        generation_request = (
                            v3_s3_state.candidate_generation_request()
                        )
                        if generation_request["mode"] == "restricted":
                            effective_proposal.update(
                                {
                                    "heuristics": list(
                                        generation_request["heuristics"]
                                    ),
                                    "neighborhood_sizes": list(
                                        generation_request["neighborhood_sizes"]
                                    ),
                                    "candidates_per_family": int(
                                        generation_request[
                                            "candidates_per_family"
                                        ]
                                    ),
                                }
                            )
                    cache_key = (
                        before_repair_hash,
                        _fingerprint(effective_proposal),
                        int(solver_seed),
                    )
                    stateful_controller = v3_s3_state
                    cache_hit = bool(
                        stateful_controller is not None
                        and stateful_cache is not None
                        and stateful_cache.get("key") == cache_key
                    )
                    state_feature_metrics: dict[str, Any] = {}
                    proposal_feature_metrics = {"proposal_feature_seconds": 0.0}
                    realized_feature_metrics = {"realized_feature_seconds": 0.0}
                    proposal_rows: list[dict[str, Any]] | None = None
                    state_analysis_seconds = 0.0
                    if cache_hit:
                        assert stateful_cache is not None
                        candidates = stateful_cache["candidates"]
                        candidate_rows = stateful_cache["candidate_rows"]
                        scores = stateful_cache["scores"]
                        margin = float(stateful_cache["margin"])
                        selected_local_index = int(
                            stateful_cache["base_selected_local_index"]
                        )
                        proposal_metrics = {
                            **dict(stateful_cache["proposal_metrics"]),
                            "proposal_seconds": 0.0,
                            "candidate_generation_seconds": 0.0,
                            "state_check_seconds": 0.0,
                            "state_check_fingerprint_seconds": 0.0,
                            "backend": "v3-s3-cache",
                            "state_check_backend": "cached-state-fingerprint",
                            "full_state_verified": False,
                            "v3_s3_cache_hit": True,
                        }
                        feature_seconds = 0.0
                        inference_seconds = 0.0
                        assert stateful_controller is not None
                        stateful_controller.note_cache_hit()
                    else:
                        verification_mode = str(
                            job.get("proposal_state_verification", "always")
                        )
                        verify_full_state = verification_mode == "always" or (
                            verification_mode == "sampled"
                            and decision_index % 20 == 0
                        )
                        candidates, proposal_metrics = generate_online_candidates(
                            environment,
                            state,
                            task_id=str(row["task_id"]),
                            solver_seed=solver_seed,
                            decision_index=decision_index,
                            proposal_config=effective_proposal,
                            state_hash=before_hash,
                            verify_full_state=verify_full_state,
                            proposal_backend=controller_runtime,
                            shadow_validation=bool(
                                job.get("proposal_shadow_validation", False)
                                and optimized_runtime_available
                            ),
                        )
                        proposal_metrics["v3_s3_cache_hit"] = False
                        if controller_mode == "official_adaptive":
                            feature_started = time.perf_counter()
                            candidate_rows = online_candidate_rows(
                                state, candidates, static_grid=static_grid
                            )
                            feature_seconds = time.perf_counter() - feature_started
                        else:
                            feature_engine_created = False
                            if feature_engine is None:
                                feature_engine = make_feature_engine(state)
                                feature_engine_created = True
                            if feature_engine_created or decision_index == 0:
                                state_feature_metrics = dict(
                                    feature_engine.last_prepare_metrics
                                )
                            else:
                                state_feature_metrics = feature_engine.prepare(
                                    state,
                                    changed_agents=sorted(pending_changed_agents),
                                )
                            pending_changed_agents.clear()
                            if policy == "realized_dynamic":
                                proposal_rows = None
                            else:
                                proposal_rows, proposal_feature_metrics = (
                                    feature_engine.proposal_rows(
                                        candidates, state_hash=before_hash
                                    )
                                )
                            if policy == "realized_dynamic":
                                candidate_rows, realized_feature_metrics = (
                                    feature_engine.realized_rows(
                                        candidates, state_hash=before_hash
                                    )
                                )
                            else:
                                assert proposal_rows is not None
                                candidate_rows = list(proposal_rows)
                            state_analysis_seconds = (
                                float(
                                    state_feature_metrics.get(
                                        "state_analysis_seconds", 0.0
                                    )
                                )
                                + float(
                                    proposal_feature_metrics.get(
                                        "state_analysis_seconds", 0.0
                                    )
                                )
                                + float(
                                    realized_feature_metrics.get(
                                        "state_analysis_seconds", 0.0
                                    )
                                )
                            )
                            feature_seconds = (
                                state_analysis_seconds
                                + float(
                                    proposal_feature_metrics.get(
                                        "proposal_feature_seconds", 0.0
                                    )
                                )
                                + float(
                                    realized_feature_metrics.get(
                                        "realized_feature_seconds", 0.0
                                    )
                                )
                            )
                        if v3_s3_state is not None:
                            selected_local_index = 0
                            scores = [0.0] * len(candidate_rows)
                            margin = 0.0
                            inference_seconds = 0.0
                        elif pairwise_selector is not None:
                            inference_started = time.perf_counter()
                            selection = pairwise_selector.select(
                                SelectionRequest(
                                    candidates=candidates,
                                    candidate_rows=candidate_rows,
                                    before_fingerprint=before_repair_hash,
                                    agent_count=int(row["agent_count"]),
                                    profile=policy,
                                )
                            )
                            if selection.candidate_index is None:
                                raise ClosedLoopExecutionError(
                                    "controller_no_candidate",
                                    "pairwise controller did not select a candidate",
                                )
                            selected_local_index = int(selection.candidate_index)
                            scores = list(
                                map(float, selection.diagnostics.get("scores", []))
                            )
                            margin = float(selection.diagnostics.get("margin", 0.0))
                            inference_seconds = (
                                time.perf_counter() - inference_started
                            )
                        else:
                            inference_started = time.perf_counter()
                            selected_local_index, scores, margin = (
                                score_online_candidates(
                                    candidate_rows, runtime_models[policy]
                                )
                            )
                            inference_seconds = (
                                time.perf_counter() - inference_started
                            )
                        if stateful_controller is not None:
                            stateful_cache = {
                                "key": cache_key,
                                "candidates": candidates,
                                "candidate_rows": candidate_rows,
                                "scores": scores,
                                "margin": margin,
                                "base_selected_local_index": selected_local_index,
                                "proposal_metrics": dict(proposal_metrics),
                                "generation_decision_index": decision_index,
                            }
                    controller["proposal"] = proposal_metrics
                    pruning_metrics = no_pruning_metrics(len(candidates))
                    retained_indices = list(range(len(candidates)))
                    base_selected_local_index = selected_local_index
                    if shadow_models:
                        assert feature_engine is not None
                        shadow_rows = feature_engine.last_shadow_rows.get(policy)
                        if shadow_rows is None or len(shadow_rows) != len(candidate_rows):
                            raise ClosedLoopExecutionError(
                                "controller_shadow_mismatch",
                                "v1/v2 shadow candidate rows are incomplete",
                            )
                        shadow_index, shadow_scores, shadow_margin = (
                            score_online_candidates(shadow_rows, shadow_models[policy])
                        )
                        maximum_score_delta = max(
                            (
                                abs(float(left) - float(right))
                                for left, right in zip(scores, shadow_scores)
                            ),
                            default=0.0,
                        )

                        def ranking_order(
                            rows: list[dict[str, Any]], values: list[float]
                        ) -> list[str]:
                            return [
                                str(rows[index]["candidate_key"])
                                for index in sorted(
                                    range(len(rows)),
                                    key=lambda index: (
                                        -round(float(values[index]), 12),
                                        str(rows[index]["candidate_key"]),
                                    ),
                                )
                            ]

                        ranking_matches = ranking_order(
                            candidate_rows, scores
                        ) == ranking_order(shadow_rows, shadow_scores)
                        if (
                            selected_local_index != shadow_index
                            or not ranking_matches
                            or maximum_score_delta > 1e-12
                        ):
                            raise ClosedLoopExecutionError(
                                "controller_shadow_mismatch",
                                "v1/v2 score, ranking, or selected candidate differs",
                            )
                        controller["v1_v2_shadow"] = {
                            "passed": True,
                            "candidate_count": len(candidate_rows),
                            "maximum_score_delta": maximum_score_delta,
                            "selected_candidate_matches": True,
                            "ranking_matches": True,
                            "margin_delta": abs(float(margin) - float(shadow_margin)),
                        }
                        controller_totals["shadow_validation_count"] += 1
                        controller_totals["shadow_score_max_delta"] = max(
                            float(controller_totals["shadow_score_max_delta"]),
                            maximum_score_delta,
                        )
                    v3_s3_seconds = 0.0
                    if v3_s3_state is not None:
                        assert v3_s3_selector is not None
                        v3_s3_select_started = time.perf_counter()
                        v3_s3_selection = v3_s3_selector.select(
                            SelectionRequest(
                                candidates=candidates,
                                candidate_rows=candidate_rows,
                                profile=policy,
                                before_fingerprint=before_repair_hash,
                                temporal_context=s3_temporal_context(
                                    v3_s3_history,
                                    int(row["agent_count"]),
                                    include_wall_time=bool(
                                        v3_s3_bundle.wall_time_history_required
                                    ),
                                ),
                                agent_count=int(row["agent_count"]),
                            )
                        )
                        v3_s3_selected_index = v3_s3_selection.candidate_index
                        v3_s3_diagnostic = dict(v3_s3_selection.diagnostics)
                        v3_s3_seconds = (
                            time.perf_counter() - v3_s3_select_started
                        )
                        selected_local_index = v3_s3_selected_index
                        controller["v3_s3"] = v3_s3_diagnostic
                        controller_totals["v3_s3_seconds"] += v3_s3_seconds
                    base_diagnostic = (
                        None
                        if v3_s3_bundle is not None
                        else feature_range_diagnostic(
                            candidate_rows[base_selected_local_index],
                            policy,
                            runtime_ranges[policy],
                        )
                    )
                    if selected_local_index is None:
                        if v3_s3_state is not None:
                            controller_stalled = True
                            break
                        raise ClosedLoopExecutionError(
                            "controller_no_candidate",
                            "controller did not select a candidate",
                        )
                    else:
                        selected_index = retained_indices[selected_local_index]
                        selected = candidates[selected_index]
                        selected_row = candidate_rows[selected_local_index]
                        random_seed = repair_random_seed(
                            str(row["task_id"]),
                            solver_seed,
                            before_hash,
                            decision_index,
                            str(selected["candidate_id"]),
                            selected["proposal_seeds"],
                        )
                        action = {
                            "mode": "explicit_neighborhood",
                            "agents": selected["agents"],
                            "random_seed": random_seed,
                        }
                        diagnostic = feature_range_diagnostic(
                            selected_row, policy, runtime_ranges[policy]
                        )
                    retained_positions = {
                        global_index: local_index
                        for local_index, global_index in enumerate(retained_indices)
                    }
                    candidate_pool = []
                    for index, candidate in enumerate(candidates):
                        local_index = retained_positions.get(index)
                        candidate_pool.append(
                            {
                                **candidate,
                                "retained": local_index is not None,
                                "score": (
                                    scores[local_index]
                                    if (
                                        local_index is not None
                                        and v3_s3_bundle is None
                                    )
                                    else None
                                ),
                                "feature_out_of_range_fraction": (
                                    feature_range_diagnostic(
                                        candidate_rows[local_index],
                                        policy,
                                        runtime_ranges[policy],
                                    )["outside_fraction"]
                                    if local_index is not None
                                    else None
                                ),
                            }
                        )
                    # Keep controller test doubles and legacy proposal backends
                    # compatible with the timing-v1 schema.  Before native
                    # phased timing was added, proposal_seconds represented
                    # the complete candidate-generation stage.
                    candidate_generation_seconds = float(
                        proposal_metrics.get(
                            "candidate_generation_seconds",
                            proposal_metrics.get("proposal_seconds", 0.0),
                        )
                    )
                    state_check_seconds = float(
                        proposal_metrics.get("state_check_seconds", 0.0)
                    )
                    state_check_fingerprint_seconds = float(
                        proposal_metrics.get("state_check_fingerprint_seconds", 0.0)
                    )
                    controller_seconds_before_repair = max(
                        time.perf_counter() - proposal_started,
                        candidate_generation_seconds
                        + state_check_seconds
                        + feature_seconds
                        + float(pruning_metrics["pruner_seconds"])
                        + inference_seconds
                        + v3_s3_seconds,
                    )
                    measured_selection_stages = (
                        candidate_generation_seconds
                        + state_check_seconds
                        + feature_seconds
                        + float(pruning_metrics["pruner_seconds"])
                        + inference_seconds
                        + v3_s3_seconds
                    )
                    controller.update(
                        {
                            "controller_mode": controller_mode,
                            "controller_runtime": controller_runtime,
                            "verification_profile": verification_profile,
                            "route": route,
                            "route_conflicts": int(state["num_of_colliding_pairs"]),
                            "route_conflict_threshold": None,
                            "feature_backend": (
                                feature_engine.backend
                                if feature_engine is not None
                                else "reference-v1"
                            ),
                            "inference_backend": (
                                ",".join(v3_s3_bundle.inference_backends)
                                if v3_s3_bundle is not None
                                else getattr(
                                    runtime_models[policy],
                                    "inference_backend",
                                    (
                                        "native-portable-tree"
                                        if getattr(
                                            runtime_models[policy],
                                            "native_predictor",
                                            None,
                                        )
                                        is not None
                                        else "python-portable-tree"
                                    ),
                                )
                            ),
                            "v3_s3_inference_backends": (
                                list(v3_s3_bundle.inference_backends)
                                if v3_s3_bundle is not None
                                else None
                            ),
                            "candidate_pool": candidate_pool,
                            "pruning": pruning_metrics,
                            "feature_timings": {
                                **state_feature_metrics,
                                **proposal_feature_metrics,
                                **realized_feature_metrics,
                            },
                            "selected_candidate_id": (
                                selected["candidate_id"] if selected is not None else None
                            ),
                            "selected_score": (
                                scores[selected_local_index]
                                if (
                                    selected_local_index is not None
                                    and v3_s3_bundle is None
                                )
                                else None
                            ),
                            "base_selected_candidate_id": (
                                candidates[
                                    retained_indices[
                                        base_selected_local_index
                                    ]
                                ]["candidate_id"]
                                if v3_s3_bundle is None
                                else None
                            ),
                            "base_selected_score": (
                                scores[base_selected_local_index]
                                if v3_s3_bundle is None
                                else None
                            ),
                            "base_selected_feature_range": base_diagnostic,
                            "score_margin": (
                                margin if v3_s3_bundle is None else None
                            ),
                            "selected_feature_range": diagnostic,
                            "feature_seconds": feature_seconds,
                            "inference_seconds": inference_seconds,
                            "candidate_generation_seconds": candidate_generation_seconds,
                            "state_check_seconds": state_check_seconds,
                            "state_check_fingerprint_seconds": state_check_fingerprint_seconds,
                            "state_analysis_seconds": float(
                                state_analysis_seconds
                            ),
                            "proposal_feature_seconds": float(
                                proposal_feature_metrics.get(
                                    "proposal_feature_seconds", 0.0
                                )
                            ),
                            "realized_feature_seconds": float(
                                realized_feature_metrics.get(
                                    "realized_feature_seconds", 0.0
                                )
                            ),
                            "ranking_inference_seconds": inference_seconds,
                            "v3_s3_seconds": v3_s3_seconds,
                            "v3_s3_cache_hit": (
                                cache_hit and v3_s3_state is not None
                            ),
                            "selection_residual_seconds": max(
                                0.0,
                                controller_seconds_before_repair
                                - measured_selection_stages,
                            ),
                            "controller_seconds_before_repair": controller_seconds_before_repair,
                        }
                    )
                    if selected is not None:
                        selected_sizes[len(selected["agents"])] += 1
                        for family in selected["selection_families"]:
                            selected_families[str(family)] += 1
                    controller_totals["proposal_count"] += int(proposal_metrics["proposal_count"])
                    controller_totals["candidate_count"] += int(proposal_metrics["candidate_count"])
                    controller_totals["candidate_count_before_pruning"] += int(
                        pruning_metrics["candidate_count_before"]
                    )
                    controller_totals["candidate_count_after_pruning"] += int(
                        pruning_metrics["candidate_count_after"]
                    )
                    controller_totals["candidate_reduction_fraction_sum"] += float(
                        pruning_metrics["reduction_fraction"]
                    )
                    controller_totals["pruner_fallback_count"] += int(
                        bool(pruning_metrics["fallback"])
                    )
                    controller_totals["pruner_ood_fallback_count"] += int(
                        pruning_metrics.get("fallback_reason")
                        in {
                            "feature_out_of_range",
                            "non_finite_feature",
                            "unsupported_actual_size",
                        }
                    )
                    controller_totals["pruner_seconds"] += float(
                        pruning_metrics["pruner_seconds"]
                    )
                    controller_totals["proposal_seconds"] += float(proposal_metrics["proposal_seconds"])
                    controller_totals["candidate_generation_seconds"] += (
                        candidate_generation_seconds
                    )
                    controller_totals["state_check_seconds"] += state_check_seconds
                    controller_totals[f"proposal_backend={proposal_metrics['backend']}"] += 1
                    controller_totals[
                        f"state_check_backend={proposal_metrics['state_check_backend']}"
                    ] += 1
                    controller_totals["full_state_verification_count"] += int(
                        bool(proposal_metrics["full_state_verified"])
                    )
                    controller_totals["feature_seconds"] += feature_seconds
                    controller_totals["inference_seconds"] += inference_seconds
                    controller_totals["controller_seconds_before_repair"] += (
                        controller_seconds_before_repair
                    )
                    for metrics in (
                        state_feature_metrics,
                        proposal_feature_metrics,
                        realized_feature_metrics,
                    ):
                        for key, value in metrics.items():
                            if key.endswith("_seconds"):
                                controller_totals[key] += float(value)
                    if diagnostic is not None:
                        controller_totals[
                            "selected_feature_outside_fraction_sum"
                        ] += float(diagnostic["outside_fraction"])
                        controller_totals["selected_feature_diagnostic_count"] += 1
                    controller_totals["learned_decisions"] += 1
                # Candidate generation and controller inference can consume the
                # remaining wall budget.  Do not enter the native solver after
                # its live deadline has expired: the environment correctly
                # rejects that call, but an expected timeout must not turn the
                # whole episode into an execution error.  Keep this before
                # route counters so only executed repairs are counted.
                if time.perf_counter() - started_wall >= wall_budget:
                    external_timeout = True
                    break
                if bool(job.get("deterministic_pp_replay", False)):
                    # Pair the low-level PP stream across controller routes.
                    # Official neighborhood generation still consumes its
                    # upstream RNG stream before PP is reseeded; explicit
                    # learned actions differ only in the selected agent set.
                    action["pp_random_seed"] = pp_replay_random_seed(
                        str(row["task_id"]),
                        solver_seed,
                        before_hash,
                        decision_index,
                        route,
                    )
                if time.perf_counter() - started_wall >= wall_budget:
                    external_timeout = True
                    break
                repair_started = time.perf_counter()
                try:
                    result = _plain(environment.step(action))
                except RuntimeError as error:
                    elapsed_after_error = time.perf_counter() - started_wall
                    if (
                        "repair episode" in str(error)
                        and "finished" in str(error)
                        and elapsed_after_error >= wall_budget
                    ):
                        external_timeout = True
                        break
                    raise
                repair_wall_seconds = time.perf_counter() - repair_started
                post_step_started = time.perf_counter()
                total_repair_wall_seconds += repair_wall_seconds
                state = result["observation"]
                metrics = result["metrics"]
                if policy == "realized_dynamic":
                    route_prefix = (
                        "official" if route == "official_adaptive" else "model"
                    )
                    controller_totals[f"{route_prefix}_decision_count"] += 1
                    controller_totals[
                        f"{route_prefix}_controller_seconds"
                    ] += float(
                        controller.get("controller_seconds_before_repair", 0.0)
                    )
                    if (
                        route == "official_adaptive"
                        and "candidate_pool" not in controller
                    ):
                        controller_totals[
                            "controller_seconds_before_repair"
                        ] += float(
                            controller.get(
                                "controller_seconds_before_repair", 0.0
                            )
                        )
                    if previous_route is not None and previous_route != route:
                        controller_totals["route_switch_count"] += 1
                    previous_route = route
                if "pp_random_seed" in action:
                    requested_pp_seed = int(action["pp_random_seed"])
                    if int(metrics.get("requested_pp_random_seed", -1)) != requested_pp_seed:
                        raise ClosedLoopExecutionError(
                            "pp_seed_mismatch",
                            "native transition did not retain the requested PP seed",
                        )
                    if metrics.get("repair_order") and int(
                        metrics.get("applied_pp_random_seed", -1)
                    ) != requested_pp_seed:
                        raise ClosedLoopExecutionError(
                            "pp_seed_not_applied",
                            "native PP did not apply the deterministic replay seed",
                        )
                try:
                    native_timing_schema = _native_repair_timing_schema(metrics)
                except (TypeError, ValueError) as error:
                    raise ClosedLoopExecutionError(
                        "invalid_native_timing", str(error)
                    ) from error
                low_level_delta = _low_level_delta(before, state)
                actual = sorted(map(int, metrics.get("neighborhood", [])))
                if policy in LEARNED_POLICIES and route == "model":
                    if not bool(metrics.get("action_valid")) or actual != sorted(action["agents"]):
                        invalid_actions += 1
                        raise ClosedLoopExecutionError(
                            "invalid_action",
                            "explicit closed-loop action was rejected or changed",
                        )
                elif not bool(metrics.get("action_valid", True)):
                    invalid_actions += 1
                    raise ClosedLoopExecutionError(
                        "invalid_action", "official closed-loop action was rejected"
                    )
                pending_changed_agents.update(actual)
                if policy == "realized_dynamic":
                    route_prefix = "official" if route == "official_adaptive" else "model"
                    route_controller_seconds = float(
                        controller.get("controller_seconds_before_repair", 0.0)
                    )
                    controller["repair_wall_seconds"] = repair_wall_seconds
                    controller["total_decision_seconds"] = (
                        route_controller_seconds + repair_wall_seconds
                    )
                    controller_totals[f"{route_prefix}_repair_seconds"] += repair_wall_seconds
                    controller_totals[f"{route_prefix}_total_decision_seconds"] += (
                        route_controller_seconds + repair_wall_seconds
                    )
                conflicts.append(int(state["num_of_colliding_pairs"]))
                elapsed_wall = time.perf_counter() - started_wall
                transition_elapsed_seconds.append(elapsed_wall)
                within_wall_budget = elapsed_wall <= wall_budget
                if within_wall_budget:
                    repair_iterations_within_budget += 1
                    budget_final_conflicts = conflicts[-1]
                    budget_final_sum_of_costs = int(state["sum_of_costs"])
                    budget_final_low_level = dict(state["low_level"])

                controller_before_repair_seconds = float(
                    controller.get("controller_seconds_before_repair", 0.0)
                )
                native_step_seconds = float(
                    metrics.get(
                        "native_step_seconds",
                        metrics.get("step_runtime", 0.0),
                    )
                )
                episode_runtime_delta_seconds = float(
                    metrics.get(
                        "episode_runtime_delta_seconds",
                        metrics.get("step_runtime", native_step_seconds),
                    )
                )
                native_neighborhood_seconds = float(
                    metrics.get("native_neighborhood_generation_seconds", 0.0)
                )
                pp_replan_seconds = float(metrics.get("pp_replan_seconds", 0.0))
                native_bookkeeping_seconds = float(
                    metrics.get("native_repair_bookkeeping_seconds", 0.0)
                )
                native_residual_seconds = float(
                    metrics.get("native_residual_seconds", 0.0)
                )
                native_state_snapshot_seconds = float(
                    metrics.get("native_state_snapshot_seconds", 0.0)
                )
                binding_state_snapshot_seconds = float(
                    metrics.get("binding_state_snapshot_seconds", 0.0)
                )
                state_to_python_seconds = float(
                    metrics.get("state_to_python_seconds", 0.0)
                )
                state_export_seconds = (
                    native_state_snapshot_seconds
                    + binding_state_snapshot_seconds
                    + state_to_python_seconds
                )
                neighborhood_selection_seconds = (
                    controller_before_repair_seconds + native_neighborhood_seconds
                )
                environment_step_residual_seconds = max(
                    0.0,
                    repair_wall_seconds
                    - native_neighborhood_seconds
                    - pp_replan_seconds
                    - native_bookkeeping_seconds
                    - state_export_seconds,
                )
                controller["neighborhood_selection_seconds"] = (
                    neighborhood_selection_seconds
                )
                controller_totals["neighborhood_selection_seconds"] += (
                    neighborhood_selection_seconds
                )
                controller_totals["pp_replan_seconds"] += pp_replan_seconds
                controller_totals["repair_bookkeeping_seconds"] += (
                    native_bookkeeping_seconds
                )
                controller_totals["state_export_seconds"] += state_export_seconds
                controller_totals["environment_step_residual_seconds"] += (
                    environment_step_residual_seconds
                )
                if route == "official_adaptive":
                    controller_totals["controller_seconds_before_repair"] += (
                        controller_before_repair_seconds
                    )

                after_fingerprint_started = time.perf_counter()
                after_hash = state_fingerprint(state)
                current_state_fingerprint = after_hash
                if v3_s3_state is not None:
                    after_repair_hash = repair_structure_fingerprint(state)
                else:
                    after_repair_hash = after_hash
                state_fingerprint_seconds = before_fingerprint_seconds + (
                    time.perf_counter() - after_fingerprint_started
                ) + float(controller.get("state_check_fingerprint_seconds", 0.0))
                if v3_s3_state is not None:
                    v3_s3_observe_started = time.perf_counter()
                    repair_outcome = classify_repair_outcome(
                        before_fingerprint=before_repair_hash,
                        after_fingerprint=after_repair_hash,
                        replan_success=bool(metrics.get("replan_success")),
                        conflicts_before=int(before["num_of_colliding_pairs"]),
                        conflicts_after=int(state["num_of_colliding_pairs"]),
                        feasible=bool(state.get("feasible")),
                    )
                    conflict_reduction = max(
                        0,
                        int(before["num_of_colliding_pairs"])
                        - int(state["num_of_colliding_pairs"]),
                    )
                    continuation_expected = v3_s3_state.observe(
                        before_fingerprint=before_repair_hash,
                        after_fingerprint=after_repair_hash,
                        repair_outcome=repair_outcome,
                        conflict_reduction=float(conflict_reduction),
                        total_seconds=float(
                            controller_before_repair_seconds
                            + repair_wall_seconds
                        ),
                        feasible=bool(state.get("feasible")),
                    )
                    v3_s3_history.append(
                        {
                            "conflict_reduction": float(conflict_reduction),
                            "repair_seconds": float(repair_wall_seconds),
                            "state_changed": (
                                before_repair_hash != after_repair_hash
                            ),
                            "no_progress": repair_outcome
                            in {"hard_failure", "accepted_noop"},
                            "neighborhood_size": len(actual),
                        }
                    )
                    observe_seconds = (
                        time.perf_counter() - v3_s3_observe_started
                    )
                    controller["v3_s3"] = {
                        **dict(controller.get("v3_s3") or {}),
                        "repair_outcome": repair_outcome,
                        "continuation_expected": bool(
                            continuation_expected
                        ),
                        "state_unchanged": (
                            before_repair_hash == after_repair_hash
                        ),
                    }
                    controller["v3_s3_seconds"] = float(
                        controller.get("v3_s3_seconds", 0.0)
                    ) + observe_seconds
                    controller_totals["v3_s3_seconds"] += observe_seconds
                transition_timings = {
                    "native_step_seconds": native_step_seconds,
                    "episode_runtime_delta_seconds": (
                        episode_runtime_delta_seconds
                    ),
                    "pre_step_orchestration_seconds": pre_step_orchestration_seconds,
                    "controller_before_repair_seconds": controller_before_repair_seconds,
                    "candidate_generation_seconds": float(
                        controller.get("candidate_generation_seconds", 0.0)
                    ),
                    "state_check_seconds": float(
                        controller.get("state_check_seconds", 0.0)
                    ),
                    "state_check_fingerprint_seconds": float(
                        controller.get("state_check_fingerprint_seconds", 0.0)
                    ),
                    "state_analysis_seconds": float(
                        controller.get("state_analysis_seconds", 0.0)
                    ),
                    "proposal_feature_seconds": float(
                        controller.get("proposal_feature_seconds", 0.0)
                    ),
                    "realized_feature_seconds": float(
                        controller.get("realized_feature_seconds", 0.0)
                    ),
                    "ranking_inference_seconds": float(
                        controller.get("ranking_inference_seconds", 0.0)
                    ),
                    "v3_s3_seconds": float(
                        controller.get("v3_s3_seconds", 0.0)
                    ),
                    "selection_residual_seconds": float(
                        controller.get("selection_residual_seconds", 0.0)
                    ),
                    "native_neighborhood_generation_seconds": native_neighborhood_seconds,
                    "neighborhood_selection_seconds": neighborhood_selection_seconds,
                    "pp_replan_seconds": pp_replan_seconds,
                    "repair_bookkeeping_seconds": native_bookkeeping_seconds,
                    "native_residual_seconds": native_residual_seconds,
                    "state_export_seconds": state_export_seconds,
                    "environment_step_residual_seconds": environment_step_residual_seconds,
                    "environment_step_wall_seconds": repair_wall_seconds,
                    "state_fingerprint_seconds": state_fingerprint_seconds,
                    "post_step_orchestration_seconds": 0.0,
                    "iteration_wall_seconds": 0.0,
                }
                transition = {
                    "schema": EPISODE_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "run_fingerprint": job["run_fingerprint"],
                    "event": "transition",
                    "episode_id": episode_id,
                    "decision_index": decision_index,
                    "action": action,
                    "before_fingerprint": before_hash,
                    "after_fingerprint": after_hash,
                    "metrics": metrics,
                    "low_level_delta": low_level_delta,
                    "repair_wall_seconds": repair_wall_seconds,
                    "elapsed_wall_seconds": elapsed_wall,
                    "within_wall_budget": within_wall_budget,
                    "native_timing_schema": native_timing_schema,
                    "timings": transition_timings,
                    "controller": controller,
                    "terminated": bool(result["terminated"]),
                    "truncated": bool(result["truncated"]),
                    "after": state,
                }
                transition_timings["post_step_orchestration_seconds"] = (
                    time.perf_counter() - post_step_started
                )
                transition_timings["iteration_wall_seconds"] = (
                    time.perf_counter() - iteration_started
                )
                controller_totals["pre_step_orchestration_seconds"] += float(
                    transition_timings["pre_step_orchestration_seconds"]
                )
                controller_totals["post_step_orchestration_seconds"] += float(
                    transition_timings["post_step_orchestration_seconds"]
                )
                controller_totals["iteration_wall_seconds"] += float(
                    transition_timings["iteration_wall_seconds"]
                )
                controller_totals["state_fingerprint_seconds"] += float(
                    transition_timings["state_fingerprint_seconds"]
                )
                if trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
                    transition = encode_transition_event(transition, before, state)
                trace_write_seconds = _emit(stream, transition)
                transition_trace_write_seconds.append(trace_write_seconds)
                controller_totals["trace_write_seconds"] += trace_write_seconds
                if elapsed_wall >= wall_budget and not bool(state["done"]):
                    external_timeout = True
                    break
            elapsed_wall = time.perf_counter() - started_wall
            episode_finalize_started = time.perf_counter()
            algorithm_elapsed = (
                transition_elapsed_seconds[-1]
                if transition_elapsed_seconds
                else initial_state_elapsed_seconds
            )
            feasible_elapsed = algorithm_elapsed if bool(state["feasible"]) else None
            success = feasible_elapsed is not None and feasible_elapsed <= wall_budget
            if not success and algorithm_elapsed >= wall_budget:
                external_timeout = True
            truncated = not success
            repair_limit_reached = (
                max_decisions > 0
                and len(conflicts) - 1 >= max_decisions
                and not success
            )
            metric_iteration_budget = job.get("metric_iteration_budget")
            fixed_auc = (
                fixed_budget_conflict_auc(
                    conflicts,
                    int(metric_iteration_budget),
                    success=success,
                )
                if metric_iteration_budget is not None
                else None
            )
            normalized_fixed_auc = (
                fixed_auc / (float(conflicts[0]) * int(metric_iteration_budget))
                if fixed_auc is not None
                and conflicts[0] > 0
                and metric_iteration_budget is not None
                else None
            )
            wall_auc = wall_clock_conflict_auc(
                conflicts, transition_elapsed_seconds, wall_budget
            )
            normalized_wall_auc = (
                wall_auc / (float(conflicts[0]) * wall_budget)
                if conflicts[0] > 0
                else None
            )
            if success:
                stop_reason = "success"
            elif controller_stalled:
                stop_reason = "controller_stalled"
            elif repair_limit_reached:
                stop_reason = "repair_limit"
            elif external_timeout or bool(state["done"]):
                stop_reason = "wall_timeout"
                external_timeout = True
            else:
                stop_reason = "truncated"
            model_decisions = int(controller_totals["model_decision_count"])
            official_decisions = int(controller_totals["official_decision_count"])
            if (
                policy == "realized_dynamic"
                and model_decisions + official_decisions != len(conflicts) - 1
            ):
                raise ClosedLoopExecutionError(
                    "route_accounting_mismatch",
                    "controller route counts do not equal repair iterations",
                )
            summary = {
                "initial_fingerprint": initial_fingerprint,
                "initial_conflicts": conflicts[0],
                "final_conflicts": conflicts[-1],
                "budget_final_conflicts": budget_final_conflicts,
                "repairable": conflicts[0] > 0,
                "success": success,
                "truncated": truncated,
                "external_timeout": external_timeout,
                "stop_reason": stop_reason,
                "stopping_rule": stopping_rule,
                "wall_time_budget_seconds": wall_budget,
                "repair_iterations": len(conflicts) - 1,
                "repair_iterations_within_budget": repair_iterations_within_budget,
                "conflict_trajectory": conflicts,
                "transition_elapsed_seconds": transition_elapsed_seconds,
                "conflict_auc": sum(
                    (conflicts[index] + conflicts[index + 1]) / 2.0
                    for index in range(len(conflicts) - 1)
                ),
                "fixed_budget_conflict_auc": fixed_auc,
                "normalized_fixed_budget_conflict_auc": normalized_fixed_auc,
                "metric_iteration_budget": metric_iteration_budget,
                "wall_clock_conflict_auc": wall_auc,
                "normalized_wall_clock_conflict_auc": normalized_wall_auc,
                "wall_time_to_feasible": feasible_elapsed if success else None,
                "capped_wall_time_to_feasible": min(feasible_elapsed, wall_budget)
                if success
                else wall_budget,
                "budget_overshoot_seconds": max(0.0, algorithm_elapsed - wall_budget),
                "native_time_to_feasible": float(state["runtime"]) if success else None,
                "repair_wall_seconds": total_repair_wall_seconds,
                "environment_construct_seconds": environment_construct_seconds,
                "reset_wall_seconds": reset_wall_seconds,
                "reset_timings": reset_timings,
                "initial_state_elapsed_seconds": initial_state_elapsed_seconds,
                "initial_fingerprint_seconds": initial_fingerprint_seconds,
                "initial_trace_write_seconds": initial_trace_write_seconds,
                "transition_trace_write_seconds": transition_trace_write_seconds,
                "trace_write_seconds": initial_trace_write_seconds
                + sum(transition_trace_write_seconds),
                "episode_observed_wall_seconds": elapsed_wall,
                "timing_unaccounted_seconds": max(
                    0.0,
                    elapsed_wall
                    - environment_construct_seconds
                    - reset_wall_seconds
                    - initial_trace_write_seconds
                    - sum(transition_trace_write_seconds)
                    - float(controller_totals["iteration_wall_seconds"]),
                ),
                "controller_mode": controller_mode,
                "controller_runtime": controller_runtime,
                "requested_controller_runtime": requested_controller_runtime,
                "verification_profile": verification_profile,
                "feature_backend": (
                    feature_engine.backend
                    if feature_engine is not None
                    else "reference-v1"
                    if policy in LEARNED_POLICIES
                    and controller_mode == "official_adaptive"
                    else "not_used"
                    if policy in LEARNED_POLICIES
                    else None
                ),
                "controller_totals": dict(controller_totals),
                "model_decision_count": model_decisions,
                "official_decision_count": official_decisions,
                "model_route_fraction": (
                    model_decisions / (model_decisions + official_decisions)
                    if model_decisions + official_decisions
                    else 0.0
                ),
                "route_switch_count": int(controller_totals["route_switch_count"]),
                "balanced_controller": None,
                "v3_s3": (
                    {
                        **v3_s3_state.summary(),
                        "combined_runtime_feature_count": len(
                            required_model_features
                        ),
                        "history_length": len(v3_s3_history),
                    }
                    if v3_s3_state is not None
                    and policy == "realized_dynamic"
                    else None
                ),
                "candidate_reduction_fraction": (
                    1.0
                    - float(controller_totals["candidate_count_after_pruning"])
                    / float(controller_totals["candidate_count_before_pruning"])
                    if controller_totals["candidate_count_before_pruning"]
                    else 0.0
                ),
                "pruner_fallback_fraction": (
                    float(controller_totals["pruner_fallback_count"])
                    / float(controller_totals["learned_decisions"])
                    if controller_totals["learned_decisions"]
                    else 0.0
                ),
                "pruner_ood_fallback_fraction": (
                    float(controller_totals["pruner_ood_fallback_count"])
                    / float(controller_totals["learned_decisions"])
                    if controller_totals["learned_decisions"]
                    else 0.0
                ),
                "mean_selected_feature_outside_fraction": (
                    float(controller_totals["selected_feature_outside_fraction_sum"])
                    / float(controller_totals["selected_feature_diagnostic_count"])
                    if controller_totals["selected_feature_diagnostic_count"]
                    else 0.0
                ),
                "selected_size_counts": {
                    str(key): value for key, value in sorted(selected_sizes.items())
                },
                "selected_family_counts": dict(sorted(selected_families.items())),
                "invalid_action_count": invalid_actions,
                "fingerprint_mismatch_count": fingerprint_mismatches,
                "final_sum_of_costs": int(state["sum_of_costs"]),
                "budget_final_sum_of_costs": budget_final_sum_of_costs,
                "final_low_level": state["low_level"],
                "budget_final_low_level": budget_final_low_level,
            }
            final_fingerprint_started = time.perf_counter()
            final_fingerprint = current_state_fingerprint
            final_fingerprint_seconds = time.perf_counter() - final_fingerprint_started
            summary["final_fingerprint_seconds"] = final_fingerprint_seconds
            finish_event = {
                "schema": EPISODE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": job["run_fingerprint"],
                "event": "finish",
                "episode_id": episode_id,
                "policy": policy,
                "success": success,
                "final_fingerprint": final_fingerprint,
                "summary": summary,
            }
            if trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
                finish_event = encode_finish_event(finish_event)
            finish_event_orchestration_seconds = (
                time.perf_counter() - episode_finalize_started
            )
            finish_trace_write_seconds = _emit(stream, finish_event)
            trace_close_started = time.perf_counter()
        trace_close_seconds = time.perf_counter() - trace_close_started
        validation_started = time.perf_counter()
        validated = validate_closed_loop_trace(
            partial_path,
            job["run_fingerprint"],
            expected_episode_id=episode_id,
            expected_policy=policy,
            expected_solver_seed=solver_seed,
            metric_iteration_budget=(
                int(job["metric_iteration_budget"])
                if job.get("metric_iteration_budget") is not None
                else None
            ),
            collection_root=output_root,
        )
        trace_validation_seconds = time.perf_counter() - validation_started
        if validated["summary"] != summary:
            raise ClosedLoopTraceError("new trace summary mismatch")
        rename_started = time.perf_counter()
        os.replace(partial_path, trace_path)
        atomic_rename_seconds = time.perf_counter() - rename_started
        metadata_started = time.perf_counter()
        metadata = trace_file_metadata(trace_path)
        trace_metadata_seconds = time.perf_counter() - metadata_started
        finalized_at = time.perf_counter()
        episode_process_wall_seconds = finalized_at - started_wall
        episode_finalization_timings = {
            "finish_event_orchestration_seconds": finish_event_orchestration_seconds,
            "finish_trace_write_seconds": finish_trace_write_seconds,
            "trace_close_seconds": trace_close_seconds,
            "trace_validation_seconds": trace_validation_seconds,
            "atomic_rename_seconds": atomic_rename_seconds,
            "trace_metadata_seconds": trace_metadata_seconds,
            "post_algorithm_finalize_seconds": finalized_at
            - episode_finalize_started,
            "episode_process_wall_seconds": episode_process_wall_seconds,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": CLOSED_LOOP_SCHEMA,
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_file": relative_trace,
            "trace_format": trace_format,
            "storage_fingerprint": storage_fp,
            **metadata,
            "trace_event_count": int(validated["event_count"]),
            "initial_state_ref": initial_state_ref,
            "status": "ok",
            "summary": summary,
            "episode_finalization_timings": episode_finalization_timings,
            "error": None,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": CLOSED_LOOP_SCHEMA,
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row.get("layout_mode"),
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_format": trace_format,
            "storage_fingerprint": storage_fp,
            "trace_file": None,
            "partial_trace_file": partial_path.relative_to(output_root).as_posix()
            if partial_path.is_file()
            else None,
            "status": "error",
            "summary": None,
            "error_kind": getattr(error, "kind", type(error).__name__),
            "error": f"{type(error).__name__}: {error}",
        }


def _with_time_budget_overrides(
    config: dict[str, Any],
    wall_time_budget_seconds: float | None,
    episode_process_timeout_seconds: float | None,
    environment_time_limit_seconds: float | None = None,
) -> dict[str, Any]:
    result = {**config, "environment": dict(config["environment"])}
    if wall_time_budget_seconds is None:
        if (
            episode_process_timeout_seconds is not None
            or environment_time_limit_seconds is not None
        ):
            raise ValueError(
                "time-limit overrides require a wall-time budget override"
            )
        return result
    wall_budget = float(wall_time_budget_seconds)
    if not math.isfinite(wall_budget) or wall_budget <= 0.0:
        raise ValueError("wall-time budget override must be finite and positive")
    environment_limit = float(
        environment_time_limit_seconds
        if environment_time_limit_seconds is not None
        else wall_budget
    )
    if not math.isfinite(environment_limit) or environment_limit <= 0.0:
        raise ValueError(
            "environment time-limit override must be finite and positive"
        )
    process_timeout = float(
        episode_process_timeout_seconds
        if episode_process_timeout_seconds is not None
        else max(wall_budget, environment_limit) + 60.0
    )
    if (
        not math.isfinite(process_timeout)
        or process_timeout <= max(wall_budget, environment_limit)
    ):
        raise ValueError(
            "episode process timeout must be finite and greater than both the "
            "wall-time budget and environment time limit"
        )
    result["wall_time_budget_seconds"] = wall_budget
    result["episode_process_timeout_seconds"] = process_timeout
    result["environment"]["time_limit"] = environment_limit
    return result


def _with_stopping_rule(
    config: dict[str, Any], stopping_rule: str
) -> dict[str, Any]:
    if stopping_rule not in STOPPING_RULES:
        raise ValueError(f"unsupported stopping rule: {stopping_rule}")
    result = {**config, "environment": dict(config["environment"])}
    result["stopping_rule"] = stopping_rule
    if stopping_rule in {"wall-clock", "wall-clock-fixed-metric"}:
        result["max_decisions"] = 0
        result["environment"]["max_repair_iterations"] = 0
    if stopping_rule == "wall-clock":
        result["metric_iteration_budget"] = None
    elif stopping_rule == "wall-clock-fixed-metric":
        metric_budget = result.get("metric_iteration_budget")
        if type(metric_budget) is not int or metric_budget <= 0:
            raise ValueError(
                "wall-clock-fixed-metric requires a positive integer "
                "metric_iteration_budget"
            )
        result["deterministic_pp_replay"] = True
    return result


def _collection_policy_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "episode_count": len(results),
        "success_count": sum(
            bool(dict(row.get("summary") or {}).get("success")) for row in results
        ),
        "error_count": sum(
            str(row.get("status")) not in {"ok", "resumed"} for row in results
        ),
        "timeout_count": sum(str(row.get("status")) == "timeout" for row in results),
    }


def run_closed_loop_collection(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    task_ids: list[str] | None = None,
    trace_format: str = TRACE_FORMAT_DELTA_GZIP_V2,
    controller: str | None = None,
    feature_backend: str = "auto",
    controller_bundle: str | Path | None = None,
    feature_shadow_validation: bool = False,
    controller_runtime: str = "reference",
    verification_profile: str = "audit",
    v3_s3_bundle: str | Path | None = None,
    job_keys: set[tuple[str, int]] | None = None,
    cohort_job_keys: set[tuple[str, int]] | None = None,
    wall_time_budget_seconds: float | None = None,
    episode_process_timeout_seconds: float | None = None,
    environment_time_limit_seconds: float | None = None,
    stopping_rule: str = "historical",
    qualification_source: str | Path | None = None,
    use_global_collection_lock: bool = True,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported closed-loop config")
    if not isinstance(config.get("deterministic_pp_replay", False), bool):
        raise ValueError("deterministic_pp_replay must be boolean")
    config = _with_time_budget_overrides(
        config,
        wall_time_budget_seconds,
        episode_process_timeout_seconds,
        environment_time_limit_seconds,
    )
    config = _with_stopping_rule(config, stopping_rule)
    if trace_format not in TRACE_FORMATS:
        raise ValueError(f"unsupported trace format: {trace_format}")
    if feature_backend not in FEATURE_BACKENDS:
        raise ValueError(f"unsupported feature backend: {feature_backend}")
    if controller_runtime not in CONTROLLER_RUNTIMES:
        raise ValueError(f"unsupported controller runtime: {controller_runtime}")
    if verification_profile not in VERIFICATION_PROFILES:
        raise ValueError(f"unsupported verification profile: {verification_profile}")
    controller_mode, controller_root, controller_manifest = resolve_controller_mode(
        project_root, controller, controller_bundle
    )
    v3_s3_root: Path | None = None
    v3_s3_manifest: dict[str, Any] | None = None
    if controller_mode == "v3-s3":
        v3_s3_root = Path(
            str(v3_s3_bundle or DEFAULT_V3_S3_BUNDLE)
        )
        if not v3_s3_root.is_absolute():
            v3_s3_root = project_root / v3_s3_root
        v3_s3_root = v3_s3_root.resolve()
        loaded_v3_s3 = load_v3_s3_bundle(v3_s3_root)
        v3_s3_manifest = loaded_v3_s3.manifest
        if not bool(v3_s3_manifest.get("native_audit_completed")):
            raise ValueError("v3-s3 requires a completed native audit")
        proposal_sizes = set(map(int, config["proposal"]["neighborhood_sizes"]))
        if proposal_sizes != {4, 8, 16}:
            raise ValueError(
                "v3-s3 requires the frozen 4/8/16 candidate space"
            )
    elif v3_s3_bundle is not None:
        raise ValueError("v3_s3_bundle is only valid with v3-s3")
    storage_fp = storage_fingerprint(trace_format)
    split = str(config["split"])
    solver_seeds = configured_solver_seeds(config)
    policies = configured_policies(config)
    phases = {"qualify", "all", *policies}
    if phase not in phases:
        raise ValueError(f"unsupported closed-loop phase: {phase}")
    all_rows = _load_dataset_rows(dataset_root, [split])
    rows = _selected_rows(all_rows, task_ids)
    available_job_keys = {
        (str(row["task_id"]), int(solver_seed))
        for row in rows
        for solver_seed in solver_seeds
    }
    normalized_job_keys = (
        {(str(task_id), int(seed)) for task_id, seed in job_keys}
        if job_keys is not None
        else None
    )
    normalized_cohort_job_keys = (
        {(str(task_id), int(seed)) for task_id, seed in cohort_job_keys}
        if cohort_job_keys is not None
        else None
    )
    if (
        normalized_cohort_job_keys is not None
        and not normalized_cohort_job_keys <= available_job_keys
    ):
        unknown = sorted(normalized_cohort_job_keys - available_job_keys)
        raise ValueError(f"closed-loop cohort contains unknown task/seed pairs: {unknown}")
    if normalized_job_keys is not None and not normalized_job_keys <= available_job_keys:
        unknown = sorted(normalized_job_keys - available_job_keys)
        raise ValueError(f"closed-loop job filter contains unknown task/seed pairs: {unknown}")
    if (
        normalized_job_keys is not None
        and normalized_cohort_job_keys is not None
        and not normalized_job_keys <= normalized_cohort_job_keys
    ):
        raise ValueError("closed-loop execution slice is outside the registered cohort")
    formal = task_ids is None and bool(config.get("formal", True))
    design = closed_loop_dataset_design(
        all_rows, split, dict(config.get("dataset_design", {}))
    )
    dataset_mode = str(config.get("dataset_design", {}).get("mode", "structured"))
    if dataset_mode in {"movingai_ood", "balanced_wall_clock"}:
        registered_ids = set(map(str, config["dataset_design"].get("historical_map_ids", [])))
        current_ids = {str(row["map_id"]) for row in all_rows}
        overlap = sorted(current_ids & registered_ids)
        isolation = {
            "passed": not overlap,
            "mode": f"{dataset_mode}_map_id",
            "current_map_ids": sorted(current_ids),
            "historical_overlap": overlap,
        }
    else:
        isolation = _seed_isolation(
            all_rows, list(config.get("reference_datasets", [])), project_root
        )
    frozen_root = Path(str(config["frozen_models"]))
    if not frozen_root.is_absolute():
        frozen_root = project_root / frozen_root
    bundle = load_frozen_policy_bundle(frozen_root, dict(config["model_registration"]))
    if controller_manifest is not None:
        expected_source_manifest = str(
            config["model_registration"].get("deployment_manifest_sha256", "")
        ).lower()
        actual_source_manifest = str(
            controller_manifest.get("source_bundle", {}).get("manifest_sha256", "")
        ).lower()
        registered_alternatives = dict(
            config["model_registration"].get("registered_controller_bundles", {})
        )
        controller_id = str(controller_manifest.get("controller_id", ""))
        alternative = dict(registered_alternatives.get(controller_id, {}))
        alternative_manifest_sha = str(
            alternative.get("controller_manifest_sha256", "")
        ).lower()
        loaded_manifest_sha = _sha256(controller_root / "controller_manifest.json")
        registered_alternative = bool(
            controller_id
            and alternative_manifest_sha
            and loaded_manifest_sha == alternative_manifest_sha
        )
        if actual_source_manifest != expected_source_manifest and not registered_alternative:
            raise ValueError("controller-v2 was built from a different v1 deployment bundle")
    effective_workers = int(workers or config["workers"])
    dataset_fp = _dataset_fingerprint(dataset_root)
    implementation = controller_implementation_fingerprint(project_root)
    effective = {
        **config,
        "task_ids_override": task_ids,
        "cohort_job_keys_override": (
            [list(value) for value in sorted(normalized_cohort_job_keys)]
            if normalized_cohort_job_keys is not None
            else None
        ),
        "controller": controller_mode,
        "feature_backend": feature_backend,
        "controller_runtime": controller_runtime,
        "verification_profile": verification_profile,
        "deterministic_pp_replay": bool(
            config.get("deterministic_pp_replay", False)
        ),
        "controller_bundle": str(controller_root),
        "feature_shadow_validation": bool(
            feature_shadow_validation
            or verification_profile == "audit"
            and controller_runtime in {"optimized", "auto"}
        ),
        "v3_s3_bundle": (
            str(v3_s3_root) if v3_s3_root is not None else None
        ),
    }
    config_fp = _fingerprint(effective)
    run_fp = _fingerprint(
        {
            "dataset_fingerprint": dataset_fp,
            "configuration_fingerprint": config_fp,
            "freeze_manifest": bundle.manifest,
            "controller_bundle_manifest": controller_manifest,
            "v3_s3_bundle_manifest": v3_s3_manifest,
            "controller_implementation": implementation,
        }
    )
    registered_job_keys = normalized_cohort_job_keys or available_job_keys
    estimate = {
        "task_count": len(rows),
        "reset_count": len(registered_job_keys),
        "solver_seeds": list(solver_seeds),
        "policies": list(policies),
        "policy_episode_count": len(registered_job_keys) * len(policies),
        "maximum_decisions_per_episode": (
            int(config["max_decisions"])
            if int(config["max_decisions"]) > 0
            else None
        ),
        "stopping_rule": stopping_rule,
        "maximum_proposals_per_decision": int(config["proposal"]["max_seed_agents"])
        * len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["trials"]),
        "maximum_candidates_per_decision": len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["candidates_per_family"]),
        "workers": effective_workers,
        "controller": controller_mode,
        "feature_backend": feature_backend,
        "controller_runtime": controller_runtime,
        "verification_profile": verification_profile,
        "wall_time_budget_seconds": float(config["wall_time_budget_seconds"]),
        "episode_process_timeout_seconds": float(
            config["episode_process_timeout_seconds"]
        ),
        "environment_time_limit_seconds": float(config["environment"]["time_limit"]),
        "environment_max_repair_iterations": int(
            config["environment"]["max_repair_iterations"]
        ),
    }
    if dry_run:
        return {
            "schema": CLOSED_LOOP_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "formal": formal,
            "run_fingerprint": run_fp,
            "trace_format": trace_format,
            "storage_fingerprint": storage_fp,
            "dataset_design": design,
            "seed_isolation": isolation,
            "frozen_models": bundle.manifest,
            "controller": controller_mode,
            "feature_backend": feature_backend,
            "controller_runtime": controller_runtime,
            "verification_profile": verification_profile,
            "feature_schema_id": (
                V3_S3_FEATURE_SCHEMA_ID
                if controller_mode == "v3-s3"
                else FEATURE_SCHEMA_ID
                if controller_mode != "official_adaptive"
                else None
            ),
            "feature_schema_sha256": (
                V3_S3_FEATURE_SCHEMA_SHA256
                if controller_mode == "v3-s3"
                else FEATURE_SCHEMA_SHA256
                if controller_mode != "official_adaptive"
                else None
            ),
            "controller_bundle": controller_manifest,
            "v3_s3_bundle": v3_s3_manifest,
            "controller_implementation": implementation,
            "estimate": estimate,
        }
    run_config = {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_fp,
        "configuration": effective,
        "configuration_fingerprint": config_fp,
        "run_fingerprint": run_fp,
        "trace_format": trace_format,
        "storage_fingerprint": storage_fp,
        "formal": formal,
        "dataset_design": design,
        "seed_isolation": isolation,
        "frozen_models": bundle.manifest,
        "controller": controller_mode,
        "feature_backend": feature_backend,
        "controller_runtime": controller_runtime,
        "verification_profile": verification_profile,
        "feature_schema_id": (
            V3_S3_FEATURE_SCHEMA_ID
            if controller_mode == "v3-s3"
            else FEATURE_SCHEMA_ID
            if controller_mode != "official_adaptive"
            else None
        ),
        "feature_schema_sha256": (
            V3_S3_FEATURE_SCHEMA_SHA256
            if controller_mode == "v3-s3"
            else FEATURE_SCHEMA_SHA256
            if controller_mode != "official_adaptive"
            else None
        ),
        "controller_bundle": controller_manifest,
        "v3_s3_bundle": v3_s3_manifest,
        "controller_implementation": implementation,
    }
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fp:
            raise ValueError("output contains a different closed-loop run")
        existing_format = str(existing.get("trace_format", TRACE_FORMAT_FULL_V1))
        existing_storage = str(
            existing.get("storage_fingerprint", storage_fingerprint(existing_format))
        )
        if existing_format != trace_format or existing_storage != storage_fp:
            raise ValueError("output contains a different closed-loop trace format")
        if not resume:
            raise ValueError("output already exists; pass resume to continue")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, run_config)
    sequence = policies if phase == "all" else (phase,)
    if phase == "all":
        sequence = ("qualify",) + policies
    summary: dict[str, Any] = {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fp,
        "trace_format": trace_format,
        "storage_fingerprint": storage_fp,
        "formal": formal,
        "controller": controller_mode,
        "feature_backend": feature_backend,
        "estimate": estimate,
    }
    for current in sequence:
        if current == "qualify":
            if qualification_source is not None:
                qualification_root = Path(qualification_source).resolve()
                source_run_path = qualification_root / "run_config.json"
                if not source_run_path.is_file():
                    raise ValueError("qualification source is missing run_config.json")
                source_run = _read_json(source_run_path)
                if _qualification_reuse_fingerprint(
                    source_run
                ) != _qualification_reuse_fingerprint(run_config):
                    raise ValueError(
                        "qualification source is incompatible with the current reset protocol"
                    )
                reused_results = _read_jsonl(
                    qualification_root / "qualification_manifest.jsonl"
                )
                expected_qualification_keys = (
                    normalized_job_keys
                    if normalized_job_keys is not None
                    else available_job_keys
                )
                results = [
                    value
                    for value in reused_results
                    if (
                        str(value["task_id"]),
                        int(value.get("solver_seed", solver_seeds[0])),
                    )
                    in expected_qualification_keys
                ]
            else:
                jobs = [
                    {
                        "row": row,
                        "solver_seed": solver_seed,
                        "dataset_root": str(dataset_root),
                        "environment": config["environment"],
                    }
                    for row in rows
                    for solver_seed in solver_seeds
                    if normalized_job_keys is None
                    or (str(row["task_id"]), int(solver_seed)) in normalized_job_keys
                ]
                with _CollectionRunLock(
                    output_root,
                    run_fp,
                    "closed-loop-qualification",
                    use_global_lock=use_global_collection_lock,
                ):
                    results = _run_jobs(
                        _qualification_worker,
                        jobs,
                        effective_workers,
                        phase="closed-loop-qualification",
                        output_root=output_root,
                        run_fingerprint=run_fp,
                        timeout_seconds=float(config["episode_process_timeout_seconds"]),
                    )
            qualification_manifest = output_root / "qualification_manifest.jsonl"
            existing_results = (
                _read_jsonl(qualification_manifest)
                if qualification_manifest.is_file()
                else []
            )
            merged = {
                (str(value["task_id"]), int(value["solver_seed"])): value
                for value in existing_results
            }
            merged.update(
                {
                    (str(value["task_id"]), int(value["solver_seed"])): value
                    for value in results
                }
            )
            results = [merged[key] for key in sorted(merged)]
            _write_jsonl(qualification_manifest, results)
            report = closed_loop_qualification_report(
                rows,
                results,
                config,
                design,
                isolation,
                formal=formal,
                expected_job_keys=normalized_cohort_job_keys,
            )
            _write_json(output_root / "qualification_report.json", report)
            summary["qualification"] = report
            if not report["passed"] and phase == "all":
                break
            continue
        qualification = closed_loop_qualification_report(
            rows,
            _read_jsonl(output_root / "qualification_manifest.jsonl"),
            config,
            design,
            isolation,
            formal=formal,
            expected_job_keys=normalized_cohort_job_keys,
        )
        if not qualification["passed"]:
            raise ValueError("closed-loop qualification failed; policy execution is forbidden")
        policy_manifest = output_root / f"{current}_manifest.jsonl"
        existing_results = (
            _read_jsonl(policy_manifest) if policy_manifest.is_file() else []
        )
        existing_by_key = {
            (str(value["task_id"]), int(value["solver_seed"])): value
            for value in existing_results
        }
        jobs = [
            {
                "row": row,
                "policy": current,
                "solver_seed": solver_seed,
                "dataset_root": str(dataset_root),
                "environment": config["environment"],
                "proposal": config["proposal"],
                "max_decisions": int(config["max_decisions"]),
                "metric_iteration_budget": (
                    int(config["metric_iteration_budget"])
                    if config.get("metric_iteration_budget") is not None
                    else None
                ),
                "wall_time_budget_seconds": float(config["wall_time_budget_seconds"]),
                "stopping_rule": stopping_rule,
                "safety_max_decisions": WALL_CLOCK_SAFETY_MAX_DECISIONS,
                "frozen_models": str(frozen_root.resolve()),
                "model_registration": config["model_registration"],
                "output_root": str(output_root),
                "run_fingerprint": run_fp,
                "trace_format": trace_format,
                "storage_fingerprint": storage_fp,
                "controller": controller_mode,
                "feature_backend": feature_backend,
                "controller_runtime": controller_runtime,
                "verification_profile": verification_profile,
                "controller_bundle": str(controller_root),
                "feature_shadow_validation": bool(
                    feature_shadow_validation
                    or verification_profile == "audit"
                    and controller_runtime in {"optimized", "auto"}
                ),
                "proposal_shadow_validation": bool(
                    verification_profile == "audit"
                    and controller_runtime in {"optimized", "auto"}
                ),
                "v3_s3_bundle": (
                    str(v3_s3_root) if v3_s3_root is not None else None
                ),
                "proposal_state_verification": (
                    "always" if verification_profile == "audit" else "sampled"
                ),
                "resume": resume,
                "require_finalization_timings": True,
                "deterministic_pp_replay": bool(
                    config.get("deterministic_pp_replay", False)
                ),
                "existing_manifest_row": existing_by_key.get(
                    (str(row["task_id"]), int(solver_seed))
                ),
            }
            for row in rows
            for solver_seed in solver_seeds
            if normalized_job_keys is None
            or (str(row["task_id"]), int(solver_seed)) in normalized_job_keys
        ]
        with _CollectionRunLock(
            output_root,
            run_fp,
            f"closed-loop-{current}",
            use_global_lock=use_global_collection_lock,
        ):
            results = _run_jobs(
                _closed_loop_episode_worker,
                jobs,
                effective_workers,
                phase=f"closed-loop-{current}",
                output_root=output_root,
                run_fingerprint=run_fp,
                timeout_seconds=float(config["episode_process_timeout_seconds"]),
            )
        merged = {
            (str(value["task_id"]), int(value["solver_seed"])): value
            for value in existing_results
        }
        merged.update(
            {
                (str(value["task_id"]), int(value["solver_seed"])): value
                for value in results
            }
        )
        results = [merged[key] for key in sorted(merged)]
        _write_jsonl(policy_manifest, results)
        summary[current] = _collection_policy_summary(results)
        if summary[current]["error_count"] and phase == "all":
            break
    _write_json(output_root / "collection_summary.json", summary)
    return summary


__all__ = [
    "CLOSED_LOOP_SCHEMA",
    "CONTROLLER_MODES",
    "DEFAULT_CONTROLLER_BUNDLE",
    "STOPPING_RULES",
    "CollectionLockError",
    "LEARNED_POLICIES",
    "REPAIR_TIMING_SCHEMA",
    "REPAIR_TIMING_SCHEMA_V1",
    "REPAIR_TIMING_SCHEMA_V2",
    "REPAIR_TIMING_SCHEMAS",
    "FIXED_POLICIES",
    "POLICIES",
    "SUPPORTED_POLICIES",
    "closed_loop_dataset_design",
    "movingai_ood_dataset_design",
    "closed_loop_qualification_report",
    "configured_policies",
    "configured_solver_seeds",
    "feature_range_diagnostic",
    "export_portable_policy_bundle",
    "fixed_budget_conflict_auc",
    "wall_clock_conflict_auc",
    "generate_online_candidates",
    "load_frozen_policy_bundle",
    "online_candidate_rows",
    "PortablePairwiseModel",
    "proposal_random_seed",
    "proposal_random_seeds",
    "repair_random_seed",
    "resolve_controller_mode",
    "run_closed_loop_collection",
    "score_online_candidates",
    "verify_portable_policy_bundle",
]
