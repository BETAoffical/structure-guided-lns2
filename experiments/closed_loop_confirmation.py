from __future__ import annotations

import collections
import itertools
import json
import math
import os
import pickle
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    episode_id as _episode_id,
    select_rows_by_task_id as _selected_rows,
    sha256_file as _sha256,
)
from experiments.context_audit import _pair_vector
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V1,
    EPISODE_SCHEMA_V2,
    TRACE_FORMAT_DELTA_GZIP_V2,
    TRACE_FORMAT_FULL_V1,
    TRACE_FORMATS,
    TraceStorageError,
    apply_extras_delta,
    apply_state_delta,
    encode_finish_event,
    encode_initial_event,
    encode_transition_event,
    open_trace_text,
    partial_trace_path,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
    storage_fingerprint,
    trace_file_metadata,
    trace_suffix,
)
from experiments.compact_controller_model import (
    compact_runtime_model,
    load_controller_bundle,
)
from experiments.feature_schema_v2 import FEATURE_SCHEMA_ID, FEATURE_SCHEMA_SHA256
from experiments.state_analysis import (
    StaticGridAnalysis,
    analyze_state,
    analyze_static_grid,
)
from experiments.neighborhood_candidates import (
    _seed_isolation,
    conflict_density,
    conflict_severity,
    no_pruning_metrics,
    select_representative_neighborhoods,
)
from experiments.online_feature_engine import (
    FEATURE_BACKENDS,
    OnlineFeatureEngine,
    _native_vector_function,
)
from experiments.neighborhood_features import (
    _feature_profiles_from_shared,
    candidate_feature_cache,
    state_dynamic_features,
    static_context_features,
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
    select_seed_agents,
    state_fingerprint,
)
from experiments.stall_guard import (
    StallGuardConfig,
    StallGuardState,
    load_stall_guard_config,
    repair_structure_fingerprint,
)
from experiments.repair_aware import (
    RepairAwareBundle,
    RepairAwareConfig,
    RepairAwareState,
    adaptive_feature_row,
    load_repair_aware_bundle,
    load_repair_aware_config,
)


CLOSED_LOOP_SCHEMA = "lns2.closed_loop_confirmation.v1"
EPISODE_SCHEMA = EPISODE_SCHEMA_V1
FIXED_POLICIES = ("fixed_target", "fixed_collision", "fixed_random")
POLICIES = ("official_adaptive", "proposal_dynamic", "realized_dynamic")
SUPPORTED_POLICIES = ("official_adaptive", *FIXED_POLICIES, "proposal_dynamic", "realized_dynamic")
LEARNED_POLICIES = ("proposal_dynamic", "realized_dynamic")
CONTROLLER_MODES = (
    "v1-full",
    "v2-full",
    "v2-stall-safe",
    "v2-repair-aware",
)
CONTROLLER_RUNTIMES = ("reference", "optimized", "auto")
VERIFICATION_PROFILES = ("audit", "deployment")
STOPPING_RULES = ("historical", "wall-clock")
WALL_CLOCK_SAFETY_MAX_DECISIONS = 100_000
REPAIR_TIMING_SCHEMA = "lns2.repair_timing.v1"
DEFAULT_CONTROLLER_BUNDLE = "artifacts/initlns-closed-loop-controller-v2"
DEFAULT_REPAIR_AWARE_BUNDLE = "build/initlns-repair-aware-controller-v1"
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
    "experiments/stall_guard.py",
    "experiments/repair_aware.py",
    "src/python_bindings.cpp",
    "src/jsonl_observer.cpp",
    "src/online_features.cpp",
    "src/online_features.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


class ClosedLoopTraceError(ValueError):
    pass


class ClosedLoopExecutionError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def controller_implementation_fingerprint(project_root: Path) -> dict[str, Any]:
    files = {
        relative: _sha256(project_root / relative)
        for relative in CONTROLLER_IMPLEMENTATION_FILES
    }
    native_module = None
    try:
        import lns2_env

        native_path = Path(str(lns2_env.__file__)).resolve()
        native_module = {"path": native_path.name, "sha256": _sha256(native_path)}
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
            str(loaded.manifest.get("default_controller", "v1-full"))
            if loaded is not None
            else "v1-full"
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
    if str(settings.get("mode", "structured")) == "movingai_ood":
        return movingai_ood_dataset_design(rows, split, settings)
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
        if len({int(row["task_seed"]) for row in tasks}) != 4:
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
            cohort.append(
                {
                    "map_id": str(source["map_id"]),
                    "task_id": str(source["task_id"]),
                    "solver_seed": solver_seed,
                    "layout_mode": str(source["layout_mode"]),
                    "task_variant": str(source["task_variant"]),
                    "agent_count": agents,
                    "initial_conflicts": conflicts,
                    "initial_feasible": conflicts == 0,
                    "conflict_density": density,
                    "conflict_severity": conflict_severity(density, thresholds),
                    "state_fingerprint": str(result["state_fingerprint"]),
                }
            )
    nonzero = [row for row in cohort if int(row["initial_conflicts"]) > 0]
    by_layout = collections.Counter(str(row["layout_mode"]) for row in nonzero)
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
    settings = dict(config["qualification"])
    qualification_mode = str(settings.get("mode", "structured"))
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
        }
        if formal
        else {
            "minimum_nonzero_states": bool(nonzero),
            "minimum_nonzero_per_layout": True,
            "minimum_active_maps": True,
            "minimum_nonzero_per_solver_seed": True,
        }
        )
    gates = {
        "dataset_design": bool(design["passed"]) if formal else True,
        "seed_isolation": bool(isolation["passed"]),
        "all_resets_valid": len(cohort) == len(expected_keys) and not errors,
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
                "initial_feasible_count": sum(int(item["initial_conflicts"]) == 0 for item in group),
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
        "initial_feasible_count": len(cohort) - len(nonzero),
        "nonzero_state_count": len(nonzero),
        "nonzero_by_layout": dict(sorted(by_layout.items())),
        "nonzero_by_solver_seed": {
            str(seed): by_solver_seed.get(seed, 0) for seed in solver_seeds
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


@dataclass
class FrozenPolicyBundle:
    models: dict[str, Any]
    ranges: dict[str, dict[str, tuple[float, float]]]
    manifest: dict[str, Any]


@dataclass
class PortablePairwiseModel:
    profile: str
    feature_names: list[str]
    baseline: float
    trees: list[list[dict[str, Any]]]
    native_predictor: Any | None = None

    def predict_positive(self, vectors: list[list[float]]) -> list[float]:
        if self.native_predictor is not None:
            return list(map(float, self.native_predictor.predict_positive(vectors)))
        probabilities = []
        for vector in vectors:
            raw = self.baseline
            for nodes in self.trees:
                index = 0
                while not bool(nodes[index]["is_leaf"]):
                    node = nodes[index]
                    value = float(vector[int(node["feature_idx"])])
                    go_left = (
                        math.isnan(value) and bool(node["missing_go_to_left"])
                    ) or (
                        not math.isnan(value)
                        and value <= float(node["num_threshold"])
                    )
                    index = int(node["left"] if go_left else node["right"])
                raw += float(nodes[index]["value"])
            if raw >= 0.0:
                probabilities.append(1.0 / (1.0 + math.exp(-raw)))
            else:
                exponential = math.exp(raw)
                probabilities.append(exponential / (1.0 + exponential))
        return probabilities


def export_portable_policy_bundle(
    frozen_root: str | Path,
    registration: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    root = Path(frozen_root).resolve()
    freeze_manifest = _read_json(root / "freeze_manifest.json")
    if bool(freeze_manifest.get("confirmation_labels_seen", True)):
        raise ValueError("frozen model manifest has seen confirmation labels")
    model_rows = {str(row["profile"]): row for row in freeze_manifest["models"]}
    expected = {
        str(name): str(value).lower()
        for name, value in dict(registration["model_sha256"]).items()
    }
    output_root = Path(output).resolve()
    exported = []
    feature_names_by_profile: dict[str, list[str]] = {}
    for profile in LEARNED_POLICIES:
        source = root / str(model_rows[profile]["model_file"])
        source_sha = _sha256(source)
        if source_sha != expected[profile]:
            raise ValueError(f"frozen model SHA256 mismatch: {profile}")
        with source.open("rb") as stream:
            model = pickle.load(stream)
        estimator = model.estimator
        if list(map(int, estimator.classes_)) != [0, 1]:
            raise ValueError("portable exporter supports only binary pairwise models")
        trees = []
        for stage in estimator._predictors:
            if len(stage) != 1:
                raise ValueError("portable exporter supports one binary tree per stage")
            predictor = stage[0]
            nodes = []
            for node in predictor.nodes:
                if bool(node["is_categorical"]):
                    raise ValueError("portable exporter does not support categorical tree nodes")
                nodes.append(
                    {
                        "value": float(node["value"]),
                        "feature_idx": int(node["feature_idx"]),
                        "num_threshold": float(node["num_threshold"]),
                        "missing_go_to_left": bool(node["missing_go_to_left"]),
                        "left": int(node["left"]),
                        "right": int(node["right"]),
                        "is_leaf": bool(node["is_leaf"]),
                    }
                )
            trees.append(nodes)
        payload = {
            "schema": "lns2.portable_pairwise_hist_gbdt.v1",
            "schema_version": 1,
            "profile": profile,
            "source_model_sha256": source_sha,
            "feature_names": list(model.feature_names),
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": trees,
        }
        path = output_root / f"pairwise__{profile}.json"
        _write_json(path, payload)
        feature_names_by_profile[profile] = list(model.feature_names)
        exported.append(
            {
                "profile": profile,
                "file": path.relative_to(output_root).as_posix(),
                "sha256": _sha256(path),
                "source_model_sha256": source_sha,
                "feature_count": len(model.feature_names),
                "tree_count": len(trees),
            }
        )
    index_path = Path(str(registration["development_index"]))
    if not index_path.is_absolute():
        index_path = Path(__file__).resolve().parents[1] / index_path
    index_path = index_path.resolve()
    expected_index = str(registration["development_index_sha256"]).lower()
    if _sha256(index_path) != expected_index:
        raise ValueError("development ranking index SHA256 mismatch")
    development_rows = _read_jsonl(index_path)
    feature_ranges = {}
    for profile, feature_names in feature_names_by_profile.items():
        values: dict[str, list[float]] = {name: [] for name in feature_names}
        for row in development_rows:
            features = dict(row["features"][profile])
            for name in feature_names:
                values[name].append(float(features.get(name, 0.0)))
        feature_ranges[profile] = {
            name: [min(numbers), max(numbers)] for name, numbers in values.items()
        }
    manifest = {
        "schema": "lns2.portable_pairwise_bundle.v2",
        "schema_version": 2,
        "models": exported,
        "feature_ranges": feature_ranges,
        "development_index_sha256": expected_index,
        "development_state_count": len({str(row["state_id"]) for row in development_rows}),
        "development_candidate_count": len(development_rows),
        "model_parameters": freeze_manifest.get("model_parameters", {}),
        "confirmation_labels_seen": False,
    }
    _write_json(output_root / "portable_manifest.json", manifest)
    return manifest


def _load_portable_models(
    portable_root: Path,
    expected_portable: dict[str, str],
    expected_source: dict[str, str],
) -> dict[str, PortablePairwiseModel]:
    manifest = _read_json(portable_root / "portable_manifest.json")
    if bool(manifest.get("confirmation_labels_seen", True)):
        raise ValueError("portable model manifest has seen confirmation labels")
    rows = {str(row["profile"]): row for row in manifest["models"]}
    models = {}
    for profile in LEARNED_POLICIES:
        row = rows[profile]
        path = portable_root / str(row["file"])
        digest = _sha256(path)
        if digest != expected_portable[profile] or digest != str(row["sha256"]):
            raise ValueError(f"portable model SHA256 mismatch: {profile}")
        payload = _read_json(path)
        if (
            str(payload["profile"]) != profile
            or str(payload["source_model_sha256"]) != expected_source[profile]
        ):
            raise ValueError(f"portable model provenance mismatch: {profile}")
        native_predictor = None
        try:
            import lns2_env

            predictor_type = getattr(lns2_env, "PortableTreeEnsemble", None)
            if predictor_type is not None:
                native_predictor = predictor_type(
                    float(payload["baseline"]), list(payload["trees"])
                )
        except ImportError:
            pass
        models[profile] = PortablePairwiseModel(
            profile=profile,
            feature_names=list(map(str, payload["feature_names"])),
            baseline=float(payload["baseline"]),
            trees=list(payload["trees"]),
            native_predictor=native_predictor,
        )
    return models


def _load_deployment_policy_bundle(
    deployment_root: Path, registration: dict[str, Any]
) -> FrozenPolicyBundle:
    manifest_path = deployment_root / "portable_manifest.json"
    expected_manifest_sha = str(registration.get("deployment_manifest_sha256", "")).lower()
    if not expected_manifest_sha or _sha256(manifest_path) != expected_manifest_sha:
        raise ValueError("deployment manifest SHA256 mismatch")
    manifest = _read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != 2:
        raise ValueError("deployment bundle must use portable schema version 2")
    if str(manifest.get("development_index_sha256", "")).lower() != str(
        registration["development_index_sha256"]
    ).lower():
        raise ValueError("deployment bundle development index provenance mismatch")
    expected_models = {
        str(name): str(value).lower()
        for name, value in dict(registration["model_sha256"]).items()
    }
    expected_portable = {
        str(name): str(value).lower()
        for name, value in dict(registration["portable_model_sha256"]).items()
    }
    models = _load_portable_models(deployment_root, expected_portable, expected_models)
    stored_ranges = dict(manifest.get("feature_ranges", {}))
    ranges: dict[str, dict[str, tuple[float, float]]] = {}
    for profile, model in models.items():
        profile_ranges = dict(stored_ranges.get(profile, {}))
        if set(profile_ranges) != set(model.feature_names):
            raise ValueError(f"deployment feature ranges are incomplete: {profile}")
        ranges[profile] = {
            name: (float(profile_ranges[name][0]), float(profile_ranges[name][1]))
            for name in model.feature_names
        }
    return FrozenPolicyBundle(models=models, ranges=ranges, manifest=manifest)


def verify_portable_policy_bundle(
    frozen_root: str | Path, registration: dict[str, Any]
) -> dict[str, Any]:
    native_registration = {
        key: value
        for key, value in registration.items()
        if key not in {"deployment_bundle", "portable_models", "portable_model_sha256"}
    }
    native = load_frozen_policy_bundle(frozen_root, native_registration)
    portable = load_frozen_policy_bundle(frozen_root, registration)
    index_path = Path(str(registration["development_index"]))
    if not index_path.is_absolute():
        index_path = Path(__file__).resolve().parents[1] / index_path
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _read_jsonl(index_path.resolve()):
        grouped[str(row["state_id"])].append(row)
    profiles = {}
    for profile in LEARNED_POLICIES:
        mismatches = 0
        maximum_score_delta = 0.0
        pair_count = 0
        for candidates in grouped.values():
            native_index, native_scores, _ = score_online_candidates(
                candidates, native.models[profile]
            )
            portable_index, portable_scores, _ = score_online_candidates(
                candidates, portable.models[profile]
            )
            mismatches += native_index != portable_index
            maximum_score_delta = max(
                maximum_score_delta,
                max(
                    abs(first - second)
                    for first, second in zip(native_scores, portable_scores)
                ),
            )
            pair_count += len(candidates) * (len(candidates) - 1) // 2
        profiles[profile] = {
            "state_count": len(grouped),
            "pair_count": pair_count,
            "selection_mismatch_count": mismatches,
            "maximum_score_delta": maximum_score_delta,
            "passed": mismatches == 0 and maximum_score_delta <= 1e-12,
        }
    feature_parity = None
    source_value = registration.get("development_candidates")
    if source_value:
        source_path = Path(str(source_value))
        if not source_path.is_absolute():
            source_path = Path(__file__).resolve().parents[1] / source_path
        source_path = source_path.resolve()
        expected_source_sha = str(registration["development_candidates_sha256"]).lower()
        if _sha256(source_path) != expected_source_sha:
            raise ValueError("development candidate source SHA256 mismatch")
        indexed = {
            (str(row["state_id"]), str(row["candidate_id"])): row
            for row in _read_jsonl(index_path.resolve())
        }
        checked = 0
        mismatches = []
        seen = set()
        for source in _read_jsonl(source_path):
            state_id = str(source["state_id"])
            for computed in online_candidate_rows(source["state"], source["candidates"]):
                key = (state_id, str(computed["candidate_id"]))
                checked += 1
                seen.add(key)
                if key not in indexed or computed["features"] != indexed[key]["features"]:
                    mismatches.append(list(key))
        missing = sorted([list(key) for key in set(indexed) - seen])
        feature_parity = {
            "candidate_count": checked,
            "mismatch_count": len(mismatches),
            "missing_count": len(missing),
            "mismatches": mismatches,
            "missing": missing,
            "passed": not mismatches and not missing and checked == len(indexed),
        }
    passed = all(row["passed"] for row in profiles.values()) and (
        feature_parity is None or bool(feature_parity["passed"])
    )
    return {
        "schema": "lns2.portable_pairwise_equivalence.v1",
        "schema_version": 1,
        "passed": passed,
        "profiles": profiles,
        "online_feature_parity": feature_parity,
    }


def load_frozen_policy_bundle(
    frozen_root: str | Path, registration: dict[str, Any]
) -> FrozenPolicyBundle:
    deployment_value = registration.get("deployment_bundle")
    if deployment_value:
        deployment_root = Path(str(deployment_value))
        if not deployment_root.is_absolute():
            deployment_root = Path(__file__).resolve().parents[1] / deployment_root
        return _load_deployment_policy_bundle(deployment_root.resolve(), registration)
    root = Path(frozen_root).resolve()
    manifest = _read_json(root / "freeze_manifest.json")
    if bool(manifest.get("confirmation_labels_seen", True)):
        raise ValueError("frozen model manifest has seen confirmation labels")
    registered_index = registration.get("development_index")
    if registered_index:
        index_path = Path(str(registered_index))
        if not index_path.is_absolute():
            index_path = Path(__file__).resolve().parents[1] / index_path
        index_path = index_path.resolve()
    else:
        index_path = Path(str(manifest["development_index"])).resolve()
    expected_index = str(registration["development_index_sha256"]).lower()
    if _sha256(index_path) != expected_index:
        raise ValueError("development ranking index SHA256 mismatch")
    model_rows = {str(row["profile"]): row for row in manifest.get("models", [])}
    expected_models = {
        str(name): str(value).lower()
        for name, value in dict(registration["model_sha256"]).items()
    }
    if not set(LEARNED_POLICIES).issubset(model_rows):
        raise ValueError("frozen model set is incomplete")
    models: dict[str, Any] = {}
    for profile in LEARNED_POLICIES:
        row = model_rows[profile]
        path = root / str(row["model_file"])
        digest = _sha256(path)
        if digest != expected_models[profile] or digest != str(row["model_sha256"]).lower():
            raise ValueError(f"frozen model SHA256 mismatch: {profile}")
    portable_value = registration.get("portable_models")
    if portable_value:
        portable_root = Path(str(portable_value))
        if not portable_root.is_absolute():
            portable_root = Path(__file__).resolve().parents[1] / portable_root
        expected_portable = {
            str(name): str(value).lower()
            for name, value in dict(registration["portable_model_sha256"]).items()
        }
        models = _load_portable_models(
            portable_root.resolve(), expected_portable, expected_models
        )
    else:
        for profile in LEARNED_POLICIES:
            row = model_rows[profile]
            path = root / str(row["model_file"])
            with path.open("rb") as stream:
                model = pickle.load(stream)
            if str(model.profile) != profile:
                raise ValueError(f"frozen model profile mismatch: {profile}")
            models[profile] = model
    development_rows = _read_jsonl(index_path)
    ranges: dict[str, dict[str, tuple[float, float]]] = {}
    for profile, model in models.items():
        values: dict[str, list[float]] = {name: [] for name in model.feature_names}
        for row in development_rows:
            features = dict(row["features"][profile])
            for name in model.feature_names:
                values[name].append(float(features.get(name, 0.0)))
        ranges[profile] = {
            name: (min(numbers), max(numbers)) for name, numbers in values.items()
        }
    return FrozenPolicyBundle(models=models, ranges=ranges, manifest=manifest)


def proposal_random_seed(
    task_id: str,
    solver_seed: int,
    state_hash: str,
    decision_index: int,
    seed_agent: int,
    heuristic: str,
    size: int,
    trial_index: int,
) -> int:
    return int(
        _fingerprint(
            {
                "namespace": "closed-loop-proposal-v1",
                "task_id": task_id,
                "solver_seed": solver_seed,
                "state_fingerprint": state_hash,
                "decision_index": decision_index,
                "seed_agent": seed_agent,
                "heuristic": heuristic,
                "size": size,
                "trial_index": trial_index,
            }
        )[:16],
        16,
    ) % (2**31)


def repair_random_seed(
    task_id: str,
    solver_seed: int,
    state_hash: str,
    decision_index: int,
    candidate_id: str,
    forbidden: Iterable[int],
) -> int:
    value = int(
        _fingerprint(
            {
                "namespace": "closed-loop-explicit-repair-v1",
                "task_id": task_id,
                "solver_seed": solver_seed,
                "state_fingerprint": state_hash,
                "decision_index": decision_index,
                "candidate_id": candidate_id,
            }
        )[:16],
        16,
    ) % (2**31)
    excluded = set(map(int, forbidden))
    while value in excluded:
        value = (value + 1) % (2**31)
    return value


def stall_guard_fallback_seed(
    task_id: str,
    solver_seed: int,
    state_anchor_fingerprint: str,
    decision_index: int,
) -> int:
    return int(
        _fingerprint(
            {
                "namespace": "closed-loop-stall-guard-official-v1",
                "task_id": task_id,
                "solver_seed": int(solver_seed),
                "state_anchor_fingerprint": state_anchor_fingerprint,
                "decision_index": int(decision_index),
            }
        )[:16],
        16,
    ) % (2**31)


def repair_aware_fallback_seed(
    task_id: str,
    solver_seed: int,
    state_anchor_fingerprint: str,
    decision_index: int,
) -> int:
    return int(
        _fingerprint(
            {
                "namespace": "closed-loop-repair-aware-official-v1",
                "task_id": task_id,
                "solver_seed": int(solver_seed),
                "state_anchor_fingerprint": state_anchor_fingerprint,
                "decision_index": int(decision_index),
            }
        )[:16],
        16,
    ) % (2**31)


def online_candidate_rows(
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    static_grid: StaticGridAnalysis | None = None,
) -> list[dict[str, Any]]:
    analysis = analyze_state(state, static_grid=static_grid)
    dynamic = state_dynamic_features(state, analysis)
    context = static_context_features(state)
    feature_cache = candidate_feature_cache(state, analysis)
    rows = []
    state_hash = state_fingerprint(state)
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows.append(
            {
                "state_id": state_hash,
                "candidate_id": candidate_id,
                "candidate_key": candidate_id,
                "features": _feature_profiles_from_shared(
                    state,
                    analysis,
                    candidate,
                    dynamic=dynamic,
                    context=context,
                    feature_cache=feature_cache,
                ),
            }
        )
    return rows


def score_online_candidates(
    rows: list[dict[str, Any]], model: Any
) -> tuple[int, list[float], float]:
    if not rows:
        raise ValueError("cannot score an empty candidate pool")
    direct_scorer = getattr(model, "score_candidates", None)
    direct_scores = direct_scorer(rows) if callable(direct_scorer) else None
    scores = (
        list(map(float, direct_scores))
        if direct_scores is not None
        else [0.0] * len(rows)
    )
    if len(scores) != len(rows):
        raise ValueError("direct candidate scorer returned the wrong number of scores")
    vectors = []
    reverse_vectors = []
    pairs = []
    if direct_scores is None:
        pair_vectors = getattr(model, "pair_vectors", None)
        if callable(pair_vectors):
            vectors, reverse_vectors, pairs = pair_vectors(rows)
        else:
            pair_vector = getattr(model, "pair_vector", None)
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    if callable(pair_vector):
                        vectors.append(pair_vector(rows[left], rows[right]))
                        reverse_vectors.append(pair_vector(rows[right], rows[left]))
                    else:
                        vectors.append(
                            _pair_vector(
                                rows[left], rows[right], model.profile, model.feature_names
                            )
                        )
                        reverse_vectors.append(
                            _pair_vector(
                                rows[right], rows[left], model.profile, model.feature_names
                            )
                        )
                    pairs.append((left, right))
    if vectors:
        predict_positive = getattr(model, "predict_positive", None)
        if callable(predict_positive):
            forward = predict_positive(vectors)
            reverse = predict_positive(reverse_vectors)
        else:
            import numpy as np

            forward = model.estimator.predict_proba(np.asarray(vectors, dtype=float))[:, 1]
            reverse = model.estimator.predict_proba(
                np.asarray(reverse_vectors, dtype=float)
            )[:, 1]
        probabilities = [
            (float(first) + (1.0 - float(second))) / 2.0
            for first, second in zip(forward, reverse)
        ]
        for probability, (left, right) in zip(probabilities, pairs):
            scores[left] += float(probability)
            scores[right] += 1.0 - float(probability)
    # Native sklearn, Python and the C++ portable evaluator can differ by a
    # handful of floating-point ulps. Treat those numerically identical scores
    # as ties so the candidate hash remains the cross-platform decision rule.
    stable_scores = [round(score, 12) for score in scores]
    order = sorted(
        range(len(rows)),
        key=lambda index: (-stable_scores[index], str(rows[index]["candidate_key"])),
    )
    margin = (
        stable_scores[order[0]] - stable_scores[order[1]]
        if len(order) > 1
        else stable_scores[order[0]]
    )
    return order[0], scores, margin


def feature_range_diagnostic(
    row: dict[str, Any], profile: str, ranges: dict[str, tuple[float, float]]
) -> dict[str, Any]:
    if "feature_values" in row:
        if str(row.get("feature_profile")) != profile:
            raise ValueError("dense feature row has the wrong profile")
        names = tuple(map(str, row.get("feature_names", ())))
        values = tuple(map(float, row["feature_values"]))
        if len(names) != len(values) or len(names) != len(set(names)):
            raise ValueError("dense feature row is invalid")
        features = dict(zip(names, values))
    else:
        features = dict(row["features"][profile])
    outside = []
    for name, (minimum, maximum) in ranges.items():
        value = float(features.get(name, 0.0))
        if value < minimum or value > maximum:
            outside.append(name)
    return {
        "feature_count": len(ranges),
        "outside_count": len(outside),
        "outside_fraction": len(outside) / len(ranges) if ranges else 0.0,
        "outside_features": sorted(outside),
    }


def generate_online_candidates(
    environment: Any,
    state: dict[str, Any],
    *,
    task_id: str,
    solver_seed: int,
    decision_index: int,
    proposal_config: dict[str, Any],
    state_hash: str | None = None,
    verify_full_state: bool = True,
    proposal_backend: str = "reference",
    shadow_validation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if proposal_backend not in CONTROLLER_RUNTIMES:
        raise ValueError(f"unsupported proposal backend: {proposal_backend}")
    state_hash = state_fingerprint(state) if state_hash is None else str(state_hash)
    get_revision = getattr(environment, "get_state_revision", None)
    revision_before = int(get_revision()) if callable(get_revision) else None
    request_generation_started = time.perf_counter()
    requests: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for seed_agent in select_seed_agents(state, int(proposal_config["max_seed_agents"])):
        for heuristic in map(str, proposal_config["heuristics"]):
            for size in map(int, proposal_config["neighborhood_sizes"]):
                for trial_index in range(int(proposal_config["trials"])):
                    random_seed = proposal_random_seed(
                        task_id,
                        solver_seed,
                        state_hash,
                        decision_index,
                        seed_agent,
                        heuristic,
                        size,
                        trial_index,
                    )
                    action = {
                        "mode": "seed",
                        "heuristic": heuristic,
                        "seed_agent": seed_agent,
                        "neighborhood_size": size,
                        "random_seed": random_seed,
                    }
                    requests.append(
                        (
                            action,
                            {
                                "family": f"{heuristic}:{size}",
                                "seed_agent": seed_agent,
                                "proposal_seed": random_seed,
                                "requested_size": size,
                            },
                        )
                    )
    request_generation_seconds = time.perf_counter() - request_generation_started
    actions = [action for action, _ in requests]
    started = time.perf_counter()
    propose_compact = getattr(environment, "propose_batch_compact", None)
    propose_batch = getattr(environment, "propose_batch", None)
    use_compact = proposal_backend in {"optimized", "auto"} and callable(
        propose_compact
    )
    if proposal_backend == "optimized" and not callable(propose_compact):
        raise RuntimeError(
            "optimized controller runtime requires propose_batch_compact"
        )
    if use_compact:
        compact_results = [_plain(value) for value in propose_compact(actions)]
        results = []
        for value in compact_results:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise RuntimeError("compact proposal batch returned an invalid row")
            results.append(
                {
                    "action_valid": bool(value[0]),
                    "generated": bool(value[1]),
                    "neighborhood": list(map(int, value[2])),
                }
            )
        backend = "compact"
    elif callable(propose_batch):
        results = [_plain(value) for value in propose_batch(actions)]
        backend = "batch"
    else:
        results = [_plain(environment.propose(action)) for action in actions]
        backend = "single_fallback"
    proposal_seconds = time.perf_counter() - started
    proposal_shadow_seconds = 0.0
    if shadow_validation:
        shadow_started = time.perf_counter()
        if use_compact:
            if not callable(propose_batch):
                raise RuntimeError("proposal shadow validation requires propose_batch")
            shadow_results = [_plain(value) for value in propose_batch(actions)]
        else:
            if not callable(propose_compact):
                raise RuntimeError(
                    "proposal shadow validation requires propose_batch_compact"
                )
            shadow_results = []
            for value in [_plain(item) for item in propose_compact(actions)]:
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    raise RuntimeError("compact proposal shadow returned an invalid row")
                shadow_results.append(
                    {
                        "action_valid": bool(value[0]),
                        "generated": bool(value[1]),
                        "neighborhood": list(map(int, value[2])),
                    }
                )
        proposal_shadow_seconds = time.perf_counter() - shadow_started
        primary_signature = [
            (
                bool(value.get("action_valid")),
                bool(value.get("generated")),
                tuple(sorted(map(int, value.get("neighborhood", [])))),
            )
            for value in results
        ]
        shadow_signature = [
            (
                bool(value.get("action_valid")),
                bool(value.get("generated")),
                tuple(sorted(map(int, value.get("neighborhood", [])))),
            )
            for value in shadow_results
        ]
        if primary_signature != shadow_signature:
            raise ClosedLoopExecutionError(
                "proposal_shadow_mismatch",
                "reference and compact proposal batches differ",
            )
    if len(results) != len(requests):
        raise RuntimeError("online proposal batch returned an unexpected result count")
    state_check_started = time.perf_counter()
    revision_after = int(get_revision()) if callable(get_revision) else None
    if revision_before is not None and revision_after != revision_before:
        raise ClosedLoopExecutionError(
            "revision_mismatch", "proposal changed the structural state revision"
        )
    full_state_verified = bool(verify_full_state or revision_before is None)
    state_check_fingerprint_seconds = 0.0
    if full_state_verified:
        after = _plain(environment.get_state())
        state_check_fingerprint_started = time.perf_counter()
        after_fingerprint = state_fingerprint(after)
        state_check_fingerprint_seconds = (
            time.perf_counter() - state_check_fingerprint_started
        )
        if after != state or after_fingerprint != state_hash:
            raise ClosedLoopExecutionError(
                "fingerprint_mismatch", "proposal changed the closed-loop repair state"
            )
    state_check_seconds = time.perf_counter() - state_check_started
    candidate_postprocess_started = time.perf_counter()
    proposals = []
    for (_, metadata), result in zip(requests, results):
        if not bool(result.get("action_valid")) or not bool(result.get("generated")):
            raise RuntimeError("valid online proposal was rejected")
        agents = sorted(map(int, result.get("neighborhood", [])))
        if not agents or len(agents) != len(set(agents)):
            raise RuntimeError("online proposal returned an invalid neighborhood")
        proposals.append(
            {
                **metadata,
                "agents": agents,
            }
        )
    candidates = select_representative_neighborhoods(
        proposals, int(proposal_config["candidates_per_family"])
    )
    if not candidates:
        raise RuntimeError("online proposal stage produced no explicit candidates")
    candidate_postprocess_seconds = (
        time.perf_counter() - candidate_postprocess_started
    )
    candidate_generation_seconds = (
        request_generation_seconds
        + proposal_seconds
        + candidate_postprocess_seconds
    )
    return candidates, {
        "proposal_count": len(proposals),
        "unique_neighborhood_count": len({tuple(row["agents"]) for row in proposals}),
        "candidate_count": len(candidates),
        "proposal_seconds": proposal_seconds,
        "proposal_shadow_seconds": proposal_shadow_seconds,
        "request_generation_seconds": request_generation_seconds,
        "candidate_postprocess_seconds": candidate_postprocess_seconds,
        "candidate_generation_seconds": candidate_generation_seconds,
        "state_check_seconds": state_check_seconds,
        "state_check_fingerprint_seconds": state_check_fingerprint_seconds,
        "state_check_backend": (
            "revision_and_full" if revision_before is not None and full_state_verified
            else "revision" if revision_before is not None
            else "full_state"
        ),
        "full_state_verified": full_state_verified,
        "state_revision": revision_after,
        "backend": backend,
        "shadow_validation": bool(shadow_validation),
        "shadow_validation_passed": bool(shadow_validation),
    }


def fixed_budget_conflict_auc(
    trajectory: list[int], budget: int, *, success: bool
) -> float:
    if budget <= 0 or not trajectory or len(trajectory) > budget + 1:
        raise ValueError("invalid fixed-budget conflict trajectory")
    values = list(map(int, trajectory))
    pad = 0 if success else values[-1]
    values.extend([pad] * (budget + 1 - len(values)))
    return sum((values[index] + values[index + 1]) / 2.0 for index in range(budget))


def wall_clock_conflict_auc(
    trajectory: list[int], transition_elapsed_seconds: list[float], budget_seconds: float
) -> float:
    """Integrate the observed conflict count up to a fixed wall-clock deadline.

    A repair that finishes after the deadline does not contribute its after-state;
    the before-state is carried to the deadline instead.  The initial conflict
    count is charged from time zero so initialization is represented consistently.
    """
    if (
        not trajectory
        or len(transition_elapsed_seconds) != len(trajectory) - 1
        or not math.isfinite(float(budget_seconds))
        or float(budget_seconds) <= 0.0
    ):
        raise ValueError("invalid wall-clock conflict trajectory")
    budget = float(budget_seconds)
    previous_time = 0.0
    current_conflicts = int(trajectory[0])
    area = 0.0
    for elapsed, after_conflicts in zip(
        transition_elapsed_seconds, trajectory[1:]
    ):
        event_time = float(elapsed)
        if not math.isfinite(event_time) or event_time < previous_time:
            raise ValueError("wall-clock transition times must be finite and ordered")
        clipped = min(event_time, budget)
        area += current_conflicts * max(0.0, clipped - previous_time)
        if event_time > budget:
            return area
        previous_time = event_time
        current_conflicts = int(after_conflicts)
    area += current_conflicts * max(0.0, budget - previous_time)
    return area


def validate_closed_loop_trace(
    path: str | Path,
    run_fingerprint: str,
    *,
    expected_episode_id: str | None = None,
    expected_policy: str | None = None,
    expected_solver_seed: int | None = None,
    metric_iteration_budget: int | None = None,
    collection_root: str | Path | None = None,
) -> dict[str, Any]:
    trace_path = Path(path)
    try:
        rows = read_trace_events(trace_path)
    except TraceStorageError as error:
        raise ClosedLoopTraceError(f"cannot read trace: {error}") from error
    if len(rows) < 2:
        raise ClosedLoopTraceError("trace must contain initial and finish events")
    event_schema = str(rows[0].get("schema"))
    if event_schema == EPISODE_SCHEMA_V1:
        trace_format = TRACE_FORMAT_FULL_V1
        expected_schema_version = SCHEMA_VERSION
    elif event_schema == EPISODE_SCHEMA_V2:
        trace_format = TRACE_FORMAT_DELTA_GZIP_V2
        expected_schema_version = 2
    else:
        raise ClosedLoopTraceError("trace contains an unexpected event schema")
    if any(str(row.get("schema")) != event_schema for row in rows):
        raise ClosedLoopTraceError("trace contains an unexpected event schema")
    if any(int(row.get("schema_version", -1)) != expected_schema_version for row in rows):
        raise ClosedLoopTraceError("trace contains an unsupported schema version")
    if trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
        expected_storage = storage_fingerprint(trace_format)
        if any(str(row.get("trace_format")) != trace_format for row in rows):
            raise ClosedLoopTraceError("compact trace format marker mismatch")
        if any(str(row.get("storage_fingerprint")) != expected_storage for row in rows):
            raise ClosedLoopTraceError("compact trace storage fingerprint mismatch")
    if any(str(row.get("run_fingerprint")) != run_fingerprint for row in rows):
        raise ClosedLoopTraceError("trace run fingerprint mismatch")
    if rows[0].get("event") != "initial" or rows[-1].get("event") != "finish":
        raise ClosedLoopTraceError("trace event boundaries are invalid")
    if any(row.get("event") != "transition" for row in rows[1:-1]):
        raise ClosedLoopTraceError("trace contains a non-transition event before finish")

    initial = rows[0]
    finish = rows[-1]
    episode_id = str(initial.get("episode_id"))
    policy = str(initial.get("policy"))
    solver_seed = int(initial.get("solver_seed", -1))
    if expected_episode_id is not None and episode_id != expected_episode_id:
        raise ClosedLoopTraceError("trace episode id mismatch")
    if expected_policy is not None and policy != expected_policy:
        raise ClosedLoopTraceError("trace policy mismatch")
    if expected_solver_seed is not None and solver_seed != expected_solver_seed:
        raise ClosedLoopTraceError("trace solver seed mismatch")
    if str(finish.get("episode_id")) != episode_id or str(finish.get("policy")) != policy:
        raise ClosedLoopTraceError("finish metadata mismatch")

    initial_state_ref = None
    if trace_format == TRACE_FORMAT_FULL_V1:
        state = initial.get("state")
        if not isinstance(state, dict):
            raise ClosedLoopTraceError("initial event is missing state")
    else:
        initial_state_ref = str(initial.get("state_blob", ""))
        if not initial_state_ref:
            raise ClosedLoopTraceError("compact initial event is missing state blob")
        try:
            blob_path = resolve_state_blob(
                trace_path,
                initial_state_ref,
                Path(collection_root).resolve() if collection_root is not None else None,
            )
            state = read_state_blob(blob_path)
        except TraceStorageError as error:
            raise ClosedLoopTraceError(str(error)) from error
        extras = initial.get("state_extras")
        if not isinstance(extras, dict):
            raise ClosedLoopTraceError("compact initial event has invalid state extras")
        if any(key in state for key in extras):
            raise ClosedLoopTraceError("compact initial state extras overlap fingerprint fields")
        state.update(extras)
    initial_hash = state_fingerprint(state)
    if str(initial.get("state_fingerprint")) != initial_hash:
        raise ClosedLoopTraceError("initial state fingerprint mismatch")
    conflicts = [int(state.get("num_of_colliding_pairs", -1))]
    learned_policy = policy in LEARNED_POLICIES
    route_counts: collections.Counter[str] = collections.Counter()
    previous_route: str | None = None
    route_switch_count = 0
    transition_elapsed_seconds: list[float] = []
    for decision_index, event in enumerate(rows[1:-1]):
        if str(event.get("episode_id")) != episode_id:
            raise ClosedLoopTraceError("transition episode id mismatch")
        if int(event.get("decision_index", -1)) != decision_index:
            raise ClosedLoopTraceError("transition decision indexes are not contiguous")
        before_hash = state_fingerprint(state)
        if str(event.get("before_fingerprint")) != before_hash:
            raise ClosedLoopTraceError("transition before fingerprint mismatch")
        if trace_format == TRACE_FORMAT_FULL_V1:
            after = event.get("after")
            if not isinstance(after, dict):
                raise ClosedLoopTraceError("transition is missing after state")
        else:
            try:
                after = apply_state_delta(state, event.get("state_delta"))
                after.update(
                    apply_extras_delta(state, event.get("state_extras_delta"))
                )
            except (TraceStorageError, TypeError, ValueError) as error:
                raise ClosedLoopTraceError(
                    f"transition state delta is invalid: {error}"
                ) from error
        after_hash = state_fingerprint(after)
        if str(event.get("after_fingerprint")) != after_hash:
            raise ClosedLoopTraceError("transition after fingerprint mismatch")
        metrics = event.get("metrics")
        action = event.get("action")
        if not isinstance(metrics, dict) or not isinstance(action, dict):
            raise ClosedLoopTraceError("transition is missing action or metrics")
        if int(metrics.get("conflicts_before", -1)) != conflicts[-1]:
            raise ClosedLoopTraceError("transition conflicts_before mismatch")
        after_conflicts = int(after.get("num_of_colliding_pairs", -1))
        if int(metrics.get("conflicts_after", -1)) != after_conflicts:
            raise ClosedLoopTraceError("transition conflicts_after mismatch")
        if event.get("low_level_delta") != _low_level_delta(state, after):
            raise ClosedLoopTraceError("transition low-level delta mismatch")
        if bool(event.get("terminated")) != bool(after.get("feasible")):
            raise ClosedLoopTraceError("transition terminated flag mismatch")
        if bool(event.get("truncated")) != (
            bool(after.get("done")) and not bool(after.get("feasible"))
        ):
            raise ClosedLoopTraceError("transition truncated flag mismatch")
        elapsed_seconds = float(event.get("elapsed_wall_seconds", -1.0))
        if (
            not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0.0
            or (
                transition_elapsed_seconds
                and elapsed_seconds < transition_elapsed_seconds[-1]
            )
        ):
            raise ClosedLoopTraceError("transition wall times are invalid")
        transition_elapsed_seconds.append(elapsed_seconds)
        timings = event.get("timings")
        if timings is not None:
            if not isinstance(timings, dict):
                raise ClosedLoopTraceError("transition timings are invalid")
            numeric_timings = {
                str(name): float(value) for name, value in timings.items()
            }
            if any(
                not math.isfinite(value) or value < 0.0
                for value in numeric_timings.values()
            ):
                raise ClosedLoopTraceError("transition timings must be non-negative")
            selection_expected = float(
                numeric_timings.get("controller_before_repair_seconds", 0.0)
            ) + float(
                numeric_timings.get(
                    "native_neighborhood_generation_seconds", 0.0
                )
            )
            tolerance = max(1e-6, 0.01 * max(selection_expected, 1e-6))
            if not math.isclose(
                float(numeric_timings.get("neighborhood_selection_seconds", 0.0)),
                selection_expected,
                rel_tol=0.01,
                abs_tol=tolerance,
            ):
                raise ClosedLoopTraceError("neighborhood selection timing does not close")
            step_partition = sum(
                float(numeric_timings.get(name, 0.0))
                for name in (
                    "native_neighborhood_generation_seconds",
                    "pp_replan_seconds",
                    "repair_bookkeeping_seconds",
                    "state_export_seconds",
                    "environment_step_residual_seconds",
                )
            )
            step_wall = float(
                numeric_timings.get(
                    "environment_step_wall_seconds",
                    event.get("repair_wall_seconds", 0.0),
                )
            )
            if step_partition > step_wall + max(1e-5, 0.01 * step_wall):
                raise ClosedLoopTraceError("environment step timing exceeds its parent")
            native_step = float(metrics.get("native_step_seconds", 0.0))
            native_partition = sum(
                float(metrics.get(name, 0.0))
                for name in (
                    "native_neighborhood_generation_seconds",
                    "native_replan_seconds",
                    "native_state_snapshot_seconds",
                    "native_repair_bookkeeping_seconds",
                    "native_residual_seconds",
                )
            )
            if native_step > 0.0 and not math.isclose(
                native_partition,
                native_step,
                rel_tol=0.01,
                abs_tol=max(1e-6, 0.01 * native_step),
            ):
                raise ClosedLoopTraceError("native step timing does not close")
        if learned_policy:
            controller = event.get("controller")
            if not isinstance(controller, dict):
                raise ClosedLoopTraceError("learned transition is missing controller data")
            route = str(controller.get("route", "model"))
            if route not in {"model", "official_adaptive"}:
                raise ClosedLoopTraceError("learned transition has an invalid route")
            route_counts[route] += 1
            if str(controller.get("controller_mode")) == "v2-stall-safe":
                guard = controller.get("stall_guard")
                if not isinstance(guard, dict):
                    raise ClosedLoopTraceError(
                        "stall-safe transition is missing guard diagnostics"
                    )
                if str(guard.get("route")) != route:
                    raise ClosedLoopTraceError("stall guard route mismatch")
            if str(controller.get("controller_mode")) == "v2-repair-aware":
                repair = controller.get("repair_aware")
                if not isinstance(repair, dict):
                    raise ClosedLoopTraceError(
                        "repair-aware transition is missing diagnostics"
                    )
                if str(repair.get("route")) != route:
                    raise ClosedLoopTraceError("repair-aware route mismatch")
                if str(repair.get("repair_outcome")) not in {
                    "hard_failure",
                    "accepted_noop",
                    "state_changed_no_reduction",
                    "conflict_reduced",
                    "feasible",
                }:
                    raise ClosedLoopTraceError(
                        "repair-aware transition has an invalid outcome"
                    )
            if previous_route is not None and previous_route != route:
                route_switch_count += 1
            previous_route = route
            actual = sorted(map(int, metrics.get("neighborhood", [])))
            if route == "official_adaptive":
                if action.get("mode") != "official":
                    raise ClosedLoopTraceError("official route did not use an official action")
                if controller.get("selected_candidate_id") is not None:
                    raise ClosedLoopTraceError("official route unexpectedly selected a candidate")
            else:
                requested = sorted(map(int, action.get("agents", [])))
                if action.get("mode") != "explicit_neighborhood" or requested != actual:
                    raise ClosedLoopTraceError("learned transition neighborhood mismatch")
                if int(action.get("random_seed", -1)) < 0:
                    raise ClosedLoopTraceError("learned transition is missing explicit random seed")
                if int(metrics.get("requested_random_seed", -1)) != int(action["random_seed"]):
                    raise ClosedLoopTraceError("learned transition repair seed mismatch")
                selected_id = str(controller.get("selected_candidate_id", ""))
                matching = [
                    candidate
                    for candidate in controller.get("candidate_pool", [])
                    if str(candidate.get("candidate_id")) == selected_id
                ]
                if len(matching) != 1 or sorted(
                    map(int, matching[0].get("agents", []))
                ) != requested:
                    raise ClosedLoopTraceError("learned transition selected candidate mismatch")
        conflicts.append(after_conflicts)
        state = after

    summary = finish.get("summary")
    if not isinstance(summary, dict):
        raise ClosedLoopTraceError("finish event is missing summary")
    final_hash = state_fingerprint(state)
    if str(finish.get("final_fingerprint")) != final_hash:
        raise ClosedLoopTraceError("finish state fingerprint mismatch")
    if str(summary.get("initial_fingerprint")) != initial_hash:
        raise ClosedLoopTraceError("summary initial fingerprint mismatch")
    expected_values = {
        "initial_conflicts": conflicts[0],
        "final_conflicts": conflicts[-1],
        "repair_iterations": len(conflicts) - 1,
        "conflict_trajectory": conflicts,
        "final_sum_of_costs": int(state.get("sum_of_costs", -1)),
        "final_low_level": state.get("low_level"),
    }
    for name, value in expected_values.items():
        if summary.get(name) != value:
            raise ClosedLoopTraceError(f"summary {name} mismatch")
    if summary.get("transition_elapsed_seconds") is not None and summary.get(
        "transition_elapsed_seconds"
    ) != transition_elapsed_seconds:
        raise ClosedLoopTraceError("summary transition_elapsed_seconds mismatch")
    if (
        summary.get("balanced_controller") is not None
        or summary.get("stall_guard") is not None
        or summary.get("repair_aware") is not None
    ):
        expected_routes = {
            "model_decision_count": int(route_counts["model"]),
            "official_decision_count": int(route_counts["official_adaptive"]),
            "route_switch_count": route_switch_count,
        }
        for name, value in expected_routes.items():
            if int(summary.get(name, -1)) != value:
                raise ClosedLoopTraceError(f"summary {name} mismatch")
        total_routes = sum(expected_routes[name] for name in (
            "model_decision_count", "official_decision_count"
        ))
        expected_fraction = (
            expected_routes["model_decision_count"] / total_routes
            if total_routes
            else 0.0
        )
        if not math.isclose(
            float(summary.get("model_route_fraction", -1.0)), expected_fraction
        ):
            raise ClosedLoopTraceError("summary model_route_fraction mismatch")
    raw_auc = sum(
        (conflicts[index] + conflicts[index + 1]) / 2.0
        for index in range(len(conflicts) - 1)
    )
    if not math.isclose(float(summary.get("conflict_auc", -1.0)), raw_auc):
        raise ClosedLoopTraceError("summary conflict AUC mismatch")
    if metric_iteration_budget is not None:
        expected_fixed_auc = fixed_budget_conflict_auc(
            conflicts,
            metric_iteration_budget,
            success=bool(summary.get("success")),
        )
        if not math.isclose(
            float(summary.get("fixed_budget_conflict_auc", -1.0)), expected_fixed_auc
        ):
            raise ClosedLoopTraceError("summary fixed-budget conflict AUC mismatch")
        normalized_fixed = summary.get("normalized_fixed_budget_conflict_auc")
        if normalized_fixed is not None and conflicts[0] > 0:
            expected_normalized_fixed = expected_fixed_auc / (
                float(conflicts[0]) * metric_iteration_budget
            )
            if not math.isclose(
                float(normalized_fixed),
                expected_normalized_fixed,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ClosedLoopTraceError(
                    "summary normalized fixed-budget conflict AUC mismatch"
                )
    wall_budget = summary.get("wall_time_budget_seconds")
    if wall_budget is not None and summary.get("wall_clock_conflict_auc") is not None:
        expected_wall_auc = wall_clock_conflict_auc(
            conflicts, transition_elapsed_seconds, float(wall_budget)
        )
        if not math.isclose(
            float(summary["wall_clock_conflict_auc"]),
            expected_wall_auc,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ClosedLoopTraceError("summary wall-clock conflict AUC mismatch")
    if bool(finish.get("success")) != bool(summary.get("success")):
        raise ClosedLoopTraceError("finish success flag mismatch")
    return {
        "events": rows,
        "summary": summary,
        "trace_format": trace_format,
        "initial_state_ref": initial_state_ref,
        "event_count": len(rows),
    }


def _valid_episode_trace(
    path: Path,
    run_fingerprint: str,
    *,
    expected_episode_id: str,
    expected_policy: str,
    expected_solver_seed: int,
    metric_iteration_budget: int | None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        validated = validate_closed_loop_trace(
            path,
            run_fingerprint,
            expected_episode_id=expected_episode_id,
            expected_policy=expected_policy,
            expected_solver_seed=expected_solver_seed,
            metric_iteration_budget=metric_iteration_budget,
        )
    except ClosedLoopTraceError:
        return None
    return validated


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
    controller_mode = str(job.get("controller", "v1-full"))
    feature_backend = str(job.get("feature_backend", "auto"))
    requested_controller_runtime = str(job.get("controller_runtime", "reference"))
    if requested_controller_runtime not in CONTROLLER_RUNTIMES:
        raise ValueError(
            f"unsupported controller runtime: {requested_controller_runtime}"
        )
    verification_profile = str(job.get("verification_profile", "audit"))
    if verification_profile not in VERIFICATION_PROFILES:
        raise ValueError(f"unsupported verification profile: {verification_profile}")
    stall_guard_config: StallGuardConfig | None = None
    repair_aware_config: RepairAwareConfig | None = None
    repair_aware_bundle: RepairAwareBundle | None = None
    if controller_mode == "v2-stall-safe":
        raw_stall_guard = job.get("stall_guard_config")
        if raw_stall_guard is None:
            raise ValueError("v2-stall-safe requires a frozen stall guard config")
        stall_guard_config = load_stall_guard_config(raw_stall_guard)
    if controller_mode == "v2-repair-aware":
        raw_repair_aware = job.get("repair_aware_config")
        raw_repair_bundle = job.get("repair_aware_bundle")
        if raw_repair_aware is None or raw_repair_bundle is None:
            raise ValueError(
                "v2-repair-aware requires frozen config and auxiliary bundle"
            )
        repair_aware_config = load_repair_aware_config(raw_repair_aware)
        repair_aware_bundle = load_repair_aware_bundle(raw_repair_bundle)
    runtime_models: dict[str, Any] = {}
    runtime_ranges: dict[str, dict[str, tuple[float, float]]] = {}
    shadow_models: dict[str, Any] = {}
    if policy in LEARNED_POLICIES:
        bundle = load_frozen_policy_bundle(job["frozen_models"], job["model_registration"])
        if controller_mode == "v1-full":
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
                if repair_aware_bundle is not None and str(
                    repair_aware_bundle.manifest.get(
                        "main_ranker_semantic_fingerprint", ""
                    )
                ) != str(
                    compact_bundle.manifest.get(
                        "main_ranker_semantic_fingerprint", ""
                    )
                ):
                    raise ValueError(
                        "repair-aware bundle was trained for a different v2 ranker"
                    )
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
            and controller_mode != "v1-full"
            and not optimized_runtime_available
        ):
            raise RuntimeError(
                "optimized controller runtime requires compact proposals and dense native features"
            )
        controller_runtime = (
            "optimized"
            if policy in LEARNED_POLICIES
            and controller_mode != "v1-full"
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
                if policy in LEARNED_POLICIES and controller_mode == "v1-full"
                else None
            )
            required_model_features = (
                set(runtime_models[policy].base_feature_names)
                if policy in LEARNED_POLICIES
                else set()
            )
            if repair_aware_bundle is not None and policy == "realized_dynamic":
                for auxiliary_model in getattr(
                    repair_aware_bundle, "models", {}
                ).values():
                    required_model_features.update(auxiliary_model.feature_names)
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
                and controller_mode != "v1-full"
                else None
            )
            pending_changed_agents: set[int] = set()
            previous_route: str | None = None
            stall_guard = (
                StallGuardState(stall_guard_config)
                if stall_guard_config is not None
                and policy == "realized_dynamic"
                else None
            )
            repair_aware = (
                RepairAwareState(repair_aware_config, repair_aware_bundle)
                if repair_aware_config is not None
                and repair_aware_bundle is not None
                and policy == "realized_dynamic"
                else None
            )
            repair_aware_cache: dict[str, Any] | None = None
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
                before_hash = state_fingerprint(before)
                before_repair_hash = (
                    repair_structure_fingerprint(before)
                    if repair_aware is not None
                    else before_hash
                )
                before_fingerprint_seconds = (
                    time.perf_counter() - before_fingerprint_started
                )
                decision_index = len(conflicts) - 1
                controller: dict[str, Any] = {}
                route = "model" if policy in LEARNED_POLICIES else "official_adaptive"
                route_started = time.perf_counter()
                pre_step_orchestration_seconds = route_started - iteration_started
                if route == "official_adaptive":
                    guard_seconds = 0.0
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
                            "stall_guard_seconds": guard_seconds,
                        }
                    )
                else:
                    proposal_started = time.perf_counter()
                    cache_key = (
                        before_repair_hash,
                        _fingerprint(job["proposal"]),
                        int(solver_seed),
                    )
                    refresh_cache = (
                        repair_aware.consume_refresh()
                        if repair_aware is not None
                        else False
                    )
                    cache_hit = bool(
                        repair_aware is not None
                        and repair_aware_cache is not None
                        and repair_aware_cache.get("key") == cache_key
                        and not refresh_cache
                    )
                    state_feature_metrics: dict[str, Any] = {}
                    proposal_feature_metrics = {"proposal_feature_seconds": 0.0}
                    realized_feature_metrics = {"realized_feature_seconds": 0.0}
                    proposal_rows: list[dict[str, Any]] | None = None
                    state_analysis_seconds = 0.0
                    repair_aware_seconds = 0.0
                    repair_predictions: dict[str, list[float]] | None = None
                    if cache_hit:
                        assert repair_aware_cache is not None
                        candidates = repair_aware_cache["candidates"]
                        candidate_rows = repair_aware_cache["candidate_rows"]
                        scores = repair_aware_cache["scores"]
                        margin = float(repair_aware_cache["margin"])
                        selected_local_index = int(
                            repair_aware_cache["base_selected_local_index"]
                        )
                        repair_predictions = repair_aware_cache["repair_predictions"]
                        proposal_metrics = {
                            **dict(repair_aware_cache["proposal_metrics"]),
                            "proposal_seconds": 0.0,
                            "candidate_generation_seconds": 0.0,
                            "state_check_seconds": 0.0,
                            "state_check_fingerprint_seconds": 0.0,
                            "backend": "repair-aware-cache",
                            "state_check_backend": "cached-state-fingerprint",
                            "full_state_verified": False,
                            "repair_aware_cache_hit": True,
                        }
                        feature_seconds = 0.0
                        inference_seconds = 0.0
                        repair_aware.note_cache_hit()
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
                            proposal_config=job["proposal"],
                            state_hash=before_hash,
                            verify_full_state=verify_full_state,
                            proposal_backend=controller_runtime,
                            shadow_validation=bool(
                                job.get("proposal_shadow_validation", False)
                                and optimized_runtime_available
                            ),
                        )
                        proposal_metrics["repair_aware_cache_hit"] = False
                        if controller_mode == "v1-full":
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
                        inference_started = time.perf_counter()
                        selected_local_index, scores, margin = score_online_candidates(
                            candidate_rows, runtime_models[policy]
                        )
                        inference_seconds = time.perf_counter() - inference_started
                        if repair_aware is not None:
                            repair_aware_cache = {
                                "key": cache_key,
                                "candidates": candidates,
                                "candidate_rows": candidate_rows,
                                "scores": scores,
                                "margin": margin,
                                "base_selected_local_index": selected_local_index,
                                "repair_predictions": repair_predictions,
                                "proposal_metrics": dict(proposal_metrics),
                                "generation_decision_index": decision_index,
                                "lazy_augmented": False,
                            }
                    if (
                        repair_aware is not None
                        and repair_aware.needs_rescue
                        and repair_aware.config.lazy_neighborhood_sizes
                        and bool(
                            repair_aware.bundle.manifest.get(
                                "size12_promoted_offline", False
                            )
                        )
                    ):
                        if repair_aware_cache is None:
                            raise ClosedLoopExecutionError(
                                "repair_aware_cache_missing",
                                "lazy rescue generation has no state cache",
                            )
                        if not bool(repair_aware_cache.get("lazy_augmented")):
                            lazy_started = time.perf_counter()
                            lazy_config = dict(job["proposal"])
                            lazy_config["neighborhood_sizes"] = list(
                                repair_aware.config.lazy_neighborhood_sizes
                            )
                            lazy_candidates, lazy_metrics = generate_online_candidates(
                                environment,
                                state,
                                task_id=str(row["task_id"]),
                                solver_seed=solver_seed,
                                decision_index=int(
                                    repair_aware_cache["generation_decision_index"]
                                ),
                                proposal_config=lazy_config,
                                state_hash=before_hash,
                                verify_full_state=False,
                                proposal_backend=controller_runtime,
                                shadow_validation=False,
                            )
                            existing_neighborhoods = {
                                tuple(map(int, candidate["agents"]))
                                for candidate in candidates
                            }
                            extra_candidates = [
                                candidate
                                for candidate in lazy_candidates
                                if tuple(map(int, candidate["agents"]))
                                not in existing_neighborhoods
                            ]
                            if extra_candidates:
                                if feature_engine is None:
                                    feature_engine = make_feature_engine(state)
                                extra_rows, extra_feature_metrics = (
                                    feature_engine.realized_rows(
                                        extra_candidates, state_hash=before_hash
                                    )
                                )
                                candidates = list(candidates) + extra_candidates
                                candidate_rows = list(candidate_rows) + extra_rows
                                _ignored_index, scores, _ignored_margin = (
                                    score_online_candidates(
                                        candidate_rows, runtime_models[policy]
                                    )
                                )
                                repair_aware_cache.update(
                                    {
                                        "candidates": candidates,
                                        "candidate_rows": candidate_rows,
                                        "scores": scores,
                                        "repair_predictions": None,
                                    }
                                )
                                repair_predictions = None
                                realized_feature_metrics["realized_feature_seconds"] = float(
                                    realized_feature_metrics.get(
                                        "realized_feature_seconds", 0.0
                                    )
                                ) + float(
                                    extra_feature_metrics.get(
                                        "realized_feature_seconds", 0.0
                                    )
                                )
                            repair_aware_cache["lazy_augmented"] = True
                            repair_aware_cache["lazy_candidate_count"] = len(
                                extra_candidates
                            )
                            repair_aware_cache["lazy_generation_seconds"] = (
                                time.perf_counter() - lazy_started
                            )
                            proposal_metrics = {
                                **proposal_metrics,
                                "lazy_neighborhood_sizes": list(
                                    repair_aware.config.lazy_neighborhood_sizes
                                ),
                                "lazy_proposal_count": int(
                                    lazy_metrics.get("proposal_count", 0)
                                ),
                                "lazy_candidate_count": len(extra_candidates),
                                "lazy_generation_seconds": float(
                                    repair_aware_cache["lazy_generation_seconds"]
                                ),
                            }
                    controller["proposal"] = proposal_metrics
                    pruning_metrics = no_pruning_metrics(len(candidates))
                    retained_indices = list(range(len(candidates)))
                    retained_candidates = [
                        candidates[index] for index in retained_indices
                    ]
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
                    guard_seconds = 0.0
                    if stall_guard is not None:
                        guard_started = time.perf_counter()
                        guard_selected_index, guard_diagnostic = stall_guard.select(
                            retained_candidates,
                            scores,
                            before_fingerprint=before_hash,
                        )
                        guard_seconds = time.perf_counter() - guard_started
                        selected_local_index = guard_selected_index
                        controller["stall_guard"] = guard_diagnostic
                        controller_totals["stall_guard_seconds"] += guard_seconds
                    if repair_aware is not None:
                        if repair_aware.predictions_required(before_repair_hash):
                            if repair_predictions is None:
                                repair_prediction_started = time.perf_counter()
                                assert repair_aware_bundle is not None
                                repair_predictions = repair_aware_bundle.predict(
                                    candidate_rows
                                )
                                repair_aware_seconds += (
                                    time.perf_counter() - repair_prediction_started
                                )
                                if repair_aware_cache is None:
                                    raise ClosedLoopExecutionError(
                                        "repair_aware_cache_missing",
                                        "repair-aware prediction has no state cache",
                                    )
                                repair_aware_cache["repair_predictions"] = (
                                    repair_predictions
                                )
                            adaptive_prediction_started = time.perf_counter()
                            assert repair_aware_bundle is not None
                            adaptive_prediction = {
                                name: values[0]
                                for name, values in repair_aware_bundle.predict(
                                    [adaptive_feature_row(candidate_rows[0])]
                                ).items()
                            }
                            repair_aware_seconds += (
                                time.perf_counter() - adaptive_prediction_started
                            )
                        else:
                            adaptive_prediction = None
                        repair_select_started = time.perf_counter()
                        repair_selected_index, repair_diagnostic = repair_aware.select(
                            candidates,
                            scores,
                            base_selected_local_index,
                            repair_predictions,
                            before_fingerprint=before_repair_hash,
                            adaptive_prediction=adaptive_prediction,
                        )
                        repair_aware_seconds += (
                            time.perf_counter() - repair_select_started
                        )
                        selected_local_index = repair_selected_index
                        controller["repair_aware"] = repair_diagnostic
                        controller_totals["repair_aware_seconds"] += (
                            repair_aware_seconds
                        )
                    base_diagnostic = feature_range_diagnostic(
                        candidate_rows[base_selected_local_index],
                        policy,
                        runtime_ranges[policy],
                    )
                    if selected_local_index is None:
                        if stall_guard is None and repair_aware is None:
                            raise ClosedLoopExecutionError(
                                "controller_no_candidate",
                                "controller did not select a candidate",
                            )
                        route = "official_adaptive"
                        selected_index = None
                        selected = None
                        diagnostic = None
                        fallback_anchor = (
                            str(guard_diagnostic["state_anchor_fingerprint"])
                            if stall_guard is not None
                            else str(repair_diagnostic["state_anchor_fingerprint"])
                        )
                        action = {
                            "mode": "official",
                            "random_seed": (
                                stall_guard_fallback_seed(
                                    str(row["task_id"]),
                                    solver_seed,
                                    fallback_anchor,
                                    decision_index,
                                )
                                if stall_guard is not None
                                else repair_aware_fallback_seed(
                                    str(row["task_id"]),
                                    solver_seed,
                                    fallback_anchor,
                                    decision_index,
                                )
                            ),
                        }
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
                                    if local_index is not None
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
                        + guard_seconds
                        + repair_aware_seconds,
                    )
                    measured_selection_stages = (
                        candidate_generation_seconds
                        + state_check_seconds
                        + feature_seconds
                        + float(pruning_metrics["pruner_seconds"])
                        + inference_seconds
                        + guard_seconds
                        + repair_aware_seconds
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
                            "inference_backend": getattr(
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
                                if selected_local_index is not None
                                else None
                            ),
                            "base_selected_candidate_id": candidates[
                                retained_indices[base_selected_local_index]
                            ]["candidate_id"],
                            "base_selected_score": scores[base_selected_local_index],
                            "base_selected_feature_range": base_diagnostic,
                            "score_margin": margin,
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
                            "stall_guard_seconds": guard_seconds,
                            "repair_aware_seconds": repair_aware_seconds,
                            "repair_aware_cache_hit": cache_hit,
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
                if (
                    (stall_guard is not None or repair_aware is not None)
                    and policy == "realized_dynamic"
                ):
                    route_prefix = "official" if route == "official_adaptive" else "model"
                    controller_totals[f"{route_prefix}_decision_count"] += 1
                    controller_totals[f"{route_prefix}_controller_seconds"] += float(
                        controller.get("controller_seconds_before_repair", 0.0)
                    )
                    if route == "official_adaptive" and "candidate_pool" not in controller:
                        controller_totals["controller_seconds_before_repair"] += float(
                            controller.get("controller_seconds_before_repair", 0.0)
                        )
                    if previous_route is not None and previous_route != route:
                        controller_totals["route_switch_count"] += 1
                    previous_route = route
                repair_started = time.perf_counter()
                result = _plain(environment.step(action))
                repair_wall_seconds = time.perf_counter() - repair_started
                post_step_started = time.perf_counter()
                total_repair_wall_seconds += repair_wall_seconds
                state = result["observation"]
                metrics = result["metrics"]
                required_native_timing_keys = {
                    "native_step_seconds",
                    "native_neighborhood_generation_seconds",
                    "native_replan_seconds",
                    "pp_replan_seconds",
                    "native_state_snapshot_seconds",
                    "native_repair_bookkeeping_seconds",
                    "native_residual_seconds",
                    "binding_solver_call_seconds",
                    "binding_state_snapshot_seconds",
                    "state_to_python_seconds",
                    "metrics_to_python_seconds",
                    "binding_total_seconds",
                    "binding_residual_seconds",
                }
                native_timing_available = required_native_timing_keys.issubset(metrics)
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
                if (
                    (stall_guard is not None or repair_aware is not None)
                    and policy == "realized_dynamic"
                ):
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
                if (
                    route == "official_adaptive"
                    and stall_guard is None
                    and repair_aware is None
                ):
                    controller_totals["controller_seconds_before_repair"] += (
                        controller_before_repair_seconds
                    )

                after_fingerprint_started = time.perf_counter()
                after_hash = state_fingerprint(state)
                after_repair_hash = (
                    repair_structure_fingerprint(state)
                    if repair_aware is not None
                    else after_hash
                )
                state_fingerprint_seconds = before_fingerprint_seconds + (
                    time.perf_counter() - after_fingerprint_started
                ) + float(controller.get("state_check_fingerprint_seconds", 0.0))
                if stall_guard is not None:
                    guard_observe_started = time.perf_counter()
                    actual_agent_ids = set(map(int, actual))
                    before_selected_paths = {
                        int(agent["id"]): agent["path"]
                        for agent in before["agents"]
                        if int(agent["id"]) in actual_agent_ids
                    }
                    after_selected_paths = {
                        int(agent["id"]): agent["path"]
                        for agent in state["agents"]
                        if int(agent["id"]) in actual_agent_ids
                    }
                    if set(before_selected_paths) != actual_agent_ids or set(
                        after_selected_paths
                    ) != actual_agent_ids:
                        raise RuntimeError(
                            "stall guard could not compare every repaired agent path"
                        )
                    controller["stall_guard"] = stall_guard.observe(
                        after_fingerprint=after_hash,
                        replan_success=bool(metrics.get("replan_success")),
                        paths_changed=before_selected_paths != after_selected_paths,
                        conflict_graph_changed=before["conflict_edges"]
                        != state["conflict_edges"],
                        sum_of_costs_changed=int(before["sum_of_costs"])
                        != int(state["sum_of_costs"]),
                        actual_neighborhood_size=len(actual),
                    )
                    guard_observe_seconds = (
                        time.perf_counter() - guard_observe_started
                    )
                    controller["stall_guard_seconds"] = float(
                        controller.get("stall_guard_seconds", 0.0)
                    ) + guard_observe_seconds
                    controller_totals["stall_guard_seconds"] += (
                        guard_observe_seconds
                    )
                if repair_aware is not None:
                    repair_observe_started = time.perf_counter()
                    observed = repair_aware.observe(
                        before_fingerprint=before_repair_hash,
                        after_fingerprint=after_repair_hash,
                        replan_success=bool(metrics.get("replan_success")),
                        conflicts_before=int(before["num_of_colliding_pairs"]),
                        conflicts_after=int(state["num_of_colliding_pairs"]),
                        feasible=bool(state.get("feasible")),
                    )
                    observe_seconds = time.perf_counter() - repair_observe_started
                    controller["repair_aware"] = {
                        **dict(controller.get("repair_aware") or {}),
                        **observed,
                    }
                    controller["repair_aware_seconds"] = float(
                        controller.get("repair_aware_seconds", 0.0)
                    ) + observe_seconds
                    controller_totals["repair_aware_seconds"] += observe_seconds
                transition_timings = {
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
                    "stall_guard_seconds": float(
                        controller.get("stall_guard_seconds", 0.0)
                    ),
                    "repair_aware_seconds": float(
                        controller.get("repair_aware_seconds", 0.0)
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
                    "native_timing_schema": (
                        REPAIR_TIMING_SCHEMA if native_timing_available else None
                    ),
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
                (stall_guard is not None or repair_aware is not None)
                and policy == "realized_dynamic"
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
                    if policy in LEARNED_POLICIES and controller_mode == "v1-full"
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
                "stall_guard": (
                    stall_guard.summary()
                    if stall_guard is not None and policy == "realized_dynamic"
                    else None
                ),
                "repair_aware": (
                    repair_aware.summary()
                    if repair_aware is not None and policy == "realized_dynamic"
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
            final_fingerprint = state_fingerprint(state)
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
) -> dict[str, Any]:
    result = {**config, "environment": dict(config["environment"])}
    if wall_time_budget_seconds is None:
        if episode_process_timeout_seconds is not None:
            raise ValueError(
                "episode process timeout override requires a wall-time budget override"
            )
        return result
    wall_budget = float(wall_time_budget_seconds)
    if not math.isfinite(wall_budget) or wall_budget <= 0.0:
        raise ValueError("wall-time budget override must be finite and positive")
    process_timeout = float(
        episode_process_timeout_seconds
        if episode_process_timeout_seconds is not None
        else wall_budget + 60.0
    )
    if not math.isfinite(process_timeout) or process_timeout <= wall_budget:
        raise ValueError(
            "episode process timeout must be finite and greater than the wall-time budget"
        )
    result["wall_time_budget_seconds"] = wall_budget
    result["episode_process_timeout_seconds"] = process_timeout
    result["environment"]["time_limit"] = wall_budget
    return result


def _with_stopping_rule(
    config: dict[str, Any], stopping_rule: str
) -> dict[str, Any]:
    if stopping_rule not in STOPPING_RULES:
        raise ValueError(f"unsupported stopping rule: {stopping_rule}")
    result = {**config, "environment": dict(config["environment"])}
    result["stopping_rule"] = stopping_rule
    if stopping_rule == "wall-clock":
        result["max_decisions"] = 0
        result["metric_iteration_budget"] = None
        result["environment"]["max_repair_iterations"] = 0
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
    stall_guard_config: str | Path | dict[str, Any] | None = None,
    repair_aware_config: str | Path | dict[str, Any] | None = None,
    repair_aware_bundle: str | Path | None = None,
    job_keys: set[tuple[str, int]] | None = None,
    cohort_job_keys: set[tuple[str, int]] | None = None,
    wall_time_budget_seconds: float | None = None,
    episode_process_timeout_seconds: float | None = None,
    stopping_rule: str = "historical",
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported closed-loop config")
    config = _with_time_budget_overrides(
        config, wall_time_budget_seconds, episode_process_timeout_seconds
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
    stall_guard_payload: dict[str, Any] | None = None
    repair_aware_payload: dict[str, Any] | None = None
    repair_aware_root: Path | None = None
    repair_aware_manifest: dict[str, Any] | None = None
    if controller_mode == "v2-stall-safe":
        if stall_guard_config is None:
            raise ValueError("v2-stall-safe requires --stall-guard-config")
        loaded_stall_guard = load_stall_guard_config(stall_guard_config)
        proposal_sizes = set(map(int, config["proposal"]["neighborhood_sizes"]))
        if not set(loaded_stall_guard.size_caps) <= proposal_sizes:
            raise ValueError("stall guard size caps are absent from the proposal config")
        stall_guard_payload = loaded_stall_guard.payload()
    elif stall_guard_config is not None:
        raise ValueError("stall_guard_config is only valid with v2-stall-safe")
    if controller_mode == "v2-repair-aware":
        if repair_aware_config is None:
            raise ValueError("v2-repair-aware requires --repair-aware-config")
        repair_aware_payload = load_repair_aware_config(
            repair_aware_config
        ).payload()
        repair_aware_root = Path(
            str(repair_aware_bundle or DEFAULT_REPAIR_AWARE_BUNDLE)
        )
        if not repair_aware_root.is_absolute():
            repair_aware_root = project_root / repair_aware_root
        repair_aware_root = repair_aware_root.resolve()
        repair_aware_manifest = load_repair_aware_bundle(
            repair_aware_root
        ).manifest
        if controller_manifest is None or str(
            repair_aware_manifest.get("main_ranker_semantic_fingerprint", "")
        ) != str(
            controller_manifest.get("main_ranker_semantic_fingerprint", "")
        ):
            raise ValueError(
                "repair-aware bundle does not match the selected v2 controller bundle"
            )
    elif repair_aware_config is not None or repair_aware_bundle is not None:
        raise ValueError(
            "repair-aware config/bundle are only valid with v2-repair-aware"
        )
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
    if str(config.get("dataset_design", {}).get("mode", "structured")) == "movingai_ood":
        registered_ids = set(map(str, config["dataset_design"].get("historical_map_ids", [])))
        current_ids = {str(row["map_id"]) for row in all_rows}
        overlap = sorted(current_ids & registered_ids)
        isolation = {
            "passed": not overlap,
            "mode": "movingai_map_id",
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
        if actual_source_manifest != expected_source_manifest:
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
        "controller_bundle": str(controller_root),
        "feature_shadow_validation": bool(
            feature_shadow_validation
            or verification_profile == "audit"
            and controller_runtime in {"optimized", "auto"}
        ),
        "stall_guard_config": stall_guard_payload,
        "repair_aware_config": repair_aware_payload,
        "repair_aware_bundle": (
            str(repair_aware_root) if repair_aware_root is not None else None
        ),
    }
    config_fp = _fingerprint(effective)
    run_fp = _fingerprint(
        {
            "dataset_fingerprint": dataset_fp,
            "configuration_fingerprint": config_fp,
            "freeze_manifest": bundle.manifest,
            "controller_bundle_manifest": controller_manifest,
            "repair_aware_bundle_manifest": repair_aware_manifest,
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
                FEATURE_SCHEMA_ID if controller_mode != "v1-full" else None
            ),
            "feature_schema_sha256": (
                FEATURE_SCHEMA_SHA256 if controller_mode != "v1-full" else None
            ),
            "controller_bundle": controller_manifest,
            "stall_guard_config": stall_guard_payload,
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
            FEATURE_SCHEMA_ID if controller_mode != "v1-full" else None
        ),
        "feature_schema_sha256": (
            FEATURE_SCHEMA_SHA256 if controller_mode != "v1-full" else None
        ),
        "controller_bundle": controller_manifest,
        "stall_guard_config": stall_guard_payload,
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
            with _CollectionRunLock(output_root, run_fp, "closed-loop-qualification"):
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
                "stall_guard_config": stall_guard_payload,
                "repair_aware_config": repair_aware_payload,
                "repair_aware_bundle": (
                    str(repair_aware_root) if repair_aware_root is not None else None
                ),
                "proposal_state_verification": (
                    "always" if verification_profile == "audit" else "sampled"
                ),
                "resume": resume,
                "require_finalization_timings": True,
                "existing_manifest_row": existing_by_key.get(
                    (str(row["task_id"]), int(solver_seed))
                ),
            }
            for row in rows
            for solver_seed in solver_seeds
            if normalized_job_keys is None
            or (str(row["task_id"]), int(solver_seed)) in normalized_job_keys
        ]
        with _CollectionRunLock(output_root, run_fp, f"closed-loop-{current}"):
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
    "repair_random_seed",
    "resolve_controller_mode",
    "run_closed_loop_collection",
    "score_online_candidates",
    "verify_portable_policy_bundle",
]
