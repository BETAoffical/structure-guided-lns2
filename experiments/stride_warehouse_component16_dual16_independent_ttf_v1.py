"""Independent three-arm TTF test on authenticated very-high-load checkpoints.

The source checkpoint cohort predates the H1 label cohort and is byte-level
map-disjoint from it.  Timed lanes are executed one at a time with a key-mod-3
rotating order.  This module produces development evidence only.
"""
from __future__ import annotations

import collections
import importlib
from pathlib import Path
from typing import Any, Mapping

from experiments._common import (
    contained_file,
    read_json,
    read_jsonl,
    registered_input,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.stride_warehouse_disruption_recovery_ttf import (
    TTF_OVERRIDE_SCHEMA,
    _common_kwargs,
    _controller_summary,
    _lane_root,
)
from experiments.warehouse_disruption_checkpoints import (
    compute_checkpoint_identity_sha256,
)
from lns2_selector.runtime.structshell_dual16 import (
    structshell_dual16_augmentation,
    validate_structshell_dual16_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.warehouse_component16_dual16_independent_ttf_config.v1"
EXPERIMENT_ID = "stride-warehouse-component16-dual16-independent-ttf-v1"
PREFLIGHT_SCHEMA = "lns2.stride.warehouse_component16_dual16_independent_ttf_preflight.v1"
REPORT_SCHEMA = "lns2.stride.warehouse_component16_dual16_independent_ttf_report.v1"
SCIENTIFIC_STATUS = "development_independent_map_disjoint_ttf_test_not_default_promotion"
CONTROLLERS = ("official_adaptive", "component16", "dual16")
INPUT_NAMES = (
    "source_experiment_config",
    "source_dataset_manifest",
    "source_checkpoint_manifest",
    "source_checkpoint_report",
    "source_runtime_config",
    "source_qualification_manifest",
    "h1_source_manifest",
    "frozen_v2_manifest",
)
TIMED_MANIFESTS = {
    "official_adaptive": "official_adaptive_manifest.jsonl",
    "component16": "realized_dynamic_manifest.jsonl",
    "dual16": "realized_dynamic_manifest.jsonl",
}
REPORT_FILENAME = "independent_ttf_report.json"

EXPECTED_COHORT = {
    "split": "confirmation",
    "load_band": "very_high",
    "task_variant": "balanced_od_d15",
    "expected_map_count": 8,
    "expected_task_count": 8,
    "expected_checkpoint_count": 16,
    "checkpoints_per_map": 2,
    "h1_expected_map_count": 6,
    "h1_expected_checkpoint_count": 36,
    "require_map_sha_disjoint_from_h1": True,
}
EXPECTED_CONTROLLER_CONTRACT = {
    "v2_bundle": "artifacts/initlns-closed-loop-controller-v2",
    "component16": "stride-structshell-component16-v1",
    "dual16": "stride-structshell-dual16-v1",
    "frozen_v2": True,
    "no_retraining": True,
    "no_default_change": True,
}
EXPECTED_RUNTIME = {
    "wall_time_budget_seconds": 60.0,
    "environment_time_limit_seconds": 60.0,
    "episode_process_timeout_seconds": 90.0,
    "qualification_process_timeout_seconds": 90.0,
    "qualification_workers": 8,
    "timed_workers": 1,
    "execution_order": "key_mod_3_rotating_strict_three_controller_serial",
    "timing_boundary": "checkpoint_restore_inclusive_ttf",
    "repair_seed_policy": "episode_stream",
    "rebuild_local_qualification_anchor_with_current_native": True,
    "rebind_authenticated_checkpoint_rows_from_source": True,
}
EXPECTED_REPORTING = {
    "primary_baseline": "official_adaptive",
    "pairwise_targets": ["component16", "dual16"],
    "include_component16_vs_dual16": True,
    "minimum_paired_win_rate": 0.7,
    "minimum_mean_capped_ttf_improvement": 0.15,
    "minimum_successes_per_hour_improvement": 0.2,
    "maximum_success_rate_loss": 0.0,
    "no_additional_timeout_or_censor": True,
    "gate_is_diagnostic_only": True,
    "formal_promotion_allowed": False,
    "global_claim_allowed": False,
    "default_replacement_allowed": False,
}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("independent TTF config must be an object")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status") != SCIENTIFIC_STATUS
    ):
        raise ValueError("independent TTF config identity changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_NAMES):
        raise ValueError("independent TTF registered-input set changed")
    for name in INPUT_NAMES:
        spec = inputs.get(name)
        if not isinstance(spec, Mapping) or set(spec) != {"path", "sha256"}:
            raise ValueError(f"registered input contract changed: {name}")
        path, digest = spec.get("path"), spec.get("sha256")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError(f"registered input path is invalid: {name}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"registered input SHA-256 is invalid: {name}")
    if dict(config.get("cohort") or {}) != EXPECTED_COHORT:
        raise ValueError("independent TTF cohort contract changed")
    if tuple(config.get("controllers") or ()) != CONTROLLERS:
        raise ValueError("independent TTF controller set or order changed")
    if dict(config.get("controller_contract") or {}) != EXPECTED_CONTROLLER_CONTRACT:
        raise ValueError("independent TTF controller contract changed")
    if dict(config.get("runtime") or {}) != EXPECTED_RUNTIME:
        raise ValueError("independent TTF runtime contract changed")
    if dict(config.get("reporting") or {}) != EXPECTED_REPORTING:
        raise ValueError("independent TTF reporting boundary changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    root = _root()
    config = read_json(config_path)
    validate_config(config)
    registered = {
        name: registered_input(root, dict(config["inputs"][name]), label=name)
        for name in INPUT_NAMES
    }
    source_manifest = registered["source_dataset_manifest"]
    checkpoint_manifest = registered["source_checkpoint_manifest"]
    frozen_v2_manifest = registered["frozen_v2_manifest"]
    if frozen_v2_manifest.parent.resolve() != (
        root / str(config["controller_contract"]["v2_bundle"])
    ).resolve():
        raise ValueError("frozen V2 bundle path changed")
    config["_registered_inputs"] = {
        name: str(value) for name, value in registered.items()
    }
    config["_dataset_root"] = str(source_manifest.parent.parent.resolve())
    config["_checkpoint_root"] = str(checkpoint_manifest.parent.resolve())
    config["_runtime_config"] = str(registered["source_runtime_config"])
    config["_v2_bundle"] = str(frozen_v2_manifest.parent.resolve())
    return config_path, root, config


def _registered(config: Mapping[str, Any], name: str) -> Path:
    return Path(str(config["_registered_inputs"][name])).resolve()


def _validate_source_experiment(config: Mapping[str, Any]) -> None:
    source = read_json(_registered(config, "source_experiment_config"))
    dataset = dict(source.get("dataset") or {})
    generation = dict(source.get("checkpoint_generation") or {})
    if (
        source.get("schema")
        != "lns2.stride.warehouse_dual16_very_high_confirmation_config.v2"
        or source.get("experiment_id")
        != "stride-warehouse-dual16-very-high-confirmation-v2"
        or int(dataset.get("expected_map_count", -1)) != 8
        or int(dataset.get("expected_task_count", -1)) != 8
        or dataset.get("split") != "confirmation"
        or dataset.get("load_band") != "very_high"
        or int(generation.get("expected_candidate_count", -1)) != 16
        or int(generation.get("disturbance_replicas_per_task", -1)) != 2
        or generation.get("selection_blind_to_controller_outcomes") is not True
        or generation.get("no_failed_candidate_replacement") is not True
    ):
        raise ValueError("source confirmation experiment contract changed")


def _validate_runtime(config: Mapping[str, Any], checkpoints: list[dict[str, Any]]) -> None:
    runtime = read_json(_registered(config, "source_runtime_config"))
    expected_seeds = sorted(int(row["screen_solver_seed"]) for row in checkpoints)
    environment = dict(runtime.get("environment") or {})
    if (
        runtime.get("split") != "confirmation"
        or sorted(map(int, runtime.get("solver_seeds") or ())) != expected_seeds
        or tuple(runtime.get("policies") or ())
        != ("official_adaptive", "realized_dynamic")
        or float(runtime.get("wall_time_budget_seconds", -1)) != 60.0
        or float(runtime.get("episode_process_timeout_seconds", -1)) != 90.0
        or int(runtime.get("workers", -1)) != 1
        or float(environment.get("time_limit", -1)) != 60.0
        or runtime.get("repair_seed_policy") != "episode_stream"
        or runtime.get("deterministic_pp_replay") is not False
    ):
        raise ValueError("registered source runtime is not the frozen 60-second contract")


def _validate_dataset_and_checkpoints(
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    set[str],
    set[str],
    set[str],
    set[str],
    set[str],
    set[str],
]:
    dataset_manifest = _registered(config, "source_dataset_manifest")
    checkpoint_manifest = _registered(config, "source_checkpoint_manifest")
    checkpoint_report = _registered(config, "source_checkpoint_report")
    h1_manifest = _registered(config, "h1_source_manifest")
    dataset_rows = [dict(row) for row in read_jsonl(dataset_manifest)]
    checkpoints = [dict(row) for row in read_jsonl(checkpoint_manifest)]
    h1_rows = [dict(row) for row in read_jsonl(h1_manifest)]
    if (
        len(dataset_rows) != 8
        or len({str(row.get("task_id")) for row in dataset_rows}) != 8
        or len({str(row.get("map_id")) for row in dataset_rows}) != 8
    ):
        raise ValueError("source dataset must contain eight unique maps and tasks")
    split_root = dataset_manifest.parent
    dataset_by_task: dict[str, dict[str, Any]] = {}
    source_map_hashes: set[str] = set()
    source_task_hashes: set[str] = set()
    for row in dataset_rows:
        task_id = str(row.get("task_id") or "")
        if (
            row.get("split") != "confirmation"
            or row.get("task_variant") != "balanced_od_d15"
            or not task_id
        ):
            raise ValueError("source dataset row contract changed")
        map_file = contained_file(split_root, row.get("map_file"), field="source map file")
        task_file = contained_file(split_root, row.get("task_file"), field="source task file")
        source_map_hashes.add(sha256_file(map_file))
        source_task_hashes.add(sha256_file(task_file))
        dataset_by_task[task_id] = row
    if len(source_map_hashes) != 8 or len(source_task_hashes) != 8:
        raise ValueError("source dataset maps/tasks are not eight byte-distinct inputs")
    report = read_json(checkpoint_report)
    source_config_sha = str(config["inputs"]["source_experiment_config"]["sha256"])
    checkpoint_manifest_sha = str(config["inputs"]["source_checkpoint_manifest"]["sha256"])
    if (
        report.get("schema")
        != "lns2.stride.warehouse_dual16_very_high_confirmation_checkpoint_report.v2"
        or report.get("passed") is not True
        or report.get("config_sha256") != source_config_sha
        or report.get("checkpoint_manifest_sha256") != checkpoint_manifest_sha
        or int(report.get("attempted_candidate_count", -1)) != 16
        or int(report.get("completed_checkpoint_count", -1)) != 16
        or int(report.get("qualified_checkpoint_count", -1)) != 16
        or int(report.get("qualified_map_count", -1)) != 8
        or list(report.get("failures") or ())
        or report.get("checkpoint_selection_controller_outcomes_consulted") is not False
        or report.get("failed_candidate_replacement") is not False
    ):
        raise ValueError("source checkpoint report did not pass its frozen contract")
    if (
        len(checkpoints) != 16
        or {int(row.get("key_index", -1)) for row in checkpoints} != set(range(16))
        or len({str(row.get("checkpoint_id")) for row in checkpoints}) != 16
    ):
        raise ValueError("source checkpoint manifest is not the frozen 16-key cohort")
    checkpoint_root = checkpoint_manifest.parent
    per_map: collections.Counter[str] = collections.Counter()
    checkpoint_ids: set[str] = set()
    source_checkpoint_hashes: set[str] = set()
    for row in checkpoints:
        task = dataset_by_task.get(str(row.get("task_id")))
        if task is None:
            raise ValueError("checkpoint task is absent from the source dataset")
        map_file = contained_file(split_root, task.get("map_file"), field="checkpoint map file")
        task_file = contained_file(split_root, task.get("task_file"), field="checkpoint task file")
        blob = contained_file(checkpoint_root, row.get("state_blob"), field="checkpoint state blob")
        checks = (
            row.get("split") == "confirmation",
            row.get("load_band") == "very_high",
            row.get("task_variant") == "balanced_od_d15",
            str(row.get("map_id")) == str(task.get("map_id")),
            int(row.get("agent_count", -1)) == int(task.get("agent_count", -2)),
            str(row.get("map_sha256")) == sha256_file(map_file),
            str(row.get("task_sha256")) == sha256_file(task_file),
            row.get("source_kind") == "checkpoint_blob_v1",
            row.get("checkpoint_selection_controller_outcomes_consulted") is False,
            row.get("controller_outcomes_consulted") is False,
            dict(row.get("qualification") or {}).get("passed") is True,
            str(row.get("checkpoint_identity_sha256"))
            == compute_checkpoint_identity_sha256(row),
            str(row.get("state_blob_sha256")) == sha256_file(blob),
        )
        if not all(checks):
            raise ValueError(f"source checkpoint identity changed: {row.get('checkpoint_id')}")
        checkpoint_ids.add(str(row["checkpoint_id"]))
        source_checkpoint_hashes.add(str(row["checkpoint_identity_sha256"]))
        per_map[str(row["map_id"])] += 1
    if per_map != collections.Counter({str(row["map_id"]): 2 for row in dataset_rows}):
        raise ValueError("source checkpoint cohort must have two replicas per map")
    if set(map(str, report.get("qualified_checkpoint_ids") or ())) != checkpoint_ids:
        raise ValueError("source checkpoint report/manifest identity mismatch")
    if (
        len(h1_rows) != 36
        or len({str(row.get("checkpoint_id")) for row in h1_rows}) != 36
        or any(dict(row.get("qualification") or {}).get("passed") is not True for row in h1_rows)
    ):
        raise ValueError("H1 source manifest is not the registered 36-checkpoint cohort")
    h1_map_hashes = {str(row.get("map_sha256") or "") for row in h1_rows}
    h1_task_hashes = {str(row.get("task_sha256") or "") for row in h1_rows}
    h1_checkpoint_hashes = {
        str(row.get("checkpoint_identity_sha256") or "") for row in h1_rows
    }
    sha_sets = (
        source_map_hashes,
        source_task_hashes,
        source_checkpoint_hashes,
        h1_map_hashes,
        h1_task_hashes,
        h1_checkpoint_hashes,
    )
    if (
        len(h1_map_hashes) != 6
        or len(h1_task_hashes) != 18
        or len(h1_checkpoint_hashes) != 36
        or len(source_checkpoint_hashes) != 16
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for values in sha_sets
            for value in values
        )
        or any(
            str(row.get("checkpoint_identity_sha256"))
            != compute_checkpoint_identity_sha256(row)
            for row in h1_rows
        )
        or source_map_hashes & h1_map_hashes
        or source_task_hashes & h1_task_hashes
        or source_checkpoint_hashes & h1_checkpoint_hashes
    ):
        raise ValueError("source maps/tasks/checkpoints are not SHA-disjoint from H1")
    qualification_rows = [
        dict(row) for row in read_jsonl(_registered(config, "source_qualification_manifest"))
    ]
    qualification_by_key = {
        (str(row.get("task_id")), int(row.get("solver_seed", -1))): row
        for row in qualification_rows
    }
    expected_keys = {
        (str(row["task_id"]), int(row["screen_solver_seed"])) for row in checkpoints
    }
    if len(qualification_rows) != 16 or set(qualification_by_key) != expected_keys:
        raise ValueError("authenticated qualification anchor key set changed")
    checkpoint_by_key = {
        (str(row["task_id"]), int(row["screen_solver_seed"])): row
        for row in checkpoints
    }
    for key, anchored in qualification_by_key.items():
        checkpoint = checkpoint_by_key[key]
        if (
            anchored.get("status") != "ok"
            or anchored.get("qualification_source_kind")
            != "authenticated_checkpoint_blob_v1"
            or str(anchored.get("checkpoint_id")) != str(checkpoint["checkpoint_id"])
            or str(anchored.get("checkpoint_identity_sha256"))
            != str(checkpoint["checkpoint_identity_sha256"])
            or str(anchored.get("state_fingerprint"))
            != str(checkpoint["expected_fingerprint"])
            or int(anchored.get("initial_conflicts", -1))
            != int(checkpoint["expected_conflicts"])
        ):
            raise ValueError("authenticated qualification anchor changed")
    checkpoints.sort(key=lambda row: int(row["key_index"]))
    return (
        checkpoints,
        source_map_hashes,
        source_task_hashes,
        source_checkpoint_hashes,
        h1_map_hashes,
        h1_task_hashes,
        h1_checkpoint_hashes,
    )


def _evidence(config_path: str | Path) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    _validate_source_experiment(config)
    (
        checkpoints,
        source_map_hashes,
        source_task_hashes,
        source_checkpoint_hashes,
        h1_map_hashes,
        h1_task_hashes,
        h1_checkpoint_hashes,
    ) = _validate_dataset_and_checkpoints(config)
    _validate_runtime(config, checkpoints)
    return {
        "config_path": path,
        "root": root,
        "config": config,
        "dataset_root": Path(str(config["_dataset_root"])),
        "checkpoint_root": Path(str(config["_checkpoint_root"])),
        "runtime_config": Path(str(config["_runtime_config"])),
        "checkpoints": checkpoints,
        "source_map_hashes": source_map_hashes,
        "source_task_hashes": source_task_hashes,
        "source_checkpoint_hashes": source_checkpoint_hashes,
        "h1_map_hashes": h1_map_hashes,
        "h1_task_hashes": h1_task_hashes,
        "h1_checkpoint_hashes": h1_checkpoint_hashes,
    }


def _schedule(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        key_index = int(checkpoint["key_index"])
        offset = key_index % len(CONTROLLERS)
        for position in range(len(CONTROLLERS)):
            controller = CONTROLLERS[(offset + position) % len(CONTROLLERS)]
            result.append(
                {
                    "schedule_index": len(result),
                    "key_index": key_index,
                    "within_key_position": position,
                    "controller": controller,
                    "checkpoint_id": str(checkpoint["checkpoint_id"]),
                    "checkpoint_identity_sha256": str(
                        checkpoint["checkpoint_identity_sha256"]
                    ),
                    "map_id": str(checkpoint["map_id"]),
                    "task_id": str(checkpoint["task_id"]),
                    "task_variant": str(checkpoint["task_variant"]),
                    "load_band": str(checkpoint["load_band"]),
                    "disturbance_replica": int(checkpoint["disturbance_replica"]),
                    "agent_count": int(checkpoint["agent_count"]),
                    "solver_seed": int(checkpoint["screen_solver_seed"]),
                }
            )
    return result


def run_preflight(config_path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    evidence = _evidence(config_path)
    config = evidence["config"]
    schedule = _schedule(evidence["checkpoints"])
    return {
        "schema": PREFLIGHT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "config_sha256": sha256_file(evidence["config_path"]),
        "output": str(Path(output).resolve()) if output is not None else None,
        "checkpoint_count": len(evidence["checkpoints"]),
        "map_count": len(evidence["source_map_hashes"]),
        "h1_map_count": len(evidence["h1_map_hashes"]),
        "source_and_h1_map_sha_intersection": sorted(
            evidence["source_map_hashes"] & evidence["h1_map_hashes"]
        ),
        "source_maps_sha_disjoint_from_h1": True,
        "source_and_h1_task_sha_intersection": sorted(
            evidence["source_task_hashes"] & evidence["h1_task_hashes"]
        ),
        "source_tasks_sha_disjoint_from_h1": True,
        "source_and_h1_checkpoint_identity_sha_intersection": sorted(
            evidence["source_checkpoint_hashes"]
            & evidence["h1_checkpoint_hashes"]
        ),
        "source_checkpoint_identities_sha_disjoint_from_h1": True,
        "controller_count": len(CONTROLLERS),
        "episode_count": len(schedule),
        "controllers": list(CONTROLLERS),
        "execution_order": config["runtime"]["execution_order"],
        "timed_workers": 1,
        "wall_time_budget_seconds": 60.0,
        "timing_boundary": "checkpoint_restore_inclusive_ttf",
        "builds_local_qualification_anchor_with_current_native": True,
        "rebinds_authenticated_checkpoint_rows_from_source": True,
        "native_or_controller_invoked": False,
        "formal_promotion_allowed": False,
        "default_replacement_allowed": False,
        "schedule": schedule,
    }


def _component16_augmentation() -> dict[str, Any]:
    module = importlib.import_module("lns2_selector.runtime.structshell_component16")
    build = getattr(module, "structshell_component16_augmentation")
    validate = getattr(module, "validate_structshell_component16_augmentation")
    augmentation = validate(build())
    if augmentation is None:
        raise ValueError("Component16 augmentation validation returned no contract")
    return dict(augmentation)


def _controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> tuple[str, dict[str, Any]]:
    common = _common_kwargs(config)
    if controller == "official_adaptive":
        return "official_adaptive", {
            **common,
            "controller": "official_adaptive",
            "feature_backend": "auto",
            "controller_runtime": "reference",
            "verification_profile": "audit",
        }
    if controller == "component16":
        augmentation = _component16_augmentation()
    elif controller == "dual16":
        validated = validate_structshell_dual16_augmentation(
            structshell_dual16_augmentation()
        )
        if validated is None:
            raise ValueError("Dual16 augmentation validation returned no contract")
        augmentation = dict(validated)
    else:
        raise ValueError(f"unknown independent TTF controller: {controller}")
    return "realized_dynamic", {
        **common,
        "controller": "v2-full",
        "controller_bundle": str(Path(str(config["_v2_bundle"])).resolve()),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "hybridstructpool_augmentation": augmentation,
    }


def _install_local_checkpoint_qualification_anchor(
    anchor: Path,
    checkpoints: list[dict[str, Any]],
    source_manifest: Path,
) -> None:
    """Replace natural reset rows with authenticated checkpoint-state rows."""

    local_manifest = anchor / "qualification_manifest.jsonl"
    natural = {
        (str(row.get("task_id")), int(row.get("solver_seed", -1))): dict(row)
        for row in read_jsonl(local_manifest)
    }
    source = {
        (str(row.get("task_id")), int(row.get("solver_seed", -1))): dict(row)
        for row in read_jsonl(source_manifest)
    }
    checkpoint_by_key = {
        (str(row["task_id"]), int(row["screen_solver_seed"])): row
        for row in checkpoints
    }
    expected = set(checkpoint_by_key)
    if (
        len(natural) != len(checkpoints)
        or len(source) != len(checkpoints)
        or set(natural) != expected
        or set(source) != expected
    ):
        raise ValueError("local/source qualification anchor key set changed")
    rebound: list[dict[str, Any]] = []
    source_sha = sha256_file(source_manifest)
    for key in sorted(expected):
        base = natural[key]
        authenticated = source[key]
        checkpoint = checkpoint_by_key[key]
        if (
            str(base.get("task_id")) != str(checkpoint["task_id"])
            or str(base.get("map_id")) != str(checkpoint["map_id"])
            or int(base.get("agent_count", -1)) != int(checkpoint["agent_count"])
            or authenticated.get("status") != "ok"
            or authenticated.get("qualification_source_kind")
            != "authenticated_checkpoint_blob_v1"
            or str(authenticated.get("checkpoint_id"))
            != str(checkpoint["checkpoint_id"])
            or str(authenticated.get("checkpoint_identity_sha256"))
            != str(checkpoint["checkpoint_identity_sha256"])
            or str(authenticated.get("state_fingerprint"))
            != str(checkpoint["expected_fingerprint"])
            or int(authenticated.get("initial_conflicts", -1))
            != int(checkpoint["expected_conflicts"])
        ):
            raise ValueError("authenticated qualification row changed during rebinding")
        rebound.append(
            {
                **base,
                "status": "ok",
                "error": None,
                "initial_complete": True,
                "initial_feasible": False,
                "repairable": True,
                "initial_conflicts": int(checkpoint["expected_conflicts"]),
                "initial_complexity": dict(checkpoint.get("native_complexity") or {}),
                "state_fingerprint": str(checkpoint["expected_fingerprint"]),
                "qualification_source_kind": "authenticated_checkpoint_blob_v1",
                "checkpoint_id": str(checkpoint["checkpoint_id"]),
                "checkpoint_identity_sha256": str(
                    checkpoint["checkpoint_identity_sha256"]
                ),
                "authenticated_source_qualification_manifest_sha256": source_sha,
            }
        )
    write_jsonl(
        local_manifest,
        sorted(rebound, key=lambda row: (str(row["task_id"]), int(row["solver_seed"]))),
    )


def _prepare_local_qualification_anchor(
    evidence: Mapping[str, Any], ttf_root: Path, *, resume: bool
) -> Path:
    config = evidence["config"]
    runtime = dict(config["runtime"])
    checkpoints = list(evidence["checkpoints"])
    keys = {
        (str(row["task_id"]), int(row["screen_solver_seed"]))
        for row in checkpoints
    }
    anchor = ttf_root / "qualification_anchor"
    anchor_exists = (anchor / "run_config.json").is_file()
    if anchor_exists and not resume:
        raise ValueError("local qualification anchor exists; pass resume")
    qualification_kwargs = {
        key: value
        for key, value in _common_kwargs(config).items()
        if key != "workers"
    }
    run_closed_loop_collection(
        evidence["dataset_root"],
        evidence["runtime_config"],
        anchor,
        phase="qualify",
        workers=int(runtime["qualification_workers"]),
        resume=anchor_exists,
        task_ids=sorted(task_id for task_id, _seed in keys),
        job_keys=keys,
        cohort_job_keys=keys,
        qualification_process_timeout_seconds=float(
            runtime["qualification_process_timeout_seconds"]
        ),
        **qualification_kwargs,
    )
    _install_local_checkpoint_qualification_anchor(
        anchor,
        checkpoints,
        _registered(config, "source_qualification_manifest"),
    )
    run_closed_loop_collection(
        evidence["dataset_root"],
        evidence["runtime_config"],
        anchor,
        phase="qualify",
        workers=1,
        resume=True,
        task_ids=sorted(task_id for task_id, _seed in keys),
        job_keys=keys,
        cohort_job_keys=keys,
        qualification_source=anchor,
        qualification_process_timeout_seconds=float(
            runtime["qualification_process_timeout_seconds"]
        ),
        **qualification_kwargs,
    )
    return anchor


def run_collection(
    config_path: str | Path, output: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    evidence = _evidence(config_path)
    config = evidence["config"]
    output_root = Path(output).resolve()
    ttf_root = output_root / "ttf"
    ttf_root.mkdir(parents=True, exist_ok=True)
    schedule = _schedule(evidence["checkpoints"])
    schedule_path = ttf_root / "execution_schedule.jsonl"
    if schedule_path.is_file() and read_jsonl(schedule_path) != schedule:
        raise ValueError("existing independent TTF execution schedule changed")
    if schedule_path.is_file() and not resume:
        raise ValueError("independent TTF output exists; pass resume")
    write_jsonl(schedule_path, schedule)
    write_json(ttf_root / "preflight.json", run_preflight(config_path, output_root))
    qualification_anchor = _prepare_local_qualification_anchor(
        evidence, ttf_root, resume=resume
    )
    checkpoints = {
        str(row["checkpoint_id"]): row for row in evidence["checkpoints"]
    }
    progress: list[dict[str, Any]] = []
    for item in schedule:
        checkpoint = checkpoints[str(item["checkpoint_id"])]
        key = (str(item["task_id"]), int(item["solver_seed"]))
        lane = _lane_root(ttf_root, item)
        lane_exists = (lane / "run_config.json").is_file()
        if lane_exists and not resume:
            raise ValueError(f"independent TTF lane exists; pass resume: {lane}")
        override = {
            key: {
                "schema": TTF_OVERRIDE_SCHEMA,
                "state_id": str(checkpoint["checkpoint_id"]),
                "initial_restore": {
                    **checkpoint,
                    "collection_root": str(evidence["checkpoint_root"].resolve()),
                },
            }
        }
        phase, kwargs = _controller_kwargs(
            evidence["root"], config, str(item["controller"])
        )
        shared = {
            "task_ids": [str(item["task_id"])],
            "job_keys": {key},
            "cohort_job_keys": {key},
            "qualification_source": qualification_anchor,
            "episode_overrides": override,
        }
        run_closed_loop_collection(
            evidence["dataset_root"],
            evidence["runtime_config"],
            lane,
            phase="qualify",
            resume=lane_exists,
            **shared,
            **kwargs,
        )
        result = run_closed_loop_collection(
            evidence["dataset_root"],
            evidence["runtime_config"],
            lane,
            phase=phase,
            resume=True,
            **shared,
            **kwargs,
        )
        progress.append({**item, "lane": str(lane), "collection": result})
        write_json(
            ttf_root / "collection_progress.json",
            {
                "schema": PREFLIGHT_SCHEMA,
                "completed_episode_count": len(progress),
                "expected_episode_count": len(schedule),
                "rows": progress,
            },
        )
    return analyze_collection(config_path, output_root)


def _pairwise_comparison(
    rows: list[dict[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    reference: str,
    target: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    indexed = {
        controller: {
            str(row["checkpoint_id"]): row
            for row in rows
            if row["controller"] == controller
        }
        for controller in (reference, target)
    }
    ids = sorted(set(indexed[reference]) & set(indexed[target]))
    left = [dict(indexed[reference][key]["summary"]) for key in ids]
    right = [dict(indexed[target][key]["summary"]) for key in ids]
    left_capped = [float(row["capped_wall_time_to_feasible"]) for row in left]
    right_capped = [float(row["capped_wall_time_to_feasible"]) for row in right]
    left_mean = sum(left_capped) / len(left_capped) if left_capped else 0.0
    right_mean = sum(right_capped) / len(right_capped) if right_capped else 0.0
    base_rate = float(summaries[reference]["successes_per_observed_hour"])
    target_rate = float(summaries[target]["successes_per_observed_hour"])
    throughput = (target_rate - base_rate) / base_rate if base_rate else None
    result = {
        "reference": reference,
        "target": target,
        "paired_key_count": len(ids),
        "paired_win_count": sum(r < l for l, r in zip(left_capped, right_capped)),
        "paired_tie_count": sum(r == l for l, r in zip(left_capped, right_capped)),
        "paired_win_rate": (
            sum(r < l for l, r in zip(left_capped, right_capped)) / len(ids)
            if ids
            else 0.0
        ),
        "mean_capped_ttf_improvement": (
            (left_mean - right_mean) / left_mean if left_mean else 0.0
        ),
        "throughput_improvement": throughput,
        "baseline_zero_success_throughput": bool(base_rate == 0 and target_rate > 0),
        "success_rate_loss": float(summaries[reference]["success_rate"])
        - float(summaries[target]["success_rate"]),
        "additional_timeout_count": sum(
            bool(r["external_timeout"]) and not bool(l["external_timeout"])
            for l, r in zip(left, right)
        ),
        "additional_censor_count": sum(
            bool(l["success"]) and not bool(r["success"])
            for l, r in zip(left, right)
        ),
    }
    result["diagnostic_screen_gate_passed"] = bool(
        len(ids) == 16
        and result["paired_win_rate"] >= float(gate["minimum_paired_win_rate"])
        and result["mean_capped_ttf_improvement"]
        >= float(gate["minimum_mean_capped_ttf_improvement"])
        and throughput is not None
        and throughput >= float(gate["minimum_successes_per_hour_improvement"])
        and result["success_rate_loss"] <= float(gate["maximum_success_rate_loss"])
        and (
            not bool(gate["no_additional_timeout_or_censor"])
            or (
                result["additional_timeout_count"] == 0
                and result["additional_censor_count"] == 0
            )
        )
    )
    result["gate_is_diagnostic_only"] = True
    return result


def analyze_collection(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    evidence = _evidence(config_path)
    config = evidence["config"]
    ttf_root = Path(output).resolve() / "ttf"
    schedule = _schedule(evidence["checkpoints"])
    schedule_path = ttf_root / "execution_schedule.jsonl"
    schedule_integrity = bool(
        schedule_path.is_file() and read_jsonl(schedule_path) == schedule
    )
    checkpoint_by_id = {
        str(row["checkpoint_id"]): row for row in evidence["checkpoints"]
    }
    collected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for item in schedule:
        manifest_path = _lane_root(ttf_root, item) / TIMED_MANIFESTS[str(item["controller"])]
        matches = [
            dict(row)
            for row in (read_jsonl(manifest_path) if manifest_path.is_file() else [])
            if str(row.get("task_id")) == str(item["task_id"])
            and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
        ]
        if len(matches) != 1 or matches[0].get("status") not in {"ok", "resumed"}:
            missing.append(dict(item))
            continue
        checkpoint = checkpoint_by_id[str(item["checkpoint_id"])]
        summary = dict(matches[0].get("summary") or {})
        collected.append(
            {
                **item,
                "summary": summary,
                "initial_fingerprint_matches": str(summary.get("initial_fingerprint"))
                == str(checkpoint["expected_fingerprint"]),
                "initial_conflicts_match": int(summary.get("initial_conflicts", -1))
                == int(checkpoint["expected_conflicts"]),
                "bounded_summary_valid": bool(
                    summary.get("ttf_clock_schema") == "lns2.ttf.reset_inclusive_wall.v1"
                    and summary.get("capped_wall_time_to_feasible") is not None
                    and float(summary.get("wall_time_budget_seconds", -1)) == 60.0
                    and int(summary.get("invalid_action_count", -1)) == 0
                    and int(summary.get("fingerprint_mismatch_count", -1)) == 0
                    and str(summary.get("stop_reason"))
                    in {
                        "success",
                        "wall_timeout",
                        "controller_stalled",
                        "native_terminal",
                    }
                ),
            }
        )
    by_controller = {
        controller: [row for row in collected if row["controller"] == controller]
        for controller in CONTROLLERS
    }
    summaries = {
        controller: _controller_summary(rows)
        for controller, rows in by_controller.items()
    }
    reporting = dict(config["reporting"])
    comparisons = {
        "component16_vs_official": _pairwise_comparison(
            collected, summaries, "official_adaptive", "component16", reporting
        ),
        "dual16_vs_official": _pairwise_comparison(
            collected, summaries, "official_adaptive", "dual16", reporting
        ),
        "component16_vs_dual16": _pairwise_comparison(
            collected, summaries, "dual16", "component16", reporting
        ),
    }
    integrity = {
        "complete_key_mod_3_rotating_strict_serial_schedule": bool(
            schedule_integrity and len(collected) == 48 and not missing
        ),
        "all_16_authenticated_checkpoints_per_controller": all(
            len(rows) == 16 for rows in by_controller.values()
        ),
        "paired_initial_fingerprints": all(
            row["initial_fingerprint_matches"] for row in collected
        ),
        "paired_initial_conflicts": all(
            row["initial_conflicts_match"] for row in collected
        ),
        "registered_reset_inclusive_60_second_ttf_clock": all(
            row["bounded_summary_valid"] for row in collected
        ),
        "source_maps_sha_disjoint_from_h1": not bool(
            evidence["source_map_hashes"] & evidence["h1_map_hashes"]
        ),
        "source_tasks_sha_disjoint_from_h1": not bool(
            evidence["source_task_hashes"] & evidence["h1_task_hashes"]
        ),
        "source_checkpoint_identities_sha_disjoint_from_h1": not bool(
            evidence["source_checkpoint_hashes"]
            & evidence["h1_checkpoint_hashes"]
        ),
        "timed_workers_one": config["runtime"]["timed_workers"] == 1,
    }
    complete = bool(integrity and all(integrity.values()))
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "config_sha256": sha256_file(evidence["config_path"]),
        "checkpoint_count": 16,
        "map_count": 8,
        "expected_episode_count": 48,
        "completed_episode_count": len(collected),
        "controllers": summaries,
        "comparisons": comparisons,
        "integrity_gates": integrity,
        "complete_pairing_and_initial_state_identity": complete,
        "missing_or_failed_schedule_rows": missing,
        "timing_boundary": "checkpoint_restore_inclusive_ttf",
        "development_independent_from_h1_label_maps": True,
        "prior_dual16_confirmation_cohort_reused": True,
        "diagnostic_gate_only": True,
        "formal_promotion_allowed": False,
        "global_claim_allowed": False,
        "default_replacement_allowed": False,
        "promotion_decision": "development_evidence_only_no_default_change",
    }
    ttf_root.mkdir(parents=True, exist_ok=True)
    write_json(ttf_root / REPORT_FILENAME, report)
    return report


plan_ttf = run_preflight
collect_ttf = run_collection
analyze_ttf = analyze_collection

__all__ = [
    "CONTROLLERS",
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "PREFLIGHT_SCHEMA",
    "REPORT_SCHEMA",
    "_controller_kwargs",
    "_schedule",
    "analyze_collection",
    "analyze_ttf",
    "collect_ttf",
    "load_config",
    "plan_ttf",
    "run_collection",
    "run_preflight",
    "validate_config",
]
