from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy

from experiments._common import closed_loop_producer_identity, registered_input, sha256_file
from experiments.balanced_wall_clock import (
    SPLIT,
    _four_neighbor_distances,
    _map_metrics,
    _movingai_passable_cells,
    _write_derived_scenario,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _CollectionRunLock,
    _fingerprint,
    _qualification_worker,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.run_output_guard import prepare_resumable_output
from experiments.warehouse_compactcut_tasks import (
    audit_compactcut_geometry,
    audit_compactcut_tasks,
    generate_compactcut_map_bank,
    generate_compactcut_tasks,
)
from generators.io import map_document, write_movingai_map
from generators.models import MapData


CONFIG_SCHEMA = "lns2.stride.warehouse_compactcut_config.v1"
RUNTIME_SCHEMA = 1
REGISTRY_SCHEMA = "lns2.stride.warehouse_compactcut_task_registry.v1"
Q0_SCHEMA = "lns2.stride.warehouse_compactcut_q0_audit.v1"
Q0_MANIFEST_SCHEMA = "lns2.stride.warehouse_compactcut_q0_manifest.v1"
QUALIFICATION_STATUS_SCHEMA = "lns2.stride.warehouse_compactcut_qualification_status.v1"
QUALIFICATION_REPORT_SCHEMA = "lns2.stride.warehouse_compactcut_qualification_report.v1"
SELECTION_SCHEMA = "lns2.stride.warehouse_compactcut_benchmark_selection.v1"
EXPERIMENT_ID = "stride-warehouse-compactcut-v1"
TASK_GENERATOR_SHA256 = "0d3f8d07bcdcdc60a412accbb1a4966f920c107834c58f5b68f767354002f2b4"

AGENT_COUNT = 120
TASK_SEEDS = (521, 557)
SOLVER_SEEDS = (71, 72, 73, 74)
STRUCTURED_VARIANT = "bidirectional_mandatory_cut_exchange"
CONTROL_VARIANT = "diagnostic_within_partition_exchange"
TASK_VARIANTS = (STRUCTURED_VARIANT, CONTROL_VARIANT)
MAPS = (
    {"id": "whcc_cfg_01", "template": "cross_four_gate", "map_seed": 2026081701, "map_role": "development"},
    {"id": "whcc_cfg_02", "template": "cross_four_gate", "map_seed": 2026081702, "map_role": "controller_held_out"},
    {"id": "whcc_cfg_03", "template": "cross_four_gate", "map_seed": 2026081703, "map_role": "development"},
    {"id": "whcc_cfg_04", "template": "cross_four_gate", "map_seed": 2026081704, "map_role": "controller_held_out"},
    {"id": "whcc_dh_01", "template": "double_horizontal", "map_seed": 2026081711, "map_role": "development"},
    {"id": "whcc_dh_02", "template": "double_horizontal", "map_seed": 2026081712, "map_role": "controller_held_out"},
    {"id": "whcc_dh_03", "template": "double_horizontal", "map_seed": 2026081713, "map_role": "development"},
    {"id": "whcc_dh_04", "template": "double_horizontal", "map_seed": 2026081715, "map_role": "controller_held_out"},
)
MAP_BY_ID = {str(row["id"]): row for row in MAPS}
Q0_BYTE_REPRODUCIBILITY = {
    "assignment_backend": "scipy.optimize.linear_sum_assignment",
    "task_generator_sha256": TASK_GENERATOR_SHA256,
    "accepted_toolchains": [
        {
            "name": "ubuntu_22_04_wsl",
            "python_major_minor": "3.10",
            "python_version_observed": "3.10.12",
            "numpy_version": "1.21.5",
            "scipy_version": "1.8.0",
        },
        {
            "name": "windows_native",
            "python_major_minor": "3.12",
            "python_version_observed": "3.12.3",
            "numpy_version": "1.26.4",
            "scipy_version": "1.13.1",
        },
    ],
    "cohort_payload_hash_parity": {
        "task_pair_count": 16,
        "identical_payload_sha256_count": 16,
        "all_identical": True,
        "claim_boundary": "registered_8_map_2_task_seed_compactcut_q0_cohort_only",
        "pair_payload_sha256": {
            "whcc_cfg_01/t0521": "49f1e004927a6542214498faf44edab06aec865588c6b2cb6be9f60fd7a5eb39",
            "whcc_cfg_01/t0557": "30911e93039ae4022001c8859d3b49169ccfb4a7b624818b315a1e7f4809bb8e",
            "whcc_cfg_02/t0521": "a886c1101ff0bdcbe1e70e6af5c7d75a84e7f6616c1fcf9bce77dd6ddd3aff95",
            "whcc_cfg_02/t0557": "2827b056f3d44b0de5c33f4dc0e3e4c7c1c746dd8625bf3705f220ed905cd0bc",
            "whcc_cfg_03/t0521": "20f7d25d104b54847a25e9cb3c5c295aef3b31d884f4570549155d04ac1e7123",
            "whcc_cfg_03/t0557": "55cc90d0f894b3e17c91396cf06ff8a64a86e38a67c8aac7d4aa2c3aedbaee0a",
            "whcc_cfg_04/t0521": "c9b137884eceda95cb650359e62e2e023cfcf9f4649afc78ef6225ebcd6352a3",
            "whcc_cfg_04/t0557": "e5e78200d6233b63dcbb0a491a895575730783597d02b3497e08d7c2899948be",
            "whcc_dh_01/t0521": "8224d4ee3068c0b7777d06f60b818d30092dd864b694edd79fb20c5cd3340756",
            "whcc_dh_01/t0557": "8de1148d1a5b42c5e94b6d7db8832e323d2a0e727177afbf133bc3ad162c001c",
            "whcc_dh_02/t0521": "186c711e456e83b54a6917aa6e6133492afa429830a9e9a6a7f8e0de872e6d59",
            "whcc_dh_02/t0557": "b704f9719b9e8fc3eb91b3ac9abaa9a36555b370df066f1cedfcb252ae095c83",
            "whcc_dh_03/t0521": "17c36c8b46085ca3203ce60dae3534965306e8c3cfcc13fb9d935d14017f5998",
            "whcc_dh_03/t0557": "ef592423706c30e438fee69e10e6a1f39d8ada73b38328f298b078e8e1efc6c8",
            "whcc_dh_04/t0521": "65c998ecc0764ce1be93c8639ce9b6432407fdc3b056760913dd70e691abbc9e",
            "whcc_dh_04/t0557": "5624d420490151eb236a311f40edfd31238010ee6d31f729af293ebbd8019a7a",
        },
    },
    "registered_byte_policy": (
        "map_task_scenario_and_endpoint_hashes_require_an_accepted_toolchain_"
        "and_are_supported_by_16_of_16_registered_pair_payload_hash_parity_only"
    ),
}
STATIC_PRE_REGISTRATION_RECORD = {
    "map_id": "whcc_dh_04",
    "divider_template": "double_horizontal",
    "gate": "minimum_partition_size_at_least_120_on_realised_grid",
    "selection_rule": (
        "first_subsequent_integer_seed_passing_the_same_static_geometry_gate"
    ),
    "candidates": [
        {"map_seed": 2026081714, "minimum_partition_size": 96, "passed": False},
        {"map_seed": 2026081715, "minimum_partition_size": 288, "passed": True},
    ],
    "selected_map_seed": 2026081715,
    "task_or_solver_result_observed": False,
    "native_reset_observed": False,
    "controller_observed": False,
}

Q0_FILENAME = "q0_compactcut_audit.json"
Q0_MANIFEST_FILENAME = "q0_manifest.jsonl"
Q0_REGISTRATION_PROPOSAL_FILENAME = "q0_task_registry_proposal.json"
DATASET_SUMMARY_FILENAME = "dataset_summary.json"
QUALIFICATION_STATUS_FILENAME = "qualification_status.json"
QUALIFICATION_REPORT_FILENAME = "qualification_report.json"
QUALIFICATION_MANIFEST_FILENAME = "qualification_manifest.jsonl"
SELECTION_FILENAME = "benchmark_selection.json"

PRODUCER_SOURCE_FILES = (
    "experiments/stride_warehouse_compactcut.py",
    "experiments/warehouse_compactcut_tasks.py",
    "experiments/balanced_wall_clock.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/run_output_guard.py",
)


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _q0_runtime_identity() -> dict[str, str]:
    return {
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "python_version_observed": platform.python_version(),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "assignment_backend": "scipy.optimize.linear_sum_assignment",
    }


def _accepted_q0_runtime_identities() -> list[dict[str, str]]:
    return [
        {key: value for key, value in dict(toolchain).items() if key != "name"}
        | {"assignment_backend": Q0_BYTE_REPRODUCIBILITY["assignment_backend"]}
        for toolchain in Q0_BYTE_REPRODUCIBILITY["accepted_toolchains"]
    ]


def _require_q0_runtime_identity() -> dict[str, str]:
    observed = _q0_runtime_identity()
    accepted = _accepted_q0_runtime_identities()
    if observed not in accepted:
        raise ValueError(
            "Warehouse compact-cut Q0 byte runtime identity changed: "
            f"accepted={accepted}, observed={observed}"
        )
    return observed


def _resolve_registered_input(
    root: Path, specification: Mapping[str, Any], label: str
) -> Path:
    return registered_input(root, dict(specification), label=label)


def _pending_input(root: Path, specification: Mapping[str, Any], label: str) -> Path:
    path = (root / str(specification.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes project root") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _load_registry(root: Path, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    registration = dict(config.get("registration") or {})
    registry_path = _resolve_registered_input(
        root,
        {
            "path": registration.get("task_registry_path"),
            "sha256": registration.get("task_registry_sha256"),
        },
        "Warehouse compact-cut task registry",
    )
    registry = _read_json(registry_path)
    tasks = dict(registry.get("tasks") or {})
    maps = dict(registry.get("maps") or {})
    registered_map_hashes = [str(dict(row).get("map_sha256") or "") for row in maps.values()]
    map_roles_match = set(maps) == set(MAP_BY_ID) and all(
        str(dict(maps[map_id]).get("map_role"))
        == str(MAP_BY_ID[map_id]["map_role"])
        for map_id in MAP_BY_ID
    )
    task_roles_match = all(
        str(dict(row).get("map_id")) in MAP_BY_ID
        and str(dict(row).get("map_role"))
        == str(MAP_BY_ID[str(dict(row).get("map_id"))]["map_role"])
        for row in tasks.values()
    )
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("experiment_id") != EXPERIMENT_ID
        or int(registry.get("map_count", -1)) != 8
        or len(maps) != 8
        or not map_roles_match
        or len(registered_map_hashes) != 8
        or not all(_is_sha256(value) for value in registered_map_hashes)
        or len(set(registered_map_hashes)) != 8
        or int(registry.get("task_count", -1)) != 32
        or len(tasks) != 32
        or not task_roles_match
        or not _is_sha256(registry.get("q0_manifest_sha256"))
        or dict(registry.get("q0_byte_reproducibility") or {})
        != Q0_BYTE_REPRODUCIBILITY
        or dict(registry.get("q0_generation_toolchain") or {})
        not in _accepted_q0_runtime_identities()
        or dict(registry.get("static_pre_registration_record") or {})
        != STATIC_PRE_REGISTRATION_RECORD
        or str(registry.get("task_generator_sha256"))
        != str(dict(config["inputs"])["task_generator"]["sha256"])
    ):
        raise ValueError("Warehouse compact-cut task registry identity changed")
    return registry_path, registry


def load_config(
    path: str | Path, *, require_frozen_registry: bool = False
) -> tuple[Path, Path, dict[str, Any], dict[str, Any] | None]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit")
        != "88cb35e79ce4c907b71f5306c3d0e961f13d029c"
    ):
        raise ValueError("Warehouse compact-cut benchmark identity changed")
    if dict(config.get("phase_contract") or {}) != {
        "allowed_phases": ["q0_geometry_and_task_registration", "q1_reset_only"],
        "controller_step_allowed": False,
        "policy_episode_allowed": False,
        "formal_ttf_allowed": False,
        "formal_schedule_allowed": False,
        "q1_requires_frozen_task_registry": True,
    }:
        raise ValueError("Warehouse compact-cut phase contract changed")
    if dict(config.get("runtime") or {}) != {
        "workers": 16,
        "qualification_process_timeout_seconds": 240.0,
        "stop_on_first_execution_error_or_process_timeout": True,
        "resume_only_after_zero_execution_errors_and_zero_process_timeouts": True,
    }:
        raise ValueError("Warehouse compact-cut reset runtime contract changed")
    if dict(config.get("q0_byte_reproducibility") or {}) != Q0_BYTE_REPRODUCIBILITY:
        raise ValueError("Warehouse compact-cut Q0 byte reproducibility changed")
    if dict(config.get("static_pre_registration_record") or {}) != (
        STATIC_PRE_REGISTRATION_RECORD
    ):
        raise ValueError("Warehouse compact-cut static preregistration changed")
    cohort = dict(config.get("cohort") or {})
    if (
        cohort.get("split") != SPLIT
        or int(cohort.get("agent_count", -1)) != AGENT_COUNT
        or tuple(map(int, cohort.get("task_seeds") or ())) != TASK_SEEDS
        or tuple(map(str, cohort.get("task_variants") or ())) != TASK_VARIANTS
        or tuple(map(int, cohort.get("solver_seeds") or ())) != SOLVER_SEEDS
        or tuple(dict(row) for row in cohort.get("maps") or ()) != MAPS
        or int(cohort.get("expected_task_count", -1)) != 32
        or int(cohort.get("expected_reset_count", -1)) != 128
        or cohort.get("claim_boundary")
        != "eight_registered_generated_compact_warehouse_maps_and_two_preregistered_cut_task_families_only"
    ):
        raise ValueError("Warehouse compact-cut cohort changed")
    if dict(config.get("q0_gates") or {}) != {
        "all_8_maps_required": True,
        "all_8_map_grid_sha256_unique_required": True,
        "all_32_tasks_required": True,
        "fixed_agent_count": 120,
        "unique_starts_required": True,
        "unique_goals_required": True,
        "zero_fixed_points_required": True,
        "all_endpoints_reachable_required": True,
        "structured_all_agents_cross_registered_mandatory_cut_required": True,
        "structured_exact_30_agents_per_direction_per_cut_cell_required": True,
        "structured_no_cut_bypass_required": True,
        "control_all_agents_remain_within_registered_partition_required": True,
        "pairing_semantics": "independent_registered_endpoints_distance_near_matched_secondary_control",
        "causal_control_claim_allowed": False,
        "maximum_relative_mean_shortest_distance_difference": 0.05,
        "maximum_relative_p95_shortest_distance_difference": 0.10,
        "deterministic_collision_free_prioritized_joint_mapf_witness_required": True,
        "solver_or_controller_invocation_allowed": False,
    }:
        raise ValueError("Warehouse compact-cut Q0 gates changed")
    if dict(config.get("q1_gates") or {}) != {
        "all_128_resets_required": True,
        "all_initial_solutions_complete_and_consistent": True,
        "structured": {
            "variant": STRUCTURED_VARIANT,
            "required_reset_count": 64,
            "minimum_initial_conflicts": 16,
            "minimum_active_conflict_agents": 32,
            "minimum_largest_conflict_component_size": 16,
            "initial_feasible": False,
            "all_resets_must_pass": True,
        },
        "diagnostic_control": {
            "variant": CONTROL_VARIANT,
            "required_reset_count": 64,
            "all_resets_must_complete_without_execution_error_or_process_timeout": True,
            "structure_thresholds_gate_benchmark_ready": False,
        },
        "no_map_task_variant_or_seed_replacement": True,
        "failure_decision": "stop_do_not_resample_replace_or_run_controller",
    }:
        raise ValueError("Warehouse compact-cut Q1 gates changed")
    if dict(config.get("result_interpretation") or {}) != {
        "control_label": "independent_registered_endpoints_distance_near_matched_within_partition_diagnostic_control_not_a_causal_or_difficulty_matched_baseline",
        "control_structure_thresholds_gate_benchmark_ready": False,
        "controller_performance_claim_allowed": False,
        "ttf_claim_allowed": False,
        "benchmark_ready_means": "q0_passed_all_128_resets_complete_error_free_and_all_64_structured_resets_pass_16_32_16",
    }:
        raise ValueError("Warehouse compact-cut result interpretation changed")

    inputs = dict(config.get("inputs") or {})
    runtime_spec = dict(inputs.get("runtime_config") or {})
    if runtime_spec != {
        "path": "configs/stride_warehouse_compactcut_runtime_v1.json",
        "sha256": "f24d29dda5d94aea10574e7320beb696b7140c2ed6ac694b40f113cdc52c76aa",
    }:
        raise ValueError("Warehouse compact-cut runtime registration changed")
    runtime_path = _resolve_registered_input(
        root, runtime_spec, "Warehouse compact-cut runtime"
    )
    generator_spec = dict(inputs.get("task_generator") or {})
    registration = dict(config.get("registration") or {})
    frozen = (
        registration.get("status") == "frozen_q0_task_hash_registration"
        and registration.get("q1_allowed") is True
        and registration.get("task_registry_path")
        == "configs/stride_warehouse_compactcut_tasks_v1.json"
        and _is_sha256(registration.get("task_registry_sha256"))
    )
    if frozen:
        if config.get("scientific_status") != (
            "preregistered_q0_task_hash_frozen_q1_reset_only_benchmark_qualification"
        ) or str(generator_spec.get("path")) != (
            "experiments/warehouse_compactcut_tasks.py"
        ) or set(registration) != {
            "status",
            "task_registry_path",
            "task_registry_sha256",
            "q1_allowed",
        } or set(generator_spec) != {"path", "sha256"} or str(
            generator_spec.get("sha256")
        ) != TASK_GENERATOR_SHA256:
            raise ValueError("Warehouse compact-cut frozen status changed")
        _resolve_registered_input(root, generator_spec, "Warehouse compact-cut task generator")
    else:
        if (
            registration
            != {
                "status": "pending_q0_task_hash_registration",
                "task_registry_path": "configs/stride_warehouse_compactcut_tasks_v1.json",
                "task_registry_sha256": None,
                "q1_allowed": False,
            }
            or config.get("scientific_status")
            != "preregistered_q0_generation_pending_task_hash_registration_reset_only"
            or generator_spec
            != {
                "path": "experiments/warehouse_compactcut_tasks.py",
                "sha256": TASK_GENERATOR_SHA256,
            }
        ):
            raise ValueError("Warehouse compact-cut pending registration changed")
        _resolve_registered_input(
            root, generator_spec, "Warehouse compact-cut task generator"
        )
    if require_frozen_registry and not frozen:
        raise ValueError(
            "Warehouse compact-cut Q1 is blocked until all 8 map hashes, all 32 "
            "task hashes, and the Q0 manifest hash are frozen in the tracked registry"
        )

    runtime = _read_json(runtime_path)
    design = dict(runtime.get("dataset_design") or {})
    if (
        int(runtime.get("schema_version", -1)) != RUNTIME_SCHEMA
        or runtime.get("formal") is not False
        or str(runtime.get("split")) != SPLIT
        or tuple(map(int, runtime.get("solver_seeds") or ())) != SOLVER_SEEDS
        or int(runtime.get("max_decisions", -1)) != 0
        or int(runtime.get("workers", -1)) != 16
        or int(design.get("map_count", -1)) != 8
        or int(design.get("instance_count", -1)) != 32
        or dict(design.get("source_counts") or {}) != {"generated": 32}
        or dict(design.get("layout_counts") or {}) != {"warehouse": 32}
    ):
        raise ValueError("Warehouse compact-cut runtime input changed")
    registry = _load_registry(root, config)[1] if frozen else None
    return path, root, config, registry


def _task_id(map_id: str, variant: str, task_seed: int) -> str:
    suffix = "bmcx" if variant == STRUCTURED_VARIANT else "dwpx"
    return f"{map_id}__{suffix}__t{task_seed:04d}__n{AGENT_COUNT:04d}"


def benchmark_task_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    del config
    rows = [
        {
            "split": SPLIT,
            "map_id": str(map_spec["id"]),
            "map_role": str(map_spec["map_role"]),
            "task_id": _task_id(str(map_spec["id"]), variant, task_seed),
            "variant": variant,
            "task_seed": task_seed,
            "agent_count": AGENT_COUNT,
        }
        for map_spec in MAPS
        for task_seed in TASK_SEEDS
        for variant in TASK_VARIANTS
    ]
    if len(rows) != 32 or len({str(row["task_id"]) for row in rows}) != 32:
        raise ValueError("Warehouse compact-cut task cohort changed")
    return sorted(rows, key=lambda row: str(row["task_id"]))


def qualification_schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {**task, "solver_seed": solver_seed}
        for task in benchmark_task_specs(config)
        for solver_seed in SOLVER_SEEDS
    ]


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config, registry = load_config(config_path)
    return {
        "schema": QUALIFICATION_STATUS_SCHEMA,
        "command": "plan",
        "experiment_id": EXPERIMENT_ID,
        "registration_status": config["registration"]["status"],
        "q1_allowed": registry is not None,
        "map_count": 8,
        "map_role_counts": {"development": 4, "controller_held_out": 4},
        "task_count": len(benchmark_task_specs(config)),
        "qualification_reset_count": len(qualification_schedule(config)),
        "structured_reset_count": 64,
        "diagnostic_control_reset_count": 64,
        "workers": 16,
        "controller_count": 0,
        "policy_episode_count": 0,
        "formal_ttf_episode_count": 0,
        "claim_boundary": config["cohort"]["claim_boundary"],
    }


def _schedule_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row.get("task_id")), int(row.get("solver_seed", -1))


def _task_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in benchmark_task_specs(config)}


def _map_bank() -> dict[str, MapData]:
    bank = dict(generate_compactcut_map_bank())
    if set(bank) != set(MAP_BY_ID) or len(bank) != 8:
        raise ValueError("Warehouse compact-cut helper map bank changed")
    for map_id, map_data in bank.items():
        spec = MAP_BY_ID[map_id]
        if map_data.map_id != map_id or int(map_data.seed) != int(spec["map_seed"]):
            raise ValueError(f"Warehouse compact-cut helper map identity changed: {map_id}")
    return bank


def _movingai_map_text(map_data: MapData) -> str:
    return "\n".join(
        [
            "type octile",
            f"height {map_data.rows}",
            f"width {map_data.cols}",
            "map",
            *map_data.grid,
            "",
        ]
    )


def _percentile95(values: list[int]) -> float:
    if not values:
        raise ValueError("Warehouse compact-cut distance vector is empty")
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _pair_distance_metrics(pair: Mapping[str, Any]) -> dict[str, float]:
    variants = dict(pair.get("variants") or {})
    structured = list(map(int, dict(variants.get(STRUCTURED_VARIANT) or {}).get("shortest_path_distances") or ()))
    control = list(map(int, dict(variants.get(CONTROL_VARIANT) or {}).get("shortest_path_distances") or ()))
    if len(structured) != AGENT_COUNT or len(control) != AGENT_COUNT:
        raise ValueError("Warehouse compact-cut distance vector length changed")
    structured_mean = sum(structured) / len(structured)
    control_mean = sum(control) / len(control)
    structured_p95 = _percentile95(structured)
    control_p95 = _percentile95(control)
    mean_relative = abs(structured_mean - control_mean) / max(structured_mean, 1.0)
    p95_relative = abs(structured_p95 - control_p95) / max(structured_p95, 1.0)
    return {
        "structured_mean": structured_mean,
        "control_mean": control_mean,
        "structured_p95": structured_p95,
        "control_p95": control_p95,
        "mean_relative_difference": mean_relative,
        "p95_relative_difference": p95_relative,
    }


def _pair_witness(pair: Mapping[str, Any], task: Mapping[str, Any]) -> Any:
    witness = task.get("prioritized_joint_mapf_witness")
    if witness is None:
        witness = task.get("serialized_feasibility_witness")
    if witness is None:
        witness = pair.get("prioritized_joint_mapf_witness")
    if witness is None:
        witness = pair.get("serialized_feasibility_witness")
    if witness is None:
        raise ValueError("Warehouse compact-cut prioritized joint MAPF witness is missing")
    return witness


def _variant_task(pair: Mapping[str, Any], variant: str) -> dict[str, Any]:
    task = dict(dict(pair.get("variants") or {}).get(variant) or {})
    if not task:
        raise ValueError(f"Warehouse compact-cut helper omitted variant {variant}")
    _pair_witness(pair, task)
    return task


def _endpoint_payload_sha256(task: Mapping[str, Any]) -> str:
    return _fingerprint(
        {
            "task_id": str(task.get("task_id")),
            "starts": task.get("starts"),
            "goals": task.get("goals"),
            "shortest_path_distances": task.get("shortest_path_distances"),
        }
    )


def _canonical_pair_payload_sha256(pair: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        pair, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_generated_pair(
    map_data: MapData, pair: Mapping[str, Any], task_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = dict(audit_compactcut_tasks(map_data, pair))
    if audit.get("passed") is not True:
        raise ValueError(
            f"Warehouse compact-cut helper task audit failed: "
            f"{map_data.map_id}/task{task_seed}: {audit.get('errors')}"
        )
    metrics = _pair_distance_metrics(pair)
    parity_key = f"{map_data.map_id}/t{task_seed:04d}"
    pair_sha = _canonical_pair_payload_sha256(pair)
    expected_pair_sha = Q0_BYTE_REPRODUCIBILITY["cohort_payload_hash_parity"][
        "pair_payload_sha256"
    ].get(parity_key)
    if pair_sha != expected_pair_sha:
        raise ValueError(
            f"Warehouse compact-cut registered pair payload changed: {parity_key}"
        )
    metrics["pair_payload_sha256"] = pair_sha
    if metrics["mean_relative_difference"] > 0.05 or metrics["p95_relative_difference"] > 0.10:
        raise ValueError(
            f"Warehouse compact-cut distance matching gate failed: "
            f"{map_data.map_id}/task{task_seed}"
        )
    for variant in TASK_VARIANTS:
        task = _variant_task(pair, variant)
        expected_id = _task_id(map_data.map_id, variant, task_seed)
        starts = [tuple(map(int, cell)) for cell in task.get("starts") or ()]
        goals = [tuple(map(int, cell)) for cell in task.get("goals") or ()]
        if (
            str(task.get("task_id")) != expected_id
            or str(task.get("map_id")) != map_data.map_id
            or str(task.get("variant")) != variant
            or int(task.get("task_seed", -1)) != task_seed
            or int(task.get("agent_count", task.get("N", -1))) != AGENT_COUNT
            or len(starts) != AGENT_COUNT
            or len(goals) != AGENT_COUNT
        ):
            raise ValueError(f"Warehouse compact-cut generated task identity changed: {expected_id}")
    return audit, metrics


def _generation_fingerprint(root: Path) -> str:
    return _fingerprint(
        {
            "experiment_id": EXPERIMENT_ID,
            "maps": list(MAPS),
            "agent_count": AGENT_COUNT,
            "task_seeds": list(TASK_SEEDS),
            "task_variants": list(TASK_VARIANTS),
            "task_generator_sha256": sha256_file(
                root / "experiments" / "warehouse_compactcut_tasks.py"
            ),
            "q0_byte_reproducibility": Q0_BYTE_REPRODUCIBILITY,
            "static_pre_registration_record": STATIC_PRE_REGISTRATION_RECORD,
        }
    )


def _parse_and_audit_scenario(
    scenario_path: Path,
    *,
    map_name: str,
    rows: int,
    cols: int,
    passable: set[tuple[int, int]],
    task: Mapping[str, Any],
) -> None:
    lines = scenario_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "version 1" or len(lines) != AGENT_COUNT + 1:
        raise ValueError(f"Warehouse compact-cut scenario shape changed: {scenario_path}")
    starts: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = []
    recorded_distances: list[int] = []
    for index, line in enumerate(lines[1:]):
        fields = line.split("\t")
        if len(fields) != 9:
            raise ValueError(
                f"Warehouse compact-cut scenario row has {len(fields)} fields: "
                f"{scenario_path}/{index}"
            )
        bucket, row_map, width, height, sx, sy, gx, gy, distance = fields
        try:
            values = tuple(map(int, (bucket, width, height, sx, sy, gx, gy, distance)))
        except ValueError as error:
            raise ValueError(
                f"Warehouse compact-cut scenario row is non-integral: {scenario_path}/{index}"
            ) from error
        bucket_i, width_i, height_i, sx_i, sy_i, gx_i, gy_i, distance_i = values
        if (
            bucket_i != 0
            or row_map != map_name
            or width_i != cols
            or height_i != rows
            or distance_i < 0
        ):
            raise ValueError(
                f"Warehouse compact-cut scenario identity changed: {scenario_path}/{index}"
            )
        starts.append((sy_i, sx_i))
        goals.append((gy_i, gx_i))
        recorded_distances.append(distance_i)
    task_starts = [tuple(map(int, cell)) for cell in task.get("starts") or ()]
    task_goals = [tuple(map(int, cell)) for cell in task.get("goals") or ()]
    task_distances = list(map(int, task.get("shortest_path_distances") or ()))
    recomputed = _four_neighbor_distances(passable, starts, goals)
    if (
        starts != task_starts
        or goals != task_goals
        or recorded_distances != task_distances
        or recomputed != task_distances
        or len(recomputed) != AGENT_COUNT
    ):
        raise ValueError(
            f"Warehouse compact-cut scenario/task coordinate or distance mismatch: {scenario_path}"
        )


def _contained_file(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the registered dataset split") from error
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def _map_metadata(map_data: MapData, geometry: Mapping[str, Any], map_sha: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_id": map_data.map_id,
        "source": "deterministically generated compact Warehouse map",
        "map_seed": int(map_data.seed),
        "template": str(MAP_BY_ID[map_data.map_id]["template"]),
        "map_role": str(MAP_BY_ID[map_data.map_id]["map_role"]),
        "map_sha256": map_sha,
        "map_document": map_document(map_data),
        "compactcut_geometry": dict(geometry),
    }


def _expected_task_payload(
    pair: Mapping[str, Any], variant: str, scenario_sha256: str
) -> dict[str, Any]:
    task = _variant_task(pair, variant)
    map_id = str(task.get("map_id"))
    return {
        **task,
        "schema_version": 1,
        "task_semantics": "registered compact Warehouse cut benchmark Q0",
        "map_role": str(MAP_BY_ID[map_id]["map_role"]),
        "pairing_semantics": pair.get("pairing_semantics"),
        "causal_control_claim": pair.get("causal_control_claim"),
        "control_interpretation": pair.get("control_interpretation"),
        "map_grid_sha256": pair.get("map_grid_sha256"),
        "geometry": pair.get("geometry"),
        "routing_registration": pair.get("routing_registration"),
        "distance_assignment": pair.get("distance_assignment"),
        "scenario_sha256": scenario_sha256,
    }


def _audit_registered_dataset(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require_q0_runtime_identity()
    dataset = output / "dataset"
    split = dataset / SPLIT
    summary_path = dataset / DATASET_SUMMARY_FILENAME
    q0_path = dataset / Q0_FILENAME
    q0_manifest_path = dataset / Q0_MANIFEST_FILENAME
    proposal_path = dataset / Q0_REGISTRATION_PROPOSAL_FILENAME
    manifest_path = split / "manifest.jsonl"
    if not all(
        path.is_file()
        for path in (
            summary_path,
            q0_path,
            q0_manifest_path,
            proposal_path,
            manifest_path,
        )
    ):
        raise ValueError("Warehouse compact-cut Q0 dataset evidence is incomplete")
    summary = _read_json(summary_path)
    q0 = _read_json(q0_path)
    proposal = _read_json(proposal_path)
    q0_rows = _read_jsonl(q0_manifest_path)
    manifest = _read_jsonl(manifest_path)
    generation_toolchain = dict(q0.get("q0_generation_toolchain") or {})
    if (
        summary.get("dataset_revision") != EXPERIMENT_ID
        or summary.get("configuration_fingerprint") != _generation_fingerprint(root)
        or int(summary.get("map_count", -1)) != 8
        or int(summary.get("task_count", -1)) != 32
        or str(summary.get("q0_manifest_sha256")) != sha256_file(q0_manifest_path)
        or str(summary.get("registration_proposal_sha256"))
        != sha256_file(proposal_path)
        or dict(summary.get("q0_byte_reproducibility") or {})
        != Q0_BYTE_REPRODUCIBILITY
        or dict(summary.get("q0_generation_toolchain") or {})
        != generation_toolchain
        or generation_toolchain not in _accepted_q0_runtime_identities()
        or dict(summary.get("static_pre_registration_record") or {})
        != STATIC_PRE_REGISTRATION_RECORD
        or q0.get("schema") != Q0_SCHEMA
        or q0.get("passed") is not True
        or int(q0.get("map_count", -1)) != 8
        or int(q0.get("task_count", -1)) != 32
        or q0.get("solver_or_controller_invoked") is not False
        or dict(q0.get("q0_byte_reproducibility") or {})
        != Q0_BYTE_REPRODUCIBILITY
        or dict(q0.get("static_pre_registration_record") or {})
        != STATIC_PRE_REGISTRATION_RECORD
        or len(q0_rows) != 32
        or len(manifest) != 32
    ):
        raise ValueError("Warehouse compact-cut Q0 dataset identity changed")
    expected_specs = _task_index(config)
    q0_by_task = {str(row.get("task_id")): dict(row) for row in q0_rows}
    manifest_by_task = {str(row.get("task_id")): dict(row) for row in manifest}
    if (
        set(q0_by_task) != set(expected_specs)
        or set(manifest_by_task) != set(expected_specs)
        or len(q0_by_task) != 32
        or len(manifest_by_task) != 32
    ):
        raise ValueError("Warehouse compact-cut Q0 task coverage changed")
    # Reject identity/path drift before invoking the comparatively expensive
    # deterministic task generator during the semantic re-audit.
    for task_id, spec in expected_specs.items():
        row = manifest_by_task[task_id]
        if (
            str(row.get("split")) != SPLIT
            or str(row.get("source_group")) != "generated"
            or str(row.get("instance_origin")) != "warehouse_compactcut_v1_q0_only"
            or str(row.get("map_id")) != str(spec["map_id"])
            or str(row.get("map_role")) != str(spec["map_role"])
            or str(row.get("task_id")) != task_id
            or str(row.get("task_variant")) != str(spec["variant"])
            or str(row.get("variant")) != str(spec["variant"])
            or int(row.get("task_seed", -1)) != int(spec["task_seed"])
            or int(row.get("agent_count", -1)) != AGENT_COUNT
            or str(row.get("map_file")) != f"maps/{spec['map_id']}.map"
            or str(row.get("map_metadata_file")) != f"maps/{spec['map_id']}.json"
            or str(row.get("scenario_file")) != f"scenarios/{task_id}.scen"
            or str(row.get("task_file")) != f"tasks/{task_id}.json"
            or str(row.get("layout_mode")) != "warehouse"
            or str(row.get("layout_variant")) != str(spec["map_id"])
            or str(row.get("scenario_type"))
            != f"warehouse_compactcut_{spec['variant']}"
        ):
            raise ValueError(f"Warehouse compact-cut manifest row changed: {task_id}")
    if registry is not None and str(registry["q0_manifest_sha256"]) != sha256_file(q0_manifest_path):
        raise ValueError("Warehouse compact-cut registered Q0 manifest changed")

    bank = _map_bank()
    registered_maps = dict(registry.get("maps") or {}) if registry else None
    map_inputs: dict[str, tuple[int, int, set[tuple[int, int]]]] = {}
    regenerated_pairs: dict[tuple[str, int], dict[str, Any]] = {}
    expected_geometry: dict[str, Any] = {}
    expected_map_registry: dict[str, Any] = {}
    expected_task_audits: list[dict[str, Any]] = []
    for map_id, map_data in bank.items():
        map_path = split / "maps" / f"{map_id}.map"
        metadata_path = split / "maps" / f"{map_id}.json"
        if not map_path.is_file() or map_path.read_text(encoding="utf-8") != _movingai_map_text(map_data):
            raise ValueError(f"Warehouse compact-cut map bytes changed: {map_id}")
        geometry = dict(audit_compactcut_geometry(map_data))
        if (
            geometry.get("passed") is not True
            or str(geometry.get("divider_template"))
            != str(MAP_BY_ID[map_id]["template"])
        ):
            raise ValueError(f"Warehouse compact-cut geometry audit failed: {map_id}")
        expected_geometry[map_id] = geometry
        map_sha = sha256_file(map_path)
        metadata = _read_json(metadata_path)
        if metadata != _normalized(_map_metadata(map_data, geometry, map_sha)):
            raise ValueError(f"Warehouse compact-cut map metadata changed: {map_id}")
        expected_map_registration = {
            "template": str(MAP_BY_ID[map_id]["template"]),
            "map_seed": int(MAP_BY_ID[map_id]["map_seed"]),
            "map_role": str(MAP_BY_ID[map_id]["map_role"]),
            "map_sha256": map_sha,
            "map_metadata_sha256": sha256_file(metadata_path),
            "geometry_sha256": _fingerprint(geometry),
        }
        expected_map_registry[map_id] = expected_map_registration
        if registered_maps is not None:
            registered_map = dict(registered_maps.get(map_id) or {})
            if registered_map != expected_map_registration:
                raise ValueError(f"Warehouse compact-cut registered map changed: {map_id}")
        rows, cols, _grid, passable = _movingai_passable_cells(map_path)
        map_inputs[map_id] = (rows, cols, passable)
        for task_seed in TASK_SEEDS:
            pair = dict(generate_compactcut_tasks(map_data, task_seed))
            pair_audit, distance_metrics = _validate_generated_pair(
                map_data, pair, task_seed
            )
            expected_task_audits.append(
                {
                    "map_id": map_id,
                    "map_role": str(MAP_BY_ID[map_id]["map_role"]),
                    "task_seed": task_seed,
                    "audit": pair_audit,
                    "distance_metrics": distance_metrics,
                }
            )
            regenerated_pairs[(map_id, task_seed)] = pair

    if len({str(row["map_sha256"]) for row in expected_map_registry.values()}) != 8:
        raise ValueError("Warehouse compact-cut Q0 map grid SHA-256 values are not unique")

    registered_tasks = dict(registry.get("tasks") or {}) if registry else None
    for task_id, spec in expected_specs.items():
        row = manifest_by_task[task_id]
        q0_row = q0_by_task[task_id]
        expected_map_file = f"maps/{spec['map_id']}.map"
        expected_metadata_file = f"maps/{spec['map_id']}.json"
        expected_scenario_file = f"scenarios/{task_id}.scen"
        expected_task_file = f"tasks/{task_id}.json"
        if (
            str(row.get("split")) != SPLIT
            or str(row.get("source_group")) != "generated"
            or str(row.get("instance_origin")) != "warehouse_compactcut_v1_q0_only"
            or str(row.get("map_id")) != str(spec["map_id"])
            or str(row.get("map_role")) != str(spec["map_role"])
            or str(row.get("task_id")) != task_id
            or str(row.get("task_variant")) != str(spec["variant"])
            or str(row.get("variant")) != str(spec["variant"])
            or int(row.get("task_seed", -1)) != int(spec["task_seed"])
            or int(row.get("agent_count", -1)) != AGENT_COUNT
            or str(row.get("map_file")) != expected_map_file
            or str(row.get("map_metadata_file")) != expected_metadata_file
            or str(row.get("scenario_file")) != expected_scenario_file
            or str(row.get("task_file")) != expected_task_file
            or str(row.get("layout_mode")) != "warehouse"
            or str(row.get("layout_variant")) != str(spec["map_id"])
            or str(row.get("scenario_type")) != f"warehouse_compactcut_{spec['variant']}"
        ):
            raise ValueError(f"Warehouse compact-cut manifest row changed: {task_id}")
        _contained_file(split, expected_map_file, f"map for {task_id}")
        _contained_file(split, expected_metadata_file, f"metadata for {task_id}")
        scenario_path = _contained_file(split, expected_scenario_file, f"scenario for {task_id}")
        task_path = _contained_file(split, expected_task_file, f"task for {task_id}")
        hashes = {"scenario_sha256": sha256_file(scenario_path), "task_sha256": sha256_file(task_path)}
        pair = regenerated_pairs[(str(spec["map_id"]), int(spec["task_seed"]))]
        expected_task = _expected_task_payload(pair, str(spec["variant"]), hashes["scenario_sha256"])
        task = _read_json(task_path)
        if task != _normalized(expected_task):
            raise ValueError(f"Warehouse compact-cut task metadata changed: {task_id}")
        rows, cols, passable = map_inputs[str(spec["map_id"])]
        _parse_and_audit_scenario(
            scenario_path,
            map_name=f"{spec['map_id']}.map",
            rows=rows,
            cols=cols,
            passable=passable,
            task=task,
        )
        distance_metrics = _pair_distance_metrics(pair)
        expected_q0 = {
            "schema": Q0_MANIFEST_SCHEMA,
            "task_id": task_id,
            "map_id": str(spec["map_id"]),
            "map_role": str(spec["map_role"]),
            "variant": str(spec["variant"]),
            "task_seed": int(spec["task_seed"]),
            "agent_count": AGENT_COUNT,
            "scenario_sha256": hashes["scenario_sha256"],
            "task_sha256": hashes["task_sha256"],
            "endpoint_payload_sha256": _endpoint_payload_sha256(task),
            "pair_payload_sha256": _canonical_pair_payload_sha256(pair),
            "pair_distance_mean_relative_difference": distance_metrics["mean_relative_difference"],
            "pair_distance_p95_relative_difference": distance_metrics["p95_relative_difference"],
            "joint_feasibility_witness_sha256": _fingerprint(
                _pair_witness(pair, _variant_task(pair, str(spec["variant"])))
            ),
            "passed": True,
        }
        if q0_row != _normalized(expected_q0):
            raise ValueError(f"Warehouse compact-cut Q0 row changed: {task_id}")
        if registered_tasks is not None:
            registered = dict(registered_tasks.get(task_id) or {})
            if registered != {
                "map_id": str(spec["map_id"]),
                "map_role": str(spec["map_role"]),
                "variant": str(spec["variant"]),
                "task_seed": int(spec["task_seed"]),
                "agent_count": AGENT_COUNT,
                "scenario_sha256": hashes["scenario_sha256"],
                "task_sha256": hashes["task_sha256"],
                "endpoint_payload_sha256": expected_q0["endpoint_payload_sha256"],
                "pair_payload_sha256": expected_q0["pair_payload_sha256"],
                "joint_feasibility_witness_sha256": expected_q0[
                    "joint_feasibility_witness_sha256"
                ],
            }:
                raise ValueError(f"Warehouse compact-cut registered task bytes changed: {task_id}")
    expected_q0 = {
        "schema": Q0_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "passed": True,
        "map_count": 8,
        "task_count": 32,
        "q0_byte_reproducibility": Q0_BYTE_REPRODUCIBILITY,
        "q0_generation_toolchain": generation_toolchain,
        "static_pre_registration_record": STATIC_PRE_REGISTRATION_RECORD,
        "map_geometry": expected_geometry,
        "task_audits": expected_task_audits,
        "solver_or_controller_invoked": False,
        "controller_step_invoked": False,
        "policy_episode_invoked": False,
        "formal_ttf_invoked": False,
        "control_interpretation": config["result_interpretation"]["control_label"],
    }
    if q0 != _normalized(expected_q0):
        raise ValueError("Warehouse compact-cut Q0 semantic audit changed")
    expected_proposal = {
        "schema": REGISTRY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "map_count": 8,
        "task_count": 32,
        "q0_manifest_sha256": sha256_file(q0_manifest_path),
        "task_generator_sha256": sha256_file(
            root / "experiments" / "warehouse_compactcut_tasks.py"
        ),
        "q0_byte_reproducibility": Q0_BYTE_REPRODUCIBILITY,
        "q0_generation_toolchain": generation_toolchain,
        "static_pre_registration_record": STATIC_PRE_REGISTRATION_RECORD,
        "maps": expected_map_registry,
        "tasks": {
            str(row["task_id"]): {
                field: row[field]
                for field in (
                    "map_id",
                    "map_role",
                    "variant",
                    "task_seed",
                    "agent_count",
                    "scenario_sha256",
                    "task_sha256",
                    "endpoint_payload_sha256",
                    "pair_payload_sha256",
                    "joint_feasibility_witness_sha256",
                )
            }
            for row in q0_rows
        },
    }
    if proposal != _normalized(expected_proposal):
        raise ValueError("Warehouse compact-cut Q0 registration proposal changed")
    if registry is not None and registry != proposal:
        raise ValueError("Warehouse compact-cut tracked registry differs from Q0 proposal")
    return summary


def prepare_q0(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    _path, root, config, registry = load_config(config_path)
    q0_toolchain = _require_q0_runtime_identity()
    output = Path(output).resolve()
    dataset = output / "dataset"
    summary_path = dataset / DATASET_SUMMARY_FILENAME
    if summary_path.is_file():
        return _audit_registered_dataset(root, output, config, registry)
    if dataset.is_dir() and any(dataset.iterdir()):
        raise ValueError("Warehouse compact-cut Q0 dataset is non-empty without trusted identity")

    split = dataset / SPLIT
    bank = _map_bank()
    map_geometry: dict[str, Any] = {}
    map_registry: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    q0_rows: list[dict[str, Any]] = []
    task_audits: list[dict[str, Any]] = []
    map_grid_sha256s: set[str] = set()
    for map_spec in MAPS:
        map_id = str(map_spec["id"])
        map_data = bank[map_id]
        geometry = dict(audit_compactcut_geometry(map_data))
        if (
            geometry.get("passed") is not True
            or str(geometry.get("divider_template")) != str(map_spec["template"])
        ):
            raise RuntimeError(f"Warehouse compact-cut Q0 geometry failed: {map_id}")
        map_geometry[map_id] = geometry
        map_path = split / "maps" / f"{map_id}.map"
        write_movingai_map(map_path, map_data)
        map_sha = sha256_file(map_path)
        if map_sha in map_grid_sha256s:
            raise RuntimeError(
                f"Warehouse compact-cut Q0 map grid SHA-256 is duplicated: {map_id}"
            )
        map_grid_sha256s.add(map_sha)
        metadata_path = split / "maps" / f"{map_id}.json"
        _write_json(metadata_path, _map_metadata(map_data, geometry, map_sha))
        map_registry[map_id] = {
            "template": str(map_spec["template"]),
            "map_seed": int(map_spec["map_seed"]),
            "map_role": str(map_spec["map_role"]),
            "map_sha256": map_sha,
            "map_metadata_sha256": sha256_file(metadata_path),
            "geometry_sha256": _fingerprint(geometry),
        }
        topology = _map_metrics(map_path)
        for task_seed in TASK_SEEDS:
            pair = dict(generate_compactcut_tasks(map_data, task_seed))
            pair_audit, distance_metrics = _validate_generated_pair(map_data, pair, task_seed)
            task_audits.append(
                {
                    "map_id": map_id,
                    "map_role": str(map_spec["map_role"]),
                    "task_seed": task_seed,
                    "audit": pair_audit,
                    "distance_metrics": distance_metrics,
                }
            )
            for variant in TASK_VARIANTS:
                task = _variant_task(pair, variant)
                task_id = str(task["task_id"])
                starts = [tuple(map(int, cell)) for cell in task["starts"]]
                goals = [tuple(map(int, cell)) for cell in task["goals"]]
                distances = list(map(int, task["shortest_path_distances"]))
                scenario_path = split / "scenarios" / f"{task_id}.scen"
                _write_derived_scenario(
                    scenario_path,
                    map_path.name,
                    map_data.rows,
                    map_data.cols,
                    starts,
                    goals,
                    distances,
                )
                task_path = split / "tasks" / f"{task_id}.json"
                task_payload = _expected_task_payload(pair, variant, sha256_file(scenario_path))
                _write_json(task_path, task_payload)
                witness_sha = _fingerprint(_pair_witness(pair, task))
                q0_rows.append(
                    {
                        "schema": Q0_MANIFEST_SCHEMA,
                        "task_id": task_id,
                        "map_id": map_id,
                        "map_role": str(map_spec["map_role"]),
                        "variant": variant,
                        "task_seed": task_seed,
                        "agent_count": AGENT_COUNT,
                        "scenario_sha256": sha256_file(scenario_path),
                        "task_sha256": sha256_file(task_path),
                        "endpoint_payload_sha256": _endpoint_payload_sha256(
                            task_payload
                        ),
                        "pair_payload_sha256": distance_metrics[
                            "pair_payload_sha256"
                        ],
                        "pair_distance_mean_relative_difference": distance_metrics[
                            "mean_relative_difference"
                        ],
                        "pair_distance_p95_relative_difference": distance_metrics[
                            "p95_relative_difference"
                        ],
                        "joint_feasibility_witness_sha256": witness_sha,
                        "passed": True,
                    }
                )
                manifest.append(
                    {
                        "split": SPLIT,
                        "source_group": "generated",
                        "instance_origin": "warehouse_compactcut_v1_q0_only",
                        "map_id": map_id,
                        "map_role": str(map_spec["map_role"]),
                        "task_id": task_id,
                        "map_file": f"maps/{map_path.name}",
                        "scenario_file": f"scenarios/{scenario_path.name}",
                        "map_metadata_file": f"maps/{metadata_path.name}",
                        "task_file": f"tasks/{task_path.name}",
                        "layout_mode": "warehouse",
                        "layout_variant": map_id,
                        "scenario_type": f"warehouse_compactcut_{variant}",
                        "task_variant": variant,
                        "variant": variant,
                        "task_seed": task_seed,
                        "agent_count": AGENT_COUNT,
                        "topology_metrics": topology,
                        "dominant_flow_ratio": 0.0,
                        "hotspot_skew": 0.0,
                        "required_bottleneck_crossing_ratio": 1.0 if variant == STRUCTURED_VARIANT else 0.0,
                        "mean_shortest_distance": sum(distances) / len(distances),
                    }
                )
    manifest.sort(key=lambda row: str(row["task_id"]))
    q0_rows.sort(key=lambda row: str(row["task_id"]))
    if len(map_grid_sha256s) != 8:
        raise RuntimeError("Warehouse compact-cut Q0 requires eight unique map grid SHA-256 values")
    if len(manifest) != 32 or len(q0_rows) != 32:
        raise RuntimeError("Warehouse compact-cut Q0 task count changed")
    _write_jsonl(split / "manifest.jsonl", manifest)
    _write_jsonl(dataset / Q0_MANIFEST_FILENAME, q0_rows)
    q0 = {
        "schema": Q0_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "passed": True,
        "map_count": 8,
        "task_count": 32,
        "q0_byte_reproducibility": Q0_BYTE_REPRODUCIBILITY,
        "q0_generation_toolchain": q0_toolchain,
        "static_pre_registration_record": STATIC_PRE_REGISTRATION_RECORD,
        "map_geometry": map_geometry,
        "task_audits": task_audits,
        "solver_or_controller_invoked": False,
        "controller_step_invoked": False,
        "policy_episode_invoked": False,
        "formal_ttf_invoked": False,
        "control_interpretation": config["result_interpretation"]["control_label"],
    }
    _write_json(dataset / Q0_FILENAME, q0)
    q0_manifest_sha = sha256_file(dataset / Q0_MANIFEST_FILENAME)
    proposal = {
        "schema": REGISTRY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "map_count": 8,
        "task_count": 32,
        "q0_byte_reproducibility": Q0_BYTE_REPRODUCIBILITY,
        "q0_generation_toolchain": q0_toolchain,
        "static_pre_registration_record": STATIC_PRE_REGISTRATION_RECORD,
        "q0_manifest_sha256": q0_manifest_sha,
        "task_generator_sha256": sha256_file(
            root / "experiments" / "warehouse_compactcut_tasks.py"
        ),
        "maps": map_registry,
        "tasks": {
            str(row["task_id"]): {
                field: row[field]
                for field in (
                    "map_id",
                    "map_role",
                    "variant",
                    "task_seed",
                    "agent_count",
                    "scenario_sha256",
                    "task_sha256",
                    "endpoint_payload_sha256",
                    "pair_payload_sha256",
                    "joint_feasibility_witness_sha256",
                )
            }
            for row in q0_rows
        },
    }
    _write_json(dataset / Q0_REGISTRATION_PROPOSAL_FILENAME, proposal)
    summary = {
        "schema_version": 1,
        "dataset_revision": EXPERIMENT_ID,
        "configuration_fingerprint": _generation_fingerprint(root),
        "source": "eight deterministically generated compact Warehouse maps",
        "task_semantics": "Q0-only paired mandatory-cut task registration",
        "map_count": 8,
        "task_count": 32,
        "q0_byte_reproducibility": Q0_BYTE_REPRODUCIBILITY,
        "q0_generation_toolchain": q0_toolchain,
        "static_pre_registration_record": STATIC_PRE_REGISTRATION_RECORD,
        "q0_manifest_sha256": q0_manifest_sha,
        "registration_proposal_sha256": sha256_file(
            dataset / Q0_REGISTRATION_PROPOSAL_FILENAME
        ),
        "solver_or_controller_invoked": False,
    }
    _write_json(dataset / DATASET_SUMMARY_FILENAME, summary)
    _audit_registered_dataset(root, output, config, registry)
    return {
        **summary,
        "q0_passed": True,
        "registration_status": config["registration"]["status"],
        "q1_allowed": registry is not None,
    }


def _qualification_failure_result(
    job: dict[str, Any], status: str, error: str
) -> dict[str, Any]:
    row = dict(job["row"])
    return {
        "split": str(row["split"]),
        "map_id": str(row["map_id"]),
        "map_role": str(row["map_role"]),
        "task_id": str(row["task_id"]),
        "layout_mode": str(row["layout_mode"]),
        "task_variant": str(row["task_variant"]),
        "agent_count": int(row["agent_count"]),
        "solver_seed": int(job["solver_seed"]),
        "status": status,
        "error": error,
    }


def _qualification_report(
    config: Mapping[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    expected = {_schedule_key(row): row for row in qualification_schedule(config)}
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_keys: list[list[Any]] = []
    for raw in rows:
        row = dict(raw)
        key = _schedule_key(row)
        if key in indexed:
            duplicate_keys.append([key[0], key[1]])
        indexed[key] = row
    unexpected_keys = sorted([[task_id, seed] for task_id, seed in set(indexed) - set(expected)])
    tasks: list[dict[str, Any]] = []
    execution_errors: list[dict[str, Any]] = []
    for key, row in sorted(indexed.items()):
        item = expected.get(key)
        status = str(row.get("status"))
        if status != "ok":
            execution_errors.append(
                {"task_id": key[0], "solver_seed": key[1], "status": status, "error": str(row.get("error"))}
            )
            continue
        if item is None:
            continue
        complexity = dict(row.get("initial_complexity") or {})
        conflicts = int(row.get("initial_conflicts", -1))
        initial_complete = row.get("initial_complete") is True
        reported_feasible = row.get("initial_feasible") is True
        expected_feasible = bool(initial_complete and conflicts == 0)
        tasks.append(
            {
                "map_id": str(item["map_id"]),
                "map_role": str(item["map_role"]),
                "task_id": str(item["task_id"]),
                "variant": str(item["variant"]),
                "task_seed": int(item["task_seed"]),
                "agent_count": AGENT_COUNT,
                "solver_seed": int(item["solver_seed"]),
                "initial_conflicts": conflicts,
                "active_conflict_agent_count": int(complexity.get("active_conflict_agent_count", -1)),
                "largest_conflict_component_size": int(complexity.get("largest_conflict_component_size", -1)),
                "initial_complete": initial_complete,
                "initial_feasible": expected_feasible,
                "reported_initial_feasible": reported_feasible,
                "initial_state_consistent": reported_feasible == expected_feasible,
                "state_fingerprint": str(row.get("state_fingerprint") or ""),
            }
        )
    by_seed = {
        seed: tuple(
            str(row["state_fingerprint"])
            for row in sorted(
                (task for task in tasks if int(task["solver_seed"]) == seed),
                key=lambda task: str(task["task_id"]),
            )
        )
        for seed in SOLVER_SEEDS
    }
    duplicate_seed_trajectories = [
        [left, right]
        for left_index, left in enumerate(SOLVER_SEEDS)
        for right in SOLVER_SEEDS[left_index + 1 :]
        if by_seed[left] and by_seed[left] == by_seed[right]
    ]
    return {
        "schema": QUALIFICATION_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": "q1_reset_only",
        "expected_reset_count": 128,
        "observed_result_count": len(indexed),
        "valid_reset_count": len(tasks),
        "execution_error_count": sum(row["status"] == "error" for row in execution_errors),
        "process_timeout_count": sum(row["status"] == "timeout" for row in execution_errors),
        "duplicate_keys": duplicate_keys,
        "unexpected_keys": unexpected_keys,
        "duplicate_solver_seed_trajectories": duplicate_seed_trajectories,
        "all_expected_results_present": set(indexed) == set(expected),
        "controller_step_invoked": False,
        "policy_episode_invoked": False,
        "formal_ttf_invoked": False,
        "tasks": sorted(tasks, key=lambda row: (str(row["task_id"]), int(row["solver_seed"]))),
        "execution_errors": execution_errors,
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metric(name: str) -> dict[str, Any]:
        values = [int(row[name]) for row in rows]
        return {
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "zero_count": sum(value == 0 for value in values),
        }
    return {
        "reset_count": len(rows),
        "initial_conflicts": metric("initial_conflicts"),
        "active_conflict_agents": metric("active_conflict_agent_count"),
        "largest_conflict_component_size": metric("largest_conflict_component_size"),
        "initial_feasible_count": sum(row.get("initial_feasible") is True for row in rows),
    }


def select_qualified_benchmark(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    tasks = [dict(row) for row in report.get("tasks") or ()]
    indexed = {_schedule_key(row): row for row in tasks}
    expected = {_schedule_key(row): row for row in qualification_schedule(config)}
    global_errors: list[str] = []
    if int(report.get("execution_error_count", -1)) != 0:
        global_errors.append("Q1 contains an execution error")
    if int(report.get("process_timeout_count", -1)) != 0:
        global_errors.append("Q1 contains a process timeout")
    if list(report.get("duplicate_keys") or ()):
        global_errors.append("Q1 contains duplicate reset keys")
    if list(report.get("unexpected_keys") or ()):
        global_errors.append("Q1 contains unexpected reset keys")
    if report.get("all_expected_results_present") is not True or set(indexed) != set(expected) or len(indexed) != 128:
        global_errors.append("Q1 reset coverage differs from the frozen 128 keys")
    invalid_common = [
        row
        for row in tasks
        if row.get("initial_complete") is not True
        or row.get("initial_state_consistent") is not True
        or not _is_sha256(row.get("state_fingerprint"))
    ]
    if invalid_common:
        global_errors.append("Q1 contains incomplete, inconsistent, or unidentified initial states")

    structured = [row for row in tasks if str(row.get("variant")) == STRUCTURED_VARIANT]
    controls = [row for row in tasks if str(row.get("variant")) == CONTROL_VARIANT]
    structured_failures = [
        {
            "task_id": str(row["task_id"]),
            "solver_seed": int(row["solver_seed"]),
            "initial_conflicts": int(row["initial_conflicts"]),
            "active_conflict_agent_count": int(row["active_conflict_agent_count"]),
            "largest_conflict_component_size": int(row["largest_conflict_component_size"]),
            "initial_feasible": bool(row["initial_feasible"]),
        }
        for row in structured
        if row.get("initial_feasible") is not False
        or int(row.get("initial_conflicts", -1)) < 16
        or int(row.get("active_conflict_agent_count", -1)) < 32
        or int(row.get("largest_conflict_component_size", -1)) < 16
    ]
    structured_pass_count = len(structured) - len(structured_failures)
    control_complete = len(controls) == 64 and all(
        row.get("initial_complete") is True and row.get("initial_state_consistent") is True
        for row in controls
    )
    benchmark_ready = bool(
        not global_errors
        and len(structured) == 64
        and structured_pass_count == 64
        and control_complete
    )
    per_map = {
        map_id: {
            variant: _distribution(
                [row for row in tasks if str(row["map_id"]) == map_id and str(row["variant"]) == variant]
            )
            for variant in TASK_VARIANTS
        }
        for map_id in MAP_BY_ID
    }
    per_role = {
        map_role: {
            variant: _distribution(
                [
                    row
                    for row in tasks
                    if str(row["map_role"]) == map_role
                    and str(row["variant"]) == variant
                ]
            )
            for variant in TASK_VARIANTS
        }
        for map_role in ("development", "controller_held_out")
    }
    return {
        "schema": SELECTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "reset_only_benchmark_qualification_not_a_ttf_claim",
        "benchmark_ready": benchmark_ready,
        "decision": (
            "benchmark_ready_for_separately_preregistered_controller_evaluation"
            if benchmark_ready
            else "stop_do_not_resample_replace_or_run_controller"
        ),
        "map_count": 8,
        "task_count": 32,
        "expected_reset_count": 128,
        "structured_expected_reset_count": 64,
        "structured_observed_reset_count": len(structured),
        "structured_pass_count": structured_pass_count,
        "structured_failure_count": len(structured_failures),
        "structured_failures": structured_failures,
        "diagnostic_control_expected_reset_count": 64,
        "diagnostic_control_observed_reset_count": len(controls),
        "diagnostic_control_complete_and_consistent": control_complete,
        "diagnostic_control_structure_thresholds_gate_benchmark_ready": False,
        "per_map_distributions": per_map,
        "per_role_distributions": per_role,
        "global_errors": global_errors,
        "controller_performance_claim": False,
        "ttf_claim": False,
        "control_interpretation": config["result_interpretation"]["control_label"],
        "claim_boundary": config["cohort"]["claim_boundary"],
    }


def _audit_qualification_evidence(
    config: Mapping[str, Any],
    qualification: Path,
    *,
    require_complete: bool,
    registry: Mapping[str, Any],
    runtime_preflight_run_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    manifest_path = qualification / QUALIFICATION_MANIFEST_FILENAME
    rows = _read_jsonl(manifest_path) if manifest_path.is_file() else []
    expected = {_schedule_key(row): row for row in qualification_schedule(config)}
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = _schedule_key(row)
        if key in indexed:
            raise ValueError("Warehouse compact-cut Q1 manifest has duplicate keys")
        if key not in expected:
            raise ValueError("Warehouse compact-cut Q1 manifest has an unknown key")
        item = expected[key]
        registry_fields = dict(row.get("registered_task_identity") or {})
        registered = {
            "map_id": str(item["map_id"]),
            "map_role": str(item["map_role"]),
            "variant": str(item["variant"]),
            "task_seed": int(item["task_seed"]),
            "agent_count": AGENT_COUNT,
        }
        registered_task = dict(registry["tasks"])[str(item["task_id"])]
        if (
            str(row.get("split")) != SPLIT
            or str(row.get("map_id")) != str(item["map_id"])
            or str(row.get("map_role")) != str(item["map_role"])
            or str(row.get("task_id")) != str(item["task_id"])
            or str(row.get("task_variant")) != str(item["variant"])
            or int(row.get("agent_count", -1)) != AGENT_COUNT
            or int(row.get("solver_seed", -1)) != int(item["solver_seed"])
            or registry_fields != registered
            or str(row.get("registered_scenario_sha256")) != str(registered_task["scenario_sha256"])
            or str(row.get("registered_task_sha256")) != str(registered_task["task_sha256"])
            or str(row.get("registered_map_sha256"))
            != str(dict(registry["maps"])[str(item["map_id"])]["map_sha256"])
            or str(row.get("registered_q0_manifest_sha256")) != str(registry["q0_manifest_sha256"])
            or str(row.get("runtime_preflight_run_fingerprint")) != runtime_preflight_run_fingerprint
            or str(row.get("status")) not in {"ok", "error", "timeout"}
        ):
            raise ValueError(f"Warehouse compact-cut Q1 row identity changed: {key}")
        if str(row.get("status")) == "ok":
            complexity = dict(row.get("initial_complexity") or {})
            if (
                row.get("error") is not None
                or not isinstance(row.get("initial_conflicts"), int)
                or int(complexity.get("agent_count", -1)) != AGENT_COUNT
                or int(complexity.get("conflict_pair_count", -1)) != int(row["initial_conflicts"])
                or int(complexity.get("active_conflict_agent_count", -1)) < 0
                or int(complexity.get("largest_conflict_component_size", -1)) < 0
                or not _is_sha256(row.get("state_fingerprint"))
            ):
                raise ValueError(f"Warehouse compact-cut Q1 reset evidence changed: {key}")
        elif not str(row.get("error") or ""):
            raise ValueError(f"Warehouse compact-cut Q1 failure lacks an error: {key}")
        indexed[key] = row
    if require_complete and set(indexed) != set(expected):
        raise ValueError("Warehouse compact-cut Q1 evidence is incomplete")
    report_path = qualification / QUALIFICATION_REPORT_FILENAME
    stored_report = _read_json(report_path) if report_path.is_file() else None
    recomputed = _qualification_report(config, rows)
    if stored_report is not None and stored_report != recomputed:
        raise ValueError("Warehouse compact-cut Q1 report/manifest mismatch")
    if require_complete and stored_report is None:
        raise ValueError("Warehouse compact-cut Q1 report is missing")
    return rows, stored_report


def _runtime_preflight(root: Path, config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    runtime_path = _resolve_registered_input(
        root, dict(config["inputs"])["runtime_config"], "Warehouse compact-cut runtime"
    )
    keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in qualification_schedule(config)}
    return run_closed_loop_collection(
        output / "dataset",
        runtime_path,
        output / "qualification-runtime-preflight-not-written",
        phase="qualify",
        workers=16,
        dry_run=True,
        controller="official_adaptive",
        job_keys=keys,
        cohort_job_keys=keys,
        qualification_process_timeout_seconds=240.0,
    )


def _producer(root: Path) -> dict[str, Any]:
    return closed_loop_producer_identity(project_root=root, source_files=PRODUCER_SOURCE_FILES, native_required=True)


def _dataset_rows(output: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(output / "dataset" / SPLIT / "manifest.jsonl")
    indexed = {str(row.get("task_id")): dict(row) for row in rows}
    if len(rows) != 32 or len(indexed) != 32:
        raise ValueError("Warehouse compact-cut Q1 dataset manifest changed")
    return indexed


def _write_qualification_status(
    qualification: Path,
    base: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    complete: bool,
    terminal_error: bool,
    benchmark_ready: bool = False,
    report_sha256: str | None = None,
) -> None:
    errors = [row for row in rows if str(row.get("status")) != "ok"]
    payload = {
        **dict(base),
        "completed_schedule_entries": len(rows),
        "complete": complete,
        "terminal_error": terminal_error,
        "error_jobs": sum(str(row.get("status")) == "error" for row in errors),
        "timeout_jobs": sum(str(row.get("status")) == "timeout" for row in errors),
        "benchmark_ready": benchmark_ready,
        "controller_step_invoked": False,
        "policy_episode_invoked": False,
        "formal_ttf_invoked": False,
    }
    if report_sha256 is not None:
        payload["report_sha256"] = report_sha256
    _write_json(qualification / QUALIFICATION_STATUS_FILENAME, payload)


def qualify(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config, registry = load_config(config_path, require_frozen_registry=True)
    assert registry is not None
    output = Path(output).resolve()
    prepare_q0(path, output)
    _audit_registered_dataset(root, output, config, registry)
    preflight = _runtime_preflight(root, config, output)
    if dry_run:
        return {
            **plan(path),
            "command": "qualify",
            "dry_run": True,
            "runtime_preflight_run_fingerprint": preflight["run_fingerprint"],
            "native_reset_invoked": False,
        }

    qualification = output / "qualification"
    schedule = [
        {**row, "runtime_preflight_run_fingerprint": preflight["run_fingerprint"]}
        for row in qualification_schedule(config)
    ]
    producer = _producer(root)
    prepared = prepare_resumable_output(
        qualification,
        status_filename=QUALIFICATION_STATUS_FILENAME,
        status_schema=QUALIFICATION_STATUS_SCHEMA,
        config_path=path,
        schedule=schedule,
        producer=producer,
        resume=resume,
        report_filename=SELECTION_FILENAME,
        report_schema=SELECTION_SCHEMA,
        label="Warehouse compact-cut Q1 qualification",
    )
    if prepared.completed_report is not None:
        _rows, report = _audit_qualification_evidence(
            config,
            qualification,
            require_complete=True,
            registry=registry,
            runtime_preflight_run_fingerprint=preflight["run_fingerprint"],
        )
        assert report is not None
        recomputed = select_qualified_benchmark(config, report)
        stored = dict(prepared.completed_report)
        for field in (
            "config_sha256",
            "q0_manifest_sha256",
            "qualification_manifest_sha256",
            "qualification_report_sha256",
            "producer_identity",
            "runtime_preflight_run_fingerprint",
        ):
            recomputed[field] = stored[field]
        if stored != recomputed:
            raise ValueError("Warehouse compact-cut completed Q1 result changed")
        return stored

    existing, _stored_report = _audit_qualification_evidence(
        config,
        qualification,
        require_complete=False,
        registry=registry,
        runtime_preflight_run_fingerprint=preflight["run_fingerprint"],
    )
    if int(prepared.status.get("completed_schedule_entries", -1)) != len(existing):
        raise ValueError("Warehouse compact-cut Q1 resume progress changed")
    if prepared.status.get("terminal_error") is True or any(str(row.get("status")) != "ok" for row in existing):
        raise ValueError(
            "Warehouse compact-cut Q1 output is terminal after an execution error or "
            "process timeout; use a new preregistered experiment"
        )
    if (qualification / QUALIFICATION_REPORT_FILENAME).is_file():
        raise ValueError("Warehouse compact-cut incomplete Q1 unexpectedly has a final report")

    schedule_index = {_schedule_key(row): index for index, row in enumerate(schedule)}
    schedule_by_key = {_schedule_key(row): row for row in schedule}
    results_by_key = {_schedule_key(row): dict(row) for row in existing}
    pending = [row for row in schedule if _schedule_key(row) not in results_by_key]
    dataset_by_task = _dataset_rows(output)
    runtime_environment = _read_json(
        _resolve_registered_input(
            root, dict(config["inputs"])["runtime_config"], "Warehouse compact-cut runtime"
        )
    )["environment"]
    jobs = [
        {
            "row": dataset_by_task[str(item["task_id"])],
            "solver_seed": int(item["solver_seed"]),
            "dataset_root": str(output / "dataset"),
            "environment": runtime_environment,
        }
        for item in pending
    ]

    def ordered_rows() -> list[dict[str, Any]]:
        return [results_by_key[key] for key in sorted(results_by_key, key=schedule_index.__getitem__)]

    def record(result: dict[str, Any]) -> None:
        key = _schedule_key(result)
        if key not in schedule_index or key in results_by_key:
            raise ValueError("Warehouse compact-cut Q1 worker returned a bad key")
        item = schedule_by_key[key]
        registered_identity = {
            "map_id": str(item["map_id"]),
            "map_role": str(item["map_role"]),
            "variant": str(item["variant"]),
            "task_seed": int(item["task_seed"]),
            "agent_count": AGENT_COUNT,
        }
        registered_task = dict(registry["tasks"])[str(item["task_id"])]
        registered_map = dict(registry["maps"])[str(item["map_id"])]
        results_by_key[key] = {
            **dict(result),
            "map_role": str(item["map_role"]),
            "registered_task_identity": registered_identity,
            "registered_scenario_sha256": registered_task["scenario_sha256"],
            "registered_task_sha256": registered_task["task_sha256"],
            "registered_map_sha256": registered_map["map_sha256"],
            "registered_q0_manifest_sha256": registry["q0_manifest_sha256"],
            "runtime_preflight_run_fingerprint": preflight["run_fingerprint"],
        }
        rows = ordered_rows()
        _write_jsonl(qualification / QUALIFICATION_MANIFEST_FILENAME, rows)
        terminal = str(result.get("status")) in {"error", "timeout"}
        _write_qualification_status(
            qualification, prepared.base_status, rows, complete=False, terminal_error=terminal
        )

    if not existing:
        _write_jsonl(qualification / QUALIFICATION_MANIFEST_FILENAME, [])
        _write_qualification_status(
            qualification, prepared.base_status, [], complete=False, terminal_error=False
        )
    if jobs:
        with _CollectionRunLock(
            qualification,
            str(prepared.base_status["run_fingerprint"]),
            "warehouse-compactcut-q1-reset-only",
        ):
            _run_jobs(
                _qualification_worker,
                jobs,
                16,
                phase="warehouse-compactcut-q1-reset-only",
                output_root=qualification,
                run_fingerprint=str(prepared.base_status["run_fingerprint"]),
                timeout_seconds=240.0,
                on_result=record,
                failure_result=_qualification_failure_result,
                stop_on_failure=True,
            )

    rows = ordered_rows()
    failed = [row for row in rows if str(row.get("status")) != "ok"]
    if len(rows) != 128 and not failed:
        raise RuntimeError(
            "Warehouse compact-cut Q1 stopped without an execution error but did not "
            "cover all 128 resets; resume the identical output"
        )
    report = _qualification_report(config, rows)
    _write_json(qualification / QUALIFICATION_REPORT_FILENAME, report)
    if not failed:
        _audit_qualification_evidence(
            config,
            qualification,
            require_complete=True,
            registry=registry,
            runtime_preflight_run_fingerprint=preflight["run_fingerprint"],
        )
    selection = select_qualified_benchmark(config, report)
    selection.update(
        {
            "terminal_error": bool(failed),
            "config_sha256": sha256_file(path),
            "q0_manifest_sha256": sha256_file(output / "dataset" / Q0_MANIFEST_FILENAME),
            "qualification_manifest_sha256": sha256_file(
                qualification / QUALIFICATION_MANIFEST_FILENAME
            ),
            "qualification_report_sha256": sha256_file(
                qualification / QUALIFICATION_REPORT_FILENAME
            ),
            "producer_identity": producer,
            "runtime_preflight_run_fingerprint": preflight["run_fingerprint"],
        }
    )
    _write_json(qualification / SELECTION_FILENAME, selection)
    _write_qualification_status(
        qualification,
        prepared.base_status,
        rows,
        complete=not failed,
        terminal_error=bool(failed),
        benchmark_ready=bool(selection["benchmark_ready"]),
        report_sha256=(sha256_file(qualification / SELECTION_FILENAME) if not failed else None),
    )
    return selection


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return qualify(config_path, output, resume=resume, dry_run=dry_run)


__all__ = [
    "AGENT_COUNT",
    "CONFIG_SCHEMA",
    "CONTROL_VARIANT",
    "EXPERIMENT_ID",
    "MAPS",
    "Q0_FILENAME",
    "Q0_MANIFEST_FILENAME",
    "Q0_REGISTRATION_PROPOSAL_FILENAME",
    "QUALIFICATION_REPORT_SCHEMA",
    "QUALIFICATION_STATUS_SCHEMA",
    "REGISTRY_SCHEMA",
    "SELECTION_SCHEMA",
    "SOLVER_SEEDS",
    "STRUCTURED_VARIANT",
    "TASK_SEEDS",
    "TASK_VARIANTS",
    "benchmark_task_specs",
    "load_config",
    "plan",
    "prepare_q0",
    "qualification_schedule",
    "qualify",
    "run",
    "select_qualified_benchmark",
]
