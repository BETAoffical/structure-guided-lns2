from __future__ import annotations

import collections
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.stride_warehouse_disruption_recovery import (
    INCUMBENT_SUPPLY_ERROR,
    _checkpoint_worker,
    _ensure_dataset,
    _project_path,
    _registered_file,
)
from experiments.warehouse_checkpoint_resume import (
    checkpoint_failure_record,
    reusable_checkpoint_report,
)


CONFIG_SCHEMA = "lns2.stride.warehouse_disruption_recovery_heldout_config.v1"
EXPERIMENT_ID = "stride-warehouse-disruption-recovery-heldout-v1"
PLAN_SCHEMA = "lns2.stride.warehouse_disruption_recovery_heldout_plan.v1"
REPORT_SCHEMA = "lns2.stride.warehouse_disruption_recovery_heldout_checkpoint_report.v1"
PLAN_FILENAME = "heldout_checkpoint_plan.json"
MANIFEST_FILENAME = "heldout_checkpoint_manifest.jsonl"
REPORT_FILENAME = "heldout_checkpoint_qualification_report.json"
WORKER_RESULTS_FILENAME = "heldout_worker_results.json"


def load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = _read_json(path)
    if config.get("schema") != CONFIG_SCHEMA or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Warehouse disruption held-out identity changed")

    dataset = dict(config.get("dataset") or {})
    dataset_config = _registered_file(
        root,
        {"path": dataset.get("config_path"), "sha256": dataset.get("config_sha256")},
        label="Warehouse disruption dataset config",
    )
    runtime = dict(config.get("runtime") or {})
    runtime_template = _registered_file(
        root,
        {"path": runtime.get("template_path"), "sha256": runtime.get("template_sha256")},
        label="Warehouse disruption runtime template",
    )
    controller = dict(config.get("controller_contract") or {})
    v2_manifest = (
        _project_path(root, controller.get("v2_bundle"), label="v2 bundle")
        / "controller_manifest.json"
    )
    if (
        not v2_manifest.is_file()
        or sha256_file(v2_manifest) != str(controller.get("v2_manifest_sha256"))
    ):
        raise ValueError("frozen V2 controller manifest changed")
    development = dict(config.get("development_evidence") or {})
    development_report_path = _registered_file(
        root,
        {
            "path": development.get("report_path"),
            "sha256": development.get("report_sha256"),
        },
        label="Warehouse disruption development TTF report",
    )
    development_report = _read_json(development_report_path)
    integrity = dict(development_report.get("integrity_gates") or {})
    selected = list(development_report.get("screen_gate_passed_controllers") or [])
    dual16 = dict(
        dict(development_report.get("comparisons_vs_official") or {}).get("dual16")
        or {}
    )
    if (
        not integrity
        or not all(value is True for value in integrity.values())
        or development_report.get("complete_pairing_and_initial_state_identity") is not True
        or list(development_report.get("missing_or_failed_schedule_rows") or [])
        or int(development_report.get("completed_episode_count", -1)) != 28
        or int(development.get("required_completed_episode_count", -1)) != 28
        or selected != ["dual16"]
        or development.get("required_selected_controller") != "dual16"
        or dual16.get("screen_gate_passed") is not True
    ):
        raise ValueError("development evidence does not uniquely select dual16")
    generation = dict(config.get("checkpoint_generation") or {})
    gate = dict(generation.get("gate") or {})
    screen_gate = dict(config.get("screen_gate") or {})
    if (
        dataset.get("split") != "controller_held_out"
        or int(dataset.get("expected_map_count", -1)) != 2
        or int(dataset.get("expected_task_count", -1)) != 8
        or tuple(dataset.get("screen_task_variants") or ())
        != ("station_rush_d10", "balanced_od_d125")
        or int(generation.get("expected_candidate_count", -1)) != 4
        or int(generation.get("minimum_qualified_count", -1)) != 3
        or int(generation.get("minimum_qualified_map_count", -1)) != 2
        or int(generation.get("workers", -1)) != 4
        or float(generation.get("process_timeout_seconds", -1)) != 180.0
        or int(generation.get("incumbent_solver_seed_base", -1)) != 2026082900
        or float(generation.get("incumbent_time_limit_seconds", -1)) != 120.0
        or int(generation.get("incumbent_max_repair_iterations", -1)) != 0
        or generation.get("global_time_rule")
        != "earliest_maximum_hotspot_occupancy_after_t0"
        or int(generation.get("selection_seed_base", -1)) != 2026083000
        or int(generation.get("delay_ticks", -1)) != 4
        or float(generation.get("delayed_agent_fraction", -1)) != 0.15
        or int(generation.get("hotspot_neighborhood_radius", -1)) != 1
        or generation.get("selection_blind_to_controller_outcomes") is not True
        or generation.get("no_failed_candidate_replacement") is not True
        or tuple(config.get("controllers") or ()) != ("official_adaptive", "dual16")
        or controller.get("dual16") != "stride-structshell-dual16-v1"
        or controller.get("frozen_after_development") is not True
        or controller.get("no_new_ranker") is not True
        or gate != {
            "minimum_conflict_pair_count": 16,
            "minimum_active_conflict_agent_count": 32,
            "minimum_largest_conflict_component_size": 16,
        }
        or int(runtime.get("screen_solver_seed_base", -1)) != 101
        or float(runtime.get("wall_time_budget_seconds", -1)) != 120.0
        or float(runtime.get("environment_time_limit_seconds", -1)) != 120.0
        or float(runtime.get("episode_process_timeout_seconds", -1)) != 150.0
        or int(runtime.get("qualification_workers", -1)) != 4
        or int(runtime.get("timed_workers", -1)) != 1
        or runtime.get("execution_order") != "rotating_strict_two_controller_serial"
        or runtime.get("timing_boundary") != "checkpoint_restore_inclusive_ttf"
        or runtime.get("repair_seed_policy") != "episode_stream"
        or screen_gate.get("primary_baseline") != "official_adaptive"
        or screen_gate.get("target_controller") != "dual16"
        or float(screen_gate.get("minimum_paired_win_rate", -1)) != 0.70
        or float(screen_gate.get("minimum_mean_capped_ttf_improvement", -1)) != 0.15
        or float(screen_gate.get("minimum_successes_per_hour_improvement", -1)) != 0.20
        or float(screen_gate.get("maximum_success_rate_loss", -1)) != 0.0
        or screen_gate.get("no_additional_timeout_or_censor") is not True
        or screen_gate.get("targeted_expert_confirmation_only") is not True
        or screen_gate.get("global_or_default_promotion_allowed") is not False
    ):
        raise ValueError("Warehouse disruption held-out contract changed")

    config["_dataset_config"] = str(dataset_config)
    config["_runtime_template"] = str(runtime_template)
    config["_dataset_root"] = str(_project_path(root, dataset["output_path"], label="dataset output"))
    return path, root, config


def _heldout_schedule(config: Mapping[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    from experiments.repair_collection import _load_dataset_rows

    dataset = dict(config["dataset"])
    variants = {name: index for index, name in enumerate(dataset["screen_task_variants"])}
    selected = [
        dict(row)
        for row in _load_dataset_rows(dataset_root, [str(dataset["split"])])
        if str(row.get("task_variant")) in variants
    ]
    selected.sort(key=lambda row: (str(row["map_id"]), variants[str(row["task_variant"])]))
    generation = dict(config["checkpoint_generation"])
    runtime = dict(config["runtime"])
    if (
        len(selected) != 4
        or len({str(row["map_id"]) for row in selected}) != 2
        or collections.Counter(str(row["task_variant"]) for row in selected)
        != {"station_rush_d10": 2, "balanced_od_d125": 2}
    ):
        raise ValueError("Warehouse disruption held-out task schedule changed")
    return [
        {
            "key_index": index,
            "key_id": f"{row['task_id']}#heldout-delay-{index:02d}",
            "checkpoint_id": f"warehouse-disruption-heldout-{index:02d}",
            "map_id": str(row["map_id"]),
            "task_id": str(row["task_id"]),
            "task_variant": str(row["task_variant"]),
            "split": str(row["split"]),
            "agent_count": int(row["agent_count"]),
            "incumbent_solver_seed": int(generation["incumbent_solver_seed_base"]) + index,
            "selection_seed": int(generation["selection_seed_base"]) + index,
            "screen_solver_seed": int(runtime["screen_solver_seed_base"]) + index,
            "row": row,
        }
        for index, row in enumerate(selected)
    ]


def plan(config_path: str | Path) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    dataset_root = _ensure_dataset(root, config)
    schedule = _heldout_schedule(config, dataset_root)
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "dataset_root": str(dataset_root),
        "candidate_count": len(schedule),
        "map_count": len({row["map_id"] for row in schedule}),
        "task_variant_counts": dict(sorted(collections.Counter(row["task_variant"] for row in schedule).items())),
        "workers": int(config["checkpoint_generation"]["workers"]),
        "controller_outcomes_consulted": False,
        "failed_candidate_replacement": False,
        "schedule": [{key: value for key, value in row.items() if key != "row"} for row in schedule],
    }


_failure = partial(
    checkpoint_failure_record,
    incumbent_supply_error=INCUMBENT_SUPPLY_ERROR,
)


def _checkpoint_report(
    path: Path,
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    qualified = [row for row in rows if dict(row.get("qualification") or {}).get("passed")]
    supply = [row for row in failures if row.get("status") == "state_supply_unavailable" or (row.get("status") == "error" and row.get("error") == INCUMBENT_SUPPLY_ERROR)]
    execution = [row for row in failures if row not in supply]
    generation = dict(config["checkpoint_generation"])
    maps = {str(row["map_id"]) for row in qualified}
    attempted = len(rows) + len(failures)
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "candidate_count": int(generation["expected_candidate_count"]),
        "attempted_candidate_count": attempted,
        "completed_checkpoint_count": len(rows),
        "qualified_checkpoint_count": len(qualified),
        "qualified_map_count": len(maps),
        "qualified_checkpoint_ids": [str(row["checkpoint_id"]) for row in qualified],
        "state_supply_unavailable_count": len(supply),
        "state_supply_unavailable": supply,
        "execution_failure_count": len(execution),
        "execution_failures": execution,
        "gate": dict(generation["gate"]),
        "minimum_qualified_count": int(generation["minimum_qualified_count"]),
        "minimum_qualified_map_count": int(generation["minimum_qualified_map_count"]),
        "passed": attempted == 4 and len(qualified) >= 3 and len(maps) >= 2 and not execution,
        "controller_step_or_ttf_invoked": False,
        "checkpoint_selection_controller_outcomes_consulted": False,
        "failed_candidate_replacement": False,
        "targeted_expert_confirmation_only": True,
        "global_or_default_promotion_allowed": False,
        "claim_boundary": str(config["claim_boundary"]),
    }


def analyze_checkpoints(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    from experiments.repair_collection import _read_jsonl

    path, _root, config = load_config(config_path)
    checkpoint_root = Path(output).resolve() / "checkpoints"
    rows = _read_jsonl(checkpoint_root / MANIFEST_FILENAME) if (checkpoint_root / MANIFEST_FILENAME).is_file() else []
    worker_path = checkpoint_root / WORKER_RESULTS_FILENAME
    failures = list(_read_json(worker_path).get("failures", [])) if worker_path.is_file() else []
    report = _checkpoint_report(path, config, rows, failures)
    _write_json(checkpoint_root / REPORT_FILENAME, report)
    return report


def prepare_checkpoints(
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    dataset_root = _ensure_dataset(root, config)
    output_path = Path(output).resolve()
    checkpoint_root = output_path / "checkpoints"
    report_path = checkpoint_root / REPORT_FILENAME
    schedule = _heldout_schedule(config, dataset_root)
    payload = plan(path)
    if resume:
        report = reusable_checkpoint_report(
            report_path=report_path,
            plan_path=output_path / PLAN_FILENAME,
            manifest_path=checkpoint_root / MANIFEST_FILENAME,
            worker_path=checkpoint_root / WORKER_RESULTS_FILENAME,
            expected_schema=REPORT_SCHEMA,
            experiment_id=EXPERIMENT_ID,
            config_sha256=sha256_file(path),
            expected_plan=payload,
            schedule=payload["schedule"],
            build_report=lambda rows, failures: _checkpoint_report(
                path, config, rows, failures
            ),
        )
        if report is not None:
            return report
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / PLAN_FILENAME, payload)
    generation = dict(config["checkpoint_generation"])
    jobs = [
        {
            "job_id": str(item["key_id"]),
            "item": item,
            "dataset_root": str(dataset_root),
            "checkpoint_root": str(checkpoint_root),
            "generation": generation,
        }
        for item in schedule
    ]
    results = _run_jobs(
        _checkpoint_worker,
        jobs,
        int(workers if workers is not None else generation["workers"]),
        phase="warehouse-heldout-checkpoints",
        output_root=checkpoint_root,
        run_fingerprint=_fingerprint({"config_sha256": sha256_file(path), "schedule": payload["schedule"]}),
        timeout_seconds=float(generation["process_timeout_seconds"]),
        failure_result=_failure,
        stop_on_failure=False,
    )
    completed = sorted((dict(row["checkpoint"]) for row in results if row.get("status") == "ok"), key=lambda row: int(row["key_index"]))
    failures = [dict(row) for row in results if row.get("status") != "ok"]
    _write_jsonl(checkpoint_root / MANIFEST_FILENAME, completed)
    _write_json(checkpoint_root / WORKER_RESULTS_FILENAME, {"completed_count": len(completed), "failures": failures})
    return analyze_checkpoints(path, output_path)


__all__ = [
    "EXPERIMENT_ID",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_checkpoints",
    "load_config",
    "plan",
    "prepare_checkpoints",
]
