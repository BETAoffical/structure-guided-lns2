from __future__ import annotations

import collections
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.stride_warehouse_disruption_recovery import (
    INCUMBENT_SUPPLY_ERROR,
    _checkpoint_worker,
    _project_path,
    _registered_file,
)
from experiments.warehouse_disruption_checkpoints import (
    compute_checkpoint_identity_sha256,
)
from experiments.warehouse_checkpoint_resume import (
    checkpoint_failure_record,
    reusable_checkpoint_report,
)


CONFIG_SCHEMA = "lns2.stride.warehouse_disruption_recovery_boundary_config.v1"
EXPERIMENT_ID = "stride-warehouse-disruption-recovery-boundary-v1"
PLAN_SCHEMA = "lns2.stride.warehouse_disruption_recovery_boundary_plan.v1"
REPORT_SCHEMA = "lns2.stride.warehouse_disruption_recovery_boundary_checkpoint_report.v1"
PLAN_FILENAME = "boundary_checkpoint_plan.json"
MANIFEST_FILENAME = "boundary_checkpoint_manifest.jsonl"
REPORT_FILENAME = "boundary_checkpoint_qualification_report.json"
WORKER_RESULTS_FILENAME = "boundary_worker_results.json"
VARIANTS = ("balanced_od_d10", "balanced_od_d125")


def load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = _read_json(path)
    if config.get("schema") != CONFIG_SCHEMA or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Warehouse disruption boundary identity changed")
    dataset = dict(config.get("dataset") or {})
    dataset_config = _registered_file(root, {"path": dataset.get("config_path"), "sha256": dataset.get("config_sha256")}, label="Boundary dataset config")
    prior = dict(config.get("prior_dataset") or {})
    _registered_file(root, {"path": prior.get("config_path"), "sha256": prior.get("config_sha256")}, label="Prior warehouse dataset config")
    evidence = dict(config.get("heldout_evidence") or {})
    evidence_path = _registered_file(root, {"path": evidence.get("report_path"), "sha256": evidence.get("report_sha256")}, label="Held-out TTF report")
    heldout = _read_json(evidence_path)
    controller = dict(config.get("controller_contract") or {})
    v2_manifest = _project_path(root, controller.get("v2_bundle"), label="v2 bundle") / "controller_manifest.json"
    if not v2_manifest.is_file() or sha256_file(v2_manifest) != str(controller.get("v2_manifest_sha256")):
        raise ValueError("frozen V2 controller manifest changed")
    runtime = dict(config.get("runtime") or {})
    screen_gate = dict(config.get("screen_gate") or {})
    runtime_template = _registered_file(root, {"path": runtime.get("template_path"), "sha256": runtime.get("template_sha256")}, label="Boundary runtime template")
    generation = dict(config.get("checkpoint_generation") or {})
    if (
        dataset.get("split") != "boundary_confirmation"
        or int(dataset.get("expected_map_count", -1)) != 6
        or int(dataset.get("expected_task_count", -1)) != 12
        or tuple(dataset.get("task_variants") or ()) != VARIANTS
        or dict(dataset.get("load_bands") or {})
        != {"balanced_od_d10": "medium_high", "balanced_od_d125": "high"}
        or tuple(prior.get("splits") or ()) != ("development", "controller_held_out")
        or int(prior.get("expected_map_count", -1)) != 6
        or heldout.get("complete_pairing_and_initial_state_identity") is not True
        or heldout.get("targeted_expert_confirmation") is not True
        or heldout.get("global_claim_allowed") is not False
        or heldout.get("default_replacement_allowed") is not False
        or not dict(heldout.get("comparison_vs_official") or {}).get("screen_gate_passed")
        or int(heldout.get("completed_episode_count", -1)) != 8
        or int(generation.get("expected_candidate_count", -1)) != 24
        or int(generation.get("disturbance_replicas_per_task", -1)) != 2
        or int(generation.get("minimum_qualified_count", -1)) != 18
        or int(generation.get("maximum_qualified_count", -1)) != 24
        or int(generation.get("minimum_qualified_map_count", -1)) != 6
        or int(generation.get("minimum_qualified_per_load_band", -1)) != 8
        or int(generation.get("minimum_qualified_per_map_band_cell", -1)) != 1
        or int(generation.get("workers", -1)) != 8
        or float(generation.get("process_timeout_seconds", -1)) != 180.0
        or int(generation.get("incumbent_solver_seed_base", -1)) != 2026090100
        or generation.get("incumbent_seed_shared_across_disturbance_replicas") is not True
        or float(generation.get("incumbent_time_limit_seconds", -1)) != 120.0
        or int(generation.get("incumbent_max_repair_iterations", -1)) != 0
        or int(generation.get("selection_seed_base", -1)) != 2026090200
        or int(generation.get("delay_ticks", -1)) != 4
        or float(generation.get("delayed_agent_fraction", -1)) != 0.15
        or int(generation.get("hotspot_neighborhood_radius", -1)) != 1
        or generation.get("global_time_rule") != "earliest_maximum_hotspot_occupancy_after_t0"
        or dict(generation.get("gate") or {}) != {"minimum_conflict_pair_count": 16, "minimum_active_conflict_agent_count": 32, "minimum_largest_conflict_component_size": 16}
        or generation.get("selection_blind_to_controller_outcomes") is not True
        or generation.get("no_failed_candidate_replacement") is not True
        or tuple(config.get("controllers") or ()) != ("official_adaptive", "dual16")
        or controller.get("dual16") != "stride-structshell-dual16-v1"
        or controller.get("frozen_after_heldout") is not True
        or controller.get("no_new_ranker") is not True
        or int(runtime.get("screen_solver_seed_base", -1)) != 201
        or float(runtime.get("wall_time_budget_seconds", -1)) != 120.0
        or float(runtime.get("environment_time_limit_seconds", -1)) != 120.0
        or float(runtime.get("episode_process_timeout_seconds", -1)) != 150.0
        or int(runtime.get("qualification_workers", -1)) != 8
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
        or screen_gate.get("bandwise_independent") is not True
        or screen_gate.get("route_policy")
        != "independent_load_band_gates"
        or tuple(screen_gate.get("allowed_route_recommendations") or ())
        != ("high_only", "both", "none")
        or screen_gate.get("targeted_expert_confirmation_only") is not True
        or screen_gate.get("global_or_default_promotion_allowed") is not False
    ):
        raise ValueError("Warehouse disruption boundary contract changed")
    config["_dataset_config"] = str(dataset_config)
    config["_dataset_root"] = str(_project_path(root, dataset["output_path"], label="boundary dataset output"))
    config["_prior_dataset_root"] = str(_project_path(root, prior["output_path"], label="prior dataset output"))
    config["_runtime_template"] = str(runtime_template)
    return path, root, config


def _ensure_datasets(config: Mapping[str, Any]) -> Path:
    boundary_root = Path(str(config["_dataset_root"]))
    prior_root = Path(str(config["_prior_dataset_root"]))
    boundary_rows = _load_dataset_rows(boundary_root, ["boundary_confirmation"])
    prior_rows = _load_dataset_rows(prior_root, ["development", "controller_held_out"])
    if len(boundary_rows) != 12 or len({str(row["task_id"]) for row in boundary_rows}) != 12:
        raise ValueError("Boundary dataset must contain twelve unique tasks")
    boundary_maps = {(str(row["map_id"]), str(row["map_file"])) for row in boundary_rows}
    prior_maps = {(str(row["split"]), str(row["map_id"]), str(row["map_file"])) for row in prior_rows}
    boundary_hashes = {sha256_file(boundary_root / "boundary_confirmation" / file) for _map, file in boundary_maps}
    prior_hashes = {sha256_file(prior_root / split / file) for split, _map, file in prior_maps}
    if len(boundary_maps) != 6 or len(boundary_hashes) != 6:
        raise ValueError("Boundary maps are not six byte-distinct layouts")
    if len(prior_maps) != 6 or ({map_id for map_id, _file in boundary_maps} & {map_id for _split, map_id, _file in prior_maps}) or boundary_hashes & prior_hashes:
        raise ValueError("Boundary maps overlap prior development or held-out maps")
    variants = collections.Counter(str(row.get("task_variant")) for row in boundary_rows)
    if variants != {"balanced_od_d10": 6, "balanced_od_d125": 6}:
        raise ValueError("Boundary task product changed")
    return boundary_root


def _boundary_schedule(config: Mapping[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in _load_dataset_rows(dataset_root, ["boundary_confirmation"])]
    variant_order = {name: index for index, name in enumerate(VARIANTS)}
    rows.sort(key=lambda row: (str(row["map_id"]), variant_order[str(row["task_variant"])]))
    generation, runtime = dict(config["checkpoint_generation"]), dict(config["runtime"])
    result: list[dict[str, Any]] = []
    for task_index, row in enumerate(rows):
        for replica in range(2):
            key_index = len(result)
            result.append({
                "key_index": key_index,
                "task_index": task_index,
                "disturbance_replica": replica,
                "key_id": f"{row['task_id']}#boundary-disturbance-{replica}",
                "checkpoint_id": f"boundary-{task_index:02d}-disturbance-{replica}",
                "map_id": str(row["map_id"]),
                "task_id": str(row["task_id"]),
                "task_variant": str(row["task_variant"]),
                "load_band": str(config["dataset"]["load_bands"][str(row["task_variant"])]),
                "split": str(row["split"]),
                "agent_count": int(row["agent_count"]),
                "incumbent_solver_seed": int(generation["incumbent_solver_seed_base"]) + task_index,
                "selection_seed": int(generation["selection_seed_base"]) + key_index,
                "screen_solver_seed": int(runtime["screen_solver_seed_base"]) + key_index,
                "row": row,
            })
    if len(result) != 24:
        raise ValueError("Boundary schedule must contain 24 candidates")
    return result


def plan(config_path: str | Path) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    dataset_root = _ensure_datasets(config)
    schedule = _boundary_schedule(config, dataset_root)
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "candidate_count": 24,
        "map_count": 6,
        "task_count": 12,
        "load_band_counts": dict(sorted(collections.Counter(row["load_band"] for row in schedule).items())),
        "workers": 8,
        "controller_outcomes_consulted": False,
        "failed_candidate_replacement": False,
        "schedule": [{key: value for key, value in row.items() if key != "row"} for row in schedule],
    }


_failure = partial(
    checkpoint_failure_record,
    incumbent_supply_error=INCUMBENT_SUPPLY_ERROR,
)


def _boundary_checkpoint_worker(job: dict[str, Any]) -> dict[str, Any]:
    result = _checkpoint_worker(job)
    if result.get("status") != "ok":
        return result
    checkpoint = dict(result["checkpoint"])
    item = dict(job["item"])
    checkpoint["disturbance_replica"] = int(item["disturbance_replica"])
    checkpoint["load_band"] = str(item["load_band"])
    checkpoint["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(
        checkpoint
    )
    return {**result, "checkpoint": checkpoint}


def _checkpoint_report(
    path: Path,
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    qualified = [row for row in rows if dict(row.get("qualification") or {}).get("passed")]
    supply = [row for row in failures if row.get("status") == "state_supply_unavailable" or (row.get("status") == "error" and row.get("error") == INCUMBENT_SUPPLY_ERROR)]
    execution = [row for row in failures if row not in supply]
    maps = {str(row["map_id"]) for row in qualified}
    bands = collections.Counter(str(row["task_variant"]) for row in qualified)
    cells = collections.Counter((str(row["map_id"]), str(row["task_variant"])) for row in qualified)
    attempted = len(rows) + len(failures)
    passed = bool(attempted == 24 and 18 <= len(qualified) <= 24 and len(maps) == 6 and all(bands[name] >= 8 for name in VARIANTS) and len(cells) == 12 and all(value >= 1 for value in cells.values()) and not execution)
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "candidate_count": 24,
        "attempted_candidate_count": attempted,
        "completed_checkpoint_count": len(rows),
        "qualified_checkpoint_count": len(qualified),
        "qualified_map_count": len(maps),
        "qualified_per_load_band": {config["dataset"]["load_bands"][name]: bands[name] for name in VARIANTS},
        "qualified_map_band_cell_count": len(cells),
        "qualified_checkpoint_ids": [str(row["checkpoint_id"]) for row in qualified],
        "state_supply_unavailable_count": len(supply),
        "state_supply_unavailable": supply,
        "execution_failure_count": len(execution),
        "execution_failures": execution,
        "gate": dict(config["checkpoint_generation"]["gate"]),
        "passed": passed,
        "controller_step_or_ttf_invoked": False,
        "checkpoint_selection_controller_outcomes_consulted": False,
        "failed_candidate_replacement": False,
        "claim_boundary": str(config["claim_boundary"]),
    }


def analyze_checkpoints(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    checkpoint_root = Path(output).resolve() / "checkpoints"
    rows = _read_jsonl(checkpoint_root / MANIFEST_FILENAME) if (checkpoint_root / MANIFEST_FILENAME).is_file() else []
    worker_path = checkpoint_root / WORKER_RESULTS_FILENAME
    failures = list(_read_json(worker_path).get("failures", [])) if worker_path.is_file() else []
    report = _checkpoint_report(path, config, rows, failures)
    _write_json(checkpoint_root / REPORT_FILENAME, report)
    return report


def prepare_checkpoints(config_path: str | Path, output: str | Path, *, workers: int | None = None, resume: bool = False) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    dataset_root = _ensure_datasets(config)
    output_path, checkpoint_root = Path(output).resolve(), Path(output).resolve() / "checkpoints"
    report_path = checkpoint_root / REPORT_FILENAME
    schedule, payload = _boundary_schedule(config, dataset_root), plan(path)
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
    jobs = [{"job_id": item["key_id"], "item": item, "dataset_root": str(dataset_root), "checkpoint_root": str(checkpoint_root), "generation": generation} for item in schedule]
    results = _run_jobs(_boundary_checkpoint_worker, jobs, int(workers if workers is not None else generation["workers"]), phase="warehouse-boundary-checkpoints", output_root=checkpoint_root, run_fingerprint=_fingerprint({"config_sha256": sha256_file(path), "schedule": payload["schedule"]}), timeout_seconds=float(generation["process_timeout_seconds"]), failure_result=_failure, stop_on_failure=False)
    completed = sorted((dict(row["checkpoint"]) for row in results if row.get("status") == "ok"), key=lambda row: int(row["key_index"]))
    failures = [dict(row) for row in results if row.get("status") != "ok"]
    _write_jsonl(checkpoint_root / MANIFEST_FILENAME, completed)
    _write_json(checkpoint_root / WORKER_RESULTS_FILENAME, {"completed_count": len(completed), "failures": failures})
    return analyze_checkpoints(path, output_path)


__all__ = ["EXPERIMENT_ID", "PLAN_SCHEMA", "REPORT_SCHEMA", "analyze_checkpoints", "load_config", "plan", "prepare_checkpoints"]
