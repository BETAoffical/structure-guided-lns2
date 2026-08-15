from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import closed_loop_producer_identity, mean, sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.stride_bounded_native_retry_continuation import _failed_job
from experiments.stride_onpolicy_controller_attribution import (
    _arm_summary,
    _collection_path,
    _decision_rows,
    _episode_override,
    _episode_summary,
    _parent_from_exact,
    _platform_diagnostics,
    _qualification_roots,
    load_registration as load_source_registration,
)
from lns2_selector.runtime.hybridstructpool import (
    hybridstructpool_runtime_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.hybridstructpool_runtime_pilot_registration.v1"
STATUS_SCHEMA = "lns2.stride.hybridstructpool_runtime_pilot_status.v1"
REPORT_SCHEMA = "lns2.stride.hybridstructpool_runtime_pilot_report.v1"
EXPERIMENT_ID = "stride-hybridstructpool-runtime-pilot-v1"
V2_ARM = "v2_full"
HYBRID_ARM = "hybridstructpool_full"
ARMS = (V2_ARM, HYBRID_ARM)


def load_registration(path: str | Path) -> tuple[Any, ...]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    source = dict(config.get("source_registration") or {})
    source_path = (root / str(source.get("path"))).resolve()
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_pool_only_initial_path_runtime_pilot"
        or config.get("experiment_id") != EXPERIMENT_ID
        or tuple(map(str, config.get("arms") or ())) != ARMS
        or sha256_file(source_path) != str(source.get("sha256"))
    ):
        raise ValueError("HybridStructPool runtime pilot registration changed")
    engineering = dict(config.get("engineering_gate") or {})
    if engineering:
        baseline = dict(engineering.get("registered_full_runtime_baseline") or {})
        baseline_path = (root / str(baseline.get("report_path"))).resolve()
        if (
            not baseline_path.is_file()
            or sha256_file(baseline_path) != str(baseline.get("report_sha256"))
        ):
            raise ValueError("HybridStructPool full-runtime baseline changed")
    execution = dict(config["execution"])
    if execution != {
        "trial_indices": [0, 1, 2, 3],
        "worker_count": 16,
        "maximum_repair_decisions": 64,
        "wall_time_fuse_seconds": 180.0,
        "process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "native_pp_order_only": True,
        "failure_rescue_enabled": False,
        "stop_on_first_execution_error_or_process_timeout": True,
    }:
        raise ValueError("HybridStructPool runtime pilot execution changed")
    source_loaded = load_source_registration(source_path)
    tasks = [dict(row) for row in source_loaded[-1]]
    map_counts = collections.Counter(str(row["map_id"]) for row in tasks)
    if (
        len(tasks) != int(config["cohort"]["unique_task_solver_key_count"])
        or dict(sorted(map_counts.items()))
        != dict(sorted(dict(config["cohort"]["map_counts"]).items()))
    ):
        raise ValueError("HybridStructPool runtime pilot cohort changed")
    return path, root, config, source_loaded, tasks


def schedule(tasks: Iterable[Mapping[str, Any]], trials: Iterable[int]) -> list[dict]:
    rows: list[dict] = []
    for task_position, task in enumerate(tasks):
        for trial_index in map(int, trials):
            for arm_position, arm in enumerate(ARMS):
                item = {
                    **dict(task),
                    "trial_index": trial_index,
                    "arm": arm,
                    "task_position": task_position,
                    "arm_position": arm_position,
                }
                item["job_id"] = _fingerprint(
                    {
                        "task_fingerprint": item["task_fingerprint"],
                        "trial_index": trial_index,
                        "arm": arm,
                    }
                )
                rows.append(item)
    return rows


def _manifest(output: Path, item: Mapping[str, Any]) -> dict | None:
    path = _collection_path(output, item) / "realized_dynamic_manifest.jsonl"
    rows = _read_jsonl(path) if path.is_file() else []
    matches = [
        dict(row)
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("HybridStructPool pilot manifest is ambiguous")
    return matches[0] if matches else None


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    path, root, _config, source_loaded, tasks = load_registration(
        job["config_path"]
    )
    _sp, _sr, _sc, inputs, exact_loaded, _source_tasks = source_loaded
    parent = _parent_from_exact(exact_loaded)
    item = dict(job["item"])
    task = {str(row["task_fingerprint"]): row for row in tasks}[
        str(item["task_fingerprint"])
    ]
    key = (str(item["task_id"]), int(item["solver_seed"]))
    all_keys = {
        (str(row["task_id"]), int(row["solver_seed"])) for row in tasks
    }
    override = _episode_override(
        task,
        trial_index=int(item["trial_index"]),
        source_cases=list(exact_loaded[-1]),
    )
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    collection = Path(job["collection_path"]).resolve()
    kwargs = {
        "controller": "v2-full",
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "stopping_rule": "historical",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
    }
    if str(item["arm"]) == HYBRID_ARM:
        kwargs["hybridstructpool_augmentation"] = (
            hybridstructpool_runtime_augmentation()
        )
    qualification_source = _qualification_roots(inputs)["slot"]
    run_closed_loop_collection(
        dataset,
        inputs["runtime_config"],
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        cohort_job_keys=all_keys,
        job_keys=all_keys,
        episode_overrides={key: override},
        qualification_source=qualification_source,
        use_global_collection_lock=False,
        **kwargs,
    )
    run_closed_loop_collection(
        dataset,
        inputs["runtime_config"],
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        cohort_job_keys=all_keys,
        job_keys={key},
        episode_overrides={key: override},
        qualification_source=qualification_source,
        use_global_collection_lock=False,
        **kwargs,
    )
    manifest = _manifest(Path(job["output_root"]), item)
    if manifest is None:
        raise RuntimeError("HybridStructPool pilot completed without manifest")
    status = str(manifest.get("status"))
    return {
        **item,
        "status": status if status in {"error", "timeout"} else "ok",
        "manifest_status": status,
        "error": manifest.get("error"),
    }


def _status(output: Path, items: list[dict], run_fp: str) -> dict:
    manifests = [_manifest(output, item) for item in items]
    complete = [row for row in manifests if row is not None]
    return {
        "schema": STATUS_SCHEMA,
        "run_fingerprint": run_fp,
        "completed_jobs": len(complete),
        "total_jobs": len(items),
        "arm_completed_counts": {
            arm: sum(
                manifest is not None
                for item, manifest in zip(items, manifests)
                if str(item["arm"]) == arm
            )
            for arm in ARMS
        },
        "error_jobs": sum(str(row.get("status")) == "error" for row in complete),
        "timeout_jobs": sum(
            str(row.get("status")) == "timeout" for row in complete
        ),
        "status": "complete" if len(complete) == len(items) else "running",
    }


def _pilot_arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    result = _arm_summary(rows, arm)
    selected = [row for row in rows if str(row["arm"]) == arm]
    result.update(
        {
            "mean_controller_seconds": mean(
                float(row["controller_seconds"]) for row in selected
            ),
            "mean_candidate_generation_seconds": mean(
                float(row["candidate_generation_seconds"]) for row in selected
            ),
            "mean_feature_seconds": mean(
                float(row["feature_seconds"]) for row in selected
            ),
            "mean_inference_seconds": mean(
                float(row["inference_seconds"]) for row in selected
            ),
            "mean_hybrid_total_seconds": mean(
                float(row["hybrid_total_seconds"]) for row in selected
            ),
            "mean_hybrid_generation_seconds": mean(
                float(row["hybrid_generation_seconds"]) for row in selected
            ),
        }
    )
    return result


def collect(
    config_path: str | Path, output: str | Path, *, resume: bool = False
) -> dict:
    path, root, config, _source_loaded, tasks = load_registration(config_path)
    output = Path(output).resolve()
    items = schedule(tasks, config["execution"]["trial_indices"])
    producer = closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_hybridstructpool_runtime_pilot.py",
            "scripts/run_stride_hybridstructpool_runtime_pilot.py",
            "experiments/closed_loop_confirmation.py",
            "lns2_selector/runtime/causalclosurepool.py",
            "lns2_selector/runtime/hybridstructpool.py",
            "lns2_selector/runtime/online_selection.py",
            "lns2_selector/runtime/topology_candidates.py",
        ),
    )
    run_fp = _fingerprint(
        {
            "registration_sha256": sha256_file(path),
            "schedule_sha256": _fingerprint(items),
            "producer": producer,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "initial_schedule.jsonl"
    if schedule_path.is_file():
        if _read_jsonl(schedule_path) != items:
            raise ValueError("HybridStructPool pilot schedule changed")
        if not resume:
            raise ValueError("pilot output exists; pass --resume")
    else:
        _write_jsonl(schedule_path, items)
    run_config = {
        "schema": f"{STATUS_SCHEMA}.run_config",
        "registration": str(path),
        "registration_sha256": sha256_file(path),
        "producer": producer,
        "schedule_sha256": _fingerprint(items),
        "run_fingerprint": run_fp,
    }
    run_path = output / "run_config.json"
    if run_path.is_file() and _read_json(run_path) != run_config:
        raise ValueError("HybridStructPool pilot run identity changed")
    _write_json(run_path, run_config)
    existing_failures = [
        row
        for row in (_manifest(output, item) for item in items)
        if row is not None and str(row.get("status")) in {"error", "timeout"}
    ]
    if existing_failures:
        raise RuntimeError("HybridStructPool pilot contains a terminal failure")
    pending = [item for item in items if _manifest(output, item) is None]
    jobs = [
        {
            "job_id": str(item["job_id"]),
            "config_path": str(path),
            "output_root": str(output),
            "collection_path": str(_collection_path(output, item)),
            "item": item,
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            int(config["execution"]["worker_count"]),
            phase="hybridstructpool-runtime-pilot-initial",
            output_root=output,
            run_fingerprint=run_fp,
            timeout_seconds=float(config["execution"]["outer_job_timeout_seconds"]),
            failure_result=_failed_job,
            stop_on_failure=True,
        )
        terminal = [row for row in results if row["status"] in {"error", "timeout"}]
        if terminal:
            status = _status(output, items, run_fp)
            status["terminal_failure"] = terminal[0]
            _write_json(output / "collection_status.json", status)
            return status
    status = _status(output, items, run_fp)
    _write_json(output / "collection_status.json", status)
    return status


def analyze(config_path: str | Path, output: str | Path) -> dict:
    _path, _root, config, _source_loaded, tasks = load_registration(config_path)
    output = Path(output).resolve()
    items = schedule(tasks, config["execution"]["trial_indices"])
    rows: list[dict] = []
    for item in items:
        manifest = _manifest(output, item)
        if manifest is None:
            rows.append({**item, "missing": True})
            continue
        row = {**item, "manifest_status": str(manifest.get("status"))}
        row.update(_episode_summary(manifest))
        summary = dict(manifest.get("summary") or {})
        controller_totals = dict(summary.get("controller_totals") or {})
        row.update(
            {
                "initial_fingerprint": str(summary.get("initial_fingerprint")),
                "repair_wall_seconds": float(summary.get("repair_wall_seconds", 0.0)),
                "capped_ttf_seconds": float(
                    summary.get("capped_wall_time_to_feasible", 180.0)
                ),
                "controller_seconds": float(
                    controller_totals.get("controller_seconds_before_repair", 0.0)
                ),
                "candidate_generation_seconds": float(
                    controller_totals.get("candidate_generation_seconds", 0.0)
                ),
                "feature_seconds": float(
                    controller_totals.get("feature_seconds", 0.0)
                ),
                "inference_seconds": float(
                    controller_totals.get("inference_seconds", 0.0)
                ),
            }
        )
        if manifest.get("status") == "ok":
            decisions = _decision_rows(_collection_path(output, item), manifest)
            row.update(_platform_diagnostics(decisions))
            row["hybrid_total_seconds"] = sum(
                float(
                    dict(
                        dict(decision.get("controller") or {}).get("proposal") or {}
                    ).get("hybridstructpool_seconds", 0.0)
                )
                for decision in decisions
            )
            row["hybrid_generation_seconds"] = sum(
                float(
                    dict(
                        dict(decision.get("controller") or {}).get("proposal") or {}
                    ).get("hybridstructpool_generation_seconds", 0.0)
                )
                for decision in decisions
            )
            row["unique_neighborhood_count"] = len(
                {
                    tuple(
                        sorted(
                            map(
                                int,
                                decision["actual_metrics"].get("neighborhood", []),
                            )
                        )
                    )
                    for decision in decisions
                }
            )
            row["paired_pp_seed_present"] = any(
                "pp_random_seed" in decision["actual_action"]
                for decision in decisions
            )
        rows.append(row)
    complete = [row for row in rows if not row.get("missing")]
    summaries = {arm: _pilot_arm_summary(complete, arm) for arm in ARMS}
    by_map = {
        map_id: {
            arm: _pilot_arm_summary(
                [row for row in complete if str(row["map_id"]) == map_id], arm
            )
            for arm in ARMS
        }
        for map_id in sorted({str(row["map_id"]) for row in complete})
    }
    gate = {
        "complete": len(complete) == 152,
        "zero_terminal_errors": all(
            str(row.get("manifest_status")) not in {"error", "timeout"}
            for row in complete
        ),
        "no_paired_pp_seed": all(
            not bool(row.get("paired_pp_seed_present")) for row in complete
        ),
        "success_preserved": summaries[HYBRID_ARM]["success_rate"]
        >= summaries[V2_ARM]["success_rate"],
        "auc_improved": summaries[HYBRID_ARM]["mean_normalized_fixed_auc"]
        <= summaries[V2_ARM]["mean_normalized_fixed_auc"],
        "repair_decisions_improved": summaries[HYBRID_ARM][
            "restricted_mean_repair_decisions"
        ]
        <= summaries[V2_ARM]["restricted_mean_repair_decisions"],
        "capped_wall_improved": summaries[HYBRID_ARM]["mean_capped_ttf_seconds"]
        <= summaries[V2_ARM]["mean_capped_ttf_seconds"],
        "per_map_platform_preserved": all(
            values[HYBRID_ARM]["platform_entry_rate"]
            <= values[V2_ARM]["platform_entry_rate"] + 0.05
            for values in by_map.values()
        ),
    }
    engineering = dict(config.get("engineering_gate") or {})
    if engineering:
        baseline = dict(engineering["registered_full_runtime_baseline"])
        hybrid = summaries[HYBRID_ARM]
        gate.update(
            {
                "full_runtime_success_preserved": hybrid["success_rate"]
                >= float(baseline["success_rate"]),
                "full_runtime_auc_preserved": hybrid["mean_normalized_fixed_auc"]
                <= float(baseline["mean_normalized_fixed_auc"]),
                "full_runtime_repair_decisions_preserved": hybrid[
                    "restricted_mean_repair_decisions"
                ]
                <= float(baseline["restricted_mean_repair_decisions"]),
                "full_runtime_platform_preserved": hybrid["platform_entry_rate"]
                <= float(baseline["platform_entry_rate"])
                + float(engineering["maximum_platform_rate_worsening"]),
                "controller_seconds_reduced": hybrid["mean_controller_seconds"]
                <= float(engineering["maximum_mean_controller_seconds"]),
                "candidate_generation_seconds_reduced": hybrid[
                    "mean_candidate_generation_seconds"
                ]
                <= float(
                    engineering["maximum_mean_candidate_generation_seconds"]
                ),
                "per_map_capped_wall_preserved": all(
                    values[HYBRID_ARM]["mean_capped_ttf_seconds"]
                    <= values[V2_ARM]["mean_capped_ttf_seconds"]
                    * float(engineering["maximum_per_map_capped_wall_ratio"])
                    for values in by_map.values()
                ),
            }
        )
        if "maximum_mean_hybrid_total_seconds" in engineering:
            gate["hybrid_total_seconds_reduced"] = hybrid[
                "mean_hybrid_total_seconds"
            ] <= float(engineering["maximum_mean_hybrid_total_seconds"])
    report = {
        "schema": REPORT_SCHEMA,
        "integrity_passed": gate["complete"]
        and gate["zero_terminal_errors"]
        and gate["no_paired_pp_seed"],
        "all_gates_passed": all(gate.values()),
        "gates": gate,
        "arm_summaries": summaries,
        "by_map": by_map,
        "episode_count": len(complete),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    _write_json(output / "initial_report.json", report)
    return report


__all__ = ["analyze", "collect", "load_registration", "schedule"]
