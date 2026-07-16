from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.closed_loop_confirmation import (
    _closed_loop_episode_worker,
    _sha256,
    controller_implementation_fingerprint,
    validate_closed_loop_trace,
)
from experiments.closed_loop_confirmation_analysis import (
    compare_policies,
    summarize_policy,
)
from experiments.policy_visited_aggregation import (
    policy_visited_dataset_design,
    policy_visited_qualification_report,
)
from experiments.policy_visited_aggregation_analysis import (
    _fingerprint_integrity,
    closed_loop_v2_acceptance,
)
from experiments.realized_ranking_confirmation import _seed_isolation
from experiments.repair_collection import (
    SCHEMA_VERSION,
    _CollectionRunLock,
    _dataset_fingerprint,
    _fingerprint,
    _load_dataset_rows,
    _qualification_worker,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)


SCHEMA = "lns2.policy_visited_independent_confirmation.v1"
POLICY_SPECS = {
    "official_adaptive": "official_adaptive",
    "frozen_v1": "realized_dynamic",
    "aggregated_v2": "realized_dynamic",
}


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported independent confirmation config")
    if str(config.get("study_role")) != "independent_confirmation":
        raise ValueError("confirmation config must declare independent_confirmation")
    if tuple(map(str, config.get("splits", []))) != ("policy_confirmation",):
        raise ValueError("confirmation split must be policy_confirmation")
    if tuple(map(int, config.get("solver_seeds", []))) != (1, 2, 3):
        raise ValueError("confirmation solver seeds must be [1, 2, 3]")
    if int(config.get("dataset_master_seed", -1)) != 20270421:
        raise ValueError("confirmation dataset master seed must be 20270421")
    qualification = dict(config.get("qualification", {}))
    if str(qualification.get("mode")) != "natural_distribution_confirmation":
        raise ValueError("confirmation requires natural distribution qualification")
    if int(config.get("bootstrap_samples", 0)) != 5000:
        raise ValueError("confirmation requires 5,000 map bootstrap samples")
    if int(config.get("workers", 0)) <= 0:
        raise ValueError("confirmation workers must be positive")


def _dataset_master_seed(dataset_root: Path) -> int:
    summary = _read_json(dataset_root / "dataset_summary.json")
    return int(summary["master_seed"])


def _write_incremental_manifest(
    path: Path, rows: dict[str, dict[str, Any]]
) -> None:
    _write_jsonl(path, [rows[key] for key in sorted(rows)])


def _policy_model_spec(
    label: str,
    project_root: Path,
    collection_config: dict[str, Any],
    training_root: Path,
    v2_registration: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if label == "aggregated_v2":
        return str(training_root), v2_registration
    frozen_root = Path(str(collection_config["frozen_models"]))
    if not frozen_root.is_absolute():
        frozen_root = project_root / frozen_root
    return str(frozen_root.resolve()), dict(collection_config["model_registration"])


def _collect_policy(
    label: str,
    rows: list[dict[str, Any]],
    solver_seeds: tuple[int, ...],
    dataset_root: Path,
    output_root: Path,
    run_fingerprint: str,
    config: dict[str, Any],
    frozen_models: str,
    registration: dict[str, Any],
    workers: int,
    resume: bool,
) -> list[dict[str, Any]]:
    policy_root = output_root / "policies" / label
    manifest_path = policy_root / "manifest.jsonl"
    existing = _read_jsonl(manifest_path) if resume and manifest_path.is_file() else []
    manifest = {str(row["episode_id"]): row for row in existing}
    jobs = [
        {
            "row": row,
            "policy": POLICY_SPECS[label],
            "solver_seed": seed,
            "dataset_root": str(dataset_root),
            "environment": config["environment"],
            "proposal": config["proposal"],
            "max_decisions": int(config["max_decisions"]),
            "metric_iteration_budget": int(config["metric_iteration_budget"]),
            "wall_time_budget_seconds": float(config["wall_time_budget_seconds"]),
            "frozen_models": frozen_models,
            "model_registration": registration,
            "output_root": str(policy_root),
            "run_fingerprint": run_fingerprint,
            "resume": resume,
        }
        for row in rows
        for seed in solver_seeds
    ]

    def record(result: dict[str, Any]) -> None:
        value = {**result, "policy_label": label}
        manifest[str(value["episode_id"])] = value
        _write_incremental_manifest(manifest_path, manifest)

    with _CollectionRunLock(output_root, run_fingerprint, f"confirmation-{label}"):
        results = _run_jobs(
            _closed_loop_episode_worker,
            jobs,
            workers,
            phase=f"policy-visited-confirmation-{label}",
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(config["episode_process_timeout_seconds"]),
            on_result=record,
        )
    normalized = [{**row, "policy_label": label} for row in results]
    _write_jsonl(
        manifest_path, sorted(normalized, key=lambda row: str(row["episode_id"]))
    )
    return normalized


def _validate_policy_traces(
    output_root: Path,
    label: str,
    rows: list[dict[str, Any]],
    run_fingerprint: str,
    budget: int,
) -> None:
    policy_root = output_root / "policies" / label
    for row in rows:
        if str(row.get("status")) not in {"ok", "resumed"}:
            continue
        validate_closed_loop_trace(
            policy_root / str(row["trace_file"]),
            run_fingerprint,
            expected_episode_id=str(row["episode_id"]),
            expected_policy=POLICY_SPECS[label],
            expected_solver_seed=int(row["solver_seed"]),
            metric_iteration_budget=budget,
        )


def run_independent_confirmation(
    dataset: str | Path,
    development_collection: str | Path,
    training: str | Path,
    offline: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if phase not in {"qualify", "collect", "analyze", "all"}:
        raise ValueError("phase must be qualify, collect, analyze, or all")
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(dataset).resolve()
    collection_root = Path(development_collection).resolve()
    training_root = Path(training).resolve()
    offline_root = Path(offline).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_config(config)
    if _dataset_master_seed(dataset_root) != int(config["dataset_master_seed"]):
        raise ValueError("independent confirmation master seed mismatch")
    offline_report = _read_json(offline_root / "offline_report.json")
    if not bool(offline_report.get("offline_acceptance", {}).get("passed")):
        raise ValueError("development offline gate did not pass; confirmation is forbidden")
    collection_run = _read_json(collection_root / "run_config.json")
    if str(collection_run.get("configuration", {}).get("study_role")) != "development":
        raise ValueError("source collection is not a natural-distribution development run")
    training_report = _read_json(training_root / "training_report.json")
    if str(training_report["source_run_fingerprint"]) != str(
        collection_run["run_fingerprint"]
    ):
        raise ValueError("training output belongs to another development collection")
    if bool(training_report.get("validation_labels_used_for_training")):
        raise ValueError("training manifest reports validation leakage")
    rows = _load_dataset_rows(dataset_root, ["policy_confirmation"])
    design = policy_visited_dataset_design(rows, config)
    isolation = _seed_isolation(rows, list(config["reference_datasets"]), project_root)
    dataset_fp = _dataset_fingerprint(dataset_root)
    v2_registration = _read_json(training_root / "model_registration.json")
    implementation = {
        "controller": controller_implementation_fingerprint(project_root),
        "confirmation_module_sha256": _sha256(Path(__file__)),
    }
    run_fp = _fingerprint(
        {
            "dataset_fingerprint": dataset_fp,
            "configuration": config,
            "development_run": collection_run["run_fingerprint"],
            "training_report": _sha256(training_root / "training_report.json"),
            "offline_report": _sha256(offline_root / "offline_report.json"),
            "v2_registration": v2_registration,
            "implementation": implementation,
        }
    )
    estimate = {
        "map_count": 6,
        "task_count": len(rows),
        "solver_seed_count": len(config["solver_seeds"]),
        "qualification_reset_count": len(rows) * len(config["solver_seeds"]),
        "policy_episode_count": len(rows) * len(config["solver_seeds"]) * 3,
    }
    if dry_run:
        return {
            "schema": SCHEMA,
            "dry_run": True,
            "run_fingerprint": run_fp,
            "dataset_design": design,
            "seed_isolation": isolation,
            "estimate": estimate,
        }
    run_config = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_role": "independent_confirmation",
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_fp,
        "development_collection": str(collection_root),
        "development_run_fingerprint": str(collection_run["run_fingerprint"]),
        "training": str(training_root),
        "training_report_sha256": _sha256(training_root / "training_report.json"),
        "offline_report_sha256": _sha256(offline_root / "offline_report.json"),
        "configuration": config,
        "run_fingerprint": run_fp,
        "dataset_design": design,
        "seed_isolation": isolation,
        "implementation": implementation,
    }
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fp:
            raise ValueError("confirmation output contains a different run")
        if not resume:
            raise ValueError("confirmation output exists; pass resume")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, run_config)
    sequence = ("qualify", "collect", "analyze") if phase == "all" else (phase,)
    summary: dict[str, Any] = {"schema": SCHEMA, "run_fingerprint": run_fp}
    effective_workers = int(workers or config["workers"])
    seeds = tuple(map(int, config["solver_seeds"]))
    for current in sequence:
        if current == "qualify":
            jobs = [
                {
                    "row": row,
                    "solver_seed": seed,
                    "dataset_root": str(dataset_root),
                    "environment": config["environment"],
                }
                for row in rows
                for seed in seeds
            ]
            with _CollectionRunLock(output_root, run_fp, "confirmation-qualification"):
                results = _run_jobs(
                    _qualification_worker,
                    jobs,
                    effective_workers,
                    phase="policy-visited-independent-qualification",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["episode_process_timeout_seconds"]),
                )
            _write_jsonl(
                output_root / "qualification_manifest.jsonl",
                sorted(results, key=lambda row: (str(row["task_id"]), int(row["solver_seed"]))),
            )
            report = policy_visited_qualification_report(
                rows, results, config, design, isolation, formal=True
            )
            _write_json(output_root / "qualification_report.json", report)
            summary["qualification"] = report
            if not report["passed"] and phase == "all":
                break
        elif current == "collect":
            qualification = _read_json(output_root / "qualification_report.json")
            if not bool(qualification.get("passed")):
                raise ValueError("confirmation qualification failed; policy runs are forbidden")
            collection_config = dict(collection_run["configuration"])
            manifests = {}
            for label in POLICY_SPECS:
                frozen_models, registration = _policy_model_spec(
                    label,
                    project_root,
                    collection_config,
                    training_root,
                    v2_registration,
                )
                manifests[label] = _collect_policy(
                    label,
                    rows,
                    seeds,
                    dataset_root,
                    output_root,
                    run_fp,
                    config,
                    frozen_models,
                    registration,
                    effective_workers,
                    resume,
                )
            summary["collection"] = {
                label: {
                    "episode_count": len(values),
                    "success_count": sum(
                        bool(row.get("summary", {}).get("success")) for row in values
                    ),
                    "error_count": sum(
                        str(row.get("status")) not in {"ok", "resumed"}
                        for row in values
                    ),
                }
                for label, values in manifests.items()
            }
        else:
            qualification = _read_json(output_root / "qualification_report.json")
            if not bool(qualification.get("passed")):
                raise ValueError("confirmation qualification did not pass")
            manifests = {
                label: _read_jsonl(output_root / "policies" / label / "manifest.jsonl")
                for label in POLICY_SPECS
            }
            budget = int(config["metric_iteration_budget"])
            for label, values in manifests.items():
                _validate_policy_traces(output_root, label, values, run_fp, budget)
            policy_summaries = {
                label: summarize_policy(values) for label, values in manifests.items()
            }
            samples = int(config["bootstrap_samples"])
            v2_vs_adaptive = compare_policies(
                manifests["official_adaptive"], manifests["aggregated_v2"], samples, budget
            )
            v2_vs_v1 = compare_policies(
                manifests["frozen_v1"], manifests["aggregated_v2"], samples, budget
            )
            v1_vs_adaptive = compare_policies(
                manifests["official_adaptive"], manifests["frozen_v1"], samples, budget
            )
            integrity = _fingerprint_integrity(manifests)
            acceptance = closed_loop_v2_acceptance(
                policy_summaries["official_adaptive"],
                policy_summaries["frozen_v1"],
                policy_summaries["aggregated_v2"],
                v2_vs_adaptive,
                v2_vs_v1,
                integrity,
                dict(config["thresholds"]),
            )
            minimum = float(
                config["thresholds"]["minimum_closed_loop_auc_improvement_over_adaptive"]
            )
            v1_robust = (
                policy_summaries["frozen_v1"]["success_count"]
                >= policy_summaries["official_adaptive"]["success_count"]
                and v1_vs_adaptive["metrics"]["fixed_budget_conflict_auc"][
                    "relative_improvement"
                ]
                >= minimum
            )
            decision = (
                "use_v2_as_rl_warm_start"
                if acceptance["passed"]
                else "retain_v1_warm_start_and_use_policy_visited_replay"
                if v1_robust
                else "keep_rl_paused_and_redesign_candidate_control"
            )
            report = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "decision": decision,
                "qualification": qualification,
                "policy_summaries": policy_summaries,
                "comparisons": {
                    "v2_vs_adaptive": v2_vs_adaptive,
                    "v2_vs_v1": v2_vs_v1,
                    "v1_vs_adaptive": v1_vs_adaptive,
                },
                "initial_fingerprint_integrity": integrity,
                "closed_loop_acceptance": acceptance,
                "v1_remains_robust": v1_robust,
                "zero_conflict_tasks_retained_in_end_to_end": True,
                "policy_effect_metrics_condition_on_repairable_tasks": True,
                "static_context_used": False,
                "rl_trained": False,
            }
            _write_json(output_root / "independent_confirmation_report.json", report)
            summary["analysis"] = {
                "decision": decision,
                "closed_loop_passed": bool(acceptance["passed"]),
                "v1_remains_robust": v1_robust,
            }
    _write_json(output_root / "confirmation_summary.json", summary)
    return summary


__all__ = ["POLICY_SPECS", "run_independent_confirmation"]
