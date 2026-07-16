from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from experiments._common import (
    select_rows_by_task_id as _selected_rows,
    state_storage_id as _state_storage_id,
)
from experiments.realized_neighborhood_probe import evaluation_seed
from experiments.realized_ranking_confirmation import (
    _dataset_design,
    _phase_summary,
    _proposal_worker,
    _seed_isolation,
    _state_rows,
)
from experiments.repair_collection import (
    SCHEMA_VERSION,
    CollectionLockError,
    _CollectionRunLock,
    _baseline_worker,
    _dataset_fingerprint,
    _fingerprint,
    _load_dataset_rows,
    _low_level_delta,
    _make_environment,
    _plain,
    _qualification_worker,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)


NATURAL_SCHEMA = "lns2.natural_distribution_confirmation.v1"


def conflict_density(conflicts: int, agent_count: int) -> float:
    if agent_count < 2:
        return 0.0
    return 2.0 * float(conflicts) / float(agent_count * (agent_count - 1))


def conflict_severity(density: float, thresholds: dict[str, Any]) -> str:
    low = float(thresholds["low_max"])
    medium = float(thresholds["medium_max"])
    if not 0.0 <= low < medium:
        raise ValueError("conflict severity thresholds must satisfy 0 <= low < medium")
    if density <= low:
        return "low"
    if density <= medium:
        return "medium"
    return "high"


def _number_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def _distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(rows),
        "initial_feasible_count": sum(int(row["initial_conflicts"]) == 0 for row in rows),
        "nonzero_count": sum(int(row["initial_conflicts"]) > 0 for row in rows),
        "conflicts": _number_summary([float(row["initial_conflicts"]) for row in rows]),
        "conflict_density": _number_summary([float(row["conflict_density"]) for row in rows]),
        "severity_counts": dict(
            sorted(collections.Counter(str(row["conflict_severity"]) for row in rows).items())
        ),
    }


def natural_qualification_report(
    rows: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    config: dict[str, Any],
    design: dict[str, Any],
    seed_isolation: dict[str, Any],
    *,
    formal: bool,
) -> dict[str, Any]:
    settings = dict(config["qualification"])
    thresholds = dict(config["severity_thresholds"])
    indexed = {str(row["task_id"]): row for row in qualification}
    errors = [
        str(row.get("error"))
        for row in qualification
        if str(row.get("status")) != "ok"
    ]
    cohort = []
    for source in rows:
        result = indexed.get(str(source["task_id"]))
        if result is None or str(result.get("status")) != "ok":
            continue
        conflicts = int(result["initial_conflicts"])
        agents = int(source["agent_count"])
        density = conflict_density(conflicts, agents)
        cohort.append(
            {
                "split": str(source["split"]),
                "map_id": str(source["map_id"]),
                "task_id": str(source["task_id"]),
                "layout_mode": str(source["layout_mode"]),
                "task_variant": str(source["task_variant"]),
                "agent_count": agents,
                "initial_conflicts": conflicts,
                "initial_feasible": conflicts == 0,
                "repairable": conflicts > 0,
                "conflict_density": density,
                "conflict_severity": conflict_severity(density, thresholds),
            }
        )
    nonzero = [row for row in cohort if row["repairable"]]
    by_layout = collections.Counter(str(row["layout_mode"]) for row in nonzero)
    active_maps = sorted({str(row["map_id"]) for row in nonzero})
    integrity = len(indexed) == len(rows) and not errors
    if formal:
        sample_gates = {
            "minimum_nonzero_states": len(nonzero)
            >= int(settings["minimum_nonzero_states"]),
            "minimum_nonzero_per_layout": all(
                by_layout.get(layout, 0)
                >= int(settings["minimum_nonzero_states_per_layout"])
                for layout in (
                    "regular_beltway",
                    "compartmentalized",
                    "dead_end_aisles",
                )
            ),
            "minimum_active_maps": len(active_maps)
            >= int(settings["minimum_active_maps"]),
        }
    else:
        sample_gates = {
            "minimum_nonzero_states": bool(nonzero),
            "minimum_nonzero_per_layout": True,
            "minimum_active_maps": True,
        }
    gates = {
        "dataset_design": bool(design["passed"]),
        "seed_isolation": bool(seed_isolation["passed"]),
        "all_resets_valid": integrity and len(cohort) == len(rows),
        **sample_gates,
    }
    grouped: dict[str, Any] = {}
    for field in ("layout_mode", "task_variant", "agent_count", "conflict_severity"):
        values: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in cohort:
            values[str(row[field])].append(row)
        grouped[field] = {
            name: _distribution_summary(group) for name, group in sorted(values.items())
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "schema": NATURAL_SCHEMA,
        "formal": formal,
        "passed": all(gates.values()),
        "decision": (
            "eligible_for_natural_confirmation"
            if all(gates.values())
            else "inconclusive_sample_do_not_resample"
        ),
        "gates": gates,
        "valid_count": len(cohort),
        "initial_feasible_count": len(cohort) - len(nonzero),
        "nonzero_state_count": len(nonzero),
        "nonzero_by_layout": dict(sorted(by_layout.items())),
        "active_map_count": len(active_maps),
        "active_maps": active_maps,
        "repairable_task_ids": sorted(str(row["task_id"]) for row in nonzero),
        "zero_conflict_task_ids": sorted(
            str(row["task_id"]) for row in cohort if not row["repairable"]
        ),
        "severity_thresholds": thresholds,
        "natural_distribution": {
            "overall": _distribution_summary(cohort),
            "grouped": grouped,
            "tasks": sorted(cohort, key=lambda row: str(row["task_id"])),
        },
        "errors": errors,
        "dataset_design": design,
        "seed_isolation": seed_isolation,
    }


def _trial_job_id(state_id: str, candidate_id: str, trial_index: int) -> str:
    return f"{state_id}__{candidate_id}__trial_{trial_index:04d}"


def _trial_result_path(output_root: Path, job: dict[str, Any]) -> Path:
    return (
        output_root
        / "explicit_trials"
        / str(job["row"]["split"])
        / _state_storage_id(str(job["state_id"]))
        / str(job["candidate_id"])
        / f"trial_{int(job['evaluation_trial_index']):04d}.json"
    )


def _evaluation_trial_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    state_row = job["state_row"]
    candidate = job["candidate"]
    state_id = str(job["state_id"])
    candidate_id = str(job["candidate_id"])
    trial_index = int(job["evaluation_trial_index"])
    agents = sorted(map(int, candidate["agents"]))
    proposal_seeds = sorted(map(int, candidate["proposal_seeds"]))
    random_seed = evaluation_seed(state_id, candidate_id, trial_index, proposal_seeds)
    common = {
        "schema_version": SCHEMA_VERSION,
        "schema": NATURAL_SCHEMA,
        "run_fingerprint": job["run_fingerprint"],
        "job_id": job["job_id"],
        "state_id": state_id,
        "candidate_id": candidate_id,
        "evaluation_trial_index": trial_index,
        "split": str(row["split"]),
        "map_id": str(row["map_id"]),
        "task_id": str(row["task_id"]),
    }
    try:
        environment = _make_environment(
            job["dataset_root"], row, job["environment"], "Adaptive"
        )
        before = _plain(environment.reset(seed=int(state_row["solver_seed"])))
        if state_fingerprint(before) != str(state_row["state_fingerprint"]):
            raise RuntimeError("explicit evaluation reset fingerprint mismatch")
        result = _plain(
            environment.step(
                {
                    "mode": "explicit_neighborhood",
                    "agents": agents,
                    "random_seed": random_seed,
                }
            )
        )
        metrics = result["metrics"]
        actual = sorted(map(int, metrics.get("neighborhood", [])))
        if not bool(metrics.get("action_valid")) or actual != agents:
            raise RuntimeError("explicit evaluation changed or rejected the candidate")
        after = result["observation"]
        conflicts_before = int(before["num_of_colliding_pairs"])
        conflicts_after = int(after["num_of_colliding_pairs"])
        low_level = _low_level_delta(before, after)
        outcome = {
            **common,
            "agents": agents,
            "selection_families": candidate["selection_families"],
            "proposal_seeds": proposal_seeds,
            "evaluation_seed": random_seed,
            "evaluation_seed_disjoint": random_seed not in set(proposal_seeds),
            "actual_neighborhood": actual,
            "action_valid": True,
            "solved": bool(after["feasible"]),
            "conflicts_before": conflicts_before,
            "conflicts_after": conflicts_after,
            "conflict_auc": (conflicts_before + conflicts_after) / 2.0,
            "sum_of_costs_after": int(after["sum_of_costs"]),
            "generated": int(low_level["generated"]),
            "runtime": float(metrics["step_runtime"]),
        }
        return {
            **common,
            "status": "ok",
            "complete": True,
            "outcome_count": 1,
            "error": None,
            "outcome": outcome,
        }
    except Exception as error:
        return {
            **common,
            "status": "error",
            "complete": False,
            "outcome_count": 0,
            "error": f"{type(error).__name__}: {error}",
            "outcome": None,
        }


def _normalize_trial_result(
    result: dict[str, Any], job: dict[str, Any], output_root: Path, run_fingerprint: str
) -> dict[str, Any]:
    normalized = {
        **result,
        "schema_version": SCHEMA_VERSION,
        "schema": NATURAL_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "job_id": job["job_id"],
        "state_id": job["state_id"],
        "candidate_id": job["candidate_id"],
        "evaluation_trial_index": int(job["evaluation_trial_index"]),
    }
    normalized["complete"] = str(normalized.get("status")) in {"ok", "resumed"}
    normalized["outcome_count"] = int(normalized["complete"])
    path = _trial_result_path(output_root, job)
    normalized["result_file"] = path.relative_to(output_root).as_posix()
    _write_json(path, normalized)
    return normalized


def _collect_trial_results(
    jobs: list[dict[str, Any]],
    output_root: Path,
    run_fingerprint: str,
    workers: int,
    timeout_seconds: float,
    resume: bool,
) -> list[dict[str, Any]]:
    resumed: list[dict[str, Any]] = []
    pending = []
    for job in jobs:
        path = _trial_result_path(output_root, job)
        if resume and path.is_file():
            value = _read_json(path)
            if (
                value.get("run_fingerprint") == run_fingerprint
                and bool(value.get("complete"))
                and str(value.get("status")) in {"ok", "resumed"}
            ):
                resumed.append({**value, "status": "resumed"})
                continue
        pending.append(job)
    indexed_jobs = {str(job["job_id"]): job for job in pending}
    completed: list[dict[str, Any]] = []

    def record(result: dict[str, Any]) -> None:
        job = indexed_jobs[str(result["job_id"])]
        completed.append(
            _normalize_trial_result(result, job, output_root, run_fingerprint)
        )

    if pending:
        _run_jobs(
            _evaluation_trial_worker,
            pending,
            workers,
            phase="natural-confirmation-evaluation-trial",
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            timeout_seconds=timeout_seconds,
            on_result=record,
        )
    rows = sorted(
        resumed + completed,
        key=lambda row: (
            str(row["state_id"]),
            str(row["candidate_id"]),
            int(row["evaluation_trial_index"]),
        ),
    )
    _write_jsonl(output_root / "evaluation_trial_manifest.jsonl", rows)
    return rows


def _aggregate_trial_results(
    output_root: Path,
    state_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    expected_trials: int,
    run_fingerprint: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trial_rows:
        grouped[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    manifests = []
    for state_row in state_rows:
        state_id = str(state_row["state_id"])
        outcomes = []
        errors = []
        for candidate in state_row["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            values = grouped.get((state_id, candidate_id), [])
            indices = sorted(int(row["evaluation_trial_index"]) for row in values)
            if indices != list(range(expected_trials)):
                errors.append(
                    {
                        "state_id": state_id,
                        "candidate_id": candidate_id,
                        "error": "missing or duplicate evaluation trials",
                        "trial_indices": indices,
                    }
                )
                continue
            for value in values:
                if str(value.get("status")) not in {"ok", "resumed"} or not value.get(
                    "outcome"
                ):
                    errors.append(
                        {
                            "state_id": state_id,
                            "candidate_id": candidate_id,
                            "evaluation_trial_index": value.get(
                                "evaluation_trial_index"
                            ),
                            "status": value.get("status"),
                            "error": value.get("error"),
                        }
                    )
                else:
                    outcomes.append(value["outcome"])
        split = str(state_row["split"])
        root = output_root / "explicit" / split / _state_storage_id(state_id)
        outcomes_path = root / "outcomes.jsonl"
        errors_path = root / "errors.jsonl"
        _write_jsonl(outcomes_path, outcomes)
        _write_jsonl(errors_path, errors)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "schema": NATURAL_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "state_id": state_id,
            "episode_id": state_row["episode_id"],
            "split": split,
            "map_id": state_row["map_id"],
            "task_id": state_row["task_id"],
            "state_count": 1,
            "candidate_count": len(state_row["candidates"]),
            "outcome_count": len(outcomes),
            "error_count": len(errors),
            "outcomes_file": outcomes_path.relative_to(output_root).as_posix(),
            "errors_file": errors_path.relative_to(output_root).as_posix(),
            "complete": not errors,
            "status": "ok" if not errors else "error",
        }
        _write_json(root / "metadata.json", manifest)
        manifests.append(manifest)
    manifests.sort(key=lambda row: str(row["state_id"]))
    _write_jsonl(output_root / "collection_manifest.jsonl", manifests)
    return manifests


def run_natural_confirmation_collection(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    phases = {"qualify", "baseline", "propose", "evaluate", "all"}
    if phase not in phases:
        raise ValueError("phase must be qualify, baseline, propose, evaluate, or all")
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported natural confirmation config")
    split = str(config["split"])
    all_rows = _load_dataset_rows(dataset_root, [split])
    rows = _selected_rows(all_rows, task_ids)
    formal = task_ids is None
    effective_workers = int(workers or config["workers"])
    design = _dataset_design(all_rows, split)
    project_root = Path(__file__).resolve().parents[1]
    isolation = _seed_isolation(
        all_rows, list(config.get("reference_datasets", [])), project_root
    )
    dataset_fp = _dataset_fingerprint(dataset_root)
    effective = {**config, "task_ids_override": task_ids}
    config_fp = _fingerprint(effective)
    run_fp = _fingerprint(
        {"dataset_fingerprint": dataset_fp, "configuration_fingerprint": config_fp}
    )
    maximum_candidates = (
        len(rows)
        * len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["candidates_per_family"])
    )
    estimate = {
        "task_count": len(rows),
        "maximum_proposals": len(rows)
        * int(config["proposal"]["max_seed_agents"])
        * len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["trials"]),
        "maximum_candidates": maximum_candidates,
        "maximum_evaluation_trials": maximum_candidates
        * int(config["evaluation_trials"]),
        "workers": effective_workers,
    }
    if dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": NATURAL_SCHEMA,
            "dry_run": True,
            "run_fingerprint": run_fp,
            "formal": formal,
            "dataset_design": design,
            "seed_isolation": isolation,
            "estimate": estimate,
        }
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "schema": NATURAL_SCHEMA,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_fp,
        "configuration": effective,
        "configuration_fingerprint": config_fp,
        "run_fingerprint": run_fp,
        "formal": formal,
        "dataset_design": design,
        "seed_isolation": isolation,
    }
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fp:
            raise ValueError("output contains a different natural confirmation run")
        if not resume:
            raise ValueError("output already exists; pass resume to continue")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, run_config)
    solver_seed = int(config["solver_seed"])
    sequence = (
        ("qualify", "baseline", "propose", "evaluate")
        if phase == "all"
        else (phase,)
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": NATURAL_SCHEMA,
        "run_fingerprint": run_fp,
        "formal": formal,
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
            ]
            with _CollectionRunLock(output_root, run_fp, "natural-qualification"):
                results = _run_jobs(
                    _qualification_worker,
                    jobs,
                    effective_workers,
                    phase="natural-qualification",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["baseline_wall_time_limit_seconds"]),
                )
            _write_jsonl(output_root / "qualification_manifest.jsonl", results)
            report = natural_qualification_report(
                rows, results, config, design, isolation, formal=formal
            )
            _write_json(output_root / "qualification_report.json", report)
            summary["qualification"] = report
            if not report["passed"] and phase == "all":
                break
        elif current == "baseline":
            report = natural_qualification_report(
                rows,
                _read_jsonl(output_root / "qualification_manifest.jsonl"),
                config,
                design,
                isolation,
                formal=formal,
            )
            if not report["passed"]:
                raise ValueError("natural qualification gate failed; baseline is forbidden")
            jobs = [
                {
                    "row": row,
                    "policy": "official_adaptive",
                    "solver_seed": solver_seed,
                    "dataset_root": str(dataset_root),
                    "environment": config["environment"],
                    "output_root": str(output_root),
                    "run_fingerprint": run_fp,
                    "resume": resume,
                }
                for row in rows
            ]
            with _CollectionRunLock(output_root, run_fp, "natural-baseline"):
                results = _run_jobs(
                    _baseline_worker,
                    jobs,
                    effective_workers,
                    phase="natural-baseline",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["baseline_wall_time_limit_seconds"]),
                )
            _write_jsonl(output_root / "baseline_manifest.jsonl", results)
            summary["baseline"] = {
                "episode_count": len(results),
                "error_count": sum(
                    str(row.get("status")) not in {"ok", "resumed"} for row in results
                ),
                "success_count": sum(
                    bool(row.get("summary", {}).get("success")) for row in results
                ),
            }
            if summary["baseline"]["error_count"] and phase == "all":
                break
        elif current == "propose":
            report = natural_qualification_report(
                rows,
                _read_jsonl(output_root / "qualification_manifest.jsonl"),
                config,
                design,
                isolation,
                formal=formal,
            )
            if not report["passed"]:
                raise ValueError("natural qualification gate failed; proposals are forbidden")
            eligible = set(map(str, report["repairable_task_ids"]))
            jobs = [
                {
                    "row": row,
                    "solver_seed": solver_seed,
                    "dataset_root": str(dataset_root),
                    "environment": config["environment"],
                    "proposal": config["proposal"],
                    "output_root": str(output_root),
                    "run_fingerprint": run_fp,
                    "resume": resume,
                }
                for row in rows
                if str(row["task_id"]) in eligible
            ]
            with _CollectionRunLock(output_root, run_fp, "natural-proposal"):
                results = _run_jobs(
                    _proposal_worker,
                    jobs,
                    effective_workers,
                    phase="natural-proposal",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["proposal_wall_time_limit_seconds"]),
                )
            _write_jsonl(output_root / "proposal_manifest.jsonl", results)
            candidate_rows = _state_rows(output_root)
            _write_jsonl(output_root / "candidates.jsonl", candidate_rows)
            summary["proposal"] = _phase_summary(
                results, ("proposal_count", "candidate_count")
            )
            if summary["proposal"]["error_count"] and phase == "all":
                break
        elif current == "evaluate":
            state_rows = _state_rows(output_root)
            indexed = {str(row["task_id"]): row for row in rows}
            jobs = []
            for state_row in state_rows:
                for candidate in state_row["candidates"]:
                    for trial_index in range(int(config["evaluation_trials"])):
                        state_id = str(state_row["state_id"])
                        candidate_id = str(candidate["candidate_id"])
                        jobs.append(
                            {
                                "job_id": _trial_job_id(
                                    state_id, candidate_id, trial_index
                                ),
                                "state_id": state_id,
                                "candidate_id": candidate_id,
                                "evaluation_trial_index": trial_index,
                                "row": indexed[str(state_row["task_id"])],
                                "state_row": state_row,
                                "candidate": candidate,
                                "solver_seed": solver_seed,
                                "dataset_root": str(dataset_root),
                                "environment": config["environment"],
                                "run_fingerprint": run_fp,
                            }
                        )
            with _CollectionRunLock(output_root, run_fp, "natural-evaluation"):
                trial_rows = _collect_trial_results(
                    jobs,
                    output_root,
                    run_fp,
                    effective_workers,
                    float(config["trial_wall_time_limit_seconds"]),
                    resume,
                )
            manifests = _aggregate_trial_results(
                output_root,
                state_rows,
                trial_rows,
                int(config["evaluation_trials"]),
                run_fp,
            )
            summary["evaluation"] = {
                "state_count": len(manifests),
                "candidate_count": sum(int(row["candidate_count"]) for row in manifests),
                "trial_count": len(trial_rows),
                "outcome_count": sum(int(row["outcome_count"]) for row in manifests),
                "error_count": sum(int(row["error_count"]) for row in manifests),
                "timeout_count": sum(
                    str(row.get("status")) == "timeout" for row in trial_rows
                ),
            }
    _write_json(output_root / "collection_summary.json", summary)
    return summary


__all__ = [
    "CollectionLockError",
    "conflict_density",
    "conflict_severity",
    "natural_qualification_report",
    "run_natural_confirmation_collection",
]
