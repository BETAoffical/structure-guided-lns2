from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any

from experiments.realized_neighborhood_probe import (
    evaluation_seed,
    select_representative_neighborhoods,
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
    _qualification_worker,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    select_seed_agents,
    state_fingerprint,
)


CONFIRMATION_SCHEMA = "lns2.realized_ranking_confirmation.v1"


def _proposal_seed(
    state_id: str, seed_agent: int, heuristic: str, size: int, trial_index: int
) -> int:
    return int(
        _fingerprint(
            {
                "namespace": "realized-ranking-proposal-v1",
                "state_id": state_id,
                "seed_agent": int(seed_agent),
                "heuristic": str(heuristic),
                "neighborhood_size": int(size),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def _family(heuristic: str, size: int) -> str:
    return f"{heuristic}:{int(size)}"


def _od_mode(task_variant: str) -> str:
    if task_variant.startswith("balanced_"):
        return "balanced"
    if task_variant.startswith("bottleneck_"):
        return "bottleneck"
    return "unknown"


def _dataset_design(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    expected_tasks = {
        "balanced_80",
        "balanced_100",
        "bottleneck_80",
        "bottleneck_100",
    }
    errors: list[str] = []
    if any(str(row.get("split")) != split for row in rows):
        errors.append("dataset contains a non-confirmation split")
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
        variants = {str(row.get("task_variant")) for row in tasks}
        if variants != expected_tasks or len(tasks) != 4:
            errors.append(f"{map_id}: incomplete four-task pairing")
        if len({int(row["map_seed"]) for row in tasks}) != 1:
            errors.append(f"{map_id}: inconsistent map seed")
        if len({int(row["task_seed"]) for row in tasks}) != len(tasks):
            errors.append(f"{map_id}: repeated task seed")
    expected_layouts = {
        "regular_beltway": 4,
        "compartmentalized": 4,
        "dead_end_aisles": 4,
    }
    if dict(sorted(layout_counts.items())) != expected_layouts:
        errors.append("layout replication is not the registered 4/4/4 design")
    if len(rows) != 48 or len(by_map) != 12:
        errors.append("dataset is not the registered 12-map/48-task design")
    return {
        "passed": not errors,
        "errors": errors,
        "map_count": len(by_map),
        "task_count": len(rows),
        "layout_counts": dict(sorted(layout_counts.items())),
    }


def _reference_rows(reference: Path) -> list[dict[str, Any]]:
    if not reference.is_dir():
        raise ValueError(f"missing reference dataset: {reference}")
    rows: list[dict[str, Any]] = []
    for manifest in sorted(reference.glob("*/manifest.jsonl")):
        rows.extend(_read_jsonl(manifest))
    if not rows:
        raise ValueError(f"reference dataset has no manifests: {reference}")
    return rows


def _seed_isolation(
    rows: list[dict[str, Any]], references: list[str], project_root: Path
) -> dict[str, Any]:
    map_seeds = {int(row["map_seed"]) for row in rows}
    task_seeds = {int(row["task_seed"]) for row in rows}
    overlap: dict[str, Any] = {}
    for value in references:
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        reference = _reference_rows(path.resolve())
        map_overlap = sorted(map_seeds & {int(row["map_seed"]) for row in reference})
        task_overlap = sorted(task_seeds & {int(row["task_seed"]) for row in reference})
        if map_overlap or task_overlap:
            overlap[str(value)] = {
                "map_seeds": map_overlap,
                "task_seeds": task_overlap,
            }
    return {
        "passed": not overlap,
        "map_seed_count": len(map_seeds),
        "task_seed_count": len(task_seeds),
        "overlap": overlap,
        "references": list(references),
    }


def qualification_report(
    rows: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    config: dict[str, Any],
    design: dict[str, Any],
    seed_isolation: dict[str, Any],
    *,
    formal: bool,
) -> dict[str, Any]:
    settings = dict(config["qualification"])
    indexed = {str(row["task_id"]): row for row in qualification}
    errors = [
        str(row.get("error"))
        for row in qualification
        if str(row.get("status")) != "ok"
    ]
    eligible = []
    for row in rows:
        result = indexed.get(str(row["task_id"]))
        if result is None or str(result.get("status")) != "ok":
            continue
        initial = int(result.get("initial_conflicts", 0))
        if (
            bool(result.get("repairable"))
            and int(settings["minimum_initial_conflicts"]) <= initial
            <= int(settings["maximum_initial_conflicts"])
            and int(row["agent_count"]) <= int(settings["maximum_agent_count"])
        ):
            eligible.append({**result, "task_variant": row["task_variant"]})
    by_layout: collections.Counter[str] = collections.Counter(
        str(row["layout_mode"]) for row in eligible
    )
    coverage: dict[str, dict[str, list[Any]]] = {}
    for row in eligible:
        value = coverage.setdefault(str(row["map_id"]), {"od": [], "density": []})
        value["od"].append(_od_mode(str(row["task_variant"])))
        value["density"].append(int(row["agent_count"]))
    paired_maps = sorted(
        map_id
        for map_id, value in coverage.items()
        if set(value["od"]) == {"balanced", "bottleneck"}
        and set(value["density"]) == {80, 100}
    )
    formal_gates = {
        "repairable_task_count": len(eligible)
        >= int(settings["minimum_repairable_tasks"]),
        "repairable_layout_coverage": all(
            by_layout.get(layout, 0)
            >= int(settings["minimum_repairable_tasks_per_layout"])
            for layout in ("regular_beltway", "compartmentalized", "dead_end_aisles")
        ),
        "paired_map_coverage": len(paired_maps) == 12,
    }
    gates = {
        "dataset_design": bool(design["passed"]),
        "seed_isolation": bool(seed_isolation["passed"]),
        "qualification_integrity": len(indexed) == len(rows) and not errors,
        **(
            formal_gates
            if formal
            else {
                "repairable_task_count": bool(eligible),
                "repairable_layout_coverage": True,
                "paired_map_coverage": True,
            }
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "formal": formal,
        "passed": all(gates.values()),
        "gates": gates,
        "valid_count": sum(str(row.get("status")) == "ok" for row in qualification),
        "repairable_count": len(eligible),
        "repairable_task_ids": sorted(str(row["task_id"]) for row in eligible),
        "repairable_by_layout": dict(sorted(by_layout.items())),
        "paired_map_count": len(paired_maps),
        "paired_maps": paired_maps,
        "errors": errors,
        "dataset_design": design,
        "seed_isolation": seed_isolation,
    }


def _proposal_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    output_root = Path(job["output_root"])
    state_id = f"{row['task_id']}__seed_{int(job['solver_seed']):04d}__official_adaptive__decision_0000"
    root = output_root / "proposals" / str(row["split"]) / state_id
    metadata_path = root / "metadata.json"
    if job["resume"] and metadata_path.is_file():
        metadata = _read_json(metadata_path)
        if metadata.get("run_fingerprint") == job["run_fingerprint"] and metadata.get("complete"):
            return {**metadata, "status": "resumed"}
    try:
        environment = _make_environment(
            job["dataset_root"], row, job["environment"], "Adaptive"
        )
        state = _plain(environment.reset(seed=int(job["solver_seed"])))
        expected_fingerprint = state_fingerprint(state)
        proposals: list[dict[str, Any]] = []
        total_seconds = 0.0
        for seed_agent in select_seed_agents(state, int(job["proposal"]["max_seed_agents"])):
            for heuristic in job["proposal"]["heuristics"]:
                for size in job["proposal"]["neighborhood_sizes"]:
                    for trial_index in range(int(job["proposal"]["trials"])):
                        random_seed = _proposal_seed(
                            state_id, seed_agent, str(heuristic), int(size), trial_index
                        )
                        action = {
                            "mode": "seed",
                            "heuristic": str(heuristic),
                            "seed_agent": int(seed_agent),
                            "neighborhood_size": int(size),
                            "random_seed": random_seed,
                        }
                        started = time.perf_counter()
                        result = _plain(environment.propose(action))
                        elapsed = time.perf_counter() - started
                        total_seconds += elapsed
                        after = _plain(environment.get_state())
                        if after != state or state_fingerprint(after) != expected_fingerprint:
                            raise RuntimeError("proposal changed the source repair state")
                        if not bool(result.get("action_valid")) or not bool(result.get("generated")):
                            raise RuntimeError("valid fixed-heuristic proposal was rejected")
                        agents = sorted(map(int, result.get("neighborhood", [])))
                        if not agents or len(agents) != len(set(agents)):
                            raise RuntimeError("proposal returned an invalid agent set")
                        proposals.append(
                            {
                                "schema": CONFIRMATION_SCHEMA,
                                "state_id": state_id,
                                "family": _family(str(heuristic), int(size)),
                                "seed_agent": int(seed_agent),
                                "proposal_seed": random_seed,
                                "proposal_trial_index": trial_index,
                                "requested_size": int(size),
                                "agents": agents,
                                "proposal_runtime": elapsed,
                            }
                        )
        candidates = select_representative_neighborhoods(
            proposals, int(job["proposal"]["candidates_per_family"])
        )
        if not candidates:
            raise RuntimeError("proposal stage produced no explicit candidates")
        context = dict(state.get("context", {}))
        state_row = {
            "schema": CONFIRMATION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": job["run_fingerprint"],
            "state_id": state_id,
            "state_fingerprint": expected_fingerprint,
            "episode_id": f"{row['task_id']}__seed_{int(job['solver_seed']):04d}__official_adaptive",
            "decision_index": 0,
            "prefix_actions": [],
            "state": state,
            "split": str(row["split"]),
            "map_id": str(row["map_id"]),
            "task_id": str(row["task_id"]),
            "layout_mode": str(row["layout_mode"]),
            "task_variant": str(row["task_variant"]),
            "agent_count": int(row["agent_count"]),
            "solver_seed": int(job["solver_seed"]),
            "candidate_count": len(candidates),
            "proposal_outcome_count": len(proposals),
            "unique_proposal_neighborhood_count": len(
                {tuple(proposal["agents"]) for proposal in proposals}
            ),
            "candidates": candidates,
            "context_fingerprint": _fingerprint(context),
        }
        proposals_path = root / "proposals.jsonl"
        state_path = root / "state.json"
        _write_jsonl(proposals_path, proposals)
        _write_json(state_path, state_row)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": job["run_fingerprint"],
            "state_id": state_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "proposal_count": len(proposals),
            "candidate_count": len(candidates),
            "proposal_runtime": total_seconds,
            "state_file": state_path.relative_to(output_root).as_posix(),
            "proposals_file": proposals_path.relative_to(output_root).as_posix(),
            "metadata_file": metadata_path.relative_to(output_root).as_posix(),
            "complete": True,
            "status": "ok",
        }
        _write_json(metadata_path, metadata)
        return metadata
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": job["run_fingerprint"],
            "state_id": state_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "proposal_count": 0,
            "candidate_count": 0,
            "complete": False,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def _evaluation_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    state_row = job["state_row"]
    state_id = str(state_row["state_id"])
    output_root = Path(job["output_root"])
    root = output_root / "explicit" / str(row["split"]) / state_id
    metadata_path = root / "metadata.json"
    if job["resume"] and metadata_path.is_file():
        metadata = _read_json(metadata_path)
        if metadata.get("run_fingerprint") == job["run_fingerprint"] and metadata.get("complete"):
            return {**metadata, "status": "resumed"}
    outcomes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for candidate in state_row["candidates"]:
        agents = sorted(map(int, candidate["agents"]))
        proposal_seeds = sorted(map(int, candidate["proposal_seeds"]))
        for trial_index in range(int(job["evaluation_trials"])):
            random_seed = evaluation_seed(
                state_id, str(candidate["candidate_id"]), trial_index, proposal_seeds
            )
            try:
                environment = _make_environment(
                    job["dataset_root"], row, job["environment"], "Adaptive"
                )
                before = _plain(environment.reset(seed=int(state_row["solver_seed"])))
                if state_fingerprint(before) != str(state_row["state_fingerprint"]):
                    raise RuntimeError("explicit evaluation reset fingerprint mismatch")
                action = {
                    "mode": "explicit_neighborhood",
                    "agents": agents,
                    "random_seed": random_seed,
                }
                result = _plain(environment.step(action))
                metrics = result["metrics"]
                actual = sorted(map(int, metrics.get("neighborhood", [])))
                if not bool(metrics.get("action_valid")) or actual != agents:
                    raise RuntimeError("explicit evaluation changed or rejected the candidate")
                after = result["observation"]
                conflicts_before = int(before["num_of_colliding_pairs"])
                conflicts_after = int(after["num_of_colliding_pairs"])
                low_level = _low_level_delta(before, after)
                outcomes.append(
                    {
                        "schema": CONFIRMATION_SCHEMA,
                        "schema_version": SCHEMA_VERSION,
                        "run_fingerprint": job["run_fingerprint"],
                        "state_id": state_id,
                        "state_fingerprint": state_row["state_fingerprint"],
                        "candidate_id": candidate["candidate_id"],
                        "agents": agents,
                        "selection_families": candidate["selection_families"],
                        "proposal_seeds": proposal_seeds,
                        "evaluation_trial_index": trial_index,
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
                )
            except Exception as error:
                errors.append(
                    {
                        "schema": CONFIRMATION_SCHEMA,
                        "state_id": state_id,
                        "candidate_id": candidate["candidate_id"],
                        "evaluation_trial_index": trial_index,
                        "evaluation_seed": random_seed,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    outcomes_path = root / "outcomes.jsonl"
    errors_path = root / "errors.jsonl"
    _write_jsonl(outcomes_path, outcomes)
    _write_jsonl(errors_path, errors)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": job["run_fingerprint"],
        "state_id": state_id,
        "episode_id": state_row["episode_id"],
        "split": row["split"],
        "map_id": row["map_id"],
        "task_id": row["task_id"],
        "state_count": 1,
        "candidate_count": len(state_row["candidates"]),
        "outcome_count": len(outcomes),
        "error_count": len(errors),
        "outcomes_file": outcomes_path.relative_to(output_root).as_posix(),
        "errors_file": errors_path.relative_to(output_root).as_posix(),
        "metadata_file": metadata_path.relative_to(output_root).as_posix(),
        "complete": not errors,
        "status": "ok" if not errors else "error",
    }
    _write_json(metadata_path, metadata)
    return metadata


def _write_incremental_manifest(path: Path, key: str):
    existing = _read_jsonl(path) if path.is_file() else []
    indexed = {str(row.get(key, row.get("task_id"))): row for row in existing}

    def record(result: dict[str, Any]) -> None:
        indexed[str(result.get(key, result.get("task_id")))] = result
        _write_jsonl(path, [indexed[name] for name in sorted(indexed)])

    return record


def _state_rows(output_root: Path) -> list[dict[str, Any]]:
    manifest = _read_jsonl(output_root / "proposal_manifest.jsonl")
    rows = []
    for metadata in manifest:
        if str(metadata.get("status")) not in {"ok", "resumed"}:
            raise ValueError("proposal manifest contains an unsuccessful state")
        rows.append(_read_json(output_root / str(metadata["state_file"])))
    return sorted(rows, key=lambda row: str(row["state_id"]))


def _phase_summary(rows: list[dict[str, Any]], count_keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        "job_count": len(rows),
        "error_count": sum(str(row.get("status")) not in {"ok", "resumed"} for row in rows),
        **{
            key: sum(int(row.get(key, 0)) for row in rows)
            for key in count_keys
        },
    }


def run_confirmation_collection(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    if phase not in {"qualify", "propose", "evaluate", "all"}:
        raise ValueError("phase must be qualify, propose, evaluate, or all")
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported confirmation collection config")
    split = str(config["split"])
    rows = _load_dataset_rows(dataset_root, [split])
    if max_tasks is not None:
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        rows = rows[:max_tasks]
    effective_workers = int(workers or config["workers"])
    dataset_fp = _dataset_fingerprint(dataset_root)
    effective = {**config, "max_tasks_override": max_tasks}
    config_fp = _fingerprint(effective)
    run_fp = _fingerprint(
        {"dataset_fingerprint": dataset_fp, "configuration_fingerprint": config_fp}
    )
    design = _dataset_design(rows, split) if max_tasks is None else {
        "passed": True,
        "errors": [],
        "map_count": len({str(row["map_id"]) for row in rows}),
        "task_count": len(rows),
        "layout_counts": dict(collections.Counter(str(row["layout_mode"]) for row in rows)),
    }
    project_root = Path(__file__).resolve().parents[1]
    isolation = _seed_isolation(
        rows, list(config.get("reference_datasets", [])), project_root
    )
    proposal_count = (
        len(rows)
        * int(config["proposal"]["max_seed_agents"])
        * len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["trials"])
    )
    maximum_candidates = (
        len(rows)
        * len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["candidates_per_family"])
    )
    estimate = {
        "task_count": len(rows),
        "maximum_proposals": proposal_count,
        "maximum_candidates": maximum_candidates,
        "maximum_evaluation_outcomes": maximum_candidates
        * int(config["evaluation_trials"]),
        "workers": effective_workers,
    }
    if dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "run_fingerprint": run_fp,
            "dataset_design": design,
            "seed_isolation": isolation,
            "estimate": estimate,
        }
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "schema": CONFIRMATION_SCHEMA,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_fp,
        "configuration": effective,
        "configuration_fingerprint": config_fp,
        "run_fingerprint": run_fp,
        "dataset_design": design,
        "seed_isolation": isolation,
    }
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fp:
            raise ValueError("output contains a different confirmation configuration")
        if not resume:
            raise ValueError("output already exists; pass resume to continue")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, run_config)
    solver_seed = int(config["solver_seed"])
    phases = ("qualify", "propose", "evaluate") if phase == "all" else (phase,)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fp,
        "estimate": estimate,
    }
    for current in phases:
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
            path = output_root / "qualification_manifest.jsonl"
            record = _write_incremental_manifest(path, "task_id")
            with _CollectionRunLock(output_root, run_fp, "confirmation-qualification"):
                _run_jobs(
                    _qualification_worker,
                    jobs,
                    effective_workers,
                    phase="confirmation-qualification",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["episode_wall_time_limit_seconds"]),
                    on_result=record,
                )
            qualification = _read_jsonl(path)
            report = qualification_report(
                rows,
                qualification,
                config,
                design,
                isolation,
                formal=max_tasks is None,
            )
            _write_json(output_root / "qualification_report.json", report)
            summary["qualification"] = report
            if not report["passed"] and phase == "all":
                break
        elif current == "propose":
            qualification = _read_jsonl(output_root / "qualification_manifest.jsonl")
            report = qualification_report(
                rows,
                qualification,
                config,
                design,
                isolation,
                formal=max_tasks is None,
            )
            if not report["passed"]:
                raise ValueError("qualification gate failed; proposal collection is forbidden")
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
            path = output_root / "proposal_manifest.jsonl"
            record = _write_incremental_manifest(path, "state_id")
            with _CollectionRunLock(output_root, run_fp, "confirmation-proposal"):
                _run_jobs(
                    _proposal_worker,
                    jobs,
                    effective_workers,
                    phase="confirmation-proposal",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["episode_wall_time_limit_seconds"]),
                    on_result=record,
                )
            manifest = _read_jsonl(path)
            candidate_rows = _state_rows(output_root)
            _write_jsonl(output_root / "candidates.jsonl", candidate_rows)
            summary["proposal"] = _phase_summary(
                manifest, ("proposal_count", "candidate_count")
            )
            if summary["proposal"]["error_count"] and phase == "all":
                break
        elif current == "evaluate":
            state_rows = _state_rows(output_root)
            indexed = {str(row["task_id"]): row for row in rows}
            jobs = [
                {
                    "row": indexed[str(state_row["task_id"])],
                    "state_row": state_row,
                    "solver_seed": solver_seed,
                    "dataset_root": str(dataset_root),
                    "environment": config["environment"],
                    "evaluation_trials": int(config["evaluation_trials"]),
                    "output_root": str(output_root),
                    "run_fingerprint": run_fp,
                    "resume": resume,
                }
                for state_row in state_rows
            ]
            path = output_root / "collection_manifest.jsonl"
            record = _write_incremental_manifest(path, "state_id")
            with _CollectionRunLock(output_root, run_fp, "confirmation-evaluation"):
                _run_jobs(
                    _evaluation_worker,
                    jobs,
                    effective_workers,
                    phase="confirmation-evaluation",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["episode_wall_time_limit_seconds"]),
                    on_result=record,
                )
            manifest = _read_jsonl(path)
            summary["evaluation"] = _phase_summary(
                manifest, ("state_count", "candidate_count", "outcome_count")
            )
    _write_json(output_root / "collection_summary.json", summary)
    return summary


__all__ = [
    "CollectionLockError",
    "qualification_report",
    "run_confirmation_collection",
]
