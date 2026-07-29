from __future__ import annotations

import collections
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.repair_collection import _read_json, _write_json
from experiments.run_output_guard import prepare_run_output


STALL_ORACLE_BATCH_SCHEMA = "lns2.stall_oracle_batch.v1"
ORACLE_CLASSIFICATIONS = {
    "candidate_pool_failure",
    "inconclusive",
    "no_confirmed_v2_failure",
    "selector_failure",
}


def normalize_oracle_plan(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    keys: set[tuple[str, str, int, int, str]] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"oracle plan row {index} is not an object")
        source = str(raw.get("source") or "")
        task_id = str(raw.get("task_id") or "")
        fingerprint = str(raw.get("before_repair_fingerprint") or "")
        solver_seed = raw.get("solver_seed")
        decision_index = raw.get("decision_index")
        trials = raw.get("trials_per_branch")
        if not source or not task_id or len(fingerprint) != 64:
            raise ValueError(f"oracle plan row {index} has invalid identity fields")
        if (
            type(solver_seed) is not int
            or solver_seed < 0
            or type(decision_index) is not int
            or decision_index < 0
            or type(trials) is not int
            or trials < 4
            or raw.get("all_candidates") is not True
        ):
            raise ValueError(f"oracle plan row {index} has invalid trial settings")
        key = (source, task_id, solver_seed, decision_index, fingerprint)
        if key in keys:
            raise ValueError("oracle plan contains a duplicate state")
        keys.add(key)
        normalized.append(
            {
                "source": source,
                "task_id": task_id,
                "solver_seed": solver_seed,
                "decision_index": decision_index,
                "before_repair_fingerprint": fingerprint,
                "trials_per_branch": trials,
                "all_candidates": True,
            }
        )
    if not normalized:
        raise ValueError("oracle plan is empty")
    return normalized


def oracle_job_id(index: int, row: dict[str, Any]) -> str:
    return (
        f"oracle-{index:03d}-d{int(row['decision_index']):04d}-"
        f"{str(row['before_repair_fingerprint'])[:10]}"
    )


def _validated_job_summary(
    job_root: Path,
    row: dict[str, Any],
    *,
    job_id: str,
) -> dict[str, Any]:
    probe = _read_json(job_root / "stalled_state_probe_report.json")
    audit_path = job_root / "audit" / "stall_oracle_report.json"
    audit = _read_json(audit_path)
    expected = str(row["before_repair_fingerprint"])
    if (
        str(probe.get("task_id")) != str(row["task_id"])
        or int(probe.get("solver_seed", -1)) != int(row["solver_seed"])
        or int(probe.get("decision_index", -1)) != int(row["decision_index"])
        or str(probe.get("before_repair_fingerprint")) != expected
        or probe.get("all_candidates") is not True
        or int(probe.get("trials_per_branch", 0))
        != int(row["trials_per_branch"])
        or str(audit.get("before_repair_fingerprint")) != expected
    ):
        raise ValueError(f"oracle job artifact identity mismatch: {job_id}")
    classification = str(audit.get("classification"))
    if classification not in ORACLE_CLASSIFICATIONS:
        raise ValueError(f"oracle job classification is invalid: {job_id}")
    stable = list(audit.get("stable_alternatives") or ())
    ranks = sorted(
        int(branch["candidate_rank"])
        for branch in stable
        if branch.get("candidate_rank") is not None
    )
    source_manifest = next(
        (
            item
            for item in _read_jsonl(Path(str(row["source"])) / "realized_dynamic_manifest.jsonl")
            if str(item.get("task_id")) == str(row["task_id"])
            and int(item.get("solver_seed", -1)) == int(row["solver_seed"])
        ),
        {},
    )
    return {
        "job_id": job_id,
        "task_id": row["task_id"],
        "solver_seed": row["solver_seed"],
        "decision_index": row["decision_index"],
        "before_repair_fingerprint": expected,
        "layout_mode": source_manifest.get("layout_mode", "unknown"),
        "agent_count": source_manifest.get("agent_count"),
        "classification": classification,
        "candidate_count": int(probe.get("candidate_count", 0)),
        "trial_count": int(audit.get("trial_count", 0)),
        "stable_alternative_count": len(stable),
        "best_stable_rank": ranks[0] if ranks else None,
        "pp_order_sensitive_neighborhood_count": int(
            audit.get("pp_order_sensitive_neighborhood_count", 0)
        ),
        "probe_report_sha256": sha256_file(
            job_root / "stalled_state_probe_report.json"
        ),
        "audit_report_sha256": sha256_file(audit_path),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run_logged(command: list[str], *, cwd: Path, log_prefix: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    log_prefix.with_suffix(".stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(
            f"oracle subprocess failed ({completed.returncode}): {' '.join(command)}"
        )


def _run_job(
    project_root: Path,
    jobs_root: Path,
    index: int,
    row: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    job_id = oracle_job_id(index, row)
    job_root = jobs_root / job_id
    audit_path = job_root / "audit" / "stall_oracle_report.json"
    if resume and audit_path.is_file():
        return _validated_job_summary(job_root, row, job_id=job_id)
    probe_command = [
        sys.executable,
        str(project_root / "scripts" / "probe_stalled_state.py"),
        "--source",
        str(row["source"]),
        "--output",
        str(job_root),
        "--task-id",
        str(row["task_id"]),
        "--solver-seed",
        str(row["solver_seed"]),
        "--decision-index",
        str(row["decision_index"]),
        "--trials",
        str(row["trials_per_branch"]),
        "--all-candidates",
    ]
    if job_root.is_dir() and any(job_root.iterdir()):
        probe_command.append("--resume")
    _run_logged(
        probe_command,
        cwd=project_root,
        log_prefix=job_root / "probe",
    )
    if not audit_path.is_file():
        _run_logged(
            [
                sys.executable,
                str(project_root / "scripts" / "audit_v2_stall_oracle.py"),
                "--source",
                str(job_root),
                "--output",
                str(job_root / "audit"),
                "--minimum-trials",
                str(row["trials_per_branch"]),
                "--stable-fraction",
                "0.75",
            ],
            cwd=project_root,
            log_prefix=job_root / "audit-run",
        )
    return _validated_job_summary(job_root, row, job_id=job_id)


def _batch_report(
    plan_path: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = collections.Counter(str(row["classification"]) for row in summaries)
    layout_counts = collections.Counter(str(row["layout_mode"]) for row in summaries)
    return {
        "schema": STALL_ORACLE_BATCH_SCHEMA,
        "complete": not errors and len(summaries) == len(rows),
        "plan": str(plan_path),
        "job_count": len(rows),
        "completed_job_count": len(summaries),
        "error_count": len(errors),
        "errors": errors,
        "classification_counts": dict(sorted(counts.items())),
        "layout_counts": dict(sorted(layout_counts.items())),
        "selector_failure_count": counts["selector_failure"],
        "candidate_pool_failure_count": counts["candidate_pool_failure"],
        "stable_alternative_count": sum(
            int(row["stable_alternative_count"]) for row in summaries
        ),
        "training_started": False,
        "controller_actions_changed": False,
    }


def report_existing_stall_oracle_jobs(
    plan: str | Path,
    jobs_source: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    plan_path = Path(plan).resolve()
    jobs_source_root = Path(jobs_source).resolve()
    output_root = Path(output).resolve()
    rows = normalize_oracle_plan(json.loads(plan_path.read_text(encoding="utf-8")))
    source_runner = jobs_source_root / "runner_config.json"
    identity = {
        "schema": STALL_ORACLE_BATCH_SCHEMA,
        "mode": "report-existing-jobs",
        "plan_sha256": sha256_file(plan_path),
        "source_runner_config_sha256": sha256_file(source_runner),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    prepare_run_output(output_root, resume=resume, identity=identity)
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        job_id = oracle_job_id(index, row)
        try:
            summaries.append(
                _validated_job_summary(
                    jobs_source_root / "jobs" / job_id,
                    row,
                    job_id=job_id,
                )
            )
        except Exception as error:
            errors.append(
                {
                    "job_id": job_id,
                    "task_id": row["task_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    summaries.sort(key=lambda row: str(row["job_id"]))
    report = _batch_report(plan_path, rows, summaries, errors)
    report["jobs_source"] = str(jobs_source_root)
    atomic_write_csv(output_root / "oracle_batch_summary.csv", summaries)
    _write_json(output_root / "oracle_batch_report.json", report)
    return report


def run_stall_oracle_plan(
    plan: str | Path,
    output: str | Path,
    *,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("oracle batch workers must be positive")
    project_root = Path(__file__).resolve().parents[1]
    plan_path = Path(plan).resolve()
    output_root = Path(output).resolve()
    rows = normalize_oracle_plan(json.loads(plan_path.read_text(encoding="utf-8")))
    identity = {
        "schema": STALL_ORACLE_BATCH_SCHEMA,
        "plan_sha256": sha256_file(plan_path),
        "job_count": len(rows),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "probe_script_sha256": sha256_file(
            project_root / "scripts" / "probe_stalled_state.py"
        ),
        "audit_script_sha256": sha256_file(
            project_root / "scripts" / "audit_v2_stall_oracle.py"
        ),
    }
    prepare_run_output(output_root, resume=resume, identity=identity)
    jobs_root = output_root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    _write_json(
        output_root / "status.json",
        {
            "schema": STALL_ORACLE_BATCH_SCHEMA,
            "status": "running",
            "total_jobs": len(rows),
            "completed_jobs": 0,
            "error_jobs": 0,
            "workers": workers,
        },
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_job,
                project_root,
                jobs_root,
                index,
                row,
                resume=resume,
            ): (index, row)
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            index, row = futures[future]
            try:
                summaries.append(future.result())
            except Exception as error:  # keep independent jobs resumable
                errors.append(
                    {
                        "job_id": oracle_job_id(index, row),
                        "task_id": row["task_id"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            _write_json(
                output_root / "status.json",
                {
                    "schema": STALL_ORACLE_BATCH_SCHEMA,
                    "status": "running" if len(summaries) + len(errors) < len(rows) else "finalizing",
                    "total_jobs": len(rows),
                    "completed_jobs": len(summaries),
                    "error_jobs": len(errors),
                    "workers": workers,
                },
            )
    summaries.sort(key=lambda row: str(row["job_id"]))
    report = _batch_report(plan_path, rows, summaries, errors)
    atomic_write_csv(output_root / "oracle_batch_summary.csv", summaries)
    _write_json(output_root / "oracle_batch_report.json", report)
    _write_json(
        output_root / "status.json",
        {
            "schema": STALL_ORACLE_BATCH_SCHEMA,
            "status": "complete" if report["complete"] else "error",
            "total_jobs": len(rows),
            "completed_jobs": len(summaries),
            "error_jobs": len(errors),
            "workers": workers,
        },
    )
    return report


__all__ = [
    "STALL_ORACLE_BATCH_SCHEMA",
    "normalize_oracle_plan",
    "oracle_job_id",
    "report_existing_stall_oracle_jobs",
    "run_stall_oracle_plan",
]
