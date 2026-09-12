"""Versioned supervision only: frozen solver, durable capture, independent full audit."""
import argparse
from pathlib import Path
import json
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from scripts import run_sa_independent_confirmation as source

pilot = source.pilot
OUT = ROOT / "build/sa-independent-confirmation-supervisor-v2"
EVIDENCE = ROOT / "artifacts/sa-independent-confirmation-v1/interruption.json"
SOLVER_FUSE, AUDIT_FUSE = 180, 1800


def failure_result(job, status, error):
    return dict(status=status, job_id=job["job_id"], stage=job["supervisor_stage"],
                case_id=job["case"]["case_id"], arm=job["arm"], error=str(error),
                plan_sha256=job["plan_sha256"], automatic_solver_retry=False)


def checked_row(path, job):
    row = read_json(path)
    if row.get("integrity_sha256") != pilot.digest({k: v for k, v in row.items() if k != "integrity_sha256"}):
        raise ValueError("result checksum mismatch")
    for k in ("job_id", "plan_sha256", "arm", "phase"):
        if row.get(k) != job[k]:
            raise ValueError("result identity mismatch: " + k)
    if row.get("status") != "ok":
        raise ValueError("not a solver result")
    if any(row.get(k) != job["case"][k] for k in ("case_id", "map_id")):
        raise ValueError("case identity mismatch")
    return row


def solver_worker(job):
    path = ROOT / job["capture_path"]
    if path.exists():
        raise ValueError("raw result exists; audit it instead of rerunning solver")
    captured = []
    def capture(row, request):
        if request is not job or captured:
            raise ValueError("unexpected frozen validation callback")
        captured.append(True)
        # This callback runs only after the frozen loop has recorded first-feasible TTF.
        row["supervision_schema"] = "lns2.sa_supervision.v2"
        row["validation_state"] = "pending_full_audit"
        row["integrity_sha256"] = pilot.digest(row)
        write_json(path, row)
    with patch.object(pilot, "validate_result", capture):
        pilot.worker(job)
    if len(captured) != 1:
        raise ValueError("frozen worker did not reach capture boundary")
    # Never expose frozen worker's misleading validated_seconds after deferring audit.
    return dict(status="ok", job_id=job["job_id"], stage="solver", raw_sha256=sha256_file(path))


def audit_worker(job):
    start = time.monotonic()
    path = ROOT / job["capture_path"]
    before = sha256_file(path)
    if before != job["raw_sha256"]:
        raise ValueError("raw result changed before audit")
    row = checked_row(path, job)
    pilot.validate_result(row, job)
    if sha256_file(path) != before:
        raise ValueError("raw result changed during audit")
    return dict(status="ok", job_id=job["job_id"], stage="audit", full_audit=True,
        raw_sha256=before, registration_sha256=job["registration_sha256"],
        audit_seconds=time.monotonic()-start, events=len(row["events"]))


def registration_payload():
    plan = source.verify()
    evidence = read_json(EVIDENCE)
    if evidence["source_plan_sha256"] != sha256_file(source.OUT / "plan.json"):
        raise ValueError("interruption evidence plan changed")
    inputs = {str(EVIDENCE.relative_to(ROOT).as_posix()): sha256_file(EVIDENCE),
              str(Path(__file__).resolve().relative_to(ROOT).as_posix()): sha256_file(Path(__file__)),
              "tests/evaluation/test_sa_supervised_confirmation.py": sha256_file(ROOT / "tests/evaluation/test_sa_supervised_confirmation.py"),
              "docs/SA_SUPERVISION_REVISION_ZH.md": sha256_file(ROOT / "docs/SA_SUPERVISION_REVISION_ZH.md")}
    for stage in ("admission", "parity", "smoke"):
        p = source.OUT / (stage + "_report.json")
        r = read_json(p)
        if not r["complete"] or r["plan_sha256"] != evidence["source_plan_sha256"]:
            raise ValueError("prerequisites incomplete")
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
        for name, h in r["files"].items():
            if sha256_file(source.OUT / stage / name) != h:
                raise ValueError("prerequisite result changed")
    for name, h in evidence["completed_formal_files"].items():
        if sha256_file(source.OUT / "timed" / name) != h:
            raise ValueError("retained original result changed")
    return dict(schema="lns2.sa_supervision_registration.v2", source_plan_sha256=evidence["source_plan_sha256"],
        inputs=inputs, imported=evidence["completed_formal_files"],
        authorized_retry=evidence["failed_job_id"], planning_budget=60, solver_fuse=SOLVER_FUSE,
        audit_fuse=AUDIT_FUSE, solver_workers=1, audit_overlaps_timing=False,
        solver_kernel_unchanged=True, full_audit_before_next_episode=True,
        diagnostic_substitution=False, max_repair_iterations=0), plan


def register():
    r, _ = registration_payload()
    p = OUT / "registration.json"
    if p.exists() and read_json(p) != r:
        raise ValueError("supervisor registration exists with different identity")
    write_json(p, r)
    return dict(registered=True, imported=len(r["imported"]), remaining=128-len(r["imported"]),
                registration_sha256=sha256_file(p))


def verify():
    expected, plan = registration_payload()
    if read_json(OUT / "registration.json") != expected:
        raise ValueError("supervisor registration changed; new run version required")
    with source.output_context():
        scheduled = pilot.jobs(plan, "timed")
    return expected, plan, scheduled


def receipt_path(job):
    return OUT / "audits" / (job["job_id"] + ".json")


def load_validated(job, registration):
    name = job["job_id"] + ".json"
    if name in registration["imported"]:
        path = source.OUT / "timed" / name
        if sha256_file(path) != registration["imported"][name]:
            raise ValueError("imported result changed")
        # Exact bytes were fully audited by v1 worker and parent, and bound before revision.
        return checked_row(path, job), path
    path = OUT / "raw" / name
    receipt = read_json(receipt_path(job))
    if (receipt.get("status") != "ok" or receipt.get("full_audit") is not True
            or receipt.get("job_id") != job["job_id"]
            or receipt.get("raw_sha256") != sha256_file(path)
            or receipt.get("registration_sha256") != sha256_file(OUT / "registration.json")):
        raise ValueError("missing or stale full audit attestation")
    return checked_row(path, job), path


def stage_run(job, stage, worker, timeout):
    request = dict(job, supervisor_stage=stage)
    directory = OUT / "attempts" / job["job_id"] / stage
    attempt = len(list(directory.glob("*.json"))) + 1
    result_path = directory / f"{attempt:03d}.json"
    rows = _run_jobs(worker, [request], workers=1, phase=stage, output_root=OUT,
        run_fingerprint=job["registration_sha256"], timeout_seconds=timeout,
        failure_result=failure_result, on_result=lambda r: write_json(result_path, r), stop_on_failure=True)
    if len(rows) != 1 or rows[0]["status"] != "ok":
        raise ValueError("stage failed; inspect durable attempt before retry: " + stage)
    return rows[0]


def collect(clear_stop=False):
    registration, _, scheduled = verify()
    identity = sha256_file(OUT / "registration.json")
    if clear_stop:
        (OUT / "STOP_AFTER_EPISODE.json").unlink(missing_ok=True)
    # Also take the original lock: old and revised collectors cannot share this dataset concurrently.
    with _CollectionRunLock(source.OUT, registration["source_plan_sha256"], "supervised-timed"), \
            _CollectionRunLock(OUT, identity, "supervised-timed", use_global_lock=False):
        for i, original in enumerate(scheduled):
            name = original["job_id"] + ".json"
            if name in registration["imported"] or receipt_path(original).exists():
                load_validated(original, registration)
                continue
            if (OUT / "STOP_AFTER_EPISODE.json").exists() or (source.OUT / "STOP_AFTER_EPISODE.json").exists():
                write_json(OUT / "run_status.json", dict(status="paused", completed=i, total=len(scheduled)))
                return dict(paused=True, completed=i)
            raw = OUT / "raw" / name
            job = dict(original, capture_path=raw.relative_to(ROOT).as_posix(), registration_sha256=identity)
            try:
                if not raw.exists():
                    if (OUT / "attempts" / job["job_id"] / "solver").exists():
                        raise ValueError("prior solver attempt without capture; no automatic retry")
                    write_json(OUT / "run_status.json", dict(status="running", stage="solver", completed=i,
                        total=len(scheduled), active_job=job["job_id"]))
                    print(f"solver {i+1}/{len(scheduled)} {job['job_id']}", flush=True)
                    stage_run(job, "solver", solver_worker, SOLVER_FUSE)
                checked_row(raw, job)
                job["raw_sha256"] = sha256_file(raw)
                write_json(OUT / "run_status.json", dict(status="running", stage="full_audit", completed=i,
                    total=len(scheduled), active_job=job["job_id"]))
                receipt = stage_run(job, "audit", audit_worker, AUDIT_FUSE)
                write_json(receipt_path(job), receipt)
                row, _ = load_validated(job, registration)
                print(json.dumps(dict(completed=i+1, ttf=row["ttf_seconds"],
                    success=row["success_within_budget"], events=len(row["events"]),
                    audit_seconds=receipt["audit_seconds"])), flush=True)
            except BaseException as error:
                write_json(OUT / "run_status.json", dict(status="error_or_interrupted", completed=i,
                    active_job=job["job_id"], error=f"{type(error).__name__}: {error}", raw_preserved=raw.exists()))
                raise
        files = {}
        for job in scheduled:
            _, path = load_validated(job, registration)
            files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        report = dict(complete=True, jobs=len(scheduled), registration_sha256=identity, files=files)
        report["audits"] = {receipt_path(j).relative_to(ROOT).as_posix(): sha256_file(receipt_path(j))
            for j in scheduled if j["job_id"] + ".json" not in registration["imported"]}
        write_json(OUT / "timed_report.json", report)
        write_json(OUT / "run_status.json", dict(status="complete", completed=len(scheduled), total=len(scheduled)))
    return report


def analyze():
    registration, plan, scheduled = verify()
    report = read_json(OUT / "timed_report.json")
    if not report["complete"] or report["jobs"] != len(scheduled) or report["registration_sha256"] != sha256_file(OUT / "registration.json"):
        raise ValueError("collection incomplete or registration mismatch")
    rows = []
    for name, h in report["audits"].items():
        if sha256_file(ROOT / name) != h:
            raise ValueError("audit attestation changed")
    for job in scheduled:
        row, path = load_validated(job, registration)
        if report["files"].get(path.relative_to(ROOT).as_posix()) != sha256_file(path):
            raise ValueError("completed collection changed")
        rows.append(row)
    summary = {}
    for arm in pilot.ARMS:
        chosen = [r for r in rows if r["arm"] == arm]
        summary[arm] = dict(episodes=len(chosen), successes=sum(r["success_within_budget"] for r in chosen),
            mean_capped_ttf=statistics.mean(pilot.capped_time(r) for r in chosen),
            median_capped_ttf=statistics.median(pilot.capped_time(r) for r in chosen),
            generated=sum(r["generated"] for r in chosen), pp_calls=sum(r["pp_calls"] for r in chosen))
    comparisons = [pilot.comparison(rows, a, b) for a, b in (("official", "official_sa"),
        ("dual16", "dual16_sa"), ("official", "dual16"), ("official", "dual16_sa"), ("official_sa", "dual16_sa"))]
    primary = comparisons[1]
    primary["confirmation_mean_ttf_passed"] = (primary["improvement_percent"] >= 5 and
        primary["paired_seconds_ci95"][1] < 0 and primary["successes"][1] >= primary["successes"][0])
    fields = ("case_id", "map_id", "arm", "ttf_seconds", "success_within_budget", "pp_calls", "generated",
              "reset_seconds", "selection_seconds", "pp_seconds", "stop_reason")
    result = dict(schema="lns2.sa_independent_analysis.supervision_v2", summary=summary, comparisons=comparisons,
        source_plan_sha256=registration["source_plan_sha256"], registration_sha256=report["registration_sha256"],
        details=[{k:r[k] for k in fields} for r in rows], no_default_promotion=True,
        claim_scope=plan["claim_scope"], success_noninferiority_established=False,
        imported_original_episodes=len(registration["imported"]), supervision_revision_is_not_solver_change=True)
    write_json(OUT / "analysis.json", result)
    return dict(summary=summary, primary=primary, no_default_promotion=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "verify", "collect", "resume", "stop", "analyze"))
    args = parser.parse_args()
    if args.phase == "stop":
        write_json(OUT / "STOP_AFTER_EPISODE.json", {"requested": True})
        result = {"stop_after_current_episode": True}
    elif args.phase in ("collect", "resume"):
        result = collect(clear_stop=args.phase == "resume")
    elif args.phase == "verify":
        r, _, jobs = verify()
        result = dict(verified=True, jobs=len(jobs), imported=len(r["imported"]))
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
