"""One isolated failed-job reproduction with durable phase evidence; never formal TTF."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from scripts import run_sa_independent_confirmation as source

OUT = ROOT / "build/sa-independent-interruption-audit-v1"


def failure_result(job, status, error):
    return dict(job_id=job["job_id"], status=status, error=error, phase="diagnostic",
                formal_timing_result=False)


def worker(job):
    start = time.monotonic()
    original = source.pilot.validate_result
    write_json(OUT / "phase.json", dict(phase="solver_start", elapsed_seconds=0, formal_timing_result=False))
    def validate(row, request):
        write_json(OUT / "phase.json", dict(phase="solver_complete_before_validation",
            elapsed_seconds=time.monotonic()-start, ttf_seconds=row["ttf_seconds"],
            loop_end_seconds=row["loop_end_seconds"], events=len(row["events"]),
            stop_reason=row["stop_reason"], formal_timing_result=False))
        write_json(OUT / "raw_before_validation.json", row)
        validation_start = time.monotonic()
        write_json(OUT / "phase.json", dict(phase="validation_started", elapsed_seconds=validation_start-start,
                                           events=len(row["events"]), formal_timing_result=False))
        original(row, request)
        write_json(OUT / "phase.json", dict(phase="validation_complete", elapsed_seconds=time.monotonic()-start,
            validation_seconds=time.monotonic()-validation_start, events=len(row["events"]), formal_timing_result=False))
    with patch.object(source.pilot, "validate_result", validate):
        row = source.pilot.worker(job)
    return dict(status="ok", job_id=job["job_id"], phase="diagnostic", formal_timing_result=False,
        elapsed_seconds=time.monotonic()-start, events=len(row["events"]),
        diagnostic_ttf_seconds=row["ttf_seconds"], loop_end_seconds=row["loop_end_seconds"],
        validated_seconds=row["validated_seconds"], pp_calls=row["pp_calls"],
        scientific_paths_validated=True, raw_sha256=sha256_file(OUT / "raw_before_validation.json"))


def main():
    if OUT.exists():
        raise ValueError("diagnostic already attempted; inspect evidence, do not automatically repeat")
    p = source.verify()
    stopped = read_json(source.OUT / "run_status.json")
    if stopped["status"] != "error_or_interrupted":
        raise ValueError("source was not interrupted")
    with source.output_context():
        job = deepcopy(next(j for j in source.pilot.jobs(p, "timed") if j["job_id"] == stopped["job"]))
    job["phase"] = "diagnostic"
    identity = sha256_file(source.OUT / "plan.json")
    with _CollectionRunLock(OUT, identity, "interruption-diagnostic"):
        write_json(OUT / "registration.json", dict(job=job, planning_budget=60, diagnostic_fuse=420,
            source_plan_sha256=identity, formal_timing_result=False,
            original_progress=read_json(source.OUT / "collection_progress.json")))
        rows = _run_jobs(worker, [job], workers=1, phase="interruption-diagnostic", output_root=OUT,
            run_fingerprint=identity, timeout_seconds=420, failure_result=failure_result,
            on_result=lambda r: write_json(OUT / "result.json", r), stop_on_failure=True)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
