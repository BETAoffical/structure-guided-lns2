from scripts.diagnose_sa_confirmation_interruption import failure_result
from experiments.repair_collection import _run_jobs


def raise_original_error(job):
    raise RuntimeError("original-worker-error")


def wait_for_fuse(job):
    import time
    time.sleep(10)


def test_failure_record_preserves_case_job_without_generic_dataset_row():
    job = {"job_id": "failed", "case": {"row": {}}}
    result = failure_result(job, "timeout", "episode exceeded 180 seconds")
    assert result["status"] == "timeout"
    assert result["error"] == "episode exceeded 180 seconds"
    assert result["job_id"] == "failed"
    assert result["formal_timing_result"] is False


def test_original_worker_exception_survives_process_boundary(tmp_path):
    rows = _run_jobs(raise_original_error, [{"job_id": "test-error", "case": {}}], workers=1,
        phase="diagnostic-error", output_root=tmp_path, run_fingerprint="test", timeout_seconds=10,
        failure_result=failure_result, stop_on_failure=True)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "original-worker-error" in rows[0]["error"]


def test_external_timeout_has_a_durable_reason_not_key_error(tmp_path):
    rows = _run_jobs(wait_for_fuse, [{"job_id": "test-timeout", "case": {}}], workers=1,
        phase="diagnostic-timeout", output_root=tmp_path, run_fingerprint="test", timeout_seconds=1,
        failure_result=failure_result, stop_on_failure=True)
    assert len(rows) == 1
    assert rows[0]["status"] == "timeout"
    assert "episode exceeded" in rows[0]["error"]
