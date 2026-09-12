from copy import deepcopy
from pathlib import Path
from contextlib import nullcontext
import pytest

from scripts import run_sa_supervised_confirmation as run


def job():
    return dict(job_id="test", plan_sha256="plan", arm="official", phase="timed",
                case=dict(case_id="case", map_id="map"), registration_sha256="registration")


def row(j):
    return dict(status="ok", **{k:j[k] for k in ("job_id", "plan_sha256", "arm", "phase")},
                **j["case"], ttf_seconds=1.25, success_within_budget=True, events=[])


def save(path, r):
    r = deepcopy(r)
    r["integrity_sha256"] = run.pilot.digest(r)
    run.write_json(path, r)


def test_capture_is_durable_before_audit_and_excludes_fake_validation(tmp_path, monkeypatch):
    j = dict(job(), capture_path=str(tmp_path / "raw.json"))
    original = run.pilot.validate_result
    def frozen(request):
        r = row(request)
        run.pilot.validate_result(r, request)
        assert run.read_json(tmp_path / "raw.json")["ttf_seconds"] == 1.25
        r["validated_seconds"] = 999
        return r
    monkeypatch.setattr(run.pilot, "worker", frozen)
    assert run.solver_worker(j)["status"] == "ok"
    assert run.pilot.validate_result is original
    result = run.checked_row(tmp_path / "raw.json", j)
    assert result["validation_state"] == "pending_full_audit"
    assert "validated_seconds" not in result
    with pytest.raises(ValueError, match="audit it"):
        run.solver_worker(j)


def test_missing_capture_is_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run.pilot, "worker", lambda j: {})
    with pytest.raises(ValueError, match="capture boundary"):
        run.solver_worker(dict(job(), capture_path=str(tmp_path / "raw.json")))


def test_full_audit_is_bound_and_called_once(tmp_path, monkeypatch):
    path = tmp_path / "raw.json"
    j = job()
    save(path, row(j))
    j.update(capture_path=str(path), raw_sha256=run.sha256_file(path))
    calls = []
    monkeypatch.setattr(run.pilot, "validate_result", lambda r, j: calls.append(r))
    result = run.audit_worker(j)
    assert result["full_audit"] and len(calls) == 1
    assert result["raw_sha256"] == j["raw_sha256"]
    j["raw_sha256"] = "wrong"
    with pytest.raises(ValueError, match="changed before audit"):
        run.audit_worker(j)
    assert len(calls) == 1


def test_failed_audit_preserves_raw(tmp_path, monkeypatch):
    path = tmp_path / "raw.json"
    j = job()
    save(path, row(j))
    h = run.sha256_file(path)
    j.update(capture_path=str(path), raw_sha256=h)
    def bad(*args):
        raise ValueError("physics mismatch")
    monkeypatch.setattr(run.pilot, "validate_result", bad)
    with pytest.raises(ValueError, match="physics mismatch"):
        run.audit_worker(j)
    assert run.sha256_file(path) == h


@pytest.mark.parametrize("field,value", [("arm", "dual16"), ("case_id", "wrong"), ("map_id", "wrong"), ("status", "error")])
def test_identity_not_just_checksum(tmp_path, field, value):
    j = job()
    r = row(j)
    r[field] = value
    p = tmp_path / "raw.json"
    save(p, r)
    with pytest.raises(ValueError):
        run.checked_row(p, j)


def test_failure_preserves_nested_job_and_actual_reason():
    r = run.failure_result(dict(job(), supervisor_stage="audit"), "timeout", "external audit fuse")
    assert r["status"] == "timeout" and r["error"] == "external audit fuse"
    assert r["case_id"] == "case" and not r["automatic_solver_retry"]


def test_reused_import_requires_exact_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(run.source, "OUT", tmp_path)
    j = job()
    p = tmp_path / "timed/test.json"
    save(p, row(j))
    registration = dict(imported={"test.json": run.sha256_file(p)})
    assert run.load_validated(j, registration)[0]["ttf_seconds"] == 1.25
    save(p, dict(row(j), ttf_seconds=2))
    with pytest.raises(ValueError, match="imported result changed"):
        run.load_validated(j, registration)


def test_raw_is_not_validated_without_attestation(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    j = job()
    save(tmp_path / "raw/test.json", row(j))
    with pytest.raises(FileNotFoundError):
        run.load_validated(j, dict(imported={}))


def test_stop_before_new_job_and_pending_raw_resumes_only_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run.source, "OUT", tmp_path / "old")
    lock_requests = []
    def lock(*args, **kwargs):
        lock_requests.append(kwargs)
        return nullcontext()
    monkeypatch.setattr(run, "_CollectionRunLock", lock)
    run.write_json(tmp_path / "registration.json", {})
    j = job()
    registration = dict(imported={}, source_plan_sha256="plan")
    monkeypatch.setattr(run, "verify", lambda: (registration, {}, [j]))
    run.write_json(tmp_path / "STOP_AFTER_EPISODE.json", {})
    assert run.collect()["paused"]
    assert lock_requests == [{}, {"use_global_lock": False}]
    save(tmp_path / "raw/test.json", row(j))
    calls = []
    def stage(request, name, fn, timeout):
        calls.append(name)
        return dict(status="ok", full_audit=True, job_id=j["job_id"], raw_sha256=request["raw_sha256"],
            registration_sha256=run.sha256_file(tmp_path / "registration.json"), audit_seconds=.1)
    monkeypatch.setattr(run, "stage_run", stage)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    r = run.collect(clear_stop=True)
    assert r["complete"] and calls == ["audit"]
    assert run.collect()["complete"] and calls == ["audit"]


@pytest.mark.parametrize("arm", run.pilot.ARMS)
def test_native_old_worker_and_deferred_worker_actions_identical(tmp_path, arm):
    native = pytest.importorskip("lns2_env")
    if Path(native.__file__).resolve() != (run.ROOT / run.pilot.NATIVE).resolve():
        pytest.skip("requires frozen SA native")
    plan = run.read_json(run.source.OUT / "plan.json")
    with run.source.output_context():
        j = next(j for j in run.pilot.jobs(plan, "smoke") if j["arm"] == arm)
    old = run.pilot.worker(j)
    request = dict(j, capture_path=str(tmp_path / "raw.json"))
    run.solver_worker(request)
    new = run.checked_row(tmp_path / "raw.json", request)
    run.pilot.validate_result(new, request)
    assert run.pilot.state_fingerprint(old["final_state"]) == run.pilot.state_fingerprint(new["final_state"])
    assert len(old["events"]) == len(new["events"])
    for a, b in zip(old["events"], new["events"]):
        for k in ("action", "pool", "selected_index", "temperature", "uniform"):
            assert a[k] == b[k]
        for k in ("repair_order", "neighborhood", "replan_success", "pp_failure_reason"):
            assert a["metrics"][k] == b["metrics"][k]
