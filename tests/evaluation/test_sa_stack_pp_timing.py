from copy import deepcopy
from contextlib import nullcontext
import math

import pytest

from scripts import time_sa_stack_pp as timing


def test_schedule_is_serial_balanced_and_adjacent():
    cases = [dict(case_id=str(i), map_id=str(i // 2)) for i in range(8)]
    r = dict(cases=cases, config={}, native={}, records={str(i): {} for i in range(8)},
             lengths={str(i): 9 for i in range(8)})
    jobs = timing.schedule(r)
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 32
    assert timing.GATE["workers"] == 1
    for i in range(0, 32, 2):
        a, b = jobs[i:i+2]
        assert (a["case"], a["repeat"]) == (b["case"], b["repeat"])
        assert a["variant"] != b["variant"]
    for i in range(16):
        assert jobs[i]["variant"] != jobs[i+16]["variant"]


@pytest.mark.parametrize("length,expected", [(1,[0]),(2,[0,1]),(6,[0,2,5]),(249,[0,124,248])])
def test_fixed_samples(length, expected):
    assert timing.positions(length) == expected


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_bad_sample_length(value):
    with pytest.raises(ValueError): timing.positions(value)


@pytest.mark.parametrize("value", [-1, True, math.inf, math.nan, "0.1"])
def test_invalid_timer(value):
    with pytest.raises(ValueError): timing.seconds(value)


def test_gate_boundaries_do_not_round_up():
    assert timing.gate_decision(5,6,0) == "eligible_for_separate_ttf_confirmation"
    for args in ((4.9999,8,1),(10,5,1),(10,8,-.0001)):
        assert timing.gate_decision(*args) == "pp_gain_unconfirmed_do_not_expand_this_timing"


def synthetic_rows():
    return [dict(case_id=str(c), map_id=str(c//2), repeat=r, variant=v,
                 rows=[dict(decision=i, sampled=True, pp_seconds=.1 if v == "reference" else .09,
                            outer_step_seconds=.2, fingerprint=str(i)*64) for i in range(3)])
            for r in range(2) for c in range(8) for v in timing.VARIANTS]


def test_summary_determinism_pairing_and_nested_timers():
    rows = synthetic_rows()
    a = timing.summarize(rows)
    assert a == timing.summarize(rows)
    assert a["pp_reduction_percent"] == pytest.approx(10)
    assert a["paired_calls"] == 48
    assert a["cases_faster"] == 8
    assert a["no_ttf"] and not a["default_changed"]
    assert a["decision"] == "eligible_for_separate_ttf_confirmation"
    rows[1]["rows"][0]["fingerprint"] = "a"*64
    with pytest.raises(ValueError, match="trajectories differ"): timing.summarize(rows)


def test_missing_and_duplicate_jobs_rejected():
    rows = synthetic_rows()
    with pytest.raises(ValueError): timing.summarize(rows[:-1])
    with pytest.raises(ValueError): timing.summarize(rows + [deepcopy(rows[0])])


def test_semantic_mismatch_stops(monkeypatch):
    monkeypatch.setattr(timing.pilot, "state_fingerprint", lambda s: s["fingerprint"])
    metrics = dict(repair_order=[2,7], neighborhood=[2,7], replan_success=True,
                   pp_failure_reason="", pp_attempted_agent_count=2, pp_inserted_agent_count=2,
                   requested_collect_pp_diagnostics=False)
    timing.repair_equal({"fingerprint":"x"}, {"fingerprint":"x"}, metrics, metrics)
    with pytest.raises(ValueError):
        timing.repair_equal({"fingerprint":"x"}, {"fingerprint":"y"}, metrics, metrics)
    with pytest.raises(ValueError):
        timing.repair_equal({"fingerprint":"x"}, {"fingerprint":"x"}, dict(metrics, repair_order=[7,2]), metrics)


def test_result_identity_and_sample_validation():
    job = dict(job_id="j", registration_sha256="r", case=dict(case_id="c",map_id="m"),
               repeat=0, variant="reference", config=dict(frozen=dict(native_sha256="n")), length=3)
    row = dict(status="ok", job_id="j", registration_sha256="r", case_id="c", map_id="m",
               repeat=0, variant="reference", native_sha256="n", full_prefix_equal=True, no_ttf=True,
               rows=synthetic_rows()[0]["rows"])
    timing.validate_job(job,row)
    for changed in (dict(row,native_sha256="bad"), dict(row,status="timeout"), dict(row,rows=row["rows"][:-1])):
        with pytest.raises(ValueError): timing.validate_job(job,changed)


def fake_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "OUT", tmp_path)
    timing.write_json(tmp_path / "registration.json", {})
    monkeypatch.setattr(timing, "verify", lambda: {})
    monkeypatch.setattr(timing, "schedule", lambda r: [dict(job_id="job")])
    monkeypatch.setattr(timing.pilot, "_CollectionRunLock", lambda *a: nullcontext())
    def forbidden(*a, **kw):
        raise AssertionError("must not start a new timed job")
    monkeypatch.setattr(timing.pilot, "_run_jobs", forbidden)


def test_safe_stop_prevents_next_job(tmp_path, monkeypatch):
    fake_collection(tmp_path, monkeypatch)
    timing.write_json(tmp_path / "STOP_AFTER_JOB.json", {})
    assert timing.collect() == dict(stopped=True,completed=0)
    assert not (tmp_path / "started" / "job.json").exists()


def test_interrupted_attempt_cannot_be_silently_retried(tmp_path, monkeypatch):
    fake_collection(tmp_path, monkeypatch)
    timing.write_json(tmp_path / "started" / "job.json", {})
    with pytest.raises(ValueError, match="incomplete prior job"):
        timing.collect()
    with pytest.raises(ValueError, match="incomplete prior job"):
        timing.collect(True)
    assert timing.read_json(tmp_path / "status.json")["status"] == "failed_or_interrupted"


def test_resume_rejects_modified_output_before_loading(tmp_path, monkeypatch):
    fake_collection(tmp_path, monkeypatch)
    identity = timing.sha256_file(tmp_path / "registration.json")
    timing.write_json(tmp_path / "manifest.json", dict(registration_sha256=identity, files={"job":"wrong"}))
    timing.write_json(tmp_path / "jobs" / "job.json", {})
    with pytest.raises(ValueError, match="saved job changed"):
        timing.collect(True)


def test_quarantined_batch_fails_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "OUT", tmp_path)
    timing.write_json(tmp_path / "QUARANTINED.json", {})
    for operation in (timing.register, timing.verify, timing.analyze):
        with pytest.raises(ValueError, match="quarantined"):
            operation()
