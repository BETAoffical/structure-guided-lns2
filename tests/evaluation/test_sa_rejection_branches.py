from copy import deepcopy
import pytest
from scripts import probe_sa_rejection_branches as probe


def test_force_only_current_uniform_and_keep_original():
    event=dict(uniform=.999,metrics=dict(acceptance_probability=.98))
    original=deepcopy(event)
    assert probe.forced_draw("keep_rejection",event)==.999
    assert probe.forced_draw("accept_once",event)==0.
    assert event==original
    with pytest.raises(ValueError): probe.forced_draw("unknown",event)
    event["metrics"]["acceptance_probability"]=0.
    with pytest.raises(ValueError): probe.forced_draw("accept_once",event)


def test_fixed_all_nine_not_selected_by_outcome():
    rows=[dict(arm="dual16_sa",accepted=False,case_id=f"case{i%5}",decision=i) for i in range(9)]
    result=probe.selected_events(dict(attempts=list(reversed(rows))+[dict(arm="dual16_sa",accepted=True)]))
    assert len(result)==9 and result==sorted(rows,key=lambda r:(r["case_id"],r["decision"]))
    with pytest.raises(ValueError): probe.selected_events(dict(attempts=rows[:-1]))
    with pytest.raises(ValueError): probe.selected_events(dict(attempts=rows[:-1]+[rows[0]]))


@pytest.mark.parametrize("field",["repair_order","neighborhood","pp_attempted_agent_count",
    "pp_inserted_agent_count","pp_old_conflict_pair_count","pp_attempt_conflict_pair_count"])
def test_attempt_comparison_rejects_changes(field):
    m=dict(repair_order=[3,8],neighborhood=[3,8],pp_attempted_agent_count=2,
        pp_inserted_agent_count=2,pp_old_conflict_pair_count=4,pp_attempt_conflict_pair_count=5)
    probe.check_attempt(m,m)
    changed=deepcopy(m)
    changed[field]=[8,3] if isinstance(m[field],list) else m[field]+1
    with pytest.raises(ValueError): probe.check_attempt(m,changed)


def test_smoke_and_full_jobs_are_separate(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,"OUT",tmp_path)
    probe.write_json(tmp_path/"plan.json",{})
    plan=dict(targets=[dict(target_id=f"t{i}",case=dict(case_id=f"c{i}")) for i in range(9)],config={},horizon=64)
    smoke=probe.jobs(plan,"smoke")
    full=probe.jobs(plan,"branches")
    assert len(smoke)==2 and all(j["horizon"]==2 for j in smoke)
    assert len(full)==18 and all(j["horizon"]==64 for j in full)
    assert len({j["job_id"] for j in full})==18


def test_no_automatic_resume_of_saved_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,"OUT",tmp_path)
    j=dict(job_id="x",plan_sha256="p",phase="branches",mode="keep_rejection")
    row=dict(j,status="error",error="prefix mismatch")
    row["integrity_sha256"]=probe.pilot.digest(row)
    probe.write_json(tmp_path/"branches/x.json",row)
    with pytest.raises(ValueError,match="no automatic retry"): probe.load(j,"branches")


def test_pending_stop_prevents_reset(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,"OUT",tmp_path)
    probe.write_json(tmp_path/"STOP_AFTER_JOB.json",dict(requested=True))
    assert probe.worker(dict(job_id="x",plan_sha256="p"))["status"]=="paused"
