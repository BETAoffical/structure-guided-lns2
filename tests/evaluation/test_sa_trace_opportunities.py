from copy import deepcopy
import math

import pytest
from scripts import audit_sa_trace_opportunities as audit


@pytest.mark.parametrize("future,base,h,done,expected",[
    ([5,3],4,4,False,"below_pre_increase"),
    ([5,4],4,2,False,"not_below_in_observed_window"),
    ([5,4],4,4,False,"right_censored"),
    ([0],4,64,True,"below_pre_increase"),
    ([],4,4,False,"right_censored"),
    ([5,5,3],4,2,False,"not_below_in_observed_window"),
])
def test_recovery_window_and_censoring(future,base,h,done,expected):
    assert audit.recovery_window(future,base,h,done)==expected


def fixture():
    events=[]
    for d,(before,after) in enumerate([(2,3),(3,2),(2,0)]):
        m=dict(action_valid=True,step_applied=True,conflicts_before=before,conflicts_after=after,
            pp_old_conflict_pair_count=before,pp_attempt_conflict_pair_count=after,
            replan_success=True,pp_rolled_back=False,acceptance_evaluated=True,
            acceptance_temperature=1000.,acceptance_uniform=0.,
            acceptance_probability=math.exp(-1/1000) if after>before else 1.,
            neighborhood=[0,1],requested_collect_pp_diagnostics=False,pp_agent_diagnostics=[])
        m.update({k:.01 for k in audit.TIMERS})
        events.append(dict(metrics=m,elapsed_seconds=float(d+2)))
    r=dict(status="ok",arm="dual16_sa",plan_sha256=audit.PLAN_SHA,case_id="c",map_id="m",
        initial_state=dict(num_of_colliding_pairs=2),final_state=dict(num_of_colliding_pairs=0,feasible=True),
        events=events,reset_seconds=1.,ttf_seconds=4.,success_within_budget=True,
        loop_end_seconds=4.1,selection_seconds=.1,pp_seconds=.03)
    return seal(r)


def seal(row):
    row["integrity_sha256"]=audit.digest({k:v for k,v in row.items() if k!="integrity_sha256"})
    return row


def test_strict_recovery_not_equal_and_no_off_by_one():
    summary,inc,attempts=audit.episode(fixture())
    assert summary["categories"]==dict(increase=1,decrease=2)
    assert len(attempts)==1 and len(inc)==1
    assert inc[0]["first_below_steps"]==2
    assert inc[0]["first_below_seconds"]==2.
    assert inc[0]["windows"]["4"]=="below_pre_increase"


def test_no_pp_is_not_a_failed_rollback():
    r=fixture()
    r["events"]=r["events"][:1]
    m=r["events"][0]["metrics"]
    m.update(replan_success=False,pp_failure_reason="not_run",conflicts_after=2,
        pp_attempted_agent_count=0,pp_inserted_agent_count=0,repair_order=[],native_replan_seconds=0.)
    m.pop("acceptance_evaluated")
    r.update(final_state=dict(num_of_colliding_pairs=2,feasible=False),ttf_seconds=None,
             success_within_budget=False,pp_seconds=0.)
    s,inc,attempts=audit.episode(seal(r))
    assert s["categories"]==dict(pp_not_run=1) and not inc and not attempts
    m["pp_attempted_agent_count"]=1
    with pytest.raises(ValueError,match="not-run"):
        audit.episode(seal(r))


@pytest.mark.parametrize("change",["hash","probability","continuity","timer","ttf","accepted_delta"])
def test_bad_inputs_fail_closed(change):
    r=deepcopy(fixture())
    if change=="hash": r["case_id"]="tampered"
    if change=="probability": r["events"][0]["metrics"]["acceptance_probability"]=.1
    if change=="continuity": r["events"][1]["metrics"]["conflicts_before"]=12
    if change=="timer": r["events"][0]["metrics"]["native_replan_seconds"]=-1.
    if change=="ttf": r["ttf_seconds"]=3.
    if change=="accepted_delta": r["events"][0]["metrics"]["pp_attempt_conflict_pair_count"]=10
    if change!="hash": seal(r)
    with pytest.raises(ValueError): audit.episode(r)
