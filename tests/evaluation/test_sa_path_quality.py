from copy import deepcopy
import os
from pathlib import Path

import pytest

from scripts import run_sa_path_quality as q
from tests.evaluation.test_path_quality_execution import _cases


def test_full_schedule_and_stage2_pairing():
    rows=q.schedule(_cases())
    assert len(rows)==len({r["job_id"] for r in rows})==252
    assert {r["controller"] for r in rows}==set(q.CONTROLLERS)
    for i in range(0,252,3):
        group=rows[i:i+3]
        assert len({r["stage2_seed"] for r in group})==1
        assert len({(r["task_id"],r["solver_seed"],r["budget_seconds"]) for r in group})==1
    assert len({q.admission_key(r) for r in rows})==56
    assert {r["protocol"] for r in rows[:84]}=={"first_feasible"}
    assert {r["budget_seconds"] for r in rows[84:168]}=={60.0}
    assert {r["budget_seconds"] for r in rows[168:]}=={120.0}


def template():
    return dict(proposal=dict(max_seed_agents=4,heuristics=["target","collision","random"],
        neighborhood_sizes=[4,8,16],trials=8,candidates_per_family=2),
        environment=dict(time_limit=2.,max_repair_iterations=0,neighborhood_size=8,replan_algorithm="PP",use_sipp=True),
        frozen_models="unused",model_registration={})


def test_dual_methods_share_proposals_environment_and_seed(tmp_path):
    c=_cases(1)[0]; t=template(); before=deepcopy(t)
    items=q.schedule([c])[:3]
    jobs={i["controller"]:q.worker_job(c,i,t,tmp_path,"binding") for i in items}
    a,b=jobs["dual16"],jobs["dual16_sa"]
    assert a["environment"]==b["environment"] and a["sa_proposal"]==b["sa_proposal"]
    assert a["sa_case_id"]==b["sa_case_id"] and a["environment"]["max_repair_iterations"]==0
    case=dict(case_id=a["sa_case_id"])
    pool=[dict(agents=[1,2])]
    assert q.action_for("dual16",case,4,pool,0)==q.action_for("dual16_sa",case,4,pool,0)
    assert t==before
    assert "sa-stack-neighbors" not in q.NATIVE


@pytest.mark.parametrize("controller",q.CONTROLLERS)
@pytest.mark.parametrize("mode",["first_feasible","fixed_budget"])
def test_real_native_tiny_paths_and_stage2(tmp_path,monkeypatch,controller,mode):
    if os.environ.get("LNS2_SA_PATH_NATIVE_TESTS")!="1":
        pytest.skip("explicit frozen native smoke required")
    pytest.importorskip("lns2_env")
    monkeypatch.setattr(q,"ROOT",tmp_path)
    (tmp_path/"tiny.map").write_text("type octile\nheight 3\nwidth 3\nmap\n...\n...\n...\n")
    (tmp_path/"tiny.scen").write_text("version 1\n0\ttiny.map\t3\t3\t0\t0\t2\t2\t4\n")
    c=dict(task_id="tiny",map_id="tiny",family="warehouse",status="static_ready_runtime_unverified",
           solver_seeds=[51,52],static_audit=dict(agent_count=1),files=dict(map_file="tiny.map",scenario_file="tiny.scen"))
    item=dict(next(i for i in q.schedule([c]) if i["controller"]==controller and i["protocol"]==mode),budget_seconds=.3)
    r=dict(cases=[c],template=template(),fingerprint="test-only",inputs={})
    job=q.worker_job(c,item,r["template"],tmp_path/"out","test-only")
    env=q._make_environment(job["dataset_root"],job["row"],job["environment"],"Adaptive")
    state=q._plain(env.reset(seed=item["solver_seed"]))
    anchor=dict(state_fingerprint=q.state_fingerprint(state))
    spec=q.spec_for(r,item,anchor,tmp_path/"out")
    q.child(spec)
    result=q.execution.read_artifact(tmp_path/"out/result.json",spec["binding"])
    assert result["status"]=="completed"
    final=q.execution.read_artifact(tmp_path/"out/final_paths.json",spec["binding"])
    assert final["final_quality"]["feasible"]
    if mode=="fixed_budget":
        assert final["stage2_seed"]==item["stage2_seed"]
        assert final["final_quality"]["soc_steps"]<=final["initial_quality"]["soc_steps"]
    q.audit_trace(spec)
