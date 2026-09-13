from copy import deepcopy
from collections import Counter

from scripts import diagnose_sa_pressure_300s as d
from scripts import run_sa_pressure_first_feasible as p
from tests.evaluation.test_sa_pressure_first_feasible import cases


def test_union_includes_either_failure_and_excludes_dual16_only():
    report=dict(episodes=[dict(task_id="a",solver_seed=61,controller="official_adaptive",success=False),
        dict(task_id="a",solver_seed=61,controller="dual16_sa",success=False),
        dict(task_id="b",solver_seed=62,controller="dual16_sa",success=False),
        dict(task_id="c",solver_seed=61,controller="dual16",success=False)])
    assert d.failure_union(report)==[("a",61),("b",62)]


def test_fixed_budget_and_two_way_order():
    old=dict(schedule=p.schedule(cases()))
    keys=sorted({(x["task_id"],x["solver_seed"]) for x in old["schedule"]})[:11]
    report=dict(episodes=[dict(task_id=t,solver_seed=s,controller="official_adaptive",success=False) for t,s in keys])
    rows=d.schedule(old,report)
    assert len(rows)==len({x["job_id"] for x in rows})==22
    assert {x["controller"] for x in rows}==set(d.CONTROLLERS)
    assert {x["budget_seconds"] for x in rows}=={300.}
    assert {x["repair_iteration_cap"] for x in rows}=={None}
    orders=Counter(tuple(x["controller"] for x in rows[i:i+2]) for i in range(0,22,2))
    assert sorted(orders.values())==[5,6]
    for i in range(0,22,2):
        assert len({(x["task_id"],x["solver_seed"]) for x in rows[i:i+2]})==1


def test_prefix_ignores_only_timing_not_science():
    a=dict(delta=dict(top_set=dict(runtime=1.,iteration=1),agents=[]),action=dict(mode="official"),
           selected_index=None,pool=[],metrics=dict(repair_order=[1,2]),temperature=1000.,uniform=.5)
    b=deepcopy(a); b["delta"]["top_set"]["runtime"]=2.
    assert d.scientific_event(a)==d.scientific_event(b)
    b["metrics"]["repair_order"]=[2,1]
    assert d.scientific_event(a)!=d.scientific_event(b)


def test_output_isolation_and_no_new_solver():
    assert d.OUT!=p.OUT and d.OUT!=d.q.OUT
    assert d.q.child.__module__=="scripts.run_sa_path_quality"
    assert "sa-wall-clock-v1" in d.q.NATIVE
