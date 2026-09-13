from collections import Counter
from copy import deepcopy

import pytest

from scripts import run_sa_pressure_first_feasible as p


def cases():
    return [dict(map_id=f"map-{m}", task_id=f"map-{m}-{d}-{od}", family="warehouse",
                 status="static_ready_runtime_unverified", solver_seeds=[61,62],
                 pressure_design=dict(density=d, mode=dict(name=od)))
            for m in range(8) for d in (.15,.2,.25) for od in ("balanced","bottleneck_eligible")]


def test_complete_first_stage_and_paired_order():
    c = cases()
    p.check_cases(c)
    rows = p.schedule(c)
    assert len(rows) == len({r["job_id"] for r in rows}) == 288
    assert {r["protocol"] for r in rows} == {"first_feasible"}
    assert {r["budget_seconds"] for r in rows} == {120.}
    assert {r["repair_iteration_cap"] for r in rows} == {None}
    assert len({p.q.admission_key(r) for r in rows}) == 96
    orders = Counter()
    for i in range(0,288,3):
        group = rows[i:i+3]
        assert len({(r["task_id"],r["solver_seed"],r["stage2_seed"]) for r in group}) == 1
        orders[tuple(r["controller"] for r in group)] += 1
    assert len(orders) == 6 and set(orders.values()) == {16}
    assert rows == p.schedule(list(reversed(c)))


def test_cohort_rejects_replacement_missing_cells_and_seeds():
    for field, value in (("solver_seeds",[51,52]), ("family","room"), ("status","quarantined")):
        c = cases()
        c[0][field] = value
        with pytest.raises(ValueError): p.check_cases(c)
    c = cases()
    c[0]["pressure_design"] = deepcopy(c[1]["pressure_design"])
    with pytest.raises(ValueError): p.check_cases(c)
    with pytest.raises(ValueError): p.check_cases(cases()[:-1])


def test_native_and_output_isolation():
    assert p.OUT != p.q.OUT
    assert "sa-wall-clock-v1" in p.q.NATIVE
    assert p.PROTOCOL["total_planning_budgets_seconds"] == []
    assert p.q.child.__module__ == "scripts.run_sa_path_quality"


def test_stop_finishes_no_new_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(p,"OUT",tmp_path)
    r = dict(fingerprint="test", schedule=p.schedule(cases()))
    monkeypatch.setattr(p,"verify",lambda *a:r)
    monkeypatch.setattr(p,"anchors",lambda *a:{})
    monkeypatch.setattr(p,"spec_for",lambda *a:dict(output=str(tmp_path/"episodes"/"unused")))
    p.q.write_json(tmp_path/"smoke.json",dict(passed=True,binding="test"))
    p.q.write_json(tmp_path/"STOP_AFTER_EPISODE.json",dict(requested=True))
    assert p.collect() == dict(stopped=True,completed=0)
    assert not (tmp_path/"episodes").exists()


def test_interrupted_episode_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(p,"OUT",tmp_path)
    r = dict(fingerprint="test", schedule=p.schedule(cases()))
    monkeypatch.setattr(p,"verify",lambda *a:r)
    monkeypatch.setattr(p,"anchors",lambda *a:{})
    folder=tmp_path/"interrupted"
    folder.mkdir()
    monkeypatch.setattr(p,"spec_for",lambda *a:dict(output=str(folder)))
    p.q.write_json(tmp_path/"smoke.json",dict(passed=True,binding="test"))
    with pytest.raises(ValueError,match="interruption"):
        p.collect()
    assert p.q.read_json(tmp_path/"status.json")["status"] == "failed_or_interrupted"
