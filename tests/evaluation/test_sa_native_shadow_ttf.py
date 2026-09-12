from contextlib import nullcontext
from copy import deepcopy

import pytest

from scripts import confirm_sa_native_shadow_ttf as run


def test_schedule_isolated_balanced(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    run.write_json(tmp_path / "registration.json", {})
    r = dict(rounds=2, cases=[dict(case_id=f"c{i}") for i in range(8)], config={}, budget=60)
    jobs = run.schedule(r)
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 32
    assert all(j["max_steps"] is None and j["budget"] == 60 and j["arm"] == "dual16_sa" for j in jobs)
    for i in range(8):
        orders = [[j["runtime_variant"] for j in jobs if j["case"]["case_id"] == f"c{i}" and j["repeat"] == n] for n in (0, 1)]
        assert orders[0] == orders[1][::-1]
    assert run.base.VARIANTS != run.VARIANTS and run.base.OUT != run.OUT


@pytest.mark.parametrize("variant", run.VARIANTS)
def test_native_two_step_equivalence(tmp_path, variant):
    native = pytest.importorskip("lns2_env")
    from pathlib import Path
    if Path(native.__file__).resolve() != (run.ROOT / run.base.pilot.NATIVE).resolve():
        pytest.skip("requires frozen SA native")
    plan = run.read_json(run.base.source.OUT / "plan.json")
    job = dict(case=plan["cases"][0], config=plan["config"], arm="dual16_sa", phase="smoke",
               budget=60., max_steps=2, job_id="native", plan_sha256="test", repeat=0,
               runtime_variant=variant, capture_path=str(tmp_path / "raw.json"))
    before = run.base.pilot.TimedPool
    a = run.base.pilot.worker(job)
    run.solver_worker(job)
    b = run.read_json(tmp_path / "raw.json")
    run.base.pilot.validate_result(b, job)
    assert run.base.compare_trajectories(a, b)["full_trajectory_equal"]
    assert run.base.pilot.TimedPool is before
    with pytest.raises(ValueError, match="raw exists"):
        run.solver_worker(job)


def test_stop_before_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run, "verify", lambda: {})
    monkeypatch.setattr(run, "schedule", lambda r: [dict(job_id="x")])
    monkeypatch.setattr(run.base.pilot, "_CollectionRunLock", lambda *a, **k: nullcontext())
    run.write_json(tmp_path / "registration.json", {})
    run.write_json(tmp_path / "preflight_report.json", dict(complete=True, cases=8, files={},
        registration_sha256=run.sha256_file(tmp_path / "registration.json")))
    run.write_json(tmp_path / "STOP_AFTER_EPISODE.json", {})
    assert run.collect() == dict(paused=True, completed=0)


def test_gate_does_not_promote_positive_noise_or_deadline_divergence():
    r = dict(summary={v:dict(successes=16) for v in run.VARIANTS},
             pairs=[dict(full_trajectory_equal=True, deadline_divergence=False)],
             mean_capped_ttf_improvement_percent=1., paired_seconds_map_bootstrap_ci95=[-.2, .1])
    assert run.decision(r) == "ttf_inconclusive_keep_component_evidence_only"
    r["paired_seconds_map_bootstrap_ci95"][1] = -.01
    assert run.decision(r) == "retain_engineering_variant_development_only"
    bad = deepcopy(r)
    bad["pairs"][0]["deadline_divergence"] = True
    assert run.decision(bad) == "do_not_retain"
    r["summary"][run.VARIANTS[1]]["successes"] = 15
    assert run.decision(r) == "do_not_retain"
