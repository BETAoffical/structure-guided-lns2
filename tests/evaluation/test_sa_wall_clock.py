from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_sa_wall_clock as run


def test_balanced_serial_orders():
    assert all(sorted(order) == list(range(4)) for order in run.ORDERS)
    for position in range(4):
        assert Counter(order[position] for order in run.ORDERS) == Counter(range(4))
    pairs = Counter((a, b) for order in run.ORDERS for a, b in zip(order, order[1:]))
    assert len(pairs) == 12 and set(pairs.values()) == {1}


def test_time_cap_without_fabricated_success():
    assert run.capped_time({"ttf_seconds": None}) == 60
    assert run.capped_time({"ttf_seconds": 0.25}) == .25
    assert run.capped_time({"ttf_seconds": 61}) == 60
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            run.capped_time({"ttf_seconds": bad})


def test_official_action_does_not_reseed_and_model_seeds_are_paired():
    c = {"case_id": "case"}
    assert run.action_for("official", c, 0, [], None) == {"mode": "official"}
    assert run.action_for("official_sa", c, 0, [], None) == {"mode": "official"}
    pool = [{"agents": [1, 4]}]
    assert run.action_for("dual16", c, 2, pool, 0) == run.action_for("dual16_sa", c, 2, pool, 0)


def test_smoke_is_separate_and_no_timed_iteration_cap(monkeypatch):
    monkeypatch.setattr(run, "sha256_file", lambda _: "sha")
    plan = {"cases": [{"case_id": f"c{i}"} for i in range(16)], "config": {}}
    assert len(run.jobs(plan, "timed")) == 64
    assert all(j["max_steps"] is None for j in run.jobs(plan, "timed"))
    assert len(run.jobs(plan, "smoke")) == 8
    assert all(j["max_steps"] == 3 for j in run.jobs(plan, "smoke"))


def test_map_bootstrap_and_tradeoff_not_zero_loss_veto():
    rows = []
    for i in range(4):
        for arm in ("a", "b"):
            rows.append(dict(case_id=f"c{i}", map_id=f"m{i//2}", arm=arm,
                ttf_seconds=10 if arm == "a" else 5, success_within_budget=True))
    a = run.comparison(rows, "a", "b")
    assert a["decision"] == "development_timing_signal"
    assert a["paired_seconds_ci95"] == [-5, -5]
    assert run.comparison(rows, "a", "b") == a
    altered = deepcopy(rows)
    altered[1]["success_within_budget"] = False
    assert run.comparison(altered, "a", "b")["decision"] == "speed_reliability_tradeoff"


@pytest.fixture
def env(tmp_path):
    native = pytest.importorskip("lns2_env")
    if Path(native.__file__).resolve() != (run.ROOT / run.NATIVE).resolve():
        pytest.skip("official-SA interface tested separately with its registered isolated native")
    m, s = tmp_path / "tiny.map", tmp_path / "tiny.scen"
    m.write_text("type octile\nheight 2\nwidth 3\nmap\n...\n...\n")
    s.write_text("version 1\n0\ttiny.map\t3\t2\t0\t0\t2\t0\t2\n0\ttiny.map\t3\t2\t2\t0\t0\t0\t2\n")
    obj = native.LNS2RepairEnv(str(m), str(s), 2, time_limit=10)
    obj.reset_paths([[0, 1, 2], [2, 1, 0]], seed=19)
    return obj


def test_official_sa_zero_budget_rolls_back(env):
    before = env.get_state()
    out = env.step_experimental_pp({"mode": "official"}, 0, "annealed", 1000, 0)
    assert out["metrics"]["action_valid"]
    assert not out["metrics"]["replan_success"]
    assert out["observation"]["agents"] == before["agents"]


@pytest.mark.parametrize("extra", [{"random_seed": 1}, {"agents": [0]}, {"pp_random_seed": 1}, {"repair_order": [0]}])
def test_experimental_official_rejects_rng_overrides(env, extra):
    before = env.get_state()
    with pytest.raises(ValueError):
        env.step_experimental_pp({"mode": "official", **extra}, 1, "annealed", 1000, .5)
    assert env.get_state()["iteration"] == before["iteration"]


def test_official_sa_zero_temperature_matches_complete_greedy(env):
    from experiments.repair_collection import state_fingerprint
    fingerprints = []
    for mode in ("complete_greedy", "annealed"):
        env.reset_paths([[0, 1, 2], [2, 1, 0]], seed=19)
        fingerprints.append(state_fingerprint(env.step_experimental_pp({"mode": "official"}, 1, mode, 0, .5)["observation"]))
    assert fingerprints[0] == fingerprints[1]


def test_stop_after_episode_and_resume_reuses_complete_result(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    scheduled = [dict(job_id=f"j{i}", plan_sha256="sha", arm="admission", case={}) for i in range(2)]
    monkeypatch.setattr(run, "jobs", lambda *_: scheduled)
    monkeypatch.setattr(run, "sha256_file", lambda _: "sha")
    monkeypatch.setattr(run, "_CollectionRunLock", lambda *_: nullcontext())
    monkeypatch.setattr(run, "saved_result", lambda path, job: run.read_json(path))
    called = []
    def fake_jobs(worker, jobs, **kw):
        called.append(jobs[0]["job_id"])
        row = dict(status="ok", job_id=jobs[0]["job_id"])
        kw["on_result"](row)
        run.write_json(tmp_path / "STOP_AFTER_EPISODE.json", {"stop": True})
        return [row]
    monkeypatch.setattr(run, "_run_jobs", fake_jobs)
    result = run.execute({}, "admission")
    assert result["paused"] and called == ["j0"]
    assert run.read_json(tmp_path / "run_status.json")["status"] == "paused"
    (tmp_path / "STOP_AFTER_EPISODE.json").unlink()
    assert run.execute({}, "admission")["complete"]
    assert called == ["j0", "j1"]


def test_saved_corruption_rejected_before_resume(tmp_path):
    path = tmp_path / "result.json"
    run.write_json(path, {"status": "ok", "integrity_sha256": "wrong"})
    with pytest.raises(ValueError, match="hash"):
        run.saved_result(path, {"arm": "admission"})
