from collections import Counter
from copy import deepcopy

import pytest

from scripts import run_sa_independent_confirmation as run


def test_configuration_frozen_without_replacing_low_conflict_tasks():
    c = run.configuration()
    assert c["densities"] == [.15, .25]
    assert c["solver_seeds"] == [101, 103]
    assert c["preparation_workers"] == 20
    assert c["zero_conflict_retained"] and not c["replacement_permitted"]


def test_timing_engine_output_override_is_reversible(tmp_path, monkeypatch):
    previous = run.pilot.OUT
    monkeypatch.setattr(run, "OUT", tmp_path)
    with pytest.raises(RuntimeError):
        with run.output_context():
            assert run.pilot.OUT == tmp_path
            raise RuntimeError("interrupted")
    assert run.pilot.OUT == previous


def test_128_jobs_pairing_and_position_balance(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run.pilot, "sha256_file", lambda _: "sha")
    plan = {"cases": [{"case_id": f"m{i//4}-c{i}"} for i in range(32)], "config": {}}
    with run.output_context():
        jobs = run.pilot.jobs(plan, "timed")
    assert len(jobs) == 128
    assert Counter(j["arm"] for j in jobs) == Counter(dict.fromkeys(run.pilot.ARMS, 32))
    for pos in range(4):
        assert Counter(jobs[i]["arm"] for i in range(pos, 128, 4)) == Counter(dict.fromkeys(run.pilot.ARMS, 8))
    assert all(j["max_steps"] is None and j["budget"] == 60 for j in jobs)


def data(tmp_path):
    rows = []
    for i in range(8):
        path = tmp_path / "confirmation" / f"map{i}.map"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"geometry{i}")
        for j, d in enumerate((15, 25)):
            rows.append(dict(map_id=f"m{i}", map_seed=i, task_seed=100+2*i+j,
                             split="confirmation", map_file=path.name, task_variant=f"balanced_od_d{d}"))
    return rows, {"seeds": [], "map_hashes": []}


def test_map_seed_and_geometry_isolation(tmp_path):
    rows, history = data(tmp_path)
    assert len(run.check_isolation(rows, history, tmp_path)) == 8
    with pytest.raises(ValueError, match="historical seed"):
        run.check_isolation(rows, {**history, "seeds": [0]}, tmp_path)
    with pytest.raises(ValueError, match="geometry duplicate"):
        run.check_isolation(rows, {**history, "map_hashes": [run.sha256_file(tmp_path / "confirmation/map0.map")]}, tmp_path)
    altered = deepcopy(rows)
    altered[1]["task_seed"] = altered[0]["task_seed"]
    with pytest.raises(ValueError, match="seed duplication"):
        run.check_isolation(altered, history, tmp_path)


def test_each_map_has_two_independent_density_tasks(tmp_path):
    rows, history = data(tmp_path)
    rows[1]["task_variant"] = "balanced_od_d15"
    with pytest.raises(ValueError, match="unpaired density"):
        run.check_isolation(rows, history, tmp_path)


def test_generation_cannot_overwrite_partial_shard(tmp_path):
    with pytest.raises(ValueError, match="partial generated"):
        run.generation_worker(dict(output=str(tmp_path)))


def test_confirmation_does_not_reimplement_the_controller():
    from scripts.run_sa_wall_clock import worker, TimedPool
    assert run.pilot.worker is worker
    assert run.pilot.TimedPool is TimedPool


def test_bad_runtime_parameters_rejected(monkeypatch):
    c = deepcopy(run.configuration())
    c["budget_seconds"] = 120
    monkeypatch.setattr(run, "read_json", lambda _: c)
    with pytest.raises(ValueError, match="frozen protocol"):
        run.configuration()
