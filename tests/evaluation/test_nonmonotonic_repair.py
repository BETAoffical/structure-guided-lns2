from copy import deepcopy
import math
import random

import pytest

from experiments.nonmonotonic_repair import acceptance_draw, probability, summarize, temperature, validate_transition


def test_temperature_and_independent_random_stream():
    random.seed(19)
    before = random.getstate()
    assert acceptance_draw(17) == acceptance_draw(17)
    assert random.getstate() == before
    assert temperature(0) == 1000
    assert temperature(1) == 990
    assert probability(2, 1) == math.exp(-2)
    assert probability(1, 0) == 0
    assert probability(0, 0) == probability(-2, 0) == 1


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_temperature(value):
    with pytest.raises(ValueError):
        probability(1, value)


def fixture():
    paths = [[0, 1, 2], [2, 1, 0], [4, 3]]
    before = {"rows": 2, "cols": 3, "obstacles": [False] * 6,
              "agents": [{"id": i, "path": p, "start": p[0], "goal": p[-1]} for i, p in enumerate(paths)],
              "sum_of_costs": 5, "num_of_colliding_pairs": 1, "conflict_edges": [[0, 1]]}
    after = deepcopy(before)
    after["agents"][0]["path"] = [0, 1, 4, 3, 4, 1, 2]
    after.update(sum_of_costs=9, num_of_colliding_pairs=2, conflict_edges=[[0, 1], [0, 2]])
    metrics = dict(action_valid=True, step_applied=True, neighborhood=[0], pp_rolled_back=False,
        conflicts_before=1, conflicts_after=2, pp_attempt_conflict_pair_count=2, pp_old_conflict_pair_count=1,
        experimental_acceptance="annealed", acceptance_temperature=1, acceptance_uniform=0.1,
        acceptance_evaluated=True, pp_inserted_agent_count=1, acceptance_probability=math.exp(-1), replan_success=True)
    return before, after, metrics


def test_increase_can_be_valid_but_is_not_allowed_in_standard():
    before, after, metrics = fixture()
    validate_transition(before, after, metrics, [0], "annealed", 1, 0.1)
    with pytest.raises(ValueError, match="standard conflicts"):
        validate_transition(before, after, metrics, [0], "standard", 1, 0.1)


@pytest.mark.parametrize("field,value", [("pp_inserted_agent_count", 0), ("acceptance_evaluated", False),
                                        ("acceptance_probability", 0.5), ("pp_rolled_back", True)])
def test_malformed_experimental_transition(field, value):
    before, after, metrics = fixture()
    metrics[field] = value
    with pytest.raises(ValueError):
        validate_transition(before, after, metrics, [0], "annealed", 1, 0.1)


def rows():
    return [dict(arm=a, case_id=c, trial=0, feasible=(a == "annealed"), censored=False,
                 generated=10, conflicts=[1, 0 if a == "annealed" else 1], auc=0.5,
                 accepted_increases=0) for c in ("a", "b") for a in ("standard", "complete_greedy", "annealed")]


def test_gate_demands_both_controls_multiple_states_and_complete_results():
    r = rows()
    assert summarize(r)["decision"] == "development_signal_only"
    r[0]["censored"] = True
    assert summarize(r)["decision"] == "inconclusive_resource"
    with pytest.raises(ValueError):
        summarize(r[:-1])
    with pytest.raises(ValueError):
        summarize(r + [r[0]])
    r = rows()
    r[2]["generated"] = 100
    assert summarize(r)["decision"] == "no_go_this_pilot"


@pytest.fixture
def native_env(tmp_path):
    native = pytest.importorskip("lns2_env")
    if not hasattr(native.LNS2RepairEnv, "step_experimental_pp"):
        pytest.skip("requires isolated experimental native")
    m, s = tmp_path / "test.map", tmp_path / "test.scen"
    m.write_text("type octile\nheight 2\nwidth 3\nmap\n...\n...\n")
    s.write_text("version 1\n0\ttest.map\t3\t2\t0\t0\t2\t0\t2\n0\ttest.map\t3\t2\t2\t0\t0\t0\t2\n")
    env = native.LNS2RepairEnv(str(m), str(s), 2, time_limit=30)
    env.reset_paths([[0, 1, 2], [2, 1, 0]], seed=19)
    return env


@pytest.mark.parametrize("mode", ["complete_greedy", "annealed"])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_native_zero_budget_always_rolls_back(native_env, mode, diagnostics):
    before = native_env.get_state()
    result = native_env.step_experimental_pp({"mode": "explicit_neighborhood", "agents": [0, 1],
        "random_seed": 19, "collect_pp_diagnostics": diagnostics}, 0.0, mode, 1e100, 0.0)
    metrics = result["metrics"]
    assert metrics["pp_failure_reason"] == "time_limit"
    assert metrics["pp_rolled_back"] and not metrics["acceptance_evaluated"]
    assert result["observation"]["agents"] == before["agents"]
    assert result["observation"]["sum_of_costs"] == before["sum_of_costs"]


@pytest.mark.parametrize("temp,draw,mode", [(float("nan"), .5, "annealed"), (1, 1, "annealed"),
                                           (1, -.1, "annealed"), (1, .5, "unknown")])
def test_invalid_native_inputs_do_not_mutate(native_env, temp, draw, mode):
    before = native_env.get_state()
    with pytest.raises(ValueError):
        native_env.step_experimental_pp({"mode": "explicit_neighborhood", "agents": [0, 1],
                                        "random_seed": 1}, 1, mode, temp, draw)
    assert native_env.get_state()["agents"] == before["agents"]
    assert native_env.get_state()["iteration"] == before["iteration"]


def test_default_metrics_stay_unchanged(native_env):
    metrics = native_env.step_with_time_limit({"mode": "explicit_neighborhood", "agents": [0, 1],
                                              "random_seed": 19}, 1)["metrics"]
    assert "experimental_acceptance" not in metrics


@pytest.mark.parametrize("agents", [[], [0, 0], [0, 99]])
def test_invalid_experimental_neighborhood_never_falls_back(native_env, agents):
    before = native_env.get_state()
    with pytest.raises(ValueError):
        native_env.step_experimental_pp({"mode": "explicit_neighborhood", "agents": agents,
                                        "random_seed": 1}, 1, "annealed", 1, .5)
    assert native_env.get_state()["agents"] == before["agents"]
    assert native_env.get_state()["iteration"] == before["iteration"]


def test_complete_paths_and_zero_temperature_parity(native_env):
    from experiments.repair_collection import state_fingerprint
    action = {"mode": "explicit_neighborhood", "agents": [0, 1], "random_seed": 19}
    results = []
    for mode, diagnostics in [("complete_greedy", False), ("annealed", True)]:
        native_env.reset_paths([[0, 1, 2], [2, 1, 0]], seed=19)
        result = native_env.step_experimental_pp({**action, "collect_pp_diagnostics": diagnostics}, 1, mode, 0, .5)
        assert result["metrics"]["acceptance_evaluated"]
        assert result["metrics"]["pp_inserted_agent_count"] == 2
        results.append(state_fingerprint(result["observation"]))
    assert results[0] == results[1]
