from copy import deepcopy
from unittest.mock import patch

import pytest

from scripts import diagnose_uniform_runtime as probe


def test_official_action_does_not_reseed():
    assert probe.action_for("official", {}, 42) == {"mode": "official"}


def test_uniform_arms_share_random_stream():
    case, pool = {"case_id": "paired"}, [{"agents": [2, 9]}]
    a = probe.action_for("uniform_standard", case, 4, pool, 0)
    assert a == probe.action_for("uniform_complete", case, 4, pool, 0)
    assert a != probe.action_for("uniform_complete", case, 5, pool, 0)


def test_only_score_is_removed():
    pool = [{"candidate_id": "x", "agents": [1, 4], "sources": ["target"], "score": 1.5}]
    before = deepcopy(pool)
    assert probe.unscored(pool) == [{"candidate_id": "x", "agents": [1,4], "sources": ["target"]}]
    assert pool == before


def test_guard_refuses_model_and_features():
    from experiments.compact_controller_model import load_controller_bundle
    # Resolve through the module while patched, not an old imported alias.
    import experiments.compact_controller_model as model
    from lns2_selector.runtime import online_selection
    with probe.no_model_guard():
        with pytest.raises(AssertionError, match="scoring called"):
            model.load_controller_bundle("unused")
        with pytest.raises(AssertionError, match="scoring called"):
            online_selection.score_online_candidates([], None)
    assert model.load_controller_bundle is load_controller_bundle


def rows():
    return [{"case_id": f"c{i}", "map_id": f"m{i//2}", "arm": a, "feasible": True,
             "initial_fingerprint": f"fp{i}", "decisions": 10, "pp_calls": 10,
             "generated": 100 if a == "official" else 90, "censored": False}
            for i in range(16) for a in probe.ARMS]


def test_gates_are_per_arm_and_exact():
    data = rows()
    result = probe.summarize(data)
    assert result["comparisons"]["uniform_standard"]["passed"]
    for r in data:
        if r["arm"] == "uniform_complete":
            r["generated"] = 96
    result = probe.summarize(data)
    assert result["comparisons"]["uniform_standard"]["passed"]
    assert not result["comparisons"]["uniform_complete"]["passed"]


def test_success_loss_is_not_hidden_by_equal_total():
    data = rows()
    next(r for r in data if r["case_id"] == "c0" and r["arm"] == "uniform_complete")["feasible"] = False
    next(r for r in data if r["case_id"] == "c1" and r["arm"] == "official")["feasible"] = False
    c = probe.summarize(data)["comparisons"]["uniform_complete"]
    assert c["losses"] == ["c0"] and c["gains"] == ["c1"] and not c["passed"]


def test_pairing_and_missing_results_refused():
    data = rows()
    with pytest.raises(ValueError, match="incomplete"):
        probe.summarize(data[:-1])
    data[0]["initial_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="unpaired"):
        probe.summarize(data)


def test_resource_censor_blocks_promotion():
    data = rows()
    data[0]["censored"] = True
    result = probe.summarize(data)
    assert all(not c["passed"] for c in result["comparisons"].values())


def test_unscored_pool_selects_without_loading_model():
    from types import SimpleNamespace
    pool = [{"candidate_id": "b", "agents": [3]}, {"candidate_id": "a", "agents": [1,2]}]
    case = {"case_id": "c", "task_id": "task", "solver_seed": 19, "proposal": {}}
    state = {"id": "state"}
    env = SimpleNamespace(get_state=lambda: state)
    with probe.no_model_guard(), patch.object(probe, "state_fingerprint", return_value="fp"), \
            patch("experiments.online_feature_engine.TopologyAnalysisCache", return_value=SimpleNamespace(analysis="topology")), \
            patch("lns2_selector.runtime.online_selection.generate_online_candidates", return_value=(pool, {})), \
            patch("lns2_selector.runtime.structshell_dual16.generate_structshell_dual16_runtime_candidates", return_value=SimpleNamespace(candidates=pool)):
        index, got = probe.UnscoredPool(case).select(env, state, 0)
    assert got == pool and index in (0,1)


def test_registered_schedule_is_bounded():
    assert probe.HORIZON == 256 and probe.WORKERS == 20
    assert probe.ARMS == ("official", "uniform_standard", "uniform_complete")
