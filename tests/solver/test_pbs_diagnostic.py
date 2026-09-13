"""Opt-in PBS native regression checks; production builds do not expose this API."""
from pathlib import Path
import math

import pytest

native = pytest.importorskip("lns2_env")
pytestmark = pytest.mark.skipif(not hasattr(native.LNS2RepairEnv,"step_diagnostic_pbs"),reason="isolated PBS diagnostic build required")
ROOT = Path(__file__).resolve().parents[2]


def environment(algorithm="PBS", fixture="open"):
    paths = [[3,4,5],[5,4,3],[0]] if fixture=="open" else [[0,1,2],[2,1,0]]
    env=native.LNS2RepairEnv(str(ROOT/f"tests/data/pbs_{fixture}.map"),str(ROOT/f"tests/data/pbs_{fixture}.scen"),
        len(paths),time_limit=60,replan_algorithm=algorithm)
    return env,env.reset_paths(paths,seed=17)


def action():
    return dict(mode="explicit_neighborhood",agents=[0,1],random_seed=17)


def test_zero_budget_exact_rollback_and_followup():
    env,before=environment()
    result=env.step_diagnostic_pbs(action(),0.,16,False)
    assert result["metrics"]["pbs_stop_reason"]=="time_limit"
    assert [a["path"] for a in before["agents"]]==[a["path"] for a in result["observation"]["agents"]]
    assert result["observation"]["conflict_edges"]==before["conflict_edges"]
    assert result["metrics"]["pbs_low_level_calls"]==0
    follow=env.step_diagnostic_pbs(action(),5.,16,False)
    assert follow["observation"]["feasible"]


def test_success_preserves_external_path():
    env,before=environment()
    result=env.step_diagnostic_pbs(action(),5.,16,False)
    assert result["observation"]["feasible"]
    assert result["observation"]["agents"][2]["path"]==before["agents"][2]["path"]


def test_corridor_failure_can_be_repeated():
    env,before=environment(fixture="corridor")
    for _ in range(2):
        result=env.step_diagnostic_pbs(action(),.1,16,False)
        assert not result["metrics"]["replan_success"]
        assert [a["path"] for a in result["observation"]["agents"]]==[a["path"] for a in before["agents"]]
        assert result["observation"]["conflict_edges"]==before["conflict_edges"]


@pytest.mark.parametrize("seconds",[-1.,math.nan,math.inf])
def test_invalid_budget(seconds):
    env,_=environment()
    with pytest.raises(ValueError): env.step_diagnostic_pbs(action(),seconds,16,False)


@pytest.mark.parametrize("members",[[0],[0,0],[0,99]])
def test_invalid_set_is_not_official_fallback(members):
    env,before=environment()
    with pytest.raises(ValueError): env.step_diagnostic_pbs(dict(action(),agents=members),1.,16,False)
    assert env.get_state()["conflict_edges"]==before["conflict_edges"]


def test_api_rejects_pp_environment():
    env,_=environment("PP")
    with pytest.raises(ValueError): env.step_diagnostic_pbs(action(),1.,16,False)


def test_warm_root_preserves_contract():
    env,before=environment()
    result=env.step_diagnostic_pbs(action(),5.,16,True)
    assert result["observation"]["feasible"]
    assert result["metrics"]["pbs_warm_root"]
    assert result["observation"]["agents"][2]["path"]==before["agents"][2]["path"]


def test_mid_search_timeout_restores_real_external_paths():
    import json
    from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
    plan=ROOT/"build/pbs-repair-admission-v1/plan.json"
    if not plan.is_file(): pytest.skip("registered historical diagnostic input unavailable")
    case=next(c for c in json.loads(plan.read_text())["cases"] if c["id"]=="room-64-64-16")
    def restore():
        env=native.LNS2RepairEnv(str(ROOT/case["files"]["map_file"]),str(ROOT/case["files"]["scenario_file"]),
            len(case["paths"]),time_limit=60,replan_algorithm="PBS")
        return env,env.reset_paths(case["paths"],seed=case["seed"])
    env,before=restore()
    act=dict(mode="explicit_neighborhood",agents=case["agents"],random_seed=case["seed"])
    result=env.step_diagnostic_pbs(act,.005,16,False)
    assert result["metrics"]["pbs_stop_reason"]=="time_limit"
    assert result["metrics"]["pbs_low_level_calls"]>0
    assert repair_structure_fingerprint(result["observation"])==repair_structure_fingerprint(before)
    next_result=env.step_diagnostic_pbs(act,5.,16,False)
    fresh,_=restore()
    fresh_result=fresh.step_diagnostic_pbs(act,5.,16,False)
    assert repair_structure_fingerprint(next_result["observation"])==repair_structure_fingerprint(fresh_result["observation"])
