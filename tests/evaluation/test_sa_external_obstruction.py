from scripts.diagnose_sa_external_obstruction import reachable_prefix
from scripts.probe_sa_certificate_augmentation import candidates


def test_prefix_attributes_swaps_and_vertex_blockers():
    graph={0:[0,1],1:[0,1,2],2:[1,2]}
    r=reachable_prefix(graph,0,{10:[1,0]},1)
    assert r["layers"]==[[0],[]]
    assert r["blocked"][0][0]["vertex_owners"]==[10]
    assert r["blocked"][0][1]["swap_owners"]==[10]


def test_forced_collision_requires_shared_singleton():
    graph={0:[0,1,3],1:[0,1,2],2:[1,2],3:[0,3,6],6:[3,6]}
    fixed={10:[2,1],20:[6,3]}
    a=reachable_prefix(graph,1,fixed,1)
    b=reachable_prefix(graph,3,fixed,1)
    assert a["layers"][-1]==b["layers"][-1]==[0]


def test_goal_waiting_is_not_removed_after_path_end():
    graph={0:[0,1],1:[0,1]}
    r=reachable_prefix(graph,0,{90:[1]},4)
    assert r["layers"]==[[0]]*5
    assert all(layer[0]["owners"]==[90] for layer in r["blocked"])


def test_candidates_preserve_anchor_and_match_control_sizes():
    case=dict(id="case",selected=[0,1],state=dict(agents=[dict(id=i) for i in range(12)]))
    result=candidates(case,[2,3,4])
    assert result==candidates(case,[4,3,2])
    assert result[0]["agents"]==[0,1]
    for candidate in result:
        assert {0,1}<=set(candidate["agents"])
        if candidate["kind"]=="control":
            assert not {2,3,4}&set(candidate["added"])
    assert len(result[-1]["added"])==3 and len(result[-2]["added"])==1
