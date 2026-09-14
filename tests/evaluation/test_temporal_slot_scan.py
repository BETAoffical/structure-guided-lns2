import random

from experiments.temporal_slot_scan import viable_layers,forced_event
from experiments.goal_slot_certificate import goal_slot
from experiments.local_path_search import at
from scripts.validate_sa_certificate_generalization import choose_episodes,candidate_sets


def test_all_time_scan_matches_registered_single_tick_definition():
    rng=random.Random(20260915)
    graph={i:sorted({i,max(0,i-1),min(5,i+1)}) for i in range(6)}
    for _ in range(80):
        fixed={}
        for aid in (7,11):
            path=[rng.randrange(6)]
            for _ in range(rng.randrange(5)): path.append(rng.choice(graph[path[-1]]))
            fixed[aid]=path
        horizon=max(len(p)+1 for p in fixed.values())
        occupancy=[{at(p,t) for p in fixed.values()} for t in range(horizon+1)]
        swaps=[{(at(p,t),at(p,t-1)) for p in fixed.values() if t and at(p,t)!=at(p,t-1)} for t in range(horizon+1)]
        start,goal=rng.randrange(6),rng.randrange(6)
        layers,_=viable_layers(graph,start,goal,occupancy,swaps)
        for tick,layer in enumerate(layers):
            expected=goal_slot(graph,start,goal,fixed,tick)
            assert layer==set(expected.get("viable_at_tick",[]))


def test_forced_swap_and_vertex_are_necessary_not_merely_possible():
    assert forced_event([{0},{1}],[{1},{0}])["kind"]=="edge"
    assert forced_event([{0},{1}],[{2},{1}])["kind"]=="vertex"
    assert forced_event([{0},{1,2}],[{2},{1}]) is None


def test_episode_selection_deduplicates_extensions_and_ignores_outcomes():
    jobs=[dict(map_hash="new",task_id=str(i),solver_seed=1,cohort="full_120",job_id=str(i),success=i%2==0) for i in range(7)]
    jobs += [dict(jobs[0],cohort="failure_union_300",job_id="extension"),dict(jobs[1],map_hash="old",task_id="excluded")]
    selected=choose_episodes(jobs,{"old"})
    assert len(selected)==3 and all(j["map_hash"]=="new" for j in selected)
    assert selected==choose_episodes(list(reversed(jobs)),{"old"})
    changed=[dict(j,success=not j["success"]) for j in jobs]
    assert [j["job_id"] for j in selected]==[j["job_id"] for j in choose_episodes(changed,{"old"})]


def test_replacement_preserves_size_and_all_current_conflict_endpoints():
    case=dict(id="state",selected=[0,1,2],certificate=dict(blockers=[4]),
              state=dict(agents=[dict(id=i) for i in range(8)],conflict_edges=[[1,2]]))
    rows=candidate_sets(case,"replace")
    assert all(len(r["members"])==3 and {1,2}<=set(r["members"]) for r in rows)
    assert next(r["members"] for r in rows if r["id"]=="certificate-4")==[1,2,4]
    assert candidate_sets(dict(case,certificate=dict(blockers=list(range(10,20)))),"add")==[]
