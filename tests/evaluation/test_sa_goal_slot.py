from experiments.goal_slot_certificate import goal_slot


def test_static_tail_allows_unbounded_late_arrival():
    graph={0:[0,1],1:[0,1,2],2:[1,2]}
    result=goal_slot(graph,0,2,{},0)
    assert result["viable_at_tick"]==[0]
    assert result["terminal_component"]==[0,1,2]


def test_temporary_goal_occupancy_requires_future_departure():
    graph={0:[0,1],1:[0,1,2],2:[1,2]}
    result=goal_slot(graph,0,1,{10:[1,2]},1)
    assert result["viable_at_tick"]==[0,1]
    assert result["terminal_component"]==[0,1]
    assert result["boundary_owners"]==[10]


def test_permanent_occupied_goal_has_no_witness():
    assert not goal_slot({0:[0,1],1:[0,1]},0,1,{9:[1]},0)["individually_feasible"]
