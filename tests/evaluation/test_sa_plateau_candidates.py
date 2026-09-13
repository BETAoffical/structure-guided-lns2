from scripts import audit_sa_plateau_candidates as audit


def test_target_is_fixed_time_boundary_not_outcome():
    events=[dict(elapsed_seconds=x) for x in (119.5,120.,120.3)]
    assert audit.target_index(events)==1


def test_pair_coverage_distinguishes_nonconflicting_padding():
    state=dict(conflict_edges=[[2,8],[8,19]])
    c=dict(agents=[0,2,8,19])
    assert audit.coverage(state,c)==dict(internal_pairs=2,incident_pairs=2,nonconflicting_members=[0])
    assert audit.coverage(state,dict(agents=[8]))["internal_pairs"]==0


def test_four_trials_share_pp_seed_across_candidates():
    t=dict(job_id="state",selected_index=1,pool=[dict(agents=[2,8]),dict(agents=[8,19])])
    e=dict(action=dict(mode="explicit_neighborhood",agents=[8,19],random_seed=99),uniform=.4)
    jobs=audit.branch_specs(t,e)
    assert len(jobs)==9
    assert jobs[0]["trial"]==-1 and jobs[0]["action"]==e["action"]
    seeds=[]
    for trial in range(4):
        paired=[j for j in jobs if j["trial"]==trial]
        assert len({j["action"]["random_seed"] for j in paired})==1
        seeds.append(paired[0]["action"]["random_seed"])
    assert len(set(seeds))==4
    assert jobs==audit.branch_specs(t,e)


def test_candidate_permutation_does_not_change_trial_randomness():
    t=dict(job_id="state",selected_index=0,pool=[dict(agents=[1,2]),dict(agents=[3,4])])
    e=dict(action={},uniform=.5)
    left=audit.branch_specs(t,e)[1:]
    right=audit.branch_specs(dict(t,pool=list(reversed(t["pool"]))),e)[1:]
    def keyed(rows):
        return {(tuple(r["action"]["agents"]),r["trial"]):r["action"]["random_seed"] for r in rows}
    assert keyed(left)==keyed(right)
