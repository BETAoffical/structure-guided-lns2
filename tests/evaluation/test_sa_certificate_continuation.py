from scripts.probe_sa_certificate_continuation import paired_action


def test_first_action_preserves_conflict_endpoints_and_rng_pairing():
    case=dict(case_id="case")
    target=dict(decision=100,job_id="job",pool=[dict(agents=[0,1,5,6])],selected_index=0,edges=[[5,6]])
    pool=target["pool"]
    result={arm:paired_action(case,target,2,0,pool,0,arm,20) for arm in
            ("frozen","add_certified","replace_padding")}
    assert result["frozen"][0]["agents"]==[0,1,5,6]
    assert result["add_certified"][0]["agents"]==[0,1,5,6,20]
    assert result["replace_padding"][0]["agents"]==[1,5,6,20]
    assert len({(a["random_seed"],u) for a,u in result.values()})==1


def test_subsequent_actions_do_not_keep_forcing_membership():
    case=dict(case_id="case")
    target=dict(decision=100,job_id="job")
    pool=[dict(agents=[7,8])]
    left=paired_action(case,target,1,1,pool,0,"frozen",20)
    right=paired_action(case,target,1,1,pool,0,"add_certified",20)
    assert left==right and left[0]["agents"]==[7,8]
