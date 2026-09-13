import math
import pytest
from scripts import probe_sa_bounded_reheat as probe


def event(delta):
    return dict(metrics=dict(pp_attempt_conflict_pair_count=delta+1,pp_old_conflict_pair_count=1,acceptance_evaluated=True))


def test_scale_uses_only_preceding_32_attempts():
    rows=[event(1000)]+[event(2) for _ in range(32)]+[event(999999)]
    assert probe.prefix_scale(rows,33)==2
    assert probe.prefix_scale([event(0),event(-1)],2)==1
    assert probe.prefix_scale([event(1),event(3)],2)==2


def test_pulse_has_fixed_end_and_original_temperature_floor():
    assert probe.selected_temperature("pulse16",1000,0,2)==pytest.approx(2/math.log(2))
    assert probe.selected_temperature("pulse16",1015,15,2)>probe.q.temperature(1015)
    assert probe.selected_temperature("pulse16",1016,16,2)==probe.q.temperature(1016)
    assert probe.selected_temperature("pulse16",0,0,1)==probe.q.temperature(0)
    assert probe.selected_temperature("frozen",1000,0,2)==probe.q.temperature(1000)


def test_typical_increase_initial_probability_is_half_when_exposed():
    temp=probe.selected_temperature("pulse16",1000,0,3)
    assert math.exp(-3/temp)==pytest.approx(.5)
    assert math.exp(-6/temp)==pytest.approx(.25)


def test_paired_rng_does_not_depend_on_arm_and_trial_zero_reproduces():
    case=dict(case_id="case")
    pool=[dict(agents=[3,8])]
    action,u=probe.random_action(case,700,0,pool,0)
    assert action==probe.q.action_for("dual16_sa",case,700,pool,0)
    assert u==probe.q.acceptance_draw(probe.q.seed("case",0,700,"accept"))
    seeds={probe.random_action(case,700,t,pool,0)[0]["random_seed"] for t in range(4)}
    assert len(seeds)==4


def test_target_selection_is_fixed_and_contains_success_controls():
    def row(k,success,rejections=0):
        return dict(job_id=k,cohort="failure_union_300",success=success,
                    bins=dict(all=dict(worse_rejected=rejections,steps=300)),
                    attempts=[dict(probability=.001,accepted=False,best_gap_before=50,decision=600+j) for j in range(rejections)])
    rows=[row("a",False,3),row("b",False,2),row("c",False,1),row("d",False,0)]
    rows += [row(str(i),True) for i in range(5)]
    targets=probe.choose_targets(rows)
    assert [t["role"] for t in targets]==["primary"]*3+["negative"]+["success_control"]*5
    assert targets[0]["decision"]==600
    assert targets[-1]["decision"]==236
    assert probe.choose_targets(list(reversed(rows)))==targets


def test_pair_distinguishes_censoring_and_no_exposure():
    base=dict(root_fingerprint="root",trial=0,feasible=True,full_window=True,counts=[3,0],generated=10,
              events=[{}],temperature_exposed_steps=0,final_fingerprint="end")
    pulse=dict(base,feasible=False,full_window=False,counts=[3,2])
    result=probe.pair_summary(base,pulse)
    assert result["loss"] and not result["paired_full_window"]
    assert result["temperature_exposed_steps"]==0
    with pytest.raises(ValueError): probe.pair_summary(base,dict(pulse,trial=1))
