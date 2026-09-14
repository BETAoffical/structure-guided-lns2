from scripts.probe_sa_capacity_native import select_candidates, local_gate


def row(name, members, status):
    return dict(id=name, candidate=dict(family="directed_single", members=members),
                original_certificate_cleared=True, augmented=dict(status=status))


def test_raw_and_filtered_duplicate_is_one_action_with_two_roles():
    cases, missing = select_candidates(dict(selected=[0,1]), [row("a", [2], "not_proved"), row("b", [3], "proved")])
    assert len(cases) == 2
    assert cases[0]["roles"] == ["raw_single", "pass_single"]
    assert cases[1]["roles"] == ["fail_single"]
    assert "pass_pair" in missing


def test_first_input_rank_is_not_outcome_selected():
    a, _ = select_candidates(dict(selected=[0,1]), [row("a", [9], "not_proved"), row("b", [3], "not_proved")])
    assert a[0]["added"] == [9]


def groups(cases=2, unsafe=5, censored=False):
    return {(str(case), role): [dict(trial=t, after=value, censored=censored) for t in range(4)]
            for case in range(cases) for role, value in (("baseline",5), ("pass_single",4), ("fail_single",unsafe), ("random_single",5))}


def test_gate_requires_two_states_and_both_controls_not_just_baseline():
    assert local_gate(groups())["permit_continuation_diagnostic"]
    assert not local_gate(groups(1))["permit_continuation_diagnostic"]
    assert not local_gate(groups(unsafe=4))["permit_continuation_diagnostic"]
    assert not local_gate(groups(censored=True))["permit_continuation_diagnostic"]
