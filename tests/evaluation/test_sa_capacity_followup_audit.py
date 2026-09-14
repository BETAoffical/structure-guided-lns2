from copy import deepcopy
import pytest

from scripts.audit_sa_capacity_followup import validate_scalars, smallest_deficiency, io


def test_distinguish_genuine_three_agent_capacity_from_embedded_pair_deficit():
    assert len(smallest_deficiency([6, 82, 213], [{1, 2}]*3)["agents"]) == 3
    assert len(smallest_deficiency([6, 82, 213], [{1}, {1}, {1, 2}])["agents"]) == 2
    assert smallest_deficiency([6, 82], [{1}, {1, 2}]) is None


def fixture():
    case = dict(id="case", state=dict(num_of_colliding_pairs=3))
    row = dict(case_id="case", trial=0, before=3, after=2, generated=10, censored=False,
               final_state=dict(num_of_colliding_pairs=2, low_level=dict(generated=10)),
               metrics=dict(pp_failure_reason="none", requested_pp_time_limit_seconds=5.,
                            requested_random_seed=int(io.semantic_fingerprint(["case", 20260915, 0])[:7], 16)))
    return row, case


def test_valid_saved_trial_scalars():
    validate_scalars(*fixture())


@pytest.mark.parametrize("field,value", [("case_id", "wrong"), ("before", 4), ("after", 0),
                                         ("generated", 11), ("censored", True)])
def test_reject_changed_saved_trial_scalars(field, value):
    row, case = fixture()
    row[field] = value
    with pytest.raises(ValueError):
        validate_scalars(row, case)


@pytest.mark.parametrize("field,value", [("requested_random_seed", -1), ("requested_pp_time_limit_seconds", 6.)])
def test_reject_unpaired_seed_or_budget(field, value):
    row, case = fixture()
    changed = deepcopy(row)
    changed["metrics"][field] = value
    with pytest.raises(ValueError):
        validate_scalars(changed, case)
