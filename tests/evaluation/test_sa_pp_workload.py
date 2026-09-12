from copy import deepcopy
import pytest
from scripts import audit_sa_pp_workload as audit


def row():
    return dict(success_within_budget=True,
                initial_state=dict(rows=2, cols=3, obstacles=[0]*6,
                                   low_level=dict(runs=1, expanded=10, generated=20)),
                final_state=dict(feasible=True, low_level=dict(runs=3, expanded=30, generated=50)),
                events=[dict(metrics=dict(pp_attempted_agent_count=2, pp_inserted_agent_count=2,
                                          requested_collect_pp_diagnostics=False))])


def test_workload_subtracts_reset_and_counts_slots_not_allocations():
    r = row()
    before = deepcopy(r)
    assert audit.workload(r) == dict(runs=2, expanded=20, generated=30, pp_attempted=2,
                                    pp_inserted=2, sit_slot_initializations_from_source=12)
    assert r == before


@pytest.mark.parametrize("value", [True, -1, 1.5, float("nan")])
def test_invalid_counter_rejected(value):
    with pytest.raises(ValueError):
        audit.count(value)


@pytest.mark.parametrize("field", ["failure", "count_mismatch", "diagnostics", "grid", "decrease"])
def test_incompatible_record_rejected(field):
    r = row()
    if field == "failure": r["success_within_budget"] = False
    if field == "count_mismatch": r["events"][0]["metrics"]["pp_attempted_agent_count"] = 1
    if field == "diagnostics": r["events"][0]["metrics"]["requested_collect_pp_diagnostics"] = True
    if field == "grid": r["initial_state"]["obstacles"] = []
    if field == "decrease": r["final_state"]["low_level"]["expanded"] = 0
    with pytest.raises(ValueError):
        audit.workload(r)
