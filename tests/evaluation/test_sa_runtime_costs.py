import pytest

from scripts import audit_sa_runtime_costs as audit


def event():
    return dict(selection_seconds=2., metrics=dict(native_replan_seconds=3., native_step_seconds=4.,
        binding_solver_call_seconds=5., binding_total_seconds=6., native_neighborhood_generation_seconds=.1,
        native_repair_bookkeeping_seconds=.2, native_state_snapshot_seconds=.3, native_residual_seconds=.4,
        binding_state_snapshot_seconds=.2, state_to_python_seconds=.5, metrics_to_python_seconds=.1,
        binding_residual_seconds=.2))


def test_disjoint_costs_do_not_add_nested_timers_twice():
    parts = audit.event_parts(event(), 10.)
    assert parts == dict(selection=2., pp=3., native_other=1., solver_wrapper=1., binding_export=1., python_loop_other=2.)
    assert sum(parts.values()) == 10.


def test_malformed_timers_fail_closed():
    for x in (True, -1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            audit.number(x)
    with pytest.raises(ValueError, match="exceed"):
        audit.event_parts(event(), 7.)
    e = event()
    e["metrics"]["native_state_snapshot_seconds"] = 1.
    with pytest.raises(ValueError, match="reconstruct"):
        audit.event_parts(e, 10.)


def test_stage_boundaries_and_idealized_bounds():
    assert [audit.stage(x) for x in (1,10,11,100,101)] == ["low_1_10"]*2 + ["medium_11_100"]*2 + ["high_over_100"]
    with pytest.raises(ValueError):
        audit.stage(0)
    row = dict(ttf=10., steps=1, parts=dict(reset=2., selection=2., pp=6.), stages={}, diagnostics={})
    summary = audit.aggregate([row])
    assert summary["counterfactual_cost_bounds"]["pp"]["halving_cost_ttf_reduction_percent"] == 30.
    assert sum(summary["parts_percent"].values()) == 100.


def test_ttf_excludes_final_logging_and_audit_but_not_previous_loop_work():
    e = event()
    e["elapsed_seconds"] = 12.
    e["metrics"].update(action_valid=True, step_applied=True, conflicts_before=3, conflicts_after=0,
                        pp_attempted_agent_count=2, neighborhood=[2,7], requested_collect_pp_diagnostics=False,
                        pp_agent_diagnostics=[], pp_failure_reason="none")
    row = dict(status="ok", arm="dual16_sa", phase="timed", runtime_variant="single_full_check",
               plan_sha256=audit.REGISTRATION_SHA, success_within_budget=True,
               initial_state=dict(num_of_colliding_pairs=3), final_state=dict(feasible=True),
               reset_seconds=2., ttf_seconds=12., loop_end_seconds=14., events=[e],
               pp_seconds=3., selection_seconds=2., job_id="j", case_id="c", map_id="m", repeat=0)
    row["integrity_sha256"] = audit.digest(row)
    result = audit.summarize_episode(row)
    assert sum(result["parts"].values()) == 12.
    assert result["parts"]["python_loop_other"] == 2.
    assert result["post_ttf_loop_seconds"] == 2.
    row["ttf_seconds"] = 14.
    row["integrity_sha256"] = audit.digest({k:v for k,v in row.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="first feasible"):
        audit.summarize_episode(row)
