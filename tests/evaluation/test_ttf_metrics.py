from __future__ import annotations

import copy
import math

import pytest

from lns2_selector.evaluation.ttf_metrics import validate_ttf_summary


@pytest.fixture
def successful_summary() -> dict:
    return {
        "success": True,
        "external_timeout": False,
        "truncated": False,
        "stop_reason": "success",
        "final_conflicts": 0,
        "repair_iterations": 1,
        "transition_elapsed_seconds": [0.6],
        "initial_state_elapsed_seconds": 0.1,
        "ttf_clock_schema": "lns2.ttf.reset_inclusive_wall.v1",
        "wall_time_budget_seconds": 60.0,
        "wall_time_to_feasible": 0.6,
        "capped_wall_time_to_feasible": 0.6,
        "ttf_observed_wall_seconds": 0.7,
        "episode_observed_wall_seconds": 0.8,
        "reset_wall_seconds": 0.1,
        "repair_wall_seconds": 0.4,
    }


def _failed_summary(summary: dict, *, timeout: bool = False) -> dict:
    return {
        **summary,
        "success": False,
        "external_timeout": timeout,
        "truncated": True,
        "stop_reason": "wall_timeout" if timeout else "native_terminal",
        "final_conflicts": 2,
        "wall_time_to_feasible": None,
        "capped_wall_time_to_feasible": 60.0,
        "ttf_observed_wall_seconds": 60.1 if timeout else 0.7,
        "episode_observed_wall_seconds": 60.2 if timeout else 0.8,
        "transition_elapsed_seconds": [60.05] if timeout else [0.6],
    }


@pytest.mark.parametrize("elapsed", [60.0, 60.1])
def test_failed_deadline_completion_requires_timeout_flag(successful_summary, elapsed):
    summary = _failed_summary(successful_summary)
    summary.update({
        "transition_elapsed_seconds": [elapsed],
        "ttf_observed_wall_seconds": 61.0,
        "episode_observed_wall_seconds": 61.0,
    })
    with pytest.raises(ValueError, match="requires external_timeout"):
        validate_ttf_summary(summary)


def test_valid_success_returns_original_dict_without_mutation(successful_summary):
    successful_summary["native_time_to_feasible"] = 0.5
    successful_summary["initial_state_elapsed_seconds"] = 0.100001
    successful_summary["transition_elapsed_seconds"] = [0.3, 0.6]
    successful_summary["repair_iterations"] = 2
    successful_summary["transition_trace_write_seconds"] = [0.001, 0.002]
    successful_summary["reset_timings"] = {"reset_total_seconds": 0.1}
    successful_summary["controller_totals"] = {"inference_seconds": 0.01}
    before = copy.deepcopy(successful_summary)
    assert validate_ttf_summary(successful_summary, wall_time_budget_seconds=60) is successful_summary
    assert successful_summary == before


@pytest.mark.parametrize("reason", ["native_terminal", "controller_stalled", "repair_limit", "truncated"])
def test_failure_before_deadline_is_capped_at_full_budget(successful_summary, reason):
    summary = _failed_summary(successful_summary)
    summary["stop_reason"] = reason
    assert validate_ttf_summary(summary) is summary


@pytest.mark.parametrize("reason", ["wall_timeout", "repair_limit"])
def test_zero_conflicts_after_deadline_remain_a_failed_result(successful_summary, reason):
    summary = _failed_summary(successful_summary, timeout=True)
    summary.update(final_conflicts=0, stop_reason=reason, repair_wall_seconds=59.95)
    assert validate_ttf_summary(summary) is summary


def test_initially_feasible_state_needs_no_repair(successful_summary):
    successful_summary.update(
        wall_time_to_feasible=0.1,
        capped_wall_time_to_feasible=0.1,
        repair_wall_seconds=0.0,
        repair_iterations=0,
        transition_elapsed_seconds=[],
    )
    assert validate_ttf_summary(successful_summary) is successful_summary


@pytest.mark.parametrize("summary", [None, [], "{}", 1, True])
def test_summary_must_be_an_object(summary):
    with pytest.raises(ValueError, match="object"):
        validate_ttf_summary(summary)


@pytest.mark.parametrize(
    "field",
    [
        "success", "external_timeout", "truncated", "stop_reason", "final_conflicts",
        "repair_iterations", "ttf_clock_schema",
        "transition_elapsed_seconds", "initial_state_elapsed_seconds",
        "wall_time_budget_seconds", "wall_time_to_feasible", "capped_wall_time_to_feasible",
        "ttf_observed_wall_seconds", "episode_observed_wall_seconds",
        "reset_wall_seconds", "repair_wall_seconds",
    ],
)
def test_required_metric_cannot_be_silently_defaulted(successful_summary, field):
    del successful_summary[field]
    with pytest.raises(ValueError, match="missing"):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize("field", ["success", "external_timeout", "truncated"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_flags_are_strict_booleans(successful_summary, field, value):
    successful_summary[field] = value
    with pytest.raises(ValueError, match="boolean"):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize(
    "field",
    [
        "wall_time_budget_seconds", "wall_time_to_feasible", "capped_wall_time_to_feasible",
        "ttf_observed_wall_seconds", "episode_observed_wall_seconds",
        "reset_wall_seconds", "repair_wall_seconds", "native_time_to_feasible",
        "initial_state_elapsed_seconds",
    ],
)
@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf, -math.inf, True, "0.1", None])
def test_times_reject_nonfinite_negative_and_coerced_values(successful_summary, field, value):
    successful_summary[field] = value
    with pytest.raises(ValueError):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize("budget", [0, -1, math.nan, math.inf, True, "60", 120])
def test_registered_budget_must_be_positive_finite_and_match(successful_summary, budget):
    with pytest.raises(ValueError):
        validate_ttf_summary(successful_summary, wall_time_budget_seconds=budget)


@pytest.mark.parametrize("value", [-1, 0.0, True, "0", None])
def test_final_conflicts_is_a_nonnegative_integer(successful_summary, value):
    successful_summary["final_conflicts"] = value
    with pytest.raises(ValueError, match="final_conflicts"):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize(
    "field",
    ["repair_iterations", "initial_conflicts", "budget_final_conflicts", "invalid_action_count", "repair_iterations_within_budget"],
)
@pytest.mark.parametrize("value", [-1, 1.0, True, "1", math.nan])
def test_summary_counts_are_nonnegative_integers(successful_summary, field, value):
    successful_summary[field] = value
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize("schema", [None, True, 1, "", "lns2.ttf.other.v1"])
def test_ttf_clock_schema_is_fixed(successful_summary, schema):
    successful_summary["ttf_clock_schema"] = schema
    with pytest.raises(ValueError, match="ttf_clock_schema"):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize(
    "changes",
    [
        {"wall_time_budget_seconds": 0},
        {"final_conflicts": 1},
        {"external_timeout": True},
        {"truncated": True},
        {"stop_reason": "native_terminal"},
        {"stop_reason": "unknown"},
        {"wall_time_to_feasible": 60.1, "capped_wall_time_to_feasible": 60.0},
        {"capped_wall_time_to_feasible": 61.0},
        {"capped_wall_time_to_feasible": 0.5},
        {"ttf_observed_wall_seconds": 0.55},
        {"episode_observed_wall_seconds": 0.65},
        {"reset_wall_seconds": 0.35},
        {"repair_wall_seconds": 0.65},
        {"initial_state_elapsed_seconds": 0.05},
        {"initial_state_elapsed_seconds": 0.65},
        {"environment_construct_seconds": 1.0},
    ],
)
def test_success_clock_and_outcome_contradictions_are_rejected(successful_summary, changes):
    successful_summary.update(changes)
    with pytest.raises(ValueError):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize(
    "changes",
    [
        {"wall_time_to_feasible": 0.6},
        {"capped_wall_time_to_feasible": 0.6},
        {"final_conflicts": 0},
        {"truncated": False},
        {"stop_reason": "success"},
        {"stop_reason": "wall_timeout"},
        {"native_time_to_feasible": 0.5},
        {"external_timeout": True},
    ],
)
def test_failure_cannot_claim_success_metrics(successful_summary, changes):
    summary = _failed_summary(successful_summary)
    summary.update(changes)
    with pytest.raises(ValueError):
        validate_ttf_summary(summary)


def test_timeout_cannot_precede_wall_deadline(successful_summary):
    summary = _failed_summary(successful_summary, timeout=True)
    summary["ttf_observed_wall_seconds"] = 59.0
    summary["transition_elapsed_seconds"] = [58.0]
    with pytest.raises(ValueError, match="deadline"):
        validate_ttf_summary(summary)


@pytest.mark.parametrize("field", ["reset_timings", "controller_totals"])
@pytest.mark.parametrize("value", [None, [], {"partial_seconds": math.nan}, {"partial_seconds": True}])
def test_optional_timing_partitions_are_validated(successful_summary, field, value):
    successful_summary[field] = value
    with pytest.raises(ValueError):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize("field", ["transition_elapsed_seconds", "transition_trace_write_seconds"])
@pytest.mark.parametrize("value", [None, {}, [math.nan], [True], [-0.1]])
def test_optional_time_series_are_validated(successful_summary, field, value):
    successful_summary[field] = value
    with pytest.raises(ValueError):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize("times", [[0.4, 0.3], [0.8]])
def test_transition_times_are_ordered_and_within_observation(successful_summary, times):
    successful_summary["transition_elapsed_seconds"] = times
    with pytest.raises(ValueError):
        validate_ttf_summary(successful_summary)


def test_unrepresentably_large_integer_time_is_a_validation_error(successful_summary):
    successful_summary["repair_wall_seconds"] = 10 ** 1000
    with pytest.raises(ValueError, match="finite"):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize("zero_step", [False, True])
def test_changing_ttf_and_cap_together_cannot_hide_algorithm_elapsed_time(successful_summary, zero_step):
    if zero_step:
        successful_summary.update(
            repair_iterations=0,
            transition_elapsed_seconds=[],
            initial_state_elapsed_seconds=0.6,
            repair_wall_seconds=0.0,
        )
    validate_ttf_summary(successful_summary)
    successful_summary.update(wall_time_to_feasible=0.55, capped_wall_time_to_feasible=0.55)
    with pytest.raises(ValueError, match="final algorithm elapsed"):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize("count", [0, 2])
def test_repair_count_must_match_transition_clock_count(successful_summary, count):
    successful_summary["repair_iterations"] = count
    with pytest.raises(ValueError, match="transition count"):
        validate_ttf_summary(successful_summary)


def test_transition_cannot_precede_initial_state_completion(successful_summary):
    successful_summary["transition_elapsed_seconds"] = [0.05]
    with pytest.raises(ValueError, match="follow reset"):
        validate_ttf_summary(successful_summary)


@pytest.mark.parametrize("algorithm_elapsed", [59.0, 60.0])
def test_late_trace_writing_does_not_turn_timely_zero_conflicts_into_timeout(successful_summary, algorithm_elapsed):
    summary = _failed_summary(successful_summary, timeout=True)
    summary.update(final_conflicts=0, transition_elapsed_seconds=[algorithm_elapsed])
    with pytest.raises(ValueError, match="finish after its deadline"):
        validate_ttf_summary(summary)
