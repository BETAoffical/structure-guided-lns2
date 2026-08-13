from pathlib import Path

import pytest

from experiments.stride_platformentry_order import (
    _episode_override,
    _kaplan_meier_restricted_mean,
    _paired_case_bootstrap,
    detect_persistent_platform,
    order_schedule,
    prepare_order_cases,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_platformentry_order_v1_registration.json"


def _decision(
    index: int,
    before: str,
    after: str,
    *,
    success: bool,
    conflicts_before: int = 10,
    conflicts_after: int = 10,
) -> dict:
    return {
        "decision_index": index,
        "before_repair_fingerprint": before,
        "after_repair_fingerprint": after,
        "repair_state_changed": before != after,
        "before_conflicts": conflicts_before,
        "actual_metrics": {"replan_success": success},
        "actual_lns2": {"outcome": {"conflicts_after": conflicts_after}},
    }


def test_registered_same_set_order_cohort_and_schedule_are_exact() -> None:
    _loaded, cases = prepare_order_cases(CONFIG)
    assert len(cases) == 36
    assert {
        case["witness"]["escape_classification"] for case in cases
    } == {"same_set_order_escape"}
    schedule = order_schedule(cases, range(4))
    assert len(schedule) == 36 * 4 * 2
    grouped: dict[tuple[str, int], dict[str, dict]] = {}
    for row in schedule:
        grouped.setdefault((row["case_id"], row["trial_index"]), {})[
            row["arm"]
        ] = row
    assert len(grouped) == 36 * 4
    for arms in grouped.values():
        assert set(arms) == {"native_order", "conflict_priority_order"}
        assert arms["native_order"]["candidate_agents"] == arms[
            "conflict_priority_order"
        ]["candidate_agents"]
        assert arms["native_order"]["first_action_pp_seed"] == arms[
            "conflict_priority_order"
        ]["first_action_pp_seed"]


def test_forced_order_changes_only_repair_order() -> None:
    loaded, cases = prepare_order_cases(CONFIG)
    _path, root, _config, _inputs, _pretail_inputs, _parent = loaded
    native = _episode_override(root, cases[0], trial_index=0, arm="native_order")
    priority = _episode_override(
        root, cases[0], trial_index=0, arm="conflict_priority_order"
    )
    native_action = native["forced_first_action"]
    priority_action = priority["forced_first_action"]
    assert native_action["agents"] == priority_action["agents"]
    assert native_action["pp_random_seed"] == priority_action["pp_random_seed"]
    assert "repair_order" not in native_action
    assert sorted(priority_action["repair_order"]) == sorted(priority_action["agents"])
    assert native["initial_restore"] == priority["initial_restore"]


def test_platform_requires_three_same_exact_rollbacks() -> None:
    short = [
        _decision(0, "a", "a", success=False),
        _decision(1, "a", "a", success=False),
        _decision(2, "a", "b", success=True, conflicts_after=9),
    ]
    assert detect_persistent_platform(short)["entered_platform"] is False
    platform = [
        _decision(0, "a", "a", success=False),
        _decision(1, "a", "a", success=False),
        _decision(2, "a", "a", success=False),
    ]
    detected = detect_persistent_platform(platform)
    assert detected["entered_platform"] is True
    assert detected["first_entry_decision"] == 0
    assert detected["maximum_consecutive_exact_rollbacks"] == 3


def test_platform_streak_resets_when_fingerprint_changes() -> None:
    rows = [
        _decision(0, "a", "a", success=False),
        _decision(1, "b", "b", success=False),
        _decision(2, "b", "b", success=False),
        _decision(3, "b", "b", success=False),
    ]
    detected = detect_persistent_platform(rows)
    assert detected["entered_platform"] is True
    assert detected["first_entry_decision"] == 1


def test_kaplan_meier_restricted_mean_preserves_right_censoring() -> None:
    rows = [
        {"repair_iterations": 10, "success": True},
        {"repair_iterations": 20, "success": False},
    ]
    assert _kaplan_meier_restricted_mean(rows, horizon=64) == pytest.approx(37.0)
    assert _kaplan_meier_restricted_mean(
        [{"repair_iterations": 4, "success": False}], horizon=64
    ) == pytest.approx(64.0)


def test_bootstrap_is_paired_by_case() -> None:
    rows = []
    for case_id, native, order in (("a", True, False), ("b", True, False)):
        rows.extend(
            [
                {"case_id": case_id, "arm": "native_order", "entered_platform": native},
                {
                    "case_id": case_id,
                    "arm": "conflict_priority_order",
                    "entered_platform": order,
                },
            ]
        )
    result = _paired_case_bootstrap(rows, replicates=200)
    assert result == {"point": -1.0, "lower_95": -1.0, "upper_95": -1.0}
