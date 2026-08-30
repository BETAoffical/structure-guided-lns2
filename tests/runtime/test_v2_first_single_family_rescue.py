from __future__ import annotations

from unittest.mock import patch

import pytest

from lns2_selector.runtime.v2_first_single_family_rescue import (
    V2FirstSingleFamilyRescueTracker,
    generate_v2_first_single_family_rescue_candidate,
    v2_first_single_family_rescue_augmentation,
    validate_v2_first_single_family_rescue_augmentation,
)


def exact_rollback_metrics() -> dict[str, object]:
    return {
        "pp_failure_reason": "conflict_bound_exceeded",
        "replan_success": False,
        "pp_rolled_back": True,
    }


def time_limit_metrics() -> dict[str, object]:
    return {
        "pp_failure_reason": "time_limit",
        "replan_success": False,
        "pp_rolled_back": True,
    }


def test_contract_is_fixed_to_component_or_hotspot_size16() -> None:
    component = v2_first_single_family_rescue_augmentation(
        "conflict_component"
    )
    hotspot = v2_first_single_family_rescue_augmentation("hotspot")

    assert validate_v2_first_single_family_rescue_augmentation(component) == component
    assert validate_v2_first_single_family_rescue_augmentation(hotspot) == hotspot
    assert component["v2_first_rescue"][
        "minimum_consecutive_v2_exact_rollbacks"
    ] == 3
    assert component["v2_first_rescue"]["maximum_rescue_offers_per_episode"] == 1
    assert component["v2_first_rescue"]["selection"] == (
        "direct_unique_structural_candidate"
    )
    assert component["v2_first_rescue"]["permanent_v2_after_offer"] is True
    with pytest.raises(ValueError):
        v2_first_single_family_rescue_augmentation("path_overlap")
    with pytest.raises(ValueError):
        v2_first_single_family_rescue_augmentation("hotspot", 24)


def test_three_exact_v2_rollbacks_offer_rescue_on_next_decision() -> None:
    tracker = V2FirstSingleFamilyRescueTracker.from_spec(
        v2_first_single_family_rescue_augmentation("conflict_component")
    )

    for expected in (1, 2, 3):
        assert not tracker.selection("state-a")["rescue_due"]
        observation = tracker.observe_v2(
            before_repair_fingerprint="state-a",
            after_repair_fingerprint="state-a",
            metrics=exact_rollback_metrics(),
        )
        assert observation["consecutive_v2_exact_rollbacks"] == expected

    selection = tracker.selection("state-a")
    assert selection["rescue_due"]
    assert selection["offered"]
    assert selection["selection_phase"] == "single_family_rescue_due"


def test_non_exact_or_changed_fingerprint_resets_streak_without_restore() -> None:
    tracker = V2FirstSingleFamilyRescueTracker("hotspot")
    for _ in range(2):
        tracker.observe_v2(
            before_repair_fingerprint="state-a",
            after_repair_fingerprint="state-a",
            metrics=exact_rollback_metrics(),
        )
    timeout = tracker.observe_v2(
        before_repair_fingerprint="state-a",
        after_repair_fingerprint="state-a",
        metrics=time_limit_metrics(),
    )
    assert not timeout["v2_exact_conflict_bound_rollback"]
    assert timeout["consecutive_v2_exact_rollbacks"] == 0

    tracker.observe_v2(
        before_repair_fingerprint="state-a",
        after_repair_fingerprint="state-b",
        metrics=exact_rollback_metrics(),
    )
    tracker.observe_v2(
        before_repair_fingerprint="state-b",
        after_repair_fingerprint="state-b",
        metrics=exact_rollback_metrics(),
    )
    tracker.observe_v2(
        before_repair_fingerprint="state-b",
        after_repair_fingerprint="state-a",
        metrics={
            "pp_failure_reason": "none",
            "replan_success": True,
            "pp_rolled_back": False,
        },
    )
    assert tracker.selection("state-a")["consecutive_v2_exact_rollbacks"] == 0
    assert not tracker.selection("state-a")["rescue_due"]


def test_unavailable_offer_consumes_episode_and_latches_permanent_v2() -> None:
    tracker = V2FirstSingleFamilyRescueTracker("conflict_component")
    for _ in range(3):
        tracker.observe_v2(
            before_repair_fingerprint="state-a",
            after_repair_fingerprint="state-a",
            metrics=exact_rollback_metrics(),
        )

    generation = tracker.record_generation(
        repair_fingerprint="state-a", candidate_id=None
    )
    assert generation == {
        "attempted": True,
        "available": False,
        "offered": True,
        "challenger_present": False,
        "structural_selected": False,
        "consumed": True,
        "repair_fingerprint": "state-a",
        "candidate_id": None,
        "fallback": "fresh_v2_only",
        "episode_rescue_consumed": True,
    }
    assert tracker.selection("state-a")["selection_phase"] == (
        "v2_only_after_offer"
    )
    assert not tracker.selection("state-b")["rescue_due"]


def test_present_challenger_is_directly_marked_executed_once() -> None:
    tracker = V2FirstSingleFamilyRescueTracker("hotspot")
    for _ in range(3):
        tracker.observe_v2(
            before_repair_fingerprint="state-a",
            after_repair_fingerprint="state-a",
            metrics=exact_rollback_metrics(),
        )
    offered = tracker.record_generation(
        repair_fingerprint="state-a", candidate_id="candidate-a"
    )
    assert offered["challenger_present"]
    executed = tracker.mark_executed(
        repair_fingerprint="state-a", candidate_id="candidate-a"
    )
    assert executed["structural_selected"]
    assert executed["consumed"]
    with pytest.raises(ValueError):
        tracker.mark_executed(
            repair_fingerprint="state-a", candidate_id="candidate-a"
        )


def test_direct_generator_returns_only_the_isolated_family_candidate() -> None:
    candidate = {
        "candidate_id": "component-16",
        "agents": list(range(16)),
        "actual_size": 16,
        "selection_families": ["conflict_component:16"],
        "proposal_seeds": [],
    }
    with patch(
        "lns2_selector.runtime.v2_first_single_family_rescue."
        "generate_structpool_candidate_subset",
        return_value=[candidate],
    ) as generate:
        result, seconds = generate_v2_first_single_family_rescue_candidate(
            {"agents": []},
            object(),
            config=v2_first_single_family_rescue_augmentation(
                "conflict_component"
            ),
        )

    assert result is not None
    assert result["candidate_id"] == "component-16"
    assert result["hybridstructpool_provenance"] == [
        "structshell_equal_four_size",
        "v2_first_single_family_rescue",
    ]
    assert seconds >= 0.0
    assert generate.call_args.kwargs["family_sizes"] == {
        "conflict_component": (16,)
    }
