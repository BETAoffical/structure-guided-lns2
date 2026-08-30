from __future__ import annotations

from unittest.mock import patch

import pytest

from lns2_selector.runtime.v2_first_consensus_rescue import (
    V2FirstConsensusRescueTracker,
    generate_v2_first_consensus_rescue_candidate,
    v2_first_consensus_rescue_augmentation,
    validate_v2_first_consensus_rescue_augmentation,
)


def exact_rollback_metrics() -> dict[str, object]:
    return {
        "pp_failure_reason": "conflict_bound_exceeded",
        "replan_success": False,
        "pp_rolled_back": True,
    }


def candidate(
    candidate_id: str, agents: list[int], families: list[str]
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "agents": agents,
        "actual_size": len(agents),
        "selection_families": families,
        "proposal_seeds": [],
    }


def test_contract_is_fixed_to_consensus16_after_three_rollbacks() -> None:
    config = v2_first_consensus_rescue_augmentation()

    assert validate_v2_first_consensus_rescue_augmentation(config) == config
    assert config["runtime_structural_family_sizes"] == {
        "conflict_component": [16],
        "spatiotemporal_hotspot": [16],
    }
    assert config["v2_first_rescue"]["selection"] == (
        "direct_only_on_exact_nonempty_agent_consensus"
    )
    assert config["v2_first_rescue"]["maximum_pp_calls_per_decision"] == 1
    assert config["v2_first_rescue"]["permanent_v2_after_offer"] is True
    with pytest.raises(ValueError):
        v2_first_consensus_rescue_augmentation(nominal_size=24)
    with pytest.raises(ValueError):
        v2_first_consensus_rescue_augmentation(
            minimum_consecutive_v2_exact_rollbacks=2
        )


def test_tracker_reuses_three_rollback_trigger_and_consumes_mismatch() -> None:
    tracker = V2FirstConsensusRescueTracker.from_spec(
        v2_first_consensus_rescue_augmentation()
    )
    for _ in range(3):
        tracker.observe_v2(
            before_repair_fingerprint="state-a",
            after_repair_fingerprint="state-a",
            metrics=exact_rollback_metrics(),
        )

    selection = tracker.selection("state-a")
    assert selection["selection_phase"] == "consensus_rescue_due"
    assert selection["consecutive_v2_exact_rollbacks"] == 3
    generation = tracker.record_generation(
        repair_fingerprint="state-a",
        candidate_id=None,
        consensus_audit={
            "component_candidate_id": "component",
            "hotspot_candidate_id": "hotspot",
            "component_agents": list(range(16)),
            "hotspot_agents": list(range(1, 17)),
            "exact_agent_consensus": False,
        },
    )
    assert generation["fallback"] == "fresh_v2_after_no_consensus"
    assert generation["consumed"]
    assert not generation["challenger_present"]
    assert tracker.selection("state-a")["selection_phase"] == (
        "v2_only_after_offer"
    )
    assert not tracker.selection("state-b")["rescue_due"]


def test_generator_returns_merged_candidate_only_for_exact_nonempty_match() -> None:
    merged = candidate(
        "shared",
        list(range(16)),
        [
            "structpool-conflict-component:16",
            "structpool-spatiotemporal-hotspot:16",
        ],
    )
    with patch(
        "lns2_selector.runtime.v2_first_consensus_rescue."
        "generate_structpool_candidate_subset",
        return_value=[merged],
    ) as generate:
        result, seconds, audit = generate_v2_first_consensus_rescue_candidate(
            {"agents": []},
            object(),
            config=v2_first_consensus_rescue_augmentation(),
        )

    assert result is not None
    assert result["candidate_id"] == "shared"
    assert result["hybridstructpool_provenance"] == [
        "structshell_equal_four_size",
        "v2_first_consensus_rescue",
    ]
    assert audit["component_candidate_id"] == "shared"
    assert audit["hotspot_candidate_id"] == "shared"
    assert audit["component_agents"] == list(range(16))
    assert audit["hotspot_agents"] == list(range(16))
    assert audit["exact_agent_consensus"] is True
    assert seconds >= 0.0
    assert generate.call_args.kwargs["family_sizes"] == {
        "conflict_component": (16,),
        "spatiotemporal_hotspot": (16,),
    }


def test_generator_rejects_distinct_agent_sets_without_choosing_a_family() -> None:
    component = candidate(
        "component",
        list(range(16)),
        ["structpool-conflict-component:16"],
    )
    hotspot = candidate(
        "hotspot",
        list(range(1, 17)),
        ["structpool-spatiotemporal-hotspot:16"],
    )
    with patch(
        "lns2_selector.runtime.v2_first_consensus_rescue."
        "generate_structpool_candidate_subset",
        return_value=[component, hotspot],
    ):
        result, _seconds, audit = generate_v2_first_consensus_rescue_candidate(
            {"agents": []},
            object(),
            config=v2_first_consensus_rescue_augmentation(),
        )

    assert result is None
    assert audit["component_candidate_id"] == "component"
    assert audit["hotspot_candidate_id"] == "hotspot"
    assert audit["exact_agent_consensus"] is False
    assert audit["generated_candidate_count"] == 2
