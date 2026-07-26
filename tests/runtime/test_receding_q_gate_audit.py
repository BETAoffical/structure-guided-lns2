from __future__ import annotations

import pytest

from experiments import receding_q_gate_audit as module


def _row(
    state: str,
    candidate: str,
    trial: int,
    auc: float,
    *,
    final_ratio: float | None = None,
    feasible: bool = False,
    seconds: float = 0.1,
) -> dict:
    return {
        "state_id": state,
        "candidate_id": candidate,
        "trial_index": trial,
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 100,
        "feasible": feasible,
        "final_conflict_ratio": (
            float(auc) if final_ratio is None else float(final_ratio)
        ),
        "normalized_step_auc": float(auc),
        "normalized_wall_auc_seconds": float(seconds),
        "observed_total_seconds": float(seconds),
    }


def test_timing_only_winner_change_is_not_quality_instability() -> None:
    rows = [
        _row("state", "a", 0, 0.1, seconds=0.100),
        _row("state", "b", 0, 0.1, seconds=0.101),
        _row("state", "a", 1, 0.1, seconds=0.101),
        _row("state", "b", 1, 0.1, seconds=0.100),
    ]
    report, states, pairs = module.analyze_quality_stability(
        rows, expected_trials=2
    )
    assert report["exact_winner_agreement_fraction"] == 0.0
    assert report["timing_only_winner_mismatch_state_count"] == 1
    assert report["true_quality_instability_state_count"] == 0
    assert report["quality_winner_set_overlap_fraction"] == 1.0
    assert report["decision"] == "quality_stability_gate_passed"
    assert states[0]["timing_only_winner_mismatch"] is True
    assert pairs[0]["winner_outcome_equivalent"] is True


def test_true_quality_winner_change_is_targeted() -> None:
    stable = [
        _row("stable", "a", 0, 0.1),
        _row("stable", "b", 0, 0.2),
        _row("stable", "a", 1, 0.1),
        _row("stable", "b", 1, 0.2),
    ]
    unstable = [
        _row("unstable", "a", 0, 0.1),
        _row("unstable", "b", 0, 0.2),
        _row("unstable", "a", 1, 0.2),
        _row("unstable", "b", 1, 0.1),
    ]
    report, states, _ = module.analyze_quality_stability(
        stable + unstable, expected_trials=2
    )
    assert report["true_quality_instability_state_ids"] == ["unstable"]
    assert any(row["true_quality_instability"] for row in states)


def test_shared_quality_winner_is_not_true_instability() -> None:
    rows = [
        _row("state", "a", 0, 0.1, seconds=0.100),
        _row("state", "b", 0, 0.1, seconds=0.101),
        _row("state", "c", 0, 0.2, seconds=0.102),
        _row("state", "a", 1, 0.2, seconds=0.102),
        _row("state", "b", 1, 0.1, seconds=0.101),
        _row("state", "c", 1, 0.1, seconds=0.100),
    ]
    report, states, _ = module.analyze_quality_stability(
        rows, expected_trials=2
    )
    assert report["true_quality_instability_state_count"] == 0
    assert report["exact_winner_cross_seed_nonoptimal_state_count"] == 1
    assert states[0]["true_quality_instability"] is False
    assert states[0]["exact_winner_cross_seed_nonoptimal"] is True


def test_quality_winner_set_respects_lexicographic_quality() -> None:
    rows = [
        _row("state", "feasible", 0, 0.5, final_ratio=0.0, feasible=True),
        _row("state", "lower-auc", 0, 0.1, final_ratio=0.1),
    ]
    assert module.quality_winner_set(rows) == {"feasible"}


def test_timing_tolerance_expands_only_the_quality_best_set() -> None:
    rows = [
        _row("state", "a", 0, 0.1, seconds=0.100),
        _row("state", "b", 0, 0.1, seconds=0.104),
        _row("state", "bad", 0, 0.2, seconds=0.001),
    ]
    assert module.timing_tolerant_winner_set(
        rows, absolute_seconds=0.005, relative_fraction=0.0
    ) == {"a", "b"}


def test_analysis_rejects_incomplete_candidate_trial_matrix() -> None:
    rows = [
        _row("state", "a", 0, 0.1),
        _row("state", "b", 0, 0.2),
        _row("state", "a", 1, 0.1),
    ]
    with pytest.raises(ValueError, match="candidate coverage mismatch"):
        module.analyze_quality_stability(rows, expected_trials=2)
