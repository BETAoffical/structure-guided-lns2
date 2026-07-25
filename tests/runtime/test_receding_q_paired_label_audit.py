from __future__ import annotations

from experiments import receding_q_paired_label_audit as module


def _row(candidate: str, trial: int, value: float) -> dict:
    return {
        "candidate_id": candidate,
        "trial_index": trial,
        "feasible": True,
        "final_conflict_ratio": value,
        "normalized_step_auc": value,
        "normalized_wall_auc_seconds": value,
        "observed_total_seconds": value,
    }


def _policy(method: str, risk_lambda: float = 0.0) -> dict:
    return {
        "policy_id": method,
        "method": method,
        "risk_lambda": risk_lambda,
        "complexity": 0,
    }


def test_average_rank_uses_each_seed_as_a_paired_block() -> None:
    rows = [
        _row("a", 0, 0.0),
        _row("a", 1, 0.0),
        _row("a", 2, 100.0),
        _row("b", 0, 10.0),
        _row("b", 1, 10.0),
        _row("b", 2, 11.0),
    ]
    assert (
        module.select_paired_label_candidate(
            rows, policy=_policy("seed_average_rank")
        )
        == "a"
    )


def test_copeland_prefers_candidate_with_majority_seed_wins() -> None:
    rows = [
        _row("a", 0, 0.0),
        _row("a", 1, 0.0),
        _row("a", 2, 10.0),
        _row("b", 0, 1.0),
        _row("b", 1, 1.0),
        _row("b", 2, 0.0),
    ]
    assert (
        module.select_paired_label_candidate(
            rows, policy=_policy("seed_pairwise_copeland")
        )
        == "a"
    )


def test_paired_centering_removes_common_seed_shift() -> None:
    rows = [
        _row("a", 0, 0.0),
        _row("a", 1, 100.0),
        _row("a", 2, 200.0),
        _row("b", 0, 1.0),
        _row("b", 1, 101.0),
        _row("b", 2, 201.0),
    ]
    assert (
        module.select_paired_label_candidate(
            rows,
            policy=_policy("paired_quality_se", risk_lambda=2.0),
        )
        == "a"
    )
