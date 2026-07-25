from __future__ import annotations

import pytest

from experiments import receding_q_variance_audit as module


def _matrix_rows(values: list[list[float]]) -> list[dict]:
    return [
        {
            "candidate_id": f"candidate-{candidate}",
            "trial_index": trial,
            "value": value,
        }
        for candidate, row in enumerate(values)
        for trial, value in enumerate(row)
    ]


def _decompose(values: list[list[float]]) -> dict:
    return module.decompose_balanced_candidate_seed(
        _matrix_rows(values), value=lambda row: float(row["value"])
    )


def test_decomposition_finds_pure_candidate_effect() -> None:
    result = _decompose([[1.0, 1.0], [3.0, 3.0]])
    assert result["candidate_fraction"] == pytest.approx(1.0)
    assert result["seed_fraction"] == pytest.approx(0.0)
    assert result["interaction_fraction"] == pytest.approx(0.0)


def test_decomposition_finds_pure_seed_effect() -> None:
    result = _decompose([[1.0, 3.0], [1.0, 3.0]])
    assert result["candidate_fraction"] == pytest.approx(0.0)
    assert result["seed_fraction"] == pytest.approx(1.0)
    assert result["interaction_fraction"] == pytest.approx(0.0)


def test_decomposition_finds_candidate_seed_interaction() -> None:
    result = _decompose([[0.0, 2.0], [2.0, 0.0]])
    assert result["candidate_fraction"] == pytest.approx(0.0)
    assert result["seed_fraction"] == pytest.approx(0.0)
    assert result["interaction_fraction"] == pytest.approx(1.0)
    assert result["closure_error"] == pytest.approx(0.0)


def test_decomposition_rejects_incomplete_matrix() -> None:
    rows = _matrix_rows([[0.0, 1.0], [2.0, 3.0]])[:-1]
    with pytest.raises(ValueError, match="balanced"):
        module.decompose_balanced_candidate_seed(
            rows, value=lambda row: float(row["value"])
        )


def test_classification_prefers_candidate_conditioned_order_probe() -> None:
    rows = []
    for metric, candidate, seed, interaction in (
        ("root_conflict_reduction", 0.25, 0.10, 0.65),
        ("root_state_changed", 0.30, 0.10, 0.60),
        ("h3_normalized_step_auc", 0.20, 0.10, 0.70),
    ):
        rows.append(
            {
                "metric": metric,
                "agent_group": "600",
                "candidate_fraction": candidate,
                "seed_fraction": seed,
                "interaction_fraction": interaction,
                "noncandidate_fraction": seed + interaction,
            }
        )
    decision, diagnostic = module.classify_variance_source(rows)
    assert decision == "probe_candidate_conditioned_pp_order"
    assert diagnostic["six_hundred_root_order_fraction"] == pytest.approx(0.75)


def test_classification_distinguishes_continuation_amplification() -> None:
    rows = []
    for metric, candidate, seed, interaction in (
        ("root_conflict_reduction", 0.80, 0.05, 0.15),
        ("root_state_changed", 0.85, 0.05, 0.10),
        ("h3_normalized_step_auc", 0.40, 0.10, 0.50),
    ):
        rows.append(
            {
                "metric": metric,
                "agent_group": "600",
                "candidate_fraction": candidate,
                "seed_fraction": seed,
                "interaction_fraction": interaction,
                "noncandidate_fraction": seed + interaction,
            }
        )
    decision, _ = module.classify_variance_source(rows)
    assert decision == "fix_h3_continuation_labels_before_pp_order"
