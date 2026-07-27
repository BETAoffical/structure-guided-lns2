from __future__ import annotations

import pytest

from experiments import v2_cost_tiebreak_error_audit as module


def _actual(
    *, reduction: float = 10.0, seconds: float = 1.0, effective: float = 1.0
) -> dict[str, float]:
    return {
        "conflict_reduction": reduction,
        "repair_seconds": seconds,
        "effective": effective,
        "no_progress": 1.0 - effective,
    }


def test_preferred_requires_quality_and_ten_percent_time_gain() -> None:
    base = _actual()
    assert module._preferred(_actual(reduction=9.8, seconds=0.9), base)
    assert not module._preferred(_actual(reduction=9.7, seconds=0.5), base)
    assert not module._preferred(_actual(reduction=10.0, seconds=0.91), base)


def test_pair_features_keep_candidate_deltas_and_base_state_context() -> None:
    base = [float(index) for index in range(len(module.V3_FEATURE_NAMES))]
    candidate = [value + 1.0 for value in base]
    values = module._pair_features(candidate, base, v2_score_gap=-0.5)
    state_start = module.V3_FEATURE_NAMES.index("state.agent_count")
    assert len(values) == module.PAIRWISE_FEATURE_COUNT
    assert values[:state_start] == [1.0] * state_start
    assert values[state_start:-1] == base[state_start:]
    assert values[-1] == -0.5


def test_paired_trial_validation_rejects_unpaired_random_seed() -> None:
    features = [
        {"split": "train", "state_id": "state", "candidate_id": candidate}
        for candidate in ("a", "b")
    ]
    trials = []
    for candidate in ("a", "b"):
        for trial in (0, 1):
            trials.append(
                {
                    "split": "train",
                    "state_id": "state",
                    "candidate_id": candidate,
                    "trial_index": trial,
                    "random_seed": trial + int(candidate == "b"),
                    "status": "ok",
                    "complete": True,
                }
            )
    with pytest.raises(ValueError, match="paired random seeds"):
        module.validate_paired_trials(features, trials)


def test_seed_stability_separates_all_and_conditional_agreement() -> None:
    pairs = [
        {
            "seed_label_agreement": 1,
            "seed_zero_label": 0,
            "seed_one_label": 0,
            "any_seed_preferred": 0,
            "trial_count": 2,
            "all_available_seed_label_agreement": 1,
        },
        {
            "seed_label_agreement": 0,
            "seed_zero_label": 1,
            "seed_one_label": 0,
            "any_seed_preferred": 1,
            "trial_count": 2,
            "all_available_seed_label_agreement": 0,
        },
    ]
    states = [
        {"oracle_candidate_agreement": 0, "oracle_route_agreement": 1}
    ]
    report = module.seed_stability_report(pairs, states)
    assert report["pair_label_agreement_fraction"] == 0.5
    assert report["conditional_pair_label_agreement_fraction"] == 0.0
    assert report["state_oracle_route_agreement_fraction"] == 1.0


def test_pairwise_weights_require_both_classes() -> None:
    rows = [{"label": 0, "map_id": "map"}]
    with pytest.raises(ValueError, match="both target classes"):
        module._balanced_pair_weights(rows)


def test_pairwise_prediction_quality_reports_perfect_classifier() -> None:
    rows = [
        {"state_id": "a", "candidate_id": "x", "label": 0},
        {"state_id": "b", "candidate_id": "y", "label": 1},
    ]
    report = module.pairwise_prediction_quality(
        rows, {("a", "x"): 0.1, ("b", "y"): 0.9}
    )
    assert report["roc_auc"] == 1.0
    assert report["accuracy_at_0_5"] == 1.0


def test_binary_roc_auc_handles_tied_scores() -> None:
    assert module._binary_roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_pairwise_threshold_selection_prefers_passing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_evaluate(states, predictions, threshold):
        passing = threshold == 0.8
        ratio = 0.89 if passing else 0.95
        metrics = {
            "comparison": {
                "effective_rate_delta": 0.0,
                "no_progress_rate_delta": 0.0,
                "conflict_reduction_ratio": 1.0,
                "repair_time_ratio": ratio,
                "efficiency_ratio": 1.0 / ratio,
            },
            "override_fraction": 0.1,
            "cell_gate": {
                "cell_count": 6,
                "noninferior_cell_count": 6,
            },
        }
        return metrics, []

    monkeypatch.setattr(module, "evaluate_pairwise_policy", fake_evaluate)
    calibration, _ = module.calibrate_pairwise_policy([], {})
    assert calibration["passed"] is True
    assert calibration["selected_threshold"] == 0.8
