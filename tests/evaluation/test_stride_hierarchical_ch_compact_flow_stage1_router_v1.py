from __future__ import annotations

from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_compact_flow_stage1_router_v1 as router


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_hierarchical_ch_compact_flow_stage1_router_v1.json"


def _raw_features(offset: float) -> dict[str, float]:
    raw = {}
    for index, name in enumerate(router.FEATURE_NAMES):
        kind, source = name.split(":", 1)
        if kind in {"context", "delta"}:
            raw[source] = float(index + 1) + offset
    return raw


def test_fixed_small_config_and_stage1_only_boundary() -> None:
    _, _, config = router.load_config(CONFIG)
    assert len(router.FEATURE_NAMES) == 18
    assert [row["model_id"] for row in config["models"]] == [
        "regularized_logistic",
        "low_capacity_hist_gradient_boosting",
    ]
    assert list(config["train_map_folds"]) == ["fold0", "fold1", "fold2", "fold3"]
    assert config["input_semantics"]["stage2_rows_loaded"] is False
    assert config["input_semantics"]["stage2_model_fit"] is False
    assert config["claim_boundary"]["promotion_or_default_export_allowed"] is False


def test_pair_features_use_only_pre_action_context_delta_and_exact_sets() -> None:
    anchor = _raw_features(0.0)
    challenger = _raw_features(2.0)
    state = {
        "state_occurrence_id": "state-a",
        "role_to_action_id": {"v2_anchor": "v", "component16": "c", "hotspot16": "h"},
        "unique_actions": [
            {"action_id": "v", "agents": [0, 1, 2, 3]},
            {"action_id": "c", "agents": [2, 3, 4, 5]},
            {"action_id": "h", "agents": [4, 5, 6, 7]},
        ],
        "arms": {
            "v2_anchor": {"features": anchor},
            "component16": {"features": challenger},
            "hotspot16": {"features": _raw_features(4.0)},
        },
    }
    row = {
        "schema": router.STAGE1_PAIR_SCHEMA,
        "pair_id": "pair-a",
        "state_occurrence_id": "state-a",
        "research_split": "train",
        "map_id": "den404d",
        "train_fold": "fold0",
        "label": "structural_win",
        "depth_band": "d0",
        "hierarchical_stratum": "three_unique",
        "structural_action_id": "c",
        "v2_action_id": "v",
        "structural_role_aliases": ["component16"],
        "runtime_or_pp_seconds_used_in_label": False,
        "sequential_design_only": True,
        "training_authorized": False,
    }
    example = router.build_pair_example(row, state)
    assert len(example["features"]) == 18
    assert example["features"][0] == anchor["state.agent_count"]
    assert example["features"][8] == pytest.approx(2.0)
    assert example["features"][-3:] == pytest.approx([0.5, 1 / 3, 2 / 3])
    assert example["target"] == 1
    assert not any(
        token in feature.lower()
        for feature in router.FEATURE_NAMES
        for token in ("outcome", "trial", "runtime", "map_id", "task_id", "result")
    )


def _synthetic_examples() -> list[dict]:
    rows = []
    pair_index = 0
    for fold_index, (fold, maps) in enumerate(router.FOLDS.items()):
        for map_index, map_id in enumerate(maps):
            role = "component16" if map_index == 0 else "hotspot16"
            depth = "d0" if fold_index % 2 == 0 else "d4plus"
            for label, signal in (
                ("structural_win", 1.0),
                ("v2_win", 0.0),
                ("ambiguous", 0.5),
            ):
                pair_index += 1
                rows.append(
                    {
                        "pair_id": f"pair-{pair_index}",
                        "state_occurrence_id": f"state-{fold}-{map_id}-{label}",
                        "map_id": map_id,
                        "research_split": "train",
                        "train_fold": fold,
                        "label": label,
                        "target": 1 if label == "structural_win" else 0 if label == "v2_win" else None,
                        "depth_band": depth,
                        "role_stratum": role,
                        "hierarchical_stratum": "three_unique",
                        "features": [signal] + [0.0] * 17,
                    }
                )
    return rows


class _SignalEstimator:
    pass


def _fake_fit(_spec: dict, rows: list[dict]) -> _SignalEstimator:
    assert rows
    assert {row["target"] for row in rows} == {0, 1}
    assert all(row["label"] != "ambiguous" for row in rows)
    return _SignalEstimator()


def _fake_predict(_estimator: _SignalEstimator, rows: list[dict]) -> list[float]:
    return [0.9 if row["features"][0] > 0.75 else 0.1 if row["features"][0] < 0.25 else 0.5 for row in rows]


def test_four_fixed_map_folds_group_all_pairs_and_ambiguous_calibrates_only() -> None:
    _, _, config = router.load_config(CONFIG)
    result = router.cross_validate_candidate(
        _synthetic_examples(),
        config["models"][0],
        config,
        fit=_fake_fit,
        predict=_fake_predict,
    )
    assert result["passed"] is True
    assert result["threshold_calibration"]["selected_threshold"] == 0.55
    assert result["threshold_calibration"]["ambiguous_fitted_as_class"] is False
    assert result["oof_metrics"]["balanced_accuracy"] == 1.0
    assert result["oof_metrics"]["coverage"] == 1.0
    assert result["oof_metrics"]["selective_accuracy"] == 1.0
    assert result["oof_metrics"]["ambiguous_selection_rate"] == 0.0
    assert all(row["state_overlap_count"] == 0 for row in result["fold_leakage_audit"])
    assert all(row["map_overlap_count"] == 0 for row in result["fold_leakage_audit"])


def test_threshold_fails_closed_to_abstain_all_when_no_train_oof_point_is_feasible() -> None:
    predictions = [
        {"label": "structural_win", "probability": 0.51},
        {"label": "v2_win", "probability": 0.49},
        {"label": "ambiguous", "probability": 0.99},
    ]
    result = router.select_abstention_threshold(
        predictions,
        {
            "maximum_ambiguous_selection_rate": 0.0,
            "minimum_decisive_coverage": 0.5,
        },
    )
    assert result["status"] == "no_feasible_threshold_abstain_all"
    assert result["selected_threshold"] == 1.01
    assert result["ambiguous_fitted_as_class"] is False


def test_readiness_requires_stage1_true_and_stage2_false() -> None:
    base = {
        "schema": router.READINESS_SCHEMA,
        "scientific_status": "sequential_labels_only_readiness",
        "complete": True,
        "splits": {
            "train": {
                "stage1": {"label_support_trainable": True},
                "stage2": {"label_support_trainable": False},
            }
        },
        "fixed_train_map_folds": {fold: list(maps) for fold, maps in router.FOLDS.items()},
        "development_has_training_folds": False,
        "sequential_design_only": True,
        "development_is_sequential_diagnostic_only": True,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_confirmation_required": True,
    }
    router._validate_readiness(base)
    base["splits"]["train"]["stage2"]["label_support_trainable"] = True
    with pytest.raises(ValueError, match="train.stage2"):
        router._validate_readiness(base)


def test_state_balancing_rejects_ambiguous_as_hard_fit_label() -> None:
    rows = _synthetic_examples()[:3]
    with pytest.raises(ValueError, match="no ambiguous"):
        router._state_class_balanced_weights(rows)
