from __future__ import annotations

from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_router_v1 as router


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_expanded_stage1_router_v1.json"
)


def _raw_features(offset: float) -> dict[str, float]:
    result = {}
    for index, name in enumerate(router.FEATURE_NAMES):
        kind, source = name.split(":", 1)
        if kind in {"context", "delta_structural_minus_v2"}:
            result[source] = float(index + 1) + offset
    return result


def test_fixed_expanded_config_has_no_development_or_stage2_boundary() -> None:
    _, _, config = router.load_config(CONFIG)
    assert len(router.FEATURE_NAMES) == 18
    assert len(router.EXPANDED_MAPS) == 16
    assert len(set(router.EXPANDED_MAPS)) == 16
    assert all(len(maps) == 4 for maps in router.FOLDS.values())
    assert [row["model_id"] for row in config["models"]] == [
        "regularized_logistic",
        "low_capacity_hist_gradient_boosting",
    ]
    assert config["input_semantics"]["stage2_rows_loaded"] is False
    assert config["claim_boundary"]["development_partition_present"] is False
    assert config["claim_boundary"]["final_claim_authorized"] is False
    assert config["claim_boundary"]["promotion_or_default_export_allowed"] is False
    assert "development_predictions" not in config["outputs"]


def test_pair_features_are_pre_action_context_delta_and_exact_sets_only() -> None:
    anchor = _raw_features(0.0)
    challenger = _raw_features(2.0)
    state = {
        "state_occurrence_id": "state-a",
        "role_to_action_id": {
            "v2_anchor": "v",
            "component16": "c",
            "hotspot16": "h",
        },
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
        "schema": router.EXPANDED_LABEL_SCHEMA,
        "pair_id": "pair-a",
        "state_occurrence_id": "state-a",
        "research_split": "expanded_train",
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
        "training_authorized": True,
        "promotion_authorized": False,
        "development_evaluation_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_final_confirmation_required": True,
        "expanded_train": True,
        "final_claim_authorized": False,
        "model_feature_direction": "structural_minus_v2",
        "repair_outcome_or_runtime_used_as_model_feature": False,
        "exact_structural_v2_sets_differ": True,
    }
    row["model_feature_names"] = list(router.FEATURE_NAMES)
    row["model_feature_vector"] = (
        [anchor[name.split(":", 1)[1]] for name in router.FEATURE_NAMES[:8]]
        + [2.0] * 7
        + [0.5, 1 / 3, 2 / 3]
    )
    example = router.build_pair_example(row, state)
    assert len(example["features"]) == 18
    assert example["features"][0] == anchor["state.agent_count"]
    assert example["features"][8] == pytest.approx(2.0)
    assert example["features"][-3:] == pytest.approx([0.5, 1 / 3, 2 / 3])
    assert example["research_split"] == "expanded_train"
    assert example["target"] == 1
    assert not any(
        token in feature.lower()
        for feature in router.FEATURE_NAMES
        for token in ("outcome", "trial", "runtime", "map_id", "task_id", "result")
    )


def _synthetic_examples() -> list[dict]:
    rows = []
    index = 0
    for fold_index, (fold, maps) in enumerate(router.FOLDS.items()):
        for map_index, map_id in enumerate(maps):
            role = "component16" if map_index % 2 == 0 else "hotspot16"
            depth = "d0" if fold_index % 2 == 0 else "d4plus"
            for label, signal in (
                ("structural_win", 1.0),
                ("v2_win", 0.0),
                ("ambiguous", 0.5),
            ):
                index += 1
                rows.append(
                    {
                        "pair_id": f"pair-{index}",
                        "state_occurrence_id": f"state-{fold}-{map_id}-{label}",
                        "map_id": map_id,
                        "research_split": "expanded_train",
                        "train_fold": fold,
                        "label": label,
                        "target": (
                            1
                            if label == "structural_win"
                            else 0
                            if label == "v2_win"
                            else None
                        ),
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
    return [
        0.9
        if row["features"][0] > 0.75
        else 0.1
        if row["features"][0] < 0.25
        else 0.5
        for row in rows
    ]


def test_all_16_maps_are_oof_grouped_and_ambiguous_only_calibrates() -> None:
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
    assert result["oof_metrics"]["ambiguous_selection_rate"] == 0.0
    assert all(row["held_map_count"] == 4 for row in result["fold_leakage_audit"])
    assert all(row["state_overlap_count"] == 0 for row in result["fold_leakage_audit"])
    assert all(row["map_overlap_count"] == 0 for row in result["fold_leakage_audit"])


def test_development_rows_are_rejected_from_expanded_oof() -> None:
    rows = _synthetic_examples()
    rows[0]["research_split"] = "development"
    _, _, config = router.load_config(CONFIG)
    with pytest.raises(ValueError, match="population"):
        router.cross_validate_candidate(
            rows,
            config["models"][0],
            config,
            fit=_fake_fit,
            predict=_fake_predict,
        )
