from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_compact_flow_expanded_stage2_router_v1 as router
from experiments._common import sha256_file


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / "configs" / "stride_hierarchical_ch_compact_flow_expanded_stage2_router_v1.json"
)


def _raw_features(offset: float) -> dict[str, float]:
    raw = {}
    for index, name in enumerate(router.FEATURE_NAMES):
        kind, source = name.split(":", 1)
        if kind in {"context", "delta_component_minus_hotspot"}:
            raw[source] = float(index + 1) + offset
    return raw


def test_config_freezes_sixteen_maps_and_support_first_boundary() -> None:
    _, _, config = router.load_config(CONFIG)
    assert config["train_map_folds"] == {
        fold: list(maps) for fold, maps in router.FOLDS.items()
    }
    assert len({map_id for maps in router.FOLDS.values() for map_id in maps}) == 16
    assert config["support_gates"]["minimum_examples_per_decisive_class"] == 2
    assert config["support_gates"]["minimum_maps_per_decisive_class"] == 2
    assert config["support_gates"]["minimum_depth_bands_per_decisive_class"] == 2
    assert config["claim_boundary"]["development_rows_allowed"] is False
    assert config["claim_boundary"]["final_rows_allowed"] is False
    assert config["claim_boundary"]["runtime_or_ttf_claim_authorized"] is False


def _state_payload(state_id: str) -> dict:
    component = _raw_features(3.0)
    hotspot = _raw_features(1.0)
    state = {
        "state_occurrence_id": state_id,
        "candidate_repair_actions_executed": False,
        "outcome_filtering": False,
        "target_outcome_fields_read": [],
        "sequential_design_only": True,
        "role_to_action_id": {
            "v2_anchor": "v",
            "component16": "c",
            "hotspot16": "h",
        },
        "unique_actions": [
            {"action_id": "v", "agents": [0, 1]},
            {"action_id": "c", "agents": [0, 1, 2, 3]},
            {"action_id": "h", "agents": [2, 3, 4, 5]},
        ],
        "arms": {
            "v2_anchor": {"features": _raw_features(0.0)},
            "component16": {"features": component},
            "hotspot16": {"features": hotspot},
        },
    }
    return {
        "complete": True,
        "sequential_design_only": True,
        "runtime_used_in_label": False,
        "final_claim_authorized": False,
        "state_occurrence_id": state_id,
        "state_row": state,
    }


def test_exact_component_hotspot_and_registered_vector_are_recomputed(tmp_path: Path) -> None:
    state_id = "state-a"
    payload = _state_payload(state_id)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    relative = state_path.relative_to(tmp_path).as_posix()
    vector = router._recompute_feature_vector(
        payload["state_row"], [0, 1, 2, 3], [2, 3, 4, 5]
    )
    row = {
        "schema": router.LABEL_SCHEMA,
        "state_occurrence_id": state_id,
        "label": "component_win",
        "research_split": "expanded_train",
        "expanded_train": True,
        "runtime_or_pp_seconds_used_in_label": False,
        "sequential_design_only": True,
        "training_authorized": True,
        "repair_outcome_or_runtime_used_as_model_feature": False,
        "development_evaluation_authorized": False,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "promotion_authorized": False,
        "exact_component_hotspot_sets_differ": True,
        "component_action_id": "c",
        "hotspot_action_id": "h",
        "exact_component_agents": [0, 1, 2, 3],
        "exact_hotspot_agents": [2, 3, 4, 5],
        "model_feature_names": list(router.FEATURE_NAMES),
        "model_feature_vector": vector,
        "source_label_row_sha256": "a" * 64,
        "source_h1_state_file": relative,
        "source_h1_state_sha256": sha256_file(state_path),
        "map_id": "den404d",
        "map_family": "dao",
        "train_fold": "fold0",
        "depth_band": "d0",
        "hierarchical_stratum": "three_unique",
    }
    example = router.build_stage2_example(row, tmp_path, router.FOLDS)
    assert example["target"] == 1
    assert len(example["features"]) == 18
    assert example["features"][8] == pytest.approx(2.0)
    assert example["features"][-3:] == pytest.approx([0.5, 1 / 3, 2 / 3])
    row["exact_hotspot_agents"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="exact C/H sets are equal"):
        router.build_stage2_example(row, tmp_path, router.FOLDS)


def _synthetic_examples(*, missing_hotspot_fold: str | None = None) -> list[dict]:
    rows = []
    counter = 0
    for fold_index, (fold, maps) in enumerate(router.FOLDS.items()):
        depth = "d0" if fold_index % 2 == 0 else "d1_3"
        for map_id in maps:
            counter += 1
            rows.append({
                "state_occurrence_id": f"amb-{counter}",
                "map_id": map_id,
                "train_fold": fold,
                "label": "ambiguous",
                "target": None,
                "depth_band": depth,
                "map_family": "dao",
                "hierarchical_stratum": "three_unique",
                "features": [0.5] + [0.0] * 17,
            })
        for label, signal, map_id in (
            ("component_win", 1.0, maps[0]),
            ("hotspot_win", 0.0, maps[1]),
        ):
            if label == "hotspot_win" and fold == missing_hotspot_fold:
                continue
            counter += 1
            rows.append({
                "state_occurrence_id": f"dec-{counter}",
                "map_id": map_id,
                "train_fold": fold,
                "label": label,
                "target": 1 if label == "component_win" else 0,
                "depth_band": depth,
                "map_family": "dao",
                "hierarchical_stratum": "three_unique",
                "features": [signal] + [0.0] * 17,
            })
    return rows


def test_support_audit_requires_both_directions_in_every_frozen_fold() -> None:
    _, _, config = router.load_config(CONFIG)
    passed = router.audit_stage2_support(
        _synthetic_examples(), router.FOLDS, config["support_gates"]
    )
    assert passed["passed"] is True
    failed = router.audit_stage2_support(
        _synthetic_examples(missing_hotspot_fold="fold2"),
        router.FOLDS,
        config["support_gates"],
    )
    assert failed["passed"] is False
    assert failed["checks"]["each_fold_requires_both_decisive_classes"] is False
    assert failed["by_fold"]["fold2"]["both_decisive_classes_present"] is False


def test_support_failure_never_calls_fit() -> None:
    _, _, config = router.load_config(CONFIG)
    examples = _synthetic_examples(missing_hotspot_fold="fold2")
    support = router.audit_stage2_support(examples, router.FOLDS, config["support_gates"])

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("fit must not be called on support failure")

    results, fitted = router.evaluate_candidates_if_supported(
        examples, support, config, workers=16, fit=forbidden_fit
    )
    assert results == []
    assert fitted is False


class _SignalEstimator:
    pass


def _fake_fit(_spec: dict, rows: list[dict]) -> _SignalEstimator:
    assert rows
    assert {row["target"] for row in rows} == {0, 1}
    assert all(row["label"] != "ambiguous" for row in rows)
    return _SignalEstimator()


def _fake_predict(_estimator: _SignalEstimator, rows: list[dict]) -> list[float]:
    return [
        0.9 if row["features"][0] > 0.75 else 0.1 if row["features"][0] < 0.25 else 0.5
        for row in rows
    ]


def test_supported_rows_run_four_map_grouped_oof_folds() -> None:
    _, _, config = router.load_config(CONFIG)
    result = router.cross_validate_candidate(
        _synthetic_examples(),
        config["models"][0],
        config,
        fit=_fake_fit,
        predict=_fake_predict,
    )
    assert result["passed"] is True
    assert result["oof_metrics"]["balanced_accuracy"] == 1.0
    assert result["oof_metrics"]["selective_accuracy"] == 1.0
    assert result["oof_metrics"]["ambiguous_selection_rate"] == 0.0
    assert all(row["state_overlap_count"] == 0 for row in result["fold_leakage_audit"])
    assert all(row["map_overlap_count"] == 0 for row in result["fold_leakage_audit"])
