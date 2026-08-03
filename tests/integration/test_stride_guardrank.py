from __future__ import annotations

import json
from pathlib import Path

from experiments.stride_guardrank import (
    CONTROLLER_ID,
    _balanced_folds,
    _calibrate_thresholds,
    _guard_predictions,
    validate_guardrank_training_config,
)


def _config() -> dict:
    root = Path(__file__).resolve().parents[2]
    return json.loads(
        (root / "configs" / "stride_guardrank_training.json").read_text(
            encoding="utf-8"
        )
    )


def _candidate(state: str, candidate: str, score: float, kind: str) -> dict:
    return {
        "state_id": state,
        "candidate_id": candidate,
        "candidate_key": candidate,
        "candidate_kind": kind,
        "repairability_score": score,
        "mean_conflicts_after": 1.0 - score,
        "mean_conflict_reduction_ratio": score,
        "progress_rate": score,
        "feasible_rate": 0.0,
        "map_id": "map",
        "split": "train",
        "source_policy": "v2-full",
        "decision_stage": "early",
        "agent_band": "medium",
        "topology_group": "control",
    }


def test_guardrank_registration_freezes_train_only_calibration() -> None:
    config = _config()
    validate_guardrank_training_config(config)
    assert config["controller_id"] == CONTROLLER_ID
    assert config["forbidden_calibration_splits"] == [
        "validation",
        "test",
        "formal_ood",
    ]
    assert config["fresh_development_validation_required"] is True
    config["legacy_validation_is_descriptive_only"] = False
    try:
        validate_guardrank_training_config(config)
    except ValueError as error:
        assert "evidence boundary" in str(error)
    else:
        raise AssertionError("validation calibration drift was accepted")


def test_guard_predictions_keep_anchor_when_evidence_is_below_threshold() -> None:
    evidence = {
        "state": {
            "anchor_candidate_id": "anchor",
            "challenger_candidate_id": "challenger",
            "challenger_kind": "base",
            "challenger_evidence": 0.7,
        }
    }
    assert _guard_predictions(
        evidence, {"base": 0.75, "boundary_only": 0.75}
    ) == {"state": "anchor"}
    assert _guard_predictions(
        evidence, {"base": 0.65, "boundary_only": 0.75}
    ) == {"state": "challenger"}


def test_calibration_can_fall_back_exactly_to_v2() -> None:
    grouped = {
        "state": [
            _candidate("state", "anchor", 1.0, "base"),
            _candidate("state", "challenger", 0.0, "boundary_only"),
        ]
    }
    evidence = {
        "state": {
            "anchor_candidate_id": "anchor",
            "challenger_candidate_id": "challenger",
            "challenger_kind": "boundary_only",
            "challenger_evidence": 0.9,
        }
    }
    calibrated = _calibrate_thresholds(
        evidence=evidence,
        grouped=grouped,
        anchor_predictions={"state": "anchor"},
        grid=[0.55, 1.01],
        exact_tolerance=0.01,
        top3_tolerance=0.01,
    )
    assert calibrated["thresholds"]["boundary_only"] == 1.01
    assert calibrated["metrics"]["exact_best_rate"] == 1.0


def test_map_folds_never_split_a_map() -> None:
    folds = _balanced_folds(
        {"den001d", "den002d", "den003d", "den004d", "lak001d"}, 4
    )
    validation = [map_id for fold in folds for map_id in fold["validation_maps"]]
    assert sorted(validation) == [
        "den001d",
        "den002d",
        "den003d",
        "den004d",
        "lak001d",
    ]
    assert len(validation) == len(set(validation))
