from __future__ import annotations

import json
from pathlib import Path

from experiments._common import sha256_file
from experiments.stride_guardrank_map_coverage import (
    validate_guardrank_map_coverage_config,
)


ROOT = Path(__file__).resolve().parents[2]


def test_guardrank_map_coverage_is_fresh_proposal_only_diagnostic() -> None:
    path = ROOT / "configs" / "stride_guardrank_map_coverage.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_guardrank_map_coverage_config(config)
    assert config["expected_task_count"] == 16
    assert config["expected_state_count"] == 32
    assert config["candidate_repair_trials_allowed"] is False
    assert config["controller_actions_allowed"] is False
    assert config["controller_outcomes_allowed"] is False
    assert config["freshness"] == {
        "map_id_overlap_with_training_cohort": 0,
        "map_id_overlap_with_formal_ood": 0,
        "quality_outcomes_consumed": False,
        "scope": "cross_map_train_extension_proposal_audit",
    }


def test_guardrank_map_coverage_registered_inputs_match_when_available() -> None:
    config = json.loads(
        (ROOT / "configs" / "stride_guardrank_map_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    for artifact in config["inputs"].values():
        path = ROOT / artifact["path"]
        if path.is_file():
            assert sha256_file(path) == artifact["sha256"]
