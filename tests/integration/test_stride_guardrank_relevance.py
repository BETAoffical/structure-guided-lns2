from __future__ import annotations

import json
from pathlib import Path

from experiments._common import sha256_file
from experiments.stride_guardrank_relevance import (
    validate_guardrank_relevance_config,
)


ROOT = Path(__file__).resolve().parents[2]


def test_guardrank_relevance_is_input_only_and_preoutcome() -> None:
    config = json.loads(
        (ROOT / "configs" / "stride_guardrank_topology_relevance.json").read_text(
            encoding="utf-8"
        )
    )
    validate_guardrank_relevance_config(config)
    assert config["expected_state_count"] == 96
    assert config["candidate_generation_allowed"] is False
    assert config["candidate_repair_trials_allowed"] is False
    assert config["controller_outcomes_allowed"] is False
    assert config["repair_labels_allowed"] is False


def test_guardrank_relevance_registered_inputs_match_when_available() -> None:
    config = json.loads(
        (ROOT / "configs" / "stride_guardrank_topology_relevance.json").read_text(
            encoding="utf-8"
        )
    )
    for artifact in config["inputs"].values():
        path = ROOT / artifact["path"]
        if path.is_file():
            assert sha256_file(path) == artifact["sha256"]
