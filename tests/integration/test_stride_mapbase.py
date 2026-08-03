from __future__ import annotations

import json
from pathlib import Path

from experiments._common import sha256_file
from experiments.repair_collection import _read_jsonl
from experiments.stride_mapbase import (
    build_mapbase_selection,
    validate_mapbase_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "stride_mapbase_collection.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_mapbase_is_base_only_multiseed_training_collection() -> None:
    config = _config()
    validate_mapbase_config(config)
    assert config["collection_id"] == "stride-mapbase-v1"
    assert config["expected_active_state_count"] == 63
    assert config["trial_indices"] == list(range(16))
    assert config["topology_boundary_candidates_allowed"] is False
    assert config["candidate_pool"]["topology_boundary"] == "forbidden"
    assert config["selection"]["research_split"] == "train"


def test_mapbase_registered_inputs_and_selection_match_when_available() -> None:
    config = _config()
    paths = {}
    for name, artifact in config["inputs"].items():
        path = ROOT / artifact["path"]
        if not path.is_file():
            return
        assert sha256_file(path) == artifact["sha256"]
        paths[name] = path
    selection = build_mapbase_selection(
        config,
        recommended_tasks=_read_jsonl(paths["recommended_tasks"]),
        relevance_states=_read_jsonl(paths["relevance_states"]),
        dataset_rows=_read_jsonl(paths["dataset_manifest"]),
        qualification_rows=_read_jsonl(paths["qualification_manifest"]),
    )
    assert len(selection) == 63
    assert len({row["map_id"] for row in selection}) == 8
    assert all(row["before_conflicts"] > 0 for row in selection)
    assert all(row["research_split"] == "train" for row in selection)
