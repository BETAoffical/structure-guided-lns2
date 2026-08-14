from __future__ import annotations

import json
from pathlib import Path

from experiments.stride_compact_nativeorder_repair import (
    COMPACT_RETRY,
    COMPACT_SINGLE,
    FULL_RETRY,
    FULL_SINGLE,
    POLICIES,
    semantic_compact_plan,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stride_compact_nativeorder_repair_v1_registration.json"


def _state() -> dict:
    return {
        "rows": 2,
        "cols": 3,
        "obstacles": [0, 0, 0, 0, 0, 0],
        "agents": [
            {"id": 0, "start": 0, "goal": 2, "path": [0, 1, 2]},
            {"id": 1, "start": 2, "goal": 0, "path": [2, 1, 0]},
            {"id": 2, "start": 3, "goal": 5, "path": [3, 4, 5]},
        ],
        "conflict_edges": [[0, 1]],
    }


def test_registration_freezes_factorial_design_without_fixed_size() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert tuple(config["policies"]["ids"]) == POLICIES
    assert POLICIES == (FULL_SINGLE, FULL_RETRY, COMPACT_SINGLE, COMPACT_RETRY)
    assert config["policies"]["compact_rule"]["fixed_target_size"] is None
    assert config["execution"]["worker_count"] == 16
    assert config["claim_boundary"]["ttf_experiment_allowed"] is False


def test_semantic_compaction_removes_only_unsupported_member() -> None:
    plan = semantic_compact_plan(_state(), [0, 1, 2])
    assert plan["eligible"] is True
    assert plan["current_conflict_core"] == [0, 1]
    assert plan["compact_agents"] == [0, 1]
    assert plan["removed_agents"] == [2]
    assert plan["actual_size"] == 2


def test_semantic_compaction_falls_back_when_full_set_is_supported() -> None:
    plan = semantic_compact_plan(_state(), [0, 1])
    assert plan["eligible"] is False
    assert plan["compact_agents"] == [0, 1]
    assert plan["removed_agents"] == []
    assert plan["rejection_reason"] == "no_unsupported_agent_to_remove"
