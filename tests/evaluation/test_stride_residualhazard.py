from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.stride_residualhazard import (
    residual_structure_metrics,
    seed_stable_direction,
    validate_registration_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_residualhazard_v1_registration.json"


def _state(paths: list[list[int]], conflict_edges: list[list[int]]) -> dict:
    agents = [
        {
            "id": index,
            "start": path[0],
            "goal": path[-1],
            "path": path,
        }
        for index, path in enumerate(paths)
    ]
    return {
        "rows": 2,
        "cols": 2,
        "obstacles": [0, 0, 0, 0],
        "agents": agents,
        "conflict_edges": conflict_edges,
        "num_of_colliding_pairs": len(conflict_edges),
    }


def test_registration_is_frozen() -> None:
    validate_registration_config(json.loads(CONFIG.read_text(encoding="utf-8")))


def test_residual_metrics_capture_new_boundary_conflict() -> None:
    before = _state([[0, 1], [3, 1], [2]], [[0, 1]])
    after = _state([[0, 1], [3], [2, 0, 1]], [[0, 2]])
    metrics = residual_structure_metrics(before, after, {0, 1})
    assert metrics["normalized_residual_conflict_count"] == pytest.approx(1.0)
    assert metrics["selected_unselected_residual_pair_ratio"] == pytest.approx(1.0)
    assert metrics["new_residual_pair_ratio"] == pytest.approx(1.0)
    assert metrics["low_degree_residual_event_ratio"] == pytest.approx(1.0)
    assert metrics["largest_residual_component_agent_ratio"] == pytest.approx(1.0)
    assert metrics["residual_conflict_cell_herfindahl"] == pytest.approx(1.0)
    assert metrics["outside_boundary_queue_agent_ratio"] == pytest.approx(1.0)


def test_seed_stability_requires_both_fixed_halves() -> None:
    stable = seed_stable_direction(([0.1] * 6 + [-0.1] * 2) * 2)
    assert stable["seed_stable_direction"] == "challenger_higher"
    split = seed_stable_direction([0.1] * 8 + [-0.1] * 8)
    assert split["seed_stable_direction"] == "uncertain"


def test_seed_stability_rejects_wrong_trial_count() -> None:
    with pytest.raises(ValueError, match="16 paired"):
        seed_stable_direction([0.1] * 15)
