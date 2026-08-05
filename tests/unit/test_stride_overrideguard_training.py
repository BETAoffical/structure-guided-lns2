from __future__ import annotations

import numpy as np

from experiments.stride_overrideguard_training import (
    _action_matrix,
    _overrideguard_predictions,
)


def test_action_matrix_preserves_candidate_anchor_orientation() -> None:
    candidates = np.asarray([[1.0, 4.0], [3.0, 2.0]], dtype=np.float32)
    forward = {
        "candidate": np.asarray([0]),
        "anchor": np.asarray([1]),
    }
    reverse = {
        "candidate": np.asarray([1]),
        "anchor": np.asarray([0]),
    }
    assert _action_matrix(candidates, forward).tolist() == [[-2.0, 2.0, 2.0, 3.0]]
    assert _action_matrix(candidates, reverse).tolist() == [[2.0, -2.0, 2.0, 3.0]]


def test_overrideguard_can_only_retain_or_veto_maprank_override() -> None:
    anchor = {"same": "a", "allow": "a", "veto": "a"}
    maprank = {"same": "a", "allow": "b", "veto": "c"}
    probabilities = {"same": 1.0, "allow": 0.8, "veto": 0.6}
    selected = _overrideguard_predictions(
        maprank_predictions=maprank,
        anchor_predictions=anchor,
        safety_probabilities=probabilities,
        safety_threshold=0.7,
    )
    assert selected == {"same": "a", "allow": "b", "veto": "a"}
    assert all(selected[state] in {anchor[state], maprank[state]} for state in selected)
