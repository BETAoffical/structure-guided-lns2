from __future__ import annotations

import numpy as np

from experiments.stride_certguard_training import (
    _certguard_predictions,
    _uncertainty_matrix,
)


def test_uncertainty_matrix_is_symmetric_abs_delta_plus_mean() -> None:
    candidates = np.asarray([[1.0, 4.0], [3.0, 2.0]], dtype=np.float32)
    forward = {
        "left": np.asarray([0]),
        "right": np.asarray([1]),
    }
    reverse = {
        "left": np.asarray([1]),
        "right": np.asarray([0]),
    }
    assert _uncertainty_matrix(candidates, forward).tolist() == [[2.0, 2.0, 2.0, 3.0]]
    assert np.array_equal(
        _uncertainty_matrix(candidates, forward),
        _uncertainty_matrix(candidates, reverse),
    )


def test_certguard_can_only_retain_or_veto_maprank_override() -> None:
    anchor = {"same": "a", "allow": "a", "veto": "a"}
    maprank = {"same": "a", "allow": "b", "veto": "c"}
    certainties = {"same": 1.0, "allow": 0.8, "veto": 0.6}
    selected = _certguard_predictions(
        maprank_predictions=maprank,
        anchor_predictions=anchor,
        certainty_probabilities=certainties,
        certainty_threshold=0.7,
    )
    assert selected == {"same": "a", "allow": "b", "veto": "a"}
    assert all(selected[state] in {anchor[state], maprank[state]} for state in selected)
