from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_temporal_anchor_incremental_readiness import (
    FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    _target_blind_prefix,
    force_decision0_abstention_probabilities,
    history_feature_vector,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_temporal_anchor_incremental_readiness_v1.json"


def _transition(index: int, *, agents: list[int], success: bool, delta: int, before: int) -> dict:
    return {
        "event": "transition",
        "decision_index": index,
        "before_fingerprint": f"before-{index}",
        "after_fingerprint": f"after-{index}",
        "action": {"forbidden_target_action": index},
        "state_delta": {"forbidden_target_delta": index},
        "metrics": {
            "neighborhood": agents,
            "repair_order": list(reversed(agents)),
            "applied_pp_random_seed": 100 + index,
            "replan_success": success,
            "conflict_delta": delta,
            "conflicts_before": before,
            "forbidden_target_metric": index,
        },
    }


class TemporalAnchorIncrementalReadinessTest(unittest.TestCase):
    def test_frozen_config_and_dimensions(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_config(config)
        self.assertEqual(len(HISTORY_FEATURE_NAMES), 16)
        self.assertEqual(len(FEATURE_NAMES), 209)
        tampered = copy.deepcopy(config)
        tampered["features"]["history_feature_names"][0] = "history.retuned"
        with self.assertRaisesRegex(ValueError, "feature contract"):
            validate_config(tampered)

    def test_history_vector_is_anchor_relative_and_decision0_is_zero(self) -> None:
        challenger = {1, 2}
        anchor = {2, 3}
        transitions = [
            {"neighborhood": [1, 2], "replan_success": False, "conflict_delta": 0, "conflicts_before": 10},
            {"neighborhood": [2, 3], "replan_success": True, "conflict_delta": 2, "conflicts_before": 8},
        ]
        values = history_feature_vector(challenger, anchor, transitions)
        for observed, expected in zip(values[:8], [2, 1, 0, 1, 0, 0, .25, .125]):
            self.assertAlmostEqual(observed, expected)
        self.assertLess(values[8], 0)  # the anchor, not the challenger, repeats the last action
        self.assertEqual(history_feature_vector(challenger, anchor, []), [0.0] * 16)

    def test_target_transition_mutation_cannot_change_prefix_features(self) -> None:
        prefix = _transition(0, agents=[1, 2], success=True, delta=1, before=5)
        target = _transition(1, agents=[7, 8], success=False, delta=-999, before=4)
        events = [{"event": "initial"}, prefix, target, {"event": "finish"}]
        expected = _target_blind_prefix(events, 1)
        mutated = copy.deepcopy(events)
        mutated[2]["action"] = {"entirely": "changed"}
        mutated[2]["metrics"] = {"future": "poison"}
        mutated[2]["state_delta"] = {"future": "poison"}
        mutated[2]["after_fingerprint"] = "future-poison"
        self.assertEqual(_target_blind_prefix(mutated, 1), expected)
        self.assertEqual(set(expected[0]), {
            "decision_index", "neighborhood", "replan_success", "conflict_delta",
            "conflicts_before", "before_fingerprint", "after_fingerprint",
        })

    def test_decision0_is_forced_to_abstention_probability(self) -> None:
        rows = [{"decision_index": 0}, {"decision_index": 2}]
        self.assertEqual(
            force_decision0_abstention_probabilities(rows, [.999, .91]), [0.0, .91]
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            force_decision0_abstention_probabilities(rows, [.9])
