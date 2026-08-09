from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.repair_collection import _read_json
from experiments.stride_safeslot_label_readiness import (
    compare_paired_current_step_trials,
    validate_safeslot_label_readiness_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_safeslot_label_readiness_v1.json"


def _trials(scores: list[float], *, candidate_id: str) -> list[dict[str, object]]:
    return [
        {
            "state_id": "state-1",
            "candidate_id": candidate_id,
            "trial_index": index,
            "pp_seed": 1000 + index,
            "before_fingerprint": "fingerprint-1",
            "before_conflicts": 100,
            "normalized_conflict_reduction": score,
        }
        for index, score in enumerate(scores)
    ]


class SafeSlotLabelReadinessTests(unittest.TestCase):
    def test_registered_config_freezes_pre_residual_boundary(self) -> None:
        config = _read_json(CONFIG)
        validate_safeslot_label_readiness_config(config)
        self.assertFalse(config["claim_boundary"]["model_training_allowed"])
        self.assertTrue(
            config["paired_pre_residual_label"][
                "residual_teacher_required_for_final_positive_label"
            ]
        )

    def test_stable_paired_advantage_is_pre_residual_positive(self) -> None:
        config = _read_json(CONFIG)
        anchor = _trials([0.20] * 16, candidate_id="anchor")
        challenger = _trials([0.25] * 16, candidate_id="challenger")
        result = compare_paired_current_step_trials(
            challenger, anchor, config["paired_pre_residual_label"]
        )
        self.assertTrue(result["pre_residual_positive"])
        self.assertAlmostEqual(result["mean_advantage"], 0.05)
        self.assertEqual(result["paired_win_fraction"], 1.0)

    def test_no_progress_regression_rejects_candidate(self) -> None:
        config = _read_json(CONFIG)
        anchor = _trials([0.01] * 16, candidate_id="anchor")
        challenger = _trials(
            [0.20] * 12 + [0.0] * 4, candidate_id="challenger"
        )
        result = compare_paired_current_step_trials(
            challenger, anchor, config["paired_pre_residual_label"]
        )
        self.assertFalse(result["pre_residual_positive"])
        self.assertGreater(
            result["challenger_no_progress_rate"], result["anchor_no_progress_rate"]
        )

    def test_seed_mismatch_is_rejected(self) -> None:
        config = _read_json(CONFIG)
        anchor = _trials([0.20] * 16, candidate_id="anchor")
        challenger = _trials([0.25] * 16, candidate_id="challenger")
        challenger = copy.deepcopy(challenger)
        challenger[5]["pp_seed"] = 999999
        with self.assertRaisesRegex(ValueError, "PP seed differs"):
            compare_paired_current_step_trials(
                challenger, anchor, config["paired_pre_residual_label"]
            )


if __name__ == "__main__":
    unittest.main()
