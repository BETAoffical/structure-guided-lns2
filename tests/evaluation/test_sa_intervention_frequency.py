import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments._common import read_json, json_fingerprint
from experiments.sa_intervention_frequency import (
    select_once, first_disagreement, horizon_outcome, completion_contrast, interpretation,
)
from scripts.run_sa_intervention_frequency import check_prefix, trace_prefix


class InterventionTests(unittest.TestCase):
    def test_once_not_first_step_only(self):
        used = False
        actions = []
        for prediction in ("a", "a", "b", "b", "c"):
            selected, used, used_now = select_once("a", prediction, used)
            actions.append((selected, used_now))
        self.assertEqual(actions, [("a", False), ("a", False), ("b", True), ("a", False), ("a", False)])

    def test_first_disagreement_ignores_results(self):
        events = [dict(decision=i, anchor_id="a", ranking={"selected":"a" if i < 3 else "b"}, success=False) for i in range(5)]
        self.assertEqual(first_disagreement(events), 3)
        for event in events:
            event["success"] = True
        self.assertEqual(first_disagreement(events), 3)
        self.assertIsNone(first_disagreement(events[:3]))
        with self.assertRaises(ValueError):
            first_disagreement(events[1:])

    def test_h32_zero_extension_and_unknown(self):
        result = dict(conflicts=[4, 3, 2, 0], success=True)
        self.assertTrue(horizon_outcome(result, 1)["success"])
        result = dict(conflicts=[4, 3, 2, 1], success=False)
        self.assertFalse(horizon_outcome(result, 1)["known"])
        self.assertEqual(horizon_outcome(result, 1, 2), dict(known=True, success=False, final_conflicts=1))
        self.assertIsNone(horizon_outcome(result, None))
        with self.assertRaises(ValueError):
            horizon_outcome(result, 3)

    def test_common_prefix_successor_uses_before_state(self):
        event = dict(decision=0, before="hash", pool=[], subset=[], anchor_id="a", features=[],
                     temperature=1, uniform=.1, action={"agents":[1]}, ranking={"scores":{"a":1}}, delta={"x":2})
        frozen = copy.deepcopy(event)
        frozen["action"] = {"agents":[2]}
        with patch("scripts.run_sa_path_quality.apply_state_delta", side_effect=lambda s,d:s | d), \
                patch("scripts.run_sa_path_quality.state_fingerprint", side_effect=json_fingerprint):
            check_prefix(event, {"x":1, "y":0}, {"x":2, "y":0}, event, frozen, True)
            with self.assertRaises(ValueError):
                check_prefix(event, {"x":1, "y":0}, {"x":2, "y":99}, event, frozen, True)
            with self.assertRaises(ValueError):
                check_prefix(event, {"x":1, "y":0}, {"x":2, "y":0}, event, frozen, False)
            changed = dict(frozen, features=[1])
            with self.assertRaises(ValueError):
                check_prefix(event, {"x":1}, {"x":2}, event, changed, True)

    def test_trace_prefix_does_not_read_future(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder/"trace.jsonl").write_text('{"decision":0}\n{"decision":1}\ninvalid future\n', encoding="utf8")
            self.assertEqual(trace_prefix(folder, 1), [{"decision":0}, {"decision":1}])

    def test_grouped_bootstrap_and_censor_bounds(self):
        rows = [dict(map_id="a", success=dict(single=1, paired=0), unknown=dict(single=0, paired=0)) for _ in range(5)]
        rows += [dict(map_id="b", success=dict(single=0, paired=1), unknown=dict(single=0, paired=0))]
        result = completion_contrast(rows, "paired", 1000, 123)
        self.assertEqual(result["observed_delta"], 0)
        self.assertEqual(result["map_wins"], 1)
        self.assertEqual(result, completion_contrast(rows[::-1], "paired", 1000, 123))
        rows[-1]["unknown"]["single"] = 1
        censored = completion_contrast(rows, "paired", 1000, 123)
        self.assertIsNone(censored["ci95"])
        self.assertEqual(censored["sample_difference_bounds"], [0, .5])

    def test_interpretation_never_promotes(self):
        base = dict(censored=False, observed_delta=.1, ci95=[.01, .2])
        self.assertEqual(interpretation(dict(paired=base, frozen=base)), "limited_evidence_repeated_takeover_hurts")
        self.assertEqual(interpretation(dict(paired=dict(base, observed_delta=0), frozen=base)), "single_intervention_does_not_resolve_failure")
        self.assertEqual(interpretation(dict(paired=dict(base, censored=True), frozen=base)), "incomplete_evidence_resource_censoring")

    def test_fixed_scope_and_no_training(self):
        cfg = read_json(Path(__file__).resolve().parents[2]/"configs/sa_intervention_frequency.json")
        self.assertEqual((cfg["expected_pairs"], cfg["expected_maps"], cfg["workers"]), (32, 8, 20))
        self.assertFalse(cfg["training"] or cfg["formal_ttf"] or cfg["automatic_promotion"])
        self.assertEqual(cfg["selection"], "first_model_disagreement_then_frozen")


if __name__ == "__main__":
    unittest.main()
