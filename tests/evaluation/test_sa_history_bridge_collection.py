from pathlib import Path
import tempfile
import unittest

from scripts.collect_sa_history_candidate_bridge import (randomization,frozen_index,stability,expected_files,
    check_root_receipt,summarize)
from scripts.audit_sa_history_information import atomic


class BridgeCollectionTests(unittest.TestCase):
    def test_paired_streams_independent_of_candidate(self):
        cfg=dict(seed=20260918)
        self.assertEqual(randomization("root",2,81,cfg),randomization("root",2,81,cfg))
        self.assertNotEqual(randomization("root",2,81,cfg),randomization("root",3,81,cfg))
        self.assertNotEqual(randomization("root",2,81,cfg),randomization("root",2,82,cfg))

    def test_frozen_tie_uses_hash_and_twelve_decimals(self):
        pool=[dict(candidate_id="b",score=.7+1e-15),dict(candidate_id="a",score=.7)]
        self.assertEqual(frozen_index(pool),1)

    def test_no_false_stability_from_all_ties(self):
        s=stability({"a":[0]*8,"b":[0]*8,"c":[0]*8})
        self.assertEqual(s["best_jaccard"],1)
        self.assertFalse(any(p["strict_both"] for p in s["pairs"]))

    def test_half_direction_reversal(self):
        s=stability({"a":[1]*4+[0]*4,"b":[0]*4+[1]*4})
        self.assertEqual(s["best_jaccard"],0)
        self.assertTrue(s["pairs"][0]["strict_both"])
        self.assertFalse(s["pairs"][0]["agree"])

    def test_missing_trial_rejected(self):
        with self.assertRaises(ValueError): stability({"a":[0]*7,"b":[0]*8})

    def test_receipt_cannot_omit_control(self):
        entry=dict(selected=["a","b","c"])
        cfg=dict(trials=8)
        self.assertEqual(len(expected_files(entry,cfg)),25)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            atomic(folder/"receipt.json",dict(binding="x",files={}))
            with self.assertRaisesRegex(ValueError,"receipt incomplete"):
                check_root_receipt(folder,entry,dict(binding="x",config=cfg))

    def test_all_ties_stop_gate_not_success(self):
        states=[]
        for i in range(8):
            rates={p:{t:0 for t in ("sustained_progress","completion")} for p in ("frozen","ordered","temporal_bag","uniform")}
            states.append(dict(id=str(i),map_id=str(i),censored=[],policy_rates=rates,
                stability={t:stability({"a":[0]*8,"b":[0]*8,"c":[0]*8}) for t in ("sustained_progress","completion")}))
        report=summarize(states,dict(states=8,bootstrap=50,seed=1))
        self.assertFalse(report["summary"]["sustained_progress"]["stability_passed"])
        self.assertTrue(report["no_policy_promotion"])


if __name__=="__main__": unittest.main()
