from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import recover_sa_onpolicy as recover


def event(n=0):
    result = {k: n for k in ("decision", "policy_sha256", "before", "pool", "proposal_order", "candidate_ids", "anchor_id",
        "features", "probabilities", "selected_id", "selection_draw", "behavior_log_probability", "action", "temperature", "uniform", "delta")}
    result["metrics"] = dict(acceptance_evaluated=True, pp_failure_reason="none", pp_rolled_back=False)
    return result


class RecoveryTests(unittest.TestCase):
    def test_only_protection_and_output_change(self):
        cfg = recover.run.read_json(recover.ROOT/recover.CONFIG)
        source = dict(binding="original", config={"output": "old", "workers": 20, "stream_seed": 7},
                      proposal=dict(max_decisions=256, node_budget=25000000, pp_safety_seconds=20.,
                                    episode_safety_seconds=300., process_fuse_seconds=360., learning_rate=.001))
        before = deepcopy(source)
        p = recover.runtime_plan(source, cfg, "receipt")
        self.assertEqual(source, before)
        self.assertEqual(p["binding"], source["binding"])
        self.assertEqual({k for k in source["proposal"] if source["proposal"][k] != p["proposal"][k]},
                         {"episode_safety_seconds", "process_fuse_seconds"})
        self.assertEqual(p["config"]["stream_seed"], 7)
        for key, value in (("change_normalizer", True), ("maximum_updates_this_stage", 2), ("episode_safety_seconds", 1200)):
            with self.assertRaises(ValueError):
                recover.runtime_plan(source, dict(cfg, **{key: value}), "x")
        source["proposal"]["node_budget"] += 1
        with self.assertRaises(ValueError):
            recover.runtime_plan(source, cfg, "x")

    def test_prefix_allows_only_old_final_incomplete_outcome_to_differ(self):
        old = [event(0), event(1)]
        old[-1]["metrics"].update(acceptance_evaluated=False, pp_failure_reason="time_limit", pp_rolled_back=True)
        new = [event(0), event(1), event(2)]
        new[1]["delta"] = "completed PP"
        self.assertEqual(recover.compare_prefix(old, new)["exact_completed_steps"], 1)
        for index, key in ((0, "delta"), (1, "before"), (1, "features"), (1, "action"), (1, "probabilities")):
            wrong = deepcopy(new)
            wrong[index][key] = "changed"
            with self.assertRaises(ValueError):
                recover.compare_prefix(old, wrong)
        with self.assertRaises(ValueError):
            recover.compare_prefix(old, new[:1])
        with self.assertRaises(ValueError):
            recover.compare_prefix([event(0)], [event(0)])
        with self.assertRaises(ValueError):
            recover.compare_prefix(old+[event(2)], new)

    def test_strict_lock_never_probes_or_kills_another_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control, out = root/"control", root/"out"
            with patch("os.kill", side_effect=AssertionError("must not probe")):
                with recover.strict_lock(out, "a", "test", control):
                    owner = recover.run.read_json(control/"active.lock")
                    with self.assertRaises(FileExistsError):
                        with recover.strict_lock(root/"other", "b", "test", control):
                            self.fail("overlap")
                    self.assertEqual(recover.run.read_json(control/"active.lock"), owner)
                self.assertFalse((control/"active.lock").exists())
                self.assertFalse((out/".collection.lock").exists())

    def test_result_reference_rejects_censor_and_changed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/"old"
            recover.run.once(folder/"result.json", recover.run.sealed(dict(binding="b", job_id="j", initial_fingerprint="i",
                 policy_sha256="p", status="ok", files={})))
            entry = dict(folder="old", origin="reused", job={"job_id":"j"}, expected_initial="i",
                         original_sha256=recover.run.sha256_file(folder/"result.json"))
            with patch.object(recover, "ROOT", root):
                recover.result_entry({"source_policy_sha256":"p"}, {"binding":"b"}, entry)
                with self.assertRaises(ValueError):
                    recover.result_entry({"source_policy_sha256":"p"}, {"binding":"b"}, dict(entry, original_sha256="bad"))
                recover.run.write_json(folder/"result.json", recover.run.sealed(dict(binding="b", job_id="j", initial_fingerprint="i",
                    policy_sha256="p", status="censored", files={})))
                with self.assertRaises(ValueError):
                    recover.result_entry({"source_policy_sha256":"p"}, {"binding":"b"}, entry)


if __name__ == "__main__":
    unittest.main()
