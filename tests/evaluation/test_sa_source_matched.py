import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_sa_source_matched import (
    check_config, check_root_identity, grouped, json_snapshot, select, signature, verify_inputs, work_budget,
)


def fixture():
    cfg = dict(sampling_seed=181, maps=2, roots_per_map=2, actual_size=16,
               allowed_families=["random:16"], planned_trials=8,
               planned_horizon=32, planned_trial_timeout_seconds=180, planned_workers=20)
    rows = []
    for m in range(2):
        for i in range(3):
            candidates = [dict(candidate_id=cid, agents=list(range(start, start + 16)),
                actual_size=16, selection_families=[family]) for cid, start, family in (
                    ("anchor", 0, "structpool-conflict-component:16"),
                    ("a", 4, "random:16"), ("b", 8, "random:16"))]
            rows.append(dict(state_id=f"m{m}-r{i}", map_id=f"m{m}", episode=f"m{m}-e{i}",
                decision=4 + i, anchor_id="anchor", pool=candidates))
    return rows, cfg


class SourceMatchedTests(unittest.TestCase):
    def test_snapshot_pins_bytes_before_a_later_config_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "config.json"
            p.write_text('{"n": 8}', encoding="utf-8")
            value, digest = json_snapshot(p)
            p.write_text('{"n": 16}', encoding="utf-8")
            self.assertEqual(value, {"n": 8})
            with patch("scripts.prepare_sa_source_matched.ROOT", root), self.assertRaises(ValueError):
                verify_inputs({"config.json": digest})

    def test_registered_helper_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "helper.json"
            p.write_text('{}', encoding="utf-8")
            _, digest = json_snapshot(p)
            with patch("scripts.prepare_sa_source_matched.ROOT", root):
                verify_inputs({"helper.json": digest})
                p.write_text('{"changed":true}', encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify_inputs({"helper.json": digest})

    def identity_fixture(self):
        r = fixture()[0][0]
        state = {k: r[k] for k in ("state_id", "map_id", "episode", "decision", "anchor_id")}
        state.update(state_fingerprint="fp", candidates=[dict(candidate_id=c["candidate_id"], agents=list(c["agents"]),
                     features={"proposal.actual_size": 16}) for c in r["pool"]])
        root = dict(state_fingerprint="fp", old_selected_id="anchor", candidates=copy.deepcopy(r["pool"]),
                    source=dict(id=r["state_id"], map_id=r["map_id"], anchor_id="anchor", state_fingerprint="fp",
                                item=dict(job_id=r["episode"]), decision=r["decision"], selected=[c["candidate_id"] for c in r["pool"]]),
                    control_event=dict(decision=r["decision"], pool=copy.deepcopy(r["pool"]), selected_index=0,
                                       action=dict(agents=list(r["pool"][0]["agents"]))))
        return state, root

    def test_cross_artifact_membership_and_source_are_checked(self):
        state, root = self.identity_fixture()
        check_root_identity(state, root)
        for mutation in (
            lambda s, r: s.update(episode="wrong"),
            lambda s, r: r["source"].update(decision=88),
            lambda s, r: s["candidates"][1]["agents"].__setitem__(0, 999),
            lambda s, r: r["candidates"][1].update(selection_families=["target:16"]),
            lambda s, r: r["control_event"].update(selected_index=1),
        ):
            s, r = self.identity_fixture()
            mutation(s, r)
            with self.assertRaises(ValueError):
                check_root_identity(s, r)

    def test_protocol_and_output_boundaries(self):
        root = Path(__file__).resolve().parents[2]
        cfg = json.loads((root / "configs/sa_source_matched_preparation.json").read_text())
        check_config(cfg)
        for change in (dict(collection_ready=True), dict(planned_horizon=64),
                       dict(output="../outside"), dict(output="build"), dict(planned_workers=21)):
            with self.assertRaises(ValueError):
                check_config(cfg | change)

    def test_same_size_is_not_same_source(self):
        rows, _ = fixture()
        groups = grouped(rows[0]["pool"], 16)
        self.assertEqual(len(groups[("random:16",)]), 2)
        self.assertEqual(len(groups[("structpool-conflict-component:16",)]), 1)

    def test_selection_is_outcome_score_and_feature_blind(self):
        rows, cfg = fixture()
        expected = select(rows, cfg)
        for r in rows:
            r.update(outcomes=[999], completed=True, features={"coverage": 1.0})
            for c in r["pool"]:
                c.update(score=123456, trials=[True] * 8, predicted_reward=-123)
        self.assertEqual(select(rows, cfg), expected)

    def test_order_invariant_and_repeatable(self):
        rows, cfg = fixture()
        expected = select(rows, cfg)
        rows.reverse()
        for r in rows:
            r["pool"].reverse()
        self.assertEqual(expected, select(rows, cfg))
        self.assertEqual(expected, select(rows, cfg))

    def test_distinct_source_episodes_and_maps(self):
        rows, cfg = fixture()
        rows[1]["episode"] = rows[0]["episode"]
        chosen = select(rows, cfg)
        for m in ("m0", "m1"):
            selected = [r for r in chosen if r["map_id"] == m]
            self.assertEqual(len(selected), 2)
            self.assertEqual(len({r["episode"] for r in selected}), 2)

    def test_insufficient_episodes_not_resampled(self):
        rows, cfg = fixture()
        for r in rows:
            r["episode"] = "one-episode"
        with self.assertRaisesRegex(ValueError, "distinct source"):
            select(rows, cfg)

    def test_mixed_sources_and_wrong_actual_size_not_silently_matched(self):
        rows, cfg = fixture()
        rows[0]["pool"][1]["selection_families"].append("target:16")
        with self.assertRaisesRegex(ValueError, "pure-family"):
            select(rows, cfg)
        rows, cfg = fixture()
        rows[0]["pool"][1]["actual_size"] = 15
        with self.assertRaisesRegex(ValueError, "actual size"):
            select(rows, cfg)

    def test_anchor_deduplicated_when_in_pair(self):
        rows, cfg = fixture()
        for r in rows:
            r["anchor_id"] = "a"
        chosen = select(rows, cfg)
        self.assertTrue(all(len(r["candidates"]) == 2 for r in chosen))

    def test_budget_includes_prefix_and_never_claims_execution(self):
        rows, cfg = fixture()
        chosen = select(rows, cfg)
        b = work_budget(chosen, cfg)
        self.assertEqual(b["trial_jobs"], 4 * 3 * 8)
        self.assertEqual(b["maximum_rollout_repairs"], 96 * 32)
        self.assertEqual(b["maximum_prefix_replay_repairs"], sum(r["decision"] for r in chosen) * 3 * 8)
        self.assertEqual(b["independent_resets"], 96)
        self.assertEqual(b["executed_jobs"], 0)
        self.assertIsNone(b["predicted_runtime_seconds"])

    def test_duplicate_membership_and_ids_rejected(self):
        rows, _ = fixture()
        pool = rows[0]["pool"]
        with self.assertRaisesRegex(ValueError, "candidate ID"):
            grouped(pool + [pool[0]], 16)
        extra = copy.deepcopy(pool[0])
        extra["candidate_id"] = "duplicate-physical-set"
        with self.assertRaisesRegex(ValueError, "membership"):
            grouped(pool + [extra], 16)

    def test_source_order_is_canonical(self):
        c = dict(candidate_id="c", agents=list(range(16)), actual_size=16,
                 selection_families=["target:16", "collision:16"])
        self.assertEqual(signature(c), (16, ("collision:16", "target:16")))


if __name__ == "__main__":
    unittest.main()
