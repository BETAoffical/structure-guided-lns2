from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from tests.native_support import isolated_sa_native
from unittest.mock import patch

import numpy as np

from experiments._common import read_json
from experiments.sa_onpolicy_actor import (NumpyActor, initial_bundle, select_with_draw, torch_actor,
    torch_distribution, update_once, validate_bundle, vectorize)
from scripts import run_sa_onpolicy as run


def plan_fixture():
    cfg = read_json(run.ROOT / run.CONFIG)
    design = read_json(run.ROOT / cfg["design"])
    return dict(config=cfg, proposal=design["proposal"], binding="b" * 64,
                split=dict(train_maps=[f"m{i}" for i in range(6)]),
                cases=[dict(task_id=f"m{i}-t{t}", map_id=f"m{i}", solver_seeds=[227, 229]) for i in range(8) for t in range(2)])


def features():
    return [{f"f{i:03d}": float(k + i / 100) for i in range(129)} for k in (-1, 1)]


class SchedulingTests(unittest.TestCase):
    def test_schedule_is_map_disjoint_and_bounded(self):
        p = plan_fixture()
        train = run.jobs_for(p, "train-0", 0)
        held = run.jobs_for(p, "evaluation", 2)
        self.assertEqual((len(train), len(held)), (96, 64))
        self.assertEqual(len({j["job_id"] for j in train + held}), 160)
        self.assertFalse({j["case"]["map_id"] for j in train} & {j["case"]["map_id"] for j in held})
        for phase, iteration in (("train-2", 2), ("train-0", 1), ("evaluation", 0)):
            with self.assertRaises(ValueError):
                run.jobs_for(p, phase, iteration)

    def test_rng_is_independent_of_arm_and_distinct_by_purpose(self):
        p = plan_fixture()
        a = [run.stream_draw(p, "evaluation", "case", 0, d, "select") for d in range(8)]
        self.assertEqual(a, [run.stream_draw(p, "evaluation", "case", 0, d, "select") for d in range(8)])
        self.assertNotEqual(a, [run.stream_draw(p, "evaluation", "case", 0, d, "accept") for d in range(8)])
        self.assertNotEqual(a, [run.stream_draw(p, "evaluation", "case", 1, d, "select") for d in range(8)])
        self.assertNotEqual(a, [run.stream_draw(p, "train-0", "case", 0, d, "select") for d in range(8)])

    def test_categorical_sampling_is_order_independent(self):
        for draw, expected in ((0., "a"), (.3, "b"), (.999999, "b")):
            self.assertEqual(run.select_with_draw({"b": .7, "a": .3}, draw), expected)
        for draw in (-1., 1., float("nan")):
            with self.assertRaises(ValueError):
                select_with_draw({"a": 1.}, draw)

    def test_safe_batch_stop_resume_and_atomic_completion(self):
        p = plan_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / p["config"]["output"]
            calls = []
            def jobs(worker, batch, *args, **kwargs):
                calls.append(len(batch))
                rows = []
                for job in batch:
                    folder = run.folder_for(out, "qualify", job)
                    run.once(folder / "result.json", run.sealed(dict(binding=p["binding"], status="ok", files={})))
                    row = dict(status="ok", job_id=job["job_id"])
                    kwargs["on_result"](row)
                    rows.append(row)
                if len(calls) == 1:
                    run.write_json(out / "STOP_AFTER_BATCH", {})
                return rows
            with patch.object(run, "ROOT", root), patch.object(run, "verify", return_value=(p, out)), \
                 patch("experiments.repair_collection._run_jobs", side_effect=jobs):
                self.assertEqual(run.collect("qualify")["status"], "paused")
                self.assertEqual(calls, [20])
                self.assertFalse((out / "qualify.complete.json").exists())
                self.assertEqual(run.collect("qualify", resume=True)["status"], "completed")
                self.assertEqual(calls, [20, 12])
                self.assertEqual(run.collect("qualify", resume=True)["status"], "completed")
                self.assertEqual(calls, [20, 12])

    def test_recorded_external_timeout_never_silently_retries(self):
        p = plan_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp) / p["config"]["output"]
            j = run.jobs_for(p, "qualify")[0]
            run.once(out / "failures" / (j["job_id"] + ".json"), {"status": "censored"})
            with patch.object(run, "ROOT", root), patch.object(run, "verify", return_value=(p, out)), \
                 patch("experiments.repair_collection._run_jobs") as mocked:
                with self.assertRaisesRegex(ValueError, "explicit recovery"):
                    run.collect("qualify", resume=True)
                mocked.assert_not_called()

    def test_result_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            run.once(folder / "state.json", {"x": 1})
            run.once(folder / "result.json", run.sealed(dict(binding="p", files={"state.json": run.sha256_file(folder / "state.json")})))
            run.result_read(folder, {"binding": "p"})
            run.write_json(folder / "state.json", {"x": 2})
            with self.assertRaises(ValueError):
                run.result_read(folder, {"binding": "p"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "existing Windows torch environment")
class ActorTests(unittest.TestCase):
    def test_initialization_is_deterministic_and_zero_residual_is_control(self):
        b = initial_bundle(features(), 3, "p")
        self.assertEqual(b, initial_bundle(features(), 3, "p"))
        p = NumpyActor(b).probabilities(["a", "b"], "a", features())
        self.assertAlmostEqual(p["a"], .95)
        self.assertEqual(p["b"], .05)

    def test_torch_numpy_parity_with_nonzero_residual(self):
        b = initial_bundle(features(), 5, "p")
        b["w2"] = [[.05] * 32]
        native = NumpyActor(b).probabilities(["a", "b"], "a", features())
        torch_p = torch_distribution(torch_actor(b), b, ["a", "b"], "a", features()).probs.detach().numpy()
        np.testing.assert_allclose(torch_p, list(native.values()), atol=1e-12, rtol=0)
        for draw in (.001, .5, .96, .999):
            self.assertEqual(select_with_draw(native, draw), select_with_draw(dict(zip(native, torch_p)), draw))

    def test_one_autograd_update_replays_probabilities_and_changes_weights(self):
        b = initial_bundle(features(), 8, "p")
        actor = NumpyActor(b)
        event = dict(candidate_ids=["a", "b"], anchor_id="a", features=features(), selected_id="b",
                     policy_sha256=actor.sha, probabilities=actor.probabilities(["a", "b"], "a", features()))
        updated, metrics = update_once(b, [dict(episode_id="e", coefficient=1.)], lambda _: [event], "data")
        self.assertEqual(updated["parent_policy"], actor.sha)
        self.assertGreater(metrics["parameter_l2_change"], 0)
        self.assertLessEqual(metrics["parameter_l2_change"], .001000000001)
        self.assertGreater(NumpyActor(updated).probabilities(["a", "b"], "a", features())["b"], event["probabilities"]["b"])
        bad = deepcopy(event)
        bad["policy_sha256"] = "old"
        with self.assertRaisesRegex(ValueError, "stale"):
            update_once(b, [dict(episode_id="e", coefficient=1.)], lambda _: [bad], "data")

    def test_feature_schema_and_weight_shapes_fail_closed(self):
        b = initial_bundle(features(), 3, "p")
        fs = features()
        fs[0]["future.success"] = 1
        with self.assertRaises(ValueError):
            vectorize(fs, b["feature_names"])
        b["scale"][0] = 0
        with self.assertRaises(ValueError):
            validate_bundle(b)


class NativeIntegrationTests(unittest.TestCase):
    @isolated_sa_native
    def test_real_pool_state_streams_and_censoring(self):
        import lns2_env
        from experiments.online_feature_engine import OnlineFeatureEngine
        from experiments.sa_history_selector import History
        from scripts import run_sa_path_quality as q
        p = plan_fixture()
        p["proposal"] = dict(p["proposal"], max_decisions=8)
        self.assertEqual(q.native_identity()["sha256"], p["config"]["native_sha256"])
        proposal = read_json(run.ROOT / "build/sa-pressure-first-feasible-v1/registration.json")["template"]["proposal"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, s = root / "tiny.map", root / "tiny.scen"
            m.write_text("type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n", encoding="utf8")
            s.write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n", encoding="utf8")
            def reset():
                env = lns2_env.LNS2RepairEnv(str(m), str(s), 2)
                state = q._plain(env.reset_paths([[0, 1, 2, 3], [3, 2, 1, 0]], seed=11))
                return env, state
            ctx = dict(case_id="tiny-s11", task_id="tiny", solver_seed=11, proposal=proposal)
            env, state = reset()
            i, pool = q.SingleFullCheckPool(ctx).select(env, state, 0)
            fs = run.features_for(state, pool, OnlineFeatureEngine(state, backend="native"), History(state),
                                  q.temperature(0), q.state_fingerprint(state))
            names = sorted(run.budget_features(fs[0], 0, 0, p["proposal"]))
            self.assertEqual(len(names), 129)
            b = dict(schema="lns2.sa.onpolicy_actor.v1", binding=p["binding"], iteration=0,
                     feature_names=names, mean=[0.] * 129, scale=[1.] * 129,
                     w1=[[0.] * 129 for _ in range(32)], b1=[0.] * 32, w2=[[0.] * 32], b2=[0.],
                     epsilon=.1, residual_bound=2.)
            results = []
            for i, arm in enumerate([*p["proposal"]["evaluation_arms"], "trained_actor"]):
                env, state = reset()
                job = dict(plan=p, case=dict(map_id="tiny"), arm=arm, phase="micro", pair_id="tiny-s11",
                           split="train", replica=0, job_id=str(i))
                bundle = b if arm in {"trained_actor", "untrained_exploration"} else None
                result = run.episode_loop(job, q, env, state, ctx, root / str(i), bundle)
                results.append(result)
                run.result_read(root / str(i), p)
            self.assertEqual(len({r["initial_fingerprint"] for r in results}), 1)
            self.assertEqual(results[2]["final_fingerprint"], results[3]["final_fingerprint"])
            self.assertEqual(results[3]["final_fingerprint"], results[4]["final_fingerprint"])
            for i in (2, 3, 4):
                events = list(run.trace_read(root / str(i)))
                self.assertTrue(events)
                self.assertTrue(all(e["policy_sha256"] == validate_bundle(b) for e in events))
            env, state = reset()
            job["plan"] = dict(p, proposal=dict(p["proposal"], episode_safety_seconds=0))
            result = run.episode_loop(job, q, env, state, ctx, root / "timeout", b)
            self.assertEqual((result["status"], result["decisions"]), ("censored", 0))


if __name__ == "__main__":
    unittest.main()
