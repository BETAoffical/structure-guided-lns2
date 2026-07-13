from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    import lns2_env  # noqa: F401
except ModuleNotFoundError:
    lns2_env = None

from experiments.repair_collection import (
    _baseline_worker,
    _counterfactual_worker,
    _read_jsonl,
    _write_jsonl,
)


@unittest.skipUnless(
    lns2_env is not None and "LNS2_TEST_MAP" in os.environ,
    "the native repair collector is tested by Linux CTest",
)
class NativeRepairCollectionTests(unittest.TestCase):
    def test_baseline_counterfactual_replay_and_resume(self) -> None:
        map_path = Path(os.environ["LNS2_TEST_MAP"]).resolve()
        scenario_path = Path(os.environ["LNS2_TEST_SCEN"]).resolve()
        row = {
            "split": "train",
            "map_id": "native-random-map",
            "task_id": "native-random-200",
            "map_file": str(map_path),
            "scenario_file": str(scenario_path),
            "layout_mode": "random",
            "layout_variant": None,
            "scenario_type": "benchmark",
            "task_variant": "benchmark",
            "agent_count": 200,
            "topology_metrics": {},
            "dominant_flow_ratio": 0.0,
            "hotspot_skew": 0.0,
            "required_bottleneck_crossing_ratio": 0.0,
            "mean_shortest_distance": 0.0,
        }
        environment = {
            "time_limit": 30.0,
            "max_repair_iterations": 4,
            "neighborhood_size": 8,
            "replan_algorithm": "PP",
            "use_sipp": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            baseline = _baseline_worker(
                {
                    "dataset_root": directory,
                    "output_root": directory,
                    "row": row,
                    "solver_seed": 29,
                    "policy": "official_adaptive",
                    "environment": environment,
                    "run_fingerprint": "native-test-run",
                    "resume": False,
                }
            )
            self.assertEqual(baseline["status"], "ok", baseline.get("error"))
            self.assertTrue(baseline["summary"]["repairable"])
            result = _counterfactual_worker(
                {
                    "dataset_root": directory,
                    "output_root": directory,
                    "row": row,
                    "manifest": baseline,
                    "environment": environment,
                    "counterfactual": {
                        "max_states_per_episode": 1,
                        "max_seed_agents": 1,
                        "heuristics": ["collision"],
                        "neighborhood_sizes": [8],
                        "trials": 1,
                        "horizons": [1],
                    },
                    "run_fingerprint": "native-test-run",
                    "resume": False,
                }
            )
            self.assertEqual(result["status"], "ok", result.get("error"))
            self.assertEqual(result["state_count"], 1)
            self.assertEqual(result["outcome_count"], 1)
            self.assertEqual(result["error_count"], 0)
            outcomes = _read_jsonl(Path(directory) / result["outcomes_file"])
            self.assertEqual(len(outcomes), 1)
            self.assertGreaterEqual(outcomes[0]["candidate_action"]["random_seed"], 0)
            self.assertEqual(outcomes[0]["horizon_outcomes"][0]["horizon"], 1)

            resumed = _counterfactual_worker(
                {
                    "dataset_root": directory,
                    "output_root": directory,
                    "row": row,
                    "manifest": baseline,
                    "environment": environment,
                    "counterfactual": {},
                    "run_fingerprint": "native-test-run",
                    "resume": True,
                }
            )
            self.assertEqual(resumed["status"], "resumed")

            bad_manifest = dict(baseline)
            bad_manifest["episode_id"] = baseline["episode_id"] + "__bad_replay"
            original_trace = Path(directory) / baseline["trace_file"]
            bad_trace = Path(directory) / "bad-replay.jsonl"
            bad_events = _read_jsonl(original_trace)
            bad_events[0]["state"]["agents"][0]["path"][-1] += 1
            _write_jsonl(bad_trace, bad_events)
            bad_manifest["trace_file"] = bad_trace.relative_to(directory).as_posix()
            mismatch = _counterfactual_worker(
                {
                    "dataset_root": directory,
                    "output_root": directory,
                    "row": row,
                    "manifest": bad_manifest,
                    "environment": environment,
                    "counterfactual": {
                        "max_states_per_episode": 1,
                        "max_seed_agents": 1,
                        "heuristics": ["collision"],
                        "neighborhood_sizes": [8],
                        "trials": 1,
                        "horizons": [1],
                    },
                    "run_fingerprint": "native-test-run",
                    "resume": False,
                }
            )
            self.assertEqual(mismatch["status"], "error")
            self.assertEqual(mismatch["outcome_count"], 0)
            self.assertEqual(mismatch["error_count"], 1)

            zero_conflict_row = dict(row)
            zero_conflict_row["task_id"] = "native-random-24"
            zero_conflict_row["agent_count"] = 24
            zero_conflict = _baseline_worker(
                {
                    "dataset_root": directory,
                    "output_root": directory,
                    "row": zero_conflict_row,
                    "solver_seed": 0,
                    "policy": "official_adaptive",
                    "environment": environment,
                    "run_fingerprint": "native-test-run",
                    "resume": False,
                }
            )
            self.assertEqual(zero_conflict["status"], "ok")
            self.assertFalse(zero_conflict["summary"]["repairable"])
            self.assertEqual(zero_conflict["summary"]["repair_iterations"], 0)


if __name__ == "__main__":
    unittest.main()
