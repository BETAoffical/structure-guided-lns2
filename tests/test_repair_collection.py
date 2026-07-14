from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path

from experiments.repair_collection import (
    CollectionLockError,
    _AtomicProcessLock,
    _counterfactual_source_eligible,
    _counterfactual_source_reason,
    _horizon_outcomes,
    _prepare_run,
    _run_jobs,
    _select_task_rows,
    _validate_config,
    candidate_actions,
    select_seed_agents,
    state_fingerprint,
    recover_counterfactual_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scheduler_worker(job: dict) -> dict:
    time.sleep(float(job.get("sleep", 0.0)))
    return {
        "schema_version": 1,
        "task_id": job["row"]["task_id"],
        "solver_seed": job["solver_seed"],
        "status": "ok",
        "error": None,
    }


def sample_state() -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": False,
        "done": False,
        "iteration": 3,
        "rows": 2,
        "cols": 3,
        "sum_of_costs": 12,
        "num_of_colliding_pairs": 3,
        "runtime": 0.5,
        "context": {"layout_mode": "ignored"},
        "low_level": {
            "expanded": 10,
            "generated": 20,
            "reopened": 1,
            "runs": 4,
        },
        "obstacles": [0, 0, 1, 0, 0, 0],
        "conflict_edges": [[0, 1], [0, 2], [2, 3]],
        "agents": [
            {
                "id": 0,
                "start": 0,
                "goal": 5,
                "path_cost": 4,
                "shortest_path_cost": 3,
                "delay": 1,
                "conflict_degree": 2,
                "path": [0, 1, 4, 5],
            },
            {
                "id": 1,
                "start": 1,
                "goal": 4,
                "path_cost": 3,
                "shortest_path_cost": 2,
                "delay": 1,
                "conflict_degree": 1,
                "path": [1, 0, 3, 4],
            },
            {
                "id": 2,
                "start": 3,
                "goal": 2,
                "path_cost": 5,
                "shortest_path_cost": 2,
                "delay": 3,
                "conflict_degree": 2,
                "path": [3, 4, 3, 0, 1, 2],
            },
            {
                "id": 3,
                "start": 5,
                "goal": 0,
                "path_cost": 4,
                "shortest_path_cost": 3,
                "delay": 1,
                "conflict_degree": 1,
                "path": [5, 4, 3, 0],
            },
        ],
    }


class RepairCollectionTests(unittest.TestCase):
    def test_atomic_lock_rejects_live_owner_and_recovers_stale_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.lock"
            owner = {
                "run_id": "first",
                "pid": os.getpid(),
                "process_start_token": None,
                "host": socket.gethostname(),
                "output_root": directory,
            }
            first = _AtomicProcessLock(path, owner)
            first.acquire()
            with self.assertRaisesRegex(CollectionLockError, "active"):
                _AtomicProcessLock(path, {**owner, "run_id": "second"}).acquire()
            first.release()

            path.write_text(
                json.dumps(
                    {
                        **owner,
                        "run_id": "stale",
                        "pid": 2**31 - 1,
                    }
                ),
                encoding="utf-8",
            )
            recovered = _AtomicProcessLock(path, {**owner, "run_id": "recovered"})
            recovered.acquire()
            self.assertTrue(list(path.parent.glob("active.lock.stale-*")))
            recovered.release()

    def test_scheduler_reports_timeout_and_updates_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [
                {
                    "row": {
                        "split": "train",
                        "map_id": "map",
                        "task_id": "slow",
                        "agent_count": 10,
                    },
                    "solver_seed": 0,
                    "sleep": 0.3,
                }
            ]
            results = _run_jobs(
                _scheduler_worker,
                jobs,
                1,
                phase="timeout-test",
                output_root=root,
                run_fingerprint="run",
                timeout_seconds=0.05,
            )
            self.assertEqual(results[0]["status"], "timeout")
            progress = json.loads(
                (root / "collection_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(progress["timeout_jobs"], 1)
            self.assertEqual(progress["completed_jobs"], 1)

    def test_scheduler_emits_each_result_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = []
            jobs = [
                {
                    "row": {
                        "split": "train",
                        "map_id": "map",
                        "task_id": task_id,
                        "agent_count": 10,
                    },
                    "solver_seed": 0,
                    "sleep": delay,
                }
                for task_id, delay in (("first", 0.01), ("second", 0.1))
            ]
            results = _run_jobs(
                _scheduler_worker,
                jobs,
                2,
                phase="incremental-test",
                output_root=root,
                run_fingerprint="run",
                on_result=lambda row: observed.append(row["task_id"]),
            )
            self.assertEqual(len(results), 2)
            self.assertEqual(observed, ["first", "second"])

    def test_task_filter_rejects_missing_and_preserves_dataset_order(self) -> None:
        rows = [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]
        self.assertEqual(
            [row["task_id"] for row in _select_task_rows(rows, ["c", "a"])],
            ["a", "c"],
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            _select_task_rows(rows, ["missing"])

    def test_hardening_smoke_config_has_a_bounded_workload(self) -> None:
        config = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "repair_collection_hardening_smoke.json"
            ).read_text(encoding="utf-8")
        )
        _validate_config(config)
        counterfactual = config["counterfactual"]
        self.assertEqual(counterfactual["maximum_agent_count"], 400)
        self.assertEqual(counterfactual["episode_wall_time_limit_seconds"], 300)
        self.assertEqual(
            counterfactual["max_states_per_episode"]
            * counterfactual["max_seed_agents"]
            * len(counterfactual["heuristics"])
            * len(counterfactual["neighborhood_sizes"])
            * counterfactual["trials"],
            24,
        )

    def test_counterfactual_source_filter_bounds_extreme_conflict_episodes(self) -> None:
        configuration = {
            "minimum_initial_conflicts": 2,
            "maximum_initial_conflicts": 200,
            "require_source_success": False,
        }
        row = {
            "summary": {
                "repairable": True,
                "initial_conflicts": 50,
                "success": False,
            }
        }
        self.assertTrue(_counterfactual_source_eligible(row, configuration))
        row["summary"]["initial_conflicts"] = 201
        self.assertFalse(_counterfactual_source_eligible(row, configuration))
        self.assertEqual(
            _counterfactual_source_reason(row, configuration),
            "above_maximum_initial_conflicts",
        )
        row["summary"]["initial_conflicts"] = 50
        configuration["require_source_success"] = True
        self.assertFalse(_counterfactual_source_eligible(row, configuration))
        configuration["require_source_success"] = False
        configuration["maximum_agent_count"] = 400
        row["agent_count"] = 600
        self.assertEqual(
            _counterfactual_source_reason(row, configuration),
            "above_maximum_agent_count",
        )

    def test_counterfactual_manifest_recovery_rejects_incomplete_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_config.json").write_text(
                json.dumps({"run_fingerprint": "run"}), encoding="utf-8"
            )
            episode = root / "counterfactual" / "train" / "episode"
            episode.mkdir(parents=True)
            for name, rows in (
                ("states.jsonl", [{"state": 1}]),
                ("outcomes.jsonl", [{"outcome": 1}]),
                ("errors.jsonl", []),
            ):
                (episode / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            metadata = {
                "complete": True,
                "episode_id": "episode",
                "run_fingerprint": "run",
                "status": "ok",
                "state_count": 1,
                "outcome_count": 1,
                "error_count": 0,
                "states_file": "counterfactual/train/episode/states.jsonl",
                "outcomes_file": "counterfactual/train/episode/outcomes.jsonl",
                "errors_file": "counterfactual/train/episode/errors.jsonl",
            }
            (episode / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 1)
            self.assertFalse(report["invalid_metadata"])
            metadata["outcome_count"] = 2
            (episode / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 0)
            self.assertEqual(
                report["invalid_metadata"][0]["reason"],
                "count_mismatch_outcomes_file",
            )

    def test_transfer_config_has_exact_splits_and_volume(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs" / "repair_transfer_pilot.json").read_text(
                encoding="utf-8"
            )
        )
        split_counts = {
            name: sum(split["layout_counts"].values())
            * int(split.get("tasks_per_map", config["tasks_per_map"]))
            for name, split in config["splits"].items()
        }
        self.assertEqual(
            split_counts,
            {
                "train": 24,
                "validation": 12,
                "test_id": 12,
                "test_ood_layout": 24,
                "test_ood_task": 12,
                "test_ood_density": 6,
                "test_joint_ood": 12,
            },
        )
        self.assertEqual(sum(split_counts.values()), 102)
        seen = set(config["splits"]["train"]["layout_counts"])
        unseen = set(config["splits"]["test_ood_layout"]["layout_counts"])
        self.assertTrue(seen.isdisjoint(unseen))

    def test_state_fingerprint_excludes_only_runtime_and_context(self) -> None:
        first = sample_state()
        second = sample_state()
        second["runtime"] = 999.0
        second["context"] = {"layout_mode": "different"}
        self.assertEqual(state_fingerprint(first), state_fingerprint(second))
        second["agents"][0]["path"][-1] = 4
        self.assertNotEqual(state_fingerprint(first), state_fingerprint(second))

    def test_candidate_selection_is_deterministic_and_stratified(self) -> None:
        state = sample_state()
        first = select_seed_agents(state, 3)
        second = select_seed_agents(state, 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertIn(0, first)
        self.assertIn(2, first)
        actions = candidate_actions(
            state,
            maximum_seeds=3,
            heuristics=["target", "collision", "random"],
            neighborhood_sizes=[4, 8],
        )
        self.assertEqual(len(actions), 18)
        self.assertEqual(
            {action["heuristic"] for action in actions},
            {"target", "collision", "random"},
        )

    def test_run_config_rejects_a_different_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            (dataset / "train").mkdir(parents=True)
            (dataset / "dataset_summary.json").write_text(
                json.dumps({"splits": {"train": {}}}), encoding="utf-8"
            )
            for name, value in (
                ("map.map", "type octile\nheight 1\nwidth 1\nmap\n.\n"),
                ("task.scen", "version 1\n"),
                ("map.json", "{}\n"),
                ("task.json", "{}\n"),
            ):
                (dataset / "train" / name).write_text(value, encoding="utf-8")
            (dataset / "train" / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "map_file": "map.map",
                        "scenario_file": "task.scen",
                        "map_metadata_file": "map.json",
                        "task_file": "task.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _prepare_run(
                dataset,
                output,
                {"schema_version": 1, "value": "first"},
                ["train"],
                resume=False,
            )
            with self.assertRaisesRegex(ValueError, "different dataset or"):
                _prepare_run(
                    dataset,
                    output,
                    {"schema_version": 1, "value": "second"},
                    ["train"],
                    resume=True,
                )

            stable_config = {"schema_version": 1, "value": "stable"}
            selected_output = root / "selected-output"
            _prepare_run(
                dataset,
                selected_output,
                stable_config,
                ["train"],
                resume=False,
                task_ids=["task-a"],
            )
            with self.assertRaisesRegex(ValueError, "different dataset or"):
                _prepare_run(
                    dataset,
                    selected_output,
                    stable_config,
                    ["train"],
                    resume=True,
                    task_ids=["task-b"],
                )

            second_output = root / "second-output"
            _prepare_run(
                dataset,
                second_output,
                stable_config,
                ["train"],
                resume=False,
            )
            (dataset / "train" / "map.map").write_text(
                "type octile\nheight 1\nwidth 1\nmap\n@\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "different dataset or"):
                _prepare_run(
                    dataset,
                    second_output,
                    stable_config,
                    ["train"],
                    resume=True,
                )

    def test_early_solution_padding_does_not_repeat_runtime(self) -> None:
        initial = sample_state()
        solved = sample_state()
        solved["feasible"] = True
        solved["done"] = True
        solved["num_of_colliding_pairs"] = 0
        points = [
            {"step": 0, "state": initial, "step_runtime": 0.0},
            {"step": 1, "state": solved, "step_runtime": 0.25},
        ]
        outcome = _horizon_outcomes(initial, points, [4])[0]
        self.assertTrue(outcome["available"])
        self.assertTrue(outcome["solved"])
        self.assertEqual(outcome["conflict_auc"], 1.5)
        self.assertEqual(outcome["branch_runtime"], 0.25)
        self.assertEqual(outcome["time_to_feasible"], 0.25)


if __name__ == "__main__":
    unittest.main()
