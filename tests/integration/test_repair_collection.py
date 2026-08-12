from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from experiments._common import episode_id as make_episode_id
from experiments.repair_collection import (
    COUNTERFACTUAL_SCHEMA,
    COUNTERFACTUAL_METADATA_SCHEMA,
    EPISODE_SCHEMA,
    NATIVE_REPAIR_TIMING_SCHEMA,
    NATIVE_SEMANTICS_SCHEMA,
    NATIVE_UNLIMITED_TIME_SENTINEL_SECONDS,
    REPAIR_COLLECTION_ARTIFACT_VERSION,
    REPAIR_COLLECTION_SCHEMA,
    REPAIR_TIME_LABEL,
    SCHEMA_VERSION,
    STATE_FINGERPRINT_KEYS,
    CollectionLockError,
    _AtomicProcessLock,
    _atomic_write_text,
    _baseline_worker,
    _counterfactual_worker,
    _counterfactual_source_eligible,
    _counterfactual_source_reason,
    _fingerprint,
    _horizon_outcomes,
    _make_environment,
    _native_step_seconds,
    _prepare_run,
    _run_metadata,
    _run_jobs,
    _select_task_rows,
    _trial_seed,
    _valid_episode_trace,
    _validate_config,
    candidate_actions,
    select_seed_agents,
    state_fingerprint,
    recover_counterfactual_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _minimal_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "dataset_summary.json").write_text(
        json.dumps({"splits": {"train": {}}}),
        encoding="utf-8",
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
    return dataset


def _v2_timing_metrics() -> dict:
    return {
        "step_runtime": 0.125,
        "native_step_seconds": 0.125,
        "episode_runtime_delta_seconds": 0.5,
        "action_valid": True,
        "conflicts_before": 3,
        "conflicts_after": 2,
        "sum_of_costs_before": 12,
        "sum_of_costs_after": 13,
    }


def _counterfactual_config() -> dict:
    return {
        "horizons": [1],
        "trials": 1,
        "max_seed_agents": 1,
        "heuristics": ["target"],
        "neighborhood_sizes": [8],
    }


def _v2_state_row(
    run_fingerprint: str = "run",
    episode_id: str = "episode",
) -> dict:
    state = sample_state()
    fingerprint = state_fingerprint(state)
    return {
        "schema": COUNTERFACTUAL_SCHEMA,
        "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
        "repair_time_label": REPAIR_TIME_LABEL,
        "run_fingerprint": run_fingerprint,
        "episode_id": episode_id,
        "state_id": f"{episode_id}__decision_0000",
        "decision_index": 0,
        "state_fingerprint": fingerprint,
        "prefix_actions": [],
        "candidate_count": 1,
        "state": state,
    }


def _v2_outcome_row(
    run_fingerprint: str = "run",
    episode_id: str = "episode",
) -> dict:
    state = sample_state()
    fingerprint = state_fingerprint(state)
    state_id = f"{episode_id}__decision_0000"
    base_action = {
        "mode": "seed",
        "heuristic": "target",
        "seed_agent": 2,
        "neighborhood_size": 8,
    }
    trial_seed = _trial_seed(episode_id, state_id, base_action, 0)
    action = {**base_action, "random_seed": trial_seed}
    after = json.loads(json.dumps(state))
    after["iteration"] += 1
    after["num_of_colliding_pairs"] = 2
    after["sum_of_costs"] = 13
    after["low_level"]["expanded"] += 1
    after["low_level"]["generated"] += 2
    after["low_level"]["runs"] += 1
    return {
        "schema": COUNTERFACTUAL_SCHEMA,
        "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
        "repair_time_label": REPAIR_TIME_LABEL,
        "run_fingerprint": run_fingerprint,
        "episode_id": episode_id,
        "state_id": state_id,
        "state_fingerprint": fingerprint,
        "candidate_index": 0,
        "candidate_action": action,
        "trial_index": 0,
        "trial_seed": trial_seed,
        "step_runtime_label": REPAIR_TIME_LABEL,
        "action_valid": True,
        "conflict_trajectory": [3, 2],
        "steps": [
            {
                "step": 0,
                "metrics": None,
                "step_runtime": 0.0,
                "action": None,
                "state_fingerprint": fingerprint,
                "conflicts": 3,
                "sum_of_costs": 12,
                "low_level": dict(state["low_level"]),
                "terminated": False,
                "truncated": False,
            },
            {
                "step": 1,
                "action": action,
                "metrics": _v2_timing_metrics(),
                "step_runtime": 0.125,
                "state_fingerprint": state_fingerprint(after),
                "conflicts": 2,
                "sum_of_costs": 13,
                "low_level": dict(after["low_level"]),
                "terminated": False,
                "truncated": False,
            },
        ],
        "horizon_outcomes": [
            {
                "horizon": 1,
                "available": True,
                "executed_steps": 1,
                "solved": False,
                "solved_step": None,
                "conflicts_after": 2,
                "conflict_reduction": 1,
                "conflict_auc": 2.5,
                "sum_of_costs_after": 13,
                "cost_improvement": -1,
                "low_level_delta": {
                    "expanded": 1,
                    "generated": 2,
                    "reopened": 0,
                    "runs": 1,
                },
                "branch_runtime": 0.125,
                "branch_runtime_label": REPAIR_TIME_LABEL,
                "time_to_feasible": None,
                "time_to_feasible_label": REPAIR_TIME_LABEL,
            }
        ],
    }


def _v2_error_row(
    run_fingerprint: str = "run",
    episode_id: str = "episode",
) -> dict:
    outcome = _v2_outcome_row(run_fingerprint, episode_id)
    action = dict(outcome["candidate_action"])
    action.pop("random_seed")
    return {
        "schema": COUNTERFACTUAL_SCHEMA,
        "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
        "repair_time_label": REPAIR_TIME_LABEL,
        "run_fingerprint": run_fingerprint,
        "episode_id": episode_id,
        "state_id": outcome["state_id"],
        "state_fingerprint": outcome["state_fingerprint"],
        "candidate_index": outcome["candidate_index"],
        "candidate_action": action,
        "trial_index": outcome["trial_index"],
        "trial_seed": outcome["trial_seed"],
        "error": "RuntimeError: simulated branch failure",
    }


def _episode_events(run_fingerprint: str) -> list[dict]:
    initial = sample_state()
    after = json.loads(json.dumps(initial))
    after["iteration"] = int(initial["iteration"]) + 1
    after["num_of_colliding_pairs"] = 2
    after["runtime"] = 0.625
    after["done"] = True
    metrics = _v2_timing_metrics()
    metrics["sum_of_costs_after"] = 12
    summary = {
        "initial_conflicts": 3,
        "final_conflicts": 2,
        "repairable": True,
        "success": False,
        "truncated": True,
        "repair_iterations": 1,
        "conflict_trajectory": [3, 2],
        "conflict_auc": 2.5,
        "initial_runtime": 0.5,
        "repair_step_runtime": 0.125,
        "repair_step_runtime_label": REPAIR_TIME_LABEL,
        "time_to_feasible": None,
        "time_to_feasible_label": REPAIR_TIME_LABEL,
        "final_sum_of_costs": 12,
    }
    common = {
        "schema": EPISODE_SCHEMA,
        "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
        "repair_time_label": REPAIR_TIME_LABEL,
        "run_fingerprint": run_fingerprint,
    }
    return [
        {
            **common,
            "event": "initial",
            "episode_id": "episode",
            "policy": "official_adaptive",
            "solver_seed": 29,
            "state_fingerprint": state_fingerprint(initial),
            "state": initial,
        },
        {
            **common,
            "event": "transition",
            "episode_id": "episode",
            "native_timing_schema": NATIVE_REPAIR_TIMING_SCHEMA,
            "action": {"mode": "official"},
            "before_fingerprint": state_fingerprint(initial),
            "after_fingerprint": state_fingerprint(after),
            "metrics": metrics,
            "low_level_delta": {
                "expanded": 0,
                "generated": 0,
                "reopened": 0,
                "runs": 0,
            },
            "terminated": False,
            "truncated": True,
            "after": after,
        },
        {
            **common,
            "event": "finish",
            "episode_id": "episode",
            "success": False,
            "final_fingerprint": state_fingerprint(after),
            "summary": summary,
        },
    ]


class RepairCollectionTests(unittest.TestCase):
    def test_native_step_seconds_requires_strict_v2_native_timing(self) -> None:
        self.assertEqual(
            _native_step_seconds(
                {
                    "step_runtime": 0.125,
                    "native_step_seconds": 0.125,
                    "episode_runtime_delta_seconds": 0.5,
                }
            ),
            0.125,
        )
        self.assertEqual(
            _native_step_seconds(
                {
                    "step_runtime": 0.125,
                    "native_step_seconds": 0.125,
                    "episode_runtime_delta_seconds": 0.12,
                }
            ),
            0.125,
        )
        with self.assertRaisesRegex(ValueError, "native_step_seconds"):
            _native_step_seconds({"step_runtime": 0.25})
        with self.assertRaisesRegex(
            ValueError,
            "episode_runtime_delta_seconds",
        ):
            _native_step_seconds(
                {
                    "step_runtime": 0.25,
                    "native_step_seconds": 0.25,
                }
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _native_step_seconds(
                {
                    "step_runtime": 0.5,
                    "native_step_seconds": 0.25,
                    "episode_runtime_delta_seconds": 0.5,
                }
            )
        with self.assertRaisesRegex(ValueError, "numeric"):
            _native_step_seconds(
                {
                    "step_runtime": 10**400,
                    "native_step_seconds": 10**400,
                    "episode_runtime_delta_seconds": 10**400,
                }
            )

    def test_make_environment_rejects_legacy_native_timing_schema(self) -> None:
        legacy_module = mock.Mock()
        legacy_module.repair_timing_schema = "lns2.repair_timing.v1"
        with (
            mock.patch(
                "experiments.repair_collection._load_environment_module",
                return_value=legacy_module,
            ),
            self.assertRaisesRegex(RuntimeError, "requires native timing schema"),
        ):
            _make_environment(
                ".",
                {"split": "train"},
                {},
                "Adaptive",
            )
        legacy_module.LNS2RepairEnv.assert_not_called()

    def test_make_environment_maps_explicit_unlimited_mode_to_native_sentinel(self) -> None:
        module = mock.Mock()
        module.repair_timing_schema = NATIVE_REPAIR_TIMING_SCHEMA
        module.native_semantics_schema = NATIVE_SEMANTICS_SCHEMA
        module.LNS2RepairEnv.return_value = object()
        row = {
            "split": "train",
            "map_file": "map.map",
            "scenario_file": "task.scen",
            "agent_count": 10,
        }
        environment = {
            "time_limit": 0.0,
            "unlimited_time": True,
            "neighborhood_size": 8,
            "replan_algorithm": "PP",
            "use_sipp": True,
            "max_repair_iterations": 0,
        }
        with mock.patch(
            "experiments.repair_collection._load_environment_module",
            return_value=module,
        ):
            _make_environment(".", row, environment, "Adaptive")
        self.assertEqual(
            module.LNS2RepairEnv.call_args.kwargs["time_limit"],
            NATIVE_UNLIMITED_TIME_SENTINEL_SECONDS,
        )

        invalid = {**environment, "time_limit": 1.0}
        with (
            mock.patch(
                "experiments.repair_collection._load_environment_module",
                return_value=module,
            ),
            self.assertRaisesRegex(ValueError, "requires time_limit=0"),
        ):
            _make_environment(".", row, invalid, "Adaptive")

    def test_episode_resume_rejects_legacy_or_non_v2_timing_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.jsonl"
            events = _episode_events("same-run")
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in events),
                encoding="utf-8",
            )
            self.assertIsNotNone(
                _valid_episode_trace(
                    path,
                    "same-run",
                    expected_episode_id="episode",
                    expected_policy="official_adaptive",
                    expected_solver_seed=29,
                )
            )
            self.assertIsNone(
                _valid_episode_trace(
                    path,
                    "same-run",
                    expected_episode_id="other-episode",
                )
            )

            legacy = json.loads(json.dumps(events))
            for row in legacy:
                row["schema"] = "lns2.repair_episode.v1"
                row["schema_version"] = 1
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in legacy),
                encoding="utf-8",
            )
            self.assertIsNone(_valid_episode_trace(path, "same-run"))

            invalid_timing = json.loads(json.dumps(events))
            del invalid_timing[1]["metrics"]["native_step_seconds"]
            path.write_text(
                "".join(
                    json.dumps(row) + "\n" for row in invalid_timing
                ),
                encoding="utf-8",
            )
            self.assertIsNone(_valid_episode_trace(path, "same-run"))

            invalid_summary = json.loads(json.dumps(events))
            invalid_summary[-1]["summary"]["repair_step_runtime"] = 999.0
            path.write_text(
                "".join(
                    json.dumps(row) + "\n" for row in invalid_summary
                ),
                encoding="utf-8",
            )
            self.assertIsNone(_valid_episode_trace(path, "same-run"))

            transition_tampering = []
            invalid_action = json.loads(json.dumps(events))
            invalid_action[1]["action"] = {"mode": "seed"}
            transition_tampering.append(("action", invalid_action))
            invalid_metrics = json.loads(json.dumps(events))
            invalid_metrics[1]["metrics"]["conflicts_before"] = 999
            transition_tampering.append(("metrics", invalid_metrics))
            invalid_low_level = json.loads(json.dumps(events))
            invalid_low_level[1]["low_level_delta"]["expanded"] = 1
            transition_tampering.append(("low_level", invalid_low_level))
            invalid_flags = json.loads(json.dumps(events))
            invalid_flags[1]["truncated"] = False
            transition_tampering.append(("terminal_flags", invalid_flags))
            non_finite = json.loads(json.dumps(events))
            non_finite[1]["unexpected"] = float("nan")
            transition_tampering.append(("non_finite", non_finite))
            non_object = json.loads(json.dumps(events))
            non_object[1] = []
            transition_tampering.append(("non_object", non_object))
            for name, tampered in transition_tampering:
                with self.subTest(name=name):
                    path.write_text(
                        "".join(json.dumps(row) + "\n" for row in tampered),
                        encoding="utf-8",
                    )
                    self.assertIsNone(
                        _valid_episode_trace(path, "same-run")
                    )

    def test_baseline_resume_preserves_completed_invalid_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "split": "train",
                "map_id": "map",
                "task_id": "task",
                "layout_mode": "layout",
                "agent_count": 4,
            }
            policy = "official_adaptive"
            solver_seed = 29
            episode_id = make_episode_id(row, solver_seed, policy)
            path = (
                root
                / "episodes"
                / "train"
                / policy
                / f"{episode_id}.jsonl"
            )
            path.parent.mkdir(parents=True)
            events = _episode_events("run")
            for event in events:
                event["episode_id"] = episode_id
            events[1]["action"] = {"mode": "seed"}
            original = "".join(json.dumps(row) + "\n" for row in events)
            path.write_text(original, encoding="utf-8")
            result = _baseline_worker(
                {
                    "row": row,
                    "policy": policy,
                    "solver_seed": solver_seed,
                    "output_root": str(root),
                    "run_fingerprint": "run",
                    "resume": True,
                }
            )
            self.assertEqual(result["status"], "error")
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_baseline_records_idle_deadline_without_counting_a_repair_step(self) -> None:
        class DeadlineEnvironment:
            def reset(self, seed: int) -> dict:
                del seed
                return sample_state()

            def step(self, action: dict) -> dict:
                del action
                terminal = json.loads(json.dumps(sample_state()))
                terminal["done"] = True
                terminal["runtime"] = 1.0
                return {
                    "observation": terminal,
                    "metrics": {"step_applied": False},
                    "terminated": False,
                    "truncated": True,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "split": "train",
                "map_id": "map",
                "task_id": "task",
                "layout_mode": "layout",
                "task_variant": None,
                "agent_count": 4,
            }
            with mock.patch(
                "experiments.repair_collection._make_environment",
                return_value=DeadlineEnvironment(),
            ):
                result = _baseline_worker(
                    {
                        "dataset_root": str(root),
                        "output_root": str(root),
                        "row": row,
                        "policy": "official_adaptive",
                        "solver_seed": 29,
                        "environment": {},
                        "run_fingerprint": "run",
                        "resume": False,
                    }
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["summary"]["repair_iterations"], 0)
            self.assertEqual(result["summary"]["repair_step_runtime"], 0.0)
            trace = root / str(result["trace_file"])
            self.assertIsNotNone(
                _valid_episode_trace(
                    trace,
                    "run",
                    expected_episode_id=str(result["episode_id"]),
                    expected_policy="official_adaptive",
                    expected_solver_seed=29,
                )
            )

    def test_counterfactual_resume_rejects_legacy_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = (
                root
                / "counterfactual"
                / "train"
                / "episode"
                / "metadata.json"
            )
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema": "lns2.counterfactual_metadata.v1",
                        "schema_version": 1,
                        "run_fingerprint": "same-run",
                        "complete": True,
                        "status": "ok",
                    }
                ),
                encoding="utf-8",
            )
            job = {
                "manifest": {
                    "episode_id": "episode",
                    "split": "train",
                    "policy": "official_adaptive",
                    "trace_file": "missing.jsonl",
                    "solver_seed": 0,
                },
                "output_root": str(root),
                "run_fingerprint": "same-run",
                "counterfactual": _counterfactual_config(),
                "resume": True,
            }
            rejected = _counterfactual_worker(job)
            self.assertEqual(rejected["status"], "error")
            self.assertNotEqual(rejected["status"], "resumed")
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8"))["schema"],
                "lns2.counterfactual_metadata.v1",
            )

            metadata_path.write_text(
                json.dumps(
                    {
                        "schema": COUNTERFACTUAL_METADATA_SCHEMA,
                        "schema_version": (
                            REPAIR_COLLECTION_ARTIFACT_VERSION
                        ),
                        "repair_time_label": REPAIR_TIME_LABEL,
                        "run_fingerprint": "same-run",
                        "episode_id": "episode",
                        "metadata_file": (
                            "counterfactual/train/episode/metadata.json"
                        ),
                        "state_count": 0,
                        "outcome_count": 0,
                        "error_count": 0,
                        "states_file": (
                            "counterfactual/train/episode/states.jsonl"
                        ),
                        "outcomes_file": (
                            "counterfactual/train/episode/outcomes.jsonl"
                        ),
                        "errors_file": (
                            "counterfactual/train/episode/errors.jsonl"
                        ),
                        "counterfactual_config": _counterfactual_config(),
                        "complete": True,
                        "status": "ok",
                    }
                ),
                encoding="utf-8",
            )
            for name in ("states.jsonl", "outcomes.jsonl", "errors.jsonl"):
                (metadata_path.parent / name).write_text("", encoding="utf-8")
            resumed = _counterfactual_worker(job)
            self.assertEqual(resumed["status"], "resumed")

            (metadata_path.parent / "outcomes.jsonl").unlink()
            rejected_missing_file = _counterfactual_worker(job)
            self.assertEqual(rejected_missing_file["status"], "error")
            self.assertNotEqual(rejected_missing_file["status"], "resumed")
            self.assertTrue(metadata_path.is_file())

    def test_counterfactual_resume_validates_all_v2_timing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "counterfactual" / "train" / "episode"
            episode.mkdir(parents=True)
            metadata_path = episode / "metadata.json"
            metadata = {
                "schema": COUNTERFACTUAL_METADATA_SCHEMA,
                "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
                "repair_time_label": REPAIR_TIME_LABEL,
                "run_fingerprint": "run",
                "episode_id": "episode",
                "metadata_file": "counterfactual/train/episode/metadata.json",
                "state_count": 1,
                "outcome_count": 1,
                "error_count": 0,
                "states_file": "counterfactual/train/episode/states.jsonl",
                "outcomes_file": "counterfactual/train/episode/outcomes.jsonl",
                "errors_file": "counterfactual/train/episode/errors.jsonl",
                "counterfactual_config": _counterfactual_config(),
                "complete": True,
                "status": "ok",
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            (episode / "states.jsonl").write_text(
                json.dumps(_v2_state_row()) + "\n",
                encoding="utf-8",
            )
            (episode / "errors.jsonl").write_text("", encoding="utf-8")
            job = {
                "manifest": {
                    "episode_id": "episode",
                    "split": "train",
                    "policy": "official_adaptive",
                    "trace_file": "missing.jsonl",
                    "solver_seed": 0,
                },
                "output_root": str(root),
                "run_fingerprint": "run",
                "counterfactual": _counterfactual_config(),
                "resume": True,
            }

            def write_outcome(row: object) -> None:
                (episode / "outcomes.jsonl").write_text(
                    json.dumps(row) + "\n",
                    encoding="utf-8",
                )

            valid = _v2_outcome_row()
            write_outcome(valid)
            self.assertEqual(_counterfactual_worker(job)["status"], "resumed")
            job["counterfactual"]["horizons"] = [1, 2]
            self.assertEqual(_counterfactual_worker(job)["status"], "error")
            job["counterfactual"]["horizons"] = [1]

            solved = json.loads(json.dumps(valid))
            solved["conflict_trajectory"] = [3, 0]
            solved["steps"][1]["conflicts"] = 0
            solved["steps"][1]["metrics"]["conflicts_after"] = 0
            solved["steps"][1]["terminated"] = True
            solved["horizon_outcomes"][0].update(
                {
                    "solved": True,
                    "solved_step": 1,
                    "conflicts_after": 0,
                    "conflict_reduction": 3,
                    "conflict_auc": 1.5,
                    "time_to_feasible": 0.125,
                }
            )
            write_outcome(solved)
            self.assertEqual(_counterfactual_worker(job)["status"], "resumed")

            invalid_rows = []
            missing_native = json.loads(json.dumps(valid))
            del missing_native["steps"][1]["metrics"]["native_step_seconds"]
            invalid_rows.append(("missing_native", missing_native))
            negative_step = json.loads(json.dumps(valid))
            negative_step["steps"][1]["step_runtime"] = -0.125
            invalid_rows.append(("negative_step", negative_step))
            mismatched_step = json.loads(json.dumps(valid))
            mismatched_step["steps"][1]["step_runtime"] = 0.25
            invalid_rows.append(("mismatched_step", mismatched_step))
            overflowing_step = json.loads(json.dumps(valid))
            overflowing_step["steps"][1]["step_runtime"] = 10**400
            invalid_rows.append(("overflowing_step", overflowing_step))
            negative_branch = json.loads(json.dumps(valid))
            negative_branch["horizon_outcomes"][0]["branch_runtime"] = -0.125
            invalid_rows.append(("negative_branch", negative_branch))
            mismatched_branch = json.loads(json.dumps(valid))
            mismatched_branch["horizon_outcomes"][0]["branch_runtime"] = 0.25
            invalid_rows.append(("mismatched_branch", mismatched_branch))
            negative_feasible = json.loads(json.dumps(valid))
            negative_feasible["horizon_outcomes"][0].update(
                {
                    "solved": True,
                    "solved_step": 1,
                    "time_to_feasible": -0.125,
                }
            )
            invalid_rows.append(("negative_feasible", negative_feasible))
            mismatched_feasible = json.loads(json.dumps(valid))
            mismatched_feasible["horizon_outcomes"][0].update(
                {
                    "solved": True,
                    "solved_step": 1,
                    "time_to_feasible": 0.25,
                }
            )
            invalid_rows.append(("mismatched_feasible", mismatched_feasible))
            invalid_trajectory = json.loads(json.dumps(valid))
            invalid_trajectory["conflict_trajectory"] = [3, 999]
            invalid_rows.append(("conflict_trajectory", invalid_trajectory))
            invalid_conflicts = json.loads(json.dumps(valid))
            invalid_conflicts["steps"][1]["conflicts"] = 999
            invalid_rows.append(("step_conflicts", invalid_conflicts))
            invalid_cost = json.loads(json.dumps(valid))
            invalid_cost["steps"][1]["sum_of_costs"] = 999
            invalid_rows.append(("step_sum_of_costs", invalid_cost))
            invalid_fingerprint = json.loads(json.dumps(valid))
            invalid_fingerprint["steps"][0]["state_fingerprint"] = "f" * 64
            invalid_rows.append(("step_fingerprint", invalid_fingerprint))
            invalid_available = json.loads(json.dumps(valid))
            invalid_available["horizon_outcomes"][0]["available"] = False
            invalid_rows.append(("available", invalid_available))
            non_boolean_available = json.loads(json.dumps(valid))
            non_boolean_available["horizon_outcomes"][0]["available"] = 1
            invalid_rows.append(
                ("non_boolean_available", non_boolean_available)
            )
            non_boolean_solved = json.loads(json.dumps(valid))
            non_boolean_solved["horizon_outcomes"][0]["solved"] = "false"
            invalid_rows.append(("non_boolean_solved", non_boolean_solved))
            invalid_conflict_metrics = json.loads(json.dumps(valid))
            invalid_conflict_metrics["horizon_outcomes"][0].update(
                {
                    "conflicts_after": 999,
                    "conflict_reduction": -996,
                    "conflict_auc": 999.0,
                }
            )
            invalid_rows.append(
                ("horizon_conflict_metrics", invalid_conflict_metrics)
            )
            invalid_cost_metrics = json.loads(json.dumps(valid))
            invalid_cost_metrics["horizon_outcomes"][0].update(
                {
                    "sum_of_costs_after": 999,
                    "cost_improvement": -989,
                }
            )
            invalid_rows.append(
                ("horizon_cost_metrics", invalid_cost_metrics)
            )
            invalid_low_level = json.loads(json.dumps(valid))
            invalid_low_level["horizon_outcomes"][0]["low_level_delta"][
                "expanded"
            ] = 999
            invalid_rows.append(("low_level_delta", invalid_low_level))
            empty_horizons = json.loads(json.dumps(valid))
            empty_horizons["horizon_outcomes"] = []
            invalid_rows.append(("empty_horizons", empty_horizons))
            duplicate_horizons = json.loads(json.dumps(valid))
            duplicate_horizons["horizon_outcomes"].append(
                json.loads(
                    json.dumps(duplicate_horizons["horizon_outcomes"][0])
                )
            )
            invalid_rows.append(("duplicate_horizons", duplicate_horizons))

            for name, row in invalid_rows:
                with self.subTest(name=name):
                    write_outcome(row)
                    result = _counterfactual_worker(job)
                    self.assertEqual(result["status"], "error")
                    self.assertTrue(metadata_path.is_file())
            write_outcome([])
            self.assertEqual(_counterfactual_worker(job)["status"], "error")

    def test_counterfactual_resume_cross_binds_state_candidate_and_trials(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "counterfactual" / "train" / "episode"
            episode.mkdir(parents=True)
            metadata_path = episode / "metadata.json"
            base_metadata = {
                "schema": COUNTERFACTUAL_METADATA_SCHEMA,
                "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
                "repair_time_label": REPAIR_TIME_LABEL,
                "run_fingerprint": "run",
                "episode_id": "episode",
                "metadata_file": "counterfactual/train/episode/metadata.json",
                "state_count": 1,
                "outcome_count": 1,
                "error_count": 0,
                "states_file": "counterfactual/train/episode/states.jsonl",
                "outcomes_file": "counterfactual/train/episode/outcomes.jsonl",
                "errors_file": "counterfactual/train/episode/errors.jsonl",
                "counterfactual_config": _counterfactual_config(),
                "complete": True,
                "status": "ok",
            }
            job = {
                "manifest": {
                    "episode_id": "episode",
                    "split": "train",
                    "policy": "official_adaptive",
                    "trace_file": "missing.jsonl",
                    "solver_seed": 0,
                },
                "output_root": str(root),
                "run_fingerprint": "run",
                "counterfactual": _counterfactual_config(),
                "resume": True,
            }

            def write_case(
                states: list[object],
                outcomes: list[object],
                *,
                outcome_count: int | None = None,
            ) -> str:
                metadata = dict(base_metadata)
                metadata["state_count"] = len(states)
                metadata["outcome_count"] = (
                    len(outcomes)
                    if outcome_count is None
                    else outcome_count
                )
                metadata_text = json.dumps(metadata)
                metadata_path.write_text(metadata_text, encoding="utf-8")
                (episode / "states.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in states),
                    encoding="utf-8",
                )
                (episode / "outcomes.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in outcomes),
                    encoding="utf-8",
                )
                (episode / "errors.jsonl").write_text("", encoding="utf-8")
                return metadata_text

            state = _v2_state_row()
            outcome = _v2_outcome_row()
            write_case([state], [outcome])
            self.assertEqual(_counterfactual_worker(job)["status"], "resumed")

            cases: list[tuple[str, list[object], list[object]]] = []
            bad_fingerprint = json.loads(json.dumps(state))
            bad_fingerprint["state_fingerprint"] = "f" * 64
            cases.append(("state_fingerprint", [bad_fingerprint], [outcome]))
            bad_prefix = json.loads(json.dumps(state))
            bad_prefix["prefix_actions"] = [{"mode": "official"}]
            cases.append(("prefix", [bad_prefix], [outcome]))
            bad_count = json.loads(json.dumps(state))
            bad_count["candidate_count"] = 2
            cases.append(("candidate_count", [bad_count], [outcome]))
            orphan = json.loads(json.dumps(outcome))
            orphan["state_id"] = "episode__decision_9999"
            cases.append(("orphan_state", [state], [orphan]))
            bad_action = json.loads(json.dumps(outcome))
            bad_action["candidate_action"]["heuristic"] = "random"
            cases.append(("candidate_action", [state], [bad_action]))
            bad_trial = json.loads(json.dumps(outcome))
            bad_trial["trial_index"] = 1
            cases.append(("trial_index", [state], [bad_trial]))
            non_finite = json.loads(json.dumps(outcome))
            non_finite["unexpected"] = float("inf")
            cases.append(("non_finite", [state], [non_finite]))
            cases.append(("non_object", [state], [[]]))

            for name, states, outcomes in cases:
                with self.subTest(name=name):
                    original_metadata = write_case(states, outcomes)
                    self.assertEqual(_counterfactual_worker(job)["status"], "error")
                    self.assertEqual(
                        metadata_path.read_text(encoding="utf-8"),
                        original_metadata,
                    )

            missing_metadata = write_case([state], [])
            self.assertEqual(_counterfactual_worker(job)["status"], "error")
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                missing_metadata,
            )
            duplicate_metadata = write_case([state], [outcome, outcome])
            self.assertEqual(_counterfactual_worker(job)["status"], "error")
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                duplicate_metadata,
            )
            metadata_path.write_text("[]", encoding="utf-8")
            self.assertEqual(_counterfactual_worker(job)["status"], "error")
            self.assertEqual(metadata_path.read_text(encoding="utf-8"), "[]")

    def test_run_identity_binds_schema_timing_and_producer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _minimal_dataset(root)
            config = {"schema_version": SCHEMA_VERSION, "value": "stable"}
            fingerprint, run_config = _run_metadata(
                dataset,
                config,
                ["train"],
            )
            self.assertEqual(
                run_config["schema"],
                REPAIR_COLLECTION_SCHEMA,
            )
            self.assertEqual(
                run_config["schema_version"],
                REPAIR_COLLECTION_ARTIFACT_VERSION,
            )
            self.assertEqual(
                run_config["input_config_schema_version"],
                SCHEMA_VERSION,
            )
            with mock.patch(
                "experiments.repair_collection.REPAIR_TIME_LABEL",
                "lns2.repair_time.changed",
            ):
                changed_timing, _ = _run_metadata(
                    dataset,
                    config,
                    ["train"],
                )
            with mock.patch(
                "experiments.repair_collection._producer_identity",
                return_value={
                    "name": "experiments.repair_collection",
                    "implementation_sha256": "changed",
                },
            ):
                changed_producer, _ = _run_metadata(
                    dataset,
                    config,
                    ["train"],
                )
            self.assertNotEqual(fingerprint, changed_timing)
            self.assertNotEqual(fingerprint, changed_producer)

            output = root / "legacy-output"
            output.mkdir()
            (output / "run_config.json").write_text(
                json.dumps(
                    {
                        "schema": "lns2.repair_collection.v1",
                        "schema_version": 1,
                        "run_fingerprint": fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                _prepare_run(
                    dataset,
                    output,
                    config,
                    ["train"],
                    resume=True,
                    metadata=(fingerprint, run_config),
                )

    def test_atomic_write_retries_transient_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            original_replace = Path.replace
            attempts = 0

            def transient_replace(source: Path, target: Path) -> Path:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("temporary sharing violation")
                return original_replace(source, target)

            with (
                mock.patch.object(Path, "replace", autospec=True, side_effect=transient_replace),
                mock.patch("experiments.repair_collection.time.sleep") as sleep,
            ):
                _atomic_write_text(path, "complete\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "complete\n")
            self.assertEqual(attempts, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_prepare_run_rejects_nonempty_output_without_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _minimal_dataset(root)
            config = json.loads(
                (
                    PROJECT_ROOT
                    / "configs"
                    / "repair_collection_hardening_smoke.json"
                ).read_text(encoding="utf-8")
            )
            fingerprint, run_config = _run_metadata(dataset, config, ["train"])
            output = root / "existing-output"
            output.mkdir()
            sentinel = output / "evidence.jsonl"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            before = sentinel.read_bytes()

            with self.assertRaisesRegex(ValueError, "non-empty"):
                _prepare_run(
                    dataset,
                    output,
                    config,
                    ["train"],
                    resume=False,
                    metadata=(fingerprint, run_config),
                )

            self.assertEqual(sentinel.read_bytes(), before)
            self.assertFalse((output / "run_config.json").exists())
            self.assertEqual(list(output.iterdir()), [sentinel])

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

    def test_scheduler_uses_custom_failure_result_for_non_dataset_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [
                {
                    "job_id": "state-a",
                    "state_record": {"state_fingerprint": "state-a"},
                    "sleep": 0.3,
                }
            ]

            def failure_result(job: dict, status: str, message: str) -> dict:
                return {
                    "state_fingerprint": job["state_record"]["state_fingerprint"],
                    "status": status,
                    "error": message,
                    "state_count": 0,
                    "outcome_count": 0,
                }

            results = _run_jobs(
                _scheduler_worker,
                jobs,
                1,
                phase="custom-timeout-test",
                output_root=root,
                run_fingerprint="run",
                timeout_seconds=0.05,
                failure_result=failure_result,
            )
            self.assertEqual(results[0]["state_fingerprint"], "state-a")
            self.assertEqual(results[0]["status"], "timeout")
            self.assertIn("exceeded", results[0]["error"])

    def test_scheduler_can_stop_dispatch_after_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                for task_id, delay in (
                    ("timeout", 0.3),
                    ("must-not-start-a", 0.0),
                    ("must-not-start-b", 0.0),
                )
            ]
            results = _run_jobs(
                _scheduler_worker,
                jobs,
                1,
                phase="stop-on-failure-test",
                output_root=root,
                run_fingerprint="run",
                timeout_seconds=0.05,
                stop_on_failure=True,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "timeout")
            progress = json.loads(
                (root / "collection_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["status"], "failed")
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
                json.dumps(
                    {
                        "schema": REPAIR_COLLECTION_SCHEMA,
                        "schema_version": (
                            REPAIR_COLLECTION_ARTIFACT_VERSION
                        ),
                        "repair_time_label": REPAIR_TIME_LABEL,
                        "run_fingerprint": "run",
                        "configuration": {
                            "counterfactual": _counterfactual_config()
                        },
                    }
                ),
                encoding="utf-8",
            )
            episode = root / "counterfactual" / "train" / "episode"
            episode.mkdir(parents=True)
            for name, rows in (
                ("states.jsonl", [_v2_state_row()]),
                (
                    "outcomes.jsonl",
                    [_v2_outcome_row()],
                ),
                ("errors.jsonl", []),
            ):
                (episode / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            metadata = {
                "schema": COUNTERFACTUAL_METADATA_SCHEMA,
                "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
                "repair_time_label": REPAIR_TIME_LABEL,
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
                "metadata_file": "counterfactual/train/episode/metadata.json",
                "counterfactual_config": _counterfactual_config(),
            }
            (episode / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 1)
            self.assertFalse(report["invalid_metadata"])
            metadata["counterfactual_config"] = {
                **_counterfactual_config(),
                "trials": 2,
            }
            (episode / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 0)
            self.assertEqual(
                report["invalid_metadata"][0]["reason"],
                "artifact_schema_or_timing_mismatch",
            )
            metadata["counterfactual_config"] = _counterfactual_config()
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
            metadata["outcome_count"] = 1
            (episode / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            invalid_timing = _v2_outcome_row()
            invalid_timing["horizon_outcomes"][0]["branch_runtime"] = 0.25
            (episode / "outcomes.jsonl").write_text(
                json.dumps(invalid_timing) + "\n",
                encoding="utf-8",
            )
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 0)
            self.assertEqual(
                report["invalid_metadata"][0]["reason"],
                "artifact_schema_or_timing_mismatch",
            )
            (episode / "outcomes.jsonl").write_text("[]\n", encoding="utf-8")
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 0)
            self.assertEqual(
                report["invalid_metadata"][0]["reason"],
                "artifact_schema_or_timing_mismatch",
            )
            orphan_outcome = _v2_outcome_row()
            orphan_outcome["state_id"] = "episode__decision_9999"
            (episode / "outcomes.jsonl").write_text(
                json.dumps(orphan_outcome) + "\n",
                encoding="utf-8",
            )
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 0)
            self.assertEqual(
                report["invalid_metadata"][0]["reason"],
                "artifact_schema_or_timing_mismatch",
            )
            error_metadata = dict(metadata)
            error_metadata.update(
                {
                    "outcome_count": 0,
                    "error_count": 1,
                    "status": "error",
                }
            )
            (episode / "metadata.json").write_text(
                json.dumps(error_metadata), encoding="utf-8"
            )
            (episode / "outcomes.jsonl").write_text("", encoding="utf-8")
            error_row = _v2_error_row()
            (episode / "errors.jsonl").write_text(
                json.dumps(error_row) + "\n", encoding="utf-8"
            )
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 1)
            tampered_error = json.loads(json.dumps(error_row))
            tampered_error["state_fingerprint"] = "f" * 64
            (episode / "errors.jsonl").write_text(
                json.dumps(tampered_error) + "\n", encoding="utf-8"
            )
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 0)
            self.assertEqual(
                report["invalid_metadata"][0]["reason"],
                "artifact_schema_or_timing_mismatch",
            )
            (episode / "metadata.json").write_text("[]", encoding="utf-8")
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 0)
            self.assertEqual(
                report["invalid_metadata"][0]["reason"],
                "invalid_metadata_json_object",
            )
            self.assertEqual(
                (episode / "metadata.json").read_text(encoding="utf-8"),
                "[]",
            )
            (episode / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            run_config = json.loads(
                (root / "run_config.json").read_text(encoding="utf-8")
            )
            run_config["configuration"]["counterfactual"]["horizons"] = []
            (root / "run_config.json").write_text(
                json.dumps(run_config),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "horizons"):
                recover_counterfactual_manifest(root)

    def test_manifest_recovery_rejects_partial_v2_run_identity(self) -> None:
        identities = (
            {
                "schema": REPAIR_COLLECTION_SCHEMA,
                "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
            },
            {
                "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
            },
            {
                "schema": REPAIR_COLLECTION_SCHEMA,
                "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
                "repair_time_label": "lns2.repair_time.wrong",
            },
        )
        for index, identity in enumerate(identities):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "run_config.json").write_text(
                    json.dumps(
                        {
                            **identity,
                            "run_fingerprint": "run",
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "incomplete or unsupported artifact identity",
                ):
                    recover_counterfactual_manifest(root)
                self.assertFalse(
                    (root / "counterfactual_manifest.jsonl").exists()
                )
                self.assertFalse((root / "summary.json").exists())

    def test_legacy_manifest_recovery_preserves_legacy_summary_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_config.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_fingerprint": "legacy-run",
                    }
                ),
                encoding="utf-8",
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
            (episode / "metadata.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "episode_id": "episode",
                        "run_fingerprint": "legacy-run",
                        "status": "ok",
                        "state_count": 1,
                        "outcome_count": 1,
                        "error_count": 0,
                        "states_file": (
                            "counterfactual/train/episode/states.jsonl"
                        ),
                        "outcomes_file": (
                            "counterfactual/train/episode/outcomes.jsonl"
                        ),
                        "errors_file": (
                            "counterfactual/train/episode/errors.jsonl"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            report = recover_counterfactual_manifest(root)
            self.assertEqual(report["recovered_count"], 1)
            summary = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertNotIn("schema", summary)
            self.assertNotIn("repair_time_label", summary)

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

    def test_state_fingerprint_fast_path_matches_historical_canonical_hash(self) -> None:
        state = sample_state()
        historical = _fingerprint(
            {key: state[key] for key in STATE_FINGERPRINT_KEYS}
        )
        self.assertEqual(state_fingerprint(state), historical)

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
        self.assertEqual(outcome["branch_runtime_label"], REPAIR_TIME_LABEL)
        self.assertEqual(outcome["time_to_feasible"], 0.25)
        self.assertEqual(outcome["time_to_feasible_label"], REPAIR_TIME_LABEL)


if __name__ == "__main__":
    unittest.main()
