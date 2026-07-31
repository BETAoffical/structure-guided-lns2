from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_lns2_warm_start_feasibility as warm


def _write_valid_completed_job(root: Path) -> tuple[dict, Path, Path]:
    final_state = {
        "initialized": True,
        "initial_solution_complete": True,
        "agents": [{"path": [0, 1]}],
        "num_of_colliding_pairs": 0,
        "sum_of_costs": 1,
        "feasible": True,
        "done": True,
        "iteration": 1,
        "runtime": 0.2,
        "rows": 1,
        "cols": 2,
        "low_level": {"generated": 1, "expanded": 1, "reopened": 0, "runs": 1},
        "obstacles": [0, 0],
        "conflict_edges": [],
    }
    source_state = {
        "initialized": True,
        "initial_solution_complete": True,
        "agents": [{"path": [0, 1]}],
        "num_of_colliding_pairs": 1,
        "sum_of_costs": 1,
        "feasible": False,
        "done": False,
        "iteration": 0,
        "runtime": 0.0,
        "rows": 1,
        "cols": 2,
        "low_level": {"generated": 1, "expanded": 1, "reopened": 0, "runs": 1},
        "obstacles": [0, 0],
        "conflict_edges": [[0, 0]],
    }
    source = {
        "source_seed": 5,
        "source_label": None,
        "dataset_root": str(root / "dataset"),
        "row": {"task_id": "task", "map_id": "map"},
        "environment": {"time_limit": 1.0},
        "evidence": {},
        "state": source_state,
    }
    source_identity = warm._source_identity(source)
    task_identity = warm._task_identity(source)
    portfolio_key = warm._fingerprint(task_identity)
    run_identity = {"schema": warm.SCHEMA, "test_run": True}
    run_fingerprint = warm._fingerprint(run_identity)
    job_identity = {
        "schema": warm.SCHEMA,
        "run_fingerprint": run_fingerprint,
        "source_identity": source_identity,
        "task_identity": task_identity,
        "portfolio_key": portfolio_key,
        "continuation_seed": 7,
        "time_limit": 1.0,
    }
    job_fingerprint = warm._fingerprint(job_identity)
    job_id = warm._job_id(5, 7, job_fingerprint)
    final_state_path = root / "jobs" / job_id / "final_state.json.gz"
    warm._atomic_write_gzip_json(final_state_path, final_state)
    job = {
        "output_root": str(root),
        "job_id": job_id,
        "run_fingerprint": run_fingerprint,
        "job_fingerprint": job_fingerprint,
        "job_identity": job_identity,
        "portfolio_key": portfolio_key,
        "source_identity": source_identity,
        "task_identity": task_identity,
        "source": source,
        "continuation_seed": 7,
        "time_limit": 1.0,
    }
    completed_at = "2026-01-01T00:00:00+00:00"
    result = {
        "schema": warm.SCHEMA,
        "status": "complete",
        "job_id": job_id,
        "run_fingerprint": job["run_fingerprint"],
        "job_fingerprint": job["job_fingerprint"],
        "portfolio_key": job["portfolio_key"],
        "source_seed": 5,
        "continuation_seed": 7,
        "time_limit": 1.0,
        "source_evidence": {},
        "source_identity_fingerprint": warm._fingerprint(source_identity),
        "warm_start_semantics": "paths-and-derived-repair-state; ALNS-and-RNG-reset",
        "success": True,
        "stop_reason": "feasible",
        "initial_conflicts": 1,
        "final_conflicts": 0,
        "initial_sum_of_costs": 1,
        "final_sum_of_costs": 1,
        "repair_iterations": 1,
        "nonreducing_repairs": 0,
        "nonreducing_fraction": 0.0,
        "selection_seconds": 0.01,
        "pp_seconds": 0.02,
        "repair_wall_seconds": 0.05,
        "observed_wall_seconds": 0.1,
        "native_runtime": 0.2,
        "restore_timings": {"reset_total_seconds": 0.01},
        "conflict_trajectory": [1, 0],
        "elapsed_seconds": [0.0, 0.05],
        "step_applied_trajectory": [True],
        "diagnostics": [],
        "final_paths_sha256": warm._paths_sha256([[0, 1]]),
        "final_state_file": final_state_path.relative_to(root).as_posix(),
        "final_state_sha256": warm._sha256(final_state_path),
        "completed_at": completed_at,
    }
    result_path = root / "jobs" / job_id / "result.json"
    status_path = root / "jobs" / job_id / "status.json"
    warm._write_json(result_path, result)
    warm._write_json(
        status_path,
        {
            "schema": warm.SCHEMA,
            "status": "complete",
            "job_id": job_id,
            "run_fingerprint": job["run_fingerprint"],
            "job_fingerprint": job["job_fingerprint"],
            "portfolio_key": job["portfolio_key"],
            "source_seed": 5,
            "continuation_seed": 7,
            "initial_conflicts": 1,
            "current_conflicts": 0,
            "repair_iterations": 1,
            "elapsed_seconds": 0.1,
            "success": True,
            "stop_reason": "feasible",
            "latest_diagnostic": None,
            "errors": 0,
            "completed_at": completed_at,
        },
    )
    warm._write_json(
        root / "run_config.json",
        {
            "schema": warm.SCHEMA,
            "run_fingerprint": run_fingerprint,
            "run_identity": run_identity,
            "jobs": [
                {
                    "job_id": job_id,
                    "job_fingerprint": job_fingerprint,
                    "job_identity": job_identity,
                    "portfolio_key": portfolio_key,
                    "source_seed": 5,
                    "continuation_seed": 7,
                }
            ],
        },
    )
    return job, result_path, status_path


class WarmStartIdentityTests(unittest.TestCase):
    def test_source_seed_parser_rejects_negative_and_duplicate_values(self) -> None:
        self.assertEqual(warm._parse_source_seeds("5, 7"), [5, 7])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            warm._parse_source_seeds("5,-1")
        with self.assertRaisesRegex(ValueError, "unique"):
            warm._parse_source_seeds("5,5")

    def test_cli_rejects_negative_continuation_seed_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            sys,
            "argv",
            [
                "run_lns2_warm_start_feasibility.py",
                "--source",
                directory,
                "--output",
                str(Path(directory) / "output"),
                "--seed-base",
                "-1",
            ],
        ), mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaisesRegex(
            SystemExit, "2"
        ):
            warm.main()

    def test_output_identity_is_stable_and_rejects_legacy_or_changed_runs(self) -> None:
        identity = {"schema": warm.SCHEMA, "input": {"seed": 7}}
        fingerprint = warm._fingerprint(identity)
        config = {
            "schema": warm.SCHEMA,
            "run_fingerprint": fingerprint,
            "run_identity": identity,
            "producer_identity": {"test": True},
            "jobs": [],
        }
        validation = {
            "schema": warm.SCHEMA,
            "status": "passed",
            "run_fingerprint": fingerprint,
            "sources": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new"
            warm._prepare_run_output(root, config, validation)
            first_validation = (root / "source_validation.json").read_bytes()
            warm._prepare_run_output(root, config, validation)
            self.assertEqual(
                (root / "source_validation.json").read_bytes(), first_validation
            )

            changed_identity = {"schema": warm.SCHEMA, "input": {"seed": 8}}
            changed = {
                **config,
                "run_fingerprint": warm._fingerprint(changed_identity),
                "run_identity": changed_identity,
            }
            with self.assertRaisesRegex(RuntimeError, "resume .* mismatch"):
                warm._prepare_run_output(root, changed, validation)

            legacy = Path(directory) / "legacy"
            legacy.mkdir()
            (legacy / "run_config.json").write_text(
                json.dumps({"schema": "lns2.warm_start_feasibility.v1"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "resume schema mismatch"):
                warm._prepare_run_output(legacy, config, validation)

    def test_portfolio_marker_is_task_and_run_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, result_path, status_path = _write_valid_completed_job(root)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            final_state_path = root / result["final_state_file"]
            final_state = warm._read_gzip_json(final_state_path)
            marker_path = warm._portfolio_marker_path(
                root, job["portfolio_key"]
            )
            warm._mark_portfolio_solved(
                marker_path,
                job,
                job_id=job["job_id"],
                result_path=result_path,
                status_path=status_path,
                result=result,
                state=final_state,
            )
            self.assertTrue(warm._portfolio_is_solved(marker_path, job))
            with self.assertRaisesRegex(RuntimeError, "run_fingerprint"):
                warm._portfolio_is_solved(
                    marker_path, {**job, "run_fingerprint": "d" * 64}
                )
            other_task = {"dataset_root": "/dataset", "task": {"task_id": "t2"}}
            self.assertNotEqual(
                marker_path,
                warm._portfolio_marker_path(root, warm._fingerprint(other_task)),
            )
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["success"] = False
            warm._write_json(status_path, status)
            with self.assertRaisesRegex(
                RuntimeError, "winning_status_sha256"
            ):
                warm._portfolio_is_solved(marker_path, job)

    def test_portfolio_marker_is_not_published_before_winner_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {
                "output_root": str(root),
                "run_fingerprint": "a" * 64,
                "portfolio_key": "b" * 64,
            }
            marker = warm._portfolio_marker_path(root, job["portfolio_key"])
            state = {
                "initialized": True,
                "initial_solution_complete": True,
                "agents": [{"path": [0]}],
                "feasible": True,
                "done": True,
                "iteration": 0,
                "runtime": 0.0,
                "num_of_colliding_pairs": 0,
                "sum_of_costs": 0,
                "rows": 1,
                "cols": 1,
                "low_level": {
                    "generated": 1,
                    "expanded": 1,
                    "reopened": 0,
                    "runs": 1,
                },
                "obstacles": [0],
                "conflict_edges": [],
            }
            result_path = root / "jobs" / "winner" / "result.json"
            status_path = root / "jobs" / "winner" / "status.json"
            result = {
                "schema": warm.SCHEMA,
                "status": "complete",
                "job_id": "winner",
                "run_fingerprint": job["run_fingerprint"],
                "portfolio_key": job["portfolio_key"],
                "success": True,
                "final_state_file": "jobs/winner/final_state.json.gz",
                "final_state_sha256": "c" * 64,
            }
            with self.assertRaisesRegex(
                RuntimeError, "durable result and status"
            ):
                warm._mark_portfolio_solved(
                    marker,
                    job,
                    job_id="winner",
                    result_path=result_path,
                    status_path=status_path,
                    result=result,
                    state=state,
                )
            self.assertFalse(marker.exists())

    def test_portfolio_marker_rejects_hash_rebound_incomplete_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, result_path, status_path = _write_valid_completed_job(root)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            final_state = warm._read_gzip_json(
                root / result["final_state_file"]
            )
            marker_path = warm._portfolio_marker_path(
                root, job["portfolio_key"]
            )
            warm._mark_portfolio_solved(
                marker_path,
                job,
                job_id=job["job_id"],
                result_path=result_path,
                status_path=status_path,
                result=result,
                state=final_state,
            )
            for field in (
                "job_fingerprint",
                "stop_reason",
                "final_conflicts",
                "conflict_trajectory",
            ):
                result.pop(field)
            warm._write_json(result_path, result)
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["winning_result_sha256"] = warm._sha256(result_path)
            warm._write_json(marker_path, marker)
            with self.assertRaisesRegex(
                RuntimeError, "failed completed-result validation"
            ):
                warm._portfolio_is_solved(marker_path, job)

    def test_invalid_completed_result_is_preserved_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_id = warm._job_id(5, 730000, "b" * 64)
            result_path = root / "jobs" / job_id / "result.json"
            warm._write_json(
                result_path,
                {
                    "schema": warm.SCHEMA,
                    "status": "complete",
                    "job_id": job_id,
                    "run_fingerprint": "wrong",
                },
            )
            original = result_path.read_bytes()
            job = {
                "output_root": str(root),
                "job_id": job_id,
                "run_fingerprint": "a" * 64,
                "job_fingerprint": "b" * 64,
                "portfolio_key": "c" * 64,
                "source_identity": {"source": 5},
                "source": {"source_seed": 5, "evidence": {}},
                "continuation_seed": 730000,
                "time_limit": 1.0,
            }
            with self.assertRaisesRegex(RuntimeError, "invalid and was preserved"):
                warm._run_job(job)
            self.assertEqual(result_path.read_bytes(), original)

    def test_completed_result_requires_strict_boolean_and_integer_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_state_path = root / "jobs" / "job" / "final_state.json.gz"
            final_state = {
                "agents": [{"path": [0, 1]}],
                "num_of_colliding_pairs": 1,
                "sum_of_costs": 1,
                "feasible": False,
            }
            warm._atomic_write_gzip_json(final_state_path, final_state)
            job = {
                "output_root": str(root),
                "job_id": "job",
                "run_fingerprint": "a" * 64,
                "job_fingerprint": "b" * 64,
                "portfolio_key": "c" * 64,
                "source_identity": {"source": 5},
                "source": {"source_seed": 5, "evidence": {}},
                "continuation_seed": 7,
                "time_limit": 1.0,
            }
            result = {
                "schema": warm.SCHEMA,
                "status": "complete",
                "job_id": "job",
                "run_fingerprint": job["run_fingerprint"],
                "job_fingerprint": job["job_fingerprint"],
                "portfolio_key": job["portfolio_key"],
                "source_seed": 5,
                "continuation_seed": 7,
                "time_limit": 1.0,
                "source_evidence": {},
                "source_identity_fingerprint": warm._fingerprint(
                    job["source_identity"]
                ),
                "success": "false",
                "stop_reason": "time_limit",
                "initial_conflicts": 1,
                "final_conflicts": 1.0,
                "initial_sum_of_costs": 1,
                "final_sum_of_costs": 1,
                "repair_iterations": 0,
                "nonreducing_repairs": 0,
                "final_paths_sha256": warm._paths_sha256([[0, 1]]),
                "final_state_file": str(final_state_path.relative_to(root)),
                "final_state_sha256": warm._sha256(final_state_path),
            }
            result_path = root / "jobs" / "job" / "result.json"
            status_path = root / "jobs" / "job" / "status.json"
            warm._write_json(result_path, result)
            warm._write_json(
                status_path,
                {
                    "schema": warm.SCHEMA,
                    "status": "complete",
                    "job_id": "job",
                    "run_fingerprint": job["run_fingerprint"],
                    "job_fingerprint": job["job_fingerprint"],
                    "portfolio_key": job["portfolio_key"],
                },
            )
            with self.assertRaisesRegex(RuntimeError, "success is not a boolean"):
                warm._validate_completed_result(result_path, status_path, job)

    def test_completed_result_recomputes_trajectory_and_status_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, result_path, status_path = _write_valid_completed_job(root)
            validated = warm._validate_completed_result(
                result_path, status_path, job
            )
            self.assertTrue(validated["success"])

            tampered = json.loads(result_path.read_text(encoding="utf-8"))
            tampered["conflict_trajectory"][-1] = 1
            warm._write_json(result_path, tampered)
            with self.assertRaisesRegex(
                RuntimeError, "trajectory final endpoint mismatch"
            ):
                warm._validate_completed_result(result_path, status_path, job)

    def test_resume_republishes_marker_only_after_complete_winner_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, _result_path, _status_path = _write_valid_completed_job(root)
            marker = warm._portfolio_marker_path(root, job["portfolio_key"])
            self.assertFalse(marker.exists())
            resumed = warm._run_job(job)
            self.assertTrue(resumed["success"])
            self.assertTrue(warm._portfolio_is_solved(marker, job))

    def test_completed_result_rejects_coordinated_path_endpoint_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, result_path, status_path = _write_valid_completed_job(root)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            final_state_path = root / result["final_state_file"]
            final_state = warm._read_gzip_json(final_state_path)
            final_state["agents"][0]["path"][-1] = 2
            warm._atomic_write_gzip_json(final_state_path, final_state)
            result["final_state_sha256"] = warm._sha256(final_state_path)
            result["final_paths_sha256"] = warm._paths_sha256([[0, 2]])
            warm._write_json(result_path, result)
            with self.assertRaisesRegex(
                RuntimeError, "path endpoints differ"
            ):
                warm._validate_completed_result(result_path, status_path, job)

    def test_restart_cross_binds_result_and_config_job_manifests(self) -> None:
        run_identity = {"schema": warm.SCHEMA, "run": 1}
        run_fingerprint = warm._fingerprint(run_identity)
        job_identity = {
            "schema": warm.SCHEMA,
            "run_fingerprint": run_fingerprint,
        }
        job_fingerprint = warm._fingerprint(job_identity)
        configured = {
            "job_id": "job",
            "job_fingerprint": job_fingerprint,
            "job_identity": job_identity,
            "portfolio_key": "p" * 64,
            "source_seed": 5,
            "continuation_seed": 7,
        }
        result = {
            "schema": warm.SCHEMA,
            "status": "complete",
            "job_id": "job",
            "run_fingerprint": run_fingerprint,
            **{key: value for key, value in configured.items() if key != "job_identity"},
            "continuation_seed": 8,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warm._write_json(
                root / "run_config.json",
                {
                    "schema": warm.SCHEMA,
                    "run_fingerprint": run_fingerprint,
                    "run_identity": run_identity,
                    "jobs": [configured],
                },
            )
            warm._write_json(
                root / "results.json",
                {
                    "schema": warm.SCHEMA,
                    "run_fingerprint": run_fingerprint,
                    "jobs": [result],
                },
            )
            with self.assertRaisesRegex(ValueError, "continuation_seed mismatch"):
                warm._restart_sources(
                    root / "results.json", [], best_per_source=False
                )

    def test_restart_rejects_checkpoint_from_another_job_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, result_path, _status_path = _write_valid_completed_job(root)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            results_path = root / "results.json"
            warm._write_json(
                results_path,
                {
                    "schema": warm.SCHEMA,
                    "run_fingerprint": job["run_fingerprint"],
                    "jobs": [result],
                },
            )
            restarted = warm._restart_sources(
                results_path, [job["source"]], best_per_source=False
            )
            self.assertEqual(len(restarted), 1)

            original_state = warm._read_gzip_json(
                root / result["final_state_file"]
            )
            other_state_path = (
                root / "jobs" / "other-job" / "final_state.json.gz"
            )
            warm._atomic_write_gzip_json(other_state_path, original_state)
            result["final_state_file"] = other_state_path.relative_to(
                root
            ).as_posix()
            result["final_state_sha256"] = warm._sha256(other_state_path)
            warm._write_json(result_path, result)
            warm._write_json(
                results_path,
                {
                    "schema": warm.SCHEMA,
                    "run_fingerprint": job["run_fingerprint"],
                    "jobs": [result],
                },
            )
            with self.assertRaisesRegex(
                ValueError, "completed artifact is invalid"
            ):
                warm._restart_sources(
                    results_path, [job["source"]], best_per_source=False
                )

    def test_restored_terminal_state_gets_an_immediate_stop_reason(self) -> None:
        self.assertEqual(
            warm._terminal_stop_reason({"feasible": True, "done": True}),
            "feasible",
        )
        self.assertEqual(
            warm._terminal_stop_reason({"feasible": False, "done": True}),
            "time_limit",
        )
        self.assertIsNone(
            warm._terminal_stop_reason({"feasible": False, "done": False})
        )

    def test_aggregate_rejects_truthy_nonboolean_success(self) -> None:
        job = {
            "job_id": "job",
            "job_fingerprint": "b" * 64,
            "portfolio_key": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warm._write_json(
                root / "jobs" / "job" / "status.json",
                {
                    "schema": warm.SCHEMA,
                    "status": "complete",
                    "job_id": "job",
                    "run_fingerprint": "a" * 64,
                    "job_fingerprint": job["job_fingerprint"],
                    "portfolio_key": job["portfolio_key"],
                    "success": "false",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "success is not boolean"):
                warm._aggregate_status(root, [job], "a" * 64)


if __name__ == "__main__":
    unittest.main()
