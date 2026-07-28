from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_lns2_warm_start_feasibility as warm


class WarmStartIdentityTests(unittest.TestCase):
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
        task = {"dataset_root": "/dataset", "task": {"task_id": "t1"}}
        key = warm._fingerprint(task)
        job = {
            "run_fingerprint": "a" * 64,
            "job_fingerprint": "b" * 64,
            "portfolio_key": key,
            "task_identity": task,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path = warm._portfolio_marker_path(root, key)
            warm._write_json(
                marker_path,
                {
                    "schema": warm.SCHEMA,
                    "run_fingerprint": job["run_fingerprint"],
                    "portfolio_key": key,
                    "success": True,
                    "winning_job_id": "job-1",
                    "final_state_fingerprint": "c" * 64,
                },
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
