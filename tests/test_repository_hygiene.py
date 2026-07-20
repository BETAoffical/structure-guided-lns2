from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from experiments._common import (
    episode_id,
    state_storage_id,
    trial_job_id,
)
from experiments.closed_loop_confirmation import CONTROLLER_IMPLEMENTATION_FILES
from experiments.policy_visited_aggregation import (
    IMPLEMENTATION_FILES as POLICY_IMPLEMENTATION_FILES,
)
from experiments.repair_order_probe import (
    IMPLEMENTATION_FILES as REPAIR_ORDER_IMPLEMENTATION_FILES,
)
from experiments.sequential_credit_audit import (
    IMPLEMENTATION_FILES as SEQUENTIAL_IMPLEMENTATION_FILES,
)
from scripts.audit_repository_hygiene import (
    PROJECT_ROOT,
    _repository_path,
    build_cleanup_plan,
    duplicate_function_groups,
    load_config,
    post_cleanup_report,
    run_check,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RepositoryHygieneTests(unittest.TestCase):
    def test_current_controlled_files_pass_hygiene_check(self) -> None:
        report = run_check(PROJECT_ROOT, load_config())
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["evidence"]["entry_count"], 24)

    def test_common_identifiers_preserve_registered_shapes(self) -> None:
        self.assertEqual(
            episode_id({"task_id": "task"}, 7, "Adaptive"),
            "task__seed_0007__Adaptive",
        )
        self.assertEqual(state_storage_id("state"), "state-0f4b9f34e988ff56")
        self.assertEqual(
            trial_job_id("state", "candidate", 3),
            "state-0f4b9f34e988ff56__candidate__trial_0003",
        )

    def test_common_module_is_covered_by_all_relevant_fingerprints(self) -> None:
        for files in (
            CONTROLLER_IMPLEMENTATION_FILES,
            POLICY_IMPLEMENTATION_FILES,
            REPAIR_ORDER_IMPLEMENTATION_FILES,
            SEQUENTIAL_IMPLEMENTATION_FILES,
        ):
            self.assertIn("experiments/_common.py", files)

    def test_repository_paths_cannot_escape(self) -> None:
        with self.assertRaises(ValueError):
            _repository_path(PROJECT_ROOT, "../outside")
        with self.assertRaises(ValueError):
            _repository_path(PROJECT_ROOT, str(PROJECT_ROOT.resolve()))

    def test_duplicate_function_detection_is_structural(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            for name in ("first.py", "second.py"):
                (root / "experiments" / name).write_text(
                    "def value(number):\n    return number + 1\n",
                    encoding="utf-8",
                )
            groups = duplicate_function_groups(
                root,
                ["experiments/first.py", "experiments/second.py"],
                {"experiments"},
            )
            self.assertEqual(len(groups), 1)
            self.assertEqual({row["name"] for row in groups[0]}, {"value"})

    def test_build_plan_protects_evidence_and_never_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "build" / "formal" / "result.json"
            smoke = root / "build" / "collector-smoke" / "trace.jsonl"
            environment = root / "build" / "venv-graph" / "python.exe"
            _write_json(formal, {"decision": "frozen"})
            smoke.parent.mkdir(parents=True)
            smoke.write_text("{}\n", encoding="utf-8")
            environment.parent.mkdir(parents=True)
            environment.write_bytes(b"python")
            result_config = root / "configs" / "result_consolidation.json"
            _write_json(
                result_config,
                {
                    "experiments": [
                        {
                            "id": "formal",
                            "source": {
                                "path": "build/formal/result.json",
                                "sha256": _sha256(formal),
                            },
                        }
                    ]
                },
            )
            config = {
                "result_consolidation_config": "configs/result_consolidation.json",
                "build": {
                    "fixed_protected_roots": ["venv-graph"],
                    "temporary_name_patterns": ["(^|[-_])smoke([-_]|$)"],
                    "temporary_exact_roots": [],
                    "cache_directory_names": ["__pycache__"],
                    "incomplete_file_suffixes": [".tmp", ".part"],
                    "dependency_metadata_names": ["run_config.json"],
                    "maximum_dependency_json_bytes": 1024 * 1024,
                },
            }
            output = root / "build" / "hygiene"
            inventory, plan = build_cleanup_plan(root, config, output)
            protected = {Path(row["path"]).name for row in plan["protected_roots"]}
            deleted = {Path(row["path"]).name for row in plan["delete_roots"]}
            self.assertIn("formal", protected)
            self.assertIn("venv-graph", protected)
            self.assertIn("hygiene", protected)
            self.assertEqual(deleted, {"collector-smoke"})
            self.assertFalse(plan["deletion_supported"])
            self.assertTrue(smoke.is_file())
            self.assertTrue(formal.is_file())
            self.assertTrue(all(not Path(row["path"]).is_absolute() for row in inventory["directories"]))
            _write_json(output / "pre_cleanup_inventory.json", inventory)
            _write_json(output / "cleanup_plan.json", plan)
            shutil.rmtree(smoke.parent)
            _, post = post_cleanup_report(root, config, output)
            self.assertTrue(post["passed"], post)
            self.assertEqual(post["removed_roots"], ["build/collector-smoke"])

    def test_evidence_mismatch_blocks_build_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "build" / "formal" / "result.json"
            _write_json(formal, {"decision": "changed"})
            _write_json(
                root / "configs" / "result_consolidation.json",
                {
                    "experiments": [
                        {
                            "id": "formal",
                            "source": {
                                "path": "build/formal/result.json",
                                "sha256": "0" * 64,
                            },
                        }
                    ]
                },
            )
            config = {
                "result_consolidation_config": "configs/result_consolidation.json",
                "build": {
                    "fixed_protected_roots": [],
                    "temporary_name_patterns": [],
                    "temporary_exact_roots": [],
                    "cache_directory_names": [],
                    "incomplete_file_suffixes": [".tmp"],
                    "dependency_metadata_names": [],
                    "maximum_dependency_json_bytes": 1024,
                },
            }
            with self.assertRaises(RuntimeError):
                build_cleanup_plan(root, config, root / "build" / "hygiene")

    def test_build_plan_classifies_nested_and_conditional_cleanup_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks = root / "build" / "old-run" / "tracks"
            legacy = root / "build" / "legacy" / "episodes"
            tracks.mkdir(parents=True)
            legacy.mkdir(parents=True)
            (tracks / "trace.jsonl").write_bytes(b"track")
            (legacy / "episode.jsonl").write_bytes(b"legacy")
            evidence_path = root / "artifacts" / "migration.json"
            _write_json(
                evidence_path,
                {
                    "schema": "migration.v1",
                    "matching_episode_count": 1,
                    "equivalence_passed": True,
                },
            )
            _write_json(root / "configs" / "result_consolidation.json", {"experiments": []})
            config = {
                "result_consolidation_config": "configs/result_consolidation.json",
                "build": {
                    "fixed_protected_roots": [],
                    "safe_delete_roots": [],
                    "safe_delete_paths": [
                        {"path": "build/old-run/tracks", "reason": "regenerable"}
                    ],
                    "conditional_delete_paths": [
                        {
                            "path": "build/legacy/episodes",
                            "reason": "migrated",
                            "expected_bytes": 6,
                            "verification_json": "artifacts/migration.json",
                            "required_values": {
                                "schema": "migration.v1",
                                "matching_episode_count": 1,
                                "equivalence_passed": True,
                            },
                        }
                    ],
                    "temporary_name_patterns": [],
                    "temporary_exact_roots": [],
                    "cache_directory_names": [],
                    "incomplete_file_suffixes": [".tmp"],
                    "dependency_metadata_names": [],
                    "maximum_dependency_json_bytes": 1024,
                },
            }
            output = root / "build" / "hygiene"
            inventory, plan = build_cleanup_plan(root, config, output)
            self.assertEqual(
                [row["path"] for row in plan["safe_delete_paths"]],
                ["build/old-run/tracks"],
            )
            self.assertEqual(
                [row["path"] for row in plan["conditional_delete_paths"]],
                ["build/legacy/episodes"],
            )
            self.assertTrue(
                plan["conditional_delete_paths"][0]["evidence_preconditions_passed"]
            )
            self.assertFalse(plan["deletion_supported"])
            self.assertTrue(tracks.is_dir())
            self.assertTrue(legacy.is_dir())

            _write_json(output / "pre_cleanup_inventory.json", inventory)
            _write_json(output / "cleanup_plan.json", plan)
            shutil.rmtree(tracks)
            shutil.rmtree(legacy)
            _, post = post_cleanup_report(root, config, output)
            self.assertTrue(post["passed"], post)
            self.assertEqual(
                post["removed_paths"],
                ["build/legacy/episodes", "build/old-run/tracks"],
            )

    def test_conditional_cleanup_is_blocked_when_evidence_does_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "build" / "legacy" / "episodes"
            target.mkdir(parents=True)
            (target / "episode.jsonl").write_bytes(b"legacy")
            _write_json(root / "artifacts" / "migration.json", {"passed": False})
            _write_json(root / "configs" / "result_consolidation.json", {"experiments": []})
            config = {
                "result_consolidation_config": "configs/result_consolidation.json",
                "build": {
                    "fixed_protected_roots": [],
                    "conditional_delete_paths": [
                        {
                            "path": "build/legacy/episodes",
                            "verification_json": "artifacts/migration.json",
                            "required_values": {"passed": True},
                        }
                    ],
                    "temporary_name_patterns": [],
                    "temporary_exact_roots": [],
                    "cache_directory_names": [],
                    "incomplete_file_suffixes": [".tmp"],
                    "dependency_metadata_names": [],
                    "maximum_dependency_json_bytes": 1024,
                },
            }
            _, plan = build_cleanup_plan(root, config, root / "build" / "hygiene")
            self.assertEqual(plan["conditional_delete_paths"], [])
            self.assertEqual(len(plan["blocked_paths"]), 1)
            self.assertTrue(target.is_dir())


if __name__ == "__main__":
    unittest.main()
