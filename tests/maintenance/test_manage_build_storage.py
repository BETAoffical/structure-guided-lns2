from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.manage_build_storage import (
    MANIFEST_SCHEMA,
    build_plan,
    collect_target_files,
    plan_is_current,
    verify_manifest,
)


class ManageBuildStorageTests(unittest.TestCase):
    def _config(self) -> dict[str, object]:
        return {
            "schema": "lns2.build_storage_compaction.v1",
            "schema_version": 1,
            "minimum_file_bytes": 1,
            "minimum_projected_savings_bytes": 1,
            "extensions": [".json", ".jsonl"],
            "target_roots": ["build/data"],
            "excluded_path_parts": ["models"],
        }

    def test_plan_only_selects_registered_text_and_estimates_savings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "build" / "data"
            data.mkdir(parents=True)
            (data / "rows.jsonl").write_text("{\"value\":1}\n" * 1000, encoding="utf-8")
            (data / "binary.bin").write_bytes(b"x" * 1000)
            (data / "models").mkdir()
            (data / "models" / "ignored.json").write_text("{}", encoding="utf-8")

            files = collect_target_files(root, self._config())
            self.assertEqual([data / "rows.jsonl"], files)
            plan = build_plan(root, self._config())
            self.assertEqual(1, plan["file_count"])
            self.assertGreater(plan["projected_savings_bytes"], 0)
            self.assertTrue(plan["compression_authorized"])
            self.assertTrue(plan_is_current(root, self._config(), plan))
            (data / "rows.jsonl").write_text("changed", encoding="utf-8")
            self.assertFalse(plan_is_current(root, self._config(), plan))

    def test_target_must_remain_below_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "outside").mkdir()
            config = self._config()
            config["target_roots"] = ["outside"]
            with self.assertRaisesRegex(ValueError, "below build"):
                collect_target_files(root, config)

    def test_manifest_verification_detects_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "build" / "data.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"value": 1}), encoding="utf-8")
            import hashlib

            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "files": [{"path": "build/data.json", "sha256": expected}],
            }
            self.assertTrue(verify_manifest(root, manifest)["passed"])
            path.write_text(json.dumps({"value": 2}), encoding="utf-8")
            self.assertFalse(verify_manifest(root, manifest)["passed"])


if __name__ == "__main__":
    unittest.main()
