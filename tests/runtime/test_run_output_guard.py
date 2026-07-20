from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_output_guard import prepare_run_output


class RunOutputGuardTests(unittest.TestCase):
    def test_new_output_records_identity_and_matching_resume_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            identity = {"runner": "test", "mode": "quick", "fingerprints": ["a"]}
            created = prepare_run_output(output, resume=False, identity=identity)
            config = output / "runner_config.json"
            before = hashlib.sha256(config.read_bytes()).hexdigest()

            resumed = prepare_run_output(output, resume=True, identity=identity)

            self.assertEqual(created, resumed)
            self.assertEqual(before, hashlib.sha256(config.read_bytes()).hexdigest())

    def test_nonempty_output_requires_explicit_resume_without_touching_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            status = output / "status.json"
            status.write_text('{"status":"complete"}\n', encoding="utf-8")
            before = status.read_bytes()

            with self.assertRaisesRegex(ValueError, "pass --resume"):
                prepare_run_output(
                    output,
                    resume=False,
                    identity={"runner": "test", "mode": "quick"},
                )

            self.assertEqual(before, status.read_bytes())
            self.assertFalse((output / "runner_config.json").exists())

    def test_resume_rejects_missing_or_mismatched_identity_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            sentinel = output / "run.log"
            sentinel.write_text("old log\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "choose a new --output"):
                prepare_run_output(
                    output,
                    resume=True,
                    identity={"runner": "test", "mode": "quick"},
                )
            self.assertEqual("old log\n", sentinel.read_text(encoding="utf-8"))

            output = Path(directory) / "protected"
            original = {"runner": "test", "mode": "quick"}
            prepare_run_output(output, resume=False, identity=original)
            config = output / "runner_config.json"
            before = config.read_bytes()
            with self.assertRaisesRegex(ValueError, "different mode"):
                prepare_run_output(
                    output,
                    resume=True,
                    identity={"runner": "test", "mode": "formal"},
                )
            self.assertEqual(before, config.read_bytes())
            self.assertEqual(
                original,
                json.loads(config.read_text(encoding="utf-8"))["identity"],
            )


if __name__ == "__main__":
    unittest.main()
