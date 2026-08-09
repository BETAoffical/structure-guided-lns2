from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments._common import read_json, sha256_file, write_json
from experiments.run_output_guard import (
    load_completed_report,
    prepare_resumable_output,
    prepare_run_output,
)


class RunOutputGuardTests(unittest.TestCase):
    @staticmethod
    def _producer(source_hash: str = "0" * 64) -> dict[str, object]:
        return {
            "schema": "lns2.producer_identity.v2",
            "source_sha256": {"runner.py": source_hash},
            "python": {"implementation": "CPython", "version": "3.10.0"},
            "packages": {},
            "native_required": False,
            "native": None,
        }

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

    def test_resumable_output_binds_config_schedule_and_producer_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            output = root / "output"
            config.write_text('{"version":1}\n', encoding="utf-8")
            schedule = [{"job": 1}, {"job": 2}]
            created = prepare_resumable_output(
                output,
                status_filename="status.json",
                status_schema="test.status.v1",
                config_path=config,
                schedule=schedule,
                producer=self._producer(),
                resume=False,
                report_filename="report.json",
                label="test run",
            )
            self.assertFalse(created.resumed)
            before = {
                path.name: path.read_bytes()
                for path in output.iterdir()
                if path.is_file()
            }

            resumed = prepare_resumable_output(
                output,
                status_filename="status.json",
                status_schema="test.status.v1",
                config_path=config,
                schedule=schedule,
                producer=self._producer(),
                resume=True,
                report_filename="report.json",
                label="test run",
            )
            self.assertTrue(resumed.resumed)
            self.assertEqual(
                before,
                {
                    path.name: path.read_bytes()
                    for path in output.iterdir()
                    if path.is_file()
                },
            )

            mutations = (
                ([{"job": 2}, {"job": 1}], self._producer()),
                (schedule, self._producer("1" * 64)),
            )
            for changed_schedule, changed_producer in mutations:
                with self.assertRaisesRegex(ValueError, "implementation fingerprint"):
                    prepare_resumable_output(
                        output,
                        status_filename="status.json",
                        status_schema="test.status.v1",
                        config_path=config,
                        schedule=changed_schedule,
                        producer=changed_producer,
                        resume=True,
                        report_filename="report.json",
                        label="test run",
                    )
                self.assertEqual(
                    before,
                    {
                        path.name: path.read_bytes()
                        for path in output.iterdir()
                        if path.is_file()
                    },
                )

            config.write_text('{"version":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "implementation fingerprint"):
                prepare_resumable_output(
                    output,
                    status_filename="status.json",
                    status_schema="test.status.v1",
                    config_path=config,
                    schedule=schedule,
                    producer=self._producer(),
                    resume=True,
                    report_filename="report.json",
                    label="test run",
                )

    def test_completed_report_is_hash_verified_and_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            output = root / "output"
            config.write_text("{}\n", encoding="utf-8")
            prepared = prepare_resumable_output(
                output,
                status_filename="status.json",
                status_schema="test.status.v1",
                config_path=config,
                schedule=[{"job": 1}],
                producer=self._producer(),
                resume=False,
                report_filename="report.json",
                report_schema="test.report.v1",
            )
            report_path = output / "report.json"
            report = {
                "schema": "test.report.v1",
                "result": "ok",
                "producer_identity": self._producer(),
            }
            write_json(report_path, report)
            write_json(
                output / "status.json",
                {
                    **prepared.base_status,
                    "completed_schedule_entries": 1,
                    "complete": True,
                    "report_sha256": sha256_file(report_path),
                },
            )
            before = report_path.read_bytes()
            self.assertEqual(
                load_completed_report(
                    output,
                    status_filename="status.json",
                    report_filename="report.json",
                    status_schema="test.status.v1",
                    report_schema="test.report.v1",
                ),
                report,
            )
            self.assertEqual(before, report_path.read_bytes())
            resumed = prepare_resumable_output(
                output,
                status_filename="status.json",
                status_schema="test.status.v1",
                config_path=config,
                schedule=[{"job": 1}],
                producer=self._producer(),
                resume=True,
                report_filename="report.json",
                report_schema="test.report.v1",
            )
            self.assertEqual(resumed.completed_report, report)
            report_path.write_text('{"result":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed report changed"):
                load_completed_report(
                    output,
                    status_filename="status.json",
                    report_filename="report.json",
                )
            write_json(report_path, report)
            write_json(
                output / "status.json",
                {
                    **prepared.base_status,
                    "completed_schedule_entries": 0,
                    "complete": True,
                    "report_sha256": sha256_file(report_path),
                },
            )
            with self.assertRaisesRegex(ValueError, "incomplete schedule coverage"):
                load_completed_report(
                    output,
                    status_filename="status.json",
                    report_filename="report.json",
                )

            write_json(
                report_path,
                {**report, "producer_identity": self._producer("1" * 64)},
            )
            write_json(
                output / "status.json",
                {
                    **prepared.base_status,
                    "completed_schedule_entries": 1,
                    "complete": True,
                    "report_sha256": sha256_file(report_path),
                },
            )
            with self.assertRaisesRegex(ValueError, "producer identity changed"):
                load_completed_report(
                    output,
                    status_filename="status.json",
                    report_filename="report.json",
                )

    def test_strict_json_rejects_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.json"
            invalid.write_text('{"metric":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_json(invalid)
            output = root / "output.json"
            with self.assertRaises(ValueError):
                write_json(output, {"metric": float("inf")})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
