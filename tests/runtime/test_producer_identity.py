from __future__ import annotations

import importlib.metadata
import platform
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experiments._common import (
    NATIVE_SEMANTICS_SCHEMA,
    contained_file,
    producer_identity,
    validate_producer_identity,
)
from experiments.run_output_guard import prepare_run_output


class ProducerIdentityResumeTest(unittest.TestCase):
    @unittest.skipUnless(
        platform.system() == "Windows", "extended paths are Windows-specific"
    )
    def test_contained_file_supports_a_long_windows_output_path(self) -> None:
        directory = tempfile.mkdtemp()
        try:
            root = Path(directory).resolve()
            relative = Path(
                *(f"descriptive-episode-segment-{index:02d}" for index in range(9)),
                "trace.jsonl.gz",
            )
            ordinary = root / relative
            self.assertGreater(len(str(ordinary)), 260)
            extended = Path("\\\\?\\" + str(ordinary))
            extended.parent.mkdir(parents=True)
            extended.write_bytes(b"trace")

            resolved = contained_file(root, relative.as_posix(), field="trace")

            self.assertEqual(resolved.read_bytes(), b"trace")
        finally:
            shutil.rmtree("\\\\?\\" + directory)

    def test_required_package_must_have_a_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "producer.py").write_text("source\n", encoding="utf-8")
            with mock.patch(
                "experiments._common.importlib.metadata.version",
                side_effect=importlib.metadata.PackageNotFoundError(
                    "missing-package"
                ),
            ), mock.patch(
                "experiments._common.importlib.import_module",
                side_effect=ImportError("missing-package"),
            ):
                with self.assertRaisesRegex(ValueError, "lacks package version"):
                    producer_identity(
                        project_root=project,
                        source_files=("producer.py",),
                        native_required=False,
                        package_names=("missing-package",),
                    )

                identity = producer_identity(
                    project_root=project,
                    source_files=("producer.py",),
                    native_required=False,
                    optional_package_names=("missing-package",),
                )
            self.assertIsNone(identity["packages"]["missing-package"])

    def test_source_native_and_package_changes_reject_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            output = root / "output"
            project.mkdir()
            source = project / "producer.py"
            native = root / "lns2_env.pyd"
            source.write_text("source-one\n", encoding="utf-8")
            native.write_bytes(b"native-one")
            module = types.SimpleNamespace(
                __file__=str(native),
                repair_timing_schema="lns2.repair_timing.v2",
                native_semantics_schema=NATIVE_SEMANTICS_SCHEMA,
            )

            def identity(*, sklearn: str = "2") -> dict:
                with mock.patch(
                    "experiments._common.importlib.metadata.version",
                    side_effect=lambda name: {
                        "numpy": "1",
                        "scikit-learn": sklearn,
                    }[name],
                ), mock.patch(
                    "experiments._common.importlib.import_module",
                    return_value=module,
                ):
                    return producer_identity(
                        project_root=project,
                        source_files=("producer.py",),
                        native_required=True,
                        package_names=("numpy", "scikit-learn"),
                    )

            baseline = identity()
            prepare_run_output(
                output,
                resume=False,
                identity={"runner": "test", "producer_identity": baseline},
            )
            config_path = output / "runner_config.json"
            before = config_path.read_bytes()

            source.write_text("source-two\n", encoding="utf-8")
            source_changed = identity()
            source.write_text("source-one\n", encoding="utf-8")
            native.write_bytes(b"native-two")
            native_changed = identity()
            native.write_bytes(b"native-one")
            package_changed = identity(sklearn="3")

            for changed in (
                source_changed,
                native_changed,
                package_changed,
            ):
                with self.assertRaisesRegex(
                    ValueError, "implementation fingerprint"
                ):
                    prepare_run_output(
                        output,
                        resume=True,
                        identity={
                            "runner": "test",
                            "producer_identity": changed,
                        },
                    )
                self.assertEqual(config_path.read_bytes(), before)

    def test_native_semantics_schema_is_required_and_strict(self) -> None:
        identity = {
            "schema": "lns2.producer_identity.v2",
            "source_sha256": {"producer.py": "0" * 64},
            "python": {"implementation": "CPython", "version": "3.10.0"},
            "packages": {},
            "native_required": True,
            "native": {
                "path": "lns2_env.so",
                "sha256": "1" * 64,
                "repair_timing_schema": "lns2.repair_timing.v2",
                "native_semantics_schema": NATIVE_SEMANTICS_SCHEMA,
            },
        }
        self.assertEqual(
            validate_producer_identity(identity, native_required=True),
            identity,
        )
        missing = {
            **identity,
            "native": {
                key: value
                for key, value in identity["native"].items()
                if key != "native_semantics_schema"
            },
        }
        with self.assertRaisesRegex(ValueError, "semantics schema"):
            validate_producer_identity(missing, native_required=True)
        coerced = {**identity, "native_required": 1}
        with self.assertRaisesRegex(ValueError, "native-required"):
            validate_producer_identity(coerced, native_required=True)


if __name__ == "__main__":
    unittest.main()
