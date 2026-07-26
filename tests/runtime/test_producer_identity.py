from __future__ import annotations

import importlib.metadata
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experiments._common import producer_identity
from experiments.run_output_guard import prepare_run_output


class ProducerIdentityResumeTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
