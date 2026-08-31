from __future__ import annotations

import unittest
import types
from unittest.mock import patch

from scripts.check_environment import TRAINING_VERSIONS, environment_report


class EnvironmentCheckTests(unittest.TestCase):
    def test_training_profile_requires_the_frozen_versions(self) -> None:
        with patch("scripts.check_environment.platform.system", return_value="Windows"), patch(
            "scripts.check_environment._package_version",
            side_effect=lambda name: TRAINING_VERSIONS.get(name),
        ):
            report = environment_report("training-windows")
        self.assertTrue(report["passed"])
        self.assertEqual(report["required_failure_count"], 0)
        self.assertFalse(report["installation_performed"])

    def test_missing_training_package_is_reported_without_installing(self) -> None:
        with patch("scripts.check_environment.platform.system", return_value="Windows"), patch(
            "scripts.check_environment._package_version",
            side_effect=lambda name: None if name == "scikit-learn" else TRAINING_VERSIONS[name],
        ):
            report = environment_report("training-windows")
        self.assertFalse(report["passed"])
        self.assertEqual(report["required_failure_count"], 1)
        self.assertFalse(report["installation_performed"])

    def test_runtime_profile_requires_native_timing_schema_v2(self) -> None:
        module = types.SimpleNamespace(
            repair_timing_schema="lns2.repair_timing.v2",
            native_semantics_schema=(
                "lns2.native_semantics.official_step_timed_extension.v3"
            ),
            __file__="/tmp/lns2_env.so",
            LNS2RepairEnv=type(
                "Environment",
                (),
                {
                    "get_last_reset_timings": lambda self: {},
                    "propose_batch_compact": lambda self, actions: [],
                    "propose_seed_grid_grouped": lambda self, *args: {},
                },
            ),
            PortableTreeEnsemble=lambda: None,
            batch_online_features=lambda *args: None,
            batch_online_feature_vectors=lambda *args: None,
            topology_conflict_events=lambda *args: None,
        )
        with patch("scripts.check_environment.platform.system", return_value="Linux"), patch(
            "scripts.check_environment.platform.release", return_value="microsoft-standard"
        ), patch(
            "scripts.check_environment._package_version", return_value="installed"
        ), patch(
            "scripts.check_environment.importlib.import_module", return_value=module
        ):
            report = environment_report("runtime-wsl")
        timing = next(
            row
            for row in report["checks"]
            if row["name"] == "lns2_env:repair-timing-schema"
        )
        self.assertTrue(timing["passed"])
        self.assertEqual(timing["expected"], "lns2.repair_timing.v2")
        semantics = next(
            row
            for row in report["checks"]
            if row["name"] == "lns2_env:native-semantics-schema"
        )
        self.assertTrue(semantics["passed"])


if __name__ == "__main__":
    unittest.main()
