from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
