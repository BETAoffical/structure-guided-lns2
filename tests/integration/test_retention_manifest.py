from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "configs" / "retention_manifest.json"
PRODUCTION_ROOTS = {"experiments", "generators", "lns2_selector", "scripts"}
EXECUTABLE_ROLES = {"active", "shared", "reproducibility", "compatibility"}


def _repository_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        value
        for value in result.stdout.split("\0")
        if value and (PROJECT_ROOT / value).is_file()
    }


class RetentionManifestTests(unittest.TestCase):
    def test_manifest_classifies_every_production_module_and_pytest_file(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "lns2.retention_manifest.v1")
        self.assertEqual(payload["schema_version"], 1)

        entries = list(payload["entries"])
        paths = [str(entry["path"]) for entry in entries]
        self.assertEqual(len(paths), len(set(paths)), "duplicate manifest paths")

        repository_files = _repository_files()
        expected = {
            path
            for path in repository_files
            if (
                path.endswith(".py")
                and path.split("/", 1)[0] in PRODUCTION_ROOTS
            )
            or (
                path.startswith("tests/")
                and Path(path).name.startswith("test_")
                and path.endswith(".py")
            )
        }
        self.assertEqual(set(paths), expected)

        for entry in entries:
            path = str(entry["path"])
            self.assertIn(entry["role"], EXECUTABLE_ROLES, path)
            self.assertTrue((PROJECT_ROOT / path).is_file(), path)
            field = "contract" if path.startswith("tests/") else "purpose"
            self.assertTrue(str(entry.get(field, "")).strip(), path)


if __name__ == "__main__":
    unittest.main()
