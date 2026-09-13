import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import diagnose_pbs_repair as audit


class PBSAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.before = dict(agents=[dict(id=0, path=[0,1]), dict(id=1, path=[2,1]), dict(id=2, path=[3])],
                           conflict_edges=[[0,1]])
        self.metrics = dict(action_valid=True, step_applied=True, neighborhood=[0,1], replan_success=False)
        self.validation = patch("scripts.run_feedback_exploration_diagnostics.validate_final")
        self.validation.start()
        self.addCleanup(self.validation.stop)

    def test_rollback_passes(self):
        audit.validate_transition(self.before, copy.deepcopy(self.before), self.metrics, [0,1])

    def test_external_change_rejected(self):
        after = copy.deepcopy(self.before)
        after["agents"][2]["path"] = [3,4]
        with self.assertRaisesRegex(ValueError, "external paths"):
            audit.validate_transition(self.before, after, self.metrics, [0,1])

    def test_failed_selected_path_change_rejected(self):
        after = copy.deepcopy(self.before)
        after["agents"][0]["path"] = [0,0,1]
        with self.assertRaisesRegex(ValueError, "roll back"):
            audit.validate_transition(self.before, after, self.metrics, [0,1])

    def test_explicit_set_change_rejected(self):
        self.metrics["neighborhood"] = [0,2]
        with self.assertRaisesRegex(ValueError, "explicit set"):
            audit.validate_transition(self.before, self.before, self.metrics, [0,1])

    def test_invalid_action_rejected(self):
        self.metrics["action_valid"] = False
        with self.assertRaisesRegex(ValueError, "invalid action"):
            audit.validate_transition(self.before, self.before, self.metrics, [0,1])

    def test_plan_tamper_rejected_before_native_read(self):
        with self.assertRaisesRegex(ValueError, "plan changed"):
            audit.verify(dict(binding="wrong", no_ttf=True))

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "row.json"
            audit.write(path, dict(status="crash", returncode=-11))
            self.assertEqual(audit.read(path)["returncode"], -11)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
