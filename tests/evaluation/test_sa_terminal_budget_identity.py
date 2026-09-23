import tempfile
from pathlib import Path
import unittest
from scripts import audit_sa_terminal_budget as a


class IdentityTests(unittest.TestCase):
    def test_only_newlines_may_differ(self):
        self.assertTrue(a.same_source(b"x\r\ny\r\n", b"x\ny\n"))
        self.assertFalse(a.same_source(b"x\r\nz\r\n", b"x\ny\n"))

    def test_changed_runtime_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "budget.py"
            p.write_text("25", encoding="utf8")
            files = {"budget.py": a.run.sha256_file(p)}
            a.verify_files(files, root)
            p.write_text("30", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "runtime dependency changed"):
                a.verify_files(files, root)

    def test_failed_then_successful_audit_keeps_both_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = a.save_attempt(root, "binding", [dict(job_id="b", status="error"), dict(job_id="a", status="ok")])
            sha = a.run.sha256_file(first)
            second = a.save_attempt(root, "binding", [dict(job_id="b", status="ok"), dict(job_id="a", status="ok")])
            self.assertNotEqual(first, second)
            self.assertEqual(sha, a.run.sha256_file(first))
            row = a.run.check_seal(a.run.read_json(second))
            self.assertEqual([r["job_id"] for r in row["results"]], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
