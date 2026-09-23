import ast
import unittest
from scripts import register_sa_support_audit_fix as fix


class AuditFixTests(unittest.TestCase):
    def test_only_audit_changes_allowed(self):
        old = '\n'.join(f'def {n}():\n    return 1' for n in ('verify', 'audit_worker', 'analyze', 'worker'))
        new = old.replace('return 1', 'return 2', 3)
        self.assertEqual(fix.audit_only_difference(old, new), sorted(fix.ALLOWED))
        with self.assertRaises(ValueError):
            fix.audit_only_difference(old, new.replace('return 1', 'return 2'))
        with self.assertRaises(ValueError):
            fix.audit_only_difference(old, new + '\nNODE_BUDGET=1000000000')

    def test_state_refresh_precedes_scoring(self):
        tree = ast.parse((fix.ROOT / fix.TARGET).read_text(encoding='utf8'))
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_worker')
        loop = next(n for n in worker.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'event')
        calls = [ast.unparse(n) for n in loop.body]
        refresh = calls.index('engine.prepare(state)')
        extraction = next(i for i, v in enumerate(calls) if v.startswith('feature_rows, _ ='))
        self.assertLess(refresh, extraction)

    def test_original_collector_ast_unchanged(self):
        import subprocess
        old = subprocess.check_output(['git', 'show', '027cf62:' + fix.TARGET], cwd=fix.ROOT, text=True)
        current = (fix.ROOT / fix.TARGET).read_text(encoding='utf8')
        self.assertEqual(fix.audit_only_difference(old, current), sorted(fix.ALLOWED))


if __name__ == '__main__':
    unittest.main()
