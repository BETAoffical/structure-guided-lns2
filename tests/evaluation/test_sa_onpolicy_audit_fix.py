from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts import register_sa_onpolicy_audit_fix as fix


class AuditAmendmentTests(unittest.TestCase):
    def test_only_exact_two_blocks_may_change(self):
        before = fix.OLD_VERIFY + "\nCOLLECT = 20\n" + fix.OLD_AUDIT
        after = before.replace(fix.OLD_VERIFY, fix.NEW_VERIFY).replace(fix.OLD_AUDIT, fix.NEW_AUDIT)
        fix.prove_scope(before, after)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            fix.prove_scope(before, after.replace("COLLECT = 20", "COLLECT = 21"))

    def test_comparison_excludes_only_delta_extras(self):
        state = dict(agents=[dict(id=2, path=[0, 1])], low_level=dict(generated=3), feasible=True)
        final = dict(state, runtime=1.2, context={"map_id": "tiny"})
        def require(value, message):
            if not value:
                raise ValueError(message)
        def compare(other):
            scope = dict(state=state, folder=Path("."), read_json=lambda _: other,
                         q=SimpleNamespace(state_fingerprint=lambda _: "constant"),
                         result={"final_fingerprint": "constant"}, require=require)
            exec("\n".join(line[4:] for line in fix.NEW_AUDIT.splitlines()), scope)
        compare(final)
        for other in (dict(final, agents=[]), dict(final, low_level={"generated": 4}),
                      dict(final, feasible=False), dict(final, unexpected=1)):
            with self.assertRaisesRegex(ValueError, "final paths"):
                compare(other)


if __name__ == "__main__":
    unittest.main()
