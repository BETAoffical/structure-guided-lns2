import copy
import json
import unittest
from scripts import repair_sa_hard_reference_report as fix


class ReferenceReportTests(unittest.TestCase):
    def raw(self):
        value = dict(behavior=[dict(sizes={8:3,16:2},minimum_post_conflicts=6)],success=8,mean=1.25)
        return json.loads(json.dumps(fix.run.sealed(value)))

    def test_exact_integer_key_cause_and_fix(self):
        raw = self.raw()
        with self.assertRaises(ValueError): fix.run.check_seal(raw)
        canonical = fix.canonical_report(raw)
        fix.run.check_seal(json.loads(json.dumps(canonical)))
        self.assertNotEqual(raw["integrity"],canonical["integrity"])

    def test_no_scientific_or_source_mutation(self):
        raw = self.raw()
        before = copy.deepcopy(raw)
        fixed = fix.canonical_report(raw)
        self.assertEqual(before,raw)
        self.assertEqual({k:v for k,v in before.items() if k != "integrity"},
                         {k:v for k,v in fixed.items() if k != "integrity"})

    def test_unrelated_tamper_rejected(self):
        raw = self.raw()
        raw["success"] = 9
        with self.assertRaises(ValueError): fix.canonical_report(raw)
        raw = self.raw()
        raw["behavior"][0]["sizes"]["16"] = 99
        with self.assertRaises(ValueError): fix.canonical_report(raw)

    def test_ambiguous_keys_rejected(self):
        raw = self.raw()
        raw["behavior"][0]["sizes"]["08"] = raw["behavior"][0]["sizes"].pop("8")
        with self.assertRaises(ValueError): fix.canonical_report(raw)


if __name__ == "__main__": unittest.main()
