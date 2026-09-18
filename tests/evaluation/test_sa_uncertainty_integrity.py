import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments._common import json_fingerprint
from experiments.sa_uncertainty_audit import fit_member
from scripts.run_sa_paired_closed_loop import sealed, once
from scripts.run_sa_uncertainty_audit import check_fit, verify_outputs
from tests.evaluation.test_sa_low_complexity_ranker import fixture


class IntegrityTests(unittest.TestCase):
    def test_resealed_wrong_fit_metadata_is_rejected(self):
        data = fixture()
        plan = dict(binding=json_fingerprint("test"), config=dict(members=20, seed=4))
        base = dict(fit_member(data, "held", 0, plan["config"]), binding=plan["binding"])
        check_fit(plan, data, "held", 0, sealed(base))
        variants = []
        for key, value in (("state_weights", {}), ("sklearn", "changed"), ("constant_targets", not base["constant_targets"])):
            variants.append(dict(base, **{key: value}))
        missing = copy.deepcopy(base)
        missing["predictions"][0]["scores"].pop("0")
        variants.append(missing)
        wrong = copy.deepcopy(base)
        wrong["predictions"][0]["selected"] = "missing"
        variants.append(wrong)
        for row in variants:
            with self.assertRaises(ValueError):
                check_fit(plan, data, "held", 0, sealed(row))

    def test_verify_outputs_rejects_missing_or_resealed_report(self):
        with TemporaryDirectory() as td:
            out = Path(td)
            with self.assertRaises(FileNotFoundError):
                verify_outputs({}, out)
            expected = dict(binding="b", fits=168, means={"frozen": .5})
            once(out / "report.json", sealed(dict(expected, fits=167)))
            with patch("scripts.run_sa_uncertainty_audit.build_report", return_value=expected):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    verify_outputs({}, out)


if __name__ == "__main__":
    unittest.main()
