import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments._common import sha256_file
from scripts.run_sa_paired_closed_loop import once, sealed
from scripts.run_sa_policy_aligned_update import load_updated_model


class LoaderTest(unittest.TestCase):
    def test_actual_loader_uses_base_not_expanded_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = ["budget.remaining", "degree"]
            payload = dict(feature_names=["delta."+n for n in base]+["mean."+n for n in base])
            once(root/"model/bundle.json", payload)
            files = {"bundle.json":sha256_file(root/"model/bundle.json")}
            once(root/"model/training.json", sealed(dict(binding="b", feature_names=base, files=files)))
            once(root/"model/receipt.json", sealed(dict(binding="b", files=files, parity_verified=True,
                training_sha256=sha256_file(root/"model/training.json"))))
            with patch("scripts.run_sa_policy_aligned_update.portable_model", return_value="loaded") as factory:
                self.assertEqual(load_updated_model(root, dict(binding="b"), native=False), "loaded")
                factory.assert_called_once_with(payload, base, native=False)
            with self.assertRaisesRegex(ValueError, "identity"):
                load_updated_model(root, dict(binding="other"), native=False)


if __name__ == "__main__":
    unittest.main()
