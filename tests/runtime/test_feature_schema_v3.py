from __future__ import annotations

import unittest

from experiments.feature_schema_v2 import (
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    PROFILE_FEATURE_NAMES,
)
from experiments.feature_schema_v3 import (
    V3_FEATURE_NAMES,
    V3_FEATURE_SCHEMA_ID,
    V3_FEATURE_SCHEMA_SHA256,
    V3_REMOVED_TRAINING_FEATURE_NAMES,
    resolve_v3_feature_names,
    v3_feature_schema_manifest,
)


class V3FeatureSchemaTests(unittest.TestCase):
    def test_fixed_projection_contains_94_features(self) -> None:
        source = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
        self.assertEqual(len(source), 124)
        self.assertEqual(len(V3_REMOVED_TRAINING_FEATURE_NAMES), 30)
        self.assertEqual(len(V3_FEATURE_NAMES), 94)
        self.assertEqual(
            V3_FEATURE_NAMES,
            tuple(
                name
                for name in source
                if name not in V3_REMOVED_TRAINING_FEATURE_NAMES
            ),
        )

    def test_manifest_is_self_consistent(self) -> None:
        manifest = v3_feature_schema_manifest()
        self.assertEqual(manifest["schema"], V3_FEATURE_SCHEMA_ID)
        self.assertEqual(manifest["sha256"], V3_FEATURE_SCHEMA_SHA256)
        self.assertEqual(tuple(manifest["feature_names"]), V3_FEATURE_NAMES)

    def test_current_and_historical_v3_schemas_resolve(self) -> None:
        self.assertEqual(
            resolve_v3_feature_names(
                V3_FEATURE_SCHEMA_ID, V3_FEATURE_SCHEMA_SHA256
            ),
            V3_FEATURE_NAMES,
        )
        self.assertEqual(
            resolve_v3_feature_names(FEATURE_SCHEMA_ID, FEATURE_SCHEMA_SHA256),
            tuple(PROFILE_FEATURE_NAMES["realized_dynamic"]),
        )

    def test_unknown_or_tampered_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
            resolve_v3_feature_names(V3_FEATURE_SCHEMA_ID, "wrong")


if __name__ == "__main__":
    unittest.main()
