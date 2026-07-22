from __future__ import annotations

import unittest

from experiments.v3_feature_audit import (
    feature_group,
    permute_candidate_features,
    project_rows,
)


class V3FeatureGroupingTests(unittest.TestCase):
    def test_feature_groups_are_stable(self) -> None:
        self.assertEqual(feature_group("state.agent_count"), "state")
        self.assertEqual(feature_group("proposal.actual_size"), "proposal.size")
        self.assertEqual(
            feature_group("proposal.family_count=target:4"), "proposal.family"
        )
        self.assertEqual(
            feature_group("proposal.seed_delay_mean"), "proposal.seed"
        )
        self.assertEqual(
            feature_group("realized.path_overlap_mean"), "realized.path"
        )
        self.assertEqual(
            feature_group("realized.internal_conflict_edges"),
            "realized.conflict",
        )


class V3FeaturePermutationTests(unittest.TestCase):
    def rows(self) -> list[dict[str, object]]:
        result = []
        for state, value in (("s0", 1.0), ("s1", 2.0), ("s2", 3.0)):
            for trial in range(2):
                result.append(
                    {
                        "state_id": state,
                        "candidate_id": f"{state}-c",
                        "map_id": "map",
                        "route": "model",
                        "actual_size": 4,
                        "trial": trial,
                        "features": [value, 10.0 + value],
                    }
                )
        return result

    def test_permutation_keeps_paired_trials_and_value_distribution(self) -> None:
        source = self.rows()
        changed = permute_candidate_features(
            source, (0,), namespace="test"
        )
        by_candidate = {}
        for row in changed:
            by_candidate.setdefault(row["candidate_id"], set()).add(
                row["features"][0]
            )
            self.assertEqual(
                row["features"][1],
                next(
                    source_row["features"][1]
                    for source_row in source
                    if source_row["trial"] == row["trial"]
                    and source_row["candidate_id"] == row["candidate_id"]
                ),
            )
        self.assertTrue(all(len(values) == 1 for values in by_candidate.values()))
        self.assertEqual(
            sorted(row["features"][0] for row in source[::2]),
            sorted(row["features"][0] for row in changed[::2]),
        )

    def test_projection_preserves_requested_order(self) -> None:
        projected = project_rows(self.rows(), (1, 0))
        self.assertEqual(projected[0]["features"], [11.0, 1.0])

    def test_size_permutation_can_cross_size_strata(self) -> None:
        rows = self.rows()
        for index, row in enumerate(rows):
            row["actual_size"] = 4 if index < 2 else 8
        changed = permute_candidate_features(
            rows,
            (0,),
            namespace="size-feature",
            stratify_by_actual_size=False,
        )
        self.assertTrue(
            any(
                float(left["features"][0]) != float(right["features"][0])
                for left, right in zip(rows, changed)
            )
        )

    def test_disagreeing_paired_features_are_rejected(self) -> None:
        rows = self.rows()
        rows[1]["features"] = [99.0, 11.0]
        with self.assertRaisesRegex(ValueError, "paired PP trials disagree"):
            permute_candidate_features(rows, (0,), namespace="test")


if __name__ == "__main__":
    unittest.main()
