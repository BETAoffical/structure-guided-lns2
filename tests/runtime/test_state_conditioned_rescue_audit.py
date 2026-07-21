from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from experiments.state_conditioned_rescue_audit import (
    ACTIONS,
    FEATURE_NAMES,
    AuditState,
    balanced_map_folds,
    resolve_recorded_source,
    select_safe_oracle_action,
)


def _metrics(
    *, escape: float, failure: float, reduction: float, seconds: float
) -> dict[str, float]:
    return {
        "state_escape_rate": escape,
        "final_hard_failure_rate": failure,
        "mean_conflict_reduction": reduction,
        "mean_repair_seconds": seconds,
        "conflict_reduction_per_second": reduction / seconds,
    }


class SafeOracleTests(unittest.TestCase):
    def test_fast_but_less_reliable_action_is_rejected(self) -> None:
        values = {
            "adaptive": _metrics(escape=0.75, failure=0.25, reduction=2, seconds=1),
            "size4": _metrics(escape=1.0, failure=0.0, reduction=3, seconds=1),
            "size8": _metrics(escape=0.5, failure=0.5, reduction=20, seconds=1),
            "size16": _metrics(escape=0.75, failure=0.25, reduction=2, seconds=2),
            "learned": _metrics(escape=0.75, failure=0.25, reduction=2, seconds=1.5),
        }
        self.assertEqual(select_safe_oracle_action(values), "size4")

    def test_action_coverage_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover"):
            select_safe_oracle_action({"adaptive": {}})


class MapFoldTests(unittest.TestCase):
    def _state(self, dataset: str, layout: str, map_index: int) -> AuditState:
        state_id = f"{dataset}-{layout}-{map_index}"
        return AuditState(
            dataset_id=dataset,
            state_id=state_id,
            map_id=f"map-{map_index}",
            map_group=f"{dataset}::map-{map_index}",
            cell=f"{layout}__agents_400",
            features=tuple(0.0 for _ in FEATURE_NAMES),
            oracle_action="adaptive",
            rows_by_action={action: [] for action in ACTIONS},
            action_metrics={},
        )

    def test_map_groups_never_cross_folds(self) -> None:
        states = [
            self._state(dataset, layout, index)
            for dataset in ("old", "new")
            for layout in (
                "compartmentalized",
                "dead_end_aisles",
                "regular_beltway",
            )
            for index in range(4)
        ]
        folds = balanced_map_folds(states)
        self.assertEqual(len(folds), 4)
        flattened = [group for fold in folds for group in fold]
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_dataset_and_map_identifiers_are_not_features(self) -> None:
        self.assertFalse(any("dataset" in name or "map_id" in name for name in FEATURE_NAMES))

    def test_wsl_recorded_source_resolves_to_windows_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "build" / "confirmation-v1"
            expected.mkdir(parents=True)
            observed = resolve_recorded_source(
                root, "/mnt/c/work/project/build/confirmation-v1"
            )
            self.assertEqual(observed, expected.resolve())


if __name__ == "__main__":
    unittest.main()
