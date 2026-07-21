from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.rescue_lite_confirmation import (
    AGENT_COUNTS,
    CONFIRMATION_SPLIT,
    LAYOUTS,
    analyze_confirmation,
    build_confirmation_source_config,
    confirmation_seed,
    confirmation_task_waves,
    run_rescue_lite_confirmation,
    select_confirmation_states,
    validate_dataset_isolation,
    validate_trial_coverage,
)


def _prepared(
    state_id: str,
    layout: str,
    agents: int,
    *,
    map_id: str,
    task_id: str,
) -> dict[str, object]:
    return {
        "valid": True,
        "state": {
            "state_id": state_id,
            "split": CONFIRMATION_SPLIT,
            "map_id": map_id,
            "layout_mode": layout,
            "task_id": task_id,
            "decision_index": (
                int(state_id.rsplit("-", 1)[-1])
                if state_id.rsplit("-", 1)[-1].isdigit()
                else 0
            ),
            "agent_count": agents,
            "cell": f"{layout}__agents_{agents}",
            "before_repair_fingerprint": "a" * 64,
        },
    }


def _trial(
    state_id: str,
    candidate_id: str,
    *,
    reduction: int,
    seconds: float,
    changed: bool,
    seed: int = 9,
) -> dict[str, object]:
    before = "a" * 64
    after = ("b" * 64) if changed else before
    return {
        "state_id": state_id,
        "candidate_id": candidate_id,
        "trial_index": 0,
        "random_seed": seed,
        "complete": True,
        "status": "ok",
        "outcome": {
            "before_fingerprint": before,
            "after_fingerprint": after,
            "state_changed": changed,
            "conflict_reduction": reduction,
            "repair_seconds": seconds,
            "pp_replan_seconds": seconds * 0.8,
            "generated": 100,
            "expanded": 200,
            "reopened": 5,
            "hard_failure": not changed,
        },
    }


class ConfirmationDesignTests(unittest.TestCase):
    def test_seed_is_paired_by_state_and_trial(self) -> None:
        self.assertEqual(confirmation_seed("state", 2), confirmation_seed("state", 2))
        self.assertNotEqual(confirmation_seed("state", 1), confirmation_seed("state", 2))

    def test_task_waves_are_layout_balanced(self) -> None:
        rows = []
        for layout in LAYOUTS:
            for map_index in range(4):
                map_id = f"{layout}-{map_index}"
                for task_index in range(4):
                    rows.append(
                        {
                            "layout_mode": layout,
                            "map_id": map_id,
                            "task_id": f"{map_id}-task-{task_index}",
                        }
                    )
        initial, expansion = confirmation_task_waves(rows)
        self.assertEqual(len(initial), 24)
        self.assertEqual(len(expansion), 24)
        self.assertFalse(set(initial) & set(expansion))

    def test_source_protocol_registers_adaptive_and_model_policies(self) -> None:
        rows = [
            {
                "layout_mode": layout,
                "map_id": f"{layout}-map",
                "task_variant": "dense_random_400",
            }
            for layout in LAYOUTS
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs" / "closed_loop_multiseed_collection.json").write_text(
                "{}", encoding="utf-8"
            )
            with mock.patch(
                "experiments.rescue_lite_confirmation._load_dataset_rows",
                return_value=rows,
            ):
                path = build_confirmation_source_config(
                    dataset=root / "dataset",
                    output=root / "output",
                    project_root=root,
                )
            protocol = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                protocol["policies"], ["official_adaptive", "realized_dynamic"]
            )

    def test_state_selection_enforces_each_cell_and_task_cap(self) -> None:
        rows = []
        for layout in LAYOUTS:
            for agents in AGENT_COUNTS:
                for index in range(4):
                    rows.append(
                        _prepared(
                            f"{layout}-{agents}-{index}",
                            layout,
                            agents,
                            map_id=f"{layout}-map-{index % 2}",
                            task_id=f"{layout}-{agents}-task-{index // 2}",
                        )
                    )
        selected, counts = select_confirmation_states(rows, quota_per_cell=3)
        self.assertEqual(len(selected), 18)
        self.assertEqual(set(counts.values()), {3})
        task_counts: dict[str, int] = {}
        for row in selected:
            task_id = str(row["state"]["task_id"])
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        self.assertLessEqual(max(task_counts.values()), 2)


class ConfirmationIsolationTests(unittest.TestCase):
    def _dataset(self, root: Path, split: str, content: str) -> Path:
        dataset = root
        (dataset / split / "maps").mkdir(parents=True)
        map_path = dataset / split / "maps" / "map.map"
        map_path.write_text(content, encoding="utf-8")
        (dataset / split / "manifest.jsonl").write_text(
            json.dumps({"map_id": f"{split}-map", "map_file": "maps/map.map"})
            + "\n",
            encoding="utf-8",
        )
        (dataset / "dataset_summary.json").write_text(
            json.dumps({"master_seed": 1, "splits": {split: {}}}),
            encoding="utf-8",
        )
        return dataset

    def test_map_content_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = self._dataset(root / "current", CONFIRMATION_SPLIT, "same")
            reference = self._dataset(root / "reference", "policy_validation", "same")
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_dataset_isolation(current, [reference])

    def test_nonempty_output_requires_resume_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "dataset.json"
            config.write_text("{}", encoding="utf-8")
            controller = root / "controller"
            repair = root / "repair"
            controller.mkdir()
            repair.mkdir()
            (controller / "controller_manifest.json").write_text("{}", encoding="utf-8")
            (repair / "repair_aware_manifest.json").write_text("{}", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pass --resume"):
                run_rescue_lite_confirmation(
                    project_root=root,
                    output=output,
                    dataset_config=config,
                    controller_bundle=controller,
                    repair_aware_bundle=repair,
                    reference_datasets=[],
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


class ConfirmationAnalysisTests(unittest.TestCase):
    def _results(self) -> list[dict[str, object]]:
        results = []
        for layout in LAYOUTS:
            for agents in AGENT_COUNTS:
                state_id = f"{layout}-{agents}"
                state = _prepared(
                    state_id,
                    layout,
                    agents,
                    map_id=f"{layout}-map",
                    task_id=f"{layout}-{agents}-task",
                )["state"]
                results.append(
                    {
                        "state": state,
                        "top_candidate_by_size": {
                            "4": "size4",
                            "8": "size8",
                            "16": "size16",
                        },
                        "learned_candidate_id": "size8",
                        "fixed_selector_seconds": 0.001,
                        "learned_selector_seconds": 0.01,
                        "trials": [
                            _trial(state_id, "size4", reduction=10, seconds=1.0, changed=True),
                            _trial(state_id, "size8", reduction=1, seconds=1.0, changed=True),
                            _trial(state_id, "size16", reduction=1, seconds=2.0, changed=True),
                            _trial(
                                state_id,
                                "official_adaptive",
                                reduction=5,
                                seconds=1.0,
                                changed=True,
                            ),
                        ],
                    }
                )
        return results

    def test_exact_confirmation_can_promote_frozen_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = analyze_confirmation(results=self._results(), output=temporary)
            self.assertEqual(report["decision"], "rescue_lite_confirmed")
            self.assertTrue(report["coverage"]["passed"])
            self.assertGreaterEqual(report["cell_gate"]["noninferior_cell_count"], 5)

    def test_paired_seed_mismatch_is_rejected(self) -> None:
        results = self._results()
        results[0]["trials"][1]["random_seed"] = 10
        with self.assertRaisesRegex(ValueError, "paired seed mismatch"):
            validate_trial_coverage(results)

    def test_hard_failure_with_changed_fingerprint_is_rejected(self) -> None:
        results = self._results()
        results[0]["trials"][0]["outcome"]["hard_failure"] = True
        with self.assertRaisesRegex(ValueError, "hard failure changed"):
            validate_trial_coverage(results)


if __name__ == "__main__":
    unittest.main()
