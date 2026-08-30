from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments._common import sha256_file
from experiments.stride_v2first_consensus16_warehouse_n700_supply import (
    DEFAULT_OUTPUT,
    EXPERIMENT_ID,
    SEED_PRIORITY,
    SOLVER_SEEDS,
    STATUS_SCHEMA,
    TASK_IDS,
    V2_DEFAULT_OUTPUT,
    V2_EXPERIMENT_ID,
    V2_STATUS_SCHEMA,
    _run_screen_jobs,
    analyze_screen,
    load_config,
    plan,
    run_screen,
    screen_schedule,
    select_screen_keys,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / "configs" / "stride_v2first_consensus16_warehouse_n700_supply_v1.json"
)
CONFIG_SHA256 = "7761da5aef07a9a13cb8aeeba22d5cd96b70abd5a1a7ade673a636a8d471073b"
OUTPUT = ROOT / DEFAULT_OUTPUT
V2_CONFIG = (
    ROOT / "configs" / "stride_v2first_consensus16_warehouse_n700_supply_v2.json"
)
V2_CONFIG_SHA256 = "45bb42756ad6f82b04dd267da252485b499d073a04b9ec85069469374fd72c0a"
V2_OUTPUT = ROOT / V2_DEFAULT_OUTPUT


class WarehouseN700SupplyRunnerTest(unittest.TestCase):
    def test_registered_dataset_and_screen_identity_are_live(self) -> None:
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)
        path, root, config = load_config(CONFIG)
        self.assertEqual(path, CONFIG.resolve())
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(tuple(config["cohort"]["candidate_solver_seeds"]), SOLVER_SEEDS)
        self.assertEqual(tuple(row["id"] for row in config["cohort"]["tasks"]), TASK_IDS)
        for task in config["cohort"]["tasks"]:
            for field in ("task_file", "scenario_file"):
                registered = task[field]
                self.assertEqual(
                    sha256_file(root / registered["path"]), registered["sha256"]
                )
        for registered in config["inputs"].values():
            self.assertEqual(
                sha256_file(root / registered["path"]), registered["sha256"]
            )

    def test_plan_is_twelve_key_non_ttf_screen_only(self) -> None:
        _path, _root, config = load_config(CONFIG)
        schedule = screen_schedule(config)
        self.assertEqual(len(schedule), 12)
        self.assertEqual(
            {(row["task_id"], row["solver_seed"]) for row in schedule},
            {(task, seed) for task in TASK_IDS for seed in SOLVER_SEEDS},
        )
        result = plan(CONFIG)
        self.assertEqual(result["schema"], STATUS_SCHEMA)
        self.assertEqual(result["candidate_key_count"], 12)
        self.assertEqual(
            (
                result["wall_time_safety_fuse_seconds"],
                result["environment_time_safety_fuse_seconds"],
                result["episode_process_timeout_seconds"],
                result["workers"],
            ),
            (200.0, 200.0, 300.0, 16),
        )
        self.assertFalse(result["included_in_ttf"])
        self.assertFalse(result["formal_or_ttf_available"])
        self.assertTrue(result["future_formal_requires_new_registration"])

    def test_v2_is_new_identity_pinned_to_zero_controller_v1_invalid(self) -> None:
        self.assertEqual(sha256_file(V2_CONFIG), V2_CONFIG_SHA256)
        path, root, config = load_config(V2_CONFIG)
        self.assertEqual(path, V2_CONFIG.resolve())
        self.assertEqual(config["experiment_id"], V2_EXPERIMENT_ID)
        evidence = config["v1_invalid_evidence"]
        self.assertEqual(
            sha256_file(root / evidence["status"]["path"]),
            evidence["status"]["sha256"],
        )
        self.assertEqual(evidence["decision_status"], "INVALID")
        self.assertEqual(evidence["completed_schedule_entries"], 0)
        self.assertEqual(evidence["controller_episode_count"], 0)
        self.assertFalse(evidence["resumption_allowed"])
        self.assertFalse(
            (
                root
                / "build/stride-v2first-consensus16-warehouse-n700-supply-v1/"
                "screen/controller/consensus16_probe/realized_dynamic_manifest.jsonl"
            ).exists()
        )
        v1_plan = plan(CONFIG)
        v2_plan = plan(V2_CONFIG)
        self.assertEqual(v2_plan["schema"], V2_STATUS_SCHEMA)
        self.assertEqual(v2_plan["experiment_id"], V2_EXPERIMENT_ID)
        self.assertTrue(v2_plan["runner_resume_fix_profile"])
        self.assertEqual(v2_plan["v1_controller_episode_count"], 0)
        self.assertEqual(v2_plan["schedule_sha256"], v1_plan["schedule_sha256"])
        self.assertEqual(v2_plan["candidate_key_count"], 12)
        self.assertFalse(v2_plan["formal_or_ttf_available"])

    def test_selection_uses_registered_priority_without_backfill(self) -> None:
        candidates = [
            {"task_id": task, "solver_seed": seed, "eligible": True}
            for task in TASK_IDS
            for seed in SOLVER_SEEDS
        ]
        selected, missing = select_screen_keys(candidates)
        self.assertFalse(missing)
        self.assertEqual(
            [(row["task_id"], row["solver_seed"]) for row in selected],
            [(task, SEED_PRIORITY[task][0]) for task in TASK_IDS],
        )
        selected, missing = select_screen_keys(
            [row for row in candidates if row["task_id"] != TASK_IDS[0]]
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(missing, [TASK_IDS[0]])

    def test_collection_wrapper_uses_positional_runtime_api_and_sixteen_workers(self) -> None:
        _path, root, config = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runtime = output / "runtime.json"
            qualification = output / "qualification"
            keys = {(task, seed) for task in TASK_IDS for seed in SOLVER_SEEDS}

            def collection_side_effect(*args, **kwargs):
                if kwargs["phase"] == "qualify":
                    collection_root = Path(args[2])
                    collection_root.mkdir(parents=True, exist_ok=True)
                    collection_root.joinpath("run_config.json").write_text(
                        "{}", encoding="utf-8"
                    )

            with (
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n700_supply."
                    "_runtime_config_path",
                    return_value=runtime,
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n700_supply."
                    "controller_kwargs",
                    return_value={},
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n700_supply."
                    "run_closed_loop_collection",
                    side_effect=collection_side_effect,
                ) as collection,
            ):
                _run_screen_jobs(root, output, config, keys, qualification)
            self.assertEqual(collection.call_count, 2)
            self.assertEqual(
                [call.kwargs["phase"] for call in collection.call_args_list],
                ["qualify", "realized_dynamic"],
            )
            self.assertEqual(
                [call.kwargs["resume"] for call in collection.call_args_list],
                [False, True],
            )
            for call in collection.call_args_list:
                self.assertEqual(call.args[1], runtime)
                self.assertEqual(call.kwargs["workers"], 16)
                self.assertEqual(call.kwargs["qualification_source"], qualification)

    def test_dry_run_is_no_write_and_foreign_outputs_fail_before_producer(self) -> None:
        existed = OUTPUT.exists()
        with patch(
            "experiments.stride_v2first_consensus16_warehouse_n700_supply."
            "_producer",
            side_effect=AssertionError("producer must not be inspected by dry-run"),
        ):
            result = run_screen(CONFIG, OUTPUT, dry_run=True)
        self.assertFalse(result["solver_or_controller_invoked"])
        self.assertEqual(OUTPUT.exists(), existed)

        with tempfile.TemporaryDirectory() as directory:
            foreign = Path(directory) / "foreign-output"
            with patch(
                "experiments.stride_v2first_consensus16_warehouse_n700_supply."
                "_producer",
                side_effect=AssertionError("guard must run before producer"),
            ):
                with self.assertRaisesRegex(ValueError, "registered isolated output"):
                    run_screen(CONFIG, foreign)
                with self.assertRaisesRegex(ValueError, "registered isolated output"):
                    analyze_screen(CONFIG, foreign)
            self.assertFalse(foreign.exists())

        v2_existed = V2_OUTPUT.exists()
        with patch(
            "experiments.stride_v2first_consensus16_warehouse_n700_supply."
            "_producer",
            side_effect=AssertionError("producer must not be inspected by dry-run"),
        ):
            result = run_screen(V2_CONFIG, V2_OUTPUT, dry_run=True)
            with self.assertRaisesRegex(ValueError, "registered isolated output"):
                run_screen(V2_CONFIG, OUTPUT, dry_run=True)
        self.assertEqual(result["schema"], V2_STATUS_SCHEMA)
        self.assertFalse(result["solver_or_controller_invoked"])
        self.assertEqual(V2_OUTPUT.exists(), v2_existed)

    def test_timeout_is_inconclusive_but_non_timeout_failure_is_invalid(self) -> None:
        _path, _root, config = load_config(CONFIG)
        rows = screen_schedule(config)
        prepared = SimpleNamespace(
            completed_report=None,
            status={},
            resumed=False,
            base_status={"total_schedule_entries": len(rows)},
        )
        common = (
            patch(
                "experiments.stride_v2first_consensus16_warehouse_n700_supply."
                "_producer",
                return_value={"identity": "test"},
            ),
            patch(
                "experiments.stride_v2first_consensus16_warehouse_n700_supply."
                "prepare_resumable_output",
                return_value=prepared,
            ),
            patch(
                "experiments.stride_v2first_consensus16_warehouse_n700_supply."
                "_reset_qualification",
                return_value=OUTPUT / "screen" / "reset_qualification",
            ),
            patch(
                "experiments.stride_v2first_consensus16_warehouse_n700_supply."
                "write_json",
            ),
        )
        for error, expected in (
            (TimeoutError("episode process timeout"), "INCONCLUSIVE_STATE_SUPPLY"),
            (ValueError("trace corrupt"), "INVALID"),
        ):
            with common[0], common[1], common[2], common[3], patch(
                "experiments.stride_v2first_consensus16_warehouse_n700_supply."
                "_run_screen_jobs",
                side_effect=error,
            ):
                result = run_screen(CONFIG, OUTPUT)
            self.assertEqual(result["decision_status"], expected)
            self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
