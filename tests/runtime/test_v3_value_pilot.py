from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experiments import v3_value_pilot as module
from experiments._common import producer_identity
from experiments.v3_value_pilot import (
    V3_VALUE_PILOT_SCHEMA,
    V3_VALUE_PILOT_PRODUCER_FILES,
    _extend_replay_repair_budget,
    _winner_key,
    analyze_value_rollouts,
    build_state_arms,
    load_resumable_value_rollout,
    run_value_label_pilot,
    validate_value_rollout,
)


def _sequence_trial(
    sequence_id: str,
    template: str,
    candidate_id: str,
    agents: list[int],
) -> dict:
    family, size, representative = template.split(":")
    return {
        "sequence_id": sequence_id,
        "templates": [
            {
                "family": family,
                "requested_size": int(size.removeprefix("size")),
                "representative": int(representative.removeprefix("rep")),
                "template_key": template,
            }
        ],
        "steps": [
            {
                "step": 1,
                "executed": True,
                "candidate_id": candidate_id,
                "agents": agents,
                "selection_seconds": 0.1,
            }
        ],
    }


def _rollout(
    state: str,
    arm: str,
    trial: int,
    *,
    feasible: bool,
    seconds: float,
    final_conflicts: int,
) -> dict:
    return {
        "state_id": state,
        "arm_id": arm,
        "trial_index": trial,
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 400,
        "initial_conflicts": 10,
        "final_conflicts": final_conflicts,
        "feasible": feasible,
        "censored": not feasible,
        "observed_total_seconds": seconds,
        "continuation_iterations": 1,
        "normalized_conflict_auc_seconds": seconds,
    }


def _semantic_rollout() -> tuple[dict, dict, dict]:
    state = {
        "state_id": "state",
        "split": "policy_train",
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 100,
        "source_stratum": "ordinary",
        "initial_conflicts": 10,
        "before_fingerprint": "before",
        "before_repair_fingerprint": "repair-before",
        "shared_initial_selection_seconds": 0.05,
    }
    arm = {
        "arm_id": "v2_full",
        "aliases": ["v2_full"],
        "candidate_id": "candidate",
        "agents": [1, 2, 3, 4],
        "template_key": "",
    }
    seed = module._paired_seed("repair-before", 0, 1)
    row = {
        "schema": V3_VALUE_PILOT_SCHEMA,
        "producer_identity_fingerprint": "producer",
        "state_id": "state",
        "split": "policy_train",
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 100,
        "source_stratum": "ordinary",
        "arm_id": "v2_full",
        "arm_aliases": ["v2_full"],
        "candidate_id": "candidate",
        "agents": [1, 2, 3, 4],
        "actual_size": 4,
        "template_key": "",
        "trial_index": 0,
        "initial_fingerprint": "before",
        "initial_repair_fingerprint": "repair-before",
        "final_fingerprint": "after",
        "final_repair_fingerprint": "repair-after",
        "initial_conflicts": 10,
        "final_conflicts": 0,
        "conflict_trajectory": [10, 0],
        "conflict_reduction": 10,
        "steps": [
            {
                "step": 1,
                "route": "explicit_first_action",
                "action": {
                    "mode": "explicit_neighborhood",
                    "agents": [1, 2, 3, 4],
                    "pp_random_seed": seed,
                },
                "repair_outcome": "feasible",
                "conflicts_before": 10,
                "conflicts_after": 0,
                "conflict_reduction": 10,
                "repair_seconds": 0.2,
                "pp_replan_seconds": 0.1,
                "low_level": {
                    "generated": 5,
                    "expanded": 4,
                    "reopened": 0,
                    "runs": 1,
                },
                "after_done": True,
                "after_feasible": True,
                "replan_success": True,
                "before_fingerprint": "before",
                "after_fingerprint": "after",
                "before_repair_fingerprint": "repair-before",
                "after_repair_fingerprint": "repair-after",
            }
        ],
        "repair_iterations": 1,
        "continuation_iterations": 0,
        "feasible": True,
        "censored": False,
        "stop_reason": "feasible",
        "shared_initial_selection_seconds": 0.05,
        "rollout_wall_seconds": 0.25,
        "observed_total_seconds": 0.30,
        "pp_replan_seconds": 0.1,
        "conflict_auc_seconds": 2.0,
        "normalized_conflict_auc_seconds": 0.2,
        "low_level": {
            "generated": 5,
            "expanded": 4,
            "reopened": 0,
            "runs": 1,
        },
        "complete": True,
    }
    return state, arm, row


class V3ValueArmTests(unittest.TestCase):
    def test_rollout_budget_extends_beyond_s3_source_horizon(self) -> None:
        replay = {"environment": {"max_repair_iterations": 12}}
        prepared = _extend_replay_repair_budget(
            replay,
            prefix_length=9,
            max_repairs=8,
        )
        self.assertEqual(
            prepared["environment"]["max_repair_iterations"],
            17,
        )
        self.assertEqual(replay["environment"]["max_repair_iterations"], 12)

    def test_duplicate_agent_sets_are_deduplicated_with_aliases(self) -> None:
        payload = {
            "trials": [
                _sequence_trial(
                    "model",
                    "collision:size4:rep0",
                    "same",
                    [1, 2, 3, 4],
                ),
                _sequence_trial(
                    "oracle",
                    "collision:size4:rep0",
                    "same",
                    [1, 2, 3, 4],
                ),
                _sequence_trial(
                    "quality",
                    "target:size8:rep0",
                    "quality",
                    list(range(8)),
                ),
            ],
            "external_baselines": [
                {
                    "controller": "v2-full",
                    "steps": [
                        {
                            "step": 1,
                            "candidate_id": "same",
                            "action": {"agents": [1, 2, 3, 4]},
                        }
                    ],
                },
                {
                    "controller": "v2-full",
                    "steps": [
                        {
                            "step": 1,
                            "candidate_id": "same",
                            "action": {"agents": [1, 2, 3, 4]},
                        }
                    ],
                },
            ],
        }
        oracle = {
            "model_sequence_id": "model",
            "oracle_s3_efficiency_sequence_id": "oracle",
            "oracle_s3_quality_time_sequence_id": "quality",
            "oracle_h1_efficiency_first_template": "target:size8:rep0",
        }
        arms = build_state_arms(payload, oracle)
        self.assertEqual(len(arms), 2)
        self.assertEqual(
            arms[0]["aliases"],
            ["v2_full", "model_s3", "oracle_s3_efficiency"],
        )
        self.assertEqual(
            arms[1]["aliases"],
            ["oracle_s3_quality_time", "oracle_h1_efficiency"],
        )

    def test_winner_prefers_feasible_before_censored(self) -> None:
        feasible = _rollout(
            "state",
            "feasible",
            0,
            feasible=True,
            seconds=10.0,
            final_conflicts=0,
        )
        censored = _rollout(
            "state",
            "censored",
            0,
            feasible=False,
            seconds=1.0,
            final_conflicts=1,
        )
        self.assertLess(_winner_key(feasible, 0.1), _winner_key(censored, 0.1))


class V3ValueAnalysisTests(unittest.TestCase):
    def test_smoke_analysis_records_signal_without_promotion(self) -> None:
        rows = [
            _rollout(
                "state",
                arm,
                trial,
                feasible=(arm == "good"),
                seconds=1.0 if arm == "good" else 2.0,
                final_conflicts=0 if arm == "good" else 5,
            )
            for arm in ("good", "bad")
            for trial in (0, 1)
        ]
        report, states, sensitivity = analyze_value_rollouts(
            rows,
            expected_jobs=4,
            smoke_only=True,
        )
        self.assertEqual(report["decision"], "smoke_completed_not_scientific")
        self.assertEqual(report["action_sensitive_state_fraction"], 1.0)
        self.assertEqual(report["uncensored_branch_fraction"], 0.5)
        self.assertEqual(states[0]["arm_count"], 2)
        self.assertTrue(sensitivity)

    def test_analysis_rejects_duplicate_keys(self) -> None:
        row = _rollout(
            "state",
            "arm",
            0,
            feasible=True,
            seconds=1.0,
            final_conflicts=0,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_value_rollouts(
                [row, dict(row)],
                expected_jobs=2,
                smoke_only=True,
            )


class V3ValueIntegrityTests(unittest.TestCase):
    def test_producer_identity_covers_local_import_dependencies(self) -> None:
        self.assertTrue(
            {
                "experiments/closed_loop_trace_storage.py",
                "experiments/feature_schema_v2.py",
                "experiments/v3_s3.py",
            }.issubset(V3_VALUE_PILOT_PRODUCER_FILES)
        )

    def test_old_schema_resume_fails_before_writing_artifacts(self) -> None:
        requested_plan = {
            "schema": V3_VALUE_PILOT_SCHEMA,
            "states": [],
        }
        identity = {
            "schema": "lns2.producer_identity.v1",
            "test": "identity",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan_path = output / "plan.json"
            config_path = output / "run_config.json"
            plan_path.write_text(json.dumps(requested_plan), encoding="utf-8")
            config_path.write_text(
                json.dumps({"schema": "lns2.v3_value_label_pilot.v1"}),
                encoding="utf-8",
            )
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with mock.patch.object(
                module,
                "producer_identity",
                return_value=identity,
            ), mock.patch.object(
                module,
                "build_value_pilot_plan",
                return_value=requested_plan,
            ):
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    run_value_label_pilot(
                        source="unused",
                        oracle_state_comparison="unused",
                        output=output,
                        state_count=1,
                        resume=True,
                    )
            after = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(after, before)

    def test_schema_and_full_rollout_validation(self) -> None:
        self.assertEqual(V3_VALUE_PILOT_SCHEMA, "lns2.v3_value_label_pilot.v2")
        state, arm, row = _semantic_rollout()
        validate_value_rollout(
            row,
            state_plan=state,
            arm_plan=arm,
            max_repairs=30,
            wall_clock_seconds=60.0,
            expected_trial_index=0,
            expected_producer_fingerprint="producer",
        )
        for field, value, message in (
            ("observed_total_seconds", float("nan"), "finite"),
            ("producer_identity_fingerprint", "wrong", "producer"),
            ("final_fingerprint", "corrupt", "final fingerprint"),
        ):
            corrupt = json.loads(json.dumps(row))
            corrupt[field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_value_rollout(
                    corrupt,
                    state_plan=state,
                    arm_plan=arm,
                    max_repairs=30,
                    wall_clock_seconds=60.0,
                    expected_trial_index=0,
                    expected_producer_fingerprint="producer",
                )

    def test_resume_rejects_and_preserves_invalid_complete_artifact(self) -> None:
        state, arm, row = _semantic_rollout()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rollout.json"
            row["observed_total_seconds"] = 99.0
            path.write_text(json.dumps(row), encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "observed_total_seconds"):
                load_resumable_value_rollout(
                    path,
                    state_plan=state,
                    arm_plan=arm,
                    max_repairs=30,
                    wall_clock_seconds=60.0,
                    expected_trial_index=0,
                    expected_producer_fingerprint="producer",
                )
            self.assertEqual(path.read_bytes(), before)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not an object"):
                load_resumable_value_rollout(
                    path,
                    state_plan=state,
                    arm_plan=arm,
                    max_repairs=30,
                    wall_clock_seconds=60.0,
                    expected_trial_index=0,
                    expected_producer_fingerprint="producer",
                )

    def test_null_after_fingerprints_fail_closed(self) -> None:
        state, arm, row = _semantic_rollout()
        row["steps"][0]["after_fingerprint"] = None
        row["steps"][0]["after_repair_fingerprint"] = None
        row["final_fingerprint"] = None
        row["final_repair_fingerprint"] = None
        with self.assertRaisesRegex(ValueError, "after fingerprint"):
            validate_value_rollout(
                row,
                state_plan=state,
                arm_plan=arm,
                max_repairs=30,
                wall_clock_seconds=60.0,
                expected_trial_index=0,
                expected_producer_fingerprint="producer",
            )

    def test_no_progress_repair_outcome_is_recomputed(self) -> None:
        state, arm, row = _semantic_rollout()
        step = row["steps"][0]
        step["repair_outcome"] = "state_changed_no_reduction"
        step["conflicts_after"] = 10
        step["conflict_reduction"] = 0
        step["after_feasible"] = False
        row["final_conflicts"] = 10
        row["conflict_trajectory"] = [10, 10]
        row["conflict_reduction"] = 0
        row["feasible"] = False
        row["censored"] = True
        row["stop_reason"] = "environment_terminal"
        validate_value_rollout(
            row,
            state_plan=state,
            arm_plan=arm,
            max_repairs=30,
            wall_clock_seconds=60.0,
            expected_trial_index=0,
            expected_producer_fingerprint="producer",
        )
        step["repair_outcome"] = "accepted_noop"
        with self.assertRaisesRegex(ValueError, "repair outcome mismatch"):
            validate_value_rollout(
                row,
                state_plan=state,
                arm_plan=arm,
                max_repairs=30,
                wall_clock_seconds=60.0,
                expected_trial_index=0,
                expected_producer_fingerprint="producer",
            )

    def test_producer_identity_binds_source_native_and_package_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            native = root / "lns2_env.pyd"
            source.write_text("one", encoding="utf-8")
            native.write_bytes(b"native-one")
            fake_module = types.SimpleNamespace(
                __file__=str(native),
                repair_timing_schema="lns2.repair_timing.v2",
            )
            with mock.patch(
                "experiments._common.importlib.metadata.version",
                side_effect=lambda name: {"numpy": "1", "scikit-learn": "2"}[
                    name
                ],
            ), mock.patch(
                "experiments._common.importlib.import_module",
                return_value=fake_module,
            ):
                first = producer_identity(
                    project_root=root,
                    source_files=("source.py",),
                    native_required=True,
                    package_names=("numpy", "scikit-learn"),
                )
                source.write_text("two", encoding="utf-8")
                second = producer_identity(
                    project_root=root,
                    source_files=("source.py",),
                    native_required=True,
                    package_names=("numpy", "scikit-learn"),
                )
                native.write_bytes(b"native-two")
                third = producer_identity(
                    project_root=root,
                    source_files=("source.py",),
                    native_required=True,
                    package_names=("numpy", "scikit-learn"),
                )
            with mock.patch(
                "experiments._common.importlib.metadata.version",
                side_effect=lambda name: {"numpy": "1", "scikit-learn": "3"}[
                    name
                ],
            ), mock.patch(
                "experiments._common.importlib.import_module",
                return_value=fake_module,
            ):
                fourth = producer_identity(
                    project_root=root,
                    source_files=("source.py",),
                    native_required=True,
                    package_names=("numpy", "scikit-learn"),
                )
            self.assertNotEqual(first["source_sha256"], second["source_sha256"])
            self.assertNotEqual(second["native"]["sha256"], third["native"]["sha256"])
            self.assertEqual(first["packages"]["scikit-learn"], "2")
            self.assertNotEqual(third["packages"], fourth["packages"])

    def test_native_required_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.py").write_text("source", encoding="utf-8")
            with mock.patch(
                "experiments._common.importlib.import_module",
                side_effect=ImportError("missing"),
            ):
                with self.assertRaisesRegex(RuntimeError, "lns2_env"):
                    producer_identity(
                        project_root=root,
                        source_files=("source.py",),
                        native_required=True,
                    )


if __name__ == "__main__":
    unittest.main()
