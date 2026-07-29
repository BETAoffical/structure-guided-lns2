from __future__ import annotations

import json
import tempfile
import math
import copy
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import configured_policies
from experiments.repair_aware import load_portable_scalar_model
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.v3_s3 import (
    S3_ACTION_TEMPLATES,
    V3_S3_FULL_FEATURE_NAMES,
    V3_S3_LEGACY_FULL_FEATURE_NAMES,
    balanced_sequence_templates,
    load_v3_s3_bundle,
    sequence_id,
)
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.trace_replay import recorded_replay_action
from experiments.v3_s3_collection import (
    V3_S3_COLLECTION_SCHEMA,
    V3_S3_COLLECTION_VERSION,
    _ambiguous_additional_sequences,
    _audit_state_sample,
    _candidate_generation_record,
    _coverage,
    _job_progress,
    _outcome_row,
    _paired_repair_action,
    _paired_seed,
    _qualification_artifact_errors,
    _rank_correlation,
    _resume_completed_artifact,
    _select_qualified_states,
    _sequence_feature,
    _sequence_trial,
    _baseline_trial,
    _state_artifact_errors,
    _strict_retest,
    _stream_manifests,
    qualification_pool,
    temporal_context,
)
from tests.data.test_repair_collection import sample_state
from experiments.v3_s3_pipeline import (
    S3_SOURCE_POLICIES,
    _pipeline_identity,
    _source_config,
    _source_task_partition,
    _verified_training_inputs,
    collect_v3_s3_sources,
    run_v3_s3_collection_stage,
    run_v3_s3_native_audit_stage,
    run_v3_s3_training_stage,
)
from experiments.v3_s3_training import (
    EXTRA_TREES_PARAMETERS,
    HGB_PARAMETERS,
    _baseline_metrics,
    _fit_target_model,
    _normalize_prediction_values,
    _portable_payload,
    _project_training_features,
    _runtime_prefix_targets,
    _runtime_reachable_steps,
    _sequence_rows,
    _strict_feature_projection,
    finalize_v3_s3_native_audit,
    train_v3_s3_controller,
)
from scripts.audit_v3_s3_source_replay import audit_source_replay


class _PythonPortableTreeEnsemble:
    def __init__(self, baseline, trees):
        self.baseline = float(baseline)
        self.trees = trees

    def predict_raw(self, vectors):
        outputs = []
        for vector in vectors:
            value = self.baseline
            for tree in self.trees:
                index = 0
                while not bool(tree[index]["is_leaf"]):
                    node = tree[index]
                    index = int(
                        node["left"]
                        if float(vector[int(node["feature_idx"])])
                        <= float(node["num_threshold"])
                        else node["right"]
                    )
                value += float(tree[index]["value"])
            outputs.append(value)
        return outputs

    def predict_positive(self, vectors):
        return [
            1.0 / (1.0 + math.exp(-value))
            for value in self.predict_raw(vectors)
        ]


def _semantic_qualification(root: Path, *, ambiguous: bool = True):
    run_fingerprint = "run-v2"
    state_id = "state"
    decision = {
        "split": "policy_train",
        "state_id": state_id,
        "map_id": "map",
        "layout_mode": "regular_beltway",
        "agent_count": 100,
        "source_stratum": "ordinary_progress",
        "decision_index": 0,
        "before_fingerprint": "full-0",
        "before_repair_fingerprint": "repair-0",
        "temporal_context": temporal_context([], 100),
    }
    candidates = []
    candidate_rows = []
    template_indices = {}
    for index, template in enumerate(S3_ACTION_TEMPLATES):
        candidate_id = f"candidate-{index}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "agents": [index],
                "selection_families": [template.family_key],
                "selection_rank_by_family": {
                    template.family_key: template.representative
                },
            }
        )
        candidate_rows.append(
            {
                "state_id": decision["before_fingerprint"],
                "candidate_id": candidate_id,
                "candidate_key": candidate_id,
                "features": {
                    "realized_dynamic": {
                        name: 0.0
                        for name in PROFILE_FEATURE_NAMES["realized_dynamic"]
                    }
                },
            }
        )
        template_indices[template.key] = index
    qualified = {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "decision": decision,
        "candidates": candidates,
        "candidate_rows": candidate_rows,
        "template_indices": template_indices,
        "timing": {"full_pool_seconds": 0.0},
    }
    qualification_path = root / "qualification.json"
    _write_json(qualification_path, qualified)
    selected = {**decision, "qualification_file": str(qualification_path)}
    sequences = list(balanced_sequence_templates(state_id))
    if ambiguous:
        sequences.extend(_ambiguous_additional_sequences(qualified, state_id))
    return run_fingerprint, qualified, selected, sequences


def _semantic_sequence_trial(decision, templates, trial_index):
    conflicts = [3, 2, 1, 0]
    steps = []
    for index, template in enumerate(templates, 1):
        terminated = index == len(templates)
        candidate_index = S3_ACTION_TEMPLATES.index(template)
        agents = [candidate_index]
        candidate = {
            "candidate_id": f"candidate-{candidate_index}",
            "agents": agents,
            "selection_families": [template.family_key],
            "selection_rank_by_family": {
                template.family_key: template.representative
            },
        }
        if index == 1:
            candidate_pool = [
                {
                    "candidate_id": f"candidate-{pool_index}",
                    "agents": [pool_index],
                    "selection_families": [pool_template.family_key],
                    "selection_rank_by_family": {
                        pool_template.family_key: pool_template.representative
                    },
                }
                for pool_index, pool_template in enumerate(S3_ACTION_TEMPLATES)
            ]
        else:
            candidate_pool = [candidate]
        steps.append(
            {
                "step": index,
                "template": template.payload(),
                "template_valid": True,
                "executed": True,
                "candidate_id": f"candidate-{candidate_index}",
                "agents": agents,
                "candidate_generation": _candidate_generation_record(
                    candidate_pool,
                    template,
                    before_fingerprint=f"full-{index - 1}",
                    decision_index=int(decision["decision_index"]) + index - 1,
                    source=(
                        "qualification_pool" if index == 1 else "restricted_dynamic"
                    ),
                    include_pool=index > 1,
                ),
                "action": {
                    "mode": "explicit_neighborhood",
                    "agents": agents,
                    "random_seed": _paired_seed(
                        "repair-0", trial_index, index
                    ),
                    "pp_random_seed": _paired_seed(
                        "repair-0", trial_index, index
                    ),
                },
                "selection_seconds": 0.0,
                "repair_seconds": 0.0,
                "pp_replan_seconds": 0.0,
                "total_seconds": 0.0,
                "conflicts_before": conflicts[index - 1],
                "conflicts_after": conflicts[index],
                "conflict_reduction": 1,
                "repair_outcome": "feasible" if terminated else "conflict_reduced",
                "replan_success": True,
                "step_applied": True,
                "repair_order": agents,
                "requested_pp_seed": _paired_seed(
                    "repair-0", trial_index, index
                ),
                "applied_pp_seed": _paired_seed(
                    "repair-0", trial_index, index
                ),
                "before_fingerprint": f"full-{index - 1}",
                "after_fingerprint": f"full-{index}",
                "before_repair_fingerprint": f"repair-{index - 1}",
                "after_repair_fingerprint": f"repair-{index}",
                "low_level_delta": {
                    "generated": 0,
                    "expanded": 0,
                    "reopened": 0,
                    "runs": 0,
                },
                "terminated": terminated,
                "truncated": False,
                "done": terminated,
            }
        )
    return {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "split": decision["split"],
        "state_id": decision["state_id"],
        "map_id": decision["map_id"],
        "layout_mode": decision["layout_mode"],
        "agent_count": decision["agent_count"],
        "source_stratum": decision["source_stratum"],
        "sequence_id": sequence_id(templates),
        "templates": [template.payload() for template in templates],
        "trial_index": trial_index,
        "initial_fingerprint": "full-0",
        "initial_repair_fingerprint": "repair-0",
        "final_fingerprint": "full-3",
        "final_repair_fingerprint": "repair-3",
        "steps": steps,
        "executed_steps": 3,
        "conflict_trajectory": conflicts,
        "conflict_reduction": 3,
        "best_conflict_reduction": 3,
        "no_progress": False,
        "feasible": True,
        "stop_reason": "feasible",
        "truncated": False,
        "total_seconds": 0.0,
        "pp_replan_seconds": 0.0,
        "generated": 0,
        "expanded": 0,
        "reopened": 0,
        "runs": 0,
        "complete": True,
    }


def _semantic_baseline(decision, controller, trial_index):
    row = _semantic_sequence_trial(
        decision, balanced_sequence_templates(decision["state_id"])[0], trial_index
    )
    row.pop("source_stratum")
    row.pop("sequence_id")
    row.pop("templates")
    row["controller"] = controller
    for step in row["steps"]:
        step.pop("template")
        step.pop("template_valid")
        step.pop("executed")
        step.pop("candidate_generation")
        if controller == "official_adaptive":
            step["candidate_id"] = "official_adaptive"
            step["action"] = {
                "mode": "official",
                "random_seed": _paired_seed(
                    row["initial_repair_fingerprint"],
                    trial_index,
                    step["step"],
                ),
                "pp_random_seed": _paired_seed(
                    row["initial_repair_fingerprint"],
                    trial_index,
                    step["step"],
                ),
            }
    return row


def _semantic_state_artifact(root: Path, *, ambiguous: bool = True):
    run_fingerprint, qualified, selected, sequences = _semantic_qualification(
        root, ambiguous=ambiguous
    )
    decision = qualified["decision"]
    features = [_sequence_feature(qualified, templates) for templates in sequences]
    trials = [
        _semantic_sequence_trial(decision, templates, trial_index)
        for templates in sequences
        for trial_index in (0, 1)
    ]
    if ambiguous:
        top = sorted(sequence_id(templates) for templates in sequences)[:6]
        lookup = {sequence_id(templates): templates for templates in sequences}
        trials.extend(
            _semantic_sequence_trial(decision, lookup[key], trial_index)
            for key in top
            for trial_index in (2, 3)
        )
    payload = {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": decision["state_id"],
        "ambiguous": ambiguous,
        "features": features,
        "trials": trials,
        "external_baselines": [
            _semantic_baseline(decision, controller, trial_index)
            for controller in ("v2-full", "official_adaptive")
            for trial_index in (0, 1)
        ],
    }
    state_path = root / "state.json"
    _write_json(state_path, payload)
    return run_fingerprint, qualified, selected, state_path, payload


class V3S3PipelineTest(unittest.TestCase):
    def test_native_audit_stage_forwards_resume_to_identity_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = root / "controller"
            controller.mkdir(parents=True)
            final_report = {
                "native_audit_completed": True,
                "decision": "v3_s3_mixed_load_pilot_failed",
            }
            final_manifest: dict[str, object] = {}
            _write_json(controller / "training_report.json", final_report)
            _write_json(
                root / "training_stage_report.json",
                {**final_report, "manifest": final_manifest},
            )
            _write_json(root / "status.json", {"started_at": "start"})
            collection = root / "collection"
            collection.mkdir()
            _write_json(collection / "collection_report.json", {"complete": True})
            bundle = types.SimpleNamespace(
                report=final_report,
                manifest=final_manifest,
            )
            audit_identity = {"schema": "native-audit-test"}
            with mock.patch(
                "experiments.v3_s3_pipeline._completed_collection_state_count",
                return_value=130,
            ), mock.patch(
                "experiments.v3_s3_pipeline.load_v3_s3_bundle",
                return_value=bundle,
            ), mock.patch(
                "experiments.v3_s3_pipeline.v3_s3_native_audit_identity",
                return_value=audit_identity,
            ), mock.patch(
                "experiments.v3_s3_pipeline.prepare_run_output"
            ) as prepare, mock.patch(
                "experiments.v3_s3_pipeline.finalize_v3_s3_native_audit",
                return_value={**final_report, "manifest": final_manifest},
            ) as finalize, mock.patch(
                "experiments.v3_s3_pipeline._completed_native_stage_is_valid",
                return_value=True,
            ):
                result = run_v3_s3_native_audit_stage(
                    output=root,
                    resume=True,
                    benchmark_rows=12,
                )
            self.assertEqual(result["decision"], final_report["decision"])
            self.assertTrue(prepare.call_args.kwargs["resume"])
            self.assertEqual(
                prepare.call_args.kwargs["identity"]["native_audit_identity"],
                audit_identity,
            )
            self.assertEqual(
                finalize.call_args.kwargs["audit_identity"], audit_identity
            )

    def test_probability_export_parity_uses_runtime_clamping(self) -> None:
        self.assertEqual(
            _normalize_prediction_values(
                "step1_no_progress_probability",
                (-0.2, 0.4, 1.3),
            ),
            [0.0, 0.4, 1.0],
        )
        self.assertEqual(
            _normalize_prediction_values(
                "sequence_net_conflict_reduction",
                (-2.0, 3.0),
            ),
            [-2.0, 3.0],
        )

    @staticmethod
    def _write_complete_collection(root: Path) -> None:
        collection = root / "collection"
        collection.mkdir(parents=True)
        paths = {
            "sequence_features": collection / "sequence_features.jsonl",
            "sequence_trials": collection / "sequence_trials.jsonl",
            "external_baselines": collection / "external_baselines.jsonl",
        }
        for name, path in paths.items():
            path.write_text(f'{{"artifact": "{name}"}}\n', encoding="utf-8")
        _write_json(
            collection / "collection_report.json",
            {
                "complete": True,
                "requested_state_count": 130,
                "completed_state_count": 130,
                "selection": {
                    "selected_by_split": {
                        "policy_train": 100,
                        "policy_validation": 30,
                    }
                },
                "sequence_features_sha256": sha256_file(
                    paths["sequence_features"]
                ),
                "sequence_trials_sha256": sha256_file(
                    paths["sequence_trials"]
                ),
                "external_baselines_sha256": sha256_file(
                    paths["external_baselines"]
                ),
            },
        )
        _write_json(root / "status.json", {"started_at": "start"})

    def test_training_inputs_are_hashed_before_status_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_complete_collection(root)
            verified = _verified_training_inputs(root)
            self.assertEqual(verified["collection_state_count"], 130)
            before_status = (root / "status.json").read_bytes()
            (root / "collection" / "sequence_trials.jsonl").write_text(
                '{"tampered": true}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                run_v3_s3_training_stage(
                    project_root=Path(__file__).resolve().parents[2],
                    output=root,
                    training_jobs=1,
                    resume=False,
                )
            self.assertEqual((root / "status.json").read_bytes(), before_status)
            self.assertFalse((root / "training-control").exists())

    def test_training_controller_is_published_only_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_complete_collection(root)

            def fake_train(*, output, **_kwargs):
                Path(output).mkdir(parents=True)
                (Path(output) / "artifact").write_text("ok\n", encoding="utf-8")
                return {"provisional_model_family": "test"}

            with mock.patch(
                "experiments.v3_s3_pipeline.train_v3_s3_controller",
                side_effect=fake_train,
            ), mock.patch(
                "experiments.v3_s3_pipeline.load_v3_s3_bundle"
            ) as loader, mock.patch(
                "experiments.v3_s3_pipeline.v3_s3_training_producer_identity",
                return_value={"schema": "test-training-producer"},
            ):
                report = run_v3_s3_training_stage(
                    project_root=Path(__file__).resolve().parents[2],
                    output=root,
                    training_jobs=1,
                    resume=False,
                )
            self.assertEqual(report["provisional_model_family"], "test")
            self.assertTrue((root / "controller" / "artifact").is_file())
            self.assertFalse((root / "controller.partial").exists())
            self.assertEqual(_read_json(root / "status.json")["status"], "waiting")
            loader.assert_called_once()

    def test_training_can_reference_completed_collection_without_copying(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            self._write_complete_collection(source)

            def fake_train(*, output, **_kwargs):
                Path(output).mkdir(parents=True)
                (Path(output) / "artifact").write_text(
                    "ok\n", encoding="utf-8"
                )
                return {"provisional_model_family": "test"}

            with mock.patch(
                "experiments.v3_s3_pipeline.train_v3_s3_controller",
                side_effect=fake_train,
            ), mock.patch(
                "experiments.v3_s3_pipeline.load_v3_s3_bundle"
            ), mock.patch(
                "experiments.v3_s3_pipeline.v3_s3_training_producer_identity",
                return_value={"schema": "test-training-producer"},
            ):
                report = run_v3_s3_training_stage(
                    project_root=Path(__file__).resolve().parents[2],
                    output=output,
                    training_jobs=1,
                    resume=False,
                    collection_source=source / "collection",
                )
            self.assertEqual(report["provisional_model_family"], "test")
            self.assertFalse((output / "collection").exists())
            reference = _read_json(output / "collection_reference.json")
            self.assertEqual(
                Path(reference["collection_root"]),
                (source / "collection").resolve(),
            )
            self.assertIsNone(reference["collection_root_relative"])

    def test_runtime_labels_stop_after_unchanged_state(self) -> None:
        steps = [
            {
                "step": 1,
                "template_valid": True,
                "executed": True,
                "before_fingerprint": "same",
                "after_fingerprint": "same",
                "conflicts_after": 9,
                "conflict_reduction": 1,
                "total_seconds": 0.4,
                "repair_outcome": "conflict_reduced",
            },
            {
                "step": 2,
                "template_valid": True,
                "executed": True,
                "before_fingerprint": "later",
                "after_fingerprint": "changed",
                "conflicts_after": 0,
                "conflict_reduction": 9,
                "total_seconds": 0.1,
                "repair_outcome": "feasible",
            },
        ]
        trial = {"steps": steps, "conflict_trajectory": [10, 9, 0]}
        self.assertEqual(len(_runtime_reachable_steps(trial)), 1)
        targets = _runtime_prefix_targets(trial)
        self.assertEqual(targets["sequence_net_conflict_reduction"], 1.0)
        self.assertEqual(targets["sequence_total_seconds"], 0.4)

    @staticmethod
    def _write_source_dataset(root: Path) -> None:
        scenarios = ("balanced", "uniform", "bottleneck", "swap")
        for split, map_count in (("policy_train", 4), ("policy_validation", 2)):
            rows = []
            for layout in (
                "regular_beltway",
                "compartmentalized",
                "dead_end_aisles",
            ):
                for map_index in range(map_count):
                    map_id = f"{split}-{layout}-{map_index}"
                    for agents in (80, 100, 200, 400, 600):
                        for scenario in scenarios:
                            variant = f"{scenario}_{agents}"
                            rows.append(
                                {
                                    "split": split,
                                    "layout_mode": layout,
                                    "map_id": map_id,
                                    "agent_count": agents,
                                    "task_variant": variant,
                                    "task_id": f"{map_id}-{variant}",
                                }
                            )
            _write_jsonl(root / split / "manifest.jsonl", rows)

    def test_source_partition_is_map_isolated_and_policy_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_source_dataset(root)
            partition, report = _source_task_partition(root)
            self.assertEqual(report["episode_count"], 180)
            self.assertEqual(
                {
                    policy: sum(
                        len(partition[(split, policy)])
                        for split in ("policy_train", "policy_validation")
                    )
                    for policy in S3_SOURCE_POLICIES
                },
                {policy: 60 for policy in S3_SOURCE_POLICIES},
            )
            train_maps = {
                tuple(task.split("-", 4)[0:4])
                for policy in S3_SOURCE_POLICIES
                for task in partition[("policy_train", policy)]
            }
            validation_maps = {
                tuple(task.split("-", 4)[0:4])
                for policy in S3_SOURCE_POLICIES
                for task in partition[("policy_validation", policy)]
            }
            self.assertFalse(train_maps & validation_maps)

    def test_source_config_keeps_the_registered_policy_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            self._write_source_dataset(dataset)
            project_root = Path(__file__).resolve().parents[2]
            config_path = _source_config(
                project_root=project_root,
                dataset=dataset,
                split="policy_train",
                policy="fixed_random",
                output=root / "output",
                workers=2,
            )
            config = _read_json(config_path)
            self.assertEqual(tuple(config["policies"]), S3_SOURCE_POLICIES)
            self.assertEqual(configured_policies(config), S3_SOURCE_POLICIES)

    def test_source_collection_qualifies_once_per_split_and_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            self._write_source_dataset(dataset)

            def fake_collection(_dataset, _config, _output, *, phase, **_kwargs):
                if phase == "qualify":
                    return {"qualification": {"passed": True}}
                return {
                    phase: {
                        "error_count": 0,
                        "episode_count": len(_kwargs["task_ids"]),
                    },
                }

            with mock.patch(
                "experiments.v3_s3_pipeline.run_closed_loop_collection",
                side_effect=fake_collection,
            ) as runner:
                report = collect_v3_s3_sources(
                    project_root=Path(__file__).resolve().parents[2],
                    dataset=dataset,
                    output=output,
                    controller_bundle=root / "controller",
                    workers=2,
                    resume=False,
                    started_at="2026-07-22T00:00:00+00:00",
                )

            self.assertTrue(report["complete"])
            calls = runner.call_args_list
            direct_qualifications = [
                call
                for call in calls
                if call.kwargs["phase"] == "qualify"
                and "qualification_source" not in call.kwargs
            ]
            reused_qualifications = [
                call
                for call in calls
                if call.kwargs["phase"] == "qualify"
                and "qualification_source" in call.kwargs
            ]
            policy_calls = [
                call for call in calls if call.kwargs["phase"] in S3_SOURCE_POLICIES
            ]
            self.assertEqual(len(direct_qualifications), 2)
            self.assertEqual(len(reused_qualifications), 6)
            self.assertEqual(len(policy_calls), 6)
            self.assertEqual(
                sorted(len(call.kwargs["task_ids"]) for call in direct_qualifications),
                [60, 120],
            )

    def test_collection_stage_records_dataset_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "dataset.json"
            config.write_text("{}\n", encoding="utf-8")
            output = root / "output"
            with mock.patch(
                "experiments.v3_s3_pipeline._pipeline_identity",
                return_value={"test": "identity"},
            ), mock.patch(
                "experiments.v3_s3_pipeline.generate_dataset",
                side_effect=RuntimeError("synthetic dataset failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic dataset failure"):
                    run_v3_s3_collection_stage(
                        project_root=Path(__file__).resolve().parents[2],
                        output=output,
                        controller_bundle=root / "controller",
                        dataset_config=config,
                        workers=1,
                        resume=False,
                        parallelism_audit=False,
                    )
            status = _read_json(output / "status.json")
            self.assertEqual(status["status"], "error")
            self.assertEqual(status["phase"], "dataset-failed")
            self.assertEqual(status["error_states"], 1)

    def test_reused_sources_are_bound_to_report_and_replay_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "dataset.json"
            config.write_text("{}\n", encoding="utf-8")
            controller = root / "controller"
            controller.mkdir()
            (controller / "controller_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            source = root / "source"
            source.mkdir()
            (source / "source_report.json").write_text(
                '{"complete": true}\n', encoding="utf-8"
            )
            (source / "source_replay_audit.json").write_text(
                '{"passed": true}\n', encoding="utf-8"
            )

            with mock.patch(
                "experiments.v3_s3_pipeline.producer_identity",
                return_value={"schema": "test-producer"},
            ) as producer:
                identity = _pipeline_identity(
                    project_root=Path(__file__).resolve().parents[2],
                    dataset_config=config,
                    controller_bundle=controller,
                    workers="auto",
                    parallelism_audit=True,
                    reuse_source_output=source,
                )

            self.assertEqual(identity["reuse_source_output"], str(source.resolve()))
            self.assertIn("reuse_source_report_sha256", identity)
            self.assertIn("reuse_source_replay_audit_sha256", identity)
            self.assertEqual(
                identity["producer_identity"], {"schema": "test-producer"}
            )
            self.assertNotIn("implementation", identity)
            producer.assert_called_once()
            producer_arguments = producer.call_args.kwargs
            self.assertTrue(producer_arguments["native_required"])
            self.assertNotIn(
                "experiments/v3_s3_training.py",
                producer_arguments["source_files"],
            )

    def test_collection_stage_records_incomplete_source_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "dataset.json"
            config.write_text("{}\n", encoding="utf-8")
            output = root / "output"

            def fake_dataset(_config, destination):
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "dataset_summary.json").write_text(
                    "{}\n", encoding="utf-8"
                )

            with mock.patch(
                "experiments.v3_s3_pipeline._pipeline_identity",
                return_value={"test": "identity"},
            ), mock.patch(
                "experiments.v3_s3_pipeline.generate_dataset",
                side_effect=fake_dataset,
            ), mock.patch(
                "experiments.v3_s3_pipeline.collect_v3_s3_sources",
                return_value={"complete": False},
            ):
                with self.assertRaisesRegex(RuntimeError, "completed with errors"):
                    run_v3_s3_collection_stage(
                        project_root=Path(__file__).resolve().parents[2],
                        output=output,
                        controller_bundle=root / "controller",
                        dataset_config=config,
                        workers=1,
                        resume=False,
                        parallelism_audit=False,
                    )
            status = _read_json(output / "status.json")
            self.assertEqual(status["status"], "error")
            self.assertEqual(status["phase"], "source-collection-failed")

    def test_rank_correlation_detects_order_changes(self) -> None:
        self.assertAlmostEqual(_rank_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(_rank_correlation([1, 2, 3], [3, 2, 1]), -1.0)

    def test_trace_replay_uses_recorded_neighborhood_not_policy_action(self) -> None:
        action = recorded_replay_action(
            {
                "action": {"mode": "official", "pp_random_seed": 991},
                "metrics": {
                    "neighborhood": [7, 2, 5],
                    "repair_order": [5, 7, 2],
                    "requested_pp_random_seed": 991,
                    "applied_pp_random_seed": 991,
                },
            }
        )
        self.assertEqual(action["mode"], "replay_neighborhood")
        self.assertEqual(action["agents"], [7, 2, 5])
        self.assertEqual(action["repair_order"], [5, 7, 2])
        self.assertEqual(action["pp_random_seed"], 991)

    def test_trace_replay_rejects_unseeded_pp_transition(self) -> None:
        with self.assertRaisesRegex(ValueError, "deterministic pp_random_seed"):
            recorded_replay_action(
                {
                    "action": {"mode": "official"},
                    "metrics": {
                        "neighborhood": [2, 5],
                        "repair_order": [5, 2],
                    }
                }
            )

    def test_trace_replay_accepts_legacy_seeded_explicit_transition(self) -> None:
        action = recorded_replay_action(
            {
                "action": {
                    "mode": "explicit_neighborhood",
                    "agents": [2, 5],
                    "random_seed": 917,
                },
                "metrics": {
                    "neighborhood": [2, 5],
                    "repair_order": [5, 2],
                    "requested_random_seed": 917,
                },
            }
        )
        self.assertEqual(
            action,
            {
                "mode": "explicit_neighborhood",
                "agents": [2, 5],
                "random_seed": 917,
            },
        )

    def test_trace_replay_rejects_legacy_explicit_seed_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "deterministic source evidence"):
            recorded_replay_action(
                {
                    "action": {
                        "mode": "explicit_neighborhood",
                        "agents": [2, 5],
                        "random_seed": 917,
                    },
                    "metrics": {
                        "neighborhood": [2, 5],
                        "repair_order": [5, 2],
                        "requested_random_seed": 918,
                    },
                }
            )

    def test_trace_replay_rejects_empty_recorded_neighborhood(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty recorded neighborhood"):
            recorded_replay_action(
                {
                    "action": {"mode": "official"},
                    "metrics": {"neighborhood": [], "repair_order": []},
                }
            )

    def test_source_replay_audit_fails_when_later_prefix_is_rejected(self) -> None:
        rows = [
            {
                "source_root": "root",
                "episode_id": "episode",
                "decision_index": 0,
                "prefix_actions": [],
                "replay_action": {"step": 0},
                "before_fingerprint": "state-0",
                "after_fingerprint": "state-1",
                "source_policy": "fixed_random",
            },
            {
                "source_root": "root",
                "episode_id": "episode",
                "decision_index": 1,
                "prefix_actions": [{"step": 0}],
                "replay_action": {"step": 1},
                "before_fingerprint": "state-1",
                "after_fingerprint": "state-2",
                "source_policy": "fixed_random",
            },
            {
                "source_root": "root",
                "episode_id": "episode",
                "decision_index": 2,
                "prefix_actions": [{"step": 0}, {"step": 1}],
                "replay_action": {"step": 2},
                "before_fingerprint": "state-2",
                "after_fingerprint": "state-3",
                "source_policy": "fixed_random",
            },
        ]

        class ReplayEnvironment:
            def step(self, _action):
                return {"observation": {"done": True, "fingerprint": "state-1"}}

        with mock.patch(
            "scripts.audit_v3_s3_source_replay.source_roots", return_value={}
        ), mock.patch(
            "scripts.audit_v3_s3_source_replay.source_decisions", return_value=rows
        ), mock.patch(
            "scripts.audit_v3_s3_source_replay._source_replay_job",
            return_value=({}, {}),
        ), mock.patch(
            "scripts.audit_v3_s3_source_replay.replay_prefix",
            return_value=(
                ReplayEnvironment(),
                {"done": False, "fingerprint": "state-0"},
            ),
        ), mock.patch(
            "scripts.audit_v3_s3_source_replay.state_fingerprint",
            side_effect=lambda state: state["fingerprint"],
        ):
            report = audit_source_replay(Path("unused"))
        self.assertEqual(report["matched_decision_state_count"], 2)
        self.assertEqual(report["rejected_episode_count"], 1)
        self.assertFalse(report["passed"])

    def test_source_replay_audit_fails_on_terminal_after_mismatch(self) -> None:
        rows = [
            {
                "source_root": "root",
                "episode_id": "episode",
                "decision_index": 0,
                "prefix_actions": [],
                "replay_action": {"step": 0},
                "before_fingerprint": "state-0",
                "after_fingerprint": "state-1",
                "source_policy": "fixed_random",
            }
        ]

        class ReplayEnvironment:
            def step(self, _action):
                return {"observation": {"done": True, "fingerprint": "wrong"}}

        with mock.patch(
            "scripts.audit_v3_s3_source_replay.source_roots", return_value={}
        ), mock.patch(
            "scripts.audit_v3_s3_source_replay.source_decisions", return_value=rows
        ), mock.patch(
            "scripts.audit_v3_s3_source_replay._source_replay_job",
            return_value=({}, {}),
        ), mock.patch(
            "scripts.audit_v3_s3_source_replay.replay_prefix",
            return_value=(
                ReplayEnvironment(),
                {"done": False, "fingerprint": "state-0"},
            ),
        ), mock.patch(
            "scripts.audit_v3_s3_source_replay.state_fingerprint",
            side_effect=lambda state: state["fingerprint"],
        ):
            report = audit_source_replay(Path("unused"))
        self.assertEqual(report["terminal_after_mismatch_count"], 1)
        self.assertFalse(report["passed"])

    def test_official_source_history_records_actual_neighborhood_size(self) -> None:
        outcome = _outcome_row(
            {
                "before_conflicts": 7,
                "before_repair_fingerprint": "before",
                "after_repair_fingerprint": "after",
                "actual_action": {"mode": "official"},
                "actual_metrics": {
                    "neighborhood": [1, 2, 3, 4, 5, 6, 7, 8],
                    "replan_success": True,
                },
                "actual_lns2": {
                    "outcome": {"conflicts_after": 5, "repair_seconds": 0.25}
                },
            }
        )
        self.assertEqual(outcome["neighborhood_size"], 8)
        self.assertEqual(outcome["conflict_reduction"], 2)

    def test_parallel_audit_sample_round_robins_layout_and_load(self) -> None:
        rows = []
        for layout in (
            "regular_beltway",
            "compartmentalized",
            "dead_end_aisles",
        ):
            for agents in (100, 400, 600):
                for repeat in range(2):
                    rows.append(
                        {
                            "layout_mode": layout,
                            "agent_count": agents,
                            "map_id": f"{layout}-{agents}-{repeat}",
                            "source_policy": S3_SOURCE_POLICIES[repeat],
                            "decision_stage": "early" if repeat == 0 else "late",
                            "source_root": f"root-{layout}",
                            "episode_id": f"episode-{agents}-{repeat}",
                            "decision_index": repeat,
                        }
                    )
        selected = _audit_state_sample(rows, count=10)
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            {
                (row["layout_mode"], row["agent_count"])
                for row in selected[:9]
            },
            {
                (layout, agents)
                for layout in (
                    "regular_beltway",
                    "compartmentalized",
                    "dead_end_aisles",
                )
                for agents in (100, 400, 600)
            },
        )

    def test_sparse_low_load_sources_backfill_without_duplicate_states(self) -> None:
        layouts = (
            "regular_beltway",
            "compartmentalized",
            "dead_end_aisles",
        )
        policies = ("fixed_random", "official_adaptive", "realized_dynamic")
        rows = []
        split_agents = {
            "policy_train": [100] * 6 + [200] * 10 + [400] * 30 + [600] * 82,
            "policy_validation": [400] * 22 + [600] * 55,
        }
        for split, agent_counts in split_agents.items():
            for index, agent_count in enumerate(agent_counts):
                layout = layouts[index % len(layouts)]
                rows.append(
                    {
                        "split": split,
                        "layout_mode": layout,
                        "agent_count": agent_count,
                        "source_no_progress": index % 5 == 0,
                        "source_repair_seconds": 0.1 + 0.01 * (index % 11),
                        "before_conflicts": 1 + index % 13,
                        "map_id": f"{split}-{layout}-{index % 6}",
                        "task_id": f"{split}-{layout}-task-{index}",
                        "source_policy": policies[index % len(policies)],
                        "decision_stage": ("early", "middle", "late")[index % 3],
                        "source_root": f"root-{split}",
                        "episode_id": f"{split}-episode-{index}",
                        "decision_index": index % 12,
                        "before_repair_fingerprint": f"{split}-fingerprint-{index}",
                    }
                )

        pool, pool_report = qualification_pool(rows)
        self.assertTrue(pool_report["passed"])
        self.assertEqual(pool_report["qualification_pool_count"], 205)
        self.assertEqual(
            pool_report["selection_plan"]["target_state_count"], 203
        )
        self.assertEqual(
            pool_report["selection_plan"]["by_split"]["policy_train"][
                "missing_agent_counts"
            ],
            [80],
        )

        with tempfile.TemporaryDirectory() as directory:
            qualification_results = []
            for index, row in enumerate(pool):
                path = Path(directory) / f"state-{index}.json"
                _write_json(path, {"decision": row})
                qualification_results.append(
                    {"status": "ok", "state_file": str(path)}
                )
            selected, selection = _select_qualified_states(qualification_results)

        self.assertEqual(len(selected), 203)
        self.assertEqual(
            selection["selected_by_split"],
            {"policy_train": 128, "policy_validation": 75},
        )
        self.assertNotIn(80, selection["selected_by_agent_count"])
        self.assertEqual(
            len({str(row["before_repair_fingerprint"]) for row in selected}),
            len(selected),
        )

    def test_paired_repair_action_seeds_native_step_and_pp(self) -> None:
        explicit = _paired_repair_action(
            "explicit_neighborhood", random_seed=123, agents=[4, 2]
        )
        self.assertEqual(explicit["random_seed"], 123)
        self.assertEqual(explicit["pp_random_seed"], 123)
        self.assertEqual(explicit["agents"], [4, 2])
        official = _paired_repair_action("official", random_seed=456)
        self.assertEqual(official["random_seed"], 456)
        self.assertEqual(official["pp_random_seed"], 456)

    def test_coverage_allows_extra_paired_trials_for_ambiguous_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_fingerprint, _qualified, selected, path, _payload = (
                _semantic_state_artifact(root)
            )
            report = _coverage(
                [selected],
                [path],
                run_fingerprint=run_fingerprint,
            )
        self.assertEqual(report["feature_count"], 48)
        self.assertEqual(report["trial_count"], 108)
        self.assertTrue(report["passed"], report["errors"])

    def test_completed_artifact_validators_reject_semantic_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, qualified, selected, _path, payload = _semantic_state_artifact(
                root, ambiguous=False
            )
            self.assertEqual(
                _qualification_artifact_errors(
                    qualified,
                    run_fingerprint=run,
                    expected_decision=qualified["decision"],
                ),
                [],
            )
            self.assertEqual(
                _state_artifact_errors(
                    payload,
                    run_fingerprint=run,
                    selected_row=selected,
                    qualified=qualified,
                ),
                [],
            )

            template_tamper = copy.deepcopy(qualified)
            first_key = next(iter(template_tamper["template_indices"]))
            template_tamper["template_indices"][first_key] = 17
            self.assertTrue(
                _qualification_artifact_errors(
                    template_tamper, run_fingerprint=run
                )
            )
            self.assertTrue(
                _qualification_artifact_errors([], run_fingerprint=run)
            )

            feature_tamper = copy.deepcopy(payload)
            feature_tamper["features"][0]["feature_names"].reverse()
            self.assertTrue(
                _state_artifact_errors(
                    feature_tamper,
                    run_fingerprint=run,
                    selected_row=selected,
                    qualified=qualified,
                )
            )

    def test_state_validator_rejects_dynamic_candidate_and_strict_type_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, qualified, selected, _path, payload = _semantic_state_artifact(
                root, ambiguous=False
            )

            dynamic_candidate = copy.deepcopy(payload)
            dynamic_candidate["trials"][0]["steps"][1]["candidate_id"] = "other"
            self.assertTrue(
                _state_artifact_errors(
                    dynamic_candidate,
                    run_fingerprint=run,
                    selected_row=selected,
                    qualified=qualified,
                )
            )

            wrong_state = copy.deepcopy(payload)
            generation = wrong_state["trials"][0]["steps"][1][
                "candidate_generation"
            ]
            generation["before_fingerprint"] = "different-state"
            generation["fingerprint"] = _fingerprint(
                {key: value for key, value in generation.items() if key != "fingerprint"}
            )
            self.assertTrue(
                _state_artifact_errors(
                    wrong_state,
                    run_fingerprint=run,
                    selected_row=selected,
                    qualified=qualified,
                )
            )

            for corrupt in (
                ("null-fingerprint", None),
                ("fractional-counter", 0.5),
                ("nonboolean-terminal", None),
            ):
                tampered = copy.deepcopy(payload)
                if corrupt[0] == "null-fingerprint":
                    tampered["trials"][0]["steps"][0]["after_fingerprint"] = None
                    tampered["trials"][0]["final_fingerprint"] = None
                elif corrupt[0] == "fractional-counter":
                    tampered["trials"][0]["steps"][0]["low_level_delta"][
                        "generated"
                    ] = corrupt[1]
                else:
                    tampered["trials"][0]["steps"][0]["terminated"] = None
                self.assertTrue(
                    _state_artifact_errors(
                        tampered,
                        run_fingerprint=run,
                        selected_row=selected,
                        qualified=qualified,
                    ),
                    corrupt[0],
                )
            terminal_tamper = copy.deepcopy(payload)
            terminal_tamper["trials"][0]["stop_reason"] = "horizon_complete"
            self.assertTrue(
                _state_artifact_errors(
                    terminal_tamper,
                    run_fingerprint=run,
                    selected_row=selected,
                    qualified=qualified,
                )
            )
            sequence_seed_tamper = copy.deepcopy(payload)
            sequence_action = sequence_seed_tamper["trials"][0]["steps"][0][
                "action"
            ]
            sequence_action["random_seed"] += 1
            sequence_action["pp_random_seed"] += 1
            self.assertTrue(
                _state_artifact_errors(
                    sequence_seed_tamper,
                    run_fingerprint=run,
                    selected_row=selected,
                    qualified=qualified,
                )
            )
            baseline_seed_tamper = copy.deepcopy(payload)
            baseline_action = baseline_seed_tamper["external_baselines"][0][
                "steps"
            ][0]["action"]
            baseline_action["random_seed"] += 1
            baseline_action["pp_random_seed"] += 1
            self.assertTrue(
                _state_artifact_errors(
                    baseline_seed_tamper,
                    run_fingerprint=run,
                    selected_row=selected,
                    qualified=qualified,
                )
            )
            nonfinite_tamper = copy.deepcopy(payload)
            nonfinite_tamper["features"][0]["feature_values"][0] = float("nan")
            self.assertTrue(
                _state_artifact_errors(
                    nonfinite_tamper,
                    run_fingerprint=run,
                    selected_row=selected,
                    qualified=qualified,
                )
            )

    def test_invalid_completed_resume_fails_closed_and_preserves_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, qualified, _selected, _sequences = _semantic_qualification(root)
            path = root / "qualification.json"
            qualified["candidate_rows"][0]["state_id"] = "tampered"
            _write_json(path, qualified)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "invalid and was preserved"):
                _resume_completed_artifact(
                    path,
                    run_fingerprint=run,
                    validator=_qualification_artifact_errors,
                    label="test qualification",
                    expected_decision=qualified["decision"],
                )
            self.assertEqual(path.read_bytes(), before)

            qualified["complete"] = False
            qualified["candidate_rows"][0]["state_id"] = qualified["decision"][
                "before_fingerprint"
            ]
            _write_json(path, qualified)
            self.assertIsNone(
                _resume_completed_artifact(
                    path,
                    run_fingerprint=run,
                    validator=_qualification_artifact_errors,
                    label="test qualification",
                    expected_decision=qualified["decision"],
                )
            )

            run, qualified, selected, state_path, payload = (
                _semantic_state_artifact(root, ambiguous=False)
            )
            payload["external_baselines"][0]["state_id"] = "other-state"
            _write_json(state_path, payload)
            before = state_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "invalid and was preserved"):
                _resume_completed_artifact(
                    state_path,
                    run_fingerprint=run,
                    validator=_state_artifact_errors,
                    label="test state",
                    selected_row=selected,
                    qualified=qualified,
                )
            self.assertEqual(state_path.read_bytes(), before)

    def test_terminal_nonfeasible_done_stops_sequence_and_baseline(self) -> None:
        before = sample_state()
        before["iteration"] = 0
        after = copy.deepcopy(before)
        after["done"] = True
        after["feasible"] = False
        after["iteration"] = 1
        decision = {
            "split": "policy_train",
            "state_id": "terminal-state",
            "map_id": "map",
            "layout_mode": "regular_beltway",
            "agent_count": 4,
            "source_stratum": "ordinary_progress",
            "task_id": "task",
            "solver_seed": 7,
            "decision_index": 0,
            "prefix_actions": [],
            "before_fingerprint": state_fingerprint(before),
        }
        templates = balanced_sequence_templates(decision["state_id"])[0]
        qualified = {
            "decision": decision,
            "candidates": [
                {
                    "candidate_id": "first",
                    "agents": [0],
                    "selection_families": [templates[0].family_key],
                    "selection_rank_by_family": {
                        templates[0].family_key: templates[0].representative
                    },
                }
            ],
            "template_indices": {templates[0].key: 0},
            "timing": {"full_pool_seconds": 0.0},
        }

        class TerminalEnvironment:
            def __init__(self):
                self.calls = 0

            def step(self, action):
                self.calls += 1
                return {
                    "observation": copy.deepcopy(after),
                    "metrics": {
                        "replan_success": False,
                        "pp_replan_seconds": 0.0,
                        "step_applied": True,
                        "neighborhood": [0],
                        "repair_order": [0],
                        "requested_pp_random_seed": action["pp_random_seed"],
                        "applied_pp_random_seed": action["pp_random_seed"],
                    },
                    "terminated": False,
                    "truncated": True,
                }

        with mock.patch(
            "experiments.v3_s3_collection._source_replay_job",
            return_value=({}, {"proposal": {}}),
        ):
            sequence_environment = TerminalEnvironment()
            with mock.patch(
                "experiments.v3_s3_collection.replay_prefix",
                return_value=(sequence_environment, copy.deepcopy(before)),
            ):
                sequence = _sequence_trial(qualified, templates, 0)
            baseline_environment = TerminalEnvironment()
            with mock.patch(
                "experiments.v3_s3_collection.replay_prefix",
                return_value=(baseline_environment, copy.deepcopy(before)),
            ):
                baseline = _baseline_trial(qualified, "official_adaptive", 0)

        self.assertEqual(sequence_environment.calls, 1)
        self.assertEqual(baseline_environment.calls, 1)
        for row in (sequence, baseline):
            self.assertEqual(
                row["stop_reason"], "deadline_or_iteration_truncation"
            )
            self.assertTrue(row["truncated"])
            self.assertEqual(row["executed_steps"], 1)

    def test_strict_retest_cache_binds_state_and_qualification_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _qualified, selected, state_path, payload = (
                _semantic_state_artifact(root, ambiguous=False)
            )
            controller = root / "controller"
            controller.mkdir()
            (controller / "bundle.json").write_text("{}\n", encoding="utf-8")
            completed = [
                {
                    "state_id": selected["state_id"],
                    "sequence_id": payload["trials"][0]["sequence_id"],
                    "trial_index": 0,
                    "passed": True,
                }
            ]
            with mock.patch(
                "experiments.v3_s3_collection._run_jobs",
                return_value=(completed, []),
            ) as run_jobs:
                first = _strict_retest(
                    selected=[selected],
                    state_files=[state_path],
                    output_root=root,
                    controller_bundle=controller,
                    run_fingerprint=run,
                    fraction=1.0,
                )
                second = _strict_retest(
                    selected=[selected],
                    state_files=[state_path],
                    output_root=root,
                    controller_bundle=controller,
                    run_fingerprint=run,
                    fraction=1.0,
                )
                payload["cache_nonce"] = 1
                _write_json(state_path, payload)
                third = _strict_retest(
                    selected=[selected],
                    state_files=[state_path],
                    output_root=root,
                    controller_bundle=controller,
                    run_fingerprint=run,
                    fraction=1.0,
                )
            self.assertEqual(first["input_sha256"], second["input_sha256"])
            self.assertNotEqual(second["input_sha256"], third["input_sha256"])
            self.assertEqual(run_jobs.call_count, 2)

    def test_resume_progress_excludes_reused_states_from_throughput(self) -> None:
        completed = [{"status": "resumed"} for _ in range(47)]
        completed.extend({"status": "ok"} for _ in range(6))
        progress = _job_progress(
            completed,
            [],
            total=201,
            elapsed_seconds=6000.0,
        )
        self.assertEqual(progress["finished_states"], 53)
        self.assertEqual(progress["resumed_states"], 47)
        self.assertEqual(progress["processed_states"], 6)
        self.assertAlmostEqual(progress["states_per_minute"], 0.06)
        self.assertAlmostEqual(
            progress["estimated_remaining_seconds"],
            148000.0,
        )

    def test_training_rows_reject_duplicate_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.jsonl"
            trials = root / "trials.jsonl"
            duplicate = {
                "state_id": "state",
                "sequence_id": "sequence",
            }
            _write_jsonl(features, [duplicate, duplicate])
            _write_jsonl(trials, [])
            with self.assertRaisesRegex(ValueError, "duplicate sequences"):
                _sequence_rows(features, trials)

    def test_training_projects_legacy_wall_time_features_by_name(self) -> None:
        legacy_values = tuple(
            999.0
            if name.startswith("history.") and name.endswith("repair_seconds")
            else float(index)
            for index, name in enumerate(V3_S3_LEGACY_FULL_FEATURE_NAMES)
        )
        projected = _project_training_features(
            V3_S3_LEGACY_FULL_FEATURE_NAMES,
            legacy_values,
        )
        self.assertEqual(len(projected), len(V3_S3_FULL_FEATURE_NAMES))
        self.assertEqual(
            projected,
            tuple(
                legacy_values[V3_S3_LEGACY_FULL_FEATURE_NAMES.index(name)]
                for name in V3_S3_FULL_FEATURE_NAMES
            ),
        )

    def test_baseline_metrics_require_exact_state_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baselines.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "split": "policy_validation",
                        "controller": "v2-full",
                        "state_id": "state-a",
                        "trial_index": trial,
                        "agent_count": 100,
                    }
                    for trial in (0, 1)
                ],
            )
            with self.assertRaisesRegex(ValueError, "state coverage mismatch"):
                _baseline_metrics(
                    path,
                    "policy_validation",
                    "v2-full",
                    expected_state_ids={"state-a", "state-b"},
                )

    def test_streamed_manifests_replace_complete_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            state.write_text(
                "{\"external_baselines\":[{\"id\":3}],"
                "\"features\":[{\"id\":1}],\"trials\":[{\"id\":2}]}\n",
                encoding="utf-8",
            )
            counts = _stream_manifests([state], root)
            self.assertEqual(counts, {"features": 1, "trials": 1, "baselines": 1})
            self.assertFalse(any(root.glob("*.partial")))
            self.assertEqual(
                (root / "sequence_features.jsonl").read_text(encoding="utf-8"),
                '{"id": 1}\n',
            )

    def test_fractional_probability_target_is_not_binarized(self) -> None:
        try:
            import numpy as np
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is only required in the Windows training profile")
        values = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        target = np.asarray([0.0, 0.5, 0.5, 1.0], dtype=float)
        name, estimator = _fit_target_model(
            "step1_no_progress_probability",
            values,
            target,
            np.ones(4, dtype=float),
            "extra_trees",
        )
        self.assertEqual(name, "step1_no_progress_probability")
        self.assertFalse(hasattr(estimator, "classes_"))
        payload = _portable_payload(name, estimator, ["x"], "fractional-test")
        portable = replace(load_portable_scalar_model(payload), native_predictor=None)
        rows = [
            {
                "feature_profile": "v3_s3",
                "feature_names": ("x",),
                "feature_values": (float(value[0]),),
            }
            for value in values
        ]
        expected = estimator.predict(values)
        observed = portable.predict(rows)
        self.assertLessEqual(
            max(abs(float(a) - float(b)) for a, b in zip(expected, observed)),
            1e-12,
        )

    def test_strict_feature_projection_removes_constants_and_aliases(self) -> None:
        rows = [
            {"features": (value, 2.0 * value, 7.0, -value)}
            for value in (1.0, 2.0, 3.0, 4.0)
        ]
        kept, report = _strict_feature_projection(rows, ("a", "b", "c", "d"))
        self.assertEqual(kept, (0,))
        self.assertEqual(report["removed"]["c"]["reason"], "constant")
        self.assertEqual(report["removed"]["b"]["canonical"], "a")
        self.assertEqual(report["removed"]["d"]["canonical"], "a")

    def test_extra_trees_portable_export_matches_sklearn(self) -> None:
        try:
            import numpy as np
            from sklearn.ensemble import ExtraTreesRegressor
        except ImportError:
            self.skipTest("scikit-learn is only required in the Windows training profile")

        values = np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]], dtype=float
        )
        target = np.asarray([0.5, 2.0, 3.0, 5.0], dtype=float)
        estimator = ExtraTreesRegressor(
            n_estimators=7,
            min_samples_leaf=1,
            max_features=2,
            random_state=7,
            n_jobs=1,
        ).fit(values, target)
        payload = _portable_payload(
            "step1_conflict_reduction", estimator, ["x", "y"], "test"
        )
        portable = replace(
            load_portable_scalar_model(payload), native_predictor=None
        )
        rows = [
            {
                "feature_profile": "v3_s3",
                "feature_names": ("x", "y"),
                "feature_values": tuple(row),
            }
            for row in values
        ]
        expected = estimator.predict(values)
        observed = portable.predict(rows)
        self.assertLessEqual(
            max(abs(float(left) - float(right)) for left, right in zip(expected, observed)),
            1e-12,
        )

    def test_extra_trees_portable_matches_float32_split_boundary(self) -> None:
        try:
            import numpy as np
            from sklearn.ensemble import ExtraTreesRegressor
        except ImportError:
            self.skipTest("scikit-learn is only required in the Windows training profile")

        values = np.asarray([[0.0], [1.0]], dtype=float)
        estimator = ExtraTreesRegressor(
            n_estimators=1,
            min_samples_leaf=1,
            max_features=1,
            random_state=17,
            n_jobs=1,
        ).fit(values, np.asarray([0.0, 1.0], dtype=float))
        threshold = float(estimator.estimators_[0].tree_.threshold[0])
        lower = np.float32(threshold)
        if float(lower) > threshold:
            lower = np.nextafter(lower, np.float32("-inf"))
        upper = np.nextafter(lower, np.float32("inf"))
        rounding_boundary = (float(lower) + float(upper)) / 2.0
        probe = (threshold + rounding_boundary) / 2.0
        self.assertNotEqual(
            float(probe <= threshold),
            float(np.float32(probe) <= threshold),
        )
        payload = _portable_payload(
            "step1_conflict_reduction",
            estimator,
            ["x"],
            "float32-boundary",
        )
        self.assertEqual(payload["input_precision"], "float32")
        portable = replace(
            load_portable_scalar_model(payload),
            native_predictor=None,
        )
        row = {
            "feature_profile": "v3_s3",
            "feature_names": ("x",),
            "feature_values": (probe,),
        }
        self.assertAlmostEqual(
            float(estimator.predict(np.asarray([[probe]], dtype=float))[0]),
            float(portable.predict([row])[0]),
            places=12,
        )

    def test_training_smoke_exports_an_independent_bundle(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is only required in the Windows training profile")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_rows = []
            trial_rows = []
            baseline_rows = []
            layouts = ("regular_beltway", "compartmentalized", "dead_end_aisles")
            agents = (80, 100, 200, 400, 600)
            state_index = 0
            for split, maps_per_layout in (("policy_train", 4), ("policy_validation", 2)):
                for layout in layouts:
                    for map_index in range(maps_per_layout):
                        map_id = f"{split}-{layout}-{map_index}"
                        state_id = f"state-{state_index}"
                        agent_count = agents[state_index % len(agents)]
                        sequences = balanced_sequence_templates(state_id)[:4]
                        for sequence_index, templates in enumerate(sequences):
                            sid = sequence_id(templates)
                            values = [
                                float(
                                    (state_index + 1) * (feature_index % 5)
                                    + sequence_index * (feature_index % 3)
                                )
                                for feature_index in range(len(V3_S3_FULL_FEATURE_NAMES))
                            ]
                            feature_rows.append(
                                {
                                    "split": split,
                                    "state_id": state_id,
                                    "sequence_id": sid,
                                    "map_id": map_id,
                                    "layout_mode": layout,
                                    "agent_count": agent_count,
                                    "source_stratum": "ordinary_progress",
                                    "templates": [value.payload() for value in templates],
                                    "feature_names": list(V3_S3_FULL_FEATURE_NAMES),
                                    "feature_values": values,
                                }
                            )
                            for trial_index in (0, 1):
                                no_progress = (state_index + sequence_index + trial_index) % 5 == 0
                                reduction = 0 if no_progress else 3 + sequence_index
                                steps = [
                                    {
                                        "step": step,
                                        "template": templates[step - 1].payload(),
                                        "template_valid": True,
                                        "executed": True,
                                        "selection_seconds": 0.02,
                                        "total_seconds": 0.2 + 0.02 * step,
                                        "conflict_reduction": reduction,
                                        "repair_outcome": (
                                            "accepted_noop" if no_progress else "conflict_reduced"
                                        ),
                                    }
                                    for step in (1, 2, 3)
                                ]
                                conflicts = 100
                                trajectory = [conflicts]
                                for step_row in steps:
                                    step_row["conflicts_before"] = conflicts
                                    conflicts -= int(reduction)
                                    step_row["conflicts_after"] = conflicts
                                    trajectory.append(conflicts)
                                trial_rows.append(
                                    {
                                        "split": split,
                                        "state_id": state_id,
                                        "sequence_id": sid,
                                        "templates": [value.payload() for value in templates],
                                        "trial_index": trial_index,
                                        "steps": steps,
                                        "conflict_trajectory": trajectory,
                                        "conflict_reduction": 3 * reduction,
                                        "total_seconds": sum(row["total_seconds"] for row in steps),
                                        "no_progress": no_progress,
                                        "feasible": False,
                                    }
                                )
                        if split in {"policy_train", "policy_validation"}:
                            for controller in ("v2-full", "official_adaptive"):
                                for trial_index in (0, 1):
                                    reduction = 5.0 if controller == "v2-full" else 4.0
                                    baseline_rows.append(
                                        {
                                            "split": split,
                                            "state_id": state_id,
                                            "controller": controller,
                                            "trial_index": trial_index,
                                            "agent_count": agent_count,
                                            "conflict_reduction": reduction,
                                            "total_seconds": 1.0,
                                            "no_progress": False,
                                            "feasible": False,
                                            "steps": [
                                                {
                                                    "step": step,
                                                    "selection_seconds": 0.1,
                                                    "total_seconds": 1.0 / 3.0,
                                                    "conflicts_before": (
                                                        100
                                                        if step == 1
                                                        else 100 - int(reduction)
                                                    ),
                                                    "conflicts_after": (
                                                        100 - int(reduction)
                                                    ),
                                                    "repair_outcome": (
                                                        "conflict_reduced"
                                                        if step == 1
                                                        else "state_changed_no_reduction"
                                                    ),
                                                }
                                                for step in (1, 2, 3)
                                            ],
                                            "conflict_trajectory": [
                                                100,
                                                100 - int(reduction),
                                                100 - int(reduction),
                                                100 - int(reduction),
                                            ],
                                        }
                                    )
                        state_index += 1
            features = root / "features.jsonl"
            trials = root / "trials.jsonl"
            baselines = root / "baselines.jsonl"
            _write_jsonl(features, feature_rows)
            _write_jsonl(trials, trial_rows)
            _write_jsonl(baselines, baseline_rows)
            with mock.patch.dict(HGB_PARAMETERS, {"max_iter": 3, "max_leaf_nodes": 3}), mock.patch.dict(
                EXTRA_TREES_PARAMETERS,
                {"n_estimators": 3, "min_samples_leaf": 1},
            ):
                report = train_v3_s3_controller(
                    sequence_features=features,
                    sequence_trials=trials,
                    external_baselines=baselines,
                    output=root / "controller",
                    training_jobs=2,
                )
            bundle = load_v3_s3_bundle(root / "controller")
            self.assertEqual(report["training_state_count"], 12)
            self.assertEqual(report["diagnostic_state_count"], 6)
            self.assertEqual(set(bundle.models), set(report["manifest"]["models"]))
            self.assertEqual(report["manifest"]["runtime_dependencies"], [])
            self.assertEqual(report["manifest"]["v2_runtime_call_count"], 0)
            self.assertEqual(report["manifest"]["adaptive_runtime_call_count"], 0)
            native_module = types.ModuleType("lns2_env")
            native_module.PortableTreeEnsemble = _PythonPortableTreeEnsemble
            native_module.repair_timing_schema = "lns2.repair_timing.v2"
            native_binary = root / "lns2_env.pyd"
            native_binary.write_bytes(b"native-audit-one")
            native_module.__file__ = str(native_binary)
            with mock.patch.dict(sys.modules, {"lns2_env": native_module}):
                finalized = finalize_v3_s3_native_audit(
                    controller_output=root / "controller", benchmark_rows=32
                )
            self.assertTrue(finalized["native_audit_completed"])
            self.assertTrue(finalized["pilot_checks"]["native_python_parity"])
            self.assertEqual(finalized["native_benchmark_sequence_count"], 32)
            self.assertEqual(finalized["native_parity_probe_count"], 32)
            report_path = root / "controller" / "training_report.json"
            manifest_path = root / "controller" / "v3_s3_manifest.json"
            finalized_bytes = (report_path.read_bytes(), manifest_path.read_bytes())
            with mock.patch.dict(
                sys.modules, {"lns2_env": native_module}
            ), mock.patch(
                "experiments.v3_s3_training.time.perf_counter",
                side_effect=AssertionError("completed audit was remeasured"),
            ):
                repeated = finalize_v3_s3_native_audit(
                    controller_output=root / "controller", benchmark_rows=32
                )
            self.assertEqual(
                repeated["native_audit_identity_fingerprint"],
                finalized["native_audit_identity_fingerprint"],
            )
            self.assertEqual(
                finalized_bytes,
                (report_path.read_bytes(), manifest_path.read_bytes()),
            )
            native_binary.write_bytes(b"native-audit-two")
            with mock.patch.dict(sys.modules, {"lns2_env": native_module}):
                with self.assertRaisesRegex(
                    ValueError, "identity differs"
                ):
                    finalize_v3_s3_native_audit(
                        controller_output=root / "controller",
                        benchmark_rows=32,
                    )
            self.assertEqual(
                finalized_bytes,
                (report_path.read_bytes(), manifest_path.read_bytes()),
            )
            tampered = json.loads(report_path.read_text(encoding="utf-8"))
            tampered["decision"] = "tampered"
            report_path.write_text(
                json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "training report SHA256 mismatch"):
                load_v3_s3_bundle(root / "controller")


if __name__ == "__main__":
    unittest.main()
