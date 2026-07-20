from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from research.engineering.balanced.balanced_controller import BalancedControllerConfig
from research.engineering.proposal_pruner.candidate_pruning import CandidatePruner
from experiments.closed_loop_confirmation import (
    _closed_loop_episode_worker,
    generate_online_candidates,
    load_frozen_policy_bundle,
    online_candidate_rows,
    resolve_controller_mode,
    score_online_candidates,
)
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.compact_controller_model import (
    CompactPortablePairwiseModel,
    compact_portable_payload,
    compact_runtime_model,
    export_controller_bundle,
    load_compact_model,
    load_controller_bundle,
)
from experiments.feature_schema_v2 import (
    PROFILE_FEATURE_NAMES,
    PROPOSAL_FAMILIES,
    REMOVED_FEATURE_NAMES,
    canonicalize_features,
    redundancy_violations,
    unsupported_actual_size,
)
from research.studies.representation.local_representation_audit import analyze_state, reconstruct_conflicts
from experiments.online_feature_engine import OnlineFeatureEngine, _native_batch_function
from experiments.repair_collection import state_fingerprint
from tests.runtime.test_closed_loop_confirmation import make_candidate, make_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _PythonBackedPortableTreeEnsemble:
    instance_count = 0

    def __init__(self, baseline: float, trees: list[list[dict]]) -> None:
        type(self).instance_count += 1
        self.reference = CompactPortablePairwiseModel(
            profile="fixture",
            input_features=[],
            base_feature_names=[],
            baseline=float(baseline),
            trees=trees,
            semantic_fingerprint="fixture",
        )

    def predict_positive(self, vectors: list[list[float]]) -> list[float]:
        return self.reference.predict_positive(vectors)


class _SignedModel:
    profile = "proposal_dynamic"
    feature_names: list[str] = []

    @staticmethod
    def pair_vector(left: dict, right: dict) -> list[float]:
        return [float(left["rank"]) - float(right["rank"])]

    @staticmethod
    def predict_positive(vectors: list[list[float]]) -> list[float]:
        return [0.9 if vector[0] > 0 else 0.1 for vector in vectors]


def _pruner_pool() -> tuple[list[dict], list[dict]]:
    candidates = []
    rows = []
    for family in PROPOSAL_FAMILIES:
        for representative, rank in (("a", 2), ("b", 1)):
            identifier = f"{family}-{representative}"
            candidates.append(
                {
                    "candidate_id": identifier,
                    "selection_families": [family],
                    "agents": [0, 1, 2, 3],
                }
            )
            rows.append(
                {
                    "candidate_id": identifier,
                    "candidate_key": identifier,
                    "rank": rank,
                    "features": {
                        "proposal_dynamic": canonicalize_features(
                            {"proposal.actual_size": 4.0}, "proposal_dynamic"
                        )
                    },
                }
            )
    return candidates, rows


def _refresh_conflicts(state: dict) -> None:
    events = reconstruct_conflicts(state["agents"])
    pairs = sorted({(event.left, event.right) for event in events})
    degree = {int(agent["id"]): 0 for agent in state["agents"]}
    for left, right in pairs:
        degree[left] += 1
        degree[right] += 1
    state["conflict_edges"] = [list(pair) for pair in pairs]
    state["num_of_colliding_pairs"] = len(pairs)
    state["feasible"] = not pairs
    state["done"] = not pairs
    for agent in state["agents"]:
        agent["conflict_degree"] = degree[int(agent["id"])]
        agent["path_cost"] = len(agent["path"]) - 1
    state["sum_of_costs"] = sum(agent["path_cost"] for agent in state["agents"])


class ControllerV2Tests(unittest.TestCase):
    def test_registered_feature_dimensions_and_redundancies(self) -> None:
        self.assertEqual(len(PROFILE_FEATURE_NAMES["proposal_dynamic"]), 82)
        self.assertEqual(len(PROFILE_FEATURE_NAMES["realized_dynamic"]), 124)
        self.assertEqual(len(REMOVED_FEATURE_NAMES), 15)
        state = make_state()
        _refresh_conflicts(state)
        candidate = make_candidate("candidate-a", [0, 1], "target:4")
        features = online_candidate_rows(state, [candidate])[0]["features"][
            "realized_dynamic"
        ]
        self.assertEqual(redundancy_violations(features), [])

    def test_compaction_remaps_linear_alias_threshold(self) -> None:
        payload = {
            "schema": "lns2.portable_pairwise_hist_gbdt.v1",
            "schema_version": 1,
            "profile": "realized_dynamic",
            "source_model_sha256": "fixture",
            "feature_names": ["state.degree_mean"],
            "baseline": 0.0,
            "trees": [
                [
                    {
                        "value": 0.0,
                        "feature_idx": 0,
                        "num_threshold": 2.0,
                        "missing_go_to_left": False,
                        "left": 1,
                        "right": 2,
                        "is_leaf": False,
                    },
                    {"value": -1.0, "is_leaf": True},
                    {"value": 1.0, "is_leaf": True},
                ]
            ],
        }
        compact = compact_portable_payload(payload)
        self.assertEqual(
            compact["input_features"],
            [{"mode": "delta", "name": "state.conflict_edge_density"}],
        )
        self.assertEqual(compact["trees"][0][0]["num_threshold"], 1.0)
        model = load_compact_model(compact)
        left = {
            "features": {"realized_dynamic": {"state.conflict_edge_density": 1.5}}
        }
        right = {
            "features": {"realized_dynamic": {"state.conflict_edge_density": 0.0}}
        }
        self.assertGreater(model.predict_positive([model.pair_vector(left, right)])[0], 0.5)

    def test_compacted_registered_ranker_is_exact(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs" / "movingai_ood_collection.json").read_text(
                encoding="utf-8"
            )
        )
        bundle = load_frozen_policy_bundle(
            PROJECT_ROOT / config["frozen_models"], config["model_registration"]
        )
        rows = []
        index_path = PROJECT_ROOT / config["model_registration"]["development_index"]
        with index_path.open(encoding="utf-8") as stream:
            first_state = None
            for line in stream:
                row = json.loads(line)
                first_state = first_state or row["state_id"]
                if row["state_id"] != first_state:
                    break
                rows.append(row)
        for profile in ("proposal_dynamic", "realized_dynamic"):
            compact = compact_runtime_model(bundle.models[profile])
            old_index, old_scores, _ = score_online_candidates(
                rows, bundle.models[profile]
            )
            new_index, new_scores, _ = score_online_candidates(rows, compact)
            self.assertEqual(old_index, new_index)
            self.assertEqual(old_scores, new_scores)
            dense_rows = []
            for row in rows:
                features = row["features"][profile]
                dense_rows.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "candidate_key": row["candidate_key"],
                        "feature_profile": profile,
                        "feature_names": tuple(compact.base_feature_names),
                        "feature_values": tuple(
                            float(features.get(name, 0.0))
                            for name in compact.base_feature_names
                        ),
                    }
                )
            dense_index, dense_scores, _ = score_online_candidates(
                dense_rows, compact
            )
            self.assertEqual(new_index, dense_index)
            self.assertEqual(new_scores, dense_scores)

    def test_python_incremental_engine_matches_reference(self) -> None:
        first = make_state()
        _refresh_conflicts(first)
        candidate = make_candidate("candidate-a", [0, 1], "target:4")
        engine = OnlineFeatureEngine(first, backend="python", shadow_validation=True)
        expected = online_candidate_rows(first, [candidate])[0]["features"]
        actual = engine.realized_rows([candidate], state_hash="first")[0][0]["features"]
        for name, value in canonicalize_features(
            expected["realized_dynamic"], "realized_dynamic"
        ).items():
            self.assertAlmostEqual(actual["realized_dynamic"][name], value, places=12)

        second = copy.deepcopy(first)
        second["agents"][0]["path"] = [0, 4, 5, 6, 2]
        _refresh_conflicts(second)
        metrics = engine.prepare(second, changed_agents=[0])
        self.assertTrue(metrics["incremental"])
        self.assertEqual(
            engine.analysis,
            analyze_state(second, static_grid=engine.static_grid),
        )
        unchanged = engine.prepare(second, changed_agents=[])
        self.assertTrue(unchanged["incremental_cache_hit"])

    def test_native_batch_engine_matches_reference_when_available(self) -> None:
        if _native_batch_function() is None:
            self.skipTest("native feature module is not built")
        state = make_state()
        candidates = [
            make_candidate("candidate-a", [0, 1], "target:4"),
            make_candidate("candidate-b", [1, 2], "collision:4"),
        ]
        engine = OnlineFeatureEngine(state, backend="native", shadow_validation=True)
        rows, _ = engine.realized_rows(candidates, state_hash="fixture")
        self.assertEqual(len(rows), 2)
        self.assertEqual(engine.backend, "native")

    def test_deployment_feature_projection_materializes_only_tree_inputs(self) -> None:
        state = make_state()
        _refresh_conflicts(state)
        candidate = make_candidate("candidate-a", [0, 1], "target:4")
        required = ("state.agent_count", "realized.path_cost_mean")
        for backend in (
            "python",
            *(('native',) if _native_batch_function() is not None else ()),
        ):
            engine = OnlineFeatureEngine(
                state,
                backend=backend,
                shadow_validation=True,
                required_features={"realized_dynamic": required},
            )
            rows, _ = engine.realized_rows([candidate], state_hash="projection")
            self.assertEqual(
                set(rows[0]["features"]["realized_dynamic"]), set(required)
            )

    def test_pruner_reduces_confident_pairs_and_falls_back_on_ood_size(self) -> None:
        candidates, rows = _pruner_pool()
        pruner = CandidatePruner(_SignedModel(), 0.8, {})
        retained, metrics = pruner.prune(candidates, rows)
        self.assertEqual(len(retained), len(PROPOSAL_FAMILIES))
        self.assertAlmostEqual(metrics["reduction_fraction"], 0.5)
        self.assertFalse(metrics["fallback"])
        rows[0]["features"]["proposal_dynamic"]["proposal.actual_size"] = 20.0
        retained, metrics = pruner.prune(candidates, rows)
        self.assertEqual(len(retained), len(candidates))
        self.assertEqual(metrics["fallback_reason"], "unsupported_actual_size")
        self.assertTrue(unsupported_actual_size({"proposal.actual_size": 16.5}))

    def test_controller_bundle_records_separate_schema_and_defaults(self) -> None:
        source = PROJECT_ROOT / "artifacts" / "initlns-closed-loop-policy-v1"
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_controller_bundle(
                source,
                directory,
                promotion_report={
                    "exact_acceleration_passed": True,
                    "feature_performance_passed": True,
                    "pruning_promotion_passed": False,
                },
            )
            loaded = load_controller_bundle(directory)
            self.assertEqual(manifest["default_controller"], "v2-full")
            self.assertEqual(
                loaded.manifest["feature_dimensions"],
                {"proposal_dynamic": 82, "realized_dynamic": 124},
            )
            self.assertIsNone(loaded.pruner_model)

    def test_bundle_loader_connects_native_predictor_and_matches_python(self) -> None:
        source = PROJECT_ROOT / "artifacts" / "initlns-closed-loop-policy-v1"
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_controller_bundle(source, directory)
            model_row = manifest["main_rankers"]["realized_dynamic"]
            payload = json.loads(
                (Path(directory) / model_row["file"]).read_text(encoding="utf-8")
            )
            python_model = load_compact_model(payload)
            fake_module = types.ModuleType("lns2_env")
            _PythonBackedPortableTreeEnsemble.instance_count = 0
            fake_module.PortableTreeEnsemble = _PythonBackedPortableTreeEnsemble
            with patch.dict(sys.modules, {"lns2_env": fake_module}):
                loaded = load_controller_bundle(directory)

            self.assertEqual(_PythonBackedPortableTreeEnsemble.instance_count, 2)
            native_model = loaded.main_models["realized_dynamic"]
            self.assertEqual(python_model.inference_backend, "python-portable-tree")
            self.assertEqual(native_model.inference_backend, "native-portable-tree")
            rows = []
            for candidate_index in range(18):
                features = {
                    name: float(
                        ((candidate_index + 3) * (feature_index + 5)) % 29
                    )
                    / 7.0
                    for feature_index, name in enumerate(native_model.feature_names)
                }
                rows.append(
                    {
                        "candidate_key": f"candidate-{candidate_index:02d}",
                        "features": {native_model.profile: features},
                    }
                )
            python_index, python_scores, python_margin = score_online_candidates(
                rows, python_model
            )
            native_index, native_scores, native_margin = score_online_candidates(
                rows, native_model
            )
            self.assertEqual(native_index, python_index)
            self.assertLessEqual(
                max(
                    abs(left - right)
                    for left, right in zip(native_scores, python_scores)
                ),
                1e-12,
            )
            self.assertLessEqual(abs(native_margin - python_margin), 1e-12)
            python_ranking = sorted(
                range(len(rows)),
                key=lambda index: (
                    -round(python_scores[index], 12),
                    rows[index]["candidate_key"],
                ),
            )
            native_ranking = sorted(
                range(len(rows)),
                key=lambda index: (
                    -round(native_scores[index], 12),
                    rows[index]["candidate_key"],
                ),
            )
            self.assertEqual(native_ranking, python_ranking)

    def test_bundle_loader_uses_python_when_native_type_is_missing(self) -> None:
        source = PROJECT_ROOT / "artifacts" / "initlns-closed-loop-policy-v1"
        with tempfile.TemporaryDirectory() as directory:
            export_controller_bundle(source, directory)
            module_without_predictor = types.ModuleType("lns2_env")
            with patch.dict(sys.modules, {"lns2_env": module_without_predictor}):
                loaded = load_controller_bundle(directory)
            self.assertTrue(
                all(
                    model.inference_backend == "python-portable-tree"
                    for model in loaded.main_models.values()
                )
            )

    def test_bundle_loader_does_not_hide_native_constructor_failure(self) -> None:
        source = PROJECT_ROOT / "artifacts" / "initlns-closed-loop-policy-v1"

        class IncompatiblePortableTreeEnsemble:
            def __init__(self, baseline: float, trees: list[list[dict]]) -> None:
                raise RuntimeError("native bundle incompatibility")

        with tempfile.TemporaryDirectory() as directory:
            export_controller_bundle(source, directory)
            fake_module = types.ModuleType("lns2_env")
            fake_module.PortableTreeEnsemble = IncompatiblePortableTreeEnsemble
            with patch.dict(sys.modules, {"lns2_env": fake_module}):
                with self.assertRaisesRegex(
                    RuntimeError, "native bundle incompatibility"
                ):
                    load_controller_bundle(directory)

    def test_registered_controller_default_resolves_to_promoted_v2_full(self) -> None:
        mode, _, manifest = resolve_controller_mode(PROJECT_ROOT, None)
        self.assertEqual(mode, "v2-full")
        self.assertIsNotNone(manifest)

    def test_revision_only_proposal_check_avoids_full_state_copy(self) -> None:
        state = make_state()

        class Environment:
            revision = 7

            def get_state_revision(self) -> int:
                return self.revision

            def propose_batch(self, actions: list[dict]) -> list[dict]:
                return [
                    {
                        "action_valid": True,
                        "generated": True,
                        "neighborhood": [0, 1],
                    }
                    for _ in actions
                ]

            def get_state(self) -> dict:
                raise AssertionError("full state copy should be skipped")

        _, metrics = generate_online_candidates(
            Environment(),
            state,
            task_id="task",
            solver_seed=1,
            decision_index=0,
            proposal_config={
                "max_seed_agents": 1,
                "heuristics": ["target"],
                "neighborhood_sizes": [4],
                "trials": 1,
                "candidates_per_family": 1,
            },
            state_hash=state_fingerprint(state),
            verify_full_state=False,
        )
        self.assertEqual(metrics["state_check_backend"], "revision")
        self.assertFalse(metrics["full_state_verified"])

    def test_compact_proposal_backend_matches_reference_shadow(self) -> None:
        state = make_state()

        class Environment:
            revision = 11

            def get_state_revision(self) -> int:
                return self.revision

            def propose_batch(self, actions: list[dict]) -> list[dict]:
                return [
                    {
                        "action_valid": True,
                        "generated": True,
                        "neighborhood": [0, 1],
                    }
                    for _ in actions
                ]

            def propose_batch_compact(self, actions: list[dict]) -> list[tuple]:
                return [(True, True, [0, 1]) for _ in actions]

            def get_state(self) -> dict:
                return state

        candidates, metrics = generate_online_candidates(
            Environment(),
            state,
            task_id="task",
            solver_seed=1,
            decision_index=0,
            proposal_config={
                "max_seed_agents": 1,
                "heuristics": ["target"],
                "neighborhood_sizes": [4],
                "trials": 1,
                "candidates_per_family": 1,
            },
            state_hash=state_fingerprint(state),
            proposal_backend="optimized",
            shadow_validation=True,
        )
        self.assertEqual(metrics["backend"], "compact")
        self.assertTrue(metrics["shadow_validation_passed"])
        self.assertEqual(candidates[0]["agents"], [0, 1])

    def test_v2_full_worker_executes_one_learned_decision(self) -> None:
        initial = make_state()
        _refresh_conflicts(initial)
        final = copy.deepcopy(initial)
        final["iteration"] = 1
        final["num_of_colliding_pairs"] = 0
        final["conflict_edges"] = []
        final["feasible"] = True
        final["done"] = True

        class Environment:
            def __init__(self) -> None:
                self.state = initial
                self.revision = 1

            def reset(self, seed: int) -> dict:
                return self.state

            def get_state_revision(self) -> int:
                return self.revision

            def get_state(self) -> dict:
                return self.state

            def propose_batch(self, actions: list[dict]) -> list[dict]:
                return [
                    {
                        "action_valid": True,
                        "generated": True,
                        "neighborhood": [0, 1],
                    }
                    for _ in actions
                ]

            def step(self, action: dict) -> dict:
                self.state = final
                self.revision += 1
                return {
                    "observation": final,
                    "metrics": {
                        "action_valid": True,
                        "neighborhood": list(action["agents"]),
                        "requested_random_seed": int(action["random_seed"]),
                        "conflicts_before": int(initial["num_of_colliding_pairs"]),
                        "conflicts_after": 0,
                    },
                    "terminated": True,
                    "truncated": False,
                }

        config = json.loads(
            (PROJECT_ROOT / "configs" / "movingai_ood_collection.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            job = {
                "row": {
                    "split": "closed_loop",
                    "map_id": "map-a",
                    "task_id": "task-a",
                    "layout_mode": "regular_beltway",
                    "task_variant": "balanced_80",
                    "agent_count": 4,
                },
                "policy": "realized_dynamic",
                "solver_seed": 1,
                "output_root": directory,
                "run_fingerprint": "v2-run",
                "resume": False,
                "dataset_root": directory,
                "environment": {},
                "max_decisions": 2,
                "metric_iteration_budget": 2,
                "wall_time_budget_seconds": 300.0,
                "proposal": {
                    "max_seed_agents": 1,
                    "heuristics": ["target"],
                    "neighborhood_sizes": [4],
                    "trials": 1,
                    "candidates_per_family": 1,
                },
                "frozen_models": str(PROJECT_ROOT / config["frozen_models"]),
                "model_registration": config["model_registration"],
                "controller": "v2-full",
                "feature_backend": "python",
                "controller_bundle": str(
                    PROJECT_ROOT / "artifacts" / "initlns-closed-loop-controller-v2"
                ),
                "feature_shadow_validation": True,
                "proposal_state_verification": "always",
            }
            with patch(
                "experiments.closed_loop_confirmation._make_environment",
                return_value=Environment(),
            ):
                result = _closed_loop_episode_worker(job)
            events = read_trace_events(Path(directory) / result["trace_file"])
            transition = next(
                event for event in events if event.get("event") == "transition"
            )
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertEqual(result["summary"]["controller_mode"], "v2-full")
        self.assertEqual(result["summary"]["repair_iterations"], 1)
        self.assertIn(
            transition["controller"]["inference_backend"],
            {"native-portable-tree", "python-portable-tree"},
        )
        self.assertGreater(
            result["summary"]["controller_totals"]["realized_feature_seconds"], 0
        )

    def test_v2_balanced_official_route_skips_features_and_is_accounted(self) -> None:
        initial = make_state()
        _refresh_conflicts(initial)
        final = copy.deepcopy(initial)
        final["iteration"] = 1
        final["num_of_colliding_pairs"] = 0
        final["conflict_edges"] = []
        final["feasible"] = True
        final["done"] = True

        class Environment:
            def reset(self, seed: int) -> dict:
                return initial

            def step(self, action: dict) -> dict:
                self.assert_official(action)
                return {
                    "observation": final,
                    "metrics": {
                        "action_valid": True,
                        "neighborhood": [0, 1],
                        "conflicts_before": int(initial["num_of_colliding_pairs"]),
                        "conflicts_after": 0,
                    },
                    "terminated": True,
                    "truncated": False,
                }

            @staticmethod
            def assert_official(action: dict) -> None:
                if action != {"mode": "official"}:
                    raise AssertionError(f"unexpected action: {action}")

        config = json.loads(
            (PROJECT_ROOT / "configs" / "movingai_ood_collection.json").read_text(
                encoding="utf-8"
            )
        )
        balanced = BalancedControllerConfig(
            conflict_threshold=16,
            pruner_threshold=None,
            source={"study_role": "fixture"},
        )
        with tempfile.TemporaryDirectory() as directory:
            job = {
                "row": {
                    "split": "closed_loop",
                    "map_id": "map-a",
                    "task_id": "task-a",
                    "layout_mode": "regular_beltway",
                    "task_variant": "balanced_80",
                    "agent_count": 4,
                },
                "policy": "realized_dynamic",
                "solver_seed": 1,
                "output_root": directory,
                "run_fingerprint": "balanced-run",
                "resume": False,
                "dataset_root": directory,
                "environment": {},
                "max_decisions": 2,
                "metric_iteration_budget": 2,
                "wall_time_budget_seconds": 300.0,
                "proposal": {
                    "max_seed_agents": 1,
                    "heuristics": ["target"],
                    "neighborhood_sizes": [4],
                    "trials": 1,
                    "candidates_per_family": 1,
                },
                "frozen_models": str(PROJECT_ROOT / config["frozen_models"]),
                "model_registration": config["model_registration"],
                "controller": "v2-balanced",
                "balanced_config": balanced.payload(),
                "feature_backend": "python",
                "controller_bundle": str(
                    PROJECT_ROOT / "artifacts" / "initlns-closed-loop-controller-v2"
                ),
                "feature_shadow_validation": False,
                "proposal_state_verification": "always",
            }
            with patch(
                "experiments.closed_loop_confirmation._make_environment",
                return_value=Environment(),
            ):
                result = _closed_loop_episode_worker(job)
        self.assertEqual(result["status"], "ok", result.get("error"))
        summary = result["summary"]
        self.assertEqual(summary["model_decision_count"], 0)
        self.assertEqual(summary["official_decision_count"], 1)
        self.assertEqual(summary["repair_iterations"], 1)
        self.assertEqual(summary["feature_backend"], "not_used")
        self.assertEqual(
            summary["controller_totals"]["official_decision_count"], 1
        )
        self.assertEqual(
            summary["controller_totals"].get("candidate_count", 0), 0
        )

    def test_v2_stall_safe_backs_off_then_uses_official_fallback(self) -> None:
        initial = make_state()
        _refresh_conflicts(initial)
        states = []
        for iteration in range(8):
            state = copy.deepcopy(initial)
            state["iteration"] = iteration
            state["low_level"] = {
                "expanded": iteration,
                "generated": iteration,
                "reopened": 0,
                "runs": iteration,
            }
            state["num_of_colliding_pairs"] = 5 if iteration < 7 else 0
            state["feasible"] = iteration == 7
            state["done"] = iteration == 7
            if iteration == 7:
                state["conflict_edges"] = []
            states.append(state)

        class Environment:
            def __init__(self) -> None:
                self.index = 0

            def reset(self, seed: int) -> dict:
                self.index = 0
                return states[0]

            def step(self, action: dict) -> dict:
                expected_sizes = (4, 4, 3, 3, 2, 2)
                if self.index < 6:
                    if action.get("mode") != "explicit_neighborhood":
                        raise AssertionError(f"unexpected model action: {action}")
                    if len(action["agents"]) != expected_sizes[self.index]:
                        raise AssertionError(f"unexpected guarded size: {action}")
                    neighborhood = list(action["agents"])
                    replan_success = False
                else:
                    if action.get("mode") != "official" or "random_seed" not in action:
                        raise AssertionError(f"unexpected fallback action: {action}")
                    neighborhood = [0, 1]
                    replan_success = True
                before = states[self.index]
                self.index += 1
                after = states[self.index]
                metrics = {
                    "action_valid": True,
                    "neighborhood": neighborhood,
                    "requested_random_seed": int(action["random_seed"]),
                    "conflicts_before": int(before["num_of_colliding_pairs"]),
                    "conflicts_after": int(after["num_of_colliding_pairs"]),
                    "replan_success": replan_success,
                }
                return {
                    "observation": after,
                    "metrics": metrics,
                    "terminated": bool(after["feasible"]),
                    "truncated": False,
                }

        class FeatureEngine:
            def __init__(self, state: dict, **_kwargs: object) -> None:
                self.backend = "fixture"
                self.last_prepare_metrics = {"state_analysis_seconds": 0.0}
                self.last_shadow_rows: dict[str, list[dict]] = {}

            def prepare(self, state: dict, *, changed_agents: list[int]) -> dict:
                return {"state_analysis_seconds": 0.0}

            def realized_rows(
                self, candidates: list[dict], *, state_hash: str
            ) -> tuple[list[dict], dict]:
                return (
                    [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "candidate_key": candidate["candidate_id"],
                            "features": {"realized_dynamic": {}},
                        }
                        for candidate in candidates
                    ],
                    {"realized_feature_seconds": 0.0},
                )

        candidate_rows = [
            {
                "candidate_id": "large",
                "agents": [0, 1, 2, 3],
                "actual_size": 4,
                "selection_families": ["target:4"],
                "proposal_count_by_family": {"target:4": 1},
                "proposal_seeds": [10],
                "seed_agents": [0],
            },
            {
                "candidate_id": "medium",
                "agents": [0, 1, 2],
                "actual_size": 3,
                "selection_families": ["target:3"],
                "proposal_count_by_family": {"target:3": 1},
                "proposal_seeds": [11],
                "seed_agents": [0],
            },
            {
                "candidate_id": "small",
                "agents": [0, 1],
                "actual_size": 2,
                "selection_families": ["target:2"],
                "proposal_count_by_family": {"target:2": 1},
                "proposal_seeds": [12],
                "seed_agents": [0],
            },
        ]

        def generated(*_args: object, **_kwargs: object) -> tuple[list[dict], dict]:
            return (
                copy.deepcopy(candidate_rows),
                {
                    "proposal_count": 3,
                    "candidate_count": 3,
                    "proposal_seconds": 0.0,
                    "candidate_generation_seconds": 0.0,
                    "state_check_seconds": 0.0,
                    "state_check_fingerprint_seconds": 0.0,
                    "state_check_backend": "fixture",
                    "full_state_verified": True,
                    "state_revision": 1,
                    "backend": "fixture",
                },
            )

        config = json.loads(
            (PROJECT_ROOT / "configs" / "movingai_ood_collection.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            job = {
                "row": {
                    "split": "closed_loop",
                    "map_id": "map-a",
                    "task_id": "task-a",
                    "layout_mode": "regular_beltway",
                    "task_variant": "balanced_80",
                    "agent_count": 4,
                },
                "policy": "realized_dynamic",
                "solver_seed": 1,
                "output_root": directory,
                "run_fingerprint": "stall-safe-run",
                "resume": False,
                "dataset_root": directory,
                "environment": {},
                "max_decisions": 7,
                "metric_iteration_budget": 7,
                "wall_time_budget_seconds": 300.0,
                "proposal": {
                    "max_seed_agents": 1,
                    "heuristics": ["target"],
                    "neighborhood_sizes": [2, 3, 4],
                    "trials": 1,
                    "candidates_per_family": 1,
                },
                "frozen_models": str(PROJECT_ROOT / config["frozen_models"]),
                "model_registration": config["model_registration"],
                "controller": "v2-stall-safe",
                "stall_guard_config": {
                    "schema": "lns2.stall_guard.v1",
                    "schema_version": 1,
                    "unchanged_state_attempts_per_level": 2,
                    "size_caps": [4, 3, 2],
                    "terminal_fallback": "official_adaptive",
                    "reset_on_state_fingerprint_change": True,
                },
                "feature_backend": "python",
                "controller_bundle": str(
                    PROJECT_ROOT / "artifacts" / "initlns-closed-loop-controller-v2"
                ),
                "feature_shadow_validation": False,
                "proposal_state_verification": "always",
            }
            with (
                patch(
                    "experiments.closed_loop_confirmation._make_environment",
                    return_value=Environment(),
                ),
                patch(
                    "experiments.closed_loop_confirmation.OnlineFeatureEngine",
                    FeatureEngine,
                ),
                patch(
                    "experiments.closed_loop_confirmation.generate_online_candidates",
                    side_effect=generated,
                ),
                patch(
                    "experiments.closed_loop_confirmation.score_online_candidates",
                    return_value=(0, [3.0, 2.0, 1.0], 1.0),
                ),
            ):
                result = _closed_loop_episode_worker(job)
            events = read_trace_events(Path(directory) / result["trace_file"])

        self.assertEqual(result["status"], "ok", result.get("error"))
        summary = result["summary"]
        self.assertEqual(summary["model_decision_count"], 6)
        self.assertEqual(summary["official_decision_count"], 1)
        self.assertEqual(summary["stall_guard"]["size_backoff_count"], 2)
        self.assertEqual(
            summary["stall_guard"]["official_fallback_decision_count"], 1
        )
        self.assertEqual(summary["stall_guard"]["rescued_state_count"], 1)
        transitions = [row for row in events if row.get("event") == "transition"]
        self.assertEqual(
            [row["controller"]["route"] for row in transitions],
            ["model"] * 6 + ["official_adaptive"],
        )
        fallback = transitions[-1]["controller"]
        self.assertIsNotNone(fallback["base_selected_candidate_id"])
        self.assertTrue(fallback["candidate_pool"])
        self.assertIsNone(fallback["selected_candidate_id"])
        self.assertEqual(
            fallback["stall_guard"]["final_neighborhood_size"], 2
        )

    def test_v2_balanced_accumulates_changes_across_skipped_model_steps(self) -> None:
        initial = make_state()
        _refresh_conflicts(initial)
        conflict_sequence = (5, 3, 2, 5, 0)
        states = []
        for iteration, conflicts in enumerate(conflict_sequence):
            state = copy.deepcopy(initial)
            state["iteration"] = iteration
            state["num_of_colliding_pairs"] = conflicts
            state["feasible"] = conflicts == 0
            state["done"] = conflicts == 0
            if conflicts == 0:
                state["conflict_edges"] = []
            states.append(state)

        class Environment:
            def __init__(self) -> None:
                self.index = 0

            def reset(self, seed: int) -> dict:
                self.index = 0
                return states[0]

            def step(self, action: dict) -> dict:
                before = states[self.index]
                expected_model = int(before["num_of_colliding_pairs"]) > 4
                if expected_model != (action["mode"] == "explicit_neighborhood"):
                    raise AssertionError(f"unexpected route action: {action}")
                neighborhood = list(action.get("agents", [self.index - 1]))
                if not expected_model:
                    neighborhood = [0] if self.index == 1 else [1]
                self.index += 1
                after = states[self.index]
                metrics = {
                    "action_valid": True,
                    "neighborhood": neighborhood,
                    "conflicts_before": int(before["num_of_colliding_pairs"]),
                    "conflicts_after": int(after["num_of_colliding_pairs"]),
                }
                if expected_model:
                    metrics["requested_random_seed"] = int(action["random_seed"])
                return {
                    "observation": after,
                    "metrics": metrics,
                    "terminated": bool(after["feasible"]),
                    "truncated": False,
                }

        class FeatureEngine:
            prepare_calls: list[list[int]] = []

            def __init__(self, state: dict, **_kwargs: object) -> None:
                self.backend = "fixture"
                self.last_prepare_metrics = {"state_analysis_seconds": 0.0}
                self.last_shadow_rows: dict[str, list[dict]] = {}

            def prepare(self, state: dict, *, changed_agents: list[int]) -> dict:
                self.prepare_calls.append(list(changed_agents))
                self.last_prepare_metrics = {"state_analysis_seconds": 0.0}
                return dict(self.last_prepare_metrics)

            def realized_rows(
                self, candidates: list[dict], *, state_hash: str
            ) -> tuple[list[dict], dict]:
                return (
                    [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "candidate_key": candidate["candidate_id"],
                            "features": {"realized_dynamic": {}},
                        }
                        for candidate in candidates
                    ],
                    {"realized_feature_seconds": 0.0},
                )

        def candidates(*_args: object, **_kwargs: object) -> tuple[list[dict], dict]:
            return (
                [
                    {
                        "candidate_id": "candidate-a",
                        "agents": [2],
                        "actual_size": 1,
                        "selection_families": ["target:4"],
                        "proposal_count_by_family": {"target:4": 1},
                        "proposal_seeds": [123],
                        "seed_agents": [2],
                    }
                ],
                {
                    "proposal_count": 1,
                    "candidate_count": 1,
                    "proposal_seconds": 0.0,
                    "state_check_seconds": 0.0,
                    "state_check_backend": "fixture",
                    "full_state_verified": True,
                    "state_revision": 1,
                    "backend": "fixture",
                },
            )

        config = json.loads(
            (PROJECT_ROOT / "configs" / "movingai_ood_collection.json").read_text(
                encoding="utf-8"
            )
        )
        balanced = BalancedControllerConfig(
            conflict_threshold=4,
            pruner_threshold=None,
            source={"study_role": "fixture"},
        )
        with tempfile.TemporaryDirectory() as directory:
            job = {
                "row": {
                    "split": "closed_loop",
                    "map_id": "map-a",
                    "task_id": "task-a",
                    "layout_mode": "regular_beltway",
                    "task_variant": "balanced_80",
                    "agent_count": 4,
                },
                "policy": "realized_dynamic",
                "solver_seed": 1,
                "output_root": directory,
                "run_fingerprint": "balanced-accumulation-run",
                "resume": False,
                "dataset_root": directory,
                "environment": {},
                "max_decisions": 4,
                "metric_iteration_budget": 4,
                "wall_time_budget_seconds": 300.0,
                "proposal": {
                    "max_seed_agents": 1,
                    "heuristics": ["target"],
                    "neighborhood_sizes": [4],
                    "trials": 1,
                    "candidates_per_family": 1,
                },
                "frozen_models": str(PROJECT_ROOT / config["frozen_models"]),
                "model_registration": config["model_registration"],
                "controller": "v2-balanced",
                "balanced_config": balanced.payload(),
                "feature_backend": "python",
                "controller_bundle": str(
                    PROJECT_ROOT / "artifacts" / "initlns-closed-loop-controller-v2"
                ),
                "feature_shadow_validation": False,
                "proposal_state_verification": "always",
            }
            with (
                patch(
                    "experiments.closed_loop_confirmation._make_environment",
                    return_value=Environment(),
                ),
                patch(
                    "experiments.closed_loop_confirmation.OnlineFeatureEngine",
                    FeatureEngine,
                ),
                patch(
                    "experiments.closed_loop_confirmation.generate_online_candidates",
                    side_effect=candidates,
                ),
            ):
                result = _closed_loop_episode_worker(job)
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertEqual(result["summary"]["model_decision_count"], 2)
        self.assertEqual(result["summary"]["official_decision_count"], 2)
        self.assertEqual(FeatureEngine.prepare_calls, [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
