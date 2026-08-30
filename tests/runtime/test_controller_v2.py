from __future__ import annotations

import copy
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.closed_loop_confirmation import (
    CONTROLLER_MODES,
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
from experiments.state_analysis import analyze_state, reconstruct_conflicts
from experiments.online_feature_engine import (
    OnlineFeatureEngine,
    TopologyAnalysisCache,
    _native_batch_function,
    _native_topology_event_function,
)
from experiments.repair_collection import state_fingerprint
from lns2_selector.controllers.v2 import PairwiseV2Selector
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
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
    def test_active_controller_modes_are_minimal(self) -> None:
        self.assertEqual(
            CONTROLLER_MODES,
            (
                "official_adaptive",
                "v2-full",
                "mixed-full-v2",
                "v3-s3",
            ),
        )

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
            superset_names = tuple(PROFILE_FEATURE_NAMES[profile])
            self.assertTrue(set(compact.base_feature_names) <= set(superset_names))
            superset_rows = []
            for row in rows:
                features = row["features"][profile]
                superset_rows.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "candidate_key": row["candidate_key"],
                        "feature_profile": profile,
                        "feature_names": superset_names,
                        "feature_values": tuple(
                            float(features.get(name, 0.0))
                            for name in superset_names
                        ),
                    }
                )
            superset_index, superset_scores, _ = score_online_candidates(
                superset_rows, compact
            )
            self.assertEqual(new_index, superset_index)
            self.assertEqual(new_scores, superset_scores)

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

    def test_topology_analysis_cache_preserves_candidate_inputs(self) -> None:
        first = make_state()
        _refresh_conflicts(first)
        cache = TopologyAnalysisCache(first)

        def assert_equivalent(state: dict) -> None:
            expected = analyze_state(state, static_grid=cache.static_grid)
            actual = cache.analysis
            self.assertIsNotNone(actual)
            assert actual is not None
            self.assertEqual(actual.events, expected.events)
            self.assertEqual(actual.pair_set, expected.pair_set)
            self.assertEqual(actual.component_id, expected.component_id)
            self.assertEqual(actual.component_members, expected.component_members)
            self.assertEqual(actual.degrees, expected.degrees)
            self.assertEqual(actual.articulation, expected.articulation)
            self.assertEqual(actual.visit_heat, expected.visit_heat)
            self.assertEqual(actual.agent_heat, expected.agent_heat)

        assert_equivalent(first)
        second = copy.deepcopy(first)
        second["agents"][0]["path"] = [0, 4, 5, 6, 2]
        _refresh_conflicts(second)
        cache.prepare(second, changed_agents=[0])
        assert_equivalent(second)

    def test_topology_analysis_cache_random_updates_match_full_reconstruction(self) -> None:
        generator = random.Random(1729)

        def random_path(length: int) -> list[int]:
            path = [generator.randrange(36)]
            while len(path) < length:
                row, column = divmod(path[-1], 6)
                neighbors = [path[-1]]
                if row:
                    neighbors.append(path[-1] - 6)
                if row < 5:
                    neighbors.append(path[-1] + 6)
                if column:
                    neighbors.append(path[-1] - 1)
                if column < 5:
                    neighbors.append(path[-1] + 1)
                path.append(generator.choice(neighbors))
            return path

        state = {
            "rows": 6,
            "cols": 6,
            "obstacles": [0] * 36,
            "agents": [
                {
                    "id": agent_id,
                    "path": random_path(8),
                }
                for agent_id in range(24)
            ],
        }
        _refresh_conflicts(state)
        cache = TopologyAnalysisCache(state, backend="python")
        native_cache = (
            TopologyAnalysisCache(state, backend="native", shadow_interval=5)
            if _native_topology_event_function() is not None
            else None
        )
        for step in range(30):
            updated = copy.deepcopy(state)
            changed = set(generator.sample(range(24), generator.randrange(1, 6)))
            for agent_id in changed:
                path_length = generator.randrange(3, 15) if step % 7 == 0 else 8
                updated["agents"][agent_id]["path"] = random_path(path_length)
            _refresh_conflicts(updated)
            cache.prepare(updated, changed_agents=sorted(changed))
            expected = analyze_state(updated, static_grid=cache.static_grid)
            self.assertIsNotNone(cache.analysis)
            assert cache.analysis is not None
            self.assertEqual(cache.analysis.events, expected.events)
            self.assertEqual(cache.analysis.pair_set, expected.pair_set)
            self.assertEqual(cache.analysis.component_id, expected.component_id)
            self.assertEqual(cache.analysis.component_members, expected.component_members)
            self.assertEqual(cache.analysis.visit_heat, expected.visit_heat)
            self.assertEqual(cache.analysis.agent_heat, expected.agent_heat)
            if native_cache is not None:
                native_cache.prepare(updated, changed_agents=sorted(changed))
                self.assertIsNotNone(native_cache.analysis)
                assert native_cache.analysis is not None
                self.assertEqual(native_cache.analysis, expected)
                self.assertTrue(native_cache.last_prepare_incremental)
                self.assertEqual(
                    native_cache.last_shadow_validation, (step + 1) % 5 == 0
                )
            state = updated

    def test_native_topology_events_match_full_reconstruction(self) -> None:
        if _native_topology_event_function() is None:
            self.skipTest("native topology event extraction is not built")
        state = make_state()
        _refresh_conflicts(state)
        cache = TopologyAnalysisCache(state, backend="native")
        self.assertEqual(cache.backend, "native")
        expected = analyze_state(state, static_grid=cache.static_grid)
        self.assertIsNotNone(cache.analysis)
        assert cache.analysis is not None
        self.assertEqual(cache.analysis.events, expected.events)
        self.assertEqual(cache.analysis.pair_set, expected.pair_set)
        self.assertEqual(cache.analysis.component_id, expected.component_id)
        self.assertEqual(cache.analysis.component_members, expected.component_members)

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

    def test_prepared_native_analysis_is_shared_without_feature_changes(self) -> None:
        if _native_batch_function() is None:
            self.skipTest("native feature module is not built")
        state = make_state()
        _refresh_conflicts(state)
        candidates = [
            make_candidate("candidate-a", [0, 1], "target:4"),
            make_candidate("candidate-b", [1, 2], "collision:4"),
        ]
        topology = TopologyAnalysisCache(state, backend="native")
        prepared = topology.last_native_prepared
        if prepared is None:
            self.skipTest("prepared native analysis is not built")

        separate = OnlineFeatureEngine(
            state, backend="native", dense_output=True
        )
        shared = OnlineFeatureEngine(
            state, backend="native", dense_output=True
        )
        shared.prepare(state, prepared_native_analysis=prepared)
        for profile, method in (
            ("proposal_dynamic", "proposal_rows"),
            ("realized_dynamic", "realized_rows"),
        ):
            expected, _ = getattr(separate, method)(
                candidates, state_hash="fixture"
            )
            actual, metrics = getattr(shared, method)(
                candidates, state_hash="fixture"
            )
            self.assertEqual(actual, expected, profile)
            self.assertEqual(metrics["prepared_analysis_reused"], 1.0)
            self.assertEqual(metrics["state_analysis_seconds"], 0.0)

        with self.assertRaisesRegex(ValueError, "another state"):
            shared.prepare(
                copy.deepcopy(state), prepared_native_analysis=prepared
            )

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

    def test_bundle_loader_rejects_ranges_for_pruned_features(self) -> None:
        source = PROJECT_ROOT / "artifacts" / "initlns-closed-loop-policy-v1"
        with tempfile.TemporaryDirectory() as directory:
            export_controller_bundle(source, directory)
            manifest_path = Path(directory) / "controller_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["main_ranges"]["realized_dynamic"][
                "unused.test_feature"
            ] = [0.0, 1.0]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "main feature ranges differ from compact inputs"
            ):
                load_controller_bundle(directory)

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

    def test_controller_mode_rejects_mismatched_bundle_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching controller bundle"):
            resolve_controller_mode(
                PROJECT_ROOT,
                "mixed-full-v2",
                "artifacts/initlns-closed-loop-controller-v2",
            )
        with self.assertRaisesRegex(ValueError, "matching controller bundle"):
            resolve_controller_mode(
                PROJECT_ROOT,
                "v2-full",
                "artifacts/initlns-mixed-full-controller-v2",
            )

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

    def test_grouped_seed_grid_backend_matches_reference_shadow(self) -> None:
        state = make_state()

        class Environment:
            revision = 13

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

            def propose_seed_grid_grouped(
                self,
                seed_agents: list[int],
                heuristics: list[str],
                sizes: list[int],
                random_seeds: list[int],
                trials: int,
            ) -> dict:
                self.assert_grid = (
                    seed_agents,
                    heuristics,
                    sizes,
                    len(random_seeds),
                    trials,
                )
                return {
                    "proposal_count": len(random_seeds),
                    "unique_neighborhood_count": 1,
                    "invalid_indices": [],
                    "rows": [([0, 1], list(range(len(random_seeds))))],
                }

            def get_state(self) -> dict:
                return state

        environment = Environment()
        candidates, metrics = generate_online_candidates(
            environment,
            state,
            task_id="task",
            solver_seed=1,
            decision_index=0,
            proposal_config={
                "max_seed_agents": 1,
                "heuristics": ["target", "collision"],
                "neighborhood_sizes": [4],
                "trials": 2,
                "candidates_per_family": 1,
            },
            state_hash=state_fingerprint(state),
            proposal_backend="optimized",
            shadow_validation=True,
        )
        self.assertEqual(metrics["backend"], "grouped_seed_grid")
        self.assertEqual(metrics["proposal_count"], 4)
        self.assertEqual(metrics["unique_neighborhood_count"], 1)
        self.assertTrue(metrics["shadow_validation_passed"])
        self.assertEqual(candidates[0]["agents"], [0, 1])

    def test_proposal_seed_override_restricts_grouped_native_grid(self) -> None:
        state = make_state()

        class Environment:
            revision = 17

            def get_state_revision(self) -> int:
                return self.revision

            def propose_seed_grid_grouped(
                self,
                seed_agents: list[int],
                heuristics: list[str],
                sizes: list[int],
                random_seeds: list[int],
                trials: int,
            ) -> dict:
                self.seed_agents = list(seed_agents)
                return {
                    "proposal_count": len(random_seeds),
                    "unique_neighborhood_count": 1,
                    "invalid_indices": [],
                    "rows": [([0, 1], list(range(len(random_seeds))))],
                }

            def get_state(self) -> dict:
                return state

        environment = Environment()
        _candidates, metrics = generate_online_candidates(
            environment,
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
            shadow_validation=False,
            seed_agents_override=[1],
        )
        self.assertEqual(environment.seed_agents, [1])
        self.assertEqual(metrics["seed_agents"], [1])
        self.assertTrue(metrics["seed_agents_overridden"])

    def test_v2_full_worker_executes_one_learned_decision(self) -> None:
        selected_profiles: list[str] = []
        original_select = PairwiseV2Selector.select

        def tracked_select(selector, request):
            selected_profiles.append(str(request.profile))
            return original_select(selector, request)

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
            with (
                patch(
                    "experiments.closed_loop_confirmation._make_environment",
                    return_value=Environment(),
                ),
                patch.object(PairwiseV2Selector, "select", new=tracked_select),
            ):
                result = _closed_loop_episode_worker(job)
            events = read_trace_events(Path(directory) / result["trace_file"])
            transition = next(
                event for event in events if event.get("event") == "transition"
            )
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertEqual(result["summary"]["controller_mode"], "v2-full")
        self.assertEqual(result["summary"]["repair_iterations"], 1)
        self.assertEqual(selected_profiles, ["realized_dynamic"])
        self.assertIn(
            transition["controller"]["inference_backend"],
            {"native-portable-tree", "python-portable-tree"},
        )
        self.assertGreater(
            result["summary"]["controller_totals"]["realized_feature_seconds"], 0
        )
