from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from research.studies.neighborhood.realized_neighborhood_ranking_audit import (
    acceptance_report,
    batch_model_selections,
    build_ranking_index,
    context_permutation_test,
    cross_validate,
    effectiveness_dominates,
    explicit_neighborhood_features,
    leave_one_map_out_folds,
    replace_context_bundle,
    train_pairwise_model,
)
from research.studies.representation.local_representation_audit import analyze_state


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _agent(identifier: int, path: list[int], delay: int = 0) -> dict:
    return {
        "id": identifier,
        "start": path[0],
        "goal": path[-1],
        "path": path,
        "path_cost": len(path) - 1,
        "shortest_path_cost": max(1, len(path) - 1),
        "conflict_degree": int(identifier in {0, 1}),
        "delay": delay,
    }


def _state(map_id: str = "map-a", split: str = "probe") -> dict:
    return {
        "rows": 1,
        "cols": 4,
        "obstacles": [0, 0, 0, 0],
        "agents": [
            _agent(0, [0, 1], 3),
            _agent(1, [1, 0], 2),
            _agent(2, [2, 2], 1),
            _agent(3, [3, 3], 0),
        ],
        "conflict_edges": [[0, 1]],
        "num_of_colliding_pairs": 1,
        "iteration": 0,
        "sum_of_costs": 4,
        "low_level": {"generated": 20, "runs": 4},
        "context": {
            "split": split,
            "map_id": map_id,
            "task_id": f"{map_id}-task",
            "layout_mode": "regular_beltway",
            "layout_variant": "fixture",
            "scenario_type": "balanced_bidirectional",
            "task_variant": "balanced_80",
            "agent_count": 4,
            "mean_shortest_distance": 1.0,
            "dominant_flow_ratio": 1.0,
            "topology_metrics": {
                "articulation_count": 2,
                "average_free_degree": 1.5,
                "dead_end_cell_count": 2,
                "route_redundancy_proxy": 0.5,
            },
        },
    }


def _candidate(identifier: str, agents: list[int], families: list[str]) -> dict:
    return {
        "candidate_id": identifier,
        "agents": agents,
        "actual_size": len(agents),
        "selection_families": families,
        "proposal_count_by_family": {family: 2 for family in families},
        "proposal_seeds": [10 + index for index in range(len(families))],
        "seed_agents": sorted(set(agents) & {0, 1}),
    }


def _collection(root: Path, split: str = "probe") -> None:
    state_id = "state-a"
    state = _state(split=split)
    candidates = [
        _candidate("candidate-a", [0, 1], ["target:4", "collision:4"]),
        _candidate("candidate-b", [0, 2], ["random:4"]),
    ]
    _write_json(
        root / "run_config.json",
        {"run_fingerprint": "run-a", "configuration": {"evaluation_trials": 8}},
    )
    _write_jsonl(
        root / "candidates.jsonl",
        [
            {
                "state_id": state_id,
                "state_fingerprint": "fingerprint-a",
                "episode_id": "episode-a",
                "split": split,
                "map_id": "map-a",
                "task_id": "map-a-task",
                "layout_mode": "regular_beltway",
                "task_variant": "balanced_80",
                "agent_count": 4,
                "state": state,
                "candidate_count": 2,
                "candidates": candidates,
            }
        ],
    )
    outcomes = []
    for candidate_number, candidate in enumerate(candidates):
        for trial in range(8):
            solved = candidate_number == 0 and trial < 4
            conflicts = 0 if solved else 1 + candidate_number
            outcomes.append(
                {
                    "state_id": state_id,
                    "candidate_id": candidate["candidate_id"],
                    "agents": candidate["agents"],
                    "actual_neighborhood": candidate["agents"],
                    "action_valid": True,
                    "evaluation_trial_index": trial,
                    "evaluation_seed": 1000 + 100 * candidate_number + trial,
                    "evaluation_seed_disjoint": True,
                    "solved": solved,
                    "conflicts_before": 1,
                    "conflicts_after": conflicts,
                    "conflict_auc": (1 + conflicts) / 2,
                    "generated": 100 + candidate_number * 10 + trial,
                    "runtime": 0.01 + candidate_number * 0.001,
                }
            )
    outcomes_path = root / "explicit" / "outcomes.jsonl"
    errors_path = root / "explicit" / "errors.jsonl"
    _write_jsonl(outcomes_path, outcomes)
    _write_jsonl(errors_path, [])
    _write_jsonl(
        root / "collection_manifest.jsonl",
        [
            {
                "state_id": state_id,
                "status": "ok",
                "error_count": 0,
                "outcomes_file": outcomes_path.relative_to(root).as_posix(),
                "errors_file": errors_path.relative_to(root).as_posix(),
            }
        ],
    )


def _index_row(map_id: str, state_id: str, candidate: int, wins: bool) -> dict:
    candidate_id = f"candidate-{candidate}"
    features = {
        "proposal_dynamic": {
            "state.conflicts": float(int(map_id[-1]) + 1),
            "proposal.choice": float(candidate),
        },
        "realized_dynamic": {
            "state.conflicts": float(int(map_id[-1]) + 1),
            "proposal.choice": float(candidate),
            "realized.coverage": float(candidate == int(wins)),
        },
        "realized_context": {
            "state.conflicts": float(int(map_id[-1]) + 1),
            "proposal.choice": float(candidate),
            "realized.coverage": float(candidate == int(wins)),
            f"context.map={map_id}": 1.0,
        },
    }
    conflicts = 0 if candidate == int(wins) else 2
    return {
        "state_id": state_id,
        "candidate_id": candidate_id,
        "candidate_key": candidate_id,
        "map_id": map_id,
        "task_id": f"{state_id}-task",
        "actual_size": 4 + candidate * 4,
        "selection_families": ["target:4" if candidate == 0 else "random:8"],
        "features": features,
        "outcome": {
            "solved_rate": float(conflicts == 0),
            "conflicts_after": float(conflicts),
            "conflict_auc": float(conflicts),
            "generated": 10.0 + candidate,
            "runtime": 0.01,
        },
        "labels": {
            "effectiveness_pareto": conflicts == 0,
            "compute_aware_pareto": conflicts == 0,
            "runtime_sensitive_pareto": conflicts == 0,
        },
    }


class RealizedNeighborhoodRankingAuditTests(unittest.TestCase):
    def test_index_aggregates_eight_trials_and_multiple_provenance_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _collection(root)
            rows, integrity = build_ranking_index(
                root,
                expected_states=1,
                expected_maps=1,
                expected_candidates=2,
                expected_outcomes=16,
            )
        self.assertTrue(integrity["passed"])
        self.assertEqual(len(rows), 2)
        first = rows[0]
        self.assertEqual(first["trial_count"], 8)
        self.assertEqual(first["outcome"]["solved_rate"], 0.5)
        self.assertEqual(first["outcome"]["conflicts_after"], 0.5)
        proposal = first["features"]["proposal_dynamic"]
        self.assertEqual(proposal["proposal.selection_family=target:4"], 1.0)
        self.assertEqual(proposal["proposal.selection_family=collision:4"], 1.0)
        self.assertTrue(first["labels"]["effectiveness_pareto"])
        for profile in first["features"].values():
            self.assertFalse(any(name.startswith(("outcome.", "label.")) for name in profile))

    def test_explicit_features_are_seed_and_member_order_independent(self) -> None:
        state = _state()
        analysis = analyze_state(state)
        forward = explicit_neighborhood_features(state, analysis, [0, 1])
        reverse = explicit_neighborhood_features(state, analysis, [1, 0])
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["realized.internal_conflict_coverage"], 1.0)

    def test_index_rejects_missing_trial_changed_neighborhood_and_non_probe_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _collection(root)
            path = root / "explicit" / "outcomes.jsonl"
            outcomes = [json.loads(line) for line in path.read_text().splitlines()]
            _write_jsonl(path, outcomes[:-1])
            with self.assertRaisesRegex(ValueError, "trials"):
                build_ranking_index(
                    root,
                    expected_states=1,
                    expected_maps=1,
                    expected_candidates=2,
                    expected_outcomes=None,
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _collection(root)
            path = root / "explicit" / "outcomes.jsonl"
            outcomes = [json.loads(line) for line in path.read_text().splitlines()]
            outcomes[0]["actual_neighborhood"] = [0, 3]
            _write_jsonl(path, outcomes)
            with self.assertRaisesRegex(ValueError, "changed"):
                build_ranking_index(
                    root,
                    expected_states=1,
                    expected_maps=1,
                    expected_candidates=2,
                    expected_outcomes=16,
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _collection(root, split="test_ood_layout")
            with self.assertRaisesRegex(ValueError, "Test/OOD"):
                build_ranking_index(
                    root,
                    expected_states=1,
                    expected_maps=1,
                    expected_candidates=2,
                    expected_outcomes=16,
                )

    def test_effectiveness_label_does_not_duplicate_horizon_one_auc(self) -> None:
        left = {"solved_rate": 0.5, "conflicts_after": 2.0, "conflict_auc": 1.0}
        right = {"solved_rate": 0.5, "conflicts_after": 2.0, "conflict_auc": 99.0}
        self.assertFalse(effectiveness_dominates(left, right))
        self.assertFalse(effectiveness_dominates(right, left))

    def test_leave_one_map_out_has_no_map_or_state_leakage(self) -> None:
        rows = []
        for map_number in range(3):
            for candidate in range(2):
                rows.append(
                    _index_row(
                        f"map-{map_number}", f"state-{map_number}", candidate, True
                    )
                )
        folds = leave_one_map_out_folds(rows)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertFalse(set(fold["train_maps"]) & set(fold["validation_maps"]))
            self.assertFalse(fold["train_states"] & fold["validation_states"])

    def test_pairwise_training_and_batched_selection_are_deterministic(self) -> None:
        rows = []
        for map_number in range(3):
            for state_number in range(2):
                state_id = f"state-{map_number}-{state_number}"
                winner = bool((map_number + state_number) % 2)
                for candidate in range(2):
                    rows.append(
                        _index_row(f"map-{map_number}", state_id, candidate, winner)
                    )
        first, _ = train_pairwise_model(rows, "realized_dynamic")
        second, _ = train_pairwise_model(rows, "realized_dynamic")
        grouped = {
            state_id: candidates
            for state_id, candidates in _group_rows(rows).items()
        }
        first_selection = batch_model_selections(grouped, first)
        second_selection = batch_model_selections(grouped, second)
        self.assertEqual(first_selection, second_selection)
        for state_id, candidates in grouped.items():
            self.assertEqual(first_selection[state_id], first.select(candidates))

    def test_context_replacement_keeps_candidates_grouped(self) -> None:
        candidates = [
            _index_row("map-0", "state-0", candidate, True)
            for candidate in range(2)
        ]
        changed = replace_context_bundle(candidates, {"context.map=donor": 1.0})
        self.assertEqual([row["candidate_id"] for row in candidates], [row["candidate_id"] for row in changed])
        for row in changed:
            context = row["features"]["realized_context"]
            self.assertEqual(context["context.map=donor"], 1.0)
            self.assertFalse(any(name == "context.map=map-0" for name in context))

    def test_cached_context_permutation_is_deterministic(self) -> None:
        rows = []
        for map_number in range(3):
            for state_number in range(2):
                state_id = f"state-{map_number}-{state_number}"
                winner = bool((map_number + state_number) % 2)
                for candidate in range(2):
                    rows.append(
                        _index_row(f"map-{map_number}", state_id, candidate, winner)
                    )
        folds = leave_one_map_out_folds(rows)
        records, models, _ = cross_validate(rows, folds)
        first = context_permutation_test(
            rows,
            folds,
            models["realized_context"],
            records["realized_dynamic"],
            records["realized_context"],
            10,
        )
        second = context_permutation_test(
            rows,
            folds,
            models["realized_context"],
            records["realized_dynamic"],
            records["realized_context"],
            10,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["cached_state_context_evaluations"], 36)

    def test_acceptance_rejects_unsupported_size_collapse(self) -> None:
        common = {
            "pareto_top1_hit_rate": 0.5,
            "mean_conflict_regret": 0.5,
            "mean_solved_rate_regret": 0.0,
            "maximum_size_share": 0.5,
        }
        summaries = {
            "uniform_random": dict(common),
            "internal_conflict_coverage": dict(common),
            "proposal_dynamic": dict(common),
            "realized_dynamic": dict(common, pareto_top1_hit_rate=0.6, mean_conflict_regret=0.4, maximum_size_share=0.81),
            "realized_context": dict(common, pareto_top1_hit_rate=0.7, mean_conflict_regret=0.3),
        }
        comparison = {
            "pareto_top1_gain": 0.1,
            "relative_conflict_regret_reduction": 0.2,
            "maps_no_worse": 6,
        }
        comparisons = {
            "realized_dynamic_vs_proposal_dynamic": dict(comparison),
            "realized_context_vs_realized_dynamic": dict(comparison),
        }
        bootstrap = {
            "hit_gain_95_ci": [-0.1, 0.2],
            "conflict_improvement_95_ci": [-0.1, 0.2],
        }
        report = acceptance_report(
            summaries,
            comparisons,
            {
                "realized_dynamic_vs_proposal_dynamic": bootstrap,
                "realized_context_vs_realized_dynamic": bootstrap,
            },
            {"hit_gain_percentile": 0.96, "conflict_reduction_percentile": 0.96},
            {"multiple_sizes_supported": True},
            {
                "minimum_top1_gain": 0.05,
                "minimum_conflict_regret_reduction": 0.05,
                "minimum_maps_no_worse": 4,
                "maximum_size_share": 0.8,
                "minimum_context_top1_gain": 0.05,
                "minimum_context_conflict_regret_reduction": 0.05,
                "minimum_context_permutation_percentile": 0.95,
            },
        )
        self.assertFalse(report["gates"]["realized_ranking"]["passed"])


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in rows:
        result.setdefault(row["state_id"], []).append(row)
    for candidates in result.values():
        candidates.sort(key=lambda row: row["candidate_id"])
    return result


if __name__ == "__main__":
    unittest.main()
