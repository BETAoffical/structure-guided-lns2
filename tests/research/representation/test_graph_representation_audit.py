from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from research.studies.representation.graph_representation_audit import (
    MODEL_NAMES,
    NODE_FEATURE_NAMES,
    Normalization,
    ProgressLog,
    StateExample,
    _baseline_summary,
    _normalization,
    _validate_config,
    acceptance,
    collate_examples,
    dominance_pairs,
    input_feature_is_forbidden,
    make_model,
    run_graph_representation_audit,
    selection_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def candidate(candidate_id: str, agents: list[int], conflicts: float) -> dict:
    pareto = conflicts == 1.0
    return {
        "state_id": "state",
        "candidate_id": candidate_id,
        "map_id": "map",
        "split": "policy_train",
        "agents": agents,
        "actual_size": len(agents),
        "task_id": "task",
        "selection_families": [f"target:{len(agents)}"],
        "flat_features": [float(len(agents)), float(conflicts)],
        "metadata_features": [float(len(agents))],
        "outcome": {
            "solved_rate": 0.0,
            "conflicts_after": conflicts,
            "conflict_auc": conflicts,
            "generated": 10.0,
            "runtime": 0.1,
        },
        "labels": {
            "effectiveness_pareto": pareto,
            "compute_aware_pareto": pareto,
            "runtime_sensitive_pareto": pareto,
        },
    }


def example(*, edges: list[list[int]] | None = None, permuted: bool = False) -> StateExample:
    node_rows = [
        [float(index + column / 10.0) for column in range(len(NODE_FEATURE_NAMES))]
        for index in range(3)
    ]
    agent_ids = [10, 20, 30]
    graph_edges = [[0, 1], [1, 2]] if edges is None else edges
    if permuted:
        permutation = [2, 0, 1]
        inverse = {old: new for new, old in enumerate(permutation)}
        node_rows = [node_rows[index] for index in permutation]
        agent_ids = [agent_ids[index] for index in permutation]
        graph_edges = [[inverse[left], inverse[right]] for left, right in graph_edges]
    return StateExample(
        state_id="state",
        map_id="map",
        split="policy_train",
        agent_ids=agent_ids,
        node_features=node_rows,
        edges=graph_edges,
        candidates=[
            candidate("a", [10, 20], 1.0),
            candidate("b", [20, 30], 3.0),
        ],
    )


class GraphRepresentationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (PROJECT_ROOT / "research/configs/representation/graph_representation_audit.json").read_text(
                encoding="utf-8"
            )
        )

    def test_registered_models_and_training_parameters_are_frozen(self) -> None:
        _validate_config(self.config)
        self.assertEqual(tuple(self.config["models"]), MODEL_NAMES)
        changed = json.loads(json.dumps(self.config))
        changed["training"]["hidden_size"] = 65
        with self.assertRaisesRegex(ValueError, "parameters"):
            _validate_config(changed)

    def test_feature_leakage_distinguishes_history_from_outcome(self) -> None:
        self.assertFalse(input_feature_is_forbidden("state.low_level_generated_per_agent"))
        self.assertTrue(input_feature_is_forbidden("outcome.generated"))
        self.assertTrue(input_feature_is_forbidden("candidate.conflicts_after"))
        self.assertTrue(input_feature_is_forbidden("context.layout_mode"))

    def test_progress_log_is_valid_incremental_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.jsonl"
            progress = ProgressLog(path, reset=True)
            progress.emit("fold_started", fold=2)
            progress.emit("fold_completed", fold=2)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["event"] for row in rows], ["fold_started", "fold_completed"])
            self.assertTrue(all(row["elapsed_seconds"] >= 0.0 for row in rows))

    def test_mismatched_output_fingerprint_is_rejected_before_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "run_config.json").write_text(
                json.dumps({"run_fingerprint": "different"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "different run fingerprint"):
                run_graph_representation_audit(
                    PROJECT_ROOT,
                    PROJECT_ROOT / "research/configs/representation/graph_representation_audit.json",
                    output,
                    phase="index",
                )

    def test_dominance_pairs_and_hash_tie_break_are_deterministic(self) -> None:
        source = example()
        self.assertEqual(dominance_pairs(source.candidates), [(0, 1)])
        scores = [{"state": [0.5, 0.5]}]
        records = selection_records([source], scores, "test")
        self.assertEqual(records["state"]["candidate_id"], "a")
        self.assertEqual(records["state"]["selected_size"], 2)
        self.assertEqual(records["state"]["conflict_regret"], 0.0)
        baseline = _baseline_summary(records)
        self.assertEqual(baseline["selected_size_counts"], {2: 1})

    def test_conflict_regret_matches_registered_normalized_definition(self) -> None:
        source = example()
        source.candidates[0]["outcome"]["conflicts_after"] = 4.0
        source.candidates[0]["labels"]["effectiveness_pareto"] = False
        source.candidates[1]["outcome"]["conflicts_after"] = 2.0
        source.candidates[1]["labels"]["effectiveness_pareto"] = True
        records = selection_records([source], [{"state": [1.0, 0.0]}], "test")
        self.assertEqual(records["state"]["conflict_regret"], 1.0)

    def test_acceptance_boundary_requires_all_gates(self) -> None:
        baseline = {
            f"s{index}": {
                "map_id": f"m{index}",
                "pareto_hit": False,
                "conflict_regret": 1.0,
                "selected_size": 4,
            }
            for index in range(12)
        }
        challenger = {
            key: {
                **value,
                "pareto_hit": index < 1,
                "conflict_regret": 0.94,
            }
            for index, (key, value) in enumerate(baseline.items())
        }
        baseline_summary = {
            "pareto_top1_hit_rate": 0.0,
            "mean_conflict_regret": 1.0,
            "maximum_size_share": 1.0,
        }
        challenger_summary = {
            "pareto_top1_hit_rate": 1 / 12,
            "mean_conflict_regret": 0.94,
            "maximum_size_share": 0.5,
        }
        local = json.loads(json.dumps(self.config))
        local["bootstrap_samples"] = 100
        result = acceptance(
            baseline_summary,
            challenger_summary,
            baseline,
            challenger,
            local,
            map_count=12,
        )
        self.assertTrue(result["passed"])
        challenger_summary["maximum_size_share"] = 0.81
        self.assertFalse(
            acceptance(
                baseline_summary,
                challenger_summary,
                baseline,
                challenger,
                local,
                map_count=12,
            )["passed"]
        )

    @unittest.skipUnless(HAS_TORCH, "graph tests require the isolated PyTorch environment")
    def test_constant_training_columns_use_unit_scale(self) -> None:
        import torch

        source = example()
        for row in source.candidates:
            row["flat_features"] = [0.0, 3.0]
        normalized = _normalization([source], "flat_features", torch.device("cpu"))
        self.assertTrue(torch.equal(normalized.std, torch.ones(2)))

    @unittest.skipUnless(HAS_TORCH, "graph tests require the isolated PyTorch environment")
    def test_agent_order_is_invariant(self) -> None:
        import torch

        device = torch.device("cpu")
        norms = {
            "node": Normalization(torch.zeros(len(NODE_FEATURE_NAMES)), torch.ones(len(NODE_FEATURE_NAMES))),
            "flat": Normalization(torch.zeros(2), torch.ones(2)),
            "metadata": Normalization(torch.zeros(1), torch.ones(1)),
        }
        torch.manual_seed(4)
        model = make_model(
            "conflict_gnn", 2, 1, len(NODE_FEATURE_NAMES), self.config
        ).eval()
        left = model(collate_examples([example()], norms, device))
        right = model(collate_examples([example(permuted=True)], norms, device))
        self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(HAS_TORCH, "graph tests require the isolated PyTorch environment")
    def test_deepsets_ignores_edges_but_gnn_uses_them(self) -> None:
        import torch

        device = torch.device("cpu")
        norms = {
            "node": Normalization(torch.zeros(len(NODE_FEATURE_NAMES)), torch.ones(len(NODE_FEATURE_NAMES))),
            "flat": Normalization(torch.zeros(2), torch.ones(2)),
            "metadata": Normalization(torch.zeros(1), torch.ones(1)),
        }
        connected = collate_examples([example()], norms, device)
        disconnected = collate_examples([example(edges=[])], norms, device)
        torch.manual_seed(7)
        deepsets = make_model(
            "agent_deepsets", 2, 1, len(NODE_FEATURE_NAMES), self.config
        ).eval()
        self.assertTrue(
            torch.allclose(deepsets(connected), deepsets(disconnected), atol=0, rtol=0)
        )
        torch.manual_seed(7)
        gnn = make_model(
            "conflict_gnn", 2, 1, len(NODE_FEATURE_NAMES), self.config
        ).eval()
        self.assertFalse(
            torch.allclose(gnn(connected), gnn(disconnected), atol=1e-7, rtol=1e-7)
        )


if __name__ == "__main__":
    unittest.main()
