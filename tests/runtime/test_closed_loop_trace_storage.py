from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.closed_loop_confirmation import (
    ClosedLoopTraceError,
    validate_closed_loop_trace,
)
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V1,
    TRACE_FORMAT_DELTA_GZIP_V2,
    TraceStorageError,
    apply_extras_delta,
    apply_state_delta,
    convert_v1_trace,
    encode_extras_delta,
    encode_state_delta,
    read_trace_events,
    write_state_blob,
)
from experiments.repair_collection import (
    _low_level_delta,
    state_fingerprint,
)


def _agent(identifier: int, path: list[int], conflicts: int) -> dict:
    return {
        "id": identifier,
        "start": path[0],
        "goal": path[-1],
        "path_cost": len(path) - 1,
        "shortest_path_cost": len(path) - 1,
        "delay": 0,
        "conflict_degree": conflicts,
        "path": path,
    }


def _state(conflicts: int) -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": conflicts == 0,
        "done": conflicts == 0,
        "iteration": 1 if conflicts == 0 else 0,
        "rows": 2,
        "cols": 3,
        "sum_of_costs": 4,
        "num_of_colliding_pairs": conflicts,
        "runtime": 0.2 if conflicts == 0 else 0.1,
        "context": {"task_id": "task-a"},
        "low_level": {
            "expanded": 8 if conflicts == 0 else 4,
            "generated": 12 if conflicts == 0 else 6,
            "reopened": 0,
            "runs": 4 if conflicts == 0 else 2,
        },
        "obstacles": [0] * 6,
        "conflict_edges": [] if conflicts == 0 else [[0, 1]],
        "agents": [
            _agent(0, [0, 1, 2], conflicts),
            _agent(1, [3, 4, 5], conflicts),
        ],
    }


def _write_v1_trace(path: Path, run_fingerprint: str = "run") -> None:
    before = _state(1)
    after = _state(0)
    summary = {
        "initial_fingerprint": state_fingerprint(before),
        "initial_conflicts": 1,
        "final_conflicts": 0,
        "repairable": True,
        "success": True,
        "truncated": False,
        "external_timeout": False,
        "repair_iterations": 1,
        "conflict_trajectory": [1, 0],
        "conflict_auc": 0.5,
        "fixed_budget_conflict_auc": 0.5,
        "wall_time_to_feasible": 0.2,
        "capped_wall_time_to_feasible": 0.2,
        "native_time_to_feasible": 0.2,
        "controller_totals": {},
        "mean_selected_feature_outside_fraction": 0.0,
        "selected_size_counts": {},
        "selected_family_counts": {},
        "invalid_action_count": 0,
        "fingerprint_mismatch_count": 0,
        "final_sum_of_costs": 4,
        "final_low_level": after["low_level"],
    }
    rows = [
        {
            "schema": EPISODE_SCHEMA_V1,
            "schema_version": 1,
            "run_fingerprint": run_fingerprint,
            "event": "initial",
            "episode_id": "episode-a",
            "policy": "official_adaptive",
            "solver_seed": 1,
            "state_fingerprint": state_fingerprint(before),
            "state": before,
        },
        {
            "schema": EPISODE_SCHEMA_V1,
            "schema_version": 1,
            "run_fingerprint": run_fingerprint,
            "event": "transition",
            "episode_id": "episode-a",
            "decision_index": 0,
            "action": {"mode": "official"},
            "before_fingerprint": state_fingerprint(before),
            "after_fingerprint": state_fingerprint(after),
            "metrics": {"conflicts_before": 1, "conflicts_after": 0},
            "low_level_delta": _low_level_delta(before, after),
            "repair_wall_seconds": 0.1,
            "elapsed_wall_seconds": 0.2,
            "controller": {},
            "terminated": True,
            "truncated": False,
            "after": after,
        },
        {
            "schema": EPISODE_SCHEMA_V1,
            "schema_version": 1,
            "run_fingerprint": run_fingerprint,
            "event": "finish",
            "episode_id": "episode-a",
            "policy": "official_adaptive",
            "success": True,
            "final_fingerprint": state_fingerprint(after),
            "summary": summary,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class ClosedLoopTraceStorageTests(unittest.TestCase):
    def test_tracked_compact_migration_evidence_matches_preserved_reports(self) -> None:
        project = Path(__file__).resolve().parents[2]
        artifact = (
            project
            / "artifacts"
            / "initlns-movingai-ood-compact-migration-v2"
        )
        evidence = json.loads(
            (artifact / "migration_evidence.json").read_text(encoding="utf-8")
        )
        self.assertTrue(evidence["equivalence_passed"])
        self.assertEqual(evidence["matching_episode_count"], 720)
        self.assertEqual(evidence["mismatch_count"], 0)
        self.assertGreater(evidence["storage_reduction_fraction"], 0.99)
        source = project / "build" / "initlns-movingai-ood-collection-v1"
        compact = project / "build" / "initlns-movingai-ood-collection-v2-compact"
        for name, expected in evidence["source_files"].items():
            path = source / name
            self.assertEqual(path.stat().st_size, expected["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected["sha256"])
        for name, expected in evidence["compact_files"].items():
            path = compact / name
            self.assertEqual(path.stat().st_size, expected["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected["sha256"])
        cleanup = json.loads(
            (artifact / "cleanup_manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(cleanup["deletion_authorized"])
        self.assertEqual(
            cleanup["target"],
            "build/initlns-movingai-ood-collection-v1/episodes",
        )

    def test_delta_round_trip_preserves_fingerprint_state(self) -> None:
        before = _state(1)
        after = _state(0)
        after["agents"][0]["path"] = [0, 3, 2]
        after["agents"][0]["nullable_diagnostic"] = None
        delta = encode_state_delta(before, after)
        reconstructed = apply_state_delta(before, delta)
        self.assertEqual(state_fingerprint(reconstructed), state_fingerprint(after))
        self.assertEqual(reconstructed["agents"], after["agents"])

    def test_extras_delta_records_new_null_field(self) -> None:
        before = _state(1)
        after = _state(0)
        after["nullable_diagnostic"] = None
        changed = encode_extras_delta(before, after)["set"]
        self.assertIn("nullable_diagnostic", changed)
        self.assertIsNone(changed["nullable_diagnostic"])
        reconstructed = apply_extras_delta(before, encode_extras_delta(before, after))
        self.assertEqual(
            reconstructed,
            {
                "runtime": 0.2,
                "context": {"task_id": "task-a"},
                "nullable_diagnostic": None,
            },
        )

    def test_extras_delta_rejects_core_field_and_missing_removal(self) -> None:
        before = _state(1)
        with self.assertRaises(TraceStorageError):
            apply_extras_delta(before, {"set": {"iteration": 2}, "remove": []})
        with self.assertRaises(TraceStorageError):
            apply_extras_delta(before, {"set": {}, "remove": ["missing"]})

    def test_invalid_edge_delta_is_rejected(self) -> None:
        delta = encode_state_delta(_state(1), _state(0))
        delta["conflict_edges"]["remove"] = [[0, 9]]
        with self.assertRaises(TraceStorageError):
            apply_state_delta(_state(1), delta)

    def test_state_blob_writes_are_deduplicated_under_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ThreadPoolExecutor(max_workers=8) as pool:
                references = list(pool.map(lambda _: write_state_blob(root, _state(1))[0], range(16)))
            self.assertEqual(len(set(references)), 1)
            self.assertEqual(len(list((root / "state_blobs").glob("*.json.gz"))), 1)

    def test_v1_to_v2_conversion_validates_and_detects_missing_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "compact"
            destination = (
                output
                / "episodes"
                / "movingai_ood"
                / "official_adaptive"
                / "episode-a.jsonl.gz"
            )
            _write_v1_trace(source)
            metadata = convert_v1_trace(source, destination, output)
            validated = validate_closed_loop_trace(
                destination,
                "run",
                expected_episode_id="episode-a",
                expected_policy="official_adaptive",
                expected_solver_seed=1,
                metric_iteration_budget=1,
                collection_root=output,
            )
            self.assertEqual(validated["trace_format"], TRACE_FORMAT_DELTA_GZIP_V2)
            self.assertEqual(validated["summary"]["conflict_trajectory"], [1, 0])
            self.assertEqual(metadata["trace_event_count"], 3)
            events = read_trace_events(destination)
            self.assertNotIn("state", events[0])
            self.assertNotIn("after", events[1])
            (output / str(metadata["initial_state_ref"])).unlink()
            with self.assertRaises(ClosedLoopTraceError):
                validate_closed_loop_trace(destination, "run", collection_root=output)

    def test_corrupt_gzip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl.gz"
            path.write_bytes(b"not-gzip")
            with self.assertRaises(ClosedLoopTraceError):
                validate_closed_loop_trace(path, "run")


if __name__ == "__main__":
    unittest.main()
