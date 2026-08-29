from __future__ import annotations

from pathlib import Path

import pytest

import experiments.closed_loop_confirmation as closed_loop
from experiments._common import sha256_file
from experiments.closed_loop_trace_storage import write_state_blob
from experiments.repair_collection import state_fingerprint
from experiments.trace_replay import target_state_from_checkpoint_blob
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


def _agent(identifier: int, path: list[int]) -> dict[str, object]:
    return {
        "id": identifier,
        "start": path[0],
        "goal": path[-1],
        "path_cost": len(path) - 1,
        "shortest_path_cost": len(path) - 1,
        "delay": 0,
        "conflict_degree": 1,
        "path": path,
    }


def _state() -> dict[str, object]:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": False,
        "done": False,
        "iteration": 0,
        "rows": 2,
        "cols": 3,
        "sum_of_costs": 4,
        "num_of_colliding_pairs": 1,
        "low_level": {
            "expanded": 4,
            "generated": 6,
            "reopened": 0,
            "runs": 2,
        },
        "obstacles": [0] * 6,
        "conflict_edges": [[0, 1]],
        "agents": [
            _agent(0, [0, 1, 2]),
            _agent(1, [3, 4, 5]),
        ],
    }


def _checkpoint(root: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    state = _state()
    relative, state_path = write_state_blob(root, state)
    checkpoint = {
        "source_kind": "checkpoint_blob_v1",
        "checkpoint_id": "warehouse-delay-0001",
        "checkpoint_identity_sha256": "1" * 64,
        "collection_root": str(root),
        "state_blob": relative,
        "state_blob_sha256": sha256_file(state_path),
        "expected_fingerprint": state_fingerprint(state),
        "repair_structure_fingerprint": repair_structure_fingerprint(state),
        "expected_conflicts": 1,
        "map_id": "warehouse-a",
        "task_id": "task-a",
        "agent_count": 2,
        "restore_seed": 17,
    }
    return checkpoint, state_path, state


def _load(root: Path, checkpoint: dict[str, object]):
    return target_state_from_checkpoint_blob(
        root,
        checkpoint,
        expected_map_id="warehouse-a",
        expected_task_id="task-a",
        expected_agent_count=2,
    )


def test_checkpoint_blob_restore_accepts_authenticated_contained_state(
    tmp_path: Path,
) -> None:
    checkpoint, state_path, state = _checkpoint(tmp_path)

    loaded, resolved = _load(tmp_path, checkpoint)

    assert resolved == state_path.resolve()
    assert loaded == state


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("state_blob_sha256", "0" * 64, "blob SHA-256"),
        ("expected_fingerprint", "0" * 64, "state fingerprint"),
        ("repair_structure_fingerprint", "0" * 64, "repair fingerprint"),
        ("expected_conflicts", 2, "conflict count"),
        ("checkpoint_id", "", "checkpoint_id"),
        ("checkpoint_identity_sha256", "not-a-sha", "identity_sha256"),
    ],
)
def test_checkpoint_blob_restore_rejects_tampered_identity_or_state_metadata(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    checkpoint, _state_path, _state_value = _checkpoint(tmp_path)
    checkpoint[field] = replacement

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, checkpoint)


def test_checkpoint_blob_restore_rejects_tampered_blob(tmp_path: Path) -> None:
    checkpoint, state_path, _state_value = _checkpoint(tmp_path)
    state_path.write_bytes(state_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="blob SHA-256"):
        _load(tmp_path, checkpoint)


def test_checkpoint_blob_restore_rejects_path_escape(tmp_path: Path) -> None:
    checkpoint, _state_path, _state_value = _checkpoint(tmp_path)
    checkpoint["state_blob"] = "../outside.json.gz"

    with pytest.raises(ValueError, match="contained relative path"):
        _load(tmp_path, checkpoint)


@pytest.mark.parametrize(
    ("current", "message"),
    [
        ({"expected_map_id": "warehouse-b"}, "map_id"),
        ({"expected_task_id": "task-b"}, "task_id"),
        ({"expected_agent_count": 3}, "agent_count"),
    ],
)
def test_checkpoint_blob_restore_rejects_cross_job_reuse(
    tmp_path: Path,
    current: dict[str, object],
    message: str,
) -> None:
    checkpoint, _state_path, _state_value = _checkpoint(tmp_path)
    expected = {
        "expected_map_id": "warehouse-a",
        "expected_task_id": "task-a",
        "expected_agent_count": 2,
        **current,
    }

    with pytest.raises(ValueError, match=message):
        target_state_from_checkpoint_blob(tmp_path, checkpoint, **expected)


def test_initial_restore_without_source_kind_keeps_legacy_trace_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state()
    trace_path = tmp_path / "episode.jsonl.gz"
    calls: list[tuple[Path, dict[str, object], int, str]] = []

    def fake_target_state_from_trace(
        root: Path,
        manifest: dict[str, object],
        *,
        decision_index: int,
        expected_fingerprint: str,
    ) -> tuple[dict[str, object], Path]:
        calls.append((root, manifest, decision_index, expected_fingerprint))
        return state, trace_path

    monkeypatch.setattr(
        closed_loop, "target_state_from_trace", fake_target_state_from_trace
    )
    initial_restore = {
        "collection_root": str(tmp_path),
        "manifest": {"trace_file": "episode.jsonl.gz"},
        "decision_index": 3,
        "expected_fingerprint": state_fingerprint(state),
        "repair_structure_fingerprint": repair_structure_fingerprint(state),
        "expected_conflicts": 1,
        "restore_seed": 17,
    }

    loaded, source_trace, source_checkpoint, source_kind = (
        closed_loop._load_initial_restore_source(
            initial_restore,
            {"map_id": "warehouse-a", "task_id": "task-a", "agent_count": 2},
        )
    )

    assert loaded == state
    assert source_trace == trace_path
    assert source_checkpoint is None
    assert source_kind == "trace"
    assert calls == [
        (
            tmp_path,
            {"trace_file": "episode.jsonl.gz"},
            3,
            state_fingerprint(state),
        )
    ]
