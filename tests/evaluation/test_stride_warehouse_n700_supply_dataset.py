from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.stride_warehouse_n700_supply_dataset import (
    AGENT_COUNT,
    ENDPOINT_SEEDS,
    EXPERIMENT_ID,
    MAP_ARCHIVE_SHA256,
    MAP_SHA256,
    TASK_IDS,
    _guard_output,
    load_config,
    plan,
    prepare_dataset,
    task_specs,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_warehouse_n700_supply_dataset_v1.json"
CLI = ROOT / "scripts" / "run_stride_warehouse_n700_supply_dataset.py"


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    }


def _scenario_endpoints(path: Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    fields = [line.split() for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    starts = [(int(row[5]), int(row[4])) for row in fields]
    goals = [(int(row[7]), int(row[6])) for row in fields]
    return starts, goals


def test_registered_identity_task_ids_and_plan_are_geometry_only() -> None:
    path, _root, config = load_config(CONFIG)
    assert path == CONFIG.resolve()
    specs = task_specs(config)
    assert tuple(row["task_id"] for row in specs) == TASK_IDS
    assert {row["agent_count"] for row in specs} == {700}
    assert {row["endpoint_seed"] for row in specs} == set(ENDPOINT_SEEDS.values())
    result = plan(CONFIG)
    assert result["experiment_id"] == EXPERIMENT_ID
    assert result["task_count"] == 4
    assert result["agent_count"] == AGENT_COUNT
    assert result["map_sha256"] == MAP_SHA256
    assert result["map_archive_sha256"] == MAP_ARCHIVE_SHA256
    assert result["solver_or_controller_invoked"] is False
    assert result["old_r2_or_n800_artifacts_modified"] is False


def test_two_fresh_materializations_are_byte_deterministic_and_q0_valid(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = prepare_dataset(CONFIG, first)
    second_summary = prepare_dataset(CONFIG, second)
    first_dataset = first / "dataset"
    second_dataset = second / "dataset"
    assert first_summary == second_summary
    assert _file_hashes(first_dataset) == _file_hashes(second_dataset)
    assert prepare_dataset(CONFIG, first) == first_summary

    manifest = [
        json.loads(line)
        for line in (first_dataset / "balanced_wall_clock" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert tuple(row["task_id"] for row in manifest) == TASK_IDS
    q0 = json.loads(
        (first_dataset / "q0_geometry_audit.json").read_text(encoding="utf-8")
    )
    assert q0["passed"] is True
    assert q0["solver_or_controller_invoked"] is False
    assert q0["old_r2_or_n800_artifacts_modified"] is False
    assert len(q0["checks"]) == 4
    assert all(row["all_endpoints_reachable"] for row in q0["checks"])
    assert all(row["passed"] for row in q0["checks"])
    for row in manifest:
        scenario = first_dataset / "balanced_wall_clock" / row["scenario_file"]
        starts, goals = _scenario_endpoints(scenario)
        assert len(starts) == len(goals) == 700
        assert len(set(starts)) == len(set(goals)) == 700
        assert all(start != goal for start, goal in zip(starts, goals))


def test_protected_old_r2_output_is_rejected_without_modification(tmp_path: Path) -> None:
    fake_root = tmp_path / "project"
    old_r2 = fake_root / "build" / "wh-f16-v1-r2"
    old_r2.mkdir(parents=True)
    sentinel = old_r2 / "sentinel.bin"
    sentinel.write_bytes(b"old-r2-must-not-change")
    before = _file_hashes(old_r2)
    with pytest.raises(ValueError, match="protected old/source root"):
        _guard_output(fake_root, old_r2)
    assert _file_hashes(old_r2) == before


def test_foreign_partial_dataset_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "foreign" / "dataset"
    dataset.mkdir(parents=True)
    marker = dataset / "foreign.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty without its identity"):
        prepare_dataset(CONFIG, tmp_path / "foreign")
    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_plan_and_prepare_dry_run_do_not_write(tmp_path: Path) -> None:
    plan_result = subprocess.run(
        [sys.executable, str(CLI), "plan", "--config", str(CONFIG)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(plan_result.stdout)["solver_or_controller_invoked"] is False
    output = tmp_path / "dry-run"
    dry_result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "prepare-q0",
            "--config",
            str(CONFIG),
            "--output",
            str(output),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(dry_result.stdout)
    assert payload["dry_run"] is True
    assert payload["output_written"] is False
    assert payload["solver_or_controller_invoked"] is False
    assert not output.exists()
