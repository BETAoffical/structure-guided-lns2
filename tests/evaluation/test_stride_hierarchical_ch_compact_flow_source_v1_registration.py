from __future__ import annotations

import collections
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.stride_hierarchical_ch_compact_flow_source_v1_registration import (
    DEVELOPMENT_MAPS,
    LEGACY_H1_MAPS,
    OD_VARIANTS,
    SEALED_FINAL_MAP_IDS,
    SOLVER_SEEDS,
    TRAIN_MAPS,
    run_preflight,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_source_v1_registration.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_real_registration_freezes_complete_map_disjoint_product(
    tmp_path: Path,
) -> None:
    report = run_preflight(CONFIG, tmp_path)
    rows = _read_jsonl(tmp_path / "registered_tasks.jsonl")

    assert report["status"] == "REGISTERED"
    assert report["passed"] is True
    assert report["scientific_status"] == "sequential_design_only"
    assert len(rows) == 192
    assert collections.Counter(row["split"] for row in rows) == {
        "train": 96,
        "development": 96,
    }
    assert report["task_product"] == {
        "od_variants": list(OD_VARIANTS),
        "solver_seeds": list(SOLVER_SEEDS),
        "loads_per_map": 3,
        "task_seeds_per_map": 2,
        "tasks_per_map": 12,
        "tasks_per_split": 96,
        "registered_task_count": 192,
        "episodes_per_split": 192,
        "expected_episode_count": 384,
        "workers": 16,
    }
    assert report["expected_task_count"] == 192
    assert report["expected_episode_count"] == 384
    assert report["solver_seeds"] == [41, 42]
    assert report["workers"] == 16
    assert report["required_capacity_counts"] == {
        "train": {"ultra": 2, "small": 2, "medium": 2, "large": 2},
        "development": {"ultra": 3, "small": 2, "medium": 2, "large": 1},
    }
    assert report["completed_v2_reset_rows_imported"] == 0
    assert report["sealed_final_semantic_access"] is False
    assert report["training_authorized"] is False


def test_each_map_has_exact_three_load_two_variant_two_task_seed_product(
    tmp_path: Path,
) -> None:
    run_preflight(CONFIG, tmp_path)
    rows = _read_jsonl(tmp_path / "registered_tasks.jsonl")
    config = _config()

    for map_id, registration in config["map_registration"].items():
        map_rows = [row for row in rows if row["map_id"] == map_id]
        assert len(map_rows) == 12
        assert {
            (row["agent_count"], row["od_variant"], row["task_seed"])
            for row in map_rows
        } == {
            (load, variant, seed)
            for load in registration["loads"]
            for variant in OD_VARIANTS
            for seed in registration["task_seeds"]
        }
        assert {row["load_tier"] for row in map_rows} == {"low", "middle", "high"}
        assert all(
            set(row[key]) == {"path", "sha256"}
            for row in map_rows
            for key in ("map", "map_metadata", "scenario", "task")
        )


def test_registered_manifest_payload_and_every_file_pin_are_stable(
    tmp_path: Path,
) -> None:
    report = run_preflight(CONFIG, tmp_path)
    manifest = tmp_path / "registered_tasks.jsonl"
    rows = _read_jsonl(manifest)

    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        "ae4d2236d574eac650aa61fb04b5271620861580d881ad0707ca92e89462f78c"
    )
    assert report["registered_task_manifest"] == {
        "path": manifest.resolve().as_posix(),
        "schema": "lns2.stride.hierarchical_ch_compact_flow_registered_task.v1",
        "sha256": "ae4d2236d574eac650aa61fb04b5271620861580d881ad0707ca92e89462f78c",
        "row_count": 192,
        "unique_map_file_count": 16,
        "unique_map_metadata_file_count": 16,
        "unique_scenario_file_count": 192,
        "unique_task_file_count": 192,
    }
    for row in rows:
        for key in ("map", "map_metadata", "scenario", "task"):
            file_path = ROOT / row[key]["path"]
            assert hashlib.sha256(file_path.read_bytes()).hexdigest() == row[key]["sha256"]


def test_historical_qualification_is_complete_and_not_imported(
    tmp_path: Path,
) -> None:
    report = run_preflight(CONFIG, tmp_path)

    assert set(report["historical_qualification"]) == set(TRAIN_MAPS + DEVELOPMENT_MAPS)
    assert all(
        evidence["valid_reset_count"] == 36
        and evidence["error_count"] == 0
        and evidence["timeout_count"] == 0
        and all(load["count"] == 12 for load in evidence["by_load"].values())
        for evidence in report["historical_qualification"].values()
    )
    assert report["historical_qualification_used_for_design"] is True
    assert report["historical_rows_imported_as_new_source_rows"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.__setitem__("workers", 20), "task-product alias"),
        (
            lambda config: config["map_registration"]["den404d"].__setitem__(
                "loads", [80, 180, 281]
            ),
            "per-map source/load/seed/capacity",
        ),
        (
            lambda config: config["parent_source_v2_terminal_failure"].__setitem__(
                "completed_v2_reset_rows_imported", 1
            ),
            "source-v2 terminal failure/import boundary",
        ),
        (
            lambda config: config["claim_boundary"].__setitem__(
                "training_authorized", True
            ),
            "claim boundary",
        ),
    ],
)
def test_registration_rejects_product_or_boundary_changes(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_selected_maps_are_disjoint_from_legacy_h1_and_sealed_final() -> None:
    train = set(TRAIN_MAPS)
    development = set(DEVELOPMENT_MAPS)
    selected = train | development

    assert len(train) == len(development) == 8
    assert not train & development
    assert not selected & set(LEGACY_H1_MAPS)
    assert not selected & set(SEALED_FINAL_MAP_IDS)
