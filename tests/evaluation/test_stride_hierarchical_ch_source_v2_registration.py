from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.stride_hierarchical_ch_source_v2_registration import (
    DEVELOPMENT_MAPS,
    TRAIN_MAPS,
    run_preflight,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_hierarchical_ch_source_v2_registration.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_real_registration_validates_static_train_development_only(tmp_path: Path) -> None:
    report = run_preflight(CONFIG, tmp_path)

    assert report["status"] == "REGISTERED"
    assert report["passed"] is True
    assert report["registered_map_count"] == 16
    assert report["expected_task_count"] == 64
    assert report["expected_episode_count"] == 128
    assert report["workers"] == 20
    assert report["source_v1_outcomes_read"] is False
    assert report["sealed_final_semantic_access"] is False
    assert set(report["static_source_observations"]) == set(TRAIN_MAPS + DEVELOPMENT_MAPS)
    assert all(
        min(observation["variant_unique_capacity"].values())
        >= observation["registered_loads"][1]
        for observation in report["static_source_observations"].values()
    )


def test_registration_rejects_load_above_static_endpoint_capacity() -> None:
    config = _config()
    config["per_map_load_grid"]["lgt605d"] = [170, 221]

    validate_config(config)
    from experiments.stride_hierarchical_ch_source_v2_registration import validate_static_sources

    with pytest.raises(ValueError, match="cannot supply registered high load"):
        validate_static_sources(ROOT, config)


def test_registration_rejects_sealed_final_map_id_leak() -> None:
    config = copy.deepcopy(_config())
    config["notes"] = ["combat2"]

    with pytest.raises(ValueError, match="sealed-final map ID leaked"):
        validate_config(config)


def test_reset_gate_is_whole_cohort_and_has_no_backfill() -> None:
    config = _config()
    gate = config["reset_only_qualification"]

    assert gate["atomic_whole_cohort_gate"] is True
    assert gate["expected_reset_count"] == {
        "train": 64,
        "development": 64,
        "total": 128,
    }
    assert gate["minimum_nonzero_states"] == {"train": 33, "development": 33}
    assert gate["minimum_active_maps"] == {"train": 8, "development": 8}
    assert gate["minimum_nonzero_resets_per_map_variant"] == 1
    assert gate["minimum_maps_with_a_reset_at_least_4_conflicts"] == {
        "train": 4,
        "development": 4,
    }
    assert gate["minimum_maps_with_a_reset_at_least_16_conflicts"] == {
        "train": 2,
        "development": 2,
    }
    assert gate["outcome_based_subset_selection_allowed"] is False
    assert gate["reserve_or_replacement_backfill_allowed"] is False
    assert gate["failure_action"] == "STATE_SUPPLY_FAIL_NO_BACKFILL"
