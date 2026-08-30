from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from experiments._common import sha256_file
from experiments.stride_hierarchical_ch_h1_v2 import (
    CONFIG_SCHEMA,
    DEVELOPMENT_MAPS,
    EXPERIMENT_ID,
    LEGACY_CURRENT_H1_MAPS,
    REPORT_SCHEMA,
    SEALED_FINAL_MAPS,
    SOURCE_LOAD_GRID,
    TRAIN_FOLDS,
    TRAIN_MAPS,
    guard_sealed_final_semantic_access,
    run_preflight,
    validate_config,
    validate_map_disjointness,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "stride_hierarchical_ch_h1_v2.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _copy_registered_fixture(tmp_path: Path) -> tuple[Path, Path]:
    config = _config()
    fixture_root = tmp_path / "fixture"
    fixture_config = fixture_root / "configs" / CONFIG_PATH.name
    fixture_config.parent.mkdir(parents=True)
    shutil.copy2(CONFIG_PATH, fixture_config)
    for registration in config["registered_sources"].values():
        for specification in registration.values():
            source = ROOT / specification["path"]
            target = fixture_root / specification["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return fixture_config, fixture_root


def test_registered_identity_sources_and_outcome_blind_source_design_are_exact(
    tmp_path: Path,
) -> None:
    config = _config()
    validate_config(config)

    assert config["schema"] == CONFIG_SCHEMA
    assert config["experiment_id"] == EXPERIMENT_ID
    splits = validate_map_disjointness(config)
    assert splits == {
        "train": TRAIN_MAPS,
        "development": DEVELOPMENT_MAPS,
        "sealed_final": SEALED_FINAL_MAPS,
    }
    assert not set(LEGACY_CURRENT_H1_MAPS) & set().union(*map(set, splits.values()))
    assert config["legacy_current_h1"]["usage"] == "legacy_sequential_dev/train-only"
    assert config["train_folds"] == {
        key: list(value) for key, value in TRAIN_FOLDS.items()
    }

    source = config["new_source_design"]
    assert source["status"] == "outcome_blind_registration_only_not_executable_yet"
    assert source["designed_splits"] == ["train", "development"]
    assert source["sealed_final_task_or_load_design_present"] is False
    assert source["task_variants"] == ["movingai_scenario_prefix"]
    assert source["source_seeds"] == [41, 42]
    assert source["max_decisions"] == 12
    assert source["per_map_load_grid"] == {
        map_id: list(loads) for map_id, loads in SOURCE_LOAD_GRID.items()
    }
    assert set(source["per_map_load_grid"]) == set(TRAIN_MAPS) | set(DEVELOPMENT_MAPS)
    assert not set(source["per_map_load_grid"]) & set(SEALED_FINAL_MAPS)
    assert source["expected_source_episode_count_per_map"] == 4
    assert source["expected_source_episode_count"] == 64
    selection = source["selection"]
    assert selection["depth_bands"] == {
        "d0": [0, 0],
        "d1_3": [1, 3],
        "d4_plus": [4, 11],
    }
    assert selection["minimum_states_per_depth_band_per_map"] == 4
    assert selection["maximum_states_per_episode"] == 4
    assert selection["maximum_states_per_map_load"] == 8
    assert selection["maximum_states_per_map"] == 16
    assert selection["target_states_per_map"] == 16
    assert selection["outcome_based_backfill_allowed"] is False
    assert selection["reserve_or_replacement_backfill_allowed"] is False

    output = tmp_path / "registration"
    first = run_preflight(CONFIG_PATH, output)
    first_bytes = (output / "registration_report.json").read_bytes()
    second = run_preflight(CONFIG_PATH, output)
    assert second == first
    assert (output / "registration_report.json").read_bytes() == first_bytes
    assert first["schema"] == REPORT_SCHEMA
    assert first["status"] == "REGISTERED"
    assert first["passed"] is True
    assert first["config_sha256"] == sha256_file(CONFIG_PATH)
    assert first["registered_map_count"] == 24
    assert first["registered_source_file_count"] == 48
    assert first["source_sha256_validation_passed"] is True
    assert first["map_splits_pairwise_disjoint"] is True
    assert first["sealed_final"]["status"] == "SEALED_HASH_VERIFIED_ONLY"
    assert first["sealed_final"]["semantic_map_or_scenario_content_read"] is False
    assert first["sealed_final"]["h1_outcome_read"] is False
    assert first["h1_outcomes_read"] is False
    assert first["outcome_fields_read"] is False
    assert first["source_episodes_generated"] is False
    assert first["h1_executed"] is False
    assert first["training_authorized"] is False


def test_checksum_tamper_fails_closed_using_copied_fixture(tmp_path: Path) -> None:
    fixture_config, fixture_root = _copy_registered_fixture(tmp_path)
    tampered = fixture_root / _config()["registered_sources"]["brc000d"]["map"]["path"]
    tampered.write_bytes(tampered.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="registered brc000d map input changed"):
        run_preflight(fixture_config, fixture_root / "registration")
    assert not (fixture_root / "registration").exists()


def test_overlap_legacy_and_source_design_leaks_fail_closed() -> None:
    overlap = copy.deepcopy(_config())
    overlap["map_splits"]["development"][0] = TRAIN_MAPS[0]
    with pytest.raises(ValueError, match="map splits overlap"):
        validate_map_disjointness(overlap)

    legacy = copy.deepcopy(_config())
    legacy["map_splits"]["train"][0] = LEGACY_CURRENT_H1_MAPS[0]
    with pytest.raises(ValueError, match="legacy sequential H1 map leaked"):
        validate_map_disjointness(legacy)

    final_task_leak = copy.deepcopy(_config())
    final_task_leak["new_source_design"]["per_map_load_grid"][
        SEALED_FINAL_MAPS[0]
    ] = [100, 200]
    final_task_leak["new_source_design"][
        "sealed_final_task_or_load_design_present"
    ] = True
    with pytest.raises(ValueError, match="source design changed"):
        validate_config(final_task_leak)


@pytest.mark.parametrize(
    ("model_frozen", "threshold_frozen", "registration_passed", "message"),
    (
        (False, False, False, "model freeze"),
        (True, False, False, "threshold freeze"),
        (True, True, False, "passed registration report"),
    ),
)
def test_sealed_final_semantic_access_requires_all_freezes(
    model_frozen: bool,
    threshold_frozen: bool,
    registration_passed: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        guard_sealed_final_semantic_access(
            model_frozen=model_frozen,
            threshold_frozen=threshold_frozen,
            registration_report_passed=registration_passed,
        )

    guard_sealed_final_semantic_access(
        model_frozen=True,
        threshold_frozen=True,
        registration_report_passed=True,
    )


def test_preflight_refuses_output_containing_h1_or_other_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "registration"
    output.mkdir()
    (output / "h1_state_labels.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-registration artifact"):
        run_preflight(CONFIG_PATH, output)
    assert not (output / "registration_report.json").exists()


def test_label_and_readiness_contract_tampering_fails_closed() -> None:
    paired = copy.deepcopy(_config())
    paired["exact_action_and_label_contract"]["paired_seed_indices"].pop()
    with pytest.raises(ValueError, match="exact-set/16-paired-seed"):
        validate_config(paired)

    stage1 = copy.deepcopy(_config())
    stage1["readiness_gates"]["stage1"][
        "minimum_examples_per_decisive_class"
    ] = 39
    with pytest.raises(ValueError, match="readiness gates"):
        validate_config(stage1)

    stage2 = copy.deepcopy(_config())
    stage2["readiness_gates"]["stage2_conditional_on_stage1_positive"][
        "minimum_maps_per_decisive_class"
    ] = 3
    with pytest.raises(ValueError, match="readiness gates"):
        validate_config(stage2)
