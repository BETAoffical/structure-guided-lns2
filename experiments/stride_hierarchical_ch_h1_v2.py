"""Outcome-blind registration for the map-disjoint hierarchical C/H study.

This module intentionally has no source-episode, H1, label, fitting, or final
evaluation implementation.  Its only mutating action writes a deterministic
registration report after validating the checksum-pinned MovingAI/DAO inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from experiments._common import registered_input, sha256_file


CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_h1_config.v2"
REPORT_SCHEMA = "lns2.stride.hierarchical_ch_h1_registration_report.v2"
EXPERIMENT_ID = "stride_hierarchical_ch_h1_v2"
DEFAULT_OUTPUT_NAME = "stride-hierarchical-ch-h1-v2"

TRAIN_MAPS = (
    "brc000d",
    "den000d",
    "hrt000d",
    "lak100d",
    "lgt605d",
    "orz999d",
    "ost000t",
    "rmtst03",
)
DEVELOPMENT_MAPS = (
    "brc203d",
    "den400d",
    "lak201d",
    "lgt602d",
    "orz302d",
    "ost004d",
    "brc997d",
    "den504d",
)
SEALED_FINAL_MAPS = (
    "brc503d",
    "combat2",
    "den501d",
    "hrt201n",
    "lak200d",
    "lgt601d",
    "orz500d",
    "ost002d",
)
LEGACY_CURRENT_H1_MAPS = (
    "den520d",
    "maze-128-128-10",
    "warehouse-10-20-10-2-1",
    "maze-128-128-1",
    "lak303d",
    "random-32-32-10",
    "room-32-32-4",
    "random-32-32-20",
    "maze-32-32-2",
    "random-64-64-10",
    "warehouse-20-40-10-2-1",
    "maze-32-32-4",
    "random-64-64-20",
    "room-64-64-8",
    "warehouse-20-40-10-2-2",
    "room-64-64-16",
)
TRAIN_FOLDS = {
    "fold0": ("brc000d", "lak100d"),
    "fold1": ("den000d", "lgt605d"),
    "fold2": ("hrt000d", "orz999d"),
    "fold3": ("ost000t", "rmtst03"),
}

ALL_NEW_MAPS = (*TRAIN_MAPS, *DEVELOPMENT_MAPS, *SEALED_FINAL_MAPS)
MAP_GROUPS = {
    map_id: (
        "combat" if map_id == "combat2" else "rmt" if map_id == "rmtst03" else map_id[:3]
    )
    for map_id in ALL_NEW_MAPS
}

SOURCE_SHA256 = {
    "brc000d": ("c5bcb3632d83e0dffe0ebcae7e8bccc57c667006f9a2d47103f5e8de528d0245", "d9bb9a1f16387c37e651a8cd33df944cba0146bb6a3bae70678bdb3db0a14f2e"),
    "den000d": ("42f390dfd8f424f3a80418c431b65448d47bc424a16e42117daf75c776e0f96b", "e4d0ad76ff92896e866b0379c24e06901a74a4dee789046fae31e8e52ecbfecb"),
    "hrt000d": ("25524f05bed5f1aa7aa9570e1a893ec611bb427867bd70b99e3efdcb38db04a0", "35780ce033f22194eca7b0b0833214602a776c54f44a17cd54cfca2c76caada6"),
    "lak100d": ("5250d01fb736f69a61724a0e424877b9c8049afd5185817206571e583a24948c", "f5eba35756568c94ce11d72e02badb9fc7f93b9cfc13562498442992e7a00e5d"),
    "lgt605d": ("3516885bcbee711653143d051404655af7d50e26dc339d82964638dd68650aa5", "8f41161fa12f286dede3c443718a3c60703c7799837696d220eff67574f1deeb"),
    "orz999d": ("a521f238741f626eeccef531d155f18a4abfd8460cbcf1e7cec1df9e9474f25a", "c414ace91f8fcc446d2f4bd7ebba2babb2a271e2d43b72874d3c13c576aadd33"),
    "ost000t": ("f968d44bcf51e6c3693c0e831c62095bbf0280779ae899265811d36b960c6c4d", "6775f4f6426733d216b64b95db7f0e8a07825503d25ae1352c1d6cffc0ca4142"),
    "rmtst03": ("c3bc8b745f5df32126c00f6e0a7614c9d8abb1f7f75ed89a4708676992b4f4ce", "5dba08539c209f476d65d647b3d22e0d64276e7985bfb8e631d6ccb845df2ad2"),
    "brc203d": ("1695b540c444f8d7e3693c48bb4481dfe0333046dd8754848d6c82850941779d", "53fcb343f843a84bed7ec26b72e46d93cd90ba1cdac2dba1b4bf846bf3ce8f5a"),
    "den400d": ("12a22ae77bdb13329d1921a72b09f0b8e5d5e9cc10ea5cd5034f15025af29184", "4cf1de8a15aaaa730099231b3b5850299a8a155e6058b20f245efda1555e4a95"),
    "lak201d": ("7b51b615a3ae35f8011cd5c47968a6d38700b74282e9467a2520451e99709886", "77bc4bd975d907fad683085aaf9b2a36e8da862f7812fdebe1a0e7975b728499"),
    "lgt602d": ("91a203b7e04ea8abd7190f53f2e3c33e91c91366b32df2a0e0c932502b07879e", "02f4466f520c67384dc3fc70592b7cb791ea94be25214d1736986f29b2a3ca81"),
    "orz302d": ("db87e99edff257a91aa3ee56469423eb0154a18b7b8155511271ba510a44c55f", "a0d5920d899f433a972e539d1dda63c53a1880c0615f96d25188760d33d169d4"),
    "ost004d": ("43008462b1e960053bfbe46fa01dbb28930f1bf9e3abae28665486985e020b31", "a32486b90e5b10992892497f46c4e3830bd65539da3e733e22c153ae78e8bed0"),
    "brc997d": ("d9b5ace8e4008c5d4442f21c0c6cb8767cf9fab90dcaf5e3d20a3d4e80ad26a9", "c40fe24febe55341d3999a273136b82904c078554c366cddf3f24fd3fbdc6454"),
    "den504d": ("4c76f876dfed2427be71c53a9322a7b2fa2fe49cd797622200c3f570fabf48c1", "a04ae58978bdb346cc9f483d537d1216e6ba4ad0f3259126b7f5829e4169d91c"),
    "brc503d": ("baf7c49918d0e4a0f79cade72991260a8371c508cbf29a33aca5010f2de63b9c", "f80e3e17ad7a24f9f14214b46d87a585fecc16c4ee2bdaa5f444d560427661d9"),
    "combat2": ("e6397f6a776e33b0dc03054a6bc694a3ce910bfeab20fdcb65283065633b5206", "a58152a3cc978262639ca610bd71963117438a16bc8d34055c2ea513478d7a57"),
    "den501d": ("1bfafa35226f03e65d5f66e816968539c99fb2ffdfe02c3da8e3ef36b246c05d", "20b8ce8c792da5f4fae3280e3addbf9b52ac263f3e751c1b65bcb0f569cb137f"),
    "hrt201n": ("93db40955080660737b8e406768924927e5d284b53506b016bf567a1127970f9", "b787d541c08598d34b4ecc746429838d794340b39d75715a7c14cf811deb733f"),
    "lak200d": ("e0460e485555e9699f329127def7db900c59f67b1421b7d0257e411e5e34bce9", "c5ff3a152f2d3c5aa7730dd8e19ecddcf36e5d127002813a7a0c58acefcdc4cb"),
    "lgt601d": ("07037697e90964f3035c5dce9fe929e783bdd0ac57e13e271dd0fba6d9dfbabe", "bbfec4349471f7e9d8498883dde675600b770de6ab55e845680d7b57c4285625"),
    "orz500d": ("b04d52995cb6af81e149c0ee88b5e609d7cf99edf78db131066189851fee823e", "343bc3dde61f3e1c31b928bfa9d8273832432c828ae5d663cb80affa56f65d1e"),
    "ost002d": ("170afbb33b12a1e3ce734e985a15bc1a20fed4319505541713f908b0a2041d94", "94dd8ec9292b5090b1a20b847bcf7cba26db2f23f43a52fe509a35caf14b44dc"),
}

SOURCE_LOAD_GRID = {
    "brc000d": (300, 500),
    "den000d": (400, 600),
    "hrt000d": (400, 600),
    "lak100d": (400, 600),
    "lgt605d": (100, 200),
    "orz999d": (400, 600),
    "ost000t": (400, 600),
    "rmtst03": (200, 400),
    "brc203d": (400, 600),
    "den400d": (300, 500),
    "lak201d": (300, 500),
    "lgt602d": (400, 600),
    "orz302d": (200, 400),
    "ost004d": (100, 200),
    "brc997d": (300, 500),
    "den504d": (300, 500),
}


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON value is forbidden: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("hierarchical C/H config must be a JSON object")
    return value


def _expected_sources() -> dict[str, dict[str, dict[str, str]]]:
    root = "build/movingai-dao-compact-source-v1"
    return {
        map_id: {
            "map": {
                "path": f"{root}/maps-all/{map_id}.map",
                "sha256": SOURCE_SHA256[map_id][0],
            },
            "scenario": {
                "path": f"{root}/scenarios-all/{map_id}.map.scen",
                "sha256": SOURCE_SHA256[map_id][1],
            },
        }
        for map_id in ALL_NEW_MAPS
    }


def _expected_new_source_design() -> dict[str, Any]:
    return {
        "status": "outcome_blind_registration_only_not_executable_yet",
        "designed_splits": ["train", "development"],
        "sealed_final_task_or_load_design_present": False,
        "source_policy": "v2-full",
        "task_semantics": "movingai_scenario_prefix",
        "task_variants": ["movingai_scenario_prefix"],
        "source_seeds": [41, 42],
        "source_seed_role": "solver_seed",
        "max_decisions": 12,
        "per_map_load_grid": {
            map_id: list(loads) for map_id, loads in SOURCE_LOAD_GRID.items()
        },
        "expected_source_episode_count_per_map": 4,
        "expected_source_episode_count": 64,
        "selection": {
            "unit": "preaction_state_before_any_h1_outcome",
            "depth_bands": {
                "d0": [0, 0],
                "d1_3": [1, 3],
                "d4_plus": [4, 11],
            },
            "minimum_states_per_depth_band_per_map": 4,
            "maximum_states_per_episode": 4,
            "maximum_states_per_map_load": 8,
            "maximum_states_per_map": 16,
            "target_states_per_map": 16,
            "maximum_source_episodes_per_map": 4,
            "rank_rule": "depth_quota_then_preaction_state_sha256_ascending",
            "allowed_selection_inputs": [
                "map_id",
                "agent_count",
                "source_seed",
                "decision_index",
                "preaction_state_sha256",
                "exact_agent_sets",
                "role_aliases",
            ],
            "forbidden_selection_inputs": [
                "repair_outcome",
                "h1_label",
                "h1_score",
                "runtime_seconds",
                "pp_seconds",
            ],
            "outcome_based_backfill_allowed": False,
            "reserve_or_replacement_backfill_allowed": False,
            "under_supply_action": "STATE_SUPPLY_FAIL_NO_BACKFILL",
        },
    }


def validate_map_disjointness(config: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = config.get("map_splits")
    if not isinstance(raw, dict) or set(raw) != {"train", "development", "sealed_final"}:
        raise ValueError("exact train/development/sealed-final map splits are required")
    splits: dict[str, tuple[str, ...]] = {}
    for name in ("train", "development", "sealed_final"):
        values = raw.get(name)
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ValueError(f"{name} map split must be a list of map IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} map split contains a duplicate")
        splits[name] = tuple(values)
    for left, right in (("train", "development"), ("train", "sealed_final"), ("development", "sealed_final")):
        overlap = set(splits[left]) & set(splits[right])
        if overlap:
            raise ValueError(f"map splits overlap: {left}/{right}: {sorted(overlap)}")
    legacy = set(LEGACY_CURRENT_H1_MAPS)
    leaked = legacy & set().union(*map(set, splits.values()))
    if leaked:
        raise ValueError(f"legacy sequential H1 map leaked into new splits: {sorted(leaked)}")
    return splits


def _validate_exact_contracts(config: dict[str, Any]) -> None:
    action = config.get("exact_action_and_label_contract")
    expected_action = {
        "roles": ["v2_anchor", "component16", "hotspot16"],
        "canonical_agent_set": "strictly_increasing_sorted_unique_agent_ids",
        "deduplicate_within_state_by_exact_agent_set": True,
        "execute_each_unique_agent_set_once": True,
        "preserve_all_role_aliases": True,
        "paired_seed_indices": list(range(16)),
        "same_state_seed_paired_across_unique_actions": True,
        "stage1": {
            "unit": "unique_structural_action_vs_unique_v2_anchor_action",
            "labels": ["structural_win", "v2_win", "ambiguous"],
            "ambiguous_is_not_negative": True,
            "requires_bidirectional_robust_comparison": True,
        },
        "stage2": {
            "unit": "unique_component_action_vs_unique_hotspot_action",
            "condition": "at_least_one_structural_action_has_stage1_structural_win",
            "labels": ["component_win", "hotspot_win", "ambiguous"],
            "ambiguous_is_not_negative": True,
            "requires_bidirectional_robust_comparison": True,
        },
        "runtime_or_pp_seconds_used_in_label": False,
    }
    if action != expected_action:
        raise ValueError("exact-set/16-paired-seed label contract changed")
    expected_gates = {
        "stage1": {
            "minimum_examples_per_decisive_class": 40,
            "minimum_maps_per_decisive_class": 8,
            "minimum_groups_per_decisive_class": 4,
            "each_train_fold_requires_both_decisive_classes": True,
        },
        "stage2_conditional_on_stage1_positive": {
            "minimum_examples_per_decisive_class": 16,
            "minimum_maps_per_decisive_class": 4,
            "minimum_groups_per_decisive_class": 3,
            "each_train_fold_requires_both_decisive_classes": True,
        },
        "all_integrity_gates_required": True,
        "failure_action": "NO_TRAIN",
    }
    if config.get("readiness_gates") != expected_gates:
        raise ValueError("hierarchical C/H readiness gates changed")


def validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "outcome_blind_map_split_registration_before_new_h1"
        or config.get("source_root") != "build/movingai-dao-compact-source-v1"
    ):
        raise ValueError("hierarchical C/H H1 v2 identity changed")
    splits = validate_map_disjointness(config)
    expected_splits = {
        "train": TRAIN_MAPS,
        "development": DEVELOPMENT_MAPS,
        "sealed_final": SEALED_FINAL_MAPS,
    }
    if splits != expected_splits:
        raise ValueError("registered map split membership or order changed")
    legacy = config.get("legacy_current_h1")
    if legacy != {
        "usage": "legacy_sequential_dev/train-only",
        "map_ids": list(LEGACY_CURRENT_H1_MAPS),
        "allowed_for_new_development_split": False,
        "allowed_for_new_sealed_final_split": False,
    }:
        raise ValueError("legacy sequential H1 boundary changed")
    folds = config.get("train_folds")
    if folds != {key: list(value) for key, value in TRAIN_FOLDS.items()}:
        raise ValueError("outcome-blind train folds changed")
    flattened_folds = [map_id for fold in folds.values() for map_id in fold]
    if len(flattened_folds) != len(set(flattened_folds)) or set(flattened_folds) != set(TRAIN_MAPS):
        raise ValueError("train folds must partition the train maps exactly once")
    if config.get("map_groups") != MAP_GROUPS:
        raise ValueError("map group registration changed")
    if config.get("registered_sources") != _expected_sources():
        raise ValueError("registered map/scenario source hashes changed")
    if config.get("new_source_design") != _expected_new_source_design():
        raise ValueError("outcome-blind train/development source design changed")
    _validate_exact_contracts(config)
    if config.get("sealed_final_policy") != {
        "integrity_sha256_read_allowed_during_registration": True,
        "semantic_map_or_scenario_read_allowed_before_freeze": False,
        "episode_generation_allowed_before_freeze": False,
        "h1_collection_or_outcome_read_allowed_before_freeze": False,
        "access_requires_model_frozen": True,
        "access_requires_threshold_frozen": True,
        "access_requires_registration_report_passed": True,
    }:
        raise ValueError("sealed-final access policy changed")
    if config.get("claim_boundary") != {
        "registration_only": True,
        "outcome_blind": True,
        "h1_outcomes_read": False,
        "source_episodes_generated": False,
        "h1_executed": False,
        "model_fit_executed": False,
        "training_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
    }:
        raise ValueError("registration-only claim boundary changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    return config_path, project_root, config


def guard_sealed_final_semantic_access(
    *,
    model_frozen: bool,
    threshold_frozen: bool,
    registration_report_passed: bool,
) -> None:
    """Fail closed unless all three preregistered final-access locks are open."""

    if model_frozen is not True:
        raise ValueError("sealed-final semantic access forbidden before model freeze")
    if threshold_frozen is not True:
        raise ValueError("sealed-final semantic access forbidden before threshold freeze")
    if registration_report_passed is not True:
        raise ValueError("sealed-final semantic access requires a passed registration report")


def _validate_sources(project_root: Path, config: dict[str, Any]) -> None:
    for map_id in ALL_NEW_MAPS:
        registration = config["registered_sources"][map_id]
        registered_input(project_root, registration["map"], label=f"{map_id} map")
        registered_input(
            project_root,
            registration["scenario"],
            label=f"{map_id} scenario",
        )


def _validate_output(output_root: Path, project_root: Path, config: dict[str, Any]) -> None:
    source_root = (project_root / str(config["source_root"])).resolve()
    target = output_root.resolve()
    if target == source_root or source_root in target.parents:
        raise ValueError("registration output must remain outside the source root")
    if not target.exists():
        return
    if not target.is_dir():
        raise ValueError("registration output must be a directory")
    entries = sorted(path.name for path in target.iterdir())
    if any(name != "registration_report.json" for name in entries):
        raise ValueError("preflight output contains a non-registration artifact")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def run_preflight(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output_root = Path(output).resolve()
    _validate_output(output_root, root, config)
    _validate_sources(root, config)

    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "REGISTERED",
        "passed": True,
        "config_path": path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(path),
        "source_root": config["source_root"],
        "registered_map_count": 24,
        "registered_source_file_count": 48,
        "source_sha256_validation_passed": True,
        "map_splits": {
            "train": list(TRAIN_MAPS),
            "development": list(DEVELOPMENT_MAPS),
            "sealed_final": list(SEALED_FINAL_MAPS),
        },
        "map_split_counts": {"train": 8, "development": 8, "sealed_final": 8},
        "map_splits_pairwise_disjoint": True,
        "legacy_current_h1_usage": "legacy_sequential_dev/train-only",
        "legacy_current_h1_map_count": 16,
        "legacy_maps_absent_from_new_splits": True,
        "train_folds": {key: list(value) for key, value in TRAIN_FOLDS.items()},
        "new_source_design": config["new_source_design"],
        "exact_agent_set_deduplication_required": True,
        "paired_seed_count": 16,
        "stage2_condition": "at_least_one_structural_action_has_stage1_structural_win",
        "readiness_gates": config["readiness_gates"],
        "sealed_final": {
            "status": "SEALED_HASH_VERIFIED_ONLY",
            "integrity_hash_read": True,
            "semantic_map_or_scenario_content_read": False,
            "source_episode_generated": False,
            "h1_outcome_read": False,
            "semantic_access_authorized": False,
            "requires_model_and_threshold_freeze": True,
        },
        "outcome_fields_read": False,
        "h1_outcomes_read": False,
        "source_episodes_generated": False,
        "h1_executed": False,
        "model_or_threshold_fit_executed": False,
        "training_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
    }
    _atomic_json(output_root / "registration_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "DEVELOPMENT_MAPS",
    "EXPERIMENT_ID",
    "LEGACY_CURRENT_H1_MAPS",
    "REPORT_SCHEMA",
    "SEALED_FINAL_MAPS",
    "SOURCE_LOAD_GRID",
    "TRAIN_FOLDS",
    "TRAIN_MAPS",
    "guard_sealed_final_semantic_access",
    "load_config",
    "run_preflight",
    "validate_config",
    "validate_map_disjointness",
]
