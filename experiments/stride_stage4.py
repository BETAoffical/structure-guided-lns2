from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_stage3 import _project_path


STRIDE_STAGE4_TRAINING_CONFIG_SCHEMA = "lns2.stride.stage4_training_config.v1"
STRIDE_STAGE4_FOLD_SCHEMA = "lns2.stride.stage4_fold_assignment.v1"
STRIDE_STAGE4_PROTOCOL_SCHEMA = "lns2.stride.stage4_protocol.v1"

def _layout_family(layout_mode: str, aliases: dict[str, str]) -> str:
    return aliases.get(layout_mode, layout_mode)


def _count_dimensions(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["total"] += 1
        counts[f"source_policy:{row['source_policy']}"] += 1
        counts[f"agent_band:{row['agent_band']}"] += 1
        counts[f"decision_stage:{row['decision_stage']}"] += 1
    return counts


def _assign_grouped_folds(
    rows: list[dict[str, Any]],
    *,
    fold_count: int,
    layout_aliases: dict[str, str],
) -> tuple[dict[str, int], list[list[str]], dict[str, str]]:
    """Assign whole maps using outcome-blind, deterministic metadata balancing."""

    if fold_count < 2:
        raise ValueError("Stage 4 requires at least two folds")

    rows_by_map: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    map_families: dict[str, str] = {}
    for row in rows:
        map_id = str(row["map_id"])
        layout_mode = str(row["layout_mode"])
        family = _layout_family(layout_mode, layout_aliases)
        previous = map_families.setdefault(map_id, family)
        if previous != family:
            raise ValueError(f"map has inconsistent layout family: {map_id}")
        rows_by_map[map_id].append(row)

    maps_by_family: defaultdict[str, list[str]] = defaultdict(list)
    for map_id, family in map_families.items():
        maps_by_family[family].append(map_id)

    global_counts = _count_dimensions(rows)
    targets = {
        name: value / float(fold_count) for name, value in global_counts.items()
    }
    fold_counts = [Counter() for _ in range(fold_count)]
    fold_maps: list[list[str]] = [[] for _ in range(fold_count)]
    family_fold_counts = [Counter() for _ in range(fold_count)]
    map_to_fold: dict[str, int] = {}

    family_order = sorted(
        maps_by_family,
        key=lambda family: (
            -sum(len(rows_by_map[map_id]) for map_id in maps_by_family[family]),
            family,
        ),
    )
    for family in family_order:
        map_order = sorted(
            maps_by_family[family],
            key=lambda map_id: (-len(rows_by_map[map_id]), map_id),
        )
        for map_id in map_order:
            map_counts = _count_dimensions(rows_by_map[map_id])

            def assignment_score(fold_index: int) -> tuple[Any, ...]:
                ratios = [
                    (fold_counts[fold_index][name] + map_counts[name]) / target
                    for name, target in targets.items()
                    if target > 0.0
                ]
                return (
                    family_fold_counts[fold_index][family],
                    max(ratios, default=0.0),
                    sum(value * value for value in ratios),
                    len(fold_maps[fold_index]),
                    fold_index,
                )

            fold_index = min(range(fold_count), key=assignment_score)
            map_to_fold[map_id] = fold_index
            fold_maps[fold_index].append(map_id)
            fold_counts[fold_index].update(map_counts)
            family_fold_counts[fold_index][family] += 1

    state_to_fold = {
        str(row["state_id"]): map_to_fold[str(row["map_id"])] for row in rows
    }
    return state_to_fold, fold_maps, map_families


def _fold_counts(
    rows: list[dict[str, Any]], state_to_fold: dict[str, int], fold_count: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fold_index in range(fold_count):
        selected = [
            row
            for row in rows
            if state_to_fold[str(row["state_id"])] == fold_index
        ]
        result.append(
            {
                "fold_index": fold_index,
                "state_count": len(selected),
                "source_policy_state_counts": dict(
                    sorted(Counter(str(row["source_policy"]) for row in selected).items())
                ),
                "agent_band_state_counts": dict(
                    sorted(Counter(str(row["agent_band"]) for row in selected).items())
                ),
                "decision_stage_state_counts": dict(
                    sorted(
                        Counter(str(row["decision_stage"]) for row in selected).items()
                    )
                ),
            }
        )
    return result


def prepare_stride_stage4_protocol(
    *, config_path: Path, output: Path, project_root: Path
) -> dict[str, Any]:
    """Freeze the Stage 4 protocol and map-grouped folds without reading labels."""

    config = _read_json(config_path)
    if config.get("schema") != STRIDE_STAGE4_TRAINING_CONFIG_SCHEMA:
        raise ValueError("unexpected STRIDE Stage 4 training-config schema")

    project_root = project_root.resolve()
    audit_path = _project_path(project_root, str(config["stage3_audit_report"]))
    audit = _read_json(audit_path)
    selection_paths = [
        _project_path(project_root, str(value))
        for value in config.get("selections", [])
    ]
    if not selection_paths:
        raise ValueError("Stage 4 requires registered selection files")

    selected_rows: list[dict[str, Any]] = []
    for path in selection_paths:
        selected_rows.extend(_read_jsonl(path))

    required_fields = {
        "state_id",
        "map_id",
        "layout_mode",
        "split",
        "source_policy",
        "agent_band",
        "decision_stage",
        "agent_count",
    }
    missing_field_count = sum(
        not required_fields.issubset(row) for row in selected_rows
    )
    state_ids = [str(row.get("state_id", "")) for row in selected_rows]
    duplicate_state_count = len(state_ids) - len(set(state_ids))

    protocol = dict(config.get("protocol", {}))
    models = dict(protocol.get("models", {}))
    feature_variants = dict(protocol.get("feature_variants", {}))
    registered_names_valid = (
        dict(models.get("frozen_anchor", {})).get("id") == "v2-full"
        and dict(models.get("frozen_anchor", {})).get("read_only") is True
        and dict(models.get("control", {})).get("id") == "stride-control-v1"
        and dict(models.get("quality", {})).get("id") == "stride-quality-v1"
    )
    feature_variants_valid = "full" in feature_variants and len(feature_variants) >= 2
    primary_label = str(protocol.get("primary_label", ""))
    forbidden_labels = {str(value) for value in protocol.get("forbidden_primary_labels", [])}

    fold_count = int(config["fold_count"])
    aliases = {
        str(name): str(value)
        for name, value in dict(config.get("layout_family_aliases", {})).items()
    }
    state_to_fold, fold_maps, map_families = _assign_grouped_folds(
        selected_rows, fold_count=fold_count, layout_aliases=aliases
    )
    counts = _fold_counts(selected_rows, state_to_fold, fold_count)

    manifest_rows = [
        {
            "schema": STRIDE_STAGE4_FOLD_SCHEMA,
            "state_id": str(row["state_id"]),
            "map_id": str(row["map_id"]),
            "layout_mode": str(row["layout_mode"]),
            "layout_family": map_families[str(row["map_id"])],
            "split": str(row["split"]),
            "source_policy": str(row["source_policy"]),
            "agent_band": str(row["agent_band"]),
            "decision_stage": str(row["decision_stage"]),
            "agent_count": int(row["agent_count"]),
            "fold_index": state_to_fold[str(row["state_id"])],
        }
        for row in sorted(selected_rows, key=lambda item: str(item["state_id"]))
    ]

    thresholds = dict(config.get("fold_gates", {}))
    expected_state_count = int(config["expected_state_count"])
    expected_policy_counts = {
        str(name): int(value)
        for name, value in dict(config["expected_state_count_per_policy"]).items()
    }
    policy_counts = Counter(str(row["source_policy"]) for row in selected_rows)
    required_families = {
        str(value) for value in config.get("required_layout_families", [])
    }
    fold_families = [
        {map_families[map_id] for map_id in map_ids} for map_ids in fold_maps
    ]
    min_policy = int(thresholds.get("min_policy_states_per_fold", 0))
    min_agent_band = int(thresholds.get("min_agent_band_states_per_fold", 0))
    expected_agent_bands = {
        str(row["agent_band"]) for row in selected_rows
    }
    stage_minima = {
        str(name): int(value)
        for name, value in dict(
            thresholds.get("min_decision_stage_states_per_fold", {})
        ).items()
    }
    map_fold_occurrences = Counter(
        map_id for map_ids in fold_maps for map_id in map_ids
    )
    audit_hash = sha256_file(audit_path)
    gates = {
        "stage3_audit_passed": audit.get("passed") is True,
        "stage3_audit_hash_frozen": audit_hash
        == str(config["stage3_audit_sha256"]),
        "stage3_state_count_matches": int(audit.get("state_count", -1))
        == expected_state_count,
        "selection_files_exist": all(path.is_file() for path in selection_paths),
        "selection_fields_complete": missing_field_count == 0,
        "selection_state_count": len(selected_rows) == expected_state_count,
        "selection_states_unique": duplicate_state_count == 0,
        "source_policy_balance": dict(policy_counts) == expected_policy_counts,
        "registered_model_names_valid": registered_names_valid,
        "feature_variants_registered": feature_variants_valid,
        "runtime_excluded_from_primary_label": (
            protocol.get("runtime_used_in_primary_label") is False
        ),
        "forbidden_primary_labels_excluded": primary_label not in forbidden_labels,
        "all_states_assigned": len(state_to_fold) == expected_state_count,
        "maps_grouped_exclusively": (
            len(map_fold_occurrences) == len(map_families)
            and all(value == 1 for value in map_fold_occurrences.values())
        ),
        "fold_state_minimum": all(
            row["state_count"] >= int(thresholds.get("min_states_per_fold", 0))
            for row in counts
        ),
        "fold_state_maximum": all(
            row["state_count"] <= int(
                thresholds.get("max_states_per_fold", expected_state_count)
            )
            for row in counts
        ),
        "fold_map_minimum": all(
            len(map_ids) >= int(thresholds.get("min_maps_per_fold", 0))
            for map_ids in fold_maps
        ),
        "each_layout_family_in_each_fold": all(
            required_families.issubset(families) for families in fold_families
        ),
        "fold_policy_minimum": all(
            all(value >= min_policy for value in row["source_policy_state_counts"].values())
            and set(row["source_policy_state_counts"]) == set(expected_policy_counts)
            for row in counts
        ),
        "fold_agent_band_minimum": all(
            set(row["agent_band_state_counts"]) == expected_agent_bands
            and all(
                value >= min_agent_band
                for value in row["agent_band_state_counts"].values()
            )
            for row in counts
        ),
        "fold_decision_stage_minimum": all(
            all(
                row["decision_stage_state_counts"].get(name, 0) >= minimum
                for name, minimum in stage_minima.items()
            )
            for row in counts
        ),
    }

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "fold_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    try:
        config_name = config_path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        config_name = config_path.name
    report = {
        "schema": STRIDE_STAGE4_PROTOCOL_SCHEMA,
        "config": config_name,
        "config_sha256": _fingerprint(config),
        "passed": all(gates.values()),
        "gates": gates,
        "label_outcomes_read": False,
        "state_count": len(selected_rows),
        "map_count": len(map_families),
        "fold_count": fold_count,
        "folds": [
            {
                **row,
                "map_count": len(fold_maps[row["fold_index"]]),
                "maps": sorted(fold_maps[row["fold_index"]]),
                "layout_families": sorted(fold_families[row["fold_index"]]),
            }
            for row in counts
        ],
        "layout_family_map_counts": dict(
            sorted(Counter(map_families.values()).items())
        ),
        "source_policy_state_counts": dict(sorted(policy_counts.items())),
        "diagnostics": {
            "missing_selection_field_count": missing_field_count,
            "duplicate_state_count": duplicate_state_count,
        },
        "stage3_audit_sha256": audit_hash,
        "selection_sha256": [sha256_file(path) for path in selection_paths],
        "fold_manifest_sha256": sha256_file(manifest_path),
        "registered_protocol": protocol,
    }
    _write_json(output / "stage4_protocol_report.json", report)
    return report


__all__ = [
    "STRIDE_STAGE4_FOLD_SCHEMA",
    "STRIDE_STAGE4_PROTOCOL_SCHEMA",
    "STRIDE_STAGE4_TRAINING_CONFIG_SCHEMA",
    "prepare_stride_stage4_protocol",
]
