from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _write_json, _write_jsonl
from experiments.stride_collection import load_stride_selection
from experiments.stride_repairability import CONTROLLER_ID


DESIGN_SCHEMA = "lns2.stride.repairability_data_design.v1"
REPORT_SCHEMA = "lns2.stride.repairability_selection_report.v1"


def _conflict_band(conflicts: int) -> str:
    if conflicts <= 10:
        return "1_10"
    if conflicts <= 100:
        return "11_100"
    if conflicts <= 500:
        return "101_500"
    return "501_plus"


def validate_repairability_data_design(config: dict[str, Any]) -> None:
    if config.get("schema") != DESIGN_SCHEMA:
        raise ValueError("unexpected repairability data design")
    if (
        config.get("scientific_status")
        != "preregistered_before_new_candidate_repair_outcomes"
        or config.get("controller_id") != CONTROLLER_ID
        or int(config.get("pilot_state_count", -1)) != 240
        or list(config.get("source_policies") or ())
        != ["official_adaptive", "v2-full"]
        or int(config.get("maximum_states_per_episode", -1)) != 2
        or int(config.get("minimum_states_per_map", -1)) != 8
        or float(config.get("topology_boundary_threshold", -1.0)) != 0.06
    ):
        raise ValueError("repairability data-design identity changed")
    if (
        list(config.get("conflict_bands") or ())
        != ["1_10", "11_100", "101_500", "501_plus"]
        or list(config.get("required_conflict_band_coverage") or ())
        != ["1_10", "11_100", "101_500"]
        or float(config.get("minimum_101_plus_state_fraction", -1.0)) != 0.10
    ):
        raise ValueError("repairability conflict-load balance changed")
    split = dict(config.get("map_split") or {})
    train = list(map(str, split.get("train") or ()))
    validation = list(map(str, split.get("validation") or ()))
    if (
        len(train) != 16
        or len(validation) != 6
        or len(set(train)) != len(train)
        or len(set(validation)) != len(validation)
        or set(train) & set(validation)
    ):
        raise ValueError("repairability map split changed")
    reserves = dict(config.get("same_split_reserves") or {})
    if set(reserves) != {
        "arena2",
        "brc300d",
        "brc502d",
        "den005d",
        "den001d",
        "den204d",
    }:
        raise ValueError("repairability reserve-map registration changed")
    load_rule = dict(config.get("qualified_load_rule") or {})
    if (
        load_rule.get("preflight_source_config")
        != "configs/stride_repairability_underload_preflight_source.json"
        or load_rule.get("confirmation_source_config")
        != "configs/stride_repairability_load_confirmation_source.json"
        or load_rule.get("reserve_extension_source_config")
        != "configs/stride_repairability_den005_reserve_source.json"
        or bool(load_rule.get("uses_candidate_outcomes"))
        or list(load_rule.get("preferred_initial_conflict_interval") or ())
        != [5, 500]
        or float(load_rule.get("minimum_nonzero_solver_seed_fraction", -1.0))
        != 2.0 / 3.0
        or set(map(str, load_rule.get("underloaded_maps_require_new_reset_only_preflight") or ()))
        != set(reserves)
    ):
        raise ValueError("repairability reset-only load rule changed")
    ratios = {
        str(name): float(value)
        for name, value in dict(
            config.get("static_low_degree_cell_ratio_by_map") or {}
        ).items()
    }
    reserve_ids = {str(row["id"]) for row in reserves.values()}
    if set(ratios) != set(train) | set(validation) | reserve_ids or any(
        not 0.0 <= value <= 1.0 for value in ratios.values()
    ):
        raise ValueError("repairability static topology registration changed")
    for primary, row in reserves.items():
        expected_split = "train" if primary in train else "validation"
        if (
            str(row.get("split")) != expected_split
            or len(str(row.get("map_sha256", ""))) != 64
            or float(row.get("static_low_degree_cell_ratio", -1.0))
            != ratios[str(row["id"])]
        ):
            raise ValueError("repairability reserve-map metadata changed")
    excluded = set(map(str, config.get("formal_ood_maps_excluded") or ()))
    if len(excluded) != 12 or excluded & (set(ratios) | reserve_ids):
        raise ValueError("repairability formal OOD isolation changed")
    gates = dict(config.get("balance_gates") or {})
    if gates != {
        "source_policy_fraction_minimum": 0.45,
        "source_policy_fraction_maximum": 0.55,
        "both_source_policies_per_map": True,
        "topology_group_fraction_minimum": 0.40,
        "early_middle_late_coverage_required": True,
        "all_registered_maps_required": True,
    }:
        raise ValueError("repairability state-balance gates changed")
    if (
        not bool(config.get("map_level_split_required"))
        or not bool(config.get("repair_outcome_blind_selection"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("repairability data evidence boundary changed")


def prepare_repairability_selection(
    *, config_path: str | Path, selection_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    selection_path = Path(selection_path).resolve()
    config = _read_json(config_path)
    validate_repairability_data_design(config)
    rows = load_stride_selection(selection_path)
    forbidden = set(map(str, config["forbidden_selection_inputs"]))
    present_names = {str(name) for row in rows for name in row}
    present_forbidden = sorted(
        name
        for name in present_names
        if name in forbidden
        or name.startswith(("candidate_", "after_", "future_"))
        or name
        in {
            "actual_action",
            "actual_lns2",
            "repair_seconds",
            "repair_state_changed",
            "replay_action",
            "step_runtime",
            "ttf",
        }
    )
    if present_forbidden:
        raise ValueError(
            f"repairability selection contains outcome fields: {present_forbidden}"
        )

    split = {
        str(map_id): research_split
        for research_split, map_ids in dict(config["map_split"]).items()
        for map_id in map_ids
    }
    minimum = int(config["minimum_states_per_map"])
    available_counts = Counter(str(row["map_id"]) for row in rows)
    reserves = dict(config["same_split_reserves"])
    effective: dict[str, str] = {}
    replacements: dict[str, str] = {}
    for primary, research_split in sorted(split.items()):
        if available_counts[primary] >= minimum:
            effective[primary] = research_split
            continue
        if primary not in reserves:
            raise ValueError(f"repairability map lacks enough states: {primary}")
        reserve = str(reserves[primary]["id"])
        if available_counts[reserve] < minimum:
            raise ValueError(
                f"repairability primary and reserve both lack states: {primary}/{reserve}"
            )
        effective[reserve] = research_split
        replacements[primary] = reserve

    selected = [row for row in rows if str(row["map_id"]) in effective]
    if len(selected) != int(config["pilot_state_count"]):
        raise ValueError("repairability selection does not contain exactly 240 states")
    unexpected_maps = sorted(set(available_counts) - set(effective))
    if unexpected_maps:
        raise ValueError(
            f"repairability source selection contains unregistered maps: {unexpected_maps}"
        )
    ratios = {
        str(name): float(value)
        for name, value in dict(config["static_low_degree_cell_ratio_by_map"]).items()
    }
    threshold = float(config["topology_boundary_threshold"])
    annotated = []
    for row in selected:
        map_id = str(row["map_id"])
        before_conflicts = int(row.get("before_conflicts", 0))
        if before_conflicts <= 0:
            raise ValueError("repairability selection requires conflicting states")
        annotated.append(
            {
                **row,
                "research_split": effective[map_id],
                "static_low_degree_cell_ratio": ratios[map_id],
                "topology_group": (
                    "boundary_relevant" if ratios[map_id] >= threshold else "control"
                ),
                "conflict_band": _conflict_band(before_conflicts),
            }
        )
    annotated.sort(key=lambda row: (str(row["research_split"]), str(row["state_id"])))

    total = len(annotated)
    policy_counts = Counter(str(row["source_policy"]) for row in annotated)
    topology_counts = Counter(str(row["topology_group"]) for row in annotated)
    stage_counts = Counter(str(row["decision_stage"]) for row in annotated)
    conflict_counts = Counter(str(row["conflict_band"]) for row in annotated)
    map_counts = Counter(str(row["map_id"]) for row in annotated)
    map_policies: dict[str, set[str]] = {}
    for row in annotated:
        map_policies.setdefault(str(row["map_id"]), set()).add(
            str(row["source_policy"])
        )
    episode_counts = Counter(str(row["episode_id"]) for row in annotated)
    gates_config = dict(config["balance_gates"])
    gates = {
        "state_count": total == int(config["pilot_state_count"]),
        "source_policy_balance": all(
            float(gates_config["source_policy_fraction_minimum"])
            <= policy_counts[policy] / total
            <= float(gates_config["source_policy_fraction_maximum"])
            for policy in config["source_policies"]
        ),
        "both_source_policies_per_map": all(
            policies == set(config["source_policies"])
            for policies in map_policies.values()
        ),
        "topology_balance": all(
            topology_counts[group] / total
            >= float(gates_config["topology_group_fraction_minimum"])
            for group in ("control", "boundary_relevant")
        ),
        "decision_stage_coverage": set(stage_counts) == {"early", "middle", "late"},
        "conflict_band_coverage": (
            set(config["required_conflict_band_coverage"])
            <= set(conflict_counts)
        ),
        "high_conflict_fraction": (
            (
                conflict_counts["101_500"]
                + conflict_counts["501_plus"]
            )
            / total
            >= float(config["minimum_101_plus_state_fraction"])
        ),
        "all_effective_maps": set(map_counts) == set(effective),
        "minimum_states_per_map": all(
            count >= minimum for count in map_counts.values()
        ),
        "episode_cap": max(episode_counts.values(), default=0)
        <= int(config["maximum_states_per_episode"]),
        "map_level_split": not (
            {
                str(row["map_id"])
                for row in annotated
                if row["research_split"] == "train"
            }
            & {
                str(row["map_id"])
                for row in annotated
                if row["research_split"] == "validation"
            }
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"repairability state selection failed gates: {gates}")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "state_selection.jsonl"
    _write_jsonl(selected_path, annotated)
    report = {
        "schema": REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "result_blind": True,
        "config_sha256": sha256_file(config_path),
        "source_selection_sha256": sha256_file(selection_path),
        "selection_sha256": sha256_file(selected_path),
        "state_count": total,
        "effective_map_count": len(effective),
        "effective_map_split": {
            research_split: sorted(
                map_id
                for map_id, assigned in effective.items()
                if assigned == research_split
            )
            for research_split in ("train", "validation")
        },
        "replacements": replacements,
        "policy_counts": dict(sorted(policy_counts.items())),
        "topology_group_counts": dict(sorted(topology_counts.items())),
        "decision_stage_counts": dict(sorted(stage_counts.items())),
        "conflict_band_counts": dict(sorted(conflict_counts.items())),
        "map_counts": dict(sorted(map_counts.items())),
        "gates": gates,
        "passed": True,
        "candidate_outcomes_read": False,
    }
    _write_json(output / "selection_report.json", report)
    return report


__all__ = [
    "prepare_repairability_selection",
    "validate_repairability_data_design",
]
