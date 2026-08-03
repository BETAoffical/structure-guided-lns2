from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_collection import (
    STRIDE_SELECTION_SCHEMA,
    STRIDE_SOURCE_POLICIES,
    _agent_band,
    _decision_stage,
    load_stride_selection,
)
from experiments.stride_repairability import CONTROLLER_ID
from experiments.trace_replay import result_blind_decision_rows


DESIGN_SCHEMA = "lns2.stride.repairability_data_design.v1"
REPORT_SCHEMA = "lns2.stride.repairability_selection_report.v1"
SOURCE_REPORT_SCHEMA = "lns2.stride.repairability_source_selection.v1"


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


def _effective_map_split(
    config: dict[str, Any], available_maps: set[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    reserves = dict(config["same_split_reserves"])
    for research_split, map_ids in dict(config["map_split"]).items():
        for primary in map(str, map_ids):
            if primary in available_maps:
                result[primary] = str(research_split)
                continue
            reserve = str(reserves.get(primary, {}).get("id", ""))
            if reserve and reserve in available_maps:
                result[reserve] = str(research_split)
                continue
            raise ValueError(
                f"repairability source map and reserve are unavailable: {primary}"
            )
    unexpected = sorted(available_maps - set(result))
    if unexpected:
        raise ValueError(
            f"repairability source contains unregistered maps: {unexpected}"
        )
    return result


def _source_selection_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_policy": str(row["source_policy"]),
        "episode_id": str(row["episode_id"]),
        "map_id": str(row["map_id"]),
        "task_id": str(row["task_id"]),
        "solver_seed": int(row["solver_seed"]),
        "decision_index": int(row["decision_index"]),
        "before_fingerprint": str(row["before_fingerprint"]),
        "before_conflicts": int(row["before_conflicts"]),
        "agent_count": int(row["agent_count"]),
        "static_low_degree_cell_ratio": float(
            row["static_low_degree_cell_ratio"]
        ),
    }


def _select_repairability_source_pool(
    pool: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose 120 states per policy using pre-action fields only."""

    allowed = {
        "schema",
        "state_id",
        "map_id",
        "task_id",
        "split",
        "source_policy",
        "decision_stage",
        "source_root",
        "episode_id",
        "before_fingerprint",
        "before_conflicts",
        "solver_seed",
        "decision_index",
        "agent_count",
        "agent_band",
        "prefix_actions",
        "static_low_degree_cell_ratio",
        "topology_group",
        "conflict_band",
        "research_split",
    }
    forbidden = sorted({name for row in pool for name in row if name not in allowed})
    if forbidden:
        raise ValueError(
            f"repairability result-blind pool has forbidden fields: {forbidden}"
        )
    target_total = int(config["pilot_state_count"])
    policies = list(map(str, config["source_policies"]))
    if target_total % len(policies):
        raise ValueError("repairability state target is not policy balanced")
    target_per_policy = target_total // len(policies)
    maximum_per_episode = int(config["maximum_states_per_episode"])
    minimum_per_map = int(config["minimum_states_per_map"])
    minimum_per_policy_map = math.ceil(minimum_per_map / len(policies))
    topology_target = math.ceil(
        float(config["balance_gates"]["topology_group_fraction_minimum"])
        * target_per_policy
    )
    high_target = math.ceil(
        float(config["minimum_101_plus_state_fraction"]) * target_per_policy
    )
    stage_target = math.ceil(target_per_policy / 3)
    effective_maps = sorted(
        {
            str(row["map_id"])
            for row in pool
        }
    )
    selected: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for policy in policies:
        candidates = [
            row for row in pool if str(row["source_policy"]) == policy
        ]
        chosen: list[dict[str, Any]] = []
        episode_counts: Counter[str] = Counter()
        map_counts: Counter[str] = Counter()
        stage_counts: Counter[str] = Counter()
        conflict_counts: Counter[str] = Counter()
        topology_counts: Counter[str] = Counter()

        def add(row: dict[str, Any]) -> None:
            chosen.append(row)
            candidates.remove(row)
            episode_counts[str(row["episode_id"])] += 1
            map_counts[str(row["map_id"])] += 1
            stage_counts[str(row["decision_stage"])] += 1
            conflict_counts[str(row["conflict_band"])] += 1
            topology_counts[str(row["topology_group"])] += 1

        for map_id in effective_maps:
            while map_counts[map_id] < minimum_per_policy_map:
                eligible = [
                    row
                    for row in candidates
                    if str(row["map_id"]) == map_id
                    and episode_counts[str(row["episode_id"])]
                    < maximum_per_episode
                ]
                if not eligible:
                    raise ValueError(
                        "repairability source lacks the per-map policy floor: "
                        f"{policy}/{map_id}"
                    )

                def map_score(row: dict[str, Any]) -> tuple[Any, ...]:
                    return (
                        int(episode_counts[str(row["episode_id"])] > 0),
                        stage_counts[str(row["decision_stage"])],
                        conflict_counts[str(row["conflict_band"])],
                        int(row["decision_index"]),
                        int(_fingerprint(_source_selection_identity(row))[:16], 16),
                    )

                add(min(eligible, key=map_score))

        while len(chosen) < target_per_policy:
            eligible = [
                row
                for row in candidates
                if episode_counts[str(row["episode_id"])] < maximum_per_episode
            ]
            if not eligible:
                break
            high_count = (
                conflict_counts["101_500"] + conflict_counts["501_plus"]
            )

            def global_score(row: dict[str, Any]) -> tuple[Any, ...]:
                is_high = str(row["conflict_band"]) in {"101_500", "501_plus"}
                group = str(row["topology_group"])
                stage = str(row["decision_stage"])
                return (
                    int(high_count < high_target and not is_high),
                    int(topology_counts[group] >= topology_target),
                    int(stage_counts[stage] >= stage_target),
                    map_counts[str(row["map_id"])],
                    conflict_counts[str(row["conflict_band"])],
                    stage_counts[stage],
                    episode_counts[str(row["episode_id"])],
                    int(_fingerprint(_source_selection_identity(row))[:16], 16),
                )

            add(min(eligible, key=global_score))

        if len(chosen) != target_per_policy:
            raise ValueError(
                f"repairability source policy lacks {target_per_policy} states: {policy}"
            )
        selected.extend(chosen)
        reports[policy] = {
            "available_state_count": sum(
                str(row["source_policy"]) == policy for row in pool
            ),
            "selected_state_count": len(chosen),
            "episode_count": len(episode_counts),
            "maximum_states_per_episode": max(episode_counts.values(), default=0),
            "map_counts": dict(sorted(map_counts.items())),
            "decision_stage_counts": dict(sorted(stage_counts.items())),
            "conflict_band_counts": dict(sorted(conflict_counts.items())),
            "topology_group_counts": dict(sorted(topology_counts.items())),
        }
    selected.sort(key=lambda row: (str(row["source_policy"]), str(row["state_id"])))
    return selected, reports


def build_repairability_source_selection(
    *, config_path: str | Path, source_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """Extract and select the preregistered 240 pre-action trace states."""

    config_path = Path(config_path).resolve()
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_repairability_data_design(config)
    run = _read_json(source_root / "run_config.json")
    source_split = str(dict(run["configuration"]).get("split", ""))
    if source_split != "balanced_wall_clock":
        raise ValueError("repairability source uses an unexpected split")
    dataset_root = Path(str(run["dataset"])).resolve()
    dataset = {
        str(row["task_id"]): row
        for row in _load_dataset_rows(dataset_root, [source_split])
    }
    available_maps = {str(row["map_id"]) for row in dataset.values()}
    effective_split = _effective_map_split(config, available_maps)
    ratios = {
        str(name): float(value)
        for name, value in dict(
            config["static_low_degree_cell_ratio_by_map"]
        ).items()
    }
    threshold = float(config["topology_boundary_threshold"])
    pool: list[dict[str, Any]] = []
    source_manifest_hashes: dict[str, str] = {}
    for policy in map(str, config["source_policies"]):
        manifest_name, registered_policy = STRIDE_SOURCE_POLICIES[policy]
        manifest_path = source_root / manifest_name
        source_manifest_hashes[policy] = sha256_file(manifest_path)
        for manifest in _read_jsonl(manifest_path):
            if str(manifest.get("status")) != "ok":
                continue
            task_id = str(manifest["task_id"])
            dataset_row = dataset.get(task_id)
            if dataset_row is None:
                raise ValueError(
                    f"repairability source task is absent from dataset: {task_id}"
                )
            map_id = str(manifest["map_id"])
            if map_id not in effective_split:
                raise ValueError(f"repairability source map is unregistered: {map_id}")
            decisions, _ = result_blind_decision_rows(source_root, manifest)
            for decision in decisions:
                decision_index = int(decision["decision_index"])
                before_conflicts = int(decision["before_conflicts"])
                if before_conflicts <= 0 or decision_index > 11:
                    continue
                identity = {
                    "source_policy": registered_policy,
                    "episode_id": str(manifest["episode_id"]),
                    "map_id": map_id,
                    "task_id": task_id,
                    "solver_seed": int(manifest["solver_seed"]),
                    "decision_index": decision_index,
                    "before_fingerprint": str(decision["before_fingerprint"]),
                    "before_conflicts": before_conflicts,
                    "agent_count": int(manifest["agent_count"]),
                    "static_low_degree_cell_ratio": ratios[map_id],
                }
                pool.append(
                    {
                        "schema": STRIDE_SELECTION_SCHEMA,
                        "state_id": "stride-repairability-"
                        + _fingerprint(identity)[:24],
                        "map_id": map_id,
                        "task_id": task_id,
                        "split": str(manifest["split"]),
                        "source_policy": registered_policy,
                        "decision_stage": _decision_stage(decision_index),
                        "source_root": str(source_root),
                        "episode_id": str(manifest["episode_id"]),
                        "before_fingerprint": identity["before_fingerprint"],
                        "before_conflicts": before_conflicts,
                        "solver_seed": identity["solver_seed"],
                        "decision_index": decision_index,
                        "agent_count": identity["agent_count"],
                        "agent_band": _agent_band(identity["agent_count"]),
                        "prefix_actions": list(decision["prefix_actions"]),
                        "static_low_degree_cell_ratio": ratios[map_id],
                        "topology_group": (
                            "boundary_relevant"
                            if ratios[map_id] >= threshold
                            else "control"
                        ),
                        "conflict_band": _conflict_band(before_conflicts),
                        "research_split": effective_split[map_id],
                    }
                )
    state_ids = [str(row["state_id"]) for row in pool]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("repairability source contains duplicate state identities")
    selected, policy_reports = _select_repairability_source_pool(pool, config)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "result_blind_source_selection.jsonl"
    _write_jsonl(raw_path, selected)
    selection_report = prepare_repairability_selection(
        config_path=config_path,
        selection_path=raw_path,
        output=output,
    )
    report = {
        "schema": SOURCE_REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "result_blind": True,
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "config_sha256": sha256_file(config_path),
        "source_run_config_sha256": sha256_file(source_root / "run_config.json"),
        "source_manifest_sha256": source_manifest_hashes,
        "available_state_count": len(pool),
        "selected_state_count": len(selected),
        "effective_map_split": {
            split: sorted(
                map_id
                for map_id, assigned in effective_split.items()
                if assigned == split
            )
            for split in ("train", "validation")
        },
        "policies": policy_reports,
        "selection_report_sha256": sha256_file(output / "selection_report.json"),
        "selection_sha256": selection_report["selection_sha256"],
        "passed": bool(selection_report["passed"]),
    }
    _write_json(output / "source_selection_report.json", report)
    return report


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
    "build_repairability_source_selection",
    "prepare_repairability_selection",
    "validate_repairability_data_design",
]
