from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


SOURCE_ROLE = "repairability_reset_only_load_confirmation"
RESERVE_ROLE = "repairability_reset_only_reserve_extension"
REPORT_SCHEMA = "lns2.stride.repairability_load_confirmation.v1"
RESERVE_REPORT_SCHEMA = "lns2.stride.repairability_reserve_confirmation.v1"
FORBIDDEN_FIELDS = {
    "candidate_repair_outcome",
    "selected_candidate",
    "controller_action",
    "future_trajectory",
    "controller_ttf",
    "repair_outcome",
    "conflicts_after",
}


def _variant(row: dict[str, Any]) -> str:
    task_id = str(row["task_id"])
    if "__derived_opposite_exchange__" in task_id:
        return "opposite_exchange"
    if "__derived_uniform_random__" in task_id:
        return "uniform_random"
    raise ValueError(f"unregistered repairability OD variant: {task_id}")


def validate_repairability_load_confirmation_source(config: dict[str, Any]) -> None:
    benchmarks = list(config.get("benchmarks") or ())
    if (
        config.get("schema_version") != 1
        or config.get("role") != SOURCE_ROLE
        or config.get("scientific_status")
        != "outcome_blind_initial_pp_confirmation"
        or config.get("dataset_revision")
        != "stride-repairability-load-confirmation-v1"
        or list(config.get("task_seeds") or ()) != [47, 59, 71, 83]
        or list(config.get("solver_seeds") or ()) != [1, 2]
        or list(config.get("task_variants") or ())
        != ["uniform_random", "opposite_exchange"]
        or int(config.get("expected_map_count", -1)) != 5
        or int(config.get("expected_instance_count", -1)) != 48
        or int(config.get("expected_qualification_job_count", -1)) != 96
        or int(config.get("minimum_conflicting_jobs_per_effective_map", -1)) != 8
    ):
        raise ValueError("repairability load-confirmation identity changed")
    identities = {
        (str(row.get("primary_map_id")), str(row.get("id")), str(row.get("research_split")))
        for row in benchmarks
    }
    if identities != {
        ("arena2", "den206d", "train"),
        ("brc502d", "brc502d", "train"),
        ("brc300d", "brc300d", "validation"),
        ("den001d", "den011d", "validation"),
        ("den204d", "den204d", "validation"),
    }:
        raise ValueError("repairability effective-map registration changed")
    if set(map(str, config.get("selection_inputs_forbidden") or ())) != {
        "candidate_repair_outcome",
        "selected_candidate",
        "controller_action",
        "future_trajectory",
        "controller_ttf",
    }:
        raise ValueError("repairability preflight outcome boundary changed")


def validate_repairability_reserve_extension_source(config: dict[str, Any]) -> None:
    benchmarks = list(config.get("benchmarks") or ())
    if (
        config.get("schema_version") != 1
        or config.get("role") != RESERVE_ROLE
        or config.get("scientific_status")
        != "outcome_blind_initial_pp_confirmation"
        or config.get("dataset_revision")
        != "stride-repairability-den005-reserve-v2"
        or list(config.get("task_seeds") or ()) != [109, 113, 127, 131]
        or list(config.get("solver_seeds") or ()) != [1, 2]
        or int(config.get("expected_map_count", -1)) != 1
        or int(config.get("expected_instance_count", -1)) != 8
        or int(config.get("expected_qualification_job_count", -1)) != 16
        or int(config.get("minimum_conflicting_jobs", -1)) != 8
        or len(benchmarks) != 1
    ):
        raise ValueError("repairability reserve-extension identity changed")
    case = dict(benchmarks[0])
    if (
        case.get("id") != "ht_mansion_n"
        or case.get("primary_map_id") != "den005d"
        or case.get("research_split") != "train"
        or list(case.get("agent_counts") or ()) != [900]
        or case.get("member_sha256")
        != "d0d82f6e8becb0d2c1e61bcbfd9d94ed6d061725e38107be2c9defc8dcbfd9b4"
    ):
        raise ValueError("repairability den005 reserve registration changed")
    if set(map(str, config.get("selection_inputs_forbidden") or ())) != {
        "candidate_repair_outcome",
        "selected_candidate",
        "controller_action",
        "future_trajectory",
        "controller_ttf",
    }:
        raise ValueError("repairability reserve outcome boundary changed")


def analyze_repairability_load_confirmation(
    *,
    source_config: str | Path,
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    source_config = Path(source_config).resolve()
    dataset = Path(dataset).resolve()
    qualification = Path(qualification).resolve()
    config = _read_json(source_config)
    validate_repairability_load_confirmation_source(config)
    dataset_manifest = dataset / "balanced_wall_clock" / "manifest.jsonl"
    qualification_manifest = qualification / "qualification_manifest.jsonl"
    qualification_report_path = qualification / "qualification_report.json"
    dataset_rows = _read_jsonl(dataset_manifest)
    rows = _read_jsonl(qualification_manifest)
    qualification_report = _read_json(qualification_report_path)

    forbidden_hits = sorted(
        {
            field
            for row in rows
            for field in FORBIDDEN_FIELDS
            if field in row
        }
    )
    dataset_index = {str(row["task_id"]): row for row in dataset_rows}
    solver_seeds = set(map(int, config["solver_seeds"]))
    expected = {
        (task_id, solver_seed)
        for task_id in dataset_index
        for solver_seed in solver_seeds
    }
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if key in indexed:
            errors.append(f"duplicate:{key[0]}:{key[1]}")
        indexed[key] = row
        source = dataset_index.get(key[0])
        if source is None or key[1] not in solver_seeds:
            errors.append(f"unexpected:{key[0]}:{key[1]}")
            continue
        if (
            str(row.get("status")) != "ok"
            or not bool(row.get("initial_complete"))
            or not str(row.get("state_fingerprint", ""))
        ):
            errors.append(f"invalid:{key[0]}:{key[1]}")
        if (
            str(row.get("map_id")) != str(source["map_id"])
            or int(row.get("agent_count", -1)) != int(source["agent_count"])
        ):
            errors.append(f"semantic:{key[0]}:{key[1]}")

    grouped: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["map_id"]), _variant(row), int(row["agent_count"]))
        ].append(row)
    summaries = []
    for (map_id, variant, agent_count), values in sorted(grouped.items()):
        conflicts = [int(row["initial_conflicts"]) for row in values]
        summaries.append(
            {
                "map_id": map_id,
                "task_variant": variant,
                "agent_count": agent_count,
                "job_count": len(values),
                "nonzero_job_count": sum(value > 0 for value in conflicts),
                "nonzero_job_fraction": sum(value > 0 for value in conflicts)
                / len(conflicts),
                "median_initial_conflicts": statistics.median(conflicts),
                "mean_initial_conflicts": statistics.fmean(conflicts),
                "minimum_initial_conflicts": min(conflicts),
                "maximum_initial_conflicts": max(conflicts),
            }
        )

    preferred_low, preferred_high = 5.0, 500.0
    chosen = []
    by_variant: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_variant[(str(row["map_id"]), str(row["task_variant"]))].append(row)
    for key, candidates in sorted(by_variant.items()):
        eligible = [
            row
            for row in candidates
            if float(row["nonzero_job_fraction"]) >= 2.0 / 3.0
            and preferred_low
            <= float(row["median_initial_conflicts"])
            <= preferred_high
        ]
        if eligible:
            selected = min(eligible, key=lambda row: int(row["agent_count"]))
            fallback = False
        else:
            selected = min(
                candidates,
                key=lambda row: (
                    abs(float(row["median_initial_conflicts"]) - 50.0),
                    -float(row["nonzero_job_fraction"]),
                    int(row["agent_count"]),
                ),
            )
            fallback = True
        chosen.append({**selected, "fallback": fallback})

    effective_index = {
        str(row["id"]): {
            "primary_map_id": str(row["primary_map_id"]),
            "research_split": str(row["research_split"]),
        }
        for row in config["benchmarks"]
    }
    nonzero_by_map = Counter(
        str(row["map_id"]) for row in rows if int(row["initial_conflicts"]) > 0
    )
    registered_loads = [
        {
            **row,
            **effective_index[str(row["map_id"])],
        }
        for row in chosen
    ]
    gates = {
        "dataset_count": len(dataset_rows) == int(config["expected_instance_count"]),
        "qualification_count": len(rows)
        == int(config["expected_qualification_job_count"]),
        "complete_product": set(indexed) == expected,
        "all_rows_valid": not errors,
        "qualification_passed": bool(qualification_report.get("passed")),
        "forbidden_outcomes_absent": not forbidden_hits,
        "all_effective_maps_present": set(nonzero_by_map) == set(effective_index),
        "minimum_conflicting_jobs_per_effective_map": all(
            nonzero_by_map[map_id]
            >= int(config["minimum_conflicting_jobs_per_effective_map"])
            for map_id in effective_index
        ),
        "all_map_variants_registered": len(registered_loads)
        == len(effective_index) * 2,
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "outcome_blind_initial_pp_confirmation",
        "source_config_sha256": sha256_file(source_config),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "qualification_manifest_sha256": sha256_file(qualification_manifest),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "counts": {
            "map_count": len(effective_index),
            "task_count": len(dataset_rows),
            "qualification_job_count": len(rows),
            "nonzero_jobs_by_map": dict(sorted(nonzero_by_map.items())),
        },
        "replacements": {
            row["primary_map_id"]: map_id
            for map_id, row in sorted(effective_index.items())
            if row["primary_map_id"] != map_id
        },
        "load_summaries": summaries,
        "registered_loads": registered_loads,
        "forbidden_fields_found": forbidden_hits,
        "row_errors": errors,
        "gates": gates,
        "passed": passed,
        "next_decision": (
            "freeze_loads_and_collect_source_states"
            if passed
            else "repair_load_confirmation_before_source_collection"
        ),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "load_confirmation_report.json", report)
    return report


def analyze_repairability_reserve_extension(
    *,
    source_config: str | Path,
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    source_config = Path(source_config).resolve()
    dataset = Path(dataset).resolve()
    qualification = Path(qualification).resolve()
    config = _read_json(source_config)
    validate_repairability_reserve_extension_source(config)
    dataset_manifest = dataset / "balanced_wall_clock" / "manifest.jsonl"
    qualification_manifest = qualification / "qualification_manifest.jsonl"
    qualification_report_path = qualification / "qualification_report.json"
    dataset_rows = _read_jsonl(dataset_manifest)
    rows = _read_jsonl(qualification_manifest)
    qualification_report = _read_json(qualification_report_path)
    expected = {
        (str(row["task_id"]), solver_seed)
        for row in dataset_rows
        for solver_seed in map(int, config["solver_seeds"])
    }
    observed = {
        (str(row.get("task_id")), int(row.get("solver_seed", -1))) for row in rows
    }
    forbidden_hits = sorted(
        {
            field
            for row in rows
            for field in FORBIDDEN_FIELDS
            if field in row
        }
    )
    valid = [
        row
        for row in rows
        if str(row.get("status")) == "ok"
        and bool(row.get("initial_complete"))
        and str(row.get("state_fingerprint", ""))
    ]
    summaries = []
    for variant in ("opposite_exchange", "uniform_random"):
        conflicts = [
            int(row["initial_conflicts"])
            for row in valid
            if _variant(row) == variant
        ]
        summaries.append(
            {
                "task_variant": variant,
                "job_count": len(conflicts),
                "nonzero_job_count": sum(value > 0 for value in conflicts),
                "median_initial_conflicts": statistics.median(conflicts)
                if conflicts
                else 0.0,
                "mean_initial_conflicts": statistics.fmean(conflicts)
                if conflicts
                else 0.0,
                "minimum_initial_conflicts": min(conflicts, default=0),
                "maximum_initial_conflicts": max(conflicts, default=0),
            }
        )
    nonzero = sum(int(row["initial_conflicts"]) > 0 for row in valid)
    gates = {
        "dataset_count": len(dataset_rows) == int(config["expected_instance_count"]),
        "qualification_count": len(rows)
        == int(config["expected_qualification_job_count"]),
        "complete_product": observed == expected,
        "all_rows_valid": len(valid) == len(rows),
        "qualification_passed": bool(qualification_report.get("passed")),
        "forbidden_outcomes_absent": not forbidden_hits,
        "minimum_conflicting_jobs": nonzero
        >= int(config["minimum_conflicting_jobs"]),
        "both_variants_repairable": all(
            int(row["nonzero_job_count"]) > 0 for row in summaries
        ),
    }
    report = {
        "schema": RESERVE_REPORT_SCHEMA,
        "scientific_status": "outcome_blind_initial_pp_confirmation",
        "source_config_sha256": sha256_file(source_config),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "qualification_manifest_sha256": sha256_file(qualification_manifest),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "primary_map_id": "den005d",
        "effective_map_id": "ht_mansion_n",
        "research_split": "train",
        "agent_count": 900,
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "variant_summaries": summaries,
        "counts": {
            "qualification_jobs": len(rows),
            "valid_jobs": len(valid),
            "nonzero_jobs": nonzero,
        },
        "forbidden_fields_found": forbidden_hits,
        "gates": gates,
        "passed": all(gates.values()),
        "next_decision": (
            "register_ht_mansion_n_as_den005d_train_reserve"
            if all(gates.values())
            else "reject_reserve_and_redesign_train_map_split"
        ),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "reserve_confirmation_report.json", report)
    return report


__all__ = [
    "analyze_repairability_load_confirmation",
    "analyze_repairability_reserve_extension",
    "validate_repairability_load_confirmation_source",
    "validate_repairability_reserve_extension_source",
]
