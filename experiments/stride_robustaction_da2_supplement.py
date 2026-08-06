from __future__ import annotations

import hashlib
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.balanced_wall_clock import (
    _largest_four_connected_component,
    _map_metrics,
    _movingai_passable_cells,
)
from experiments.repair_collection import _read_json, _write_json
from experiments.stride_robustaction_expansion import (
    DA2_SUPPLEMENT_SCHEMA,
    TOPOLOGY_GROUPS,
    topology_group,
    validate_robustaction_expansion_design,
)


REPORT_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_supplement_static_audit.v1"
)


def _selection_key(row: dict[str, Any], target: int) -> tuple[Any, ...]:
    return (
        abs(int(row["largest_four_connected_component"]) - target),
        -float(row["static_low_degree_cell_ratio"]),
        str(row["map_id"]),
    )


def select_static_da2_maps(
    rows: Iterable[dict[str, Any]], static: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the preregistered static-only DA2 filtering and family ranking."""

    minimum = int(static["minimum_largest_component"])
    maximum = int(static["maximum_largest_component"])
    target = int(static["target_largest_component"])
    blocked = tuple(map(str, static["blocked_scene_prefixes"]))
    thresholds = list(map(float, static["topology_thresholds"]))
    selected_counts = {
        str(name): int(count)
        for name, count in dict(static["selected_group_counts"]).items()
    }

    candidates: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        map_id = str(row["map_id"])
        component = int(row["largest_four_connected_component"])
        if any(map_id.startswith(prefix) for prefix in blocked):
            continue
        if not minimum <= component <= maximum:
            continue
        row["map_family"] = map_id.split("_", 1)[0]
        row["topology_group"] = topology_group(
            float(row["static_low_degree_cell_ratio"]), thresholds
        )
        candidates.append(row)

    representatives: list[dict[str, Any]] = []
    for group in TOPOLOGY_GROUPS:
        families = sorted(
            {
                str(row["map_family"])
                for row in candidates
                if str(row["topology_group"]) == group
            }
        )
        for family in families:
            family_rows = [
                row
                for row in candidates
                if str(row["topology_group"]) == group
                and str(row["map_family"]) == family
            ]
            representatives.append(min(family_rows, key=lambda row: _selection_key(row, target)))

    selected: list[dict[str, Any]] = []
    for group in TOPOLOGY_GROUPS:
        group_rows = [
            row for row in representatives if str(row["topology_group"]) == group
        ]
        selected.extend(
            sorted(group_rows, key=lambda row: _selection_key(row, target))[
                : selected_counts[group]
            ]
        )
    return (
        sorted(candidates, key=lambda row: str(row["map_id"])),
        selected,
    )


def _registered_inputs(
    project_root: Path, specs: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, raw in sorted(specs.items()):
        spec = dict(raw)
        path = project_root / str(spec["path"])
        observed = sha256_file(path) if path.is_file() else None
        matches = observed == str(spec["sha256"])
        rows[name] = {
            "path": str(spec["path"]),
            "registered_sha256": str(spec["sha256"]),
            "observed_sha256": observed,
            "matches": matches,
        }
        if not matches:
            errors.append(f"registered_input_changed:{name}")
    return rows, errors


def audit_da2_supplement_design(
    *,
    config_path: str | Path,
    archive: str | Path,
    map_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Verify the DA2 supplement without starting PP or reading repair outcomes."""

    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    archive = Path(archive).resolve()
    map_root = Path(map_root).resolve()
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_robustaction_expansion_design(config)
    if config.get("schema") != DA2_SUPPLEMENT_SCHEMA:
        raise ValueError("static audit does not reference the DA2 supplement")

    registered_inputs, errors = _registered_inputs(
        project_root, dict(config["predecessor_evidence"])
    )
    archive_sha = sha256_file(archive)
    expected_archive_sha = str(config["map_archive"]["sha256"])
    expected_member_count = int(config["map_archive"]["expected_member_count"])

    all_rows: list[dict[str, Any]] = []
    archive_members: list[str] = []
    extracted_member_count = 0
    with zipfile.ZipFile(archive) as bundle:
        archive_members = sorted(
            name for name in bundle.namelist() if name.lower().endswith(".map")
        )
        for member in archive_members:
            map_id = Path(member).stem
            path = map_root / member
            archive_member_sha = hashlib.sha256(bundle.read(member)).hexdigest()
            if not path.is_file():
                errors.append(f"extracted_missing:{member}")
                continue
            extracted_member_count += 1
            extracted_sha = sha256_file(path)
            if archive_member_sha != extracted_sha:
                errors.append(f"archive_extract_mismatch:{member}")
            metrics = _map_metrics(path)
            _rows, _cols, _grid, passable = _movingai_passable_cells(path)
            component = _largest_four_connected_component(passable)
            all_rows.append(
                {
                    "map_id": map_id,
                    "member": member,
                    "member_sha256": extracted_sha,
                    "free_cell_count": int(metrics["free_cell_count"]),
                    "largest_four_connected_component": len(component),
                    "static_low_degree_cell_ratio": float(
                        metrics["low_degree_cell_ratio"]
                    ),
                    "static_obstacle_ratio": float(metrics["obstacle_ratio"]),
                }
            )

    static = dict(config["static_selection"])
    candidates, selected = select_static_da2_maps(all_rows, static)
    candidate_groups = Counter(str(row["topology_group"]) for row in candidates)
    selected_groups = Counter(str(row["topology_group"]) for row in selected)
    selected_by_id = {str(row["map_id"]): row for row in selected}
    registered_by_id = {
        str(row["id"]): dict(row) for row in config["benchmarks"]
    }
    selected_mismatches: list[str] = []
    for map_id in sorted(set(selected_by_id) | set(registered_by_id)):
        selected_row = selected_by_id.get(map_id)
        registered = registered_by_id.get(map_id)
        if selected_row is None or registered is None:
            selected_mismatches.append(f"selection_mismatch:{map_id}")
            continue
        checks = {
            "member": str(selected_row["member"]) == str(registered["member"]),
            "member_sha256": str(selected_row["member_sha256"])
            == str(registered["member_sha256"]),
            "free_cell_count": int(selected_row["free_cell_count"])
            == int(registered["free_cell_count"]),
            "largest_four_connected_component": int(
                selected_row["largest_four_connected_component"]
            )
            == int(registered["largest_four_connected_component"]),
            "topology_group": str(selected_row["topology_group"])
            == str(registered["topology_group"]),
            "map_family": str(selected_row["map_family"])
            == str(registered["map_family"]),
            "static_low_degree_cell_ratio": abs(
                float(selected_row["static_low_degree_cell_ratio"])
                - float(registered["static_low_degree_cell_ratio"])
            )
            <= 1e-15,
            "static_obstacle_ratio": abs(
                float(selected_row["static_obstacle_ratio"])
                - float(registered["static_obstacle_ratio"])
            )
            <= 1e-15,
        }
        if not all(checks.values()):
            selected_mismatches.append(
                f"registered_metric_mismatch:{map_id}:"
                + ",".join(name for name, passed in checks.items() if not passed)
            )
    errors.extend(selected_mismatches)

    expected_candidate_groups = {
        str(name): int(count)
        for name, count in dict(static["candidate_group_counts"]).items()
    }
    expected_selected_groups = {
        str(name): int(count)
        for name, count in dict(static["selected_group_counts"]).items()
    }
    tracked = set(map(str, static["tracked_da2_map_ids"]))
    all_map_ids = {str(row["map_id"]) for row in all_rows}
    gates = {
        "registered_predecessors_immutable": all(
            bool(row["matches"]) for row in registered_inputs.values()
        ),
        "archive_sha_matches": archive_sha == expected_archive_sha,
        "archive_member_count_exact": len(archive_members) == expected_member_count,
        "all_archive_maps_extracted_and_identical": (
            extracted_member_count == expected_member_count
            and len(all_rows) == expected_member_count
            and not any(error.startswith("archive_extract") for error in errors)
        ),
        "tracked_scenes_present": tracked <= all_map_ids,
        "candidate_count_exact": len(candidates)
        == int(static["candidate_map_count_after_static_filter"]),
        "candidate_topology_balance_exact": dict(candidate_groups)
        == expected_candidate_groups,
        "selection_reproduces_registration": (
            set(selected_by_id) == set(registered_by_id) and not selected_mismatches
        ),
        "selected_topology_balance_exact": dict(selected_groups)
        == expected_selected_groups,
        "minimum_selected_family_count": len(
            {str(row["map_family"]) for row in selected}
        )
        >= int(config["minimum_distinct_map_families"]),
        "selection_is_static_and_outcome_blind": (
            bool(config["selection_boundary"]["outcome_blind"])
            and not bool(static["solver_or_repair_outcomes_read"])
        ),
        "no_solver_or_controller_run": True,
        "no_performance_measurement_run": True,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "static_da2_supplement_audited_before_initial_pp",
        "data_line_id": str(config["data_line_id"]),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "archive_sha256": archive_sha,
            "registered_predecessors": registered_inputs,
        },
        "counts": {
            "archive_map_count": len(archive_members),
            "extracted_map_count": extracted_member_count,
            "candidate_map_count": len(candidates),
            "selected_map_count": len(selected),
            "selected_family_count": len(
                {str(row["map_family"]) for row in selected}
            ),
        },
        "candidate_group_counts": dict(candidate_groups),
        "selected_group_counts": dict(selected_groups),
        "selected_maps": selected,
        "errors": errors,
        "gates": gates,
        "passed": not errors and all(gates.values()),
        "solver_or_controller_run": False,
        "performance_measurements_run": False,
        "next_decision": str(config["next_decision_on_static_audit_pass"]),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "static_audit_report.json", report)
    return report


__all__ = [
    "REPORT_SCHEMA",
    "audit_da2_supplement_design",
    "select_static_da2_maps",
]
