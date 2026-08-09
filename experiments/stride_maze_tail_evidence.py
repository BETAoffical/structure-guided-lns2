from __future__ import annotations

import collections
import shutil
import statistics
import zipfile
from pathlib import Path
from typing import Any

from experiments._common import registered_input, sha256_file
from experiments.balanced_wall_clock import (
    SPLIT,
    _derived_endpoint_seed,
    _derived_endpoints,
    _four_neighbor_distances,
    _largest_four_connected_component,
    _map_metrics,
    _movingai_passable_cells,
    _write_derived_scenario,
    _write_jsonl_atomic,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)


SOURCE_SCHEMA = "lns2.stride.maze_tail_evidence_source.v1"
CONFIG_SCHEMA = "lns2.stride.maze_tail_evidence_preflight_config.v1"
REPORT_SCHEMA = "lns2.stride.maze_tail_evidence_preflight_report.v1"
EXPERIMENT_ID = "stride-maze-tail-evidence-preflight-v1"
PARENT_COMMIT = "13a7b705e56c1b00767500be21c3cc99d37160af"
EXPECTED_MAPS = (
    "maze-32-32-4",
    "maze-128-128-1",
    "maze-128-128-2",
    "maze-128-128-10",
)
EXPECTED_TASK_SEEDS = (811, 853)
EXPECTED_SOLVER_SEEDS = (17, 29, 43)
EXPECTED_VARIANTS = ("uniform_random", "opposite_exchange")


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="Maze tail evidence")


def _validate_source(source: dict[str, Any]) -> None:
    if (
        source.get("schema") != SOURCE_SCHEMA
        or source.get("scientific_status")
        != "preregistered_before_reset_only_qualification"
        or source.get("experiment_id") != EXPERIMENT_ID
        or source.get("pre_registration_parent_commit") != PARENT_COMMIT
        or source.get("dataset_revision")
        != "stride-maze-tail-evidence-candidates-v1"
        or source.get("task_semantics")
        != "project_derived_static_od_not_official_movingai_scenarios"
        or int(source.get("master_seed", -1)) != 20260809
        or tuple(map(int, source.get("task_seeds") or ()))
        != EXPECTED_TASK_SEEDS
        or tuple(map(str, source.get("task_variants") or ()))
        != EXPECTED_VARIANTS
        or tuple(map(int, source.get("solver_seeds") or ()))
        != EXPECTED_SOLVER_SEEDS
        or int(source.get("expected_map_count", -1)) != 4
        or int(source.get("expected_task_count", -1)) != 64
        or int(source.get("expected_reset_count", -1)) != 192
    ):
        raise ValueError("Maze tail evidence source identity changed")
    benchmarks = [dict(value) for value in source.get("benchmarks") or ()]
    if tuple(str(value.get("id")) for value in benchmarks) != EXPECTED_MAPS:
        raise ValueError("Maze tail evidence map registration changed")
    expected_ladders = {
        "maze-32-32-4": (80, 120, 160, 200),
        "maze-128-128-1": (60, 80, 100, 120),
        "maze-128-128-2": (300, 400, 500, 600),
        "maze-128-128-10": (400, 600, 800, 1000),
    }
    for row in benchmarks:
        map_id = str(row["id"])
        archive = dict(row.get("map_archive") or {})
        if (
            str(row.get("layout_family")) != "maze"
            or tuple(map(int, row.get("agent_counts") or ()))
            != expected_ladders[map_id]
            or not str(archive.get("url", "")).endswith(f"/{map_id}.map.zip")
            or str(archive.get("member")) != f"{map_id}.map"
            or len(str(archive.get("sha256", ""))) != 64
            or len(str(archive.get("member_sha256", ""))) != 64
        ):
            raise ValueError(f"Maze tail evidence benchmark changed: {map_id}")
    if dict(source.get("outcome_boundary") or {}) != {
        "dataset_generation_uses_controller_outcomes": False,
        "dataset_generation_uses_repair_outcomes": False,
        "task_or_load_replacement_after_outcomes": False,
        "wide_corridor_zero_conflict_control_is_retained": True,
    }:
        raise ValueError("Maze tail evidence source outcome boundary changed")


def load_maze_tail_evidence_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_dataset_materialization_and_reset_only_qualification"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != PARENT_COMMIT
        or config.get("dataset")
        != "build/stride-maze-tail-evidence-candidate-dataset-v1"
        or config.get("split") != SPLIT
        or config.get("runtime")
        != "configs/stride_maze_tail_evidence_preflight_runtime_v1.json"
        or config.get("controller_bundle")
        != "artifacts/initlns-closed-loop-controller-v2"
        or tuple(map(int, config.get("solver_seeds") or ()))
        != EXPECTED_SOLVER_SEEDS
        or int(config.get("expected_map_count", -1)) != 4
        or int(config.get("expected_task_count", -1)) != 64
        or int(config.get("expected_reset_count", -1)) != 192
    ):
        raise ValueError("Maze tail evidence preflight identity changed")
    ladders = [dict(value) for value in config.get("candidate_ladders") or ()]
    if tuple(str(value.get("map_id")) for value in ladders) != EXPECTED_MAPS:
        raise ValueError("Maze tail evidence preflight ladders changed")
    rule = dict(config.get("selection_rule") or {})
    if (
        list(rule.get("group_by") or ())
        != ["map_id", "task_variant_family", "conflict_band"]
        or tuple(map(str, rule.get("task_variants") or ()))
        != EXPECTED_VARIANTS
        or int(rule.get("require_each_solver_seed_at_least_conflicts", -1)) != 16
        or rule.get("require_all_solver_seeds_ok_complete_and_consistent") is not True
        or int(rule.get("maximum_tasks_per_map_variant_band", -1)) != 1
        or list(rule.get("tie_break") or ())
        != [
            "absolute_target_distance",
            "lower_agent_count",
            "lower_task_seed",
            "task_id",
        ]
        or int(rule.get("minimum_selected_map_count", -1)) != 3
        or int(rule.get("minimum_selected_task_count", -1)) != 8
        or int(rule.get("minimum_map_coverage_per_task_variant", -1)) != 2
        or int(rule.get("minimum_selected_task_seed_count", -1)) != 2
        or rule.get("require_both_conflict_bands") is not True
        or rule.get("retain_unqualified_maps_as_registered_negative_controls")
        is not True
    ):
        raise ValueError("Maze tail evidence selection rule changed")
    bands = [dict(value) for value in rule.get("conflict_bands") or ()]
    if bands != [
        {
            "id": "moderate",
            "minimum_mean_initial_conflicts": 16.0,
            "maximum_mean_initial_conflicts": 99.999999,
            "target_mean_initial_conflicts": 50.0,
        },
        {
            "id": "high",
            "minimum_mean_initial_conflicts": 100.0,
            "maximum_mean_initial_conflicts": 400.0,
            "target_mean_initial_conflicts": 200.0,
        },
    ]:
        raise ValueError("Maze tail evidence conflict bands changed")
    boundary = dict(config.get("outcome_boundary") or {})
    if (
        set(boundary.get("allowed_observed_fields") or ())
        != {
            "reset_status",
            "initial_complete",
            "initial_conflicts",
            "initial_complexity",
            "initial_state_consistent",
            "state_fingerprint",
        }
        or boundary.get("candidate_repair_outcomes_read") is not False
        or boundary.get("controller_outcomes_read") is not False
        or boundary.get("ttf_outcomes_read") is not False
        or boundary.get("fresh_map_generalization_claim") is not False
        or boundary.get("tail_incidence_claim") is not False
        or boundary.get("training_allowed") is not False
    ):
        raise ValueError("Maze tail evidence preflight outcome boundary changed")
    if set(config.get("forbidden_selection_inputs") or ()) != {
        "conflicts_after",
        "repair_iterations",
        "repair_runtime",
        "pp_replan_seconds",
        "selected_candidate",
        "controller_action",
        "controller_ttf",
        "future_trajectory",
    }:
        raise ValueError("Maze tail evidence forbidden inputs changed")
    expected_inputs = {
        "source_config",
        "runtime_config",
        "controller_manifest",
        "known_tail_report",
        "closurepool_temporal_report",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("Maze tail evidence input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    source = _read_json(inputs["source_config"])
    _validate_source(source)
    runtime = _read_json(inputs["runtime_config"])
    if (
        runtime.get("experiment_runtime_id") != EXPERIMENT_ID
        or runtime.get("formal") is not False
        or runtime.get("solver_seeds") != list(EXPECTED_SOLVER_SEEDS)
        or runtime.get("dataset_design", {}).get("map_count") != 4
        or runtime.get("dataset_design", {}).get("instance_count") != 64
        or runtime.get("qualification", {}).get("mode")
        != "maze_tail_evidence_reset_only_v1"
        or runtime.get("qualification", {}).get("enforce_registered_thresholds")
        is not True
    ):
        raise ValueError("Maze tail evidence runtime changed")
    known_tail = _read_json(inputs["known_tail_report"])
    if (
        known_tail.get("integrity_passed") is not True
        or known_tail.get("regression_passed") is not False
    ):
        raise ValueError("Maze tail evidence known-tail premise changed")
    temporal = _read_json(inputs["closurepool_temporal_report"])
    if (
        temporal.get("integrity_passed") is not True
        or temporal.get("decision")
        != "human_cross_case_review_required_no_rule_freeze"
        or temporal.get("claim_boundary", {}).get("model_training_allowed")
        is not False
    ):
        raise ValueError("Maze tail evidence temporal premise changed")
    return path, root, config, inputs, source


def prepare_maze_tail_evidence_dataset(
    fetched: str | Path,
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    _path, _root, config, _inputs, source = load_maze_tail_evidence_config(
        config_path
    )
    fetched_root = Path(fetched).resolve()
    output_root = Path(output).resolve()
    expected_output = (_root / str(config["dataset"])).resolve()
    if output_root != expected_output:
        raise ValueError("Maze tail evidence dataset output changed")
    configuration_fingerprint = _fingerprint(source)
    summary_path = output_root / "dataset_summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("configuration_fingerprint") != configuration_fingerprint:
            raise ValueError("Maze tail dataset belongs to another source config")
        return existing
    if output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("Maze tail dataset output is non-empty without a summary")

    split_root = output_root / SPLIT
    manifest: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    map_ids: set[str] = set()
    master_seed = int(source["master_seed"])
    for raw_case in source["benchmarks"]:
        case = dict(raw_case)
        map_id = str(case["id"])
        if map_id in map_ids:
            raise ValueError(f"Maze tail source repeats map: {map_id}")
        map_ids.add(map_id)
        archive_spec = dict(case["map_archive"])
        archive_path = fetched_root / "_archives" / Path(
            str(archive_spec["url"])
        ).name
        if not archive_path.is_file():
            raise ValueError(f"Maze tail map archive is missing: {archive_path}")
        archive_sha = sha256_file(archive_path)
        if archive_sha != str(archive_spec["sha256"]):
            raise ValueError(f"Maze tail archive SHA mismatch: {map_id}")
        map_path = split_root / "maps" / f"{map_id}.map"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        partial = map_path.with_name(map_path.name + ".partial")
        with zipfile.ZipFile(archive_path) as bundle:
            try:
                with bundle.open(str(archive_spec["member"])) as source_stream:
                    with partial.open("wb") as output_stream:
                        shutil.copyfileobj(source_stream, output_stream)
            except KeyError as error:
                partial.unlink(missing_ok=True)
                raise ValueError(
                    f"Maze tail archive member is missing: {map_id}"
                ) from error
        partial.replace(map_path)
        map_sha = sha256_file(map_path)
        if map_sha != str(archive_spec["member_sha256"]):
            raise ValueError(f"Maze tail map SHA mismatch: {map_id}")
        rows, cols, _grid, passable = _movingai_passable_cells(map_path)
        component = _largest_four_connected_component(passable)
        metrics = _map_metrics(map_path)
        metadata_path = split_root / "maps" / f"{map_id}.json"
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "benchmark_id": map_id,
                "source": "MovingAI 2D grid benchmark",
                "source_page": str(source["source"]),
                "source_archive_url": str(archive_spec["url"]),
                "source_archive_sha256": archive_sha,
                "source_member": str(archive_spec["member"]),
                "map_sha256": map_sha,
                "largest_four_connected_component": len(component),
                "topology_metrics": metrics,
            },
        )
        archives.append(
            {
                "map_id": map_id,
                "archive": archive_path.name,
                "archive_sha256": archive_sha,
                "member_sha256": map_sha,
            }
        )
        for task_seed in map(int, source["task_seeds"]):
            for variant in map(str, source["task_variants"]):
                for agent_count in map(int, case["agent_counts"]):
                    endpoint_seed = _derived_endpoint_seed(
                        master_seed, map_id, task_seed, variant, agent_count
                    )
                    starts, goals = _derived_endpoints(
                        component, agent_count, variant, endpoint_seed
                    )
                    distances = _four_neighbor_distances(passable, starts, goals)
                    task_id = (
                        f"{map_id}__derived_{variant}__task_seed_{task_seed:04d}"
                        f"__agents_{agent_count:04d}"
                    )
                    scenario_path = split_root / "scenarios" / f"{task_id}.scen"
                    _write_derived_scenario(
                        scenario_path,
                        map_path.name,
                        rows,
                        cols,
                        starts,
                        goals,
                        distances,
                    )
                    task_path = split_root / "tasks" / f"{task_id}.json"
                    task_payload = {
                        "schema_version": 1,
                        "task_semantics_version": 1,
                        "task_semantics": (
                            "project-derived static MAPF OD on an official "
                            "MovingAI 2D benchmark map"
                        ),
                        "benchmark_id": map_id,
                        "source_map_sha256": map_sha,
                        "od_variant": variant,
                        "task_seed": task_seed,
                        "endpoint_seed": endpoint_seed,
                        "agent_count": agent_count,
                        "agent_density_largest_component": agent_count / len(component),
                        "unique_starts": len(set(starts)) == agent_count,
                        "unique_goals": len(set(goals)) == agent_count,
                        "fixed_point_count": sum(
                            start == goal for start, goal in zip(starts, goals)
                        ),
                        "distance_metric": "four_neighbor_unit",
                        "minimum_shortest_distance": min(distances),
                        "maximum_shortest_distance": max(distances),
                        "mean_shortest_distance": statistics.fmean(distances),
                        "scenario_sha256": sha256_file(scenario_path),
                    }
                    _write_json(task_path, task_payload)
                    manifest.append(
                        {
                            "split": SPLIT,
                            "source_group": "movingai",
                            "instance_origin": "movingai_map_project_derived_od",
                            "map_id": map_id,
                            "task_id": task_id,
                            "map_file": f"maps/{map_path.name}",
                            "scenario_file": f"scenarios/{scenario_path.name}",
                            "map_metadata_file": f"maps/{metadata_path.name}",
                            "task_file": f"tasks/{task_path.name}",
                            "layout_mode": "maze",
                            "layout_variant": map_id,
                            "scenario_type": f"movingai_map_derived_{variant}",
                            "task_variant": (
                                f"{variant}_seed_{task_seed}_agents_{agent_count}"
                            ),
                            "task_variant_family": variant,
                            "task_seed": task_seed,
                            "agent_count": agent_count,
                            "topology_metrics": metrics,
                            "dominant_flow_ratio": 0.0,
                            "hotspot_skew": 0.0,
                            "required_bottleneck_crossing_ratio": 0.0,
                            "mean_shortest_distance": task_payload[
                                "mean_shortest_distance"
                            ],
                        }
                    )

    if len(map_ids) != 4 or len(manifest) != 64:
        raise ValueError("Maze tail dataset dimensions differ from registration")
    task_ids = [str(row["task_id"]) for row in manifest]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Maze tail dataset repeats task IDs")
    manifest.sort(key=lambda row: str(row["task_id"]))
    manifest_path = split_root / "manifest.jsonl"
    _write_jsonl_atomic(manifest_path, manifest)
    summary = {
        "schema_version": 1,
        "dataset_revision": str(source["dataset_revision"]),
        "configuration_fingerprint": configuration_fingerprint,
        "source_config_sha256": sha256_file(_inputs["source_config"]),
        "source": "MovingAI maps with project-derived static MAPF OD tasks",
        "task_semantics": "derived_not_official_mapf_scenarios",
        "archive_registration": archives,
        "manifest_sha256": sha256_file(manifest_path),
        "splits": {
            SPLIT: {
                "map_count": len(map_ids),
                "instance_count": len(manifest),
                "source_counts": {"movingai": len(manifest)},
                "layout_counts": {"maze": len(manifest)},
            }
        },
    }
    _write_json(summary_path, summary)
    return summary


def run_maze_tail_evidence_qualification(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    _path, root, config, inputs, source = load_maze_tail_evidence_config(
        config_path
    )
    dataset = (root / str(config["dataset"])).resolve()
    summary = _read_json(dataset / "dataset_summary.json")
    if summary.get("configuration_fingerprint") != _fingerprint(source):
        raise ValueError("Maze tail qualification dataset source changed")
    manifest = _read_jsonl(dataset / SPLIT / "manifest.jsonl")
    if len(manifest) != 64:
        raise ValueError("Maze tail qualification dataset is incomplete")
    return run_closed_loop_collection(
        dataset,
        inputs["runtime_config"],
        Path(output).resolve(),
        phase="qualify",
        workers=1,
        resume=resume,
        dry_run=dry_run,
        controller="v2-full",
        controller_bundle=(root / str(config["controller_bundle"])).resolve(),
        feature_backend="native",
        controller_runtime="optimized",
        verification_profile="deployment",
        stopping_rule="run-to-completion",
    )


def _variant_family(row: dict[str, Any]) -> str:
    direct = str(row.get("task_variant_family", ""))
    if direct in EXPECTED_VARIANTS:
        return direct
    scenario = str(row.get("scenario_type", ""))
    for variant in EXPECTED_VARIANTS:
        if scenario.endswith(variant):
            return variant
    raise ValueError(f"Maze tail task has no registered variant: {row.get('task_id')}")


def select_maze_tail_tasks(
    manifest_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], set[str]]:
    source = {str(row["task_id"]): dict(row) for row in manifest_rows}
    if len(source) != len(manifest_rows):
        raise ValueError("Maze tail manifest repeats task IDs")
    solver_seeds = tuple(map(int, config["solver_seeds"]))
    expected = {(task_id, seed) for task_id in source for seed in solver_seeds}
    forbidden_names = set(map(str, config["forbidden_selection_inputs"]))
    forbidden: set[str] = set()
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    for raw in qualification_rows:
        row = dict(raw)
        forbidden.update(forbidden_names & set(row))
        key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if key in indexed:
            errors.append(f"duplicate:{key[0]}:{key[1]}")
        indexed[key] = row
        task = source.get(key[0])
        if key not in expected or task is None:
            errors.append(f"unexpected:{key[0]}:{key[1]}")
        elif (
            str(row.get("map_id")) != str(task["map_id"])
            or int(row.get("agent_count", -1)) != int(task["agent_count"])
        ):
            errors.append(f"identity:{key[0]}:{key[1]}")
    if set(indexed) != expected:
        errors.append("incomplete_reset_product")

    threshold = int(
        config["selection_rule"]["require_each_solver_seed_at_least_conflicts"]
    )
    bands = [dict(value) for value in config["selection_rule"]["conflict_bands"]]
    summaries: list[dict[str, Any]] = []
    candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for task_id, task in sorted(source.items()):
        results = [indexed.get((task_id, seed), {}) for seed in solver_seeds]
        valid = all(
            result.get("status") == "ok"
            and result.get("initial_complete") is True
            and result.get("initial_state_consistent", True) is True
            and bool(result.get("state_fingerprint"))
            for result in results
        )
        conflicts = [int(result.get("initial_conflicts", -1)) for result in results]
        mean_conflicts = statistics.fmean(conflicts) if valid else None
        eligible_floor = valid and min(conflicts) >= threshold
        band_id = None
        target = None
        if eligible_floor and mean_conflicts is not None:
            for band in bands:
                if (
                    float(band["minimum_mean_initial_conflicts"])
                    <= mean_conflicts
                    <= float(band["maximum_mean_initial_conflicts"])
                ):
                    band_id = str(band["id"])
                    target = float(band["target_mean_initial_conflicts"])
                    break
        variant = _variant_family(task)
        summary = {
            "map_id": str(task["map_id"]),
            "task_id": task_id,
            "task_variant_family": variant,
            "task_seed": int(task["task_seed"]),
            "agent_count": int(task["agent_count"]),
            "reset_count": len(results),
            "all_resets_valid": valid,
            "minimum_initial_conflicts": min(conflicts) if valid else None,
            "maximum_initial_conflicts": max(conflicts) if valid else None,
            "mean_initial_conflicts": mean_conflicts,
            "each_seed_meets_activation_floor": eligible_floor,
            "conflict_band": band_id,
            "state_fingerprints": [
                str(result.get("state_fingerprint", "")) for result in results
            ],
            "eligible": band_id is not None,
        }
        summaries.append(summary)
        if band_id is not None and target is not None:
            summary["absolute_target_distance"] = abs(mean_conflicts - target)
            candidates[(summary["map_id"], variant, band_id)].append(summary)

    selected: list[dict[str, Any]] = []
    for key in sorted(candidates):
        winner = min(
            candidates[key],
            key=lambda row: (
                float(row["absolute_target_distance"]),
                int(row["agent_count"]),
                int(row["task_seed"]),
                str(row["task_id"]),
            ),
        )
        selected.append(dict(winner))
    selected.sort(
        key=lambda row: (
            str(row["map_id"]),
            str(row["task_variant_family"]),
            str(row["conflict_band"]),
            str(row["task_id"]),
        )
    )
    return summaries, selected, errors, forbidden


def analyze_maze_tail_evidence_qualification(
    config_path: str | Path,
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    path, root, config, _inputs, source = load_maze_tail_evidence_config(
        config_path
    )
    dataset = Path(dataset).resolve()
    qualification = Path(qualification).resolve()
    output = Path(output).resolve()
    if dataset != (root / str(config["dataset"])).resolve():
        raise ValueError("Maze tail analysis dataset changed")
    manifest_path = dataset / SPLIT / "manifest.jsonl"
    summary_path = dataset / "dataset_summary.json"
    qualification_manifest_path = qualification / "qualification_manifest.jsonl"
    qualification_report_path = qualification / "qualification_report.json"
    run_config_path = qualification / "run_config.json"
    manifest_rows = _read_jsonl(manifest_path)
    qualification_rows = _read_jsonl(qualification_manifest_path)
    summaries, selected, errors, forbidden = select_maze_tail_tasks(
        manifest_rows, qualification_rows, config
    )
    output.mkdir(parents=True, exist_ok=True)
    selected_maps = {str(row["map_id"]) for row in selected}
    selected_seeds = {int(row["task_seed"]) for row in selected}
    selected_bands = {str(row["conflict_band"]) for row in selected}
    variant_map_coverage = {
        variant: len(
            {
                str(row["map_id"])
                for row in selected
                if row["task_variant_family"] == variant
            }
        )
        for variant in EXPECTED_VARIANTS
    }
    registered_maps = {str(row["map_id"]) for row in manifest_rows}
    negative_controls = sorted(registered_maps - selected_maps)
    rule = dict(config["selection_rule"])
    gates = {
        "registered_dataset_dimensions": len(manifest_rows) == 64
        and registered_maps == set(EXPECTED_MAPS),
        "complete_reset_product": len(qualification_rows) == 192 and not errors,
        "forbidden_outcomes_absent": not forbidden,
        "minimum_selected_map_count": len(selected_maps)
        >= int(rule["minimum_selected_map_count"]),
        "minimum_selected_task_count": len(selected)
        >= int(rule["minimum_selected_task_count"]),
        "minimum_variant_map_coverage": all(
            count >= int(rule["minimum_map_coverage_per_task_variant"])
            for count in variant_map_coverage.values()
        ),
        "minimum_selected_task_seed_count": len(selected_seeds)
        >= int(rule["minimum_selected_task_seed_count"]),
        "both_conflict_bands_present": selected_bands == {"moderate", "high"},
        "all_registered_maps_retained_through_reset": registered_maps
        == set(EXPECTED_MAPS),
    }
    passed = all(gates.values())
    _write_jsonl_atomic(output / "reset_task_summaries.jsonl", summaries)
    _write_jsonl_atomic(output / "selected_cohort.jsonl", selected)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_reset_only_outcome_blind_preflight",
        "experiment_id": EXPERIMENT_ID,
        "passed": passed,
        "training_allowed": False,
        "controller_outcomes_read": False,
        "candidate_repair_outcomes_read": False,
        "ttf_outcomes_read": False,
        "fresh_map_generalization_claim": False,
        "tail_incidence_claim": False,
        "registered_map_count": len(registered_maps),
        "registered_task_count": len(manifest_rows),
        "reset_count": len(qualification_rows),
        "selected_map_count": len(selected_maps),
        "selected_task_count": len(selected),
        "selected_solver_key_count": len(selected) * len(EXPECTED_SOLVER_SEEDS),
        "selected_maps": sorted(selected_maps),
        "selected_task_seeds": sorted(selected_seeds),
        "selected_conflict_bands": sorted(selected_bands),
        "variant_map_coverage": variant_map_coverage,
        "registered_negative_controls_without_selected_tasks": negative_controls,
        "gates": gates,
        "errors": errors,
        "forbidden_fields_found": sorted(forbidden),
        "next_step": config[
            "next_step_on_pass" if passed else "next_step_on_failure"
        ],
        "inputs": {
            "config_sha256": sha256_file(path),
            "source_config_sha256": sha256_file(
                root / str(config["inputs"]["source_config"]["path"])
            ),
            "source_fingerprint": _fingerprint(source),
            "dataset_summary_sha256": sha256_file(summary_path),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "qualification_manifest_sha256": sha256_file(
                qualification_manifest_path
            ),
            "qualification_report_sha256": sha256_file(
                qualification_report_path
            ),
            "run_config_sha256": sha256_file(run_config_path),
        },
        "artifacts": {
            "reset_task_summaries": "reset_task_summaries.jsonl",
            "reset_task_summaries_sha256": sha256_file(
                output / "reset_task_summaries.jsonl"
            ),
            "selected_cohort": "selected_cohort.jsonl",
            "selected_cohort_sha256": sha256_file(
                output / "selected_cohort.jsonl"
            ),
        },
    }
    _write_json(output / "maze_tail_evidence_preflight_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "SOURCE_SCHEMA",
    "analyze_maze_tail_evidence_qualification",
    "load_maze_tail_evidence_config",
    "prepare_maze_tail_evidence_dataset",
    "run_maze_tail_evidence_qualification",
    "select_maze_tail_tasks",
]
