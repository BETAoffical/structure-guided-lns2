from __future__ import annotations

from pathlib import Path

import experiments.stride_hierarchical_ch_compact_flow_balanced_source_v1 as balanced
import experiments.stride_hierarchical_ch_compact_flow_source_v1 as base
from tests.evaluation.test_stride_hierarchical_ch_compact_flow_source_v1 import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    _write_qualification,
    compact_flow_fixture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_balanced_source_v1.json"
)


def _enable_balanced_profile(fixture: dict) -> dict:
    ctx = fixture["ctx"]
    ctx["source_profile"] = base._source_profile(
        {
            "schema": balanced.CONFIG_SCHEMA,
            "experiment_id": balanced.EXPERIMENT_ID,
            "qualification_gate_profile": balanced.QUALIFICATION_GATE_PROFILE,
        }
    )
    ctx["source_config"]["qualification_gate_profile"] = (
        balanced.QUALIFICATION_GATE_PROFILE
    )
    return ctx


def _rename_development_map_to_den207d(fixture: dict) -> None:
    ctx = fixture["ctx"]
    old_map_id = ctx["map_splits"]["development"][-1]
    ctx["map_splits"]["development"][-1] = "den207d"
    ctx["per_map_loads"]["den207d"] = ctx["per_map_loads"].pop(old_map_id)
    ctx["per_map_task_seeds"]["den207d"] = ctx["per_map_task_seeds"].pop(
        old_map_id
    )
    ctx["map_capacity_strata"]["den207d"] = ctx["map_capacity_strata"].pop(
        old_map_id
    )
    for row in fixture["manifests"]["development"]:
        if row["map_id"] == old_map_id:
            row["map_id"] = "den207d"


def _qualification_with_zero_uniform_maps(
    fixture: dict, output: Path, zero_maps: set[str]
) -> tuple[set[tuple[str, int]], list[dict]]:
    split = "development"
    jobs = {
        (str(row["task_id"]), solver_seed)
        for row in fixture["manifests"][split]
        for solver_seed in base.SOLVER_SEEDS
    }
    _write_qualification(
        fixture, output=output, split=split, jobs=jobs, conflicts=16
    )
    task_by_id = {
        str(row["task_id"]): row for row in fixture["manifests"][split]
    }
    rows = _read_jsonl(output / "qualification_manifest.jsonl")
    for row in rows:
        task = task_by_id[str(row["task_id"])]
        if task["map_id"] in zero_maps and task["od_variant"] == "uniform_random":
            row["initial_conflicts"] = 0
            row["initial_feasible"] = True
    _write_jsonl(output / "qualification_manifest.jsonl", rows)
    return jobs, rows


def test_repository_config_has_independent_balanced_identity() -> None:
    ctx = balanced.load_registered_source_context(CONFIG)
    assert ctx["source_profile"]["experiment_id"] == balanced.EXPERIMENT_ID
    assert ctx["source_profile"]["gate_profile"] == (
        balanced.QUALIFICATION_GATE_PROFILE
    )
    assert ctx["source_profile"]["gate_profile_sha256"]


def test_plan_and_run_fingerprint_inputs_record_balanced_gate_profile(
    compact_flow_fixture: dict,
) -> None:
    ctx = _enable_balanced_profile(compact_flow_fixture)
    trust_path = compact_flow_fixture["output"] / "materialization_trust_report.json"
    trust = _read_json(trust_path)
    trust["schema"] = balanced.MATERIALIZATION_SCHEMA
    trust["experiment_id"] = balanced.EXPERIMENT_ID
    _write_json(trust_path, trust)

    report = balanced.plan_source_collection(
        "ignored.json", compact_flow_fixture["output"]
    )
    assert report["experiment_id"] == balanced.EXPERIMENT_ID
    assert report["qualification_gate_profile"] == balanced.QUALIFICATION_GATE_PROFILE
    assert report["qualification_gate_profile_sha256"] == ctx["source_profile"][
        "gate_profile_sha256"
    ]
    assert report["plan_fingerprint_payload"][
        "qualification_gate_profile_sha256"
    ] == ctx["source_profile"]["gate_profile_sha256"]
    assert report["plan_fingerprint"]

    schedule = _read_jsonl(
        compact_flow_fixture["output"] / "source_schedule.jsonl"
    )
    assert {row["qualification_gate_profile_id"] for row in schedule} == {
        "split_level_balanced_od_coverage_v1"
    }
    collection_config = _read_json(
        compact_flow_fixture["output"]
        / "collection_configs"
        / "development.json"
    )
    qualification = collection_config["qualification"]
    assert qualification["source_gate_profile"] == balanced.QUALIFICATION_GATE_PROFILE
    assert qualification["source_gate_profile_sha256"] == ctx["source_profile"][
        "gate_profile_sha256"
    ]


def test_den207d_zero_uniform_passes_when_split_od_coverage_is_sufficient(
    compact_flow_fixture: dict,
) -> None:
    ctx = _enable_balanced_profile(compact_flow_fixture)
    _rename_development_map_to_den207d(compact_flow_fixture)
    output = compact_flow_fixture["output"] / "balanced-pass"
    jobs, _rows = _qualification_with_zero_uniform_maps(
        compact_flow_fixture, output, {"den207d"}
    )
    audit = base._qualification_audit(
        ctx,
        "development",
        output,
        jobs,
        compact_flow_fixture["manifests"]["development"],
    )
    assert audit is not None
    assert audit["observed"]["nonzero_resets_by_map_od_variant"][
        "den207d/uniform_random"
    ] == 0
    assert audit["observed"]["nonzero_resets_by_od_variant"][
        "uniform_random"
    ] == 84
    assert audit["observed"]["active_map_counts_by_od_variant"][
        "uniform_random"
    ] == 7
    assert audit["gates"]["minimum_nonzero_resets_per_od_variant"] is True
    assert audit["gates"]["minimum_active_maps_per_od_variant"] is True
    assert "minimum_nonzero_resets_per_map_od_variant" not in audit["gates"]
    assert audit["passed"] is True


def test_balanced_split_od_coverage_fails_below_count_and_map_thresholds(
    compact_flow_fixture: dict,
) -> None:
    ctx = _enable_balanced_profile(compact_flow_fixture)
    _rename_development_map_to_den207d(compact_flow_fixture)
    maps = set(ctx["map_splits"]["development"][:4]) | {"den207d"}
    output = compact_flow_fixture["output"] / "balanced-fail"
    jobs, _rows = _qualification_with_zero_uniform_maps(
        compact_flow_fixture, output, maps
    )
    audit = base._qualification_audit(
        ctx,
        "development",
        output,
        jobs,
        compact_flow_fixture["manifests"]["development"],
    )
    assert audit is not None
    assert audit["observed"]["nonzero_resets_by_od_variant"][
        "uniform_random"
    ] == 36
    assert audit["observed"]["active_map_counts_by_od_variant"][
        "uniform_random"
    ] == 3
    assert audit["gates"]["minimum_nonzero_resets_per_od_variant"] is False
    assert audit["gates"]["minimum_active_maps_per_od_variant"] is False
    assert audit["gates"]["minimum_nonzero_resets_per_map"] is True
    assert audit["gates"]["minimum_nonzero_high_load_resets_per_map"] is True
    assert audit["passed"] is False
