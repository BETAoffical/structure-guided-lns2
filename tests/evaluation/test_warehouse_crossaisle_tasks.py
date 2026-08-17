from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

from experiments import warehouse_crossaisle_tasks as crossaisle


ROOT = Path(__file__).resolve().parents[2]
MAP_PATHS = {
    "warehouse-10-20-10-2-1": ROOT
    / "build/movingai-dev/maps/warehouse-10-20-10-2-1.map",
    "warehouse-10-20-10-2-2": ROOT
    / "build/initlns-movingai-ood-dataset-v1/movingai_ood/maps/warehouse-10-20-10-2-2.map",
    "warehouse-20-40-10-2-1": ROOT
    / "build/movingai-dev/maps/warehouse-20-40-10-2-1.map",
    "warehouse-20-40-10-2-2": ROOT
    / "build/initlns-movingai-ood-dataset-v1/movingai_ood/maps/warehouse-20-40-10-2-2.map",
}


def _map_path(map_id: str) -> Path:
    path = MAP_PATHS[map_id]
    if not path.is_file():
        pytest.skip(f"checksum-pinned generated Warehouse map unavailable: {path}")
    return path


@pytest.mark.parametrize("map_id", tuple(MAP_PATHS))
def test_detects_exact_registered_mirrored_cross_aisles(map_id: str) -> None:
    geometry = crossaisle.detect_cross_aisle_geometry(_map_path(map_id), map_id)
    expected = crossaisle.EXPECTED_GEOMETRY[map_id]

    assert geometry["map_sha256"] == crossaisle.REGISTERED_WAREHOUSE_MAP_SHA256[map_id]
    assert geometry["internal_group_count"] == expected["internal_group_count"]
    assert geometry["aisle_width"] == crossaisle.REGISTERED_AISLE_WIDTHS[map_id]
    assert geometry["excluded_central_group"] == expected["excluded_central_group"]
    assert geometry["chosen_groups"] == expected["chosen_groups"]
    assert geometry["group_count"] == crossaisle.CROSS_AISLE_GROUP_COUNT
    assert geometry["mirrored_about_excluded_center"] is True
    assert geometry["excluded_central_group"] not in geometry["chosen_groups"]
    assert geometry["left_staging_capacity"] == expected["staging_capacity"]
    assert geometry["right_staging_capacity"] == expected["staging_capacity"]
    assert all(
        item["full_width_traversal"]
        and item["shelf_band_above"]
        and item["shelf_band_below"]
        and item["left_staging_capacity"] >= 20
        and item["right_staging_capacity"] >= 20
        for item in geometry["chosen_group_audits"]
    )

    upper, lower = geometry["chosen_groups"][:6], geometry["chosen_groups"][6:]
    center2 = sum(geometry["excluded_central_group"])
    assert all(
        sum(up) + sum(down) == 2 * center2
        for up, down in zip(upper, reversed(lower))
    )


def test_task_specs_freeze_dimensions_and_agent_formula() -> None:
    specs = crossaisle.crossaisle_task_specs()

    assert len(specs) == 4 * 2 * 2 * 2
    assert len({row["task_id"] for row in specs}) == len(specs)
    assert {row["variant"] for row in specs} == set(crossaisle.TASK_VARIANTS)
    assert {row["q"] for row in specs} == set(crossaisle.SUPPORTED_Q)
    for row in specs:
        assert row["agent_count"] == (
            2
            * crossaisle.CROSS_AISLE_GROUP_COUNT
            * crossaisle.REGISTERED_AISLE_WIDTHS[row["map_id"]]
            * row["q"]
        )


@pytest.mark.parametrize("map_id", tuple(MAP_PATHS))
@pytest.mark.parametrize("q", crossaisle.SUPPORTED_Q)
@pytest.mark.parametrize("task_seed", crossaisle.DEFAULT_TASK_SEEDS)
def test_all_registered_pairs_pass_q0(
    map_id: str, q: int, task_seed: int
) -> None:
    path = _map_path(map_id)
    payload = crossaisle.generate_paired_crossaisle_tasks(path, map_id, task_seed, q)
    audit = crossaisle.audit_paired_crossaisle_tasks(path, payload)

    assert audit["passed"], audit["errors"]
    assert audit["errors"] == []
    assert payload["agent_count"] == (
        2
        * crossaisle.CROSS_AISLE_GROUP_COUNT
        * crossaisle.REGISTERED_AISLE_WIDTHS[map_id]
        * q
    )
    structured = payload["variants"][crossaisle.STRUCTURED_VARIANT]
    matched = payload["variants"][crossaisle.MATCHED_VARIANT]
    assert structured["starts"] == matched["starts"]
    assert sorted(structured["goals"]) == sorted(matched["goals"])
    assert len({tuple(cell) for cell in structured["starts"]}) == payload["agent_count"]
    assert len({tuple(cell) for cell in structured["goals"]}) == payload["agent_count"]
    assert all(start != goal for start, goal in zip(structured["starts"], structured["goals"]))
    assert audit["metrics"][crossaisle.STRUCTURED_VARIANT]["reciprocal_ratio"] == 1.0
    assert audit["metrics"][crossaisle.STRUCTURED_VARIANT]["same_group_ratio"] == 1.0
    assert audit["metrics"][crossaisle.MATCHED_VARIANT]["reciprocal_ratio"] == 0.0
    assert audit["metrics"][crossaisle.MATCHED_VARIANT]["same_group_ratio"] == 0.0
    assert audit["metrics"]["paired_distance"]["mean_relative_difference"] <= 0.05
    assert audit["metrics"]["paired_distance"]["p95_relative_difference"] <= 0.10


def test_generation_is_deterministic_and_seed_changes_pairing() -> None:
    map_id = "warehouse-10-20-10-2-1"
    path = _map_path(map_id)
    first = crossaisle.generate_paired_crossaisle_tasks(path, map_id, 419, 16)
    repeated = crossaisle.generate_paired_crossaisle_tasks(path, map_id, 419, 16)
    other_seed = crossaisle.generate_paired_crossaisle_tasks(path, map_id, 463, 16)

    assert first == repeated
    assert first["geometry"] == other_seed["geometry"]
    assert (
        first["variants"][crossaisle.STRUCTURED_VARIANT]["starts"]
        != other_seed["variants"][crossaisle.STRUCTURED_VARIANT]["starts"]
    )
    assert (
        first["variants"][crossaisle.STRUCTURED_VARIANT]["goals"]
        != other_seed["variants"][crossaisle.STRUCTURED_VARIANT]["goals"]
    )


def test_q0_rejects_endpoint_tampering() -> None:
    map_id = "warehouse-10-20-10-2-1"
    path = _map_path(map_id)
    payload = crossaisle.generate_paired_crossaisle_tasks(path, map_id, 419, 16)
    tampered = copy.deepcopy(payload)
    task = tampered["variants"][crossaisle.MATCHED_VARIANT]
    task["goals"][0] = list(task["starts"][0])

    audit = crossaisle.audit_paired_crossaisle_tasks(path, tampered)

    assert audit["passed"] is False
    assert any(
        error in audit["errors"]
        for error in (
            f"{crossaisle.MATCHED_VARIANT}:endpoints_not_unique",
            f"{crossaisle.MATCHED_VARIANT}:fixed_point",
        )
    )


@pytest.mark.parametrize("bad_cell", ([9999, 9999], [12, 12], [13]))
def test_q0_returns_errors_for_malformed_or_unregistered_endpoint(
    bad_cell: list[int],
) -> None:
    map_id = "warehouse-10-20-10-2-1"
    path = _map_path(map_id)
    payload = crossaisle.generate_paired_crossaisle_tasks(path, map_id, 419, 16)
    payload["variants"][crossaisle.MATCHED_VARIANT]["starts"][0] = bad_cell

    audit = crossaisle.audit_paired_crossaisle_tasks(path, payload)

    assert audit["passed"] is False
    assert (
        f"{crossaisle.MATCHED_VARIANT}:endpoint_outside_registered_lanes"
        in audit["errors"]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("task_id", "forged-task"),
        ("task_seed", 999),
        ("q", 20),
        ("agent_count", 1),
    ),
)
def test_q0_rejects_variant_identity_tampering(
    field: str, replacement: object
) -> None:
    map_id = "warehouse-10-20-10-2-1"
    path = _map_path(map_id)
    payload = crossaisle.generate_paired_crossaisle_tasks(path, map_id, 419, 16)
    tampered = copy.deepcopy(payload)
    tampered["variants"][crossaisle.STRUCTURED_VARIANT][field] = replacement

    audit = crossaisle.audit_paired_crossaisle_tasks(path, tampered)

    assert audit["passed"] is False
    assert f"{crossaisle.STRUCTURED_VARIANT}:identity_mismatch" in audit["errors"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("pairing_semantics", "weaker-contract"), ("causal_control_claim", True)),
)
def test_q0_rejects_pairing_contract_tampering(
    field: str, replacement: object
) -> None:
    map_id = "warehouse-10-20-10-2-1"
    path = _map_path(map_id)
    payload = crossaisle.generate_paired_crossaisle_tasks(path, map_id, 419, 16)
    payload[field] = replacement

    audit = crossaisle.audit_paired_crossaisle_tasks(path, payload)

    assert audit["passed"] is False
    assert "pairing_semantics_mismatch" in audit["errors"]


def test_rejects_wrong_map_identity_and_unsupported_q() -> None:
    path = _map_path("warehouse-10-20-10-2-1")
    with pytest.raises(ValueError, match="checksum mismatch"):
        crossaisle.detect_cross_aisle_geometry(path, "warehouse-10-20-10-2-2")
    with pytest.raises(ValueError, match="unsupported q"):
        crossaisle.generate_paired_crossaisle_tasks(
            path, "warehouse-10-20-10-2-1", 419, 24
        )
    with pytest.raises(ValueError, match="unsupported q"):
        crossaisle.crossaisle_task_specs(q_values=(16, 24))


def test_helper_remains_geometry_only() -> None:
    source = inspect.getsource(crossaisle)
    assert "lns2_env" not in source
    assert "run_closed_loop" not in source
    assert "reset(" not in source

    assert crossaisle.RESET_QUALIFICATION_GATES == {
        crossaisle.STRUCTURED_VARIANT: {
            "initial_feasible": False,
            "minimum_initial_conflicts": 16,
            "minimum_active_conflict_agents": 32,
            "minimum_largest_component": 16,
        },
        crossaisle.MATCHED_VARIANT: {
            "initial_feasible": False,
            "minimum_initial_conflicts": 1,
        },
    }
