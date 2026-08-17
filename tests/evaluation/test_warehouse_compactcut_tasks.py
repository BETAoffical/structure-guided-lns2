from __future__ import annotations

import copy
import inspect

import pytest

from experiments import warehouse_compactcut_tasks as compactcut


@pytest.fixture(scope="module")
def map_bank():
    return compactcut.generate_compactcut_map_bank()


@pytest.fixture(scope="module")
def registered_pairs(map_bank):
    pairs = {}
    for map_id, map_data in map_bank.items():
        for task_seed in compactcut.DEFAULT_TASK_SEEDS:
            payload = compactcut.generate_compactcut_tasks(map_data, task_seed)
            audit = compactcut.audit_compactcut_tasks(map_data, payload)
            assert audit["passed"], audit["errors"]
            pairs[(map_id, task_seed)] = (payload, audit)
    return pairs


def test_exact_four_public_task_geometry_apis_exist() -> None:
    assert callable(compactcut.generate_compactcut_map_bank)
    assert callable(compactcut.audit_compactcut_geometry)
    assert callable(compactcut.generate_compactcut_tasks)
    assert callable(compactcut.audit_compactcut_tasks)


def test_eight_registered_maps_have_real_capacity_two_cuts(map_bank) -> None:
    assert tuple(map_bank) == tuple(spec["map_id"] for spec in compactcut.MAP_SPECS)
    assert len(map_bank) == 8
    assert len(
        {
            map_data.metadata["compactcut_registration"]["grid_sha256"]
            for map_data in map_bank.values()
        }
    ) == 8
    for map_id, map_data in map_bank.items():
        audit = compactcut.audit_compactcut_geometry(map_data)
        assert audit["passed"], (map_id, audit["errors"])
        assert audit["height"] == 28
        assert audit["width"] == 39
        assert audit["cut_capacity"] == 2
        assert audit["selected_cut"]["valid_two_cell_cut"] is True
        assert audit["selected_cut"]["component_count_after_cut"] == 2
        assert audit["selected_cut"]["both_gate_subsets_restore_connectivity"] is True
        assert audit["bypass_after_selected_cut_removal"] is False


def test_all_sixteen_registered_pairs_pass_q0(registered_pairs) -> None:
    assert len(registered_pairs) == 16
    for (map_id, task_seed), (payload, audit) in registered_pairs.items():
        assert payload["map_id"] == map_id
        assert payload["task_seed"] == task_seed
        assert payload["agent_count"] == 120
        assert payload["pairing_semantics"] == (
            "independent_registered_endpoints_distance_near_matched_secondary_control"
        )
        assert payload["causal_control_claim"] is False
        assert payload["distance_assignment"]["structured_target_twice"] == 1
        assert audit["metrics"]["paired_distance"]["mean_relative_difference"] <= 0.05
        assert audit["metrics"]["paired_distance"]["p95_relative_difference"] <= 0.10
        for variant in compactcut.TASK_VARIANTS:
            task = payload["variants"][variant]
            witness = task["prioritized_joint_mapf_witness"]
            assert len(task["starts"]) == len(task["goals"]) == 120
            assert len({tuple(cell) for cell in task["starts"]}) == 120
            assert len({tuple(cell) for cell in task["goals"]}) == 120
            assert all(start != goal for start, goal in zip(task["starts"], task["goals"]))
            assert witness["schema"] == compactcut.SERIALIZED_WITNESS_SCHEMA
            assert witness["event_count"] == witness["makespan"]
            assert len(witness["move_events"]) == witness["event_count"]
        structured_metrics = audit["metrics"][compactcut.STRUCTURED_VARIANT]
        diagnostic_metrics = audit["metrics"][compactcut.DIAGNOSTIC_VARIANT]
        assert structured_metrics["cross_partition_count"] == 120
        assert structured_metrics["within_partition_count"] == 0
        assert diagnostic_metrics["within_partition_count"] == 120
        assert diagnostic_metrics["cross_partition_count"] == 0
        assert structured_metrics["prioritized_joint_mapf_witness"][
            "all_structured_agents_traversed_cut"
        ] is True
        assert set(
            structured_metrics["prioritized_joint_mapf_witness"][
                "structured_gate_direction_counts"
            ].values()
        ) == {30}
        assert len(
            structured_metrics["prioritized_joint_mapf_witness"][
                "structured_gate_direction_counts"
            ]
        ) == 4
        assert diagnostic_metrics["prioritized_joint_mapf_witness"][
            "diagnostic_cut_traversal_count"
        ] == 0


def test_generation_is_byte_structure_deterministic(map_bank, registered_pairs) -> None:
    expected = registered_pairs[("whcc_cfg_01", 521)][0]
    repeated = compactcut.generate_compactcut_tasks(map_bank["whcc_cfg_01"], 521)
    other_seed = registered_pairs[("whcc_cfg_01", 557)][0]

    assert repeated == expected
    assert repeated["geometry"] == other_seed["geometry"]
    assert repeated["variants"] != other_seed["variants"]


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    (
        (
            lambda payload: payload["routing_registration"].__setitem__(
                "sample_namespace", [999]
            ),
            "routing_registration_mismatch",
        ),
        (
            lambda payload: payload["distance_assignment"].__setitem__(
                "diagnostic_target_offset", 19
            ),
            "distance_assignment_mismatch",
        ),
        (
            lambda payload: payload["variants"][
                compactcut.DIAGNOSTIC_VARIANT
            ]["goals"][0].__setitem__(0, 999),
            f"{compactcut.DIAGNOSTIC_VARIANT}:registered_endpoint_or_assignment_mismatch",
        ),
        (
            lambda payload: payload["variants"][
                compactcut.STRUCTURED_VARIANT
            ]["prioritized_joint_mapf_witness"]["move_events"][0].__setitem__(
                "time", 7
            ),
            f"{compactcut.STRUCTURED_VARIANT}:deterministic_witness_mismatch",
        ),
    ),
)
def test_task_audit_rejects_registered_identity_tampering(
    map_bank, registered_pairs, tamper, expected_error
) -> None:
    payload = copy.deepcopy(registered_pairs[("whcc_cfg_01", 521)][0])
    tamper(payload)

    audit = compactcut.audit_compactcut_tasks(map_bank["whcc_cfg_01"], payload)

    assert audit["passed"] is False
    assert expected_error in audit["errors"]


def test_pairing_contract_and_task_seed_are_frozen(map_bank, registered_pairs) -> None:
    payload = copy.deepcopy(registered_pairs[("whcc_cfg_01", 521)][0])
    payload["pairing_semantics"] = "same-goals"
    audit = compactcut.audit_compactcut_tasks(map_bank["whcc_cfg_01"], payload)
    assert audit["passed"] is False
    assert "pairing_contract_mismatch" in audit["errors"]

    with pytest.raises(ValueError, match="unregistered compact-cut task_seed"):
        compactcut.generate_compactcut_tasks(map_bank["whcc_cfg_01"], 999)


def test_geometry_audit_rejects_registration_tampering(map_bank) -> None:
    map_data = copy.deepcopy(map_bank["whcc_cfg_01"])
    map_data.metadata["compactcut_registration"]["cut_capacity"] = 3

    audit = compactcut.audit_compactcut_geometry(map_data)

    assert audit["passed"] is False
    assert "registration_mismatch" in audit["errors"]


def test_helper_is_geometry_only_and_does_not_call_reset_or_solver() -> None:
    source = inspect.getsource(compactcut)
    assert "lns2_env" not in source
    assert "run_closed_loop" not in source
    assert "reset(" not in source
    assert "step(" not in source
