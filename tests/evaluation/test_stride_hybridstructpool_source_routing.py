from pathlib import Path

from experiments.stride_hybridstructpool_source_routing import (
    CONTROLLERS,
    _controller_kwargs,
    _select_promoted,
    load_config,
    schedule,
    zero_solver_audit,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_hybridstructpool_source_routing_v1.json"


def test_registration_schedule_is_complete_and_rotated() -> None:
    _path, _root, config = load_config(CONFIG)
    rows = schedule(config)
    assert len(rows) == 40
    assert {row["controller"] for row in rows} == set(CONTROLLERS)
    keys = {(row["group_id"], row["task_id"], row["solver_seed"]) for row in rows}
    assert len(keys) == 8
    assert all(
        sum(
            row["controller"] == controller and row["within_key_position"] == position
            for row in rows
        )
        >= 1
        for position, controller in enumerate(CONTROLLERS)
    )


def test_controller_modes_share_v2_but_route_only_registered_sources() -> None:
    _path, root, config = load_config(CONFIG)
    v2 = _controller_kwargs(root, config, "v2_only")
    full = _controller_kwargs(root, config, "full_v8")
    structural = _controller_kwargs(root, config, "structshell_only")
    causal = _controller_kwargs(root, config, "causal_only")
    routed = _controller_kwargs(root, config, "routed_structshell")
    assert all(
        row["controller"] == "v2-full"
        for row in (v2, full, structural, causal, routed)
    )
    assert "hybridstructpool_augmentation" not in v2
    assert full["hybridstructpool_augmentation"]["pool_id"] == (
        "stride-hybridstructpool-v1"
    )
    assert structural["hybridstructpool_augmentation"]["source_mode"] == (
        "structshell_only"
    )
    assert causal["hybridstructpool_augmentation"]["source_mode"] == "causal_only"
    assert routed["hybridstructpool_augmentation"]["activation_gate"]["gate_id"] == (
        "stride-highstress-conflict-structure-v1"
    )
    assert all(
        row["deterministic_pp_replay"] is False
        for row in (v2, full, structural, causal, routed)
    )


def test_zero_solver_audit_preserves_exact_source_memberships() -> None:
    report = zero_solver_audit(CONFIG)
    assert report["integrity_passed"] is True
    assert report["state_count"] == 78
    assert report["candidate_membership_totals"] == {
        "v2_only": 1367,
        "structshell_only": 2502,
        "causal_only": 2297,
        "full_v8": 3432,
        "routed_structshell": 2502,
    }


def test_near_tie_uses_group_regression_then_selection_time() -> None:
    eligible = ["structshell_only", "routed_structshell", "causal_only"]
    summaries = {
        "structshell_only": {
            "mean_raw_wall_time_to_feasible": 10.0,
            "mean_neighborhood_selection_seconds": 0.8,
        },
        "routed_structshell": {
            "mean_raw_wall_time_to_feasible": 10.05,
            "mean_neighborhood_selection_seconds": 0.7,
        },
        "causal_only": {
            "mean_raw_wall_time_to_feasible": 10.2,
            "mean_neighborhood_selection_seconds": 0.1,
        },
    }
    regressions = {
        "structshell_only": 0.03,
        "routed_structshell": 0.01,
        "causal_only": 0.0,
    }
    assert (
        _select_promoted(eligible, summaries, regressions, 0.01)
        == "routed_structshell"
    )
