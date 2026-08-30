from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from experiments.stride_fresh_matched_v2_active_supply_overlay_v1 import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    FOLDS,
    build_plan,
    select_active_supply_states,
    validate_config,
    wave_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "configs" / "stride_fresh_matched_v2_active_supply_overlay_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _map_family(map_id: str) -> str:
    if map_id in {"den520d", "lak303d"}:
        return "game"
    return map_id.split("-", 1)[0]


def _arm(role: str, agents: range) -> dict:
    members = list(agents)
    selection_families = {
        "v2_anchor": ["collision"],
        "component16": ["structpool-conflict-component:16"],
        "hotspot16": ["structpool-spatiotemporal-hotspot:16"],
    }[role]
    return {
        "role": role,
        "candidate_id": f"fixture-{role}-{'-'.join(map(str, members))}",
        "agents": members,
        "actual_size": len(members),
        "selection_families": selection_families,
        "features": {},
    }


def _preflight_row(
    fold: str,
    map_id: str,
    episode_index: int,
    decision_index: int,
) -> dict:
    episode_id = f"{fold}-{map_id}-episode-{episode_index}"
    return {
        "origin": "synthetic_preaction",
        "source_id": f"fixture-{fold}",
        "episode_id": episode_id,
        "task_id": f"fixture-task-{fold}-{map_id}-{episode_index}",
        "map_id": map_id,
        "map_family": _map_family(map_id),
        "solver_seed": 41 + episode_index,
        "decision_index": decision_index,
        "before_fingerprint": f"before-{episode_id}-{decision_index}",
        "before_conflicts": 10,
        "agent_count": 128,
        "arms": {
            "v2_anchor": _arm("v2_anchor", range(0, 8)),
            "component16": _arm("component16", range(16, 32)),
            "hotspot16": _arm("hotspot16", range(48, 64)),
        },
        "eligible": True,
        "ineligibility_reasons": [],
        "repair_fingerprint_preserved": True,
        "target_outcome_fields_read": False,
        "candidate_repair_actions_executed": False,
    }


def _complete_q0_rows() -> list[dict]:
    # Two independent source episodes per map make all quotas feasible under
    # the preregistered episode cap (4) and map cap (16).
    return [
        _preflight_row(fold, map_id, episode_index, decision_index)
        for fold, map_ids in FOLDS.items()
        for map_id in map_ids
        for episode_index in range(2)
        for decision_index in range(12)
    ]


def _assert_hard_supply_failure(
    selected: list[dict], report: dict, config: dict
) -> None:
    assert selected == []
    assert report["status"] == "STATE_SUPPLY_FAIL"
    assert report["selected_state_count"] == 0
    assert report["h1_logical_trial_count"] == 0
    assert report.get("reserve_backfill_used", False) is False
    assert config["state_selection"]["failure_action"] == (
        "STATE_SUPPLY_FAIL_h1_zero_no_reserve"
    )
    assert config["claim_boundary"]["reserve_backfill_allowed"] is False


def test_registered_config_live_pins_and_plan_are_exact() -> None:
    config = _config()
    validate_config(config, project_root=ROOT)
    plan = build_plan(CONFIG_PATH)

    assert config["schema"] == CONFIG_SCHEMA
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["v1_read_only"]["expected_episode_count"] == 48
    assert config["v1_read_only"]["expected_zero_prefix_inactive_episode_count"] == 25
    assert config["v1_read_only"]["mutation_allowed"] is False
    assert plan["wave_a_episode_count"] == 20
    assert plan["combined_source_episode_count"] == 68
    assert plan["target_state_count"] == 96
    assert plan["states_per_fold"] == 24
    assert plan["logical_h1_trial_count"] == 4608
    assert plan["wave_a_workers"] == plan["h1_workers"] == 16
    assert plan["reserve_backfill_allowed"] is False
    assert plan["v1_mutation_allowed"] is False

    changed = copy.deepcopy(config)
    changed["v1_read_only"]["source_report"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_config(changed, project_root=ROOT)


def test_wave_a_schedule_is_exactly_ten_tasks_times_seeds_41_and_42() -> None:
    config = _config()
    schedule = wave_schedule(config)

    assert len(schedule) == 20
    by_task = defaultdict(list)
    for row in schedule:
        by_task[(row["source_id"], row["task_id"])].append(row["solver_seed"])
    assert len(by_task) == 10
    assert all(sorted(seeds) == [41, 42] for seeds in by_task.values())
    assert len({row["map_id"] for row in schedule}) == 5
    assert Counter(row["map_family"] for row in schedule) == {
        "random": 8,
        "maze": 8,
        "room": 4,
    }
    assert {
        (row["map_id"], row["task_id"].rsplit("__agents_", 1)[-1])
        for row in schedule
    } == {
        ("random-32-32-20", "0400"),
        ("maze-128-128-1", "0600"),
        ("maze-32-32-4", "0200"),
        ("room-64-64-16", "0600"),
        ("random-64-64-10", "0600"),
    }


def test_q0_deterministically_selects_exact_balanced_96() -> None:
    config = _config()
    rows = _complete_q0_rows()
    selected, report = select_active_supply_states(rows, config)
    reversed_selected, reversed_report = select_active_supply_states(
        reversed(rows), config
    )

    assert report["status"] == reversed_report["status"] == "ok"
    assert len(selected) == report["selected_state_count"] == 96
    assert report["h1_logical_trial_count"] == 4608
    assert [row["state_occurrence_id"] for row in selected] == [
        row["state_occurrence_id"] for row in reversed_selected
    ]
    map_to_fold = {
        map_id: fold for fold, map_ids in FOLDS.items() for map_id in map_ids
    }
    actual_fold_counts = Counter(map_to_fold[row["map_id"]] for row in selected)
    actual_depth_counts = Counter(
        (map_to_fold[row["map_id"]], row["depth_band"]) for row in selected
    )
    actual_episode_counts = Counter(
        (row["origin"], row["source_id"], row["episode_id"])
        for row in selected
    )
    actual_map_counts = Counter(row["map_id"] for row in selected)
    assert actual_fold_counts == {fold: 24 for fold in FOLDS}
    assert report["fold_counts"] == dict(actual_fold_counts)
    assert all(
        actual_depth_counts[(fold, band)] >= 4
        for fold in FOLDS
        for band in ("d0", "d1_3", "d4_plus")
    )
    assert max(actual_episode_counts.values()) <= 4
    assert max(actual_map_counts.values()) <= 16
    assert len(actual_map_counts) >= 10
    assert len({row["map_family"] for row in selected}) >= 4
    assert report["maximum_episode_count"] == max(actual_episode_counts.values())
    assert report["maximum_map_count"] == max(actual_map_counts.values())
    assert report["selected_map_count"] == len(actual_map_counts)
    assert report["selected_family_count"] == len(
        {row["map_family"] for row in selected}
    )
    assert all(row["target_outcome_fields_read"] is False for row in selected)
    assert all(row["depth_band"] in {"d0", "d1_3", "d4_plus"} for row in selected)
    for row in selected:
        arms = row["arms"]
        assert arms["component16"]["actual_size"] == 16
        assert arms["hotspot16"]["actual_size"] == 16
        assert len({tuple(arms[arm]["agents"]) for arm in arms}) == 3


def test_q0_missing_depth_band_is_hard_failure_without_reserve() -> None:
    config = _config()
    rows = [row for row in _complete_q0_rows() if row["decision_index"] < 4]

    selected, report = select_active_supply_states(rows, config)

    _assert_hard_supply_failure(selected, report, config)
    assert any("d4_plus_supply" in reason for reason in report["failure_reasons"])


def test_q0_duplicate_structural_arms_are_ineligible_and_hard_fail() -> None:
    config = _config()
    rows = _complete_q0_rows()
    for row in rows:
        row["arms"]["hotspot16"] = copy.deepcopy(row["arms"]["component16"])
        row["arms"]["hotspot16"]["role"] = "hotspot16"

    selected, report = select_active_supply_states(rows, config)

    _assert_hard_supply_failure(selected, report, config)
    assert report["eligible_preaction_state_count"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_result", "success"),
        ("final_success", True),
        ("pp_seconds", 0.25),
    ],
)
def test_q0_rejects_any_forbidden_outcome_field(
    field: str, value: object
) -> None:
    config = _config()
    rows = _complete_q0_rows()
    # One contaminated prefix invalidates Q0; it cannot be silently dropped and
    # replaced by another prefix because reserve/outcome-based deletion is banned.
    rows[0]["forbidden_fixture"] = {field: value}

    selected, report = select_active_supply_states(rows, config)

    _assert_hard_supply_failure(selected, report, config)
    assert any("forbidden" in reason for reason in report["failure_reasons"])
