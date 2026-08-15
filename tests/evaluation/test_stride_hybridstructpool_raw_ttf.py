from pathlib import Path

from experiments.stride_hybridstructpool_raw_ttf import (
    CONTROLLERS,
    _controller_kwargs,
    load_config,
    schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_hybridstructpool_raw_ttf_quick_v1.json"


def test_registration_and_schedule_are_complete_and_rotated() -> None:
    _path, _root, config = load_config(CONFIG)
    rows = schedule(config)
    assert len(rows) == 24
    assert {row["controller"] for row in rows} == set(CONTROLLERS)
    keys = {(row["group_id"], row["task_id"], row["solver_seed"]) for row in rows}
    assert len(keys) == 8
    first_by_key = {}
    for row in rows:
        if row["within_key_position"] == 0:
            first_by_key[(row["group_id"], row["task_id"], row["solver_seed"])] = row["controller"]
    assert set(first_by_key.values()) == set(CONTROLLERS)


def test_controller_contract_keeps_official_separate_from_learned_arms() -> None:
    _path, root, config = load_config(CONFIG)
    official = _controller_kwargs(root, config, "official_adaptive")
    v2 = _controller_kwargs(root, config, "v2_full")
    hybrid = _controller_kwargs(root, config, "hybridstructpool_full")
    assert official["controller"] == "official_adaptive"
    assert "controller_bundle" not in official
    assert v2["controller"] == "v2-full"
    assert "hybridstructpool_augmentation" not in v2
    assert hybrid["controller"] == "v2-full"
    assert hybrid["hybridstructpool_augmentation"]["full_union_required"] is True
    assert all(not row["deterministic_pp_replay"] for row in (official, v2, hybrid))
