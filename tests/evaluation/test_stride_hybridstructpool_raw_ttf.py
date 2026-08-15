from pathlib import Path

from experiments.stride_hybridstructpool_raw_ttf import (
    CONTROLLERS,
    _controller_kwargs,
    load_config,
    schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_hybridstructpool_raw_ttf_quick_v1.json"
CONFIG_V2 = ROOT / "configs" / "stride_hybridstructpool_raw_ttf_quick_v2.json"
CONFIG_V3 = ROOT / "configs" / "stride_hybridstructpool_raw_ttf_quick_v3.json"


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


def test_v2_registration_reuses_the_same_strict_serial_comparison() -> None:
    _path, _root, config = load_config(CONFIG_V2)
    rows = schedule(config)
    assert len(rows) == 24
    assert config["pre_registration_parent_commit"] == (
        "52096acb6120cc02c59a352bb7f72fdb9532fb6b"
    )
    assert config["inputs"]["hybrid_runtime_report"]["sha256"] == (
        "f89cecfb57e763daa34b4e284db658206ecaf516f57b366ade8914b8e809fce8"
    )
    assert config["comparison"]["workers_for_timed_episodes"] == 1


def test_v3_registration_reuses_comparison_with_v8_only() -> None:
    _path, _root, config = load_config(CONFIG_V3)
    rows = schedule(config)
    assert len(rows) == 24
    assert config["pre_registration_parent_commit"] == "4096fa8"
    assert config["inputs"]["hybrid_runtime_report"]["sha256"] == (
        "e70fffb34245c6d3506b713847066870e5a91f8a54c37901ffdf215ec3714f50"
    )
    assert config["hybridstructpool_augmentation"]["runtime_filter_id"] == "none"
    assert config["comparison"]["workers_for_timed_episodes"] == 1
