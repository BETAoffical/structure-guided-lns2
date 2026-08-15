from __future__ import annotations

from pathlib import Path

from experiments.stride_hybridstructpool_runtime_pilot import (
    ARMS,
    load_registration,
    schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stride_hybridstructpool_runtime_pilot_v1_registration.json"


def test_registration_reuses_the_complete_registered_maze_cohort() -> None:
    _path, _root, config, _source, tasks = load_registration(CONFIG)
    assert len(tasks) == 19
    assert config["execution"]["repair_seed_policy"] == "episode_stream"
    assert config["execution"]["deterministic_pp_replay"] is False
    assert config["execution"]["failure_rescue_enabled"] is False


def test_schedule_is_strictly_paired_and_pool_only() -> None:
    _path, _root, config, _source, tasks = load_registration(CONFIG)
    rows = schedule(tasks, config["execution"]["trial_indices"])
    assert len(rows) == 152
    paired: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        key = (str(row["task_fingerprint"]), int(row["trial_index"]))
        paired.setdefault(key, set()).add(str(row["arm"]))
    assert len(paired) == 76
    assert all(value == set(ARMS) for value in paired.values())

