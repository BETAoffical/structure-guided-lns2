import copy

import pytest

from scripts import run_warehouse_repair_confirmation as run


def results():
    return [{"status": "ok", "failures": [], "checkpoints": [
        {"key_index": index * 2 + replica, "checkpoint_id": f"cp-{index * 2 + replica}",
         "map_id": f"map-{index // 2}", "qualification": {"passed": True}}
        for replica in range(2)]} for index in range(16)]


def test_checkpoint_gate_keeps_all_positions():
    data = results()
    report = run.checkpoint_report(data)
    assert report["passed"] and report["attempted"] == 32 and report["qualified_maps"] == 8
    for item in data[:4]:
        for cp in item["checkpoints"]:
            cp["qualification"]["passed"] = False
    report = run.checkpoint_report(data)
    assert report["qualified"] == 24 and not report["passed"]  # Only six maps survive.
    assert len(report["checkpoints"]) == 32


def test_insufficient_or_duplicate_positions_fail_closed():
    assert not run.checkpoint_report(results()[:-1])["passed"]
    data = results()
    data.append(copy.deepcopy(data[0]))
    with pytest.raises(ValueError, match="duplicate"):
        run.checkpoint_report(data)


def test_native_error_never_becomes_supply_failure():
    item = {"task_index": 0, "row": {"task_id": "t", "map_id": "m"},
            "slots": [{"checkpoint_slot": 0}, {"checkpoint_slot": 1}]}
    result = run.failure_result({"item": item}, "error", "native mismatch")
    assert len(run.checkpoint_report([result])["errors"]) == 2


def test_new_protocol_does_not_reintroduce_iteration_limit(monkeypatch):
    config = run.read_json(run.ROOT / run.CONFIG)
    assert run.environment_config(config)["max_repair_iterations"] == 0
    monkeypatch.setattr(run, "_load_dataset_rows", lambda *a: [{"task_id": f"task-{i:02d}"} for i in range(16)])
    plan = run.task_plan(config)
    assert len(plan) == 16
    for item in plan:
        assert len(item["slots"]) == 2
        assert len({s["incumbent_seed"] for s in item["slots"]}) == 1
        assert len({s["selection_seed"] for s in item["slots"]}) == 2
