from copy import deepcopy
from itertools import product

import pytest

from scripts import diagnose_feedback_equivalence as diagnostic


def values():
    return [{"snapshot": snapshot, "restore": restore, "condition_key": "same",
             "signature": {"accepted": False, "accepted_paths": None, "after_conflicts": 3},
             "search_delta": {"generated": 10}, "censored": False,
             "selected_paths_before": [snapshot], "selected_paths_after": [snapshot]}
            for snapshot, restore in product(("8", "9"), ("direct", "prefix"))]


def test_rollback_old_paths_can_differ_without_false_mismatch():
    result = diagnostic.compare_cell(values())
    assert result["same_outcome"] and result["same_condition"]


def test_accepted_path_difference_is_not_equivalent():
    rows = values()
    for v in rows:
        v["signature"] = {"accepted": True, "accepted_paths": "a"}
    rows[-1]["signature"]["accepted_paths"] = "b"
    assert not diagnostic.compare_cell(rows)["same_outcome"]


def test_censoring_and_search_variation_are_separate():
    rows = values()
    rows[0]["censored"] = True
    rows[1]["search_delta"]["generated"] = 11
    result = diagnostic.compare_cell(rows)
    assert result["censored"] and result["same_outcome"] and not result["same_search_counters"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "four_snapshots"])
def test_incomplete_or_duplicate_cells_fail(mutation):
    rows = values()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[0] = deepcopy(rows[1])
    else:
        for i, row in enumerate(rows):
            row["snapshot"] = str(i)
            row["restore"] = "direct"
    with pytest.raises(ValueError):
        diagnostic.compare_cell(rows)


def test_job_factorial_seed_and_order_pairing(monkeypatch):
    monkeypatch.setattr(diagnostic, "sha256_file", lambda _: "plan-hash")
    group = {"group_id": 1, "agents": [31, 4], "snapshots": {"8": {}, "9": {}}}
    jobs = list(diagnostic.jobs_for({"groups": [group], "config": {}}))
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 16
    for trial in (0, 1):
        selected = [j for j in jobs if j["trial"] == trial]
        assert len({j["seed"] for j in selected}) == 1
        assert {tuple(j["order"]) for j in selected} == {(4, 31), (31, 4)}
    assert jobs[0]["seed"] != jobs[1]["seed"]


def test_deterministic_control_selection():
    def case(checkpoint, controller, recovery):
        return {"checkpoint": checkpoint, "controller": controller, "rows": [
            {"decision": 1, "repair_problem": "key", "paths": "a", "before_conflicts": 1,
             "after_conflicts": 1, "censored": False},
            {"decision": 2, "repair_problem": "key", "paths": "b", "before_conflicts": 1,
             "after_conflicts": 0 if recovery else 1, "censored": False}]}
    cases = [case(f"repair-confirm-{c}", "dual16", False) for c in ("09", "13", "22")]
    cases += [case("repair-confirm-08", "dual16", True), case("repair-confirm-11", "v2-full", True)]
    assert diagnostic.select_pairs(cases) == diagnostic.select_pairs(list(reversed(cases)))


def test_resume_identity_rejects_changed_seed(monkeypatch):
    monkeypatch.setattr(diagnostic, "sha256_file", lambda _: "plan-hash")
    group = {"group_id": 1, "agents": [31, 4], "snapshots": {"8": {}, "9": {}}}
    job = next(diagnostic.jobs_for({"groups": [group], "config": {}}))
    result = {k: job[k] for k in ("job_id", "plan_sha256", "snapshot", "restore", "order", "seed", "trial")}
    result.update(status="ok", group_id=1)
    diagnostic.validate_result(result, job)
    result["seed"] += 1
    with pytest.raises(ValueError):
        diagnostic.validate_result(result, job)


def test_blob_paths_are_relative_to_episode_root_not_trace_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "BASE", tmp_path)
    relative = "state_blobs/test.json.gz"
    assert diagnostic.snapshot_blob_path({"job_id": "job"}, relative) == (
        tmp_path / "timed/job/first_phase/state_blobs/test.json.gz").resolve()
    with pytest.raises(ValueError):
        diagnostic.snapshot_blob_path({"job_id": "job"}, "../outside.json.gz")
