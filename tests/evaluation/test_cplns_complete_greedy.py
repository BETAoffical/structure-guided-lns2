from copy import deepcopy
import json

import pytest

from scripts import diagnose_cplns_complete_greedy as audit


def test_only_early_stop_is_removed_acceptance_and_deadline_are_kept():
    text = "prefix\nbool InitLNS::runPP() {\nwhile (getTime() < time_limit) {\n" + audit.EARLY_STOP
    text += "}\nif (tl_use_simulated_annealing && complete) can_accept();\nreturn rollback();\n}\n"
    text += "bool InitLNS::getInitialSolution() { return true; }\nsuffix"
    assert audit.patch_pp(text) == text.replace(audit.EARLY_STOP, "", 1)
    with pytest.raises(ValueError):
        audit.patch_pp(text.replace(audit.EARLY_STOP, ""))
    with pytest.raises(ValueError):
        audit.patch_pp(text.replace(audit.EARLY_STOP, audit.EARLY_STOP * 2))


def test_schedule_only_changes_binary_and_not_sa_or_old_task_paths():
    config = json.loads(audit.real.CONFIG.read_text())
    ref = json.loads((audit.ROOT / "configs/cplns_sequential_reference_v1.json").read_text())
    cases = [{"case_id": c, "map_id": c, "agent_count": 299} for c in config["cases"]]
    plan = {"jobs": audit.real.schedule(config, cases, ref)}
    old = {j["job_id"]: j for j in plan["jobs"]}
    jobs = audit.jobs(plan)
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 20
    for job in jobs:
        assert job["argv"][1:] == old[job["source_job_id"]]["argv"][1:]
        options = dict(zip(job["argv"][1::2], job["argv"][2::2]))
        assert options["--sa"] == "false" and options["--sa_max_con_fail"] == "2"
        assert options["--cutoffTime"] == "30"
    with pytest.raises(ValueError):
        audit.jobs({"jobs": plan["jobs"][:-4]})


def rows():
    return [{"case_id": "a", "seed": 0, "profile": arm, "map_id": "m", "initial_digest": "same",
             "first_pp_order": [2, 0, 1], "total_pp_calls": calls, "final_feasible": True,
             "feasible_within_observed_steps": True, "accepted_worse": 0}
            for arm, calls in ((audit.BASE, 100), (audit.ARM, 50), (audit.SA, 40))]


def test_summary_records_three_way_paired_effort_without_time_claims():
    result = audit.compare(rows())
    assert result["decision"] == "bounded_decomposition_complete_no_promotion"
    assert result["profiles"][audit.ARM]["mean_pp_calls"] == 50
    assert result["comparisons"][audit.ARM + "_vs_" + audit.BASE]["fewer_pp_calls"] == 1
    assert "ttf" not in result["profiles"][audit.ARM]


@pytest.mark.parametrize("field,value", [("initial_digest", "changed"), ("first_pp_order", [0, 1, 2])])
def test_pairing_mismatch_blocks_interpretation(field, value):
    data = rows()
    data[1][field] = value
    assert audit.compare(data)["decision"] == "pairing_failure_stop"


def test_duplicate_or_missing_comparator_is_rejected():
    data = rows()
    with pytest.raises(ValueError):
        audit.compare(data + [deepcopy(data[0])])
    with pytest.raises(ValueError):
        audit.compare(data[:2])


def test_failure_with_few_calls_is_not_counted_as_an_effort_win():
    data = rows()
    data[1].update(final_feasible=False, total_pp_calls=1)
    result = audit.compare(data)["comparisons"][audit.ARM + "_vs_" + audit.BASE]
    assert result["common_success"] == result["fewer_pp_calls"] == 0


@pytest.mark.parametrize("worsening,restarts,raises", [(0, 1, False), (1, 1, True), (0, 2, True)])
def test_full_count_and_greedy_contract(tmp_path, monkeypatch, worsening, restarts, raises):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    folder = tmp_path / "output/jobs"
    folder.mkdir(parents=True)
    events = [{"event": "pp_order", "order": [0, 1]}, {"event": "final", "decisions_seen": 200}]
    (folder / "one.events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    row = {"observed_steps": 128, "accepted_worse": worsening, "conflicts": [2, 1],
           "initial_conflicts": 2, "upstream_restart_counter": restarts}
    monkeypatch.setattr(audit.real, "analyze_job", lambda _: dict(row))
    job = {"job_id": "one", "output": "output", "profile": audit.ARM}
    if raises:
        with pytest.raises(ValueError):
            audit.analyze(job)
    else:
        result = audit.analyze(job)
        assert result["total_pp_calls"] == 200 and result["first_pp_order"] == [0, 1]
