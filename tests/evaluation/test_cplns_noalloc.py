import json

import pytest

from scripts import verify_cplns_noalloc as audit
from scripts import verify_cplns_reference as reference
from scripts import verify_cplns_stationary_count as native_counter


def test_schedule_has_distinct_seeds_and_full_init_log_level():
    config = json.loads(audit.CONFIG.read_text())
    ref = json.loads(reference.CONFIG.read_text())
    jobs = audit.schedule(config, ref)
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 96
    assert config["seeds"] == [0, 2]
    for i in range(0, len(jobs), 3):
        assert jobs[i]["argv"][1:] == jobs[i + 1]["argv"][1:] == jobs[i + 2]["argv"][1:]
        options = dict(zip(jobs[i]["argv"][1::2], jobs[i]["argv"][2::2]))
        assert options["--screen"] == "4" and options["--numSolver"] == "1"
    assert config["no_ttf_or_promotion"]


def test_initial_path_log_contract_rejects_missing_and_duplicate_agents():
    text = "Agent 0: 0\t1\t\nAgent 1: 3\t2\t\n"
    assert audit.initial_paths(text, 2) == {0: [0, 1], 1: [3, 2]}
    assert audit.initial_paths("Neighbors: 0, 1,\n", 2) is None
    with pytest.raises(ValueError):
        audit.initial_paths("Agent 0: 0 1\nAgent 0: 0 1\n", 2)
    assert not audit.compare_logs("x", "x", 2, 512)["initial_paths_equal"]


def test_pointer_neighborhood_difference_not_removed_from_log():
    prefix = "Agent 0: 0 1\nAgent 1: 3 2\n"
    left = prefix + "Neighbors: 0, 1,\n" * 20
    right = prefix + "Neighbors: 0, 2,\n" * 20
    result = audit.compare_logs(left, right, 2, 512)
    assert result["initial_paths_equal"] and not result["prefix_equal"]
    assert result["first_difference"] == 2


def test_counter_patch_changes_only_empty_path_guard():
    function = "bool InitLNS::updateCollidingPairs() { if (path.size() < 2) return succ; }\n"
    text = function + "void InitLNS::chooseDestroyHeuristicbyALNS() {}"
    transformed = audit.extract_counters(text)
    assert "CounterFixture::original" in transformed
    assert "if (path.size() < 2) return succ;" in transformed
    assert "CounterFixture::corrected" in transformed
    assert "if (path.empty()) return succ;" in transformed
    with pytest.raises(ValueError):
        audit.extract_counters(text.replace("path.size() < 2", "path.size() <= 2"))


@pytest.mark.parametrize("paths,target,expected", [
    ({7: [1], 14: [0, 1, 2, 1, 2]}, 7, [[7, 14]]),
    ({0: [0, 1], 1: [1, 0]}, 0, [[0, 1]]),
    ({0: [0, 1], 1: [3, 2, 1, 2]}, 0, [[0, 1]]),
    ({0: [], 1: [1]}, 0, []),
    ({0: [0], 1: [1]}, 0, []),
])
def test_independent_target_pairs(paths, target, expected):
    assert audit.target_pairs(paths, target) == expected


def test_noalloc_header_has_only_pod_thread_storage():
    text = audit.HEADER.read_text()
    assert "std::ofstream" not in text and "std::string" not in text
    assert "char data[8192]" in text and "O_EXCL" in text
    assert "CLOCK_MONOTONIC" in text and "errno == EINTR" in text
    assert "rand(" not in text


def test_native_counter_only_changes_one_guard_and_reuses_registered_tasks():
    source = "abc if (path.size() < 2) return succ; xyz"
    assert native_counter.corrected_source(source) == "abc if (path.empty()) return succ; xyz"
    with pytest.raises(ValueError):
        native_counter.corrected_source(source + source)
    config = json.loads(audit.CONFIG.read_text())
    ref = json.loads(reference.CONFIG.read_text())
    originals = {j["job_id"]: j for j in audit.schedule(config, ref)}
    jobs = native_counter.jobs(config, ref)
    assert len(jobs) == 32
    for job in jobs:
        assert job["argv"][1:] == originals[job["source_job_id"]]["argv"][1:]
        assert job["argv"][0] != originals[job["source_job_id"]]["argv"][0]
