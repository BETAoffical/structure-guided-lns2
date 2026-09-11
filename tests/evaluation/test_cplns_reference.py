from copy import deepcopy
import hashlib
import json
import zipfile

import pytest

from scripts import verify_cplns_reference as ref


FINAL = "final colliding 0 costs 56 costs_wc 56 restart 0 a*/s 20 iter 0 0 iter/s 0 t1 0 t2 0 subopt 1.000 mem 0.01 t_share 0.000 total 0.025\n"


def test_four_profiles_do_not_conflate_sa_and_restart():
    config = json.loads(ref.CONFIG.read_text())
    jobs = ref.schedule(config)
    assert len(jobs) == 16 and len({j["job_id"] for j in jobs}) == 16
    for job in jobs:
        options = dict(zip(job["argv"][1::2], job["argv"][2::2]))
        assert options["--numSolver"] == options["--numSolverPP"] == "1"
        assert options["--astar_wh"] == "1" and options["--maxIterations"] == "0"
        assert options["--initDestoryStrategy"] == "Adaptive"
        assert options["--sa"] == ("true" if job["profile"].startswith("author_sa_") else "false")
        assert options["--sa_max_con_fail"] == ("2" if "no_restart" in job["profile"] else "0.4")
    with pytest.raises(ValueError):
        ref.command(config, "unknown", "author_sa_restart", 0)
    config["common_options"]["numSolver"] = "20"
    with pytest.raises(ValueError):
        ref.command(config, "open", "author_sa_restart", 0)


@pytest.mark.parametrize("name", ["open", "order_reversal_corridor"])
def test_fixture_unique_starts_goals_and_determinism(name):
    grid, scen = ref.fixture_content(name)
    assert (grid, scen) == ref.fixture_content(name)
    rows = [r.split() for r in scen.splitlines()[1:]]
    assert len(rows) == 8
    assert len({tuple(r[4:6]) for r in rows}) == len({tuple(r[6:8]) for r in rows}) == 8


def test_parser_does_not_trust_zero_exit_or_unvalidated_success():
    assert not ref.parse_output("", 0)["valid"]
    assert not ref.parse_output(FINAL, 0)["valid"]
    log = "validateSolution LNS:491\n" + FINAL
    row = ref.parse_output(log, 0)
    assert row["valid"] and row["status"] == "reported_feasible"
    assert row["coarse_t1_seconds"] == 0 and not row["independent_path_validation"]
    assert not ref.parse_output(log, -6)["valid"]
    assert not ref.parse_output(log, 0, True)["valid"]
    assert not ref.parse_output(log + FINAL, 0)["valid"]
    incomplete = ref.parse_output(FINAL.replace("colliding 0", "colliding 3"), 0)
    assert incomplete["valid"] and incomplete["status"] == "reported_incomplete"


def test_summary_tampering_and_log_mutation_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ref, "ROOT", tmp_path)
    log = tmp_path / "out/smoke/job.log"
    log.parent.mkdir(parents=True)
    log.write_text("validateSolution LNS:491\n" + FINAL)
    job = dict(job_id="job", fixture="open", profile="author_sa_restart", seed=0, argv=["plns"], directory="out")
    row = {**ref.parse_output(log.read_text(), 0), **job,
           "process_returncode": 0, "process_timed_out": False, "log_sha256": ref.sha256_file(log)}
    ref.validate_result(row, job)
    changed = deepcopy(row)
    changed["soc"] = 1
    with pytest.raises(ValueError):
        ref.validate_result(changed, job)
    log.write_text("changed")
    with pytest.raises(ValueError):
        ref.validate_result(row, job)


def test_archive_source_identity_and_modification_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(ref, "ROOT", tmp_path)
    out = tmp_path / "out"
    source = out / "cplns-abc"
    source.mkdir(parents=True)
    archive = out / "upstream-e3fbcc82.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("cplns-abc/README.md", "upstream")
    (source / "README.md").write_text("upstream")
    config = dict(output="out", source="out/cplns-abc", commit="abc",
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    assert len(ref.check_archive(config)) == 1
    (source / "README.md").write_text("modified")
    with pytest.raises(ValueError):
        ref.check_archive(config)
