import tempfile
from pathlib import Path

import pytest

from experiments.closed_loop_trace_storage import write_state_blob
from lns2_selector.evaluation.path_quality_execution import read_artifact, spec_fingerprint, supervise_episode
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from scripts import warehouse_repair_confirmation_runtime as rt


def test_timing_requires_separate_authorization(monkeypatch):
    monkeypatch.setattr(rt, "verify_prepared", lambda **k: pytest.fail("must fail before loading inputs"))
    with pytest.raises(PermissionError):
        rt.collect()


def test_serial_stop_resume_and_no_retry(tmp_path):
    saved, calls, stop = {}, [], [False]
    specs = [{"item": {"controller": str(i)}, "binding": str(i)} for i in range(4)]
    def execute(spec, **kwargs):
        calls.append(spec["binding"])
        value = {"error": False, "status": "external_timeout" if len(calls) == 2 else "completed"}
        saved[spec["binding"]] = value
        stop[0] = len(calls) == 2
        return value
    kwargs = dict(execute=execute, validate_result=lambda s: saved.get(s["binding"]), stop_requested=lambda: stop[0])
    result = rt.run_serial(specs, tmp_path, resume=False, **kwargs)
    assert result["completed"] == 2 and result["status"] == "paused_after_episode"
    stop[0] = False
    result = rt.run_serial(specs, tmp_path, resume=True, **kwargs)
    assert result["status"] == "complete" and calls == ["0", "1", "2", "3"]


def test_error_stops_before_next_episode(tmp_path):
    calls = []
    def execute(spec, **kwargs):
        calls.append(spec)
        return {"error": True}
    with pytest.raises(ValueError, match="episode error"):
        rt.run_serial([{"binding": "a"}, {"binding": "b"}], tmp_path, resume=False,
                      stop_requested=lambda: False, execute=execute, validate_result=lambda s: None)
    assert len(calls) == 1


def test_schedule_preserves_all_three_paired_methods():
    config = rt.read_json(rt.ROOT / rt.CONFIG)
    cps = [{"key_index": i, "checkpoint_id": str(i), "checkpoint_identity_sha256": str(i),
            "task_id": str(i // 2), "map_id": str(i // 4), "screen_solver_seed": 100 + i,
            "qualification": {"passed": i != 0}} for i in range(32)]
    items = rt.bound_schedule(config, {"checkpoints": cps})
    assert len(items) == 93
    for start in range(0, len(items), 3):
        group = items[start:start + 3]
        assert {v["controller"] for v in group} == set(config["controllers"])
        assert len({v["solver_seed"] for v in group}) == 1
        assert len({v["checkpoint_id"] for v in group}) == 1


def test_paired_gate_and_map_bootstrap():
    gate = rt.read_json(rt.ROOT / rt.CONFIG)["gate"]
    left = {str(i): {"map_id": str(i // 4), "capped_delivery_seconds": 2.0,
                     "process_wall_seconds": 3.0, "success": True, "censored": False} for i in range(32)}
    right = {k: {**v, "capped_delivery_seconds": 1.0, "process_wall_seconds": 2.0} for k, v in left.items()}
    result = rt.comparison(left, right, gate)
    assert result["passed"] and result["delivery_improvement"] == 0.5
    assert result["map_bootstrap_95ci"] == [0.5, 0.5]
    assert rt.comparison(left, right, gate) == result
    assert not rt.comparison(left, left, gate)["passed"]
    right.pop("0")
    assert not rt.comparison(left, right, gate)["passed"]


@pytest.mark.parametrize("collision", [False, True])
def test_native_micro_three_methods_paths_and_resume(collision):
    pytest.importorskip("lns2_env")
    config = rt.read_json(rt.ROOT / rt.CONFIG)
    if not (rt.ROOT / config["frozen"]["native_path"]).exists():
        pytest.skip("registered native binary is unavailable")
    rt.install_native(config)
    with tempfile.TemporaryDirectory(prefix="warehouse-readiness-micro-", dir=rt.ROOT / "build") as temporary:
        root = Path(temporary)
        map_file = root / "tiny.map"
        scen = root / "tiny.scen"
        map_file.write_text("type octile\nheight 5\nwidth 7\nmap\n" + ".......\n" * 5, encoding="utf-8")
        paths = [list(range(14, 21)), list(range(20, 13, -1))] if collision else [[14, 15], [20, 19]]
        paths += [[0, 1], [6, 5], [28, 29], [34, 33]]
        scen.write_text("version 1\n" + "".join(
            f"0\ttiny.map\t7\t5\t{p[0] % 7}\t{p[0] // 7}\t{p[-1] % 7}\t{p[-1] // 7}\t{len(p) - 1}\n" for p in paths), encoding="utf-8")
        row = {"split": ".", "map_file": str(map_file), "scenario_file": str(scen),
               "map_metadata_file": str(map_file), "task_file": str(scen),
               "map_id": "tiny", "task_id": "tiny-task", "agent_count": 6, "layout_mode": "warehouse"}
        env = rt._make_environment(str(root), row, {**rt.environment_config(config), "time_limit": 60}, "Adaptive")
        state = rt._plain(env.reset_paths(paths, seed=123))
        assert bool(state["feasible"]) is not collision
        blob_ref, blob = write_state_blob(root, state)
        cp = {"source_kind": "checkpoint_blob_v1", "checkpoint_id": "tiny-cp", "map_id": "tiny", "task_id": "tiny-task",
              "agent_count": 6, "state_blob": blob_ref, "state_blob_sha256": rt.sha256_file(blob),
              "expected_fingerprint": rt.state_fingerprint(state), "expected_conflicts": state["num_of_colliding_pairs"],
              "repair_structure_fingerprint": repair_structure_fingerprint(state), "restore_seed": 123}
        cp["checkpoint_identity_sha256"] = rt.compute_checkpoint_identity_sha256(cp)
        template = rt.read_json(rt.ROOT / rt.TEMPLATE)
        outputs = []
        for controller in config["controllers"]:
            item = {"controller": controller, "task_id": "tiny-task", "solver_seed": 456,
                    "protocol": "first_feasible", "budget_seconds": 5}
            spec = rt.make_spec(config, cp, row, item, {}, root / controller, template)
            spec["worker_job"]["episode_override"]["initial_restore"]["collection_root"] = str(root)
            spec["process_timeout_seconds"] = 15
            spec["binding"] = spec_fingerprint(spec)
            summary = rt.execute_episode(spec, authorized=True, resume=False)
            assert summary["status"] == "completed", (controller, summary)
            assert summary["success_by_deadline"]
            assert rt.validated_episode(spec) == summary
            result = read_artifact(Path(spec["output"]) / "result.json", spec["binding"])
            assert result["dispatch_wall_seconds"] >= result["first_feasible_elapsed_seconds"]
            assert supervise_episode(spec, authorized=True, resume=True) == summary
            initial = read_artifact(Path(spec["output"]) / "initial.json", spec["binding"])
            first = read_artifact(Path(spec["output"]) / "first_phase_result.json", spec["binding"])
            assert first["summary"]["repair_iterations"] > 0 if collision else first["summary"]["repair_iterations"] == 0
            outputs.append(initial["state_fingerprint"])
        assert len(set(outputs)) == 1
