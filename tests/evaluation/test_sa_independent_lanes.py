import multiprocessing
import os
import signal
import sys
import time
from unittest.mock import patch

import pytest

from scripts import diagnose_sa_independent_lanes as probe


def trace(steps, success=True, fingerprint="a", censored=False):
    return {"transitions": [{}] * steps, "feasible": success, "initial_fingerprint": fingerprint, "censored": censored}


def test_fixed_budget_counts_both_lanes():
    value = probe.trace_portfolio(trace(256, False), trace(20))
    assert value == {"feasible": True, "repair_calls": 40, "used": [20, 20]}
    assert probe.trace_portfolio(trace(129), trace(256, False))["feasible"] is False
    assert probe.trace_portfolio(trace(128), trace(256, False))["repair_calls"] == 255


def test_no_unspent_future_credit():
    assert probe.trace_portfolio(trace(3), trace(1))["repair_calls"] == 2
    assert probe.trace_portfolio(trace(1), trace(1))["repair_calls"] == 1


def test_zero_conflict_no_repair():
    assert probe.trace_portfolio(trace(0), trace(0))["repair_calls"] == 0


@pytest.mark.parametrize("right", [trace(2, fingerprint="b"), trace(2, censored=True), trace(2, False)])
def test_invalid_prefix_refused(right):
    with pytest.raises(ValueError):
        probe.trace_portfolio(trace(256, False), right)


def test_fixed_schedule():
    assert [probe.schedule_index(i) for i in range(6)] == [0, 1, 0, 1, 0, 1]
    with pytest.raises(ValueError):
        probe.schedule_index(-1)


def test_ipc_timeout():
    parent, child = multiprocessing.Pipe()
    try:
        with pytest.raises(TimeoutError):
            probe.receive(parent, .01)
        child.send({"error": "isolated failure"})
        with pytest.raises(RuntimeError, match="isolated failure"):
            probe.receive(parent, 1)
    finally:
        parent.close()
        child.close()


def sleeping_child():
    time.sleep(30)


def test_explicit_child_cleanup():
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=sleeping_child)
    process.start()
    child.close()
    probe.stop_children([(process, parent)])
    assert not process.is_alive()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux native subprocess contract")
def test_parent_guard_rejects_wrong_parent():
    with pytest.raises(RuntimeError, match="parent exited"):
        probe.parent_death_guard(-1)
    # Restore the test worker's previous default rather than leave its signal changed.
    probe.ctypes.CDLL(None).prctl(1, 0, 0, 0, 0)


def guarded_wait(connection, parent_pid):
    probe.parent_death_guard(parent_pid)
    connection.send(os.getpid())
    time.sleep(30)


def nested_parent(connection):
    child = multiprocessing.get_context("spawn").Process(target=guarded_wait, args=(connection, os.getpid()))
    child.start()
    child.join()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux native subprocess contract")
def test_outer_worker_kill_does_not_leave_running_lane():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    parent = context.Process(target=nested_parent, args=(writer,))
    parent.start()
    writer.close()
    child_pid = None
    try:
        # Nested spawn may queue behind 20 test workers; cleanup still has its own 5s bound.
        assert reader.poll(60)
        child_pid = reader.recv()
        parent.terminate()
        parent.join(5)
        for _ in range(100):
            status = probe.Path(f"/proc/{child_pid}/status")
            if not status.exists():
                break
            try:
                text = status.read_text()
            except FileNotFoundError:
                break
            if "State:\tZ" in text:
                break
            time.sleep(.05)
        else:
            pytest.fail("running native lane survived parent termination")
    finally:
        if parent.is_alive():
            parent.kill()
            parent.join(5)
        reader.close()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_digest_tamper_rejected(tmp_path):
    path = tmp_path / "result.json"
    probe.write_json(path, {"status": "ok", "integrity_sha256": "wrong"})
    with pytest.raises(ValueError, match="integrity"):
        probe.load_result(path, {})


def test_zero_conflict_worker_skips_selector_and_step():
    class EmptyLane:
        def __init__(self, *_):
            self.state = {"feasible": True}
        def step(self):
            raise AssertionError("zero-conflict step")
        def result(self):
            return {"feasible": True, "generated": 0}
    job = {"job_id": "zero", "plan_sha256": "x", "arm": "standard", "horizon": 256,
           "case": {"case_id": "zero", "map_id": "m"}}
    with patch.object(probe, "Lane", EmptyLane):
        row = probe.worker(job)
    assert row["repair_calls"] == row["generated"] == 0
    assert row["feasible"] and not row["censored"]


def test_new_protocol_keeps_native_and_default_frozen():
    assert probe.ARMS == ("standard", "rank_sa", "uniform_greedy", "portfolio")
    assert probe.SEEDS == {"round1": 19, "round2": 23}
    assert probe.HORIZON == 256 and probe.WORKERS == 20
