import ast
from copy import deepcopy
from contextlib import nullcontext
import inspect
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest

from experiments import sa_single_check_runtime as runtime
from scripts import confirm_sa_single_check_runtime as run
from tests.runtime.test_closed_loop_confirmation import make_state
from lns2_selector.runtime import online_selection as online


def test_selector_ast_changes_only_one_flag():
    def normalized(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        body = tree.body[0].body
        tree.body[0].body = [n for n in body if not isinstance(n, ast.ImportFrom)]
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "verify_full_state":
                node.value = ast.Constant(True)
        return ast.dump(tree, include_attributes=False)
    assert normalized(runtime.SingleFullCheckPool.select) == normalized(run.pilot.TimedPool.select)


def test_private_bindings_preserve_frozen_bytecode_and_globals():
    def frozen(job):
        selected = TimedPool(job)
        validate_result(selected, job)
        return selected
    globals_before = dict(frozen.__globals__)
    captured = []
    a = runtime.isolated_worker(frozen, lambda j: ("a", j), lambda r, j: captured.append(r))
    b = runtime.isolated_worker(frozen, lambda j: ("b", j), lambda r, j: None)
    assert a.__code__ is b.__code__ is frozen.__code__
    assert a(3) == ("a", 3) and b(3) == ("b", 3)
    assert captured == [("a", 3)]
    assert frozen.__globals__ == globals_before


@pytest.mark.parametrize("revision", [True, False])
def test_revision_and_no_revision_full_fallback(revision):
    state = make_state()
    calls = []
    env = SimpleNamespace(get_state=lambda: (calls.append(1) or deepcopy(state)),
        propose_batch=lambda actions: [dict(action_valid=True, generated=True, neighborhood=[0, 1]) for _ in actions])
    if revision:
        env.get_state_revision = lambda: 7
    _, metrics = online.generate_online_candidates(env, state, task_id="t", solver_seed=1,
        decision_index=0, proposal_config=dict(max_seed_agents=1, heuristics=["collision"],
            neighborhood_sizes=[4], trials=1, candidates_per_family=1), verify_full_state=False)
    assert len(calls) == (0 if revision else 1)


def test_revision_mismatch_is_not_skipped():
    state = make_state()
    revisions = iter([1, 2])
    env = SimpleNamespace(get_state_revision=lambda: next(revisions), get_state=lambda: deepcopy(state),
        propose_batch=lambda actions: [dict(action_valid=True, generated=True, neighborhood=[0, 1]) for _ in actions])
    with pytest.raises(online.ClosedLoopExecutionError, match="revision"):
        online.generate_online_candidates(env, state, task_id="t", solver_seed=1, decision_index=0,
            proposal_config=dict(max_seed_agents=1, heuristics=["collision"], neighborhood_sizes=[4],
                                 trials=1, candidates_per_family=1), verify_full_state=False)


def test_outer_guard_rejects_mutation_and_empty_scoring(monkeypatch):
    state = make_state()
    env = SimpleNamespace(get_state=lambda: {**state, "iteration": state["iteration"] + 1})
    pool = object.__new__(runtime.SingleFullCheckPool)
    pool.case = dict(task_id="t", solver_seed=1, proposal={})
    pool.model = object()
    pool.topology = SimpleNamespace(analysis={}, last_native_prepared=None, prepare=lambda s: None)
    pool.engine = SimpleNamespace(prepare=lambda *a, **k: None, realized_rows=lambda *a, **k: ([], {}))
    monkeypatch.setattr(runtime, "generate_online_candidates", lambda *a, **k: ([], {}))
    monkeypatch.setattr(runtime, "generate_structshell_dual16_runtime_candidates", lambda *a, **k: SimpleNamespace(candidates=[]))
    monkeypatch.setattr(runtime, "score_online_candidates", lambda *a: (0, [], {}))
    with pytest.raises(ValueError, match="proposal mutated state"):
        pool.select(env, state, 0)
    with pytest.raises((ValueError, RuntimeError)):
        online.score_online_candidates([], pool.model)


def test_schedule_balanced_and_action_seeds_independent_of_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    run.write_json(tmp_path / "registration.json", {})
    cases = [dict(case_id=f"c{i}") for i in range(8)]
    jobs = run.schedule(dict(rounds=2, cases=cases, config={}, budget=60))
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 32
    assert all(j["arm"] == "dual16_sa" and j["max_steps"] is None for j in jobs)
    for i in range(8):
        first = [j["runtime_variant"] for j in jobs if j["case"]["case_id"] == f"c{i}" and j["repeat"] == 0]
        second = [j["runtime_variant"] for j in jobs if j["case"]["case_id"] == f"c{i}" and j["repeat"] == 1]
        assert first == second[::-1]


def test_capture_pending_audit_and_no_global_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    def frozen(job):
        r = dict(status="ok", ttf_seconds=1.25)
        validate_result(r, job)
        r["validated_seconds"] = 999
    monkeypatch.setattr(run.pilot, "worker", frozen)
    old = run.pilot.validate_result
    j = dict(capture_path="raw.json", runtime_variant="single_full_check", repeat=0, job_id="x")
    run.solver_worker(j)
    result = run.read_json(tmp_path / "raw.json")
    assert result["ttf_seconds"] == 1.25 and "validated_seconds" not in result
    assert run.pilot.validate_result is old
    with pytest.raises(ValueError, match="raw exists"):
        run.solver_worker(j)


def test_stop_does_not_start_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run, "verify", lambda: {})
    monkeypatch.setattr(run.pilot, "_CollectionRunLock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(run, "schedule", lambda r: [dict(job_id="x")])
    run.write_json(tmp_path / "registration.json", {})
    run.write_json(tmp_path / "preflight_report.json", dict(complete=True, cases=8, files={},
        registration_sha256=run.sha256_file(tmp_path / "registration.json")))
    run.write_json(tmp_path / "STOP_AFTER_EPISODE.json", {})
    assert run.collect() == dict(paused=True, completed=0)


@pytest.mark.parametrize("variant", run.VARIANTS)
def test_native_fixed_steps_equal_frozen_loop(tmp_path, variant):
    native = pytest.importorskip("lns2_env")
    if Path(native.__file__).resolve() != (run.ROOT / run.pilot.NATIVE).resolve():
        pytest.skip("requires frozen SA native")
    plan = run.read_json(run.source.OUT / "plan.json")
    case = plan["cases"][0]
    job = dict(case=case, config=plan["config"], arm="dual16_sa", phase="smoke", budget=60.,
        max_steps=2, job_id="native", plan_sha256="test", repeat=0, runtime_variant=variant,
        capture_path=str(tmp_path / "raw.json"))
    a = run.pilot.worker(job)
    run.solver_worker(job)
    b = run.read_json(tmp_path / "raw.json")
    run.pilot.validate_result(b, job)
    assert run.compare_trajectories(a, b)["full_trajectory_equal"]
