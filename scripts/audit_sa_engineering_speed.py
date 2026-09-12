"""Bounded action-equivalent selector microbenchmark; never a formal TTF run."""
import cProfile
from copy import deepcopy
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from lns2_selector.runtime import online_selection
from scripts import run_sa_independent_confirmation as source
from scripts import run_sa_supervised_confirmation as supervisor

OUT = ROOT / "build/sa-engineering-speed-audit-v1"
MAP_INDICES = (0, 2, 4, 6)
REPEATS = 8


class SingleFullCheckPool(source.pilot.TimedPool):
    """Diagnostic variant: retain final full check, avoid the earlier duplicate."""

    def select(self, env, state, decision):
        original = online_selection.generate_online_candidates
        def generate(*args, **kwargs):
            kwargs["verify_full_state"] = False
            return original(*args, **kwargs)
        # Only used by this single-process diagnostic, never installed in production.
        with patch.object(online_selection, "generate_online_candidates", generate):
            return super().select(env, state, decision)


class CountingEnvironment:
    def __init__(self, env):
        self.env = env
        self.snapshots = 0
        self.corrupt_snapshot = False

    def get_state(self):
        self.snapshots += 1
        result = self.env.get_state()
        if self.corrupt_snapshot:
            result["iteration"] += 1
        return result

    def __getattr__(self, key):
        return getattr(self.env, key)


def existing_breakdown():
    report = read_json(supervisor.OUT / "timed_report.json")
    aggregates = {a: dict(loop=0., reset=0., selection=0., pp=0., steps=0) for a in source.pilot.ARMS}
    for name, h in report["files"].items():
        p = ROOT / name
        if sha256_file(p) != h:
            raise ValueError("formal record changed")
        r = read_json(p)
        a = aggregates[r["arm"]]
        for dst, key in (("loop", "loop_end_seconds"), ("reset", "reset_seconds"),
                         ("selection", "selection_seconds"), ("pp", "pp_seconds")):
            a[dst] += r[key]
        a["steps"] += len(r["events"])
    for a in aggregates.values():
        a["other"] = a["loop"] - a["reset"] - a["selection"] - a["pp"]
    return aggregates


def run():
    if (OUT / "report.json").exists():
        raise ValueError("audit already complete; do not overwrite evidence")
    plan = source.verify()
    maps = sorted({c["map_id"] for c in plan["cases"]})
    cases = [c for c in plan["cases"] if c["map_id"] in {maps[i] for i in MAP_INDICES} and c["solver_seed"] == 101]
    if len(cases) != 8:
        raise ValueError("fixed diagnostic cohort changed")
    config = dict(schema="lns2.sa_engineering_probe.v1", cases=[c["case_id"] for c in cases],
        repeats=REPEATS, maximum_states_per_case=3, pp_diagnostic_cap=5, environment_budget=300,
        formal_ttf=False, frozen_source_plan_sha256=sha256_file(source.OUT / "plan.json"),
        script_sha256=sha256_file(Path(__file__)), variant="single_final_full_fingerprint_plus_revision_checks")
    write_json(OUT / "registration.json", config)
    breakdown = existing_breakdown()
    profile = cProfile.Profile()
    rows = []
    with source.pilot._CollectionRunLock(OUT, sha256_file(OUT / "registration.json"), "engineering-probe"):
        for case_index, case in enumerate(cases):
            job = dict(case=case, config=plan["config"], budget=300)
            env = CountingEnvironment(source.pilot.make_env(job))
            state = source.pilot._plain(env.reset(seed=case["solver_seed"]))
            if source.pilot.state_fingerprint(state) != case["expected_initial_fingerprint"]:
                raise ValueError("diagnostic reset differs from registered initial state")
            base, variant = source.pilot.TimedPool(case), SingleFullCheckPool(case)
            for decision in range(3):
                if state["feasible"]:
                    break
                # Per-call copies avoid giving either cache an unrealistic stable object identity.
                before = source.pilot.state_fingerprint(env.get_state())
                expected = base.select(env, deepcopy(state), decision)
                actual = variant.select(env, deepcopy(state), decision)
                if expected != actual:
                    raise ValueError("candidate metadata, score or choice mismatch")
                samples = {"base": [], "single_check": []}
                counts = {"base": [], "single_check": []}
                for repeat in range(REPEATS):
                    order = (("base", base), ("single_check", variant))
                    if (repeat + case_index + decision) % 2:
                        order = order[::-1]
                    for label, selector in order:
                        snapshot = deepcopy(state)
                        env.snapshots = 0
                        start = time.perf_counter()
                        result = selector.select(env, snapshot, decision)
                        elapsed = time.perf_counter() - start
                        if result != expected:
                            raise ValueError("repeated candidate/score/choice mismatch")
                        samples[label].append(elapsed)
                        counts[label].append(env.snapshots)
                if source.pilot.state_fingerprint(env.get_state()) != before:
                    raise ValueError("probe mutated native state")
                profile.runcall(base.select, env, deepcopy(state), decision)
                # A corrupt final observation must still fail closed in the diagnostic variant.
                if decision == 0:
                    env.corrupt_snapshot = True
                    try:
                        variant.select(env, deepcopy(state), decision)
                    except ValueError as error:
                        if "proposal mutated state" not in str(error):
                            raise
                    else:
                        raise ValueError("final fingerprint guard was bypassed")
                    finally:
                        env.corrupt_snapshot = False
                medians = {k: statistics.median(v) for k, v in samples.items()}
                item = dict(case_id=case["case_id"], map_id=case["map_id"], decision=decision,
                    agents=len(state["agents"]), conflicts=state["num_of_colliding_pairs"],
                    candidate_count=len(expected[1]), selection_sha256=source.pilot.digest(expected),
                    medians=medians, samples=samples, snapshot_counts=counts,
                    improvement_percent=100*(1-medians["single_check"]/medians["base"]))
                rows.append(item)
                write_json(OUT / "progress.json", dict(states=len(rows), last_case=case["case_id"], decision=decision))
                if decision < 2:
                    action = source.pilot.action_for("dual16_sa", case, decision, expected[1], expected[0])
                    temp = source.pilot.temperature(decision)
                    draw = source.pilot.acceptance_draw(source.pilot.seed(case["case_id"], 0, decision, "accept"))
                    step = source.pilot._plain(env.step_experimental_pp(action, 5., "annealed", temp, draw))
                    source.pilot.validate_transition(state, step["observation"], step["metrics"],
                        step["metrics"]["neighborhood"], "annealed", temp, draw)
                    state = step["observation"]
            print(f"case {case_index+1}/{len(cases)} states={len(rows)}", flush=True)
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
    report = dict(schema="lns2.sa_engineering_probe_report.v1", complete=True, formal_ttf=False,
        registered=sha256_file(OUT / "registration.json"), cases=len(cases), states=len(rows),
        all_candidates_scores_choices_equal=True, final_guard_fault_injection_passed=True,
        breakdown=breakdown, rows=rows, profile=stream.getvalue(),
        median_state_improvement_percent=statistics.median(r["improvement_percent"] for r in rows),
        improved_states=sum(r["improvement_percent"] > 0 for r in rows),
        fixed_trajectory_projection_only=True, default_changed=False)
    write_json(OUT / "report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows", "profile")}, indent=2))


if __name__ == "__main__":
    run()
