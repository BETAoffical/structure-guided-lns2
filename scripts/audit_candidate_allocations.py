"""Exact candidate-postprocessing equivalence and bounded component timing."""

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._common import json_fingerprint, read_json, sha256_file, write_json
from experiments import neighborhood_candidates as current

BASELINE = "42caf1a3a4c9e4dca4a9a09f7241f59030e7a350"
SOURCE = "build/sa-selector-component-profile-v1/plan.json"
SOURCE_SHA = "d2e774856b1f39dab32d435bef233ca7812c222a78075e693f1aae4fc0ecae09"
OUT = ROOT / "build/sa-candidate-allocation-v1"


@lru_cache(maxsize=1)
def reference():
    source = subprocess.check_output(
        ["git", "show", BASELINE + ":experiments/neighborhood_candidates.py"], cwd=ROOT)
    namespace = {"__name__": "candidate_allocation_frozen_reference"}
    exec(compile(source, BASELINE + ":neighborhood_candidates.py", "exec"), namespace)
    return namespace["select_representative_neighborhood_groups"], hashlib.sha256(source).hexdigest()


def require_equal(old, new, label):
    if old != new:
        raise ValueError(label + " mismatch")


def bind(function, **overrides):
    value = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                         function.__name__, function.__defaults__, function.__closure__)
    value.__kwdefaults__ = function.__kwdefaults__
    return value


def prepare():
    if (OUT / "plan.json").exists():
        raise ValueError("plan exists; do not replace registration")
    require_equal(sha256_file(ROOT / SOURCE), SOURCE_SHA, "source plan SHA")
    source = read_json(ROOT / SOURCE)
    require_equal((len(source["cases"]), sum(len(c["decisions"]) for c in source["cases"])), (8, 24), "cohort")
    inputs = {SOURCE: SOURCE_SHA}
    for item in source["cases"]:
        inputs[item["record"]["path"]] = item["record"]["sha256"]
        row = item["case"]["row"]
        base = Path(source["config"]["dataset"]["output"]) / row["split"]
        for key in ("map_file", "scenario_file"):
            name = (base / row[key]).as_posix()
            inputs[name] = sha256_file(ROOT / name)
    for p in (ROOT / "artifacts/initlns-closed-loop-controller-v2").glob("*.json"):
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    frozen = source["config"]["frozen"]
    inputs[frozen["native_path"]] = frozen["native_sha256"]
    for name in ("scripts/audit_candidate_allocations.py", "experiments/neighborhood_candidates.py",
                 "tests/runtime/test_candidate_allocations.py", "docs/SA_CANDIDATE_ALLOCATION_PROTOCOL_ZH.md"):
        inputs[name] = sha256_file(ROOT / name)
    for name, digest in inputs.items():
        require_equal(sha256_file(ROOT / name), digest, "registered file " + name)
    plan = dict(schema="lns2.candidate_allocation_audit.v1", baseline_commit=BASELINE,
                reference_source_sha256=reference()[1], inputs=inputs, selector_repeats=4,
                component_rounds=6, component_batch=50, workers=1, states=24,
                source=SOURCE, no_pp=True, no_ttf=True,
                state_contract="restore_saved_paths_with_reset_counters_not_full_historical_replay")
    write_json(OUT / "plan.json", plan)
    return dict(prepared=True, states=24, pp_calls=0, selector_repeats=4, workers=1)


def verify():
    plan = read_json(OUT / "plan.json")
    for name, digest in plan["inputs"].items():
        require_equal(sha256_file(ROOT / name), digest, "registered file " + name)
    require_equal(plan["reference_source_sha256"], reference()[1], "reference source")
    require_equal((plan["selector_repeats"], plan["component_rounds"], plan["component_batch"],
                   plan["workers"], plan["states"], plan["no_pp"], plan["no_ttf"]),
                  (4, 6, 50, 1, 24, True, True), "fixed bounds")
    return plan


def time_components(groups, count, plan):
    old = reference()[0]
    new = current.select_representative_neighborhood_groups
    expected = old(groups, count)
    require_equal(expected, new(groups, count), "representatives")
    timings = {"reference": [], "optimized": []}
    for repeat in range(plan["component_rounds"]):
        order = (("reference", old), ("optimized", new))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, function in order:
            start = time.perf_counter()
            for _ in range(plan["component_batch"]):
                value = function(groups, count)
            timings[name].append((time.perf_counter() - start) / plan["component_batch"])
            require_equal(value, expected, "timed representatives")
    return timings


def checked_states(saved, decisions):
    from experiments.closed_loop_trace_storage import apply_state_delta
    state = saved["initial_state"]
    for decision, event in enumerate(saved["events"]):
        if decision in decisions:
            yield decision, state
        state = apply_state_delta(state, event["delta"])


def make_selector(case, representative):
    from experiments.sa_single_check_runtime import SingleFullCheckPool
    from lns2_selector.runtime.online_selection import generate_online_candidates, score_online_candidates
    captured = {}

    def capture_groups(groups, count):
        captured["groups"], captured["count"] = groups, count
        return representative(groups, count)

    def capture_features(rows, model):
        captured["features"] = rows
        return score_online_candidates(rows, model)

    selector = SingleFullCheckPool(case)
    select = bind(SingleFullCheckPool.select,
                  generate_online_candidates=bind(generate_online_candidates,
                      select_representative_neighborhood_groups=capture_groups),
                  score_online_candidates=capture_features)
    return selector, select, captured


def run():
    plan = verify()
    if (OUT / "started.json").exists():
        raise ValueError("attempt already exists; inspect instead of overwriting/repeating timing")
    write_json(OUT / "started.json", dict(plan_sha256=sha256_file(OUT / "plan.json")))
    from experiments.repair_collection import _make_environment, _plain, state_fingerprint
    from scripts.run_warehouse_repair_confirmation import install_native, environment_config
    source = read_json(ROOT / plan["source"])
    install_native(source["config"])
    output = []
    for case_index, item in enumerate(source["cases"]):
        saved = read_json(ROOT / item["record"]["path"])
        case = item["case"]
        for decision, historical in checked_states(saved, item["decisions"]):
            env = _make_environment(str(ROOT / source["config"]["dataset"]["output"]), case["row"],
                                   dict(environment_config(source["config"]), time_limit=120), "Adaptive")
            agents = sorted(historical["agents"], key=lambda a: a["id"])
            require_equal([a["id"] for a in agents], list(range(len(agents))), "native agent IDs")
            paths = [a["path"] for a in agents]
            state = _plain(env.reset_paths(paths, seed=case["solver_seed"]))
            require_equal([a["path"] for a in state["agents"]], paths, "restored paths")
            require_equal(state["conflict_edges"], historical["conflict_edges"], "restored conflicts")
            require_equal(state["low_level"]["runs"], 0, "no low-level search")
            before = state_fingerprint(state)
            old = make_selector(case, reference()[0])
            new = make_selector(case, current.select_representative_neighborhood_groups)
            warm = []
            for selector, function, _ in (old, new):
                warm.append(function(selector, env, deepcopy(state), decision))
            require_equal(warm[0], warm[1], "candidate metadata/scores/selection")
            require_equal(old[2]["features"], new[2]["features"], "feature vectors")
            require_equal(old[2]["groups"], new[2]["groups"], "native groups")
            groups = deepcopy(old[2]["groups"])
            component = time_components(groups, old[2]["count"], plan)
            full = {"reference": [], "optimized": []}
            for repeat in range(plan["selector_repeats"]):
                order = (("reference", old), ("optimized", new))
                if (repeat + case_index) % 2:
                    order = tuple(reversed(order))
                for name, (selector, function, _) in order:
                    fresh_state = deepcopy(state)
                    start = time.perf_counter()
                    selected = function(selector, env, fresh_state, decision)
                    full[name].append(time.perf_counter() - start)
                    require_equal(selected, warm[0], "repeated choice")
                require_equal(old[2]["features"], new[2]["features"], "repeated features")
            require_equal(state_fingerprint(env.get_state()), before, "proposal read-only state")
            row = dict(case_id=case["case_id"], decision=decision, groups=len(groups),
                       requests=sum(len(g["sources"]) for g in groups),
                       candidates=len(warm[0][1]), historical_fingerprint=state_fingerprint(historical),
                       restored_fingerprint=before, candidate_sha256=json_fingerprint(warm[0]),
                       feature_sha256=json_fingerprint(old[2]["features"]),
                       component_seconds=component, selector_seconds=full, exact=True, pp_calls=0)
            write_json(OUT / "states" / f"{case_index:02d}-{decision:04d}.json", row)
            output.append(row)
            print(f"state {len(output)}/24 exact; groups={len(groups)}", flush=True)
    require_equal(len(output), plan["states"], "complete states")
    summary = {}
    for metric in ("component_seconds", "selector_seconds"):
        old = [statistics.median(r[metric]["reference"]) for r in output]
        new = [statistics.median(r[metric]["optimized"]) for r in output]
        summary[metric] = dict(reference_median_sum=sum(old), optimized_median_sum=sum(new),
                               reduction_fraction=1 - sum(new) / sum(old),
                               faster_states=sum(b < a for a, b in zip(old, new)))
    report = dict(schema="lns2.candidate_allocation_result.v1", plan_sha256=sha256_file(OUT / "plan.json"),
                  state_files={p.relative_to(OUT).as_posix(): sha256_file(p) for p in sorted((OUT / "states").glob("*.json"))},
                  states=24, pp_calls=0, mismatches=0, no_ttf=True, summary=summary,
                  decision="component_only_no_end_to_end_claim")
    write_json(OUT / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "run"))
    args = parser.parse_args()
    result = {"prepare": prepare, "verify": verify, "run": run}[args.phase]()
    print(json.dumps(result if args.phase != "verify" else dict(verified=True), indent=2))


if __name__ == "__main__":
    main()
