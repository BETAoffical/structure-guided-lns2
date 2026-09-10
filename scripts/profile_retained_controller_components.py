"""Bounded, solver-free attribution on the six retained development states."""
from __future__ import annotations

import argparse
import cProfile
import gzip
import hashlib
import json
import multiprocessing
import pstats
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file as digest, write_json

PACKAGE = "build/path-quality-pressure-deadline-recovery-v1/review-20260909/mechanism-preparation-v1/manifest.json"
EVIDENCE = "artifacts/initlns-research-decision-review-v1/evidence.json"
PROFILE = "realized_dynamic"
NATIVE = "build/linux/proposal-deadline-fix-validation/lns2_env.cpython-310-x86_64-linux-gnu.so"
BUNDLE = "artifacts/initlns-closed-loop-controller-v2"
CANDIDATE_FIELDS = (
    "candidate_id", "agents", "actual_size", "selection_families",
    "proposal_count_by_family", "seed_agents", "proposal_seeds",
    "selection_rank_by_family", "hybridstructpool_provenance",
)


def read(relative):
    return read_json(ROOT / relative)


def semantic_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_files(expected):
    for name, expected_sha in expected.items():
        if digest(ROOT / name) != expected_sha:
            raise ValueError("input SHA mismatch: " + name)


def input_candidates(rows):
    return [{name: row[name] for name in CANDIDATE_FIELDS if name in row} for row in rows]


def profile_call(function):
    profiler = cProfile.Profile()
    profiler.enable()
    result = function()
    profiler.disable()
    entries = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        path = Path(filename)
        if path.is_absolute():
            filename = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        entries.append(dict(file=filename, line=line, function=name, calls=calls,
                            primitive_calls=primitive, self_seconds=own, cumulative_seconds=cumulative))
    return result, sorted(entries, key=lambda row: -row["cumulative_seconds"])


def measure(function):
    times = []
    outputs = []
    for _ in range(3):
        started = time.perf_counter()
        outputs.append(function())
        times.append(time.perf_counter() - started)
    return outputs, dict(seconds=times, median_seconds=statistics.median(times))


def run_case(case):
    sys.path.insert(0, str(ROOT / Path(NATIVE).parent))
    sys.path.insert(0, str(ROOT))
    import lns2_env
    package = read(PACKAGE)
    if Path(lns2_env.__file__).resolve() != (ROOT / NATIVE).resolve():
        raise ValueError("unexpected imported native module")
    verify_files({NATIVE: package["files"][NATIVE]})
    from experiments.compact_controller_model import load_controller_bundle
    from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
    from lns2_selector.runtime.online_selection import feature_range_diagnostic, score_online_candidates
    from experiments.repair_collection import state_fingerprint

    state = read(case["state_file"])
    if state_fingerprint(state) != case["full_state_fingerprint"]:
        raise ValueError("saved state fingerprint mismatch")
    with gzip.open(ROOT / case["source_trace"], "rt", encoding="utf-8") as stream:
        transition = next(json.loads(line) for line in stream
                          if json.loads(line).get("event") == "transition"
                          and json.loads(line).get("decision_index") == case["decision_index"])
    if transition["before_fingerprint"] != case["full_state_fingerprint"]:
        raise ValueError("trace state binding mismatch")
    original = transition["controller"]["candidate_pool"]
    candidates = input_candidates(original)
    if len(candidates) != case["candidate_count"]:
        raise ValueError("candidate count mismatch")
    snapshot = semantic_digest([state, candidates])
    bundle = load_controller_bundle(ROOT / BUNDLE)
    model = bundle.main_models[PROFILE]
    engine = OnlineFeatureEngine(state, backend="native", dense_output=True,
                                required_features={PROFILE: model.base_feature_names})
    stages = {}
    profiles = {}
    # Analyze all six states for mechanism attribution; only Dual16 uses this
    # topology preparation online. V2 measurements here are controls, not costs.
    started = time.perf_counter()
    cache = TopologyAnalysisCache(state, static_grid=engine.static_grid, backend="native")
    stages["topology_initial"] = dict(seconds=time.perf_counter() - started)
    analyses, stages["topology_unchanged_prepare"] = measure(lambda: cache.prepare(state, changed_agents=[]))
    if any(value != analyses[0] for value in analyses):
        raise ValueError("unchanged topology result differs")
    _, profiles["topology_unchanged_prepare"] = profile_call(lambda: cache.prepare(state, changed_agents=[]))
    if case["controller"] == "dual16":
        engine.prepare(state, changed_agents=[], prepared_native_analysis=cache.last_native_prepared)
    else:
        engine.prepare(state, changed_agents=[])
    output, stages["feature_rows"] = measure(lambda: engine.realized_rows(candidates, state_hash=case["full_state_fingerprint"]))
    rows = output[0][0]
    if any(value[0] != rows for value in output):
        raise ValueError("repeated feature vectors differ")
    _, profiles["feature_rows"] = profile_call(lambda: engine.realized_rows(candidates, state_hash=case["full_state_fingerprint"]))
    scored, stages["ranking"] = measure(lambda: score_online_candidates(rows, model))
    index, scores, margin = scored[0]
    if any(value != scored[0] for value in scored):
        raise ValueError("repeated ranking differs")
    selected = candidates[index]["candidate_id"]
    delta = max(abs(score - float(row["score"])) for score, row in zip(scores, original))
    if selected != case["selected_candidate_id"] or delta > 1e-10:
        raise ValueError(f"historical ranking mismatch: {selected}, delta={delta}")

    def annotate():
        diagnostics = [feature_range_diagnostic(row, PROFILE, bundle.main_ranges[PROFILE]) for row in rows]
        feature_range_diagnostic(rows[index], PROFILE, bundle.main_ranges[PROFILE])
        feature_range_diagnostic(rows[index], PROFILE, bundle.main_ranges[PROFILE])
        return [{**candidate, "retained": True, "score": scores[i],
                 "feature_out_of_range_fraction": diagnostics[i]["outside_fraction"]}
                for i, candidate in enumerate(candidates)]

    annotations, stages["candidate_annotation"] = measure(annotate)
    for current, old in zip(annotations[0], original):
        if current["feature_out_of_range_fraction"] != old["feature_out_of_range_fraction"]:
            raise ValueError("historical range diagnostic mismatch")
    _, profiles["candidate_annotation"] = profile_call(annotate)
    if semantic_digest([state, candidates]) != snapshot:
        raise ValueError("pure component mutated its inputs")
    return dict(case_id=case["case_id"], controller=case["controller"], role=case["role"],
                agents=len(state["agents"]), conflicts=state["num_of_colliding_pairs"],
                candidates=len(candidates), selected=selected, maximum_historical_score_delta=delta,
                feature_digest=semantic_digest(rows), inputs_unchanged=True,
                topology_online=case["controller"] == "dual16", stages=stages, profiles=profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "build"):
        raise ValueError("output must be under ignored build")
    output.mkdir(parents=True, exist_ok=False)
    package = read(PACKAGE)
    admission = read(EVIDENCE)["runtime_admission"]
    expected = dict(admission["source_registration"]["matched_source_files"])
    expected[admission["source_registration"]["path"]] = admission["source_registration"]["sha256"]
    expected[NATIVE] = package["files"][NATIVE]
    expected[PACKAGE] = digest(ROOT / PACKAGE)
    expected[EVIDENCE] = digest(ROOT / EVIDENCE)
    cases = package["cases"]
    if len(cases) != 6 or sum(c["role"] == "fast_control" for c in cases) != 2:
        raise ValueError("fixed six-case scope changed")
    for case in cases:
        expected[case["state_file"]] = package["files"][case["state_file"]]
        expected[case["source_trace"]] = case["source_trace_sha256"]
    expected.update({name: sha for name, sha in package["files"].items() if name.startswith(BUNDLE + "/")})
    verify_files(expected)
    protocol = dict(schema="lns2.component_attribution.v1", workers=6, repeats=3,
                    cases=[c["case_id"] for c in cases], input_sha256=expected,
                    script_sha256=digest(Path(__file__)), solver_calls=0,
                    boundary="Parallel pure-function profiling only; not TTF or counterfactual speedup. No candidate generation, reset, repair, prefix replay or training.")
    write_json(output / "protocol.json", protocol)
    status = dict(status="running")
    write_json(output / "run_status.json", status)
    try:
        results = []
        with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [(c, pool.submit(run_case, c)) for c in cases]
            for case, future in futures:
                result = future.result(timeout=120)
                results.append(result)
                write_json(output / (case["case_id"] + ".json"), result)
                print(case["case_id"], "verified", flush=True)
        verify_files(expected)
        write_json(output / "report.json", dict(protocol=protocol, results=results))
        status = dict(status="completed", cases=len(results), solver_calls=0)
    except BaseException as error:
        status = dict(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "run_status.json", status)


if __name__ == "__main__":
    main()
