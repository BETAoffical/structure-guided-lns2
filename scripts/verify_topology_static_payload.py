"""Compare only static-payload reuse against the frozen pre-edit implementation."""
import argparse
import hashlib
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import profile_retained_controller_components as retained
from experiments._common import write_json

BASELINE = "e633e9b02fc68279c8f5230aef559c76e2cfb9ab"
ENGINE = "experiments/online_feature_engine.py"
EVIDENCE = "artifacts/initlns-runtime-component-attribution-v1/evidence.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "build"):
        raise ValueError("output must be under build")
    evidence = retained.read(EVIDENCE)
    retained.verify_files({evidence[k]["path"]: evidence[k]["sha256"]
                           for k in ("report", "protocol", "script")})
    previous = retained.read(evidence["protocol"]["path"])
    expected = dict(previous["input_sha256"])
    old_source = subprocess.check_output(["git", "show", BASELINE + ":" + ENGINE], cwd=ROOT)
    old_sha = hashlib.sha256(old_source).hexdigest()
    if old_sha != expected[ENGINE]:
        raise ValueError("baseline source differs from original registration")
    expected[ENGINE] = retained.digest(ROOT / ENGINE)
    expected["scripts/verify_topology_static_payload.py"] = retained.digest(Path(__file__))
    retained.verify_files(expected)
    sys.path.insert(0, str(ROOT / Path(retained.NATIVE).parent))
    import experiments.online_feature_engine as current
    baseline = types.ModuleType("_retained_topology_baseline")
    baseline.__file__ = str(ROOT / ENGINE)
    sys.modules[baseline.__name__] = baseline
    exec(compile(old_source, "retained_baseline/" + ENGINE, "exec"), baseline.__dict__)
    cases = retained.read(retained.PACKAGE)["cases"]
    if [case["case_id"] for case in cases] != previous["cases"]:
        raise ValueError("fixed six-state cohort changed")
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema="lns2.static_payload_reuse_check.v1", baseline_commit=BASELINE,
                    baseline_source_sha256=old_sha, input_sha256=expected,
                    workers=1, repeats_per_stage=3, alternating_order=True,
                    cases=previous["cases"], solver_calls=0,
                    boundary="Serial component comparison on retained states, not full episode TTF. No candidate generation or solver replay.")
    write_json(output / "protocol.json", protocol)
    write_json(output / "run_status.json", dict(status="running"))
    results = []
    try:
        for position, case in enumerate(cases):
            classes = {"baseline": baseline.TopologyAnalysisCache,
                       "reuse": current.TopologyAnalysisCache}
            order = ["baseline", "reuse"] if position % 2 == 0 else ["reuse", "baseline"]
            runs = {}
            for name in order:
                with patch.object(current, "TopologyAnalysisCache", classes[name]):
                    runs[name] = retained.run_case(case)
            before, after = runs["baseline"], runs["reuse"]
            for name in ("feature_digest", "selected", "candidates", "inputs_unchanged"):
                if before[name] != after[name]:
                    raise ValueError("component semantics differ: " + name)
            if before["maximum_historical_score_delta"] != 0 or after["maximum_historical_score_delta"] != 0:
                raise ValueError("historical scores are not exactly equal")
            profiles = after["profiles"]["topology_unchanged_prepare"]
            if any(row["function"] == "_native_static_payload" for row in profiles):
                raise ValueError("static payload is still rebuilt on unchanged prepare")
            old_time = before["stages"]["topology_unchanged_prepare"]["median_seconds"]
            new_time = after["stages"]["topology_unchanged_prepare"]["median_seconds"]
            result = dict(case_id=case["case_id"], controller=case["controller"], role=case["role"],
                          order=order, runs=runs, topology_improvement_percent=100 * (1 - new_time / old_time))
            results.append(result)
            write_json(output / (case["case_id"] + ".json"), result)
            print(case["case_id"], "exact", round(result["topology_improvement_percent"], 2), flush=True)
        retained.verify_files(expected)
        write_json(output / "report.json", dict(protocol=protocol, results=results))
        write_json(output / "run_status.json", dict(status="completed", cases=len(results), solver_calls=0))
    except BaseException as error:
        write_json(output / "run_status.json", dict(status="failed", error=str(error)))
        raise


if __name__ == "__main__":
    main()
