"""Serial TTF confirmation with frozen loop, supervision and private bindings."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.sa_single_check_runtime import SingleFullCheckPool
from experiments.sa_native_shadow_heat_runtime import NativeShadowHeatPool
from scripts import confirm_sa_single_check_runtime as base
from scripts import audit_sa_native_shadow_heat as component

OUT = ROOT / "build/sa-native-shadow-ttf-v1"
VARIANTS = ("reference_single_full_check", "native_shadow_heat")
COMPONENT_SHA = "2b20f8ef8f529a4fbf5e35b08e601918e816f6eaf65881e81e49ab6b19a95958"
GATE = dict(positive_mean_ttf=True, no_success_loss=True, full_trajectory_equal=True,
            bootstrap_seconds_upper_at_most=0.0)


def verify_component():
    component.verify()
    report = read_json(component.OUT / "report.json")
    if sha256_file(component.OUT / "report.json") != COMPONENT_SHA:
        raise ValueError("component report changed")
    if (report["decision"], report["total"]["calls"], report["shadow"]["calls"], report["replay_steps"]) != (
            "retain_for_independent_ttf_confirmation", 792, 35, 792):
        raise ValueError("component gate incomplete")
    for name, h in report["files"].items():
        path = component.OUT / "cases" / name
        row = read_json(path)
        if sha256_file(path) != h or row.get("status") != "ok" or not row.get("full_prefix_equal"):
            raise ValueError("component evidence changed")
    return report


def register():
    if (OUT / "registration.json").exists():
        raise ValueError("registration already exists")
    report = verify_component()
    r = deepcopy(base.verify())
    names = ["scripts/confirm_sa_native_shadow_ttf.py",
             "tests/evaluation/test_sa_native_shadow_ttf.py", "docs/SA_NATIVE_SHADOW_TTF_PROTOCOL_ZH.md"]
    r["inputs"].update({n: sha256_file(ROOT / n) for n in names})
    r.update(schema="lns2.sa_native_shadow_ttf.v1", variants=list(VARIANTS), gate=GATE,
             component_report_sha256=COMPONENT_SHA,
             source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    write_json(OUT / "registration.json", r)
    identity = sha256_file(OUT / "registration.json")
    files = {}
    for name, h in report["files"].items():
        path = OUT / "preflight" / name
        write_json(path, dict(component_file=(component.OUT / "cases" / name).relative_to(ROOT).as_posix(),
                             component_sha256=h, registration_sha256=identity, already_replayed=True))
        files[name] = sha256_file(path)
    write_json(OUT / "preflight_report.json", dict(complete=True, cases=8, states=792, files=files,
               registration_sha256=identity, reused_equivalence_not_new_timing=True))
    return dict(cases=8, timing_jobs=32, workers=1, budget=60, safe_stop_after_episode=True)


def verify():
    base.verify()
    verify_component()
    r = component.bound(base.verify, OUT=OUT, VARIANTS=VARIANTS)()
    if r.get("component_report_sha256") != COMPONENT_SHA or r.get("gate") != GATE:
        raise ValueError("confirmation gate changed")
    return r


def schedule(r):
    return component.bound(base.schedule, OUT=OUT, ROOT=ROOT, VARIANTS=VARIANTS)(r)


def solver_worker(job):
    # The old worker chooses two selectors; privately replace only those bindings.
    proxy = SimpleNamespace(**dict(vars(base.pilot), TimedPool=SingleFullCheckPool))
    return component.bound(base.solver_worker, VARIANTS=VARIANTS, pilot=proxy,
                           SingleFullCheckPool=NativeShadowHeatPool)(job)


def load_validated(job):
    return component.bound(base.load_validated, OUT=OUT)(job)


def stage_run(job, stage, worker, timeout):
    return component.bound(base.stage_run, OUT=OUT)(job, stage, worker, timeout)


def collect(resume=False):
    return component.bound(base.collect, OUT=OUT, verify=verify, schedule=schedule,
                           load_validated=load_validated, stage_run=stage_run,
                           solver_worker=solver_worker)(resume)


def decision(result):
    a, b = (result["summary"][v] for v in VARIANTS)
    equal = all(p["full_trajectory_equal"] and not p["deadline_divergence"] for p in result["pairs"])
    if not equal or b["successes"] < a["successes"]:
        return "do_not_retain"
    if result["mean_capped_ttf_improvement_percent"] > 0 and result["paired_seconds_map_bootstrap_ci95"][1] <= 0:
        return "retain_engineering_variant_development_only"
    return "ttf_inconclusive_keep_component_evidence_only"


def analyze():
    result = component.bound(base.analyze, OUT=OUT, VARIANTS=VARIANTS, verify=verify,
                             schedule=schedule, load_validated=load_validated)()
    result.update(schema="lns2.sa_native_shadow_ttf_analysis.v1", gate=GATE, decision=decision(result))
    write_json(OUT / "analysis.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "verify", "collect", "resume", "stop", "analyze"))
    phase = p.parse_args().phase
    if phase == "stop":
        write_json(OUT / "STOP_AFTER_EPISODE.json", dict(requested=True))
        result = dict(stop_after_current_episode=True)
    elif phase == "resume":
        result = collect(True)
    elif phase == "verify":
        result = dict(verified=bool(verify()))
    else:
        result = globals()[phase]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
