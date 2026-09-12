"""Bounded frozen-prefix selector profiling, not an end-to-end TTF trial."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.sa_single_check_runtime import SingleFullCheckPool
from scripts import confirm_sa_single_check_runtime as prior

pilot = prior.pilot
OUT = ROOT / "build/sa-selector-component-profile-v1"


def positions(length):
    if type(length) is not int or length < 1:
        raise ValueError("nonempty historical trajectory required")
    return sorted({0, (length - 1) // 2, length - 1})


class MethodProxy:
    def __init__(self, obj, owner, methods):
        self.obj, self.owner, self.methods = obj, owner, methods

    def __getattr__(self, name):
        value = getattr(self.obj, name)
        return self.owner.wrap(self.methods[name], value) if name in self.methods else value


class ProfilePool(SingleFullCheckPool):
    """Private dependency bindings preserve the frozen select bytecode."""

    def __init__(self, case, measured=True):
        super().__init__(case)
        self.measured = measured
        self.parts, self.details, self.features = Counter(), {}, None
        original = SingleFullCheckPool.select
        bindings = dict(original.__globals__)
        score = bindings["score_online_candidates"]

        def capture(rows, model):
            self.features = rows
            return score(rows, model)

        bindings["score_online_candidates"] = self.wrap("scoring", capture)
        if measured:
            for name, label in (("state_fingerprint", "fingerprints"),
                                ("generate_online_candidates", "proposal"),
                                ("generate_structshell_dual16_runtime_candidates", "dual16_augmentation")):
                bindings[name] = self.wrap(label, bindings[name])
            for name, label, methods in (
                ("OnlineFeatureEngine", "feature_engine_init", {"prepare":"feature_prepare", "realized_rows":"feature_rows"}),
                ("TopologyAnalysisCache", "topology_init", {"prepare":"topology_prepare"})):
                constructor = bindings[name]
                def make(*args, _ctor=constructor, _label=label, _methods=methods, **kwargs):
                    return MethodProxy(self.wrap(_label, _ctor)(*args, **kwargs), self, _methods)
                bindings[name] = make
        self.frozen_call = FunctionType(original.__code__, bindings, original.__name__,
                                       original.__defaults__, original.__closure__)

    def wrap(self, label, function):
        def wrapped(*args, **kwargs):
            if not self.measured:
                return function(*args, **kwargs)
            started = time.perf_counter()
            value = function(*args, **kwargs)
            self.parts[label] += time.perf_counter() - started
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
                self.details[label] = {k:v for k,v in value[1].items() if k.endswith("seconds") and isinstance(v,(float,int))}
            return value
        return wrapped

    def select(self, env, state, decision):
        self.parts, self.details = Counter(), {}
        wrapped_env = MethodProxy(env, self, {"get_state":"final_get_state"}) if self.measured else env
        started = time.perf_counter()
        result = self.frozen_call(self, wrapped_env, state, decision)
        self.total_seconds = time.perf_counter() - started
        remainder = self.total_seconds - sum(self.parts.values())
        if remainder < -1e-7:
            raise ValueError("component timers overlap")
        self.parts["other_and_instrumentation"] = max(0., remainder)
        return result


def prepare():
    if (OUT / "plan.json").exists():
        raise ValueError("plan already exists")
    registration = prior.verify()
    cases = []
    for case in registration["cases"]:
        record = registration["records"][case["case_id"]]
        if sha256_file(ROOT/record["path"]) != record["sha256"]:
            raise ValueError("historical trace changed")
        row = read_json(ROOT/record["path"])
        cases.append(dict(case=case, record=record, decisions=positions(len(row["events"])),
                          historical_steps=len(row["events"])))
    files = ["scripts/profile_sa_selector_components.py", "tests/evaluation/test_sa_selector_components.py",
             "docs/SA_SELECTOR_COMPONENT_PROFILE_PROTOCOL_ZH.md", "experiments/sa_single_check_runtime.py"]
    write_json(OUT/"plan.json", dict(schema="lns2.sa_selector_component_profile.v1", cases=cases,
        config=registration["config"], inputs={n:sha256_file(ROOT/n) for n in files},
        previous_registration_sha256=sha256_file(prior.OUT/"registration.json"),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        workers=1, repeats_after_first_visit=2, job_fuse_seconds=600, replay_pp_seconds=300.,
        selected_states=sum(len(c["decisions"]) for c in cases), no_ttf=True, no_default_change=True))
    return dict(cases=len(cases), states=sum(len(c["decisions"]) for c in cases),
                prefix_repairs=sum(c["historical_steps"] for c in cases), selector_measurements=72, workers=1)


def verify():
    prior.verify()
    plan = read_json(OUT/"plan.json")
    if plan["previous_registration_sha256"] != sha256_file(prior.OUT/"registration.json"):
        raise ValueError("registration changed")
    for name, h in plan["inputs"].items():
        if sha256_file(ROOT/name) != h:
            raise ValueError("profiling source changed: " + name)
    if (plan["workers"], plan["repeats_after_first_visit"], plan["selected_states"]) != (1,2,24):
        raise ValueError("fixed sampling or repeat budget changed")
    return plan


def worker(job):
    item, plan = job["item"], job["plan"]
    record = item["record"]
    if sha256_file(ROOT/record["path"]) != record["sha256"]:
        raise ValueError("historical source changed")
    saved = read_json(ROOT/record["path"])
    env = pilot.make_env(dict(job,budget=3000.))
    state = pilot._plain(env.reset(seed=job["case"]["solver_seed"]))
    expected = saved["initial_state"]
    reference, measured = ProfilePool(job["case"],False), ProfilePool(job["case"],True)
    samples = []
    if pilot.state_fingerprint(state) != pilot.state_fingerprint(expected):
        raise ValueError("initial replay mismatch")
    for d, event in enumerate(saved["events"]):
        if d in item["decisions"]:
            order = (reference,measured) if (job["case_index"] + item["decisions"].index(d)) % 2 == 0 else (measured,reference)
            values = [selector.select(env,deepcopy(state),d) for selector in order]
            if values[0] != values[1] or values[0] != (event["selected_index"],event["pool"]) or reference.features != measured.features:
                raise ValueError("features/candidates/scores/action mismatch")
            for repeat in range(plan["repeats_after_first_visit"] + 1):
                if repeat:
                    value = measured.select(env,deepcopy(state),d)
                    if value != values[0] or measured.features != reference.features:
                        raise ValueError("repeated selection mismatch")
                samples.append(dict(decision=d, repeat=repeat, first_visit=repeat==0,
                    initialization_visit=d==0 and repeat==0, profiled_first=order[0] is measured,
                    conflicts=state["num_of_colliding_pairs"], candidates=len(values[0][1]),
                    state_sha256=pilot.state_fingerprint(state), feature_sha256=pilot.digest(measured.features),
                    selection_sha256=pilot.digest(values[0]), total_seconds=measured.total_seconds,
                    parts=dict(measured.parts), nested_details=deepcopy(measured.details)))
            if pilot.state_fingerprint(env.get_state()) != pilot.state_fingerprint(state):
                raise ValueError("profiling changed state")
        step = pilot._plain(env.step_experimental_pp(event["action"],plan["replay_pp_seconds"],
                                                   "annealed",event["temperature"],event["uniform"]))
        state = step["observation"]
        expected = pilot.apply_state_delta(expected,event["delta"])
        if pilot.state_fingerprint(state) != pilot.state_fingerprint(expected):
            raise ValueError(f"prefix replay mismatch at {d}")
        for key in ("repair_order","neighborhood","replan_success","pp_failure_reason"):
            if step["metrics"][key] != event["metrics"][key]:
                raise ValueError("PP identity mismatch: " + key)
    pilot.validate_final(state)
    return dict(status="ok",job_id=job["job_id"],plan_sha256=job["plan_sha256"],
                samples=samples, replay_steps=len(saved["events"]), full_prefix_equal=True, no_ttf=True)


def collect(resume=False):
    plan = verify()
    identity = sha256_file(OUT/"plan.json")
    with pilot._CollectionRunLock(OUT, identity, "serial-component-profile"):
        if resume:
            (OUT/"STOP_AFTER_CASE.json").unlink(missing_ok=True)
        results = []
        for i, item in enumerate(plan["cases"]):
            job = dict(job_id=item["case"]["case_id"],case=item["case"],config=plan["config"],
                       item=item,case_index=i,plan=plan,plan_sha256=identity)
            path = OUT/"cases"/(job["job_id"]+".json")
            if path.exists():
                row = read_json(path)
                if row.get("integrity_sha256") != pilot.digest({k:v for k,v in row.items() if k!="integrity_sha256"}) or row.get("plan_sha256") != identity or row.get("status") != "ok" or row.get("job_id") != job["job_id"]:
                    raise ValueError("prior result invalid; no automatic retry")
            else:
                if (OUT/"STOP_AFTER_CASE.json").exists():
                    return dict(paused=True,completed=i)
                attempt = OUT/"attempts"/(job["job_id"]+".json")
                if attempt.exists():
                    raise ValueError("interrupted attempt without result; inspect before retry")
                write_json(attempt,dict(job_id=job["job_id"],plan_sha256=identity,started=True))
                print(f"case {i+1}/8 prefix + 3-state profile",flush=True)
                def failure(j,status,error):
                    return dict(status=status,job_id=j["job_id"],plan_sha256=identity,error=str(error))
                def save(row):
                    row["integrity_sha256"] = pilot.digest(row)
                    write_json(path,row)
                batch = pilot._run_jobs(worker,[job],workers=1,phase="profile",output_root=OUT,
                    run_fingerprint=identity,timeout_seconds=plan["job_fuse_seconds"],failure_result=failure,
                    on_result=save,stop_on_failure=True)
                row = read_json(path)
                if len(batch)!=1 or row["status"]!="ok":
                    raise ValueError("profile failed; inspect preserved result")
            results.append(row)
            write_json(OUT/"progress.json",dict(completed=i+1,total=8))
        report = dict(complete=True,plan_sha256=identity,cases=8,
            states=sum(len(c["decisions"]) for c in plan["cases"]),
            measurements=sum(len(r["samples"]) for r in results),
            replay_steps=sum(r["replay_steps"] for r in results),no_ttf=True,
            files={r["job_id"]+".json":sha256_file(OUT/"cases"/(r["job_id"]+".json")) for r in results})
        for label, predicate in (("first_visits",lambda s:s["first_visit"]),
                                 ("first_visits_excluding_initialization",lambda s:s["first_visit"] and not s["initialization_visit"]),
                                 ("repeated_visits",lambda s:not s["first_visit"])):
            samples = [s for r in results for s in r["samples"] if predicate(s)]
            totals = sum((Counter(s["parts"]) for s in samples),Counter())
            total = sum(s["total_seconds"] for s in samples)
            report[label] = dict(measurements=len(samples),seconds=total,parts=dict(totals),
                                 percentages={k:100*v/total for k,v in totals.items()})
        write_json(OUT/"report.json",report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","collect","resume","stop"))
    phase = parser.parse_args().phase
    if phase=="stop":
        write_json(OUT/"STOP_AFTER_CASE.json",dict(requested=True)); result=dict(stop_after_case=True)
    elif phase=="verify": result=dict(verified=bool(verify()))
    elif phase=="resume": result=collect(resume=True)
    else: result=globals()[phase]()
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
