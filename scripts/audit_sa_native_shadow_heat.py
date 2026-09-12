"""Reuse the frozen paired component protocol with one private runtime binding."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.sa_native_shadow_heat_runtime import NativeShadowHeatPool
from scripts import audit_sa_incremental_heat as base

OUT=ROOT/"build/sa-native-shadow-heat-audit-v2"
PARENT_SHA="2aaf946ce4777c7eff56e3dc6aa8420ebf914b537d8353918c54f203d7629f95"


def bound(function, **overrides):
    return FunctionType(function.__code__,dict(function.__globals__,**overrides),
                        function.__name__,function.__defaults__,function.__closure__)


def prepare():
    if (OUT/"plan.json").exists(): raise ValueError("plan already exists")
    plan=deepcopy(base.verify())
    if sha256_file(base.OUT/"plan.json")!=PARENT_SHA: raise ValueError("parent protocol changed")
    names=["experiments/sa_native_shadow_heat_runtime.py","scripts/audit_sa_native_shadow_heat.py",
           "tests/evaluation/test_sa_native_shadow_heat.py","docs/SA_NATIVE_SHADOW_HEAT_PROTOCOL_ZH.md"]
    plan["inputs"].update({n:sha256_file(ROOT/n) for n in names})
    plan.update(schema="lns2.sa_native_shadow_heat.v2",ancestor_plan_sha256=PARENT_SHA,
        runtime="NativeShadowHeatPool",reference_runtime="SingleFullCheckPool",
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    write_json(OUT/"plan.json",plan)
    return dict(cases=8,paired_selections=792,prefix_repairs=792,expected_shadow_checks=35,workers=1,no_ttf=True)


def verify():
    base.verify()
    if sha256_file(base.OUT/"plan.json")!=PARENT_SHA: raise ValueError("parent protocol changed")
    plan=bound(base.verify,OUT=OUT)()
    if (plan["ancestor_plan_sha256"],plan["runtime"],plan["reference_runtime"])!=(PARENT_SHA,"NativeShadowHeatPool","SingleFullCheckPool"):
        raise ValueError("runtime registration mismatch")
    return plan


def worker(job):
    return bound(base.worker,IncrementalHeatPool=NativeShadowHeatPool)(job)


def collect(resume=False):
    return bound(base.collect,OUT=OUT,verify=verify,worker=worker)(resume)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","collect","resume","stop"))
    phase=parser.parse_args().phase
    if phase=="stop":
        write_json(OUT/"STOP_AFTER_CASE.json",dict(requested=True)); result=dict(stop_after_case=True)
    elif phase=="resume": result=collect(True)
    elif phase=="verify": result=dict(verified=bool(verify()))
    else: result=globals()[phase]()
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
