"""Evaluation-only recovery; preserve audited labels and the once-trained model."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, json_fingerprint, contained_file
from experiments.sa_paired_completion import require
from scripts.run_sa_paired_closed_loop import once, sealed, check_seal
from scripts import run_sa_policy_aligned_update as active

CONFIG = "configs/sa_policy_aligned_update_evaluation_v1b.json"
SOURCE = "build/sa-policy-aligned-update-v1"
FIXED = "scripts/run_sa_policy_aligned_update.py"
FROZEN = SOURCE+"/frozen-source/run_sa_policy_aligned_update.py"
CODE = [CONFIG, FIXED, "scripts/recover_sa_policy_aligned_update_evaluation.py",
        "tests/evaluation/test_sa_policy_update_loader.py", "docs/SA_POLICY_UPDATE_LOADER_RECOVERY_ZH.md"]


def prepare():
    cfg = read_json(ROOT/CONFIG)
    old = read_json(ROOT/SOURCE/"plan.json")
    require(old["binding"] == json_fingerprint({k:v for k,v in old.items() if k != "binding"}), "old plan changed")
    require({k:v for k,v in cfg.items() if k != "output"} ==
            {k:v for k,v in old["config"].items() if k != "output"}, "scientific settings changed")
    out, src = ROOT/cfg["output"], ROOT/SOURCE
    require(not out.exists(), "recovery output already exists")
    require(not list((src/"evaluation").glob("*/result.json")), "old evaluation has results; scope audit required")
    require(sha256_file(ROOT/FROZEN) == old["inputs"][FIXED], "original source backup changed")
    inputs = {}
    for path, digest in old["inputs"].items():
        actual = FROZEN if path == FIXED else path
        require(sha256_file(contained_file(ROOT, actual, field="original input")) == digest, "old input changed: "+path)
        inputs[actual] = digest
    audit = check_seal(read_json(src/"labels.audit.json"))
    require(audit["binding"] == old["binding"] and audit["complete"] and audit["censored"] == 0, "labels not audited")
    for jid, digest in audit["results"].items():
        path = SOURCE+"/labels/"+jid+"/result.json"
        require(sha256_file(ROOT/path) == digest, "audited label changed")
        inputs[path] = digest
    preflight = check_seal(read_json(src/"preflight.complete.json"))
    require(preflight["binding"] == old["binding"] and preflight["jobs"] == 12, "original controls missing")
    for path, digest in preflight["files"].items():
        require(sha256_file(src/path) == digest, "original control changed")
        inputs[(src/path).relative_to(ROOT).as_posix()] = digest
    training = check_seal(read_json(src/"model/training.json"))
    receipt = check_seal(read_json(src/"model/receipt.json"))
    require(training["binding"] == receipt["binding"] == old["binding"] and receipt["parity_verified"], "old model identity")
    require(receipt["training_sha256"] == sha256_file(src/"model/training.json") and
            training["files"] == receipt["files"], "old model receipt")
    for path,digest in training["files"].items():
        require(sha256_file(src/"model"/path) == digest, "trained model changed")
    for path in [SOURCE+"/"+p for p in (
        "plan.json", "labels.audit.json", "labels.complete.json", "preflight.complete.json", "run_status.json",
        "model/training.json", "model/receipt.json", "model/bundle.json", "model/fixtures.json")]+CODE:
        inputs[path] = sha256_file(ROOT/path)
    p = {k:v for k,v in old.items() if k not in {"binding", "inputs", "config", "commit"}}
    p.update(config=cfg, inputs=inputs, recovery=dict(source=SOURCE, source_binding=old["binding"],
        fits=0, new_labels=0, permitted_change="updated model loader uses registered base feature names"),
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    p["binding"] = json_fingerprint(p)
    once(out/"plan.json", p)
    (out/"model").mkdir()
    for name, digest in training["files"].items():
        shutil.copyfile(src/"model"/name, out/"model"/name)
        require(sha256_file(out/"model"/name) == digest, "copy changed model")
    metadata = {k:v for k,v in training.items() if k not in {"binding", "integrity"}}
    metadata.update(binding=p["binding"], reused_from=SOURCE, fits_this_recovery=0)
    once(out/"model/training.json", sealed(metadata))
    once(out/"preflight.complete.json", sealed(dict(binding=p["binding"], jobs=12,
        reused_source=SOURCE+"/preflight.complete.json", fresh_solver_runs=0)))
    return dict(binding=p["binding"], fits=0, new_labels=0, episodes=24)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "verify-model", "evaluate", "analyze", "request-stop"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    active.CONFIG = CONFIG
    if args.phase == "prepare": result = prepare()
    elif args.phase == "verify":
        p,_ = active.verify(); result = dict(binding=p["binding"], recovery=p["recovery"])
    elif args.phase == "verify-model": result = active.verify_model()
    elif args.phase == "evaluate": result = active.collect("evaluation", args.resume, args.limit)
    elif args.phase == "analyze": result = active.analyze()
    else:
        _,out = active.verify(); once(out/"STOP_AFTER_BATCH",dict(requested=True)); result = dict(safe_stop=True)
    import json
    print(json.dumps(result))


if __name__ == "__main__":
    main()
