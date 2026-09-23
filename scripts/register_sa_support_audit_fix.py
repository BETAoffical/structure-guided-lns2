"""Register an audit-only amendment without replacing the collected evidence."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run

TARGET = "scripts/probe_sa_original_terminal_support.py"
SELF = "scripts/register_sa_support_audit_fix.py"
TEST = "tests/evaluation/test_sa_original_terminal_support_audit.py"
ALLOWED = {"verify", "audit_worker", "analyze"}


def audit_only_difference(before, after):
    def partition(text):
        functions, other = {}, []
        for node in ast.parse(text).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name] = ast.dump(node, include_attributes=False)
            else:
                other.append(ast.dump(node, include_attributes=False))
        return functions, other
    old, old_top = partition(before)
    new, new_top = partition(after)
    run.require(old.keys() == new.keys() and old_top == new_top, "no top-level/collector additions")
    changed = {name for name in old if old[name] != new[name]}
    run.require(changed == ALLOWED, "only identity verification and offline audit may change")
    return sorted(changed)


def source_check(reg):
    blob = subprocess.check_output(["git", "show", reg["source_commit"] + ":" + TARGET], cwd=ROOT)
    expected = reg["inputs"][TARGET]
    run.require(expected in {hashlib.sha256(blob).hexdigest(), hashlib.sha256(blob.replace(b"\n", b"\r\n")).hexdigest()},
                "original Git source identity")
    return audit_only_difference(blob.decode("utf8"), (ROOT / TARGET).read_text(encoding="utf8"))


def register():
    from scripts import probe_sa_original_terminal_support as probe
    out = ROOT / probe.config()["output"]
    reg = run.check_seal(run.read_json(out / probe.REGISTRATION))
    run.require(not (out / "audit.json").exists() and not (out / "report.json").exists(), "do not supersede completed audit")
    changed = source_check(reg)
    files = {}
    for phase, count in (("control", 6), ("branches", 24)):
        p = out / (phase + ".complete.json")
        proof = run.check_seal(run.read_json(p))
        run.require(proof["binding"] == reg["binding"] and proof["jobs"] == len(proof["files"]) == count, "completed evidence")
        files[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
        for jid, sha in proof["files"].items():
            path = out / phase / jid / "result.json"
            run.require(run.sha256_file(path) == sha, "result changed")
            row = run.result_read(path.parent, reg)
            run.require(row["status"] == "ok", "no unknown solver results")
            files[path.relative_to(ROOT).as_posix()] = sha
    failure = out / "audit-progress/collection_progress.json"
    run.require(run.read_json(failure)["status"] == "error", "retain original audit failure")
    files[failure.relative_to(ROOT).as_posix()] = run.sha256_file(failure)
    files.update({p: run.sha256_file(ROOT / p) for p in (SELF, TEST)})
    proof = dict(binding=reg["binding"], before=reg["inputs"][TARGET], after=run.sha256_file(ROOT / TARGET),
        changed_functions=changed, files=files, solver_reruns=0, scientific_logic_changed=False,
        reason="Refresh the offline engine at every state; use a matching audit error adapter.")
    run.once(out / "audit_amendment.json", run.sealed(proof))
    return proof


def check_amendment(reg, out, actual):
    proof = run.check_seal(run.read_json(out / "audit_amendment.json"))
    run.require(proof["binding"] == reg["binding"] and proof["before"] == reg["inputs"][TARGET] and
                proof["after"] == actual and proof["changed_functions"] == source_check(reg) and
                proof["solver_reruns"] == 0 and not proof["scientific_logic_changed"], "audit amendment identity")
    for path, sha in proof["files"].items():
        run.require(run.sha256_file(run.contained_file(ROOT, path, field="audit amendment")) == sha, "amendment input changed")


if __name__ == "__main__":
    print(json.dumps(register(), indent=2), flush=True)
