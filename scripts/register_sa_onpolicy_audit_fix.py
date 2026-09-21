"""Register a strictly audit-only amendment, retaining the original run binding."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file
from experiments.sa_paired_completion import require
from scripts.run_sa_paired_closed_loop import check_seal, once, sealed

RUNNER = "scripts/run_sa_onpolicy.py"
SELF = "scripts/register_sa_onpolicy_audit_fix.py"
TEST = "tests/evaluation/test_sa_onpolicy_audit_fix.py"
BEFORE = "runner-before-audit-fix.py"
RECEIPT = "audit-only-amendment.json"
OLD_VERIFY = '''    for name, digest in plan["inputs"].items():
        require(sha256_file(contained_file(ROOT, name, field="registered input")) == digest, "changed input: " + name)'''
NEW_VERIFY = '''    changes = {name: sha256_file(contained_file(ROOT, name, field="registered input"))
               for name, digest in plan["inputs"].items()
               if sha256_file(contained_file(ROOT, name, field="registered input")) != digest}
    if changes:
        from scripts.register_sa_onpolicy_audit_fix import validate_amendment
        validate_amendment(ROOT, plan, out, changes)'''
OLD_AUDIT = '''    require(q.state_fingerprint(state) == result["final_fingerprint"] and state == read_json(folder / "final.json"), "audit final paths")'''
NEW_AUDIT = '''    final = read_json(folder / "final.json")
    # Deltas deliberately omit wall time and external context, not solver data.
    require(q.state_fingerprint(state) == q.state_fingerprint(final) == result["final_fingerprint"] and
            {k: v for k, v in state.items() if k not in {"runtime", "context"}} ==
            {k: v for k, v in final.items() if k not in {"runtime", "context"}}, "audit final paths")'''


def prove_scope(before, after):
    require(before.count(OLD_VERIFY) == before.count(OLD_AUDIT) == 1, "unexpected original source")
    expected = before.replace(OLD_VERIFY, NEW_VERIFY).replace(OLD_AUDIT, NEW_AUDIT)
    require(expected == after, "change exceeds audit-only scope")


def validate_amendment(root, plan, out, changes):
    receipt = check_seal(read_json(out / RECEIPT))
    require(receipt["binding"] == plan["binding"] and receipt["old_sha256"] == plan["inputs"][RUNNER], "amendment binding")
    require(changes == {RUNNER: receipt["new_sha256"]}, "unregistered input change")
    require(sha256_file(out / BEFORE) == receipt["old_sha256"], "original source backup changed")
    require(receipt["supporting_files"] == {p: sha256_file(root / p) for p in (SELF, TEST)}, "amendment code changed")
    prove_scope((out / BEFORE).read_text(encoding="utf8"), (root / RUNNER).read_text(encoding="utf8"))


def register():
    from experiments.repair_collection import _CollectionRunLock
    cfg = read_json(ROOT / "configs/sa_onpolicy_execution.json")
    out = ROOT / cfg["output"]
    p = read_json(out / "plan.json")
    with _CollectionRunLock(out, p["binding"], "audit-only-amendment"):
        require((out / "STOP_AFTER_BATCH").exists(), "pause collection first")
        require(not (out / "models/actor-1.json").exists(), "unexpected completed update")
        require(sha256_file(out / BEFORE) == p["inputs"][RUNNER], "wrong backup")
        prove_scope((out / BEFORE).read_text(encoding="utf8"), (ROOT / RUNNER).read_text(encoding="utf8"))
        for name, digest in p["inputs"].items():
            if name != RUNNER:
                require(sha256_file(ROOT / name) == digest, "other input changed: " + name)
        once(out / RECEIPT, sealed(dict(binding=p["binding"], old_sha256=p["inputs"][RUNNER],
             new_sha256=sha256_file(ROOT / RUNNER), reason="Compare deterministic solver fields, excluding runtime/context omitted by delta schema",
             supporting_files={n: sha256_file(ROOT / n) for n in (SELF, TEST)},
             reused_results={f.parent.name: sha256_file(f) for f in sorted((out / "train-0").glob("*/result.json"))},
             changed_collection=False, changed_training=False, original_plan_preserved=True)))
    from scripts.run_sa_onpolicy import verify
    verify()
    return dict(registered=True, binding=p["binding"], no_trajectory_rewrite=True)


if __name__ == "__main__":
    print(register())
