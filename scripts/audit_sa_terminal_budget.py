"""Supplement runtime identity and audit without changing a running collector."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_sa_terminal_budget as probe

run = probe.run
GUARD_FILES = ("scripts/audit_sa_terminal_budget.py", "tests/evaluation/test_sa_terminal_budget_identity.py")


def source_python(commit):
    entries = subprocess.check_output(["git", "ls-tree", "-r", "-z", commit], cwd=ROOT).split(b"\0")
    selected = []
    for entry in filter(None, entries):
        meta, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = meta.split()
        path = raw_path.decode("utf8")
        if kind == b"blob" and path.endswith(".py"):
            run.require(mode in {b"100644", b"100755"}, "no source symlinks")
            selected.append((path, oid))
    process = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT,
        input=b"\n".join(oid for _, oid in selected)+b"\n", capture_output=True, check=True)
    data, offset, result = process.stdout, 0, {}
    for path, oid in selected:
        end = data.index(b"\n", offset)
        actual, kind, size = data[offset:end].split()
        run.require(actual == oid and kind == b"blob", "git blob identity")
        offset = end+1
        content = data[offset:offset+int(size)]
        run.require(len(content) == int(size) and data[offset+int(size):offset+int(size)+1] == b"\n", "git blob length")
        offset += int(size)+1
        result[path] = content
    run.require(offset == len(data), "git batch remainder")
    return result


def same_source(actual, frozen):
    return actual.replace(b"\r\n", b"\n") == frozen.replace(b"\r\n", b"\n")


def pin():
    reg, out = probe.verify()
    files = {}
    for path, content in source_python(reg["source_commit"]).items():
        actual = run.contained_file(ROOT, path, field="runtime Python").read_bytes()
        run.require(same_source(actual, content), "Python changed since launch commit: " + path)
        files[path] = hashlib.sha256(actual).hexdigest()
    files.update({p: run.sha256_file(ROOT / p) for p in GUARD_FILES})
    proof = dict(binding=reg["binding"], source_commit=reg["source_commit"], files=files,
        supplemental_after_launch=True, collector_unchanged=True,
        note="Checks current sources against the pre-launch commit; not a new scientific registration.")
    run.once(out / "runtime_identity_supplement.json", run.sealed(proof))
    return dict(files=len(files), source_commit=reg["source_commit"], collector_unchanged=True)


def verify_files(files, root=ROOT):
    for path, sha in files.items():
        run.require(run.sha256_file(run.contained_file(root, path, field="runtime identity")) == sha,
                    "runtime dependency changed: " + path)


def verify():
    reg, out = probe.verify()
    proof = run.check_seal(run.read_json(out / "runtime_identity_supplement.json"))
    run.require(proof["binding"] == reg["binding"] and proof["source_commit"] == reg["source_commit"], "identity supplement")
    verify_files(proof["files"])
    return reg, out, proof


def save_attempt(out, binding, results):
    paths = sorted((out / "audit-attempts").glob("attempt-*.json"))
    index = max((int(p.stem.split("-")[-1]) for p in paths), default=-1)+1
    path = out / "audit-attempts" / f"attempt-{index:04d}.json"
    run.once(path, run.sealed(dict(binding=binding, results=sorted(results, key=lambda r: r["job_id"]))))
    return path


def analyze():
    from experiments.repair_collection import _run_jobs
    reg, out, _ = verify()
    jobs = probe.jobs_for(reg)
    complete = run.check_seal(run.read_json(out / "extensions.complete.json"))
    run.require(complete["binding"] == reg["binding"] and complete["jobs"] == len(jobs) and complete["files"] ==
        {j["job_id"]: run.sha256_file(probe.folder(j) / "result.json") for j in jobs}, "complete collection")
    with probe.recovery.strict_lock(out, reg["binding"], "guarded-audit"):
        if (out / "audit.json").exists():
            proof = run.check_seal(run.read_json(out / "audit.json"))
            rows = proof["results"]
            run.require(proof["binding"] == reg["binding"] and len(rows) == len(jobs) and
                all(r["status"] == "ok" for r in rows) and
                {r["job_id"]: r["result_sha256"] for r in rows} == complete["files"], "existing audit identity")
        else:
            attempt_count = len(list((out / "audit-attempts").glob("attempt-*.json")))
            rows = _run_jobs(probe.audit_worker, jobs, 4, phase="extension-audit",
                output_root=out / "guarded-audit-progress" / f"attempt-{attempt_count:04d}",
                run_fingerprint=reg["binding"], timeout_seconds=960.,
                failure_result=lambda j, status, error: dict(job_id=j["job_id"], status=status, error=error))
            rows = sorted(rows, key=lambda r: r["job_id"])
            save_attempt(out, reg["binding"], rows)
            if len(rows) != 4 or any(r["status"] != "ok" for r in rows):
                run.write_json(out / "run_status.json", dict(status="audit_failed", binding=reg["binding"]))
                raise ValueError("complete extension audit required; attempt retained")
            verify()
            run.once(out / "audit.json", run.sealed(dict(binding=reg["binding"], results=rows)))
        results = [probe.read_result(j) for j in jobs]
        report = dict(probe.summary(results), binding=reg["binding"], rows=results, audit=rows,
                      source_rows=reg["records"], audit_sha256=run.sha256_file(out / "audit.json"),
                      runtime_identity_sha256=run.sha256_file(out / "runtime_identity_supplement.json"))
        run.once(out / "report.json", run.sealed(json.loads(json.dumps(report))))
        run.write_json(out / "run_status.json", dict(status="complete", report_sha256=run.sha256_file(out / "report.json")))
    return probe.summary(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pin", "verify", "analyze"))
    args = parser.parse_args()
    if args.phase == "pin":
        result = pin()
    elif args.phase == "analyze":
        result = analyze()
    else:
        reg, _, proof = verify()
        result = dict(binding=reg["binding"], files=len(proof["files"]), verified=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
