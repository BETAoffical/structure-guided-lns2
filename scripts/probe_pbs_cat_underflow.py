"""Run the five fixed expected-failure CAT diagnostic cases in isolated processes."""
import os
from pathlib import Path
import resource
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_pbs_repair as audit


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    executable = ROOT / "build/linux/pbs-admission-asan-v1/pbs_cat_underflow_probe"
    output = audit.OUT / "minimal"
    audit.require(not (output / "report.json").exists(), "minimal probe already exists")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("empty_local_external_hit", "local_only", "mixed_populated", "no_hit", "external_table_only"):
        log = output / (mode + ".log")
        with log.open("w") as stream:
            try:
                result = subprocess.run([str(executable), mode], stdout=stream, stderr=subprocess.STDOUT,
                    timeout=10, env={**os.environ, "ASAN_OPTIONS":"detect_leaks=0:disable_coredump=1"})
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = "timeout_unknown"
        rows.append(dict(mode=mode, returncode=code, log=log.name, sha256=audit.sha256_file(log)))
    report = dict(schema="lns2.pbs_cat_minimal_diagnostic.v1", rows=rows, no_solver=True,
                  executable_sha256=audit.sha256_file(executable),
                  source_sha256=audit.sha256_file(ROOT/"tests/diagnostics/pbs_cat_underflow_probe.cpp"),
                  constraint_table_sha256=audit.sha256_file(ROOT/"third_party/mapf_lns2/src/ConstraintTable.cpp"))
    audit.write(output / "report.json", report)
    print(rows)


if __name__ == "__main__":
    main()
