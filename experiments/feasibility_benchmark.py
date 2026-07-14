from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkCase:
    benchmark_id: str
    map_path: Path
    scenario_path: Path
    agent_count: int
    seed: int

    def run_id(self, solver: str) -> str:
        return (
            f"{solver}__{self.benchmark_id}__agents_{self.agent_count:04d}"
            f"__seed_{self.seed:04d}"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(dataset: str | Path, seeds: Iterable[int]) -> list[BenchmarkCase]:
    root = Path(dataset).resolve()
    manifest = _read_jsonl(root / "manifest.jsonl")
    if not manifest:
        raise ValueError(f"MovingAI manifest is missing or empty: {root / 'manifest.jsonl'}")
    cases: list[BenchmarkCase] = []
    for row in manifest:
        map_path = (root / str(row["map_file"])).resolve()
        scenario_path = (root / str(row["scenario_file"])).resolve()
        if not map_path.is_file() or not scenario_path.is_file():
            raise ValueError(f"benchmark files are missing for {row['id']}")
        for agent_count in row["agent_counts"]:
            for seed in seeds:
                cases.append(
                    BenchmarkCase(
                        str(row["id"]),
                        map_path,
                        scenario_path,
                        int(agent_count),
                        int(seed),
                    )
                )
    return cases


def _last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else None


def _number(row: dict[str, str] | None, name: str) -> float | None:
    if row is None or row.get(name, "") == "":
        return None
    return float(row[name])


def _integer(row: dict[str, str] | None, name: str) -> int | None:
    value = _number(row, name)
    return int(value) if value is not None else None


def parse_lns2_result(prefix: Path, return_code: int | None) -> dict[str, Any]:
    init = _last_csv_row(Path(str(prefix) + "-initLNS.csv"))
    final_candidates = [
        Path(str(prefix) + "-LNS.csv"),
        Path(str(prefix) + "-PP.csv"),
    ]
    final = next(
        (row for row in (_last_csv_row(path) for path in final_candidates) if row),
        None,
    )
    source = init or final
    solution_cost = _integer(source, "solution cost")
    success = return_code == 0 and solution_cost is not None and solution_cost >= 0
    runtime = _number(source, "runtime")
    preprocessing = _number(source, "preprocessing runtime") or 0.0
    return {
        "success": success,
        "runtime": runtime,
        "preprocessing_runtime": preprocessing,
        "time_to_feasible": runtime + preprocessing if runtime is not None else None,
        "solution_cost": solution_cost,
        "initial_conflicts": _integer(init, "initial collisions"),
        "final_conflicts": _integer(init, "num of collisions"),
        "conflict_auc": _number(init, "area under curve"),
        "low_level_expanded": _integer(source, "LL expanded nodes"),
        "low_level_generated": _integer(source, "LL generated"),
        "low_level_calls": _integer(source, "LL runs"),
    }


def parse_gpbs_result(path: Path) -> dict[str, Any]:
    row = _last_csv_row(path)
    solution_cost = _integer(row, "solution cost")
    success = solution_cost is not None and solution_cost >= 0
    runtime = _number(row, "runtime")
    preprocessing = _number(row, "preprocessing runtime") or 0.0
    return {
        "success": success,
        "runtime": runtime,
        "preprocessing_runtime": preprocessing,
        "time_to_feasible": (
            runtime + preprocessing if success and runtime is not None else None
        ),
        "solution_cost": solution_cost,
        "initial_conflicts": None,
        "final_conflicts": 0 if success else None,
        "conflict_auc": None,
        "low_level_expanded": _integer(row, "#low-level expanded"),
        "low_level_generated": _integer(row, "#low-level generated"),
        "low_level_calls": _integer(row, "#low-level search calls"),
        "high_level_expanded": _integer(row, "#high-level expanded"),
        "high_level_generated": _integer(row, "#high-level generated"),
    }


def solver_command(
    solver: str,
    binary: Path,
    case: BenchmarkCase,
    time_limit: float,
    run_root: Path,
) -> tuple[list[str], Path]:
    if solver == "lns2_repair":
        prefix = run_root / "stats"
        command = [
            str(binary),
            "--map",
            str(case.map_path),
            "--agents",
            str(case.scenario_path),
            "--agentNum",
            str(case.agent_count),
            "--cutoffTime",
            str(time_limit),
            "--seed",
            str(case.seed),
            "--neighborSize",
            "8",
            "--initDestroyStrategy",
            "Adaptive",
            "--replanAlgo",
            "PP",
            "--sipp",
            "true",
            "--screen",
            "0",
            "--output",
            str(prefix),
        ]
        return command, prefix
    if solver == "gpbs":
        stats = run_root / "stats.csv"
        command = [
            str(binary),
            "--map",
            str(case.map_path),
            "--agents",
            str(case.scenario_path),
            "--agentNum",
            str(case.agent_count),
            "--cutoffTime",
            str(int(time_limit)),
            "--seed",
            str(case.seed),
            "--screen",
            "0",
            "--output",
            str(stats),
            "--solver",
            "GPBS",
            "--tr",
            "true",
            "--ic",
            "true",
            "--rr",
            "true",
            "--rth",
            "0",
            "--sipp",
            "true",
        ]
        return command, stats
    raise ValueError(f"unknown solver: {solver}")


def run_case(
    solver: str,
    binary: str | Path,
    case: BenchmarkCase,
    time_limit: float,
    output: str | Path,
) -> dict[str, Any]:
    output_root = Path(output).resolve()
    run_id = case.run_id(solver)
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    executable = Path(binary).resolve()
    command, result_path = solver_command(
        solver, executable, case, time_limit, run_root
    )
    started = time.perf_counter()
    timed_out = False
    error: str | None = None
    return_code: int | None = None
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(5.0, time_limit + 10.0),
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exception:
        timed_out = True
        stdout = str(exception.stdout or "")
        stderr = str(exception.stderr or "")
        error = "external process exceeded time limit plus 10-second grace period"
    except OSError as exception:
        error = str(exception)
    wall_runtime = time.perf_counter() - started
    (run_root / "stdout.txt").write_text(stdout, encoding="utf-8")
    (run_root / "stderr.txt").write_text(stderr, encoding="utf-8")

    metrics = (
        parse_lns2_result(result_path, return_code)
        if solver == "lns2_repair"
        else parse_gpbs_result(result_path)
    )
    if timed_out or error:
        metrics["success"] = False
    return {
        "schema_version": 1,
        "run_id": run_id,
        "solver": solver,
        "benchmark_id": case.benchmark_id,
        "map_file": str(case.map_path),
        "scenario_file": str(case.scenario_path),
        "agent_count": case.agent_count,
        "seed": case.seed,
        "time_limit": time_limit,
        "command": command,
        "return_code": return_code,
        "timed_out": timed_out,
        "error": error,
        "wall_runtime": wall_runtime,
        **metrics,
    }


def run_benchmark(
    dataset: str | Path,
    output: str | Path,
    solvers: dict[str, str | Path],
    seeds: Iterable[int],
    time_limit: float,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seed_values = [int(seed) for seed in seeds]
    if time_limit <= 0:
        raise ValueError("time limit must be positive")
    if "gpbs" in solvers and not float(time_limit).is_integer():
        raise ValueError("GPBS accepts integer-second time limits")
    resolved_binaries: dict[str, Path] = {}
    for solver, binary in solvers.items():
        resolved = Path(binary).resolve()
        if not resolved.is_file():
            raise ValueError(f"solver binary does not exist: {resolved}")
        resolved_binaries[solver] = resolved
    dataset_root = Path(dataset).resolve()
    dataset_manifest = dataset_root / "manifest.jsonl"
    if not dataset_manifest.is_file():
        raise ValueError(f"MovingAI manifest is missing: {dataset_manifest}")
    configuration = {
        "schema_version": 1,
        "dataset": str(dataset_root),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "seeds": seed_values,
        "time_limit": float(time_limit),
        "solvers": {
            solver: {
                "binary": str(binary),
                "binary_sha256": _sha256(binary),
            }
            for solver, binary in sorted(resolved_binaries.items())
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    configuration["configuration_fingerprint"] = fingerprint
    config_path = output_root / "run_config.json"
    if resume and config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config.get("configuration_fingerprint") != fingerprint:
            raise ValueError("resume configuration fingerprint mismatch")
    elif resume and (output_root / "manifest.jsonl").is_file():
        raise ValueError("cannot resume a benchmark without run_config.json")
    config_path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = output_root / "manifest.jsonl"
    existing = _read_jsonl(manifest_path) if resume else []
    completed_ids = {str(row["run_id"]) for row in existing}
    pending = [
        (solver, case)
        for case in load_cases(dataset_root, seed_values)
        for solver in solvers
        if case.run_id(solver) not in completed_ids
    ]
    if limit is not None:
        pending = pending[:limit]
    rows = list(existing)
    mode = "a" if resume and manifest_path.exists() else "w"
    with manifest_path.open(mode, encoding="utf-8", newline="\n") as stream:
        for solver, case in pending:
            row = run_case(
                solver, resolved_binaries[solver], case, time_limit, output_root
            )
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
    summary: dict[str, Any] = {"schema_version": 1, "run_count": len(rows)}
    for solver in solvers:
        solver_rows = [row for row in rows if row["solver"] == solver]
        summary[solver] = {
            "runs": len(solver_rows),
            "successes": sum(bool(row["success"]) for row in solver_rows),
            "timeouts": sum(bool(row["timed_out"]) for row in solver_rows),
            "errors": sum(bool(row["error"]) for row in solver_rows),
        }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows
