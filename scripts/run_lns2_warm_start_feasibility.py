from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.closed_loop_trace_storage import (  # noqa: E402
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
)
from experiments._common import (  # noqa: E402
    contained_file as _contained_output_file,
    producer_identity,
    sha256_file as _sha256,
)
from experiments.repair_collection import (  # noqa: E402
    _load_dataset_rows,
    _make_environment,
    _plain,
    _read_json,
    _read_jsonl,
    _utc_now,
    _write_json,
    state_fingerprint,
)


SCHEMA = "lns2.warm_start_feasibility.v2"

WARM_START_PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_trace_storage.py",
    "experiments/repair_collection.py",
    "scripts/run_lns2_warm_start_feasibility.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/BasicLNS.h",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/PathTable.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/inc/SingleAgentSolver.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
    "third_party/mapf_lns2/src/PathTable.cpp",
)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_finite_json(item)
            for key, item in value.items()
        )
    return False


def _paths_sha256(paths: list[list[int]]) -> str:
    payload = json.dumps(paths, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _strict_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _state_paths(state: dict[str, Any], *, field: str) -> list[list[int]]:
    agents = state.get("agents")
    if not isinstance(agents, list):
        raise ValueError(f"{field}.agents must be a list")
    paths: list[list[int]] = []
    for agent_index, agent in enumerate(agents):
        if not isinstance(agent, dict) or not isinstance(agent.get("path"), list):
            raise ValueError(f"{field}.agents[{agent_index}].path must be a list")
        path = agent["path"]
        if any(type(location) is not int for location in path):
            raise ValueError(
                f"{field}.agents[{agent_index}].path must contain integers"
            )
        paths.append(list(path))
    return paths


def _atomic_write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}")
    with gzip.open(partial, "wt", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    os.replace(partial, path)


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint is not a JSON object: {path}")
    return value


def _initial_state(
    collection_root: Path, trace_path: Path, event: dict[str, Any]
) -> dict[str, Any]:
    if str(event.get("schema")) != EPISODE_SCHEMA_V2:
        state = event.get("state")
        if not isinstance(state, dict):
            raise ValueError("source trace is missing its initial state")
        return dict(state)
    state = read_state_blob(
        resolve_state_blob(trace_path, str(event["state_blob"]), collection_root)
    )
    extras = event.get("state_extras")
    if not isinstance(extras, dict):
        raise ValueError("source trace has invalid initial state extras")
    state.update(extras)
    return state


def _reconstruct_final_state(
    collection_root: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_path = collection_root / str(manifest["trace_file"])
    if not trace_path.is_file():
        raise ValueError(f"source trace is missing: {trace_path}")
    actual_sha = _sha256(trace_path)
    if actual_sha != str(manifest["trace_sha256"]):
        raise ValueError("source trace SHA256 mismatch")
    events = read_trace_events(trace_path)
    if len(events) != int(manifest["trace_event_count"]):
        raise ValueError("source trace event count mismatch")
    if not events or str(events[-1].get("event")) != "finish":
        raise ValueError("source trace does not end with a finish event")

    state = _initial_state(collection_root, trace_path, events[0])
    for event in events[1:-1]:
        if str(event.get("schema")) == EPISODE_SCHEMA_V2:
            before = state
            state = apply_state_delta(before, event.get("state_delta"))
            state.update(
                apply_extras_delta(before, event.get("state_extras_delta"))
            )
        else:
            state = dict(event["after"])
    if not _is_finite_json(state):
        raise ValueError("reconstructed source final state is not finite JSON")
    final_fingerprint = state_fingerprint(state)
    if final_fingerprint != str(events[-1]["final_fingerprint"]):
        raise ValueError("reconstructed source final fingerprint mismatch")
    summary = dict(manifest.get("summary") or {})
    if _strict_int(
        state.get("num_of_colliding_pairs"), field="source final conflicts"
    ) != _strict_int(summary.get("final_conflicts"), field="source summary conflicts"):
        raise ValueError("source final conflict count mismatch")
    if _strict_int(
        state.get("sum_of_costs"), field="source final SOC"
    ) != _strict_int(summary.get("final_sum_of_costs"), field="source summary SOC"):
        raise ValueError("source final SOC mismatch")
    paths = _state_paths(state, field="source final state")
    return state, {
        "collection_root": str(collection_root.resolve()),
        "episode_id": str(manifest["episode_id"]),
        "solver_seed": int(manifest["solver_seed"]),
        "trace_file": str(manifest["trace_file"]),
        "trace_sha256": actual_sha,
        "trace_event_count": len(events),
        "final_fingerprint": final_fingerprint,
        "final_conflicts": int(state["num_of_colliding_pairs"]),
        "final_sum_of_costs": int(state["sum_of_costs"]),
        "paths_sha256": _paths_sha256(paths),
    }


def _source(
    roots: list[Path], source_seed: int
) -> dict[str, Any]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        manifest_path = root / "official_adaptive_manifest.jsonl"
        if not manifest_path.is_file():
            continue
        for row in _read_jsonl(manifest_path):
            if (
                int(row.get("solver_seed", -1)) == source_seed
                and str(row.get("status")) in {"ok", "resumed"}
            ):
                matches.append((root, row))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one source episode for seed {source_seed}; "
            f"found {len(matches)}"
        )
    root, manifest = matches[0]
    run_config = _read_json(root / "run_config.json")
    dataset_root = Path(str(run_config["dataset"]))
    configuration = dict(run_config["configuration"])
    source_environment = dict(configuration["environment"])
    state, evidence = _reconstruct_final_state(root, manifest)
    rows = _load_dataset_rows(dataset_root, [str(manifest["split"])])
    task_rows = [row for row in rows if str(row["task_id"]) == str(manifest["task_id"])]
    if len(task_rows) != 1:
        raise ValueError("source task is not unique in the dataset manifest")
    return {
        "source_seed": source_seed,
        "dataset_root": str(dataset_root),
        "row": task_rows[0],
        "environment": source_environment,
        "state": state,
        "evidence": evidence,
    }


def _restart_sources(
    results_path: Path,
    base_sources: list[dict[str, Any]],
    *,
    best_per_source: bool,
) -> list[dict[str, Any]]:
    previous_root = results_path.parent
    restart_results_sha256 = _sha256(results_path)
    payload = _read_json(results_path)
    if str(payload.get("schema")) != SCHEMA:
        raise ValueError(
            "restart results use an unsupported schema; keep old warm-start "
            "artifacts read-only and start the v2 runner in a new output directory"
        )
    previous_run_fingerprint = str(payload.get("run_fingerprint", ""))
    previous_config_path = previous_root / "run_config.json"
    if not previous_run_fingerprint or not previous_config_path.is_file():
        raise ValueError("restart results lack their v2 run identity")
    previous_config = _read_json(previous_config_path)
    if (
        str(previous_config.get("schema")) != SCHEMA
        or str(previous_config.get("run_fingerprint", ""))
        != previous_run_fingerprint
        or _fingerprint(previous_config.get("run_identity"))
        != previous_run_fingerprint
    ):
        raise ValueError("restart results and run configuration identities differ")
    payload_jobs = payload.get("jobs")
    config_jobs = previous_config.get("jobs")
    if not isinstance(payload_jobs, list) or not isinstance(config_jobs, list):
        raise ValueError("restart results or run configuration has no job manifest")
    if any(not isinstance(row, dict) for row in (*payload_jobs, *config_jobs)):
        raise ValueError("restart job manifest contains a non-object row")
    result_by_id = {str(row.get("job_id", "")): row for row in payload_jobs}
    config_by_id = {str(row.get("job_id", "")): row for row in config_jobs}
    if (
        "" in result_by_id
        or "" in config_by_id
        or len(result_by_id) != len(payload_jobs)
        or len(config_by_id) != len(config_jobs)
        or set(result_by_id) != set(config_by_id)
    ):
        raise ValueError("restart result and configuration job coverage differs")
    for job_id, row in result_by_id.items():
        configured = config_by_id[job_id]
        job_identity = configured.get("job_identity")
        if (
            not isinstance(job_identity, dict)
            or _fingerprint(job_identity)
            != str(configured.get("job_fingerprint", ""))
            or job_identity.get("run_fingerprint") != previous_run_fingerprint
        ):
            raise ValueError("restart run configuration has an invalid job identity")
        for key in (
            "job_fingerprint",
            "portfolio_key",
            "source_seed",
            "continuation_seed",
        ):
            if row.get(key) != configured.get(key):
                raise ValueError(f"restart job {job_id} {key} mismatch")
        for key in ("source_seed", "continuation_seed"):
            _strict_int(row.get(key), field=f"restart job {job_id} {key}")
            _strict_int(
                configured.get(key), field=f"configured job {job_id} {key}"
            )
        if row.get("schema") != SCHEMA or row.get(
            "run_fingerprint"
        ) != previous_run_fingerprint:
            raise ValueError(f"restart job {job_id} run identity mismatch")
        if str(row.get("status")) == "complete":
            for key in ("final_conflicts", "final_sum_of_costs"):
                _strict_int(row.get(key), field=f"restart job {job_id} {key}")
            _strict_bool(
                row.get("success"), field=f"restart job {job_id} success"
            )
    rows = [
        dict(row)
        for row in payload_jobs
        if str(row.get("status")) == "complete"
    ]
    if not rows:
        raise ValueError("restart results contain no complete jobs")
    if best_per_source:
        selected: list[dict[str, Any]] = []
        for source_seed in sorted({int(row["source_seed"]) for row in rows}):
            matches = [row for row in rows if int(row["source_seed"]) == source_seed]
            selected.append(
                min(
                    matches,
                    key=lambda row: (
                        int(row["final_conflicts"]),
                        int(row["continuation_seed"]),
                    ),
                )
            )
        rows = selected
    base_by_seed = {int(source["source_seed"]): source for source in base_sources}
    restarted = []
    for row in rows:
        source_seed = int(row["source_seed"])
        if source_seed not in base_by_seed:
            raise ValueError(f"restart source seed {source_seed} has no base source")
        state_path = _contained_output_file(
            previous_root,
            row.get("final_state_file"),
            field="restart final_state_file",
        )
        if _sha256(state_path) != str(row["final_state_sha256"]):
            raise ValueError("restart checkpoint SHA256 mismatch")
        state = _read_gzip_json(state_path)
        if not _is_finite_json(state):
            raise ValueError("restart checkpoint is not finite JSON")
        paths = _state_paths(state, field="restart checkpoint")
        if _paths_sha256(paths) != str(row["final_paths_sha256"]):
            raise ValueError("restart checkpoint path hash mismatch")
        if _strict_int(
            state.get("num_of_colliding_pairs"), field="checkpoint conflicts"
        ) != _strict_int(row.get("final_conflicts"), field="result conflicts"):
            raise ValueError("restart checkpoint conflict count mismatch")
        if _strict_int(
            state.get("sum_of_costs"), field="checkpoint SOC"
        ) != _strict_int(row.get("final_sum_of_costs"), field="result SOC"):
            raise ValueError("restart checkpoint SOC mismatch")
        source = dict(base_by_seed[source_seed])
        source["state"] = state
        source["evidence"] = {
            "checkpoint_job_id": str(row["job_id"]),
            "checkpoint_run_fingerprint": previous_run_fingerprint,
            "checkpoint_job_fingerprint": str(row["job_fingerprint"]),
            "restart_results_file": str(results_path.resolve()),
            "restart_results_sha256": restart_results_sha256,
            "checkpoint_file": str(state_path.resolve()),
            "checkpoint_sha256": str(row["final_state_sha256"]),
            "paths_sha256": str(row["final_paths_sha256"]),
            "final_conflicts": int(row["final_conflicts"]),
            "final_sum_of_costs": int(row["final_sum_of_costs"]),
            "previous_observed_wall_seconds": float(row["observed_wall_seconds"]),
            "source_seed": source_seed,
            "warm_start_semantics": str(row["warm_start_semantics"]),
        }
        source["source_label"] = str(row["job_id"])
        restarted.append(source)
    return restarted


def _source_identity(source: dict[str, Any]) -> dict[str, Any]:
    state = dict(source["state"])
    if not _is_finite_json(state):
        raise ValueError("warm-start source state is not finite JSON")
    paths = _state_paths(state, field="warm-start source state")
    identity = {
        "source_seed": int(source["source_seed"]),
        "source_label": (
            str(source["source_label"])
            if source.get("source_label") is not None
            else None
        ),
        "dataset_root": str(Path(str(source["dataset_root"])).resolve()),
        "task": dict(source["row"]),
        "environment": dict(source["environment"]),
        "state_fingerprint": state_fingerprint(state),
        "paths_sha256": _paths_sha256(paths),
        "conflicts": _strict_int(
            state.get("num_of_colliding_pairs"), field="source conflicts"
        ),
        "sum_of_costs": _strict_int(
            state.get("sum_of_costs"), field="source SOC"
        ),
        "evidence": dict(source["evidence"]),
    }
    if not _is_finite_json(identity):
        raise ValueError("warm-start source identity is not finite JSON")
    return identity


def _task_identity(source: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "dataset_root": str(Path(str(source["dataset_root"])).resolve()),
        "task": dict(source["row"]),
    }
    if not _is_finite_json(identity):
        raise ValueError("warm-start task identity is not finite JSON")
    return identity


def _execution_identity() -> dict[str, Any]:
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(map(int, os.sched_getaffinity(0)))
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "worker_thread_limits": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }


def _verify_restored(
    source_state: dict[str, Any], restored: dict[str, Any]
) -> None:
    source_paths = _state_paths(source_state, field="source state")
    restored_paths = _state_paths(restored, field="restored state")
    if restored_paths != source_paths:
        raise ValueError("restored paths differ from the source final paths")
    for key in ("num_of_colliding_pairs", "sum_of_costs", "feasible"):
        if restored[key] != source_state[key]:
            raise ValueError(f"restored {key} differs from the source final state")
    if sorted(restored["conflict_edges"]) != sorted(source_state["conflict_edges"]):
        raise ValueError("restored conflict edges differ from the source final state")
    if int(restored["iteration"]) != 0:
        raise ValueError("restored repair iteration did not reset to zero")


def _terminal_stop_reason(state: dict[str, Any]) -> str | None:
    feasible = _strict_bool(state.get("feasible"), field="state feasible")
    done = _strict_bool(state.get("done"), field="state done")
    if feasible:
        return "feasible"
    if done:
        return "time_limit"
    return None


def _job_id(
    source_seed: int, continuation_seed: int, job_fingerprint: str
) -> str:
    return (
        f"source_{source_seed:04d}__continuation_{continuation_seed:06d}"
        f"__{job_fingerprint[:12]}"
    )


def _portfolio_marker_path(output_root: Path, portfolio_key: str) -> Path:
    return output_root / "portfolio" / portfolio_key / "solved.json"


def _portfolio_is_solved(path: Path, job: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    marker = _read_json(path)
    expected = {
        "schema": SCHEMA,
        "run_fingerprint": str(job["run_fingerprint"]),
        "portfolio_key": str(job["portfolio_key"]),
        "success": True,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(
                f"incompatible portfolio marker field {key}: {path}"
            )
    if not str(marker.get("winning_job_id", "")) or not str(
        marker.get("final_state_fingerprint", "")
    ):
        raise RuntimeError(f"portfolio marker is incomplete: {path}")
    return True


def _mark_portfolio_solved(
    path: Path,
    job: dict[str, Any],
    *,
    job_id: str,
    state: dict[str, Any],
) -> None:
    _write_json(
        path,
        {
            "schema": SCHEMA,
            "run_fingerprint": str(job["run_fingerprint"]),
            "portfolio_key": str(job["portfolio_key"]),
            "success": True,
            "winning_job_id": job_id,
            "final_state_fingerprint": state_fingerprint(state),
            "solved_at": _utc_now(),
        },
    )


def _validate_completed_result(
    result_path: Path, status_path: Path, job: dict[str, Any]
) -> dict[str, Any]:
    result = _read_json(result_path)
    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "status": "complete",
        "job_id": str(job["job_id"]),
        "run_fingerprint": str(job["run_fingerprint"]),
        "job_fingerprint": str(job["job_fingerprint"]),
        "portfolio_key": str(job["portfolio_key"]),
        "source_seed": int(job["source"]["source_seed"]),
        "continuation_seed": int(job["continuation_seed"]),
        "time_limit": float(job["time_limit"]),
        "source_evidence": dict(job["source"]["evidence"]),
        "source_identity_fingerprint": _fingerprint(job["source_identity"]),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            errors.append(f"{key} mismatch")
    if not _is_finite_json(result):
        errors.append("result is not finite JSON")
    for key in (
        "source_seed",
        "continuation_seed",
        "initial_conflicts",
        "final_conflicts",
        "initial_sum_of_costs",
        "final_sum_of_costs",
        "repair_iterations",
        "nonreducing_repairs",
    ):
        if type(result.get(key)) is not int:
            errors.append(f"{key} is not an integer")
    if type(result.get("success")) is not bool:
        errors.append("success is not a boolean")
    if type(result.get("time_limit")) not in {int, float} or isinstance(
        result.get("time_limit"), bool
    ):
        errors.append("time_limit is not numeric")
    if str(result.get("stop_reason")) not in {
        "feasible",
        "plateau",
        "portfolio_solved",
        "time_limit",
    }:
        errors.append("invalid stop_reason")

    try:
        final_state_path = _contained_output_file(
            Path(job["output_root"]),
            result.get("final_state_file"),
            field="final_state_file",
        )
        if _sha256(final_state_path) != str(result.get("final_state_sha256", "")):
            errors.append("final state SHA256 mismatch")
        final_state = _read_gzip_json(final_state_path)
        if not _is_finite_json(final_state):
            errors.append("final state is not finite JSON")
        final_paths = _state_paths(final_state, field="final state")
        if _paths_sha256(final_paths) != str(
            result.get("final_paths_sha256", "")
        ):
            errors.append("final paths SHA256 mismatch")
        if _strict_int(
            final_state.get("num_of_colliding_pairs"),
            field="final state conflicts",
        ) != _strict_int(result.get("final_conflicts"), field="result conflicts"):
            errors.append("final conflict count mismatch")
        if _strict_int(
            final_state.get("sum_of_costs"), field="final state SOC"
        ) != _strict_int(result.get("final_sum_of_costs"), field="result SOC"):
            errors.append("final SOC mismatch")
        if _strict_bool(
            final_state.get("feasible"), field="final state feasible"
        ) is not _strict_bool(result.get("success"), field="result success"):
            errors.append("final feasibility mismatch")
    except (KeyError, TypeError, ValueError, OSError) as error:
        errors.append(f"final state validation failed: {error}")

    if not status_path.is_file():
        errors.append("complete status file is missing")
    else:
        status = _read_json(status_path)
        for key in (
            "schema",
            "status",
            "job_id",
            "run_fingerprint",
            "job_fingerprint",
            "portfolio_key",
        ):
            if status.get(key) != expected.get(key):
                errors.append(f"status {key} mismatch")
    if errors:
        raise RuntimeError(
            f"completed warm-start artifact is invalid and was preserved: "
            f"{result_path}: " + "; ".join(errors)
        )
    return result


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    output_root = Path(job["output_root"])
    job_id = str(job["job_id"])
    job_root = output_root / "jobs" / job_id
    result_path = job_root / "result.json"
    status_path = job_root / "status.json"
    if result_path.is_file():
        return _validate_completed_result(result_path, status_path, job)

    source = dict(job["source"])
    source_state = dict(source["state"])
    paths = _state_paths(source_state, field="warm-start source state")
    environment_config = dict(source["environment"])
    environment_config["time_limit"] = float(job["time_limit"])
    environment_config["max_repair_iterations"] = 0
    started = time.perf_counter()
    status: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "job_id": job_id,
        "run_fingerprint": str(job["run_fingerprint"]),
        "job_fingerprint": str(job["job_fingerprint"]),
        "portfolio_key": str(job["portfolio_key"]),
        "source_seed": int(source["source_seed"]),
        "continuation_seed": int(job["continuation_seed"]),
        "initial_conflicts": int(source_state["num_of_colliding_pairs"]),
        "current_conflicts": int(source_state["num_of_colliding_pairs"]),
        "repair_iterations": 0,
        "errors": 0,
        "started_at": _utc_now(),
    }
    _write_json(status_path, status)
    try:
        environment = _make_environment(
            source["dataset_root"],
            source["row"],
            environment_config,
            "Adaptive",
        )
        restored = _plain(
            environment.reset_paths(paths, seed=int(job["continuation_seed"]))
        )
        _verify_restored(source_state, restored)
        restore_timings = _plain(environment.get_last_reset_timings())
        state = restored
        trajectory = [int(state["num_of_colliding_pairs"])]
        elapsed = [0.0]
        repair_iterations = 0
        nonreducing_repairs = 0
        selection_seconds = 0.0
        pp_seconds = 0.0
        repair_seconds = 0.0
        diagnostics: list[dict[str, Any]] = []
        stop_reason = _terminal_stop_reason(state) or "time_limit"
        last_drop_elapsed = 0.0
        diagnostic_interval = float(job.get("diagnostic_interval", 0.0))
        next_diagnostic = diagnostic_interval
        portfolio_stop_path = _portfolio_marker_path(
            output_root, str(job["portfolio_key"])
        )
        next_status = time.perf_counter() + 30.0
        if stop_reason == "feasible" and not _portfolio_is_solved(
            portfolio_stop_path, job
        ):
            _mark_portfolio_solved(
                portfolio_stop_path, job, job_id=job_id, state=state
            )
        while _terminal_stop_reason(state) is None:
            if _portfolio_is_solved(portfolio_stop_path, job):
                stop_reason = "portfolio_solved"
                break
            before_conflicts = int(state["num_of_colliding_pairs"])
            step_started = time.perf_counter()
            step = _plain(environment.step({"mode": "official"}))
            repair_seconds += time.perf_counter() - step_started
            state = dict(step["observation"])
            metrics = dict(step["metrics"])
            repair_iterations += int(bool(metrics.get("step_applied")))
            after_conflicts = int(state["num_of_colliding_pairs"])
            if after_conflicts >= before_conflicts:
                nonreducing_repairs += 1
            else:
                last_drop_elapsed = time.perf_counter() - started
            selection_seconds += float(
                metrics.get("native_neighborhood_generation_seconds", 0.0)
            )
            pp_seconds += float(metrics.get("pp_replan_seconds", 0.0))
            trajectory.append(after_conflicts)
            current_elapsed = time.perf_counter() - started
            elapsed.append(current_elapsed)
            terminal_reason = _terminal_stop_reason(state)
            if terminal_reason is not None:
                stop_reason = terminal_reason
            if terminal_reason == "feasible":
                if not _portfolio_is_solved(portfolio_stop_path, job):
                    _mark_portfolio_solved(
                        portfolio_stop_path,
                        job,
                        job_id=job_id,
                        state=state,
                    )
            if terminal_reason is not None:
                break
            if diagnostic_interval > 0 and current_elapsed >= next_diagnostic:
                window = float(job["plateau_window"])
                target = current_elapsed - window
                window_index = 0
                for index, value in enumerate(elapsed):
                    if float(value) <= target:
                        window_index = index
                    else:
                        break
                window_start_conflicts = int(trajectory[window_index])
                window_drop = window_start_conflicts - after_conflicts
                minimum_drop = max(
                    int(job["plateau_min_absolute_drop"]),
                    math.ceil(
                        window_start_conflicts
                        * float(job["plateau_min_relative_drop"])
                    ),
                )
                seconds_without_drop = current_elapsed - last_drop_elapsed
                plateau = (
                    current_elapsed >= window
                    and window_drop < minimum_drop
                    and seconds_without_drop
                    >= float(job["plateau_no_improvement_seconds"])
                )
                diagnostic = {
                    "elapsed_seconds": current_elapsed,
                    "current_conflicts": after_conflicts,
                    "window_seconds": window,
                    "window_start_conflicts": window_start_conflicts,
                    "window_drop": window_drop,
                    "minimum_required_drop": minimum_drop,
                    "seconds_without_drop": seconds_without_drop,
                    "plateau": plateau,
                    "diagnosed_at": _utc_now(),
                }
                diagnostics.append(diagnostic)
                status["latest_diagnostic"] = diagnostic
                _write_json(status_path, status)
                next_diagnostic += diagnostic_interval
                if plateau:
                    stop_reason = "plateau"
                    break
            if time.perf_counter() >= next_status:
                status.update(
                    {
                        "current_conflicts": after_conflicts,
                        "repair_iterations": repair_iterations,
                        "elapsed_seconds": time.perf_counter() - started,
                        "updated_at": _utc_now(),
                    }
                )
                _write_json(status_path, status)
                next_status = time.perf_counter() + 30.0

        final_paths = _state_paths(state, field="final state")
        final_state_path = job_root / "final_state.json.gz"
        _atomic_write_gzip_json(final_state_path, state)
        result = {
            "schema": SCHEMA,
            "status": "complete",
            "job_id": job_id,
            "run_fingerprint": str(job["run_fingerprint"]),
            "job_fingerprint": str(job["job_fingerprint"]),
            "portfolio_key": str(job["portfolio_key"]),
            "source_seed": int(source["source_seed"]),
            "continuation_seed": int(job["continuation_seed"]),
            "source_evidence": source["evidence"],
            "source_identity_fingerprint": _fingerprint(
                job["source_identity"]
            ),
            "warm_start_semantics": "paths-and-derived-repair-state; ALNS-and-RNG-reset",
            "time_limit": float(job["time_limit"]),
            "success": _strict_bool(state.get("feasible"), field="state feasible"),
            "stop_reason": stop_reason,
            "initial_conflicts": int(source_state["num_of_colliding_pairs"]),
            "final_conflicts": int(state["num_of_colliding_pairs"]),
            "initial_sum_of_costs": int(source_state["sum_of_costs"]),
            "final_sum_of_costs": int(state["sum_of_costs"]),
            "repair_iterations": repair_iterations,
            "nonreducing_repairs": nonreducing_repairs,
            "nonreducing_fraction": (
                nonreducing_repairs / repair_iterations if repair_iterations else 0.0
            ),
            "selection_seconds": selection_seconds,
            "pp_seconds": pp_seconds,
            "repair_wall_seconds": repair_seconds,
            "observed_wall_seconds": time.perf_counter() - started,
            "native_runtime": float(state["runtime"]),
            "restore_timings": restore_timings,
            "conflict_trajectory": trajectory,
            "elapsed_seconds": elapsed,
            "diagnostics": diagnostics,
            "final_paths_sha256": _paths_sha256(final_paths),
            "final_state_file": str(final_state_path.relative_to(output_root)),
            "final_state_sha256": _sha256(final_state_path),
            "completed_at": _utc_now(),
        }
        _write_json(result_path, result)
        _write_json(
            status_path,
            {
                **status,
                "status": "complete",
                "current_conflicts": result["final_conflicts"],
                "repair_iterations": repair_iterations,
                "elapsed_seconds": result["observed_wall_seconds"],
                "success": result["success"],
                "stop_reason": stop_reason,
                "latest_diagnostic": diagnostics[-1] if diagnostics else None,
                "completed_at": result["completed_at"],
            },
        )
        return result
    except BaseException as error:
        failure = {
            **status,
            "status": "error",
            "errors": 1,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "failed_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(status_path, failure)
        return failure


def _prepare_run_output(
    output_root: Path,
    run_config: dict[str, Any],
    source_validation: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    legacy_marker = output_root / "portfolio_solved.json"
    if legacy_marker.exists():
        raise RuntimeError(
            "legacy global portfolio marker found; preserve this output and use "
            "a new directory for the v2 runner"
        )

    config_path = output_root / "run_config.json"
    validation_path = output_root / "source_validation.json"
    if not config_path.exists():
        existing = [path.name for path in output_root.iterdir()]
        if existing:
            raise RuntimeError(
                "warm-start output has no compatible v2 run identity; preserve "
                "it and use a new directory: " + ", ".join(sorted(existing))
            )
        _write_json(config_path, run_config)
        _write_json(
            validation_path,
            {**source_validation, "validated_at": _utc_now()},
        )
        return

    existing_config = _read_json(config_path)
    if existing_config.get("schema") != run_config.get("schema"):
        raise RuntimeError(
            "warm-start resume schema mismatch; preserve the existing output "
            "and use a new directory"
        )
    if not isinstance(existing_config.get("run_identity"), dict) or _fingerprint(
        existing_config["run_identity"]
    ) != str(existing_config.get("run_fingerprint", "")):
        raise RuntimeError("existing warm-start run identity is internally invalid")
    if existing_config != run_config:
        raise RuntimeError(
            "warm-start resume configuration or job manifest mismatch; "
            "preserve the existing output and use a new directory"
        )
    if not validation_path.is_file():
        raise RuntimeError(
            "existing warm-start source validation is missing; preserve the "
            "output and use a new directory"
        )
    existing_validation = _read_json(validation_path)
    for key, value in source_validation.items():
        if existing_validation.get(key) != value:
            raise RuntimeError(
                f"warm-start source validation {key} mismatch; preserve the "
                "existing output and use a new directory"
            )


def _aggregate_status(
    output_root: Path,
    jobs: list[dict[str, Any]],
    run_fingerprint: str,
) -> dict[str, Any]:
    rows = []
    for job in jobs:
        path = output_root / "jobs" / str(job["job_id"]) / "status.json"
        if path.is_file():
            row = _read_json(path)
            expected = {
                "schema": SCHEMA,
                "job_id": str(job["job_id"]),
                "run_fingerprint": run_fingerprint,
                "job_fingerprint": str(job["job_fingerprint"]),
                "portfolio_key": str(job["portfolio_key"]),
            }
            for key, value in expected.items():
                if row.get(key) != value:
                    raise RuntimeError(
                        f"warm-start status {key} mismatch and was preserved: {path}"
                    )
            if not _is_finite_json(row):
                raise RuntimeError(
                    f"warm-start status is not finite JSON and was preserved: {path}"
                )
            if row.get("status") == "complete" and type(row.get("success")) is not bool:
                raise RuntimeError(
                    f"warm-start complete status success is not boolean: {path}"
                )
            rows.append(row)
    return {
        "schema": SCHEMA,
        "run_fingerprint": run_fingerprint,
        "status": (
            "error"
            if any(row.get("status") == "error" for row in rows)
            else "complete"
            if len(rows) == len(jobs)
            and all(row.get("status") == "complete" for row in rows)
            else "running"
        ),
        "total_jobs": len(jobs),
        "complete_jobs": sum(row.get("status") == "complete" for row in rows),
        "running_jobs": sum(row.get("status") == "running" for row in rows),
        "error_jobs": sum(row.get("status") == "error" for row in rows),
        "successful_jobs": sum(row.get("success") is True for row in rows),
        "jobs": [
            {
                key: row.get(key)
                for key in (
                    "job_id",
                    "job_fingerprint",
                    "portfolio_key",
                    "status",
                    "source_seed",
                    "continuation_seed",
                    "current_conflicts",
                    "repair_iterations",
                    "elapsed_seconds",
                    "success",
                    "stop_reason",
                    "latest_diagnostic",
                    "errors",
                )
            }
            for row in rows
        ],
        "updated_at": _utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continue LNS2 feasibility search from validated saved paths."
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--source-seeds", default="5,7")
    parser.add_argument("--restart-results")
    parser.add_argument("--best-per-source", action="store_true")
    parser.add_argument("--branches-per-source", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=730000)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--diagnostic-interval-seconds", type=float, default=1800.0)
    parser.add_argument("--plateau-window-seconds", type=float, default=1800.0)
    parser.add_argument("--plateau-min-absolute-drop", type=int, default=2)
    parser.add_argument("--plateau-min-relative-drop", type=float, default=0.01)
    parser.add_argument("--plateau-no-improvement-seconds", type=float, default=900.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    finite_values = (
        args.time_limit,
        args.diagnostic_interval_seconds,
        args.plateau_window_seconds,
        args.plateau_min_relative_drop,
        args.plateau_no_improvement_seconds,
    )
    if not all(math.isfinite(value) for value in finite_values):
        parser.error("time, diagnostic, and plateau values must be finite")
    if args.branches_per_source <= 0 or args.workers <= 0 or args.time_limit <= 0:
        parser.error("branch, worker, and time-limit values must be positive")
    if (
        args.diagnostic_interval_seconds < 0
        or args.plateau_window_seconds <= 0
        or args.plateau_min_absolute_drop < 0
        or args.plateau_min_relative_drop < 0
        or args.plateau_no_improvement_seconds < 0
    ):
        parser.error("diagnostic and plateau values must be non-negative")

    output_root = Path(args.output).resolve()
    roots = [Path(value).resolve() for value in args.source]
    source_seeds = [int(value) for value in args.source_seeds.split(",") if value]
    if not source_seeds:
        parser.error("at least one source seed is required")
    base_sources = [_source(roots, source_seed) for source_seed in source_seeds]
    sources = (
        _restart_sources(
            Path(args.restart_results).resolve(),
            base_sources,
            best_per_source=bool(args.best_per_source),
        )
        if args.restart_results
        else base_sources
    )
    source_identities = [_source_identity(source) for source in sources]
    native_producer_identity = producer_identity(
        project_root=PROJECT_ROOT,
        source_files=WARM_START_PRODUCER_FILES,
        native_required=True,
    )
    restart_evidence = (
        {
            "path": str(Path(args.restart_results).resolve()),
            "sha256": _sha256(Path(args.restart_results).resolve()),
        }
        if args.restart_results
        else None
    )
    effective_workers = min(args.workers, len(sources) * args.branches_per_source)
    run_identity = {
        "schema": SCHEMA,
        "producer_identity": native_producer_identity,
        "execution_identity": _execution_identity(),
        "source_roots": [str(root) for root in roots],
        "source_seeds": source_seeds,
        "source_identities": source_identities,
        "restart_results": restart_evidence,
        "best_per_source": bool(args.best_per_source),
        "branches_per_source": args.branches_per_source,
        "seed_base": args.seed_base,
        "time_limit": args.time_limit,
        "diagnostic_interval_seconds": args.diagnostic_interval_seconds,
        "plateau": {
            "window_seconds": args.plateau_window_seconds,
            "minimum_absolute_drop": args.plateau_min_absolute_drop,
            "minimum_relative_drop": args.plateau_min_relative_drop,
            "no_improvement_seconds": args.plateau_no_improvement_seconds,
        },
        "workers": effective_workers,
    }
    run_fingerprint = _fingerprint(run_identity)

    jobs: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        source_identity = source_identities[source_index]
        task_identity = _task_identity(source)
        portfolio_key = _fingerprint(task_identity)
        for branch in range(args.branches_per_source):
            continuation_seed = (
                args.seed_base + source_index * args.branches_per_source + branch
            )
            job_identity = {
                "schema": SCHEMA,
                "run_fingerprint": run_fingerprint,
                "source_identity": source_identity,
                "task_identity": task_identity,
                "portfolio_key": portfolio_key,
                "continuation_seed": continuation_seed,
                "time_limit": args.time_limit,
                "diagnostic_interval": args.diagnostic_interval_seconds,
                "plateau_window": args.plateau_window_seconds,
                "plateau_min_absolute_drop": args.plateau_min_absolute_drop,
                "plateau_min_relative_drop": args.plateau_min_relative_drop,
                "plateau_no_improvement_seconds": (
                    args.plateau_no_improvement_seconds
                ),
            }
            job_fingerprint = _fingerprint(job_identity)
            jobs.append(
                {
                    "job_id": _job_id(
                        source["source_seed"],
                        continuation_seed,
                        job_fingerprint,
                    ),
                    "run_fingerprint": run_fingerprint,
                    "job_fingerprint": job_fingerprint,
                    "job_identity": job_identity,
                    "source_identity": source_identity,
                    "task_identity": task_identity,
                    "portfolio_key": portfolio_key,
                    "source": source,
                    "continuation_seed": continuation_seed,
                    "time_limit": args.time_limit,
                    "diagnostic_interval": args.diagnostic_interval_seconds,
                    "plateau_window": args.plateau_window_seconds,
                    "plateau_min_absolute_drop": args.plateau_min_absolute_drop,
                    "plateau_min_relative_drop": args.plateau_min_relative_drop,
                    "plateau_no_improvement_seconds": args.plateau_no_improvement_seconds,
                    "output_root": str(output_root),
                }
            )
    run_config = {
        "schema": SCHEMA,
        "run_fingerprint": run_fingerprint,
        "run_identity": run_identity,
        "producer_identity": native_producer_identity,
        "jobs": [
            {
                "job_id": job["job_id"],
                "job_fingerprint": job["job_fingerprint"],
                "job_identity": job["job_identity"],
                "portfolio_key": job["portfolio_key"],
                "source_seed": job["source"]["source_seed"],
                "continuation_seed": job["continuation_seed"],
            }
            for job in jobs
        ],
    }
    source_validation = {
        "schema": SCHEMA,
        "status": "passed",
        "run_fingerprint": run_fingerprint,
        "semantics": "validated warm start, not exact ALNS/RNG continuation",
        "sources": source_identities,
    }
    _prepare_run_output(output_root, run_config, source_validation)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "sources": source_seeds,
                    "run_fingerprint": run_fingerprint,
                }
            )
        )
        return 0

    _write_json(
        output_root / "status.json",
        _aggregate_status(output_root, jobs, run_fingerprint),
    )
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(args.workers, len(jobs))
    ) as executor:
        futures = {executor.submit(_run_job, job): job for job in jobs}
        while futures:
            done, _ = concurrent.futures.wait(
                futures, timeout=30.0, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                results.append(future.result())
                del futures[future]
            _write_json(
                output_root / "status.json",
                _aggregate_status(output_root, jobs, run_fingerprint),
            )

    results.sort(key=lambda row: str(row["job_id"]))
    _write_json(
        output_root / "results.json",
        {
            "schema": SCHEMA,
            "run_fingerprint": run_fingerprint,
            "jobs": results,
        },
    )
    final_status = _aggregate_status(output_root, jobs, run_fingerprint)
    _write_json(output_root / "status.json", final_status)
    print(json.dumps(final_status, ensure_ascii=False, sort_keys=True))
    return 1 if final_status["error_jobs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
