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


SCHEMA = "lns2.warm_start_feasibility.v3"

WARM_START_PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_trace_storage.py",
    "experiments/repair_collection.py",
    "scripts/run_lns2_warm_start_feasibility.py",
    "include/structure_guided/instance_validation.hpp",
    "include/structure_guided/native_semantics.hpp",
    "src/instance_validation.cpp",
    "src/online_features.cpp",
    "src/online_features.h",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/ConstraintTable.h",
    "third_party/mapf_lns2/inc/BasicLNS.h",
    "third_party/mapf_lns2/inc/CBS/CBSNode.h",
    "third_party/mapf_lns2/inc/CBS/ECBSNode.h",
    "third_party/mapf_lns2/inc/CBS/GCBSNode.h",
    "third_party/mapf_lns2/inc/CBS/PBS.h",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/PathTable.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/inc/ReservationTable.h",
    "third_party/mapf_lns2/inc/SIPP.h",
    "third_party/mapf_lns2/inc/SingleAgentSolver.h",
    "third_party/mapf_lns2/inc/SpaceTimeAStar.h",
    "third_party/mapf_lns2/inc/WeightedSampling.h",
    "third_party/mapf_lns2/src/BasicLNS.cpp",
    "third_party/mapf_lns2/src/CBS/Conflict.cpp",
    "third_party/mapf_lns2/src/CBS/MDD.cpp",
    "third_party/mapf_lns2/src/ConstraintTable.cpp",
    "third_party/mapf_lns2/src/InitLNS.cpp",
    "third_party/mapf_lns2/src/PathTable.cpp",
    "third_party/mapf_lns2/src/ReservationTable.cpp",
    "third_party/mapf_lns2/src/SIPP.cpp",
    "third_party/mapf_lns2/src/SingleAgentSolver.cpp",
    "third_party/mapf_lns2/src/SpaceTimeAStar.cpp",
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


def _strict_nonnegative_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _state_paths(state: dict[str, Any], *, field: str) -> list[list[int]]:
    agents = state.get("agents")
    if not isinstance(agents, list):
        raise ValueError(f"{field}.agents must be a list")
    paths: list[list[int]] = []
    for agent_index, agent in enumerate(agents):
        if not isinstance(agent, dict) or not isinstance(agent.get("path"), list):
            raise ValueError(f"{field}.agents[{agent_index}].path must be a list")
        path = agent["path"]
        if not path or any(type(location) is not int for location in path):
            raise ValueError(
                f"{field}.agents[{agent_index}].path must contain at least one integer"
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
    base_by_seed = {int(source["source_seed"]): source for source in base_sources}
    if len(base_by_seed) != len(base_sources):
        raise ValueError("restart base sources contain duplicate source seeds")
    validated_complete_rows: list[dict[str, Any]] = []
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
            source_seed = int(row["source_seed"])
            if source_seed not in base_by_seed:
                raise ValueError(
                    f"restart source seed {source_seed} has no base source"
                )
            template_job = {
                "run_fingerprint": previous_run_fingerprint,
                "portfolio_key": configured["portfolio_key"],
                "source": dict(base_by_seed[source_seed]),
                "output_root": str(previous_root),
            }
            try:
                previous_job = _configured_job(
                    previous_root, job_id, template_job
                )
                canonical_result_path = (
                    previous_root / "jobs" / job_id / "result.json"
                )
                canonical_status_path = (
                    previous_root / "jobs" / job_id / "status.json"
                )
                if not canonical_result_path.is_file():
                    raise RuntimeError("canonical result file is missing")
                canonical_result = _read_json(canonical_result_path)
                if canonical_result != row:
                    raise RuntimeError(
                        "aggregate results row differs from canonical result"
                    )
                validated = _validate_completed_result(
                    canonical_result_path,
                    canonical_status_path,
                    previous_job,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"restart job {job_id} completed artifact is invalid"
                ) from error
            validated_complete_rows.append(validated)
    rows = validated_complete_rows
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


def _parse_source_seeds(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("at least one source seed is required")
    try:
        seeds = [int(part) for part in parts]
    except ValueError as error:
        raise ValueError("source seeds must be integers") from error
    if any(seed < 0 for seed in seeds):
        raise ValueError("source seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("source seeds must be unique")
    return seeds


def _source_matching_identity(
    template: dict[str, Any], expected_identity: dict[str, Any]
) -> dict[str, Any]:
    candidate = dict(template)
    try:
        if _source_identity(candidate) == expected_identity:
            return candidate
    except (KeyError, TypeError, ValueError):
        pass
    evidence = expected_identity.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("configured source identity cannot be reconstructed")
    checkpoint_reference = evidence.get("checkpoint_file")
    checkpoint_sha256 = evidence.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_reference, str)
        or not checkpoint_reference
        or not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
    ):
        raise ValueError("configured source identity lacks checkpoint evidence")
    checkpoint_path = Path(checkpoint_reference).resolve()
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("configured source checkpoint evidence changed")
    candidate["state"] = _read_gzip_json(checkpoint_path)
    candidate["evidence"] = dict(evidence)
    candidate["source_label"] = expected_identity.get("source_label")
    if _source_identity(candidate) != expected_identity:
        raise ValueError("configured source checkpoint identity mismatch")
    return candidate


def _configured_job(
    output_root: Path,
    job_id: str,
    template_job: dict[str, Any],
) -> dict[str, Any]:
    config_path = output_root / "run_config.json"
    if not config_path.is_file():
        raise RuntimeError("warm-start run_config is missing")
    run_config = _read_json(config_path)
    if not isinstance(run_config, dict):
        raise RuntimeError("warm-start run_config is not an object")
    run_fingerprint = str(template_job["run_fingerprint"])
    if (
        run_config.get("schema") != SCHEMA
        or run_config.get("run_fingerprint") != run_fingerprint
        or not isinstance(run_config.get("run_identity"), dict)
        or _fingerprint(run_config["run_identity"]) != run_fingerprint
    ):
        raise RuntimeError("warm-start run_config identity mismatch")
    configured_jobs = run_config.get("jobs")
    if not isinstance(configured_jobs, list) or any(
        not isinstance(row, dict) for row in configured_jobs
    ):
        raise RuntimeError("warm-start run_config job manifest is invalid")
    matches = [
        dict(row)
        for row in configured_jobs
        if row.get("job_id") == job_id
    ]
    if len(matches) != 1:
        raise RuntimeError("portfolio winner is not unique in run_config")
    configured = matches[0]
    identity = configured.get("job_identity")
    if (
        not isinstance(identity, dict)
        or _fingerprint(identity) != configured.get("job_fingerprint")
        or identity.get("run_fingerprint") != run_fingerprint
        or configured.get("portfolio_key") != template_job.get("portfolio_key")
        or identity.get("portfolio_key") != template_job.get("portfolio_key")
    ):
        raise RuntimeError("portfolio winner job identity is invalid")
    source_identity = identity.get("source_identity")
    task_identity = identity.get("task_identity")
    if not isinstance(source_identity, dict) or not isinstance(task_identity, dict):
        raise RuntimeError("portfolio winner lacks source/task identity")
    try:
        source = _source_matching_identity(
            dict(template_job["source"]), source_identity
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError("portfolio winner source identity is invalid") from error
    if _task_identity(source) != task_identity:
        raise RuntimeError("portfolio winner task identity mismatch")
    source_seed = source.get("source_seed")
    continuation_seed = configured.get("continuation_seed")
    if (
        type(source_seed) is not int
        or configured.get("source_seed") != source_seed
        or type(continuation_seed) is not int
        or identity.get("continuation_seed") != continuation_seed
        or _job_id(
            source_seed,
            continuation_seed,
            str(configured["job_fingerprint"]),
        )
        != job_id
    ):
        raise RuntimeError("portfolio winner seed/job id mismatch")
    time_limit = identity.get("time_limit")
    if (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not math.isfinite(float(time_limit))
        or float(time_limit) <= 0.0
    ):
        raise RuntimeError("portfolio winner time limit is invalid")
    return {
        **template_job,
        "job_id": job_id,
        "job_fingerprint": str(configured["job_fingerprint"]),
        "job_identity": identity,
        "source_identity": source_identity,
        "task_identity": task_identity,
        "source": source,
        "continuation_seed": continuation_seed,
        "time_limit": float(time_limit),
        "output_root": str(output_root),
    }


def _portfolio_marker_path(output_root: Path, portfolio_key: str) -> Path:
    return output_root / "portfolio" / portfolio_key / "solved.json"


def _portfolio_is_solved(path: Path, job: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    marker = _read_json(path)
    if not isinstance(marker, dict) or not _is_finite_json(marker):
        raise RuntimeError(f"portfolio marker is not a finite JSON object: {path}")
    expected = {
        "schema": SCHEMA,
        "run_fingerprint": str(job["run_fingerprint"]),
        "portfolio_key": str(job["portfolio_key"]),
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(
                f"incompatible portfolio marker field {key}: {path}"
            )
    if marker.get("success") is not True:
        raise RuntimeError(f"portfolio marker success is not true: {path}")
    winning_job_id = marker.get("winning_job_id")
    if not isinstance(winning_job_id, str) or not winning_job_id:
        raise RuntimeError(f"portfolio marker winning_job_id is invalid: {path}")

    output_root = Path(job["output_root"]).resolve()
    expected_marker_path = _portfolio_marker_path(
        output_root, str(job["portfolio_key"])
    ).resolve()
    if path.resolve() != expected_marker_path:
        raise RuntimeError(f"portfolio marker path is not task scoped: {path}")
    winner_job = _configured_job(output_root, winning_job_id, job)
    if marker.get("winning_job_fingerprint") != winner_job["job_fingerprint"]:
        raise RuntimeError(f"portfolio marker winner fingerprint mismatch: {path}")
    winner_root = (output_root / "jobs" / winning_job_id).resolve()
    canonical_result_path = winner_root / "result.json"
    canonical_status_path = winner_root / "status.json"
    canonical_final_state_path = winner_root / "final_state.json.gz"
    try:
        result_path = _contained_output_file(
            output_root,
            marker.get("winning_result_file"),
            field="winning_result_file",
        )
        status_path = _contained_output_file(
            output_root,
            marker.get("winning_status_file"),
            field="winning_status_file",
        )
        final_state_path = _contained_output_file(
            output_root,
            marker.get("final_state_file"),
            field="final_state_file",
        )
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"portfolio marker file reference is invalid: {path}") from error
    if (
        result_path.resolve() != canonical_result_path
        or status_path.resolve() != canonical_status_path
        or final_state_path.resolve() != canonical_final_state_path
    ):
        raise RuntimeError(f"portfolio marker does not use canonical winner files: {path}")
    for artifact_path, field in (
        (result_path, "winning_result_sha256"),
        (status_path, "winning_status_sha256"),
        (final_state_path, "final_state_sha256"),
    ):
        expected_sha = marker.get(field)
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or _sha256(artifact_path) != expected_sha
        ):
            raise RuntimeError(
                f"portfolio marker {field} does not match its artifact: {path}"
            )
    try:
        result = _validate_completed_result(
            result_path, status_path, winner_job
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"portfolio winner failed completed-result validation: {path}"
        ) from error
    if (
        result.get("success") is not True
        or result.get("stop_reason") != "feasible"
        or result.get("final_conflicts") != 0
    ):
        raise RuntimeError(f"portfolio winner is not a feasible completion: {path}")
    if (
        result.get("final_state_file") != marker.get("final_state_file")
        or result.get("final_state_sha256") != marker.get("final_state_sha256")
    ):
        raise RuntimeError(f"portfolio winner final state binding mismatch: {path}")
    final_state = _read_gzip_json(final_state_path)
    if _strict_bool(
        final_state.get("feasible"), field="portfolio final state feasible"
    ) is not True:
        raise RuntimeError(f"portfolio winner final state is not feasible: {path}")
    fingerprint = marker.get("final_state_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or state_fingerprint(final_state) != fingerprint
    ):
        raise RuntimeError(f"portfolio marker final state fingerprint mismatch: {path}")
    return True


def _mark_portfolio_solved(
    path: Path,
    job: dict[str, Any],
    *,
    job_id: str,
    result_path: Path,
    status_path: Path,
    result: dict[str, Any],
    state: dict[str, Any],
) -> None:
    output_root = Path(job["output_root"]).resolve()
    if result.get("success") is not True or result.get("status") != "complete":
        raise RuntimeError("portfolio marker requires a complete successful result")
    if not result_path.is_file() or not status_path.is_file():
        raise RuntimeError("portfolio marker requires durable result and status files")
    validated = _validate_completed_result(result_path, status_path, job)
    if (
        validated.get("success") is not True
        or validated.get("stop_reason") != "feasible"
        or validated.get("final_conflicts") != 0
    ):
        raise RuntimeError("portfolio marker requires a feasible completed result")
    final_state_path = _contained_output_file(
        output_root,
        result.get("final_state_file"),
        field="final_state_file",
    )
    if _sha256(final_state_path) != result.get("final_state_sha256"):
        raise RuntimeError("portfolio marker final state SHA256 mismatch")
    if state_fingerprint(_read_gzip_json(final_state_path)) != state_fingerprint(
        state
    ):
        raise RuntimeError("portfolio marker in-memory final state mismatch")
    relative_result = result_path.resolve().relative_to(output_root).as_posix()
    relative_status = status_path.resolve().relative_to(output_root).as_posix()
    _write_json(
        path,
        {
            "schema": SCHEMA,
            "run_fingerprint": str(job["run_fingerprint"]),
            "portfolio_key": str(job["portfolio_key"]),
            "success": True,
            "winning_job_id": job_id,
            "winning_job_fingerprint": str(result["job_fingerprint"]),
            "winning_result_file": relative_result,
            "winning_result_sha256": _sha256(result_path),
            "winning_status_file": relative_status,
            "winning_status_sha256": _sha256(status_path),
            "final_state_file": str(result["final_state_file"]),
            "final_state_sha256": str(result["final_state_sha256"]),
            "final_state_fingerprint": state_fingerprint(state),
            "solved_at": _utc_now(),
        },
    )


def _validate_completed_result(
    result_path: Path, status_path: Path, job: dict[str, Any]
) -> dict[str, Any]:
    result = _read_json(result_path)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"completed warm-start artifact is invalid and was preserved: "
            f"{result_path}: result is not a JSON object"
        )
    errors: list[str] = []
    output_root = Path(job["output_root"]).resolve()
    job_root = (output_root / "jobs" / str(job["job_id"])).resolve()
    if result_path.resolve() != job_root / "result.json":
        errors.append("result path is not canonical for its job")
    if status_path.resolve() != job_root / "status.json":
        errors.append("status path is not canonical for its job")
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
        value = result.get(key)
        if type(value) is not int or value < 0:
            errors.append(f"{key} is not a non-negative integer")
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
    if (
        result.get("warm_start_semantics")
        != "paths-and-derived-repair-state; ALNS-and-RNG-reset"
    ):
        errors.append("warm-start semantics marker mismatch")
    if not isinstance(result.get("completed_at"), str) or not result.get(
        "completed_at"
    ):
        errors.append("completed_at is not a non-empty string")
    if not isinstance(result.get("diagnostics"), list):
        errors.append("diagnostics is not a list")

    raw_source_state = dict(job.get("source") or {}).get("state")
    if not isinstance(raw_source_state, dict):
        source_state: dict[str, Any] = {}
        errors.append("job source state is missing")
    else:
        source_state = dict(raw_source_state)
        if result.get("initial_conflicts") != source_state.get(
            "num_of_colliding_pairs"
        ):
            errors.append("initial conflict count differs from source state")
        if result.get("initial_sum_of_costs") != source_state.get("sum_of_costs"):
            errors.append("initial SOC differs from source state")

    for field in (
        "selection_seconds",
        "pp_seconds",
        "repair_wall_seconds",
        "observed_wall_seconds",
        "native_runtime",
        "nonreducing_fraction",
    ):
        try:
            _strict_nonnegative_float(result.get(field), field=field)
        except ValueError as error:
            errors.append(str(error))
    try:
        nonreducing_fraction = _strict_nonnegative_float(
            result.get("nonreducing_fraction"), field="nonreducing_fraction"
        )
        repair_iterations = _strict_int(
            result.get("repair_iterations"), field="repair_iterations"
        )
        nonreducing_repairs = _strict_int(
            result.get("nonreducing_repairs"), field="nonreducing_repairs"
        )
        if nonreducing_repairs > repair_iterations:
            errors.append("nonreducing repairs exceed repair iterations")
        expected_fraction = (
            nonreducing_repairs / repair_iterations
            if repair_iterations
            else 0.0
        )
        if not math.isclose(
            nonreducing_fraction, expected_fraction, rel_tol=1e-12, abs_tol=1e-12
        ):
            errors.append("nonreducing fraction mismatch")
    except ValueError as error:
        errors.append(str(error))

    try:
        trajectory = result.get("conflict_trajectory")
        elapsed = result.get("elapsed_seconds")
        step_applied = result.get("step_applied_trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError("conflict_trajectory must be a non-empty list")
        if any(type(value) is not int or value < 0 for value in trajectory):
            raise ValueError(
                "conflict_trajectory must contain non-negative integers"
            )
        if not isinstance(elapsed, list) or len(elapsed) != len(trajectory):
            raise ValueError(
                "elapsed_seconds length must match conflict_trajectory"
            )
        elapsed_values = [
            _strict_nonnegative_float(value, field="elapsed_seconds")
            for value in elapsed
        ]
        if elapsed_values[0] != 0.0 or any(
            right < left
            for left, right in zip(elapsed_values, elapsed_values[1:])
        ):
            raise ValueError("elapsed_seconds must start at zero and be ordered")
        if (
            not isinstance(step_applied, list)
            or len(step_applied) + 1 != len(trajectory)
            or any(type(value) is not bool for value in step_applied)
        ):
            raise ValueError(
                "step_applied_trajectory must bind every trajectory transition"
            )
        if trajectory[0] != result.get("initial_conflicts"):
            errors.append("conflict trajectory initial endpoint mismatch")
        if trajectory[-1] != result.get("final_conflicts"):
            errors.append("conflict trajectory final endpoint mismatch")
        applied_count = sum(step_applied)
        if applied_count != result.get("repair_iterations"):
            errors.append("repair iteration count differs from applied transitions")
        derived_nonreducing = sum(
            applied and after >= before
            for before, after, applied in zip(
                trajectory, trajectory[1:], step_applied
            )
        )
        if derived_nonreducing != result.get("nonreducing_repairs"):
            errors.append(
                "nonreducing repair count differs from conflict trajectory"
            )
        observed = _strict_nonnegative_float(
            result.get("observed_wall_seconds"), field="observed_wall_seconds"
        )
        if elapsed_values[-1] > observed + 1e-9:
            errors.append("trajectory elapsed time exceeds observed wall time")
    except ValueError as error:
        errors.append(str(error))

    try:
        selection = _strict_nonnegative_float(
            result.get("selection_seconds"), field="selection_seconds"
        )
        pp = _strict_nonnegative_float(
            result.get("pp_seconds"), field="pp_seconds"
        )
        repair = _strict_nonnegative_float(
            result.get("repair_wall_seconds"), field="repair_wall_seconds"
        )
        observed = _strict_nonnegative_float(
            result.get("observed_wall_seconds"), field="observed_wall_seconds"
        )
        if selection + pp > repair + 1e-6:
            errors.append("native repair timing components exceed repair wall time")
        if repair > observed + 1e-6:
            errors.append("repair wall time exceeds observed wall time")
        restore_timings = result.get("restore_timings")
        if not isinstance(restore_timings, dict):
            errors.append("restore_timings is not an object")
        else:
            for name, value in restore_timings.items():
                _strict_nonnegative_float(value, field=f"restore_timings.{name}")
    except ValueError as error:
        errors.append(str(error))

    try:
        final_state_path = _contained_output_file(
            output_root,
            result.get("final_state_file"),
            field="final_state_file",
        )
        if final_state_path.resolve() != job_root / "final_state.json.gz":
            errors.append("final state path is not canonical for its job")
        if _sha256(final_state_path) != str(result.get("final_state_sha256", "")):
            errors.append("final state SHA256 mismatch")
        final_state = _read_gzip_json(final_state_path)
        if not _is_finite_json(final_state):
            errors.append("final state is not finite JSON")
        final_paths = _state_paths(final_state, field="final state")
        if source_state:
            source_paths = _state_paths(source_state, field="source state")
            if len(final_paths) != len(source_paths):
                errors.append("final path count differs from source state")
            elif any(
                final_path[0] != source_path[0]
                or final_path[-1] != source_path[-1]
                for source_path, final_path in zip(source_paths, final_paths)
            ):
                errors.append("final path endpoints differ from source state")
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
        final_done = _strict_bool(
            final_state.get("done"), field="final state done"
        )
        final_iteration = _strict_int(
            final_state.get("iteration"), field="final state iteration"
        )
        if final_iteration != result.get("repair_iterations"):
            errors.append("final state iteration differs from repair iterations")
        native_runtime = _strict_nonnegative_float(
            final_state.get("runtime"), field="final state runtime"
        )
        if not math.isclose(
            native_runtime,
            _strict_nonnegative_float(
                result.get("native_runtime"), field="result native_runtime"
            ),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append("final state runtime differs from result native runtime")
        stop_reason = result.get("stop_reason")
        success = result.get("success")
        if success is True and (
            stop_reason != "feasible"
            or final_done is not True
            or result.get("final_conflicts") != 0
        ):
            errors.append("successful result has inconsistent terminal semantics")
        if success is False and stop_reason == "feasible":
            errors.append("failed result uses feasible stop_reason")
        if stop_reason == "time_limit" and final_done is not True:
            errors.append("time-limit result does not end in a terminal state")
        if stop_reason in {"plateau", "portfolio_solved"} and final_done is not False:
            errors.append("early-stop result unexpectedly ends in a terminal state")
    except (KeyError, TypeError, ValueError, OSError) as error:
        errors.append(f"final state validation failed: {error}")

    if not status_path.is_file():
        errors.append("complete status file is missing")
    else:
        status = _read_json(status_path)
        if not isinstance(status, dict) or not _is_finite_json(status):
            errors.append("complete status is not a finite JSON object")
            status = {}
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
        status_expected = {
            "source_seed": result.get("source_seed"),
            "continuation_seed": result.get("continuation_seed"),
            "initial_conflicts": result.get("initial_conflicts"),
            "current_conflicts": result.get("final_conflicts"),
            "repair_iterations": result.get("repair_iterations"),
            "success": result.get("success"),
            "stop_reason": result.get("stop_reason"),
            "elapsed_seconds": result.get("observed_wall_seconds"),
            "completed_at": result.get("completed_at"),
            "latest_diagnostic": (
                result["diagnostics"][-1]
                if isinstance(result.get("diagnostics"), list)
                and result["diagnostics"]
                else None
            ),
        }
        for key, value in status_expected.items():
            if status.get(key) != value:
                errors.append(f"status {key} differs from result")
        if status.get("errors") != 0:
            errors.append("complete status has nonzero errors")
    if (
        result.get("stop_reason") == "portfolio_solved"
        and not errors
        and not _portfolio_is_solved(
            _portfolio_marker_path(
                Path(job["output_root"]), str(job["portfolio_key"])
            ),
            job,
        )
    ):
        errors.append("portfolio_solved result lacks a valid winner marker")
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
        completed = _validate_completed_result(result_path, status_path, job)
        if completed["success"] is True:
            portfolio_path = _portfolio_marker_path(
                output_root, str(job["portfolio_key"])
            )
            if not _portfolio_is_solved(portfolio_path, job):
                final_state_path = _contained_output_file(
                    output_root,
                    completed["final_state_file"],
                    field="final_state_file",
                )
                _mark_portfolio_solved(
                    portfolio_path,
                    job,
                    job_id=job_id,
                    result_path=result_path,
                    status_path=status_path,
                    result=completed,
                    state=_read_gzip_json(final_state_path),
                )
        return completed

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
        step_applied_trajectory: list[bool] = []
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
            step_applied = _strict_bool(
                metrics.get("step_applied"), field="metrics step_applied"
            )
            step_applied_trajectory.append(step_applied)
            repair_iterations += int(step_applied)
            after_conflicts = int(state["num_of_colliding_pairs"])
            if step_applied and after_conflicts >= before_conflicts:
                nonreducing_repairs += 1
            elif step_applied:
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
            "step_applied_trajectory": step_applied_trajectory,
            "diagnostics": diagnostics,
            "final_paths_sha256": _paths_sha256(final_paths),
            "final_state_file": str(final_state_path.relative_to(output_root)),
            "final_state_sha256": _sha256(final_state_path),
            "completed_at": _utc_now(),
        }
        _write_json(result_path, result)
        complete_status = {
            **status,
            "status": "complete",
            "current_conflicts": result["final_conflicts"],
            "repair_iterations": repair_iterations,
            "elapsed_seconds": result["observed_wall_seconds"],
            "success": result["success"],
            "stop_reason": stop_reason,
            "latest_diagnostic": diagnostics[-1] if diagnostics else None,
            "completed_at": result["completed_at"],
        }
        _write_json(status_path, complete_status)
        if result["success"] is True and not _portfolio_is_solved(
            portfolio_stop_path, job
        ):
            _mark_portfolio_solved(
                portfolio_stop_path,
                job,
                job_id=job_id,
                result_path=result_path,
                status_path=status_path,
                result=result,
                state=state,
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
    try:
        source_seeds = _parse_source_seeds(args.source_seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.seed_base < 0:
        parser.error("--seed-base must be non-negative")
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
