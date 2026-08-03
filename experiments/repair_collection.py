from __future__ import annotations

import collections
import datetime as dt
import errno
import hashlib
import json
import math
import multiprocessing
import os
import random
import signal
import socket
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable

from experiments._common import (
    NATIVE_SEMANTICS_SCHEMA,
    contained_file,
    episode_id as _episode_id,
    producer_identity as _structured_producer_identity,
    read_jsonl as _read_jsonl,
)
from experiments.state_analysis import summarize_initial_state_complexity


SCHEMA_VERSION = 1
REPAIR_COLLECTION_ARTIFACT_VERSION = 2
REPAIR_COLLECTION_SCHEMA = "lns2.repair_collection.v2"
EPISODE_SCHEMA = "lns2.repair_episode.v2"
COUNTERFACTUAL_SCHEMA = "lns2.counterfactual.v2"
COUNTERFACTUAL_METADATA_SCHEMA = "lns2.counterfactual_metadata.v2"
NATIVE_REPAIR_TIMING_SCHEMA = "lns2.repair_timing.v2"
REPAIR_TIME_LABEL = "lns2.repair_time.native_step_seconds.v2"
NATIVE_UNLIMITED_TIME_SENTINEL_SECONDS = 1e100
POLICY_DESTROY_STRATEGIES = {
    "official_adaptive": "Adaptive",
    "fixed_target": "Target",
    "fixed_collision": "Collision",
    "fixed_random": "Random",
}
STATE_FINGERPRINT_KEYS = (
    "initialized",
    "initial_solution_complete",
    "feasible",
    "done",
    "iteration",
    "rows",
    "cols",
    "sum_of_costs",
    "num_of_colliding_pairs",
    "low_level",
    "obstacles",
    "conflict_edges",
    "agents",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = PROJECT_ROOT / "build" / ".repair_collection_control"
LOCK_POLL_SECONDS = 0.05
PROCESS_STOP_GRACE_SECONDS = 5.0
REPAIR_COLLECTION_IMPLEMENTATION_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/BasicLNS.h",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


class CollectionLockError(RuntimeError):
    pass


def _repair_time_semantics() -> dict[str, Any]:
    return {
        "label": REPAIR_TIME_LABEL,
        "source_metric": "metrics.native_step_seconds",
        "aggregation": "sum_over_executed_repair_calls",
        "scope": (
            "Native C++ repair-step wall time. Includes native neighborhood "
            "generation, replanning, native state snapshot, bookkeeping, and "
            "native residual time; excludes Python/controller orchestration, "
            "time between calls, and binding/Python conversion outside the "
            "native step."
        ),
        "required_native_timing_schema": NATIVE_REPAIR_TIMING_SCHEMA,
    }


def _producer_identity() -> dict[str, Any]:
    return _structured_producer_identity(
        project_root=PROJECT_ROOT,
        source_files=REPAIR_COLLECTION_IMPLEMENTATION_FILES,
        native_required=True,
    )


def _collection_identity() -> dict[str, Any]:
    return {
        "schema": REPAIR_COLLECTION_SCHEMA,
        "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
        "repair_time_semantics": _repair_time_semantics(),
        "producer": _producer_identity(),
    }


def _artifact_fields(schema: str = REPAIR_COLLECTION_SCHEMA) -> dict[str, Any]:
    return {
        "schema": str(schema),
        "schema_version": REPAIR_COLLECTION_ARTIFACT_VERSION,
        "repair_time_label": REPAIR_TIME_LABEL,
    }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _process_start_token(pid: int) -> str | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except (OSError, UnicodeError):
        return None
    return fields[21] if len(fields) > 21 else None


def _process_matches(owner: dict[str, Any]) -> bool:
    if str(owner.get("host")) != socket.gethostname():
        return True
    try:
        pid = int(owner["pid"])
    except (KeyError, TypeError, ValueError):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno in {errno.ESRCH, errno.EINVAL} or getattr(
            error, "winerror", None
        ) == 87:
            return False
        raise
    expected = owner.get("process_start_token")
    actual = _process_start_token(pid)
    return expected is None or actual is None or str(expected) == actual


def _archive_stale_lock(path: Path) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path.replace(path.with_name(f"{path.name}.stale-{stamp}-{uuid.uuid4().hex[:8]}"))


class _AtomicProcessLock:
    def __init__(self, path: Path, owner: dict[str, Any]) -> None:
        self.path = path
        self.owner = owner
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                try:
                    existing = _read_json(self.path)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    raise CollectionLockError(
                        f"collection lock is unreadable: {self.path}: {error}"
                    ) from error
                if _process_matches(existing):
                    raise CollectionLockError(
                        "another repair collection is active: "
                        f"pid={existing.get('pid')} output={existing.get('output_root')}"
                    )
                _archive_stale_lock(self.path)
                continue
            try:
                with os.fdopen(
                    descriptor, "w", encoding="utf-8", newline="\n"
                ) as stream:
                    json.dump(self.owner, stream, ensure_ascii=False, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                self.path.unlink(missing_ok=True)
                raise
            self.acquired = True
            return

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = _read_json(self.path)
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {}
        if existing.get("run_id") == self.owner.get("run_id"):
            self.path.unlink(missing_ok=True)
        self.acquired = False


class _CollectionRunLock:
    def __init__(
        self,
        output_root: Path,
        run_fingerprint: str,
        phase: str,
        *,
        use_global_lock: bool = True,
    ) -> None:
        pid = os.getpid()
        self.owner = {
            **_artifact_fields(),
            "run_id": uuid.uuid4().hex,
            "run_fingerprint": run_fingerprint,
            "phase": phase,
            "pid": pid,
            "process_start_token": _process_start_token(pid),
            "host": socket.gethostname(),
            "started_at": _utc_now(),
            "output_root": str(output_root),
        }
        self.global_lock = (
            _AtomicProcessLock(CONTROL_ROOT / "active.lock", self.owner)
            if use_global_lock
            else None
        )
        self.output_lock = _AtomicProcessLock(output_root / ".collection.lock", self.owner)

    def __enter__(self) -> dict[str, Any]:
        if self.global_lock is not None:
            self.global_lock.acquire()
        try:
            self.output_lock.acquire()
        except BaseException:
            if self.global_lock is not None:
                self.global_lock.release()
            raise
        return self.owner

    def __exit__(self, *_: Any) -> None:
        self.output_lock.release()
        if self.global_lock is not None:
            self.global_lock.release()


def collection_status(output: str | Path) -> dict[str, Any]:
    output_root = Path(output).resolve()
    lock_path = output_root / ".collection.lock"
    progress_path = output_root / "collection_progress.json"
    owner = _read_json(lock_path) if lock_path.is_file() else None
    return {
        "output_root": str(output_root),
        "active": bool(owner and _process_matches(owner)),
        "lock": owner,
        "progress": _read_json(progress_path) if progress_path.is_file() else None,
    }


def cancel_collection(output: str | Path) -> dict[str, Any]:
    status = collection_status(output)
    owner = status.get("lock")
    if not owner or not status["active"]:
        return {**status, "cancel_requested": False, "reason": "not_active"}
    if str(owner.get("host")) != socket.gethostname():
        raise CollectionLockError("cannot cancel a collection owned by another host")
    pid = int(owner["pid"])
    if not _process_matches(owner):
        return {**status, "cancel_requested": False, "reason": "stale_lock"}
    os.kill(pid, signal.SIGTERM)
    return {**status, "cancel_requested": True, "signal": "SIGTERM"}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        # DrvFS can briefly report a sharing violation when Windows tooling is
        # reading the destination. Keep the atomic replace while tolerating the
        # transient lock instead of aborting a resumable collection.
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.025 * (2**attempt), 0.5))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def state_fingerprint(state: dict[str, Any]) -> str:
    """Hash deterministic solver state while excluding wall-clock and context."""

    # Repair states are already normalized to JSON-native dictionaries/lists by
    # the binding boundary (or by json.load during replay).  Calling the generic
    # ``_fingerprint`` helper would recursively copy every agent path solely to
    # normalize values that are already plain.  Large 400/600-agent states make
    # that defensive copy a measurable per-repair cost, so serialize the schema-
    # constrained state directly while retaining the exact canonical JSON and
    # SHA256 representation used by historical traces.
    deterministic_state = {key: state[key] for key in STATE_FINGERPRINT_KEYS}
    canonical = json.dumps(
        deterministic_state,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def select_seed_agents(
    state: dict[str, Any],
    maximum: int,
    *,
    state_hash: str | None = None,
) -> list[int]:
    if maximum <= 0:
        raise ValueError("maximum seed count must be positive")
    conflicting = [
        agent for agent in state["agents"] if int(agent["conflict_degree"]) > 0
    ]
    selected: list[int] = []

    def add(values: Iterable[dict[str, Any]]) -> None:
        for value in values:
            agent_id = int(value["id"])
            if agent_id not in selected and len(selected) < maximum:
                selected.append(agent_id)

    add(
        sorted(
            conflicting,
            key=lambda item: (
                -int(item["conflict_degree"]),
                -int(item["delay"]),
                int(item["id"]),
            ),
        )[:2]
    )
    add(
        sorted(
            conflicting,
            key=lambda item: (
                -int(item["delay"]),
                -int(item["conflict_degree"]),
                int(item["id"]),
            ),
        )[:2]
    )
    remaining = [
        int(item["id"])
        for item in conflicting
        if int(item["id"]) not in selected
    ]
    # Online callers already fingerprint the complete solver state for action
    # seeding and trace validation. Reusing that digest avoids hashing every
    # path a second time merely to shuffle the final seed-agent tie group.
    fingerprint = state_fingerprint(state) if state_hash is None else str(state_hash)
    rng = random.Random(int(fingerprint[:16], 16))
    rng.shuffle(remaining)
    for agent_id in remaining:
        if len(selected) >= maximum:
            break
        selected.append(agent_id)
    return selected


def candidate_actions(
    state: dict[str, Any],
    maximum_seeds: int,
    heuristics: list[str],
    neighborhood_sizes: list[int],
) -> list[dict[str, Any]]:
    supported = {"target", "collision", "random"}
    if not heuristics or any(value not in supported for value in heuristics):
        raise ValueError("counterfactual heuristics must be target, collision, or random")
    if not neighborhood_sizes or any(value <= 0 for value in neighborhood_sizes):
        raise ValueError("counterfactual neighborhood sizes must be positive")
    return [
        {
            "mode": "seed",
            "heuristic": heuristic,
            "seed_agent": seed_agent,
            "neighborhood_size": size,
        }
        for seed_agent in select_seed_agents(state, maximum_seeds)
        for heuristic in heuristics
        for size in neighborhood_sizes
    ]


def _load_environment_module() -> Any:
    try:
        import lns2_env
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "lns2_env is unavailable; build the native module and set PYTHONPATH"
        ) from error
    return lns2_env


def _context(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "split",
        "map_id",
        "task_id",
        "layout_mode",
        "layout_variant",
        "scenario_type",
        "task_variant",
        "agent_count",
        "topology_metrics",
        "dominant_flow_ratio",
        "hotspot_skew",
        "required_bottleneck_crossing_ratio",
        "mean_shortest_distance",
    )
    return {key: _plain(row.get(key)) for key in keys}


def _make_environment(
    dataset_root: str,
    row: dict[str, Any],
    environment_config: dict[str, Any],
    destroy_strategy: str,
) -> Any:
    module = _load_environment_module()
    native_timing_schema = str(
        getattr(module, "repair_timing_schema", "")
    )
    if native_timing_schema != NATIVE_REPAIR_TIMING_SCHEMA:
        raise RuntimeError(
            "repair collection requires native timing schema "
            f"{NATIVE_REPAIR_TIMING_SCHEMA}; got "
            f"{native_timing_schema or 'missing'}"
        )
    native_semantics_schema = str(
        getattr(module, "native_semantics_schema", "")
    )
    if native_semantics_schema != NATIVE_SEMANTICS_SCHEMA:
        raise RuntimeError(
            "repair collection requires native semantics schema "
            f"{NATIVE_SEMANTICS_SCHEMA}; got "
            f"{native_semantics_schema or 'missing'}"
        )
    split_root = Path(dataset_root) / str(row["split"])
    unlimited_time = bool(environment_config.get("unlimited_time", False))
    configured_time_limit = float(environment_config["time_limit"])
    if unlimited_time and configured_time_limit != 0.0:
        raise ValueError("unlimited native time requires time_limit=0")
    native_time_limit = (
        NATIVE_UNLIMITED_TIME_SENTINEL_SECONDS
        if unlimited_time
        else configured_time_limit
    )
    return module.LNS2RepairEnv(
        str(split_root / str(row["map_file"])),
        str(split_root / str(row["scenario_file"])),
        agent_count=int(row["agent_count"]),
        time_limit=native_time_limit,
        neighborhood_size=int(environment_config["neighborhood_size"]),
        destroy_strategy=destroy_strategy,
        replan_algorithm=str(environment_config["replan_algorithm"]),
        use_sipp=bool(environment_config["use_sipp"]),
        max_repair_iterations=int(environment_config["max_repair_iterations"]),
        screen=0,
        context=_context(row),
    )


def _low_level_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int]:
    return {
        key: int(after["low_level"][key]) - int(before["low_level"][key])
        for key in ("expanded", "generated", "reopened", "runs")
    }


def _native_step_seconds(metrics: dict[str, Any]) -> float:
    """Validate and read the strict v2 native repair-step time metric."""

    required = {
        "native_step_seconds",
        "step_runtime",
        "episode_runtime_delta_seconds",
    }
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(
            "repair collection requires native timing v2 metrics; "
            f"missing {missing}"
        )
    raw_values = tuple(metrics[key] for key in sorted(required))
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_values
    ):
        raise ValueError("native timing v2 metrics must be numeric")
    try:
        native_step = float(metrics["native_step_seconds"])
        step_runtime = float(metrics["step_runtime"])
        episode_delta = float(metrics["episode_runtime_delta_seconds"])
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("native timing v2 metrics must be numeric") from error
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (native_step, step_runtime, episode_delta)
    ):
        raise ValueError("native timing v2 metrics must be finite and non-negative")
    tolerance = max(1e-6, 0.01 * max(native_step, step_runtime, 1e-6))
    if not math.isclose(
        step_runtime,
        native_step,
        rel_tol=0.01,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "native timing v2 step_runtime does not match native_step_seconds"
        )
    # episode_delta and native_step have different timing boundaries.  The
    # native value includes the final state snapshot; the episode value may
    # instead include controller work between native calls, so neither bounds
    # the other.
    return native_step


def _is_finite_json_tree(value: Any) -> bool:
    """Return whether a value is a finite, JSON-shaped artifact tree."""

    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_finite_json_tree(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_finite_json_tree(item)
            for key, item in value.items()
        )
    return False


def _conflict_auc(values: list[int]) -> float:
    return sum(
        (float(values[index]) + float(values[index + 1])) / 2.0
        for index in range(len(values) - 1)
    )


def _qualification_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    solver_seed = int(job["solver_seed"])
    try:
        environment = _make_environment(
            job["dataset_root"], row, job["environment"], "Adaptive"
        )
        state = _plain(environment.reset(seed=solver_seed))
        initial_complexity = summarize_initial_state_complexity(state)
        return {
            **_artifact_fields(),
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "initial_conflicts": int(state["num_of_colliding_pairs"]),
            "repairable": not bool(state["done"]),
            "initial_feasible": bool(state["feasible"]),
            "initial_complete": bool(state["initial_solution_complete"]),
            "state_fingerprint": state_fingerprint(state),
            "initial_complexity": initial_complexity,
            "status": "ok",
            "error": None,
        }
    except Exception as error:
        return {
            **_artifact_fields(),
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def _valid_episode_trace(
    path: Path,
    run_fingerprint: str,
    *,
    expected_episode_id: str | None = None,
    expected_policy: str | None = None,
    expected_solver_seed: int | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        rows = _read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not rows
        or any(
            not isinstance(row, dict) or not _is_finite_json_tree(row)
            for row in rows
        )
        or rows[0].get("event") != "initial"
    ):
        return None
    if rows[-1].get("event") != "finish":
        return None
    if any(str(row.get("event")) != "transition" for row in rows[1:-1]):
        return None
    for row in rows:
        if (
            str(row.get("schema")) != EPISODE_SCHEMA
            or row.get("schema_version")
            != REPAIR_COLLECTION_ARTIFACT_VERSION
            or str(row.get("repair_time_label")) != REPAIR_TIME_LABEL
            or str(row.get("run_fingerprint")) != str(run_fingerprint)
        ):
            return None
    initial = rows[0]
    finish = rows[-1]
    episode_id = str(initial.get("episode_id") or "")
    policy = str(initial.get("policy") or "")
    solver_seed = _strict_int(initial.get("solver_seed"))
    if solver_seed is None:
        return None
    if (
        not episode_id
        or not policy
        or any(str(row.get("episode_id") or "") != episode_id for row in rows)
        or (
            expected_episode_id is not None
            and episode_id != str(expected_episode_id)
        )
        or (expected_policy is not None and policy != str(expected_policy))
        or (
            expected_solver_seed is not None
            and solver_seed != int(expected_solver_seed)
        )
    ):
        return None
    initial_state = initial.get("state")
    if not isinstance(initial_state, dict):
        return None
    try:
        if str(initial.get("state_fingerprint")) != state_fingerprint(
            initial_state
        ):
            return None
        initial_conflicts = _strict_int(
            initial_state["num_of_colliding_pairs"]
        )
        initial_cost = _strict_int(initial_state["sum_of_costs"])
        initial_low_level = _validated_low_level(initial_state["low_level"])
        if (
            initial_conflicts is None
            or initial_conflicts < 0
            or initial_cost is None
            or initial_cost < 0
            or initial_low_level is None
            or not isinstance(initial_state.get("feasible"), bool)
            or not isinstance(initial_state.get("done"), bool)
        ):
            return None
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    transitions = [
        row for row in rows if str(row.get("event")) == "transition"
    ]
    state = initial_state
    conflicts = [initial_conflicts]
    native_step_times: list[float] = []
    try:
        for transition in transitions:
            if (
                str(transition.get("native_timing_schema"))
                != NATIVE_REPAIR_TIMING_SCHEMA
                or not isinstance(transition.get("metrics"), dict)
                or transition.get("action") != {"mode": "official"}
                or state.get("done") is not False
            ):
                return None
            if str(transition.get("before_fingerprint")) != state_fingerprint(state):
                return None
            after = transition.get("after")
            if (
                not isinstance(after, dict)
                or str(transition.get("after_fingerprint"))
                != state_fingerprint(after)
            ):
                return None
            if (
                not isinstance(after.get("feasible"), bool)
                or not isinstance(after.get("done"), bool)
                or not isinstance(transition.get("terminated"), bool)
                or not isinstance(transition.get("truncated"), bool)
                or transition["terminated"] is not after["feasible"]
                or transition["truncated"]
                is not (after["done"] and not after["feasible"])
                or after["done"]
                is not (transition["terminated"] or transition["truncated"])
            ):
                return None
            before_conflicts = _strict_int(state.get("num_of_colliding_pairs"))
            after_conflicts = _strict_int(after.get("num_of_colliding_pairs"))
            before_cost = _strict_int(state.get("sum_of_costs"))
            after_cost = _strict_int(after.get("sum_of_costs"))
            before_low_level = _validated_low_level(state.get("low_level"))
            after_low_level = _validated_low_level(after.get("low_level"))
            metrics = transition["metrics"]
            if (
                before_conflicts is None
                or before_conflicts < 0
                or after_conflicts is None
                or after_conflicts < 0
                or before_cost is None
                or before_cost < 0
                or after_cost is None
                or after_cost < 0
                or before_low_level is None
                or after_low_level is None
                or _strict_int(metrics.get("conflicts_before"))
                != before_conflicts
                or _strict_int(metrics.get("conflicts_after"))
                != after_conflicts
                or _strict_int(metrics.get("sum_of_costs_before"))
                != before_cost
                or _strict_int(metrics.get("sum_of_costs_after"))
                != after_cost
                or metrics.get("action_valid") is not True
            ):
                return None
            expected_low_level_delta = {
                key: after_low_level[key] - before_low_level[key]
                for key in before_low_level
            }
            if (
                any(value < 0 for value in expected_low_level_delta.values())
                or _validated_low_level(transition.get("low_level_delta"))
                != expected_low_level_delta
            ):
                return None
            native_step_times.append(
                _native_step_seconds(dict(metrics))
            )
            state = after
            conflicts.append(after_conflicts)
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    try:
        finish_state = finish.get("state")
        if finish_state is not None:
            if not isinstance(finish_state, dict):
                return None
            finish_fingerprint = state_fingerprint(finish_state)
            if str(finish.get("final_fingerprint")) != finish_fingerprint:
                return None
            if finish_fingerprint != state_fingerprint(state):
                prior_payload = {
                    key: state[key] for key in STATE_FINGERPRINT_KEYS
                }
                final_payload = {
                    key: finish_state[key] for key in STATE_FINGERPRINT_KEYS
                }
                prior_payload["done"] = final_payload["done"]
                if (
                    state.get("done") is not False
                    or finish_state.get("done") is not True
                    or finish_state.get("feasible") is not False
                    or prior_payload != final_payload
                ):
                    return None
            state = finish_state
        finish_matches = (
            str(finish.get("final_fingerprint")) == state_fingerprint(state)
            and isinstance(finish.get("success"), bool)
            and finish["success"] is state["feasible"]
            and state["done"] is True
        )
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    if not finish_matches:
        return None
    summary = finish.get("summary")
    if (
        not isinstance(summary, dict)
        or str(summary.get("repair_step_runtime_label")) != REPAIR_TIME_LABEL
        or str(summary.get("time_to_feasible_label")) != REPAIR_TIME_LABEL
    ):
        return None
    native_step_total = sum(native_step_times)
    try:
        feasible = state["feasible"]
        done = state["done"]
        final_sum_of_costs = _strict_int(state["sum_of_costs"])
        initial_runtime = _finite_number(initial_state["runtime"])
        if (
            final_sum_of_costs is None
            or final_sum_of_costs < 0
            or initial_runtime is None
            or initial_runtime < 0.0
        ):
            return None
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    expected_time_to_feasible = native_step_total if feasible else None
    exact_expected = {
        "initial_conflicts": conflicts[0],
        "final_conflicts": conflicts[-1],
        "repairable": conflicts[0] > 0,
        "success": feasible,
        "truncated": bool(done and not feasible),
        "repair_iterations": len(transitions),
        "conflict_trajectory": conflicts,
        "final_sum_of_costs": final_sum_of_costs,
    }
    if any(summary.get(key) != value for key, value in exact_expected.items()):
        return None
    try:
        float_expected = {
            "conflict_auc": _conflict_auc(conflicts),
            "initial_runtime": initial_runtime,
            "repair_step_runtime": native_step_total,
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in float_expected.values()
        ):
            return None
        if any(
            not math.isclose(
                float(summary[key]),
                value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for key, value in float_expected.items()
        ):
            return None
        if expected_time_to_feasible is None:
            if summary.get("time_to_feasible") is not None:
                return None
        elif not math.isclose(
            float(summary["time_to_feasible"]),
            expected_time_to_feasible,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return None
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    return summary


def _baseline_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    policy = str(job["policy"])
    solver_seed = int(job["solver_seed"])
    episode_id = _episode_id(row, solver_seed, policy)
    output_root = Path(job["output_root"])
    trace_path = output_root / "episodes" / str(row["split"]) / policy / f"{episode_id}.jsonl"
    relative_trace = trace_path.relative_to(output_root).as_posix()
    if job["resume"]:
        summary = _valid_episode_trace(
            trace_path,
            job["run_fingerprint"],
            expected_episode_id=episode_id,
            expected_policy=policy,
            expected_solver_seed=solver_seed,
        )
        if summary is not None:
            return {
                **_artifact_fields(),
                "episode_id": episode_id,
                "split": row["split"],
                "map_id": row["map_id"],
                "task_id": row["task_id"],
                "layout_mode": row["layout_mode"],
                "task_variant": row.get("task_variant"),
                "agent_count": int(row["agent_count"]),
                "solver_seed": solver_seed,
                "policy": policy,
                "trace_file": relative_trace,
                "status": "resumed",
                "summary": summary,
                "error": None,
            }
        if trace_path.is_file():
            try:
                existing_rows = _read_jsonl(trace_path)
            except (OSError, json.JSONDecodeError):
                existing_rows = None
            completed_or_corrupt = existing_rows is None
            if existing_rows:
                rows_are_objects = all(
                    isinstance(item, dict) and _is_finite_json_tree(item)
                    for item in existing_rows
                )
                incomplete_shape = (
                    rows_are_objects
                    and existing_rows[0].get("event") == "initial"
                    and all(
                        item.get("event") == "transition"
                        for item in existing_rows[1:]
                    )
                )
                completed_or_corrupt = not incomplete_shape
            if completed_or_corrupt:
                return {
                    **_artifact_fields(),
                    "episode_id": episode_id,
                    "split": row["split"],
                    "map_id": row["map_id"],
                    "task_id": row["task_id"],
                    "layout_mode": row["layout_mode"],
                    "task_variant": row.get("task_variant"),
                    "agent_count": int(row["agent_count"]),
                    "solver_seed": solver_seed,
                    "policy": policy,
                    "trace_file": relative_trace,
                    "status": "error",
                    "summary": None,
                    "error": (
                        "existing completed or corrupt episode trace failed "
                        "integrity validation; preserving it unchanged"
                    ),
                }
    try:
        environment = _make_environment(
            job["dataset_root"],
            row,
            job["environment"],
            POLICY_DESTROY_STRATEGIES[policy],
        )
        state = _plain(environment.reset(seed=solver_seed))
        conflicts = [int(state["num_of_colliding_pairs"])]
        events: list[dict[str, Any]] = [
            {
                **_artifact_fields(EPISODE_SCHEMA),
                "run_fingerprint": job["run_fingerprint"],
                "event": "initial",
                "episode_id": episode_id,
                "policy": policy,
                "solver_seed": solver_seed,
                "state_fingerprint": state_fingerprint(state),
                "state": state,
            }
        ]
        step_runtime = 0.0
        while not bool(state["done"]):
            before = state
            action = {"mode": "official"}
            result = _plain(environment.step(action))
            state = result["observation"]
            metrics = result["metrics"]
            if metrics.get("step_applied") is False:
                if not bool(result["truncated"]) or not bool(state["done"]):
                    raise RuntimeError(
                        "a non-applied repair step did not return a truncated terminal state"
                    )
                break
            step_runtime += _native_step_seconds(metrics)
            conflicts.append(int(state["num_of_colliding_pairs"]))
            events.append(
                {
                    **_artifact_fields(EPISODE_SCHEMA),
                    "run_fingerprint": job["run_fingerprint"],
                    "event": "transition",
                    "episode_id": episode_id,
                    "native_timing_schema": NATIVE_REPAIR_TIMING_SCHEMA,
                    "action": action,
                    "before_fingerprint": state_fingerprint(before),
                    "after_fingerprint": state_fingerprint(state),
                    "metrics": metrics,
                    "low_level_delta": _low_level_delta(before, state),
                    "terminated": bool(result["terminated"]),
                    "truncated": bool(result["truncated"]),
                    "after": state,
                }
            )
        summary = {
            "initial_conflicts": conflicts[0],
            "final_conflicts": conflicts[-1],
            "repairable": conflicts[0] > 0,
            "success": bool(state["feasible"]),
            "truncated": bool(state["done"] and not state["feasible"]),
            "repair_iterations": len(conflicts) - 1,
            "conflict_trajectory": conflicts,
            "conflict_auc": _conflict_auc(conflicts),
            "initial_runtime": float(events[0]["state"]["runtime"]),
            "repair_step_runtime": step_runtime,
            "repair_step_runtime_label": REPAIR_TIME_LABEL,
            "time_to_feasible": step_runtime if state["feasible"] else None,
            "time_to_feasible_label": REPAIR_TIME_LABEL,
            "final_sum_of_costs": int(state["sum_of_costs"]),
        }
        events.append(
            {
                **_artifact_fields(EPISODE_SCHEMA),
                "run_fingerprint": job["run_fingerprint"],
                "event": "finish",
                "episode_id": episode_id,
                "success": bool(state["feasible"]),
                "final_fingerprint": state_fingerprint(state),
                "state": state,
                "summary": summary,
            }
        )
        _write_jsonl(trace_path, events)
        return {
            **_artifact_fields(),
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_file": relative_trace,
            "status": "ok",
            "summary": summary,
            "error": None,
        }
    except Exception as error:
        return {
            **_artifact_fields(),
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_file": None,
            "status": "error",
            "summary": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _decision_states(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events or events[0].get("event") != "initial":
        raise ValueError("baseline trace does not start with an initial event")
    decisions: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    initial = events[0]["state"]
    if not initial["done"]:
        decisions.append(
            {"decision_index": 0, "state": initial, "prefix_actions": []}
        )
    index = 1
    for event in events:
        if event.get("event") != "transition":
            continue
        prefix.append(_plain(event["action"]))
        state = event["after"]
        if not state["done"]:
            decisions.append(
                {
                    "decision_index": index,
                    "state": state,
                    "prefix_actions": list(prefix),
                }
            )
        index += 1
    return decisions


def _select_evenly(values: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if maximum <= 0:
        raise ValueError("maximum state count must be positive")
    if len(values) <= maximum:
        return values
    if maximum == 1:
        return [values[0]]
    indices = {
        round(index * (len(values) - 1) / (maximum - 1))
        for index in range(maximum)
    }
    return [values[index] for index in sorted(indices)]


def _trial_seed(
    episode_id: str,
    state_id: str,
    action: dict[str, Any],
    trial_index: int,
) -> int:
    value = _fingerprint(
        {
            "episode_id": episode_id,
            "state_id": state_id,
            "action": action,
            "trial_index": trial_index,
        }
    )
    return int(value[:16], 16) % (2**31)


def _horizon_outcomes(
    initial: dict[str, Any],
    points: list[dict[str, Any]],
    horizons: list[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for horizon in horizons:
        if len(points) - 1 >= horizon:
            available = True
            point = points[horizon]
            selected = points[: horizon + 1]
        elif points[-1]["state"]["feasible"]:
            available = True
            point = points[-1]
            selected = list(points)
            while len(selected) < horizon + 1:
                selected.append(
                    {
                        **point,
                        "step": len(selected),
                        "step_runtime": 0.0,
                    }
                )
        else:
            available = False
            point = points[-1]
            selected = points
        conflicts = [int(item["state"]["num_of_colliding_pairs"]) for item in selected]
        solved_step = next(
            (
                int(item["step"])
                for item in selected
                if bool(item["state"]["feasible"])
            ),
            None,
        )
        results.append(
            {
                "horizon": horizon,
                "available": available,
                "executed_steps": min(horizon, len(points) - 1),
                "solved": bool(point["state"]["feasible"]),
                "solved_step": solved_step,
                "conflicts_after": int(point["state"]["num_of_colliding_pairs"]),
                "conflict_reduction": int(initial["num_of_colliding_pairs"])
                - int(point["state"]["num_of_colliding_pairs"]),
                "conflict_auc": _conflict_auc(conflicts) if available else None,
                "sum_of_costs_after": int(point["state"]["sum_of_costs"]),
                "cost_improvement": int(initial["sum_of_costs"])
                - int(point["state"]["sum_of_costs"]),
                "low_level_delta": _low_level_delta(initial, point["state"]),
                "branch_runtime": sum(
                    float(item.get("step_runtime", 0.0)) for item in selected[1:]
                ),
                "branch_runtime_label": REPAIR_TIME_LABEL,
                "time_to_feasible": (
                    sum(
                        float(item.get("step_runtime", 0.0))
                        for item in selected[1 : solved_step + 1]
                    )
                    if solved_step is not None
                    else None
                ),
                "time_to_feasible_label": REPAIR_TIME_LABEL,
            }
        )
    return results


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _valid_state_fingerprint(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_low_level(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int] = {}
    for key in ("expanded", "generated", "reopened", "runs"):
        parsed = _strict_int(value.get(key))
        if parsed is None or parsed < 0:
            return None
        result[key] = parsed
    if set(value) != set(result):
        return None
    return result


def _normalized_horizons(values: Any) -> list[int] | None:
    if not isinstance(values, (list, tuple)) or not values:
        return None
    horizons: list[int] = []
    for value in values:
        parsed = _strict_int(value)
        if parsed is None or parsed <= 0:
            return None
        horizons.append(parsed)
    if len(horizons) != len(set(horizons)):
        return None
    return sorted(horizons)


def _normalized_counterfactual_validation_config(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not _is_finite_json_tree(value):
        return None
    trials = _strict_int(value.get("trials"))
    maximum_seeds = _strict_int(value.get("max_seed_agents"))
    heuristics = value.get("heuristics")
    neighborhood_sizes = value.get("neighborhood_sizes")
    horizons = _normalized_horizons(value.get("horizons"))
    if (
        trials is None
        or trials <= 0
        or maximum_seeds is None
        or maximum_seeds <= 0
        or not isinstance(heuristics, list)
        or not heuristics
        or any(
            not isinstance(item, str)
            or item not in {"target", "collision", "random"}
            for item in heuristics
        )
        or not isinstance(neighborhood_sizes, list)
        or not neighborhood_sizes
        or any(
            _strict_int(item) is None or item <= 0
            for item in neighborhood_sizes
        )
        or horizons is None
    ):
        return None
    return {
        "trials": trials,
        "max_seed_agents": maximum_seeds,
        "heuristics": list(heuristics),
        "neighborhood_sizes": list(neighborhood_sizes),
        "horizons": horizons,
    }


def _counterfactual_outcome_timing_is_valid(
    row: dict[str, Any],
    *,
    expected_horizons: list[int],
) -> bool:
    if str(row.get("step_runtime_label")) != REPAIR_TIME_LABEL:
        return False
    normalized_horizons = _normalized_horizons(expected_horizons)
    if normalized_horizons is None:
        return False
    steps = row.get("steps")
    horizons = row.get("horizon_outcomes")
    if (
        not isinstance(steps, list)
        or len(steps) < 2
        or not isinstance(horizons, list)
        or len(horizons) != len(normalized_horizons)
    ):
        return False

    step_times: list[float] = []
    step_conflicts: list[int] = []
    step_costs: list[int] = []
    step_fingerprints: list[str] = []
    step_low_levels: list[dict[str, int]] = []
    try:
        for expected_step, step_value in enumerate(steps):
            if not isinstance(step_value, dict):
                return False
            step = dict(step_value)
            if _strict_int(step.get("step")) != expected_step:
                return False
            stored = _finite_number(step.get("step_runtime"))
            conflicts = _strict_int(step.get("conflicts"))
            sum_of_costs = _strict_int(step.get("sum_of_costs"))
            fingerprint = step.get("state_fingerprint")
            low_level = _validated_low_level(step.get("low_level"))
            terminated = step.get("terminated")
            truncated = step.get("truncated")
            if (
                stored is None
                or stored < 0.0
                or conflicts is None
                or conflicts < 0
                or sum_of_costs is None
                or sum_of_costs < 0
                or not _valid_state_fingerprint(fingerprint)
                or low_level is None
                or not isinstance(terminated, bool)
                or not isinstance(truncated, bool)
                or (terminated and truncated)
                or ((terminated or truncated) and expected_step != len(steps) - 1)
            ):
                return False
            if expected_step == 0:
                if (
                    step.get("action") is not None
                    or step.get("metrics") is not None
                    or terminated
                    or truncated
                    or not math.isclose(
                        stored, 0.0, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    return False
            else:
                metrics = step.get("metrics")
                if (
                    not isinstance(metrics, dict)
                    or not isinstance(metrics.get("action_valid"), bool)
                    or (
                        expected_step > 1
                        and step.get("action") != {"mode": "official"}
                    )
                ):
                    return False
                expected = _native_step_seconds(dict(metrics))
                if not math.isclose(
                    stored, expected, rel_tol=1e-9, abs_tol=1e-9
                ):
                    return False
                metric_expectations = {
                    "conflicts_before": step_conflicts[-1],
                    "conflicts_after": conflicts,
                    "sum_of_costs_before": step_costs[-1],
                    "sum_of_costs_after": sum_of_costs,
                }
                if any(
                    _strict_int(metrics.get(key)) != expected_value
                    for key, expected_value in metric_expectations.items()
                ):
                    return False
                if terminated is not (conflicts == 0):
                    return False
            step_times.append(stored)
            step_conflicts.append(conflicts)
            step_costs.append(sum_of_costs)
            step_fingerprints.append(str(fingerprint))
            step_low_levels.append(low_level)

        if step_conflicts[0] <= 0:
            return False
        trajectory = row.get("conflict_trajectory")
        if (
            not isinstance(trajectory, list)
            or len(trajectory) != len(step_conflicts)
            or any(
                _strict_int(value) != expected
                for value, expected in zip(trajectory, step_conflicts)
            )
            or row.get("state_fingerprint") != step_fingerprints[0]
        ):
            return False
        action_valid = row.get("action_valid")
        first_action_valid = steps[1]["metrics"].get("action_valid")
        if (
            not isinstance(action_valid, bool)
            or not isinstance(first_action_valid, bool)
            or action_valid is not first_action_valid
        ):
            return False

        actual_horizons: list[int] = []
        for expected_horizon, horizon_value in zip(
            normalized_horizons, horizons
        ):
            if not isinstance(horizon_value, dict):
                return False
            horizon_row = dict(horizon_value)
            horizon = _strict_int(horizon_row.get("horizon"))
            if horizon != expected_horizon:
                return False
            actual_horizons.append(horizon)
            executed_steps = _strict_int(horizon_row.get("executed_steps"))
            expected_executed = min(horizon, len(step_times) - 1)
            if executed_steps != expected_executed:
                return False
            if len(step_times) - 1 >= horizon:
                expected_available = True
                point_index = horizon
                selected_conflicts = step_conflicts[: horizon + 1]
            elif step_conflicts[-1] == 0:
                expected_available = True
                point_index = len(step_times) - 1
                selected_conflicts = list(step_conflicts)
                selected_conflicts.extend(
                    [step_conflicts[-1]]
                    * (horizon + 1 - len(selected_conflicts))
                )
            else:
                expected_available = False
                point_index = len(step_times) - 1
                selected_conflicts = list(step_conflicts)
            if (
                not isinstance(horizon_row.get("available"), bool)
                or horizon_row["available"] is not expected_available
            ):
                return False
            if str(horizon_row.get("branch_runtime_label")) != REPAIR_TIME_LABEL:
                return False
            if (
                str(horizon_row.get("time_to_feasible_label"))
                != REPAIR_TIME_LABEL
            ):
                return False
            branch_runtime = _finite_number(horizon_row.get("branch_runtime"))
            expected_branch = math.fsum(
                step_times[1 : expected_executed + 1]
            )
            if (
                branch_runtime is None
                or branch_runtime < 0.0
                or not math.isclose(
                    branch_runtime,
                    expected_branch,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                return False

            expected_solved_step = next(
                (
                    index
                    for index, conflicts in enumerate(selected_conflicts)
                    if conflicts == 0
                ),
                None,
            )
            expected_solved = step_conflicts[point_index] == 0
            if (
                not isinstance(horizon_row.get("solved"), bool)
                or horizon_row["solved"] is not expected_solved
            ):
                return False
            solved_step_value = horizon_row.get("solved_step")
            if expected_solved_step is None:
                if (
                    solved_step_value is not None
                    or horizon_row.get("time_to_feasible") is not None
                ):
                    return False
            else:
                solved_step = _strict_int(solved_step_value)
                if solved_step != expected_solved_step:
                    return False
                time_to_feasible = _finite_number(
                    horizon_row.get("time_to_feasible")
                )
                expected_time = math.fsum(step_times[1 : solved_step + 1])
                if (
                    time_to_feasible is None
                    or time_to_feasible < 0.0
                    or not math.isclose(
                        time_to_feasible,
                        expected_time,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                ):
                    return False
            expected_auc = (
                _conflict_auc(selected_conflicts)
                if expected_available
                else None
            )
            conflict_auc = horizon_row.get("conflict_auc")
            if expected_auc is None:
                if conflict_auc is not None:
                    return False
            else:
                parsed_auc = _finite_number(conflict_auc)
                if (
                    parsed_auc is None
                    or parsed_auc < 0.0
                    or not math.isclose(
                        parsed_auc,
                        expected_auc,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                ):
                    return False
            integer_expectations = {
                "conflicts_after": step_conflicts[point_index],
                "conflict_reduction": (
                    step_conflicts[0] - step_conflicts[point_index]
                ),
                "sum_of_costs_after": step_costs[point_index],
                "cost_improvement": (
                    step_costs[0] - step_costs[point_index]
                ),
            }
            if any(
                _strict_int(horizon_row.get(key)) != expected
                for key, expected in integer_expectations.items()
            ):
                return False
            expected_low_level = {
                key: (
                    step_low_levels[point_index][key]
                    - step_low_levels[0][key]
                )
                for key in step_low_levels[0]
            }
            if any(value < 0 for value in expected_low_level.values()):
                return False
            if (
                _validated_low_level(horizon_row.get("low_level_delta"))
                != expected_low_level
            ):
                return False
        if actual_horizons != normalized_horizons:
            return False
    except (KeyError, OverflowError, TypeError, ValueError):
        return False
    return True


def _counterfactual_artifact_rows_are_valid(
    rows_by_field: dict[str, list[Any]],
    *,
    run_fingerprint: str,
    episode_id: str,
    counterfactual_config: dict[str, Any],
) -> bool:
    config = _normalized_counterfactual_validation_config(
        counterfactual_config
    )
    if config is None or set(rows_by_field) != {
        "states_file",
        "outcomes_file",
        "errors_file",
    }:
        return False
    for field, rows in rows_by_field.items():
        for row in rows:
            if not isinstance(row, dict) or not _is_finite_json_tree(row):
                return False
            if (
                str(row.get("schema")) != COUNTERFACTUAL_SCHEMA
                or row.get("schema_version")
                != REPAIR_COLLECTION_ARTIFACT_VERSION
                or str(row.get("repair_time_label")) != REPAIR_TIME_LABEL
                or str(row.get("run_fingerprint")) != str(run_fingerprint)
                or str(row.get("episode_id")) != str(episode_id)
            ):
                return False

    state_rows = rows_by_field["states_file"]
    states: dict[str, dict[str, Any]] = {}
    decision_indices: list[int] = []
    try:
        for row in state_rows:
            state = row.get("state")
            state_id = row.get("state_id")
            decision_index = _strict_int(row.get("decision_index"))
            prefix_actions = row.get("prefix_actions")
            candidate_count = _strict_int(row.get("candidate_count"))
            conflicts = (
                _strict_int(state.get("num_of_colliding_pairs"))
                if isinstance(state, dict)
                else None
            )
            if (
                not isinstance(state, dict)
                or not isinstance(state_id, str)
                or not state_id
                or decision_index is None
                or decision_index < 0
                or state_id
                != f"{episode_id}__decision_{decision_index:04d}"
                or state_id in states
                or not isinstance(prefix_actions, list)
                or len(prefix_actions) != decision_index
                or any(
                    action != {"mode": "official"}
                    for action in prefix_actions
                )
                or candidate_count is None
                or candidate_count < 0
                or state.get("done") is not False
                or state.get("feasible") is not False
                or conflicts is None
                or conflicts <= 0
                or _validated_low_level(state.get("low_level")) is None
            ):
                return False
            fingerprint = state_fingerprint(state)
            if (
                row.get("state_fingerprint") != fingerprint
                or not _valid_state_fingerprint(fingerprint)
            ):
                return False
            actions = candidate_actions(
                state,
                config["max_seed_agents"],
                config["heuristics"],
                config["neighborhood_sizes"],
            )
            if candidate_count != len(actions):
                return False
            states[state_id] = {
                "fingerprint": fingerprint,
                "actions": actions,
            }
            decision_indices.append(decision_index)
    except (KeyError, OverflowError, TypeError, ValueError):
        return False
    if (
        decision_indices != sorted(decision_indices)
        or len(decision_indices) != len(set(decision_indices))
        or (decision_indices and decision_indices[0] != 0)
    ):
        return False

    observed: set[tuple[str, int, int]] = set()
    for field in ("outcomes_file", "errors_file"):
        for row in rows_by_field[field]:
            state_id = row.get("state_id")
            candidate_index = _strict_int(row.get("candidate_index"))
            trial_index = _strict_int(row.get("trial_index"))
            if (
                not isinstance(state_id, str)
                or state_id not in states
                or candidate_index is None
                or candidate_index < 0
                or candidate_index >= len(states[state_id]["actions"])
                or trial_index is None
                or trial_index < 0
                or trial_index >= config["trials"]
                or row.get("state_fingerprint")
                != states[state_id]["fingerprint"]
            ):
                return False
            key = (state_id, candidate_index, trial_index)
            if key in observed:
                return False
            observed.add(key)
            action = states[state_id]["actions"][candidate_index]
            trial_seed = _trial_seed(
                episode_id,
                state_id,
                action,
                trial_index,
            )
            if _strict_int(row.get("trial_seed")) != trial_seed:
                return False
            if field == "outcomes_file":
                expected_action = {**action, "random_seed": trial_seed}
                steps = row.get("steps")
                if (
                    row.get("candidate_action") != expected_action
                    or not isinstance(steps, list)
                    or len(steps) < 2
                    or not isinstance(steps[1], dict)
                    or steps[1].get("action") != expected_action
                    or not _counterfactual_outcome_timing_is_valid(
                        row,
                        expected_horizons=config["horizons"],
                    )
                ):
                    return False
            elif (
                row.get("candidate_action") != action
                or not isinstance(row.get("error"), str)
                or not row["error"]
            ):
                return False

    expected = {
        (state_id, candidate_index, trial_index)
        for state_id, state in states.items()
        for candidate_index in range(len(state["actions"]))
        for trial_index in range(config["trials"])
    }
    return observed == expected


def _valid_counterfactual_resume_metadata(
    metadata: dict[str, Any],
    run_fingerprint: str,
    output_root: Path,
    *,
    expected_episode_id: str,
    expected_metadata_file: str,
    counterfactual_config: dict[str, Any],
) -> bool:
    requested_config = (
        _normalized_counterfactual_validation_config(counterfactual_config)
        if counterfactual_config
        else None
    )
    stored_config_value = metadata.get("counterfactual_config")
    stored_config = (
        _normalized_counterfactual_validation_config(stored_config_value)
        if isinstance(stored_config_value, dict)
        else None
    )
    if not isinstance(stored_config_value, dict) or stored_config is None:
        return False
    if (
        requested_config is not None
        and stored_config is not None
        and requested_config != stored_config
    ):
        return False
    validation_config = stored_config
    metadata_parent = Path(expected_metadata_file).parent
    expected_artifact_files = {
        "states_file": (metadata_parent / "states.jsonl").as_posix(),
        "outcomes_file": (metadata_parent / "outcomes.jsonl").as_posix(),
        "errors_file": (metadata_parent / "errors.jsonl").as_posix(),
    }
    if not _is_finite_json_tree(metadata) or not (
        str(metadata.get("schema")) == COUNTERFACTUAL_METADATA_SCHEMA
        and metadata.get("schema_version")
        == REPAIR_COLLECTION_ARTIFACT_VERSION
        and str(metadata.get("repair_time_label")) == REPAIR_TIME_LABEL
        and str(metadata.get("run_fingerprint")) == str(run_fingerprint)
        and str(metadata.get("episode_id")) == str(expected_episode_id)
        and str(metadata.get("metadata_file")) == str(expected_metadata_file)
        and metadata.get("complete") is True
        and str(metadata.get("status")) == "ok"
        and all(
            str(metadata.get(field)) == expected
            for field, expected in expected_artifact_files.items()
        )
    ):
        return False
    expected_counts = {
        "states_file": _strict_int(metadata.get("state_count")),
        "outcomes_file": _strict_int(metadata.get("outcome_count")),
        "errors_file": _strict_int(metadata.get("error_count")),
    }
    if any(value is None for value in expected_counts.values()):
        return False
    if any(value < 0 for value in expected_counts.values() if value is not None):
        return False
    rows_by_field: dict[str, list[dict[str, Any]]] = {}
    try:
        for field, expected_count in expected_counts.items():
            path = contained_file(output_root, metadata.get(field), field=field)
            rows = _read_jsonl(path)
            if len(rows) != expected_count:
                return False
            rows_by_field[field] = rows
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if expected_counts["errors_file"] != 0:
        return False
    return _counterfactual_artifact_rows_are_valid(
        rows_by_field,
        run_fingerprint=run_fingerprint,
        episode_id=expected_episode_id,
        counterfactual_config=validation_config,
    )


def _counterfactual_worker(job: dict[str, Any]) -> dict[str, Any]:
    manifest = job["manifest"]
    episode_id = str(manifest["episode_id"])
    output_root = Path(job["output_root"])
    episode_root = output_root / "counterfactual" / str(manifest["split"]) / episode_id
    metadata_path = episode_root / "metadata.json"
    relative_metadata = metadata_path.relative_to(output_root).as_posix()
    if job["resume"] and metadata_path.is_file():
        try:
            metadata = _read_json(metadata_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            return {
                **_artifact_fields(COUNTERFACTUAL_METADATA_SCHEMA),
                "run_fingerprint": job["run_fingerprint"],
                "episode_id": episode_id,
                "split": manifest["split"],
                "state_count": 0,
                "outcome_count": 0,
                "error_count": 1,
                "metadata_file": relative_metadata,
                "complete": False,
                "status": "error",
                "error": (
                    "existing counterfactual metadata is not a valid JSON "
                    f"object: {type(error).__name__}: {error}"
                ),
            }
        if _valid_counterfactual_resume_metadata(
            metadata,
            str(job["run_fingerprint"]),
            output_root,
            expected_episode_id=episode_id,
            expected_metadata_file=relative_metadata,
            counterfactual_config=job.get("counterfactual", {}),
        ):
            metadata = dict(metadata)
            metadata["status"] = "resumed"
            return metadata
        identity_compatible = (
            _is_finite_json_tree(metadata)
            and str(metadata.get("schema")) == COUNTERFACTUAL_METADATA_SCHEMA
            and metadata.get("schema_version")
            == REPAIR_COLLECTION_ARTIFACT_VERSION
            and str(metadata.get("repair_time_label")) == REPAIR_TIME_LABEL
            and str(metadata.get("run_fingerprint"))
            == str(job["run_fingerprint"])
        )
        if not identity_compatible or metadata.get("complete") is True:
            return {
                **_artifact_fields(COUNTERFACTUAL_METADATA_SCHEMA),
                "run_fingerprint": job["run_fingerprint"],
                "episode_id": episode_id,
                "split": manifest["split"],
                "state_count": 0,
                "outcome_count": 0,
                "error_count": 1,
                "metadata_file": relative_metadata,
                "complete": False,
                "status": "error",
                "error": (
                    "existing counterfactual metadata has an incompatible "
                    "artifact schema, timing semantics, or run identity"
                ),
            }
    metadata_path.unlink(missing_ok=True)
    try:
        trace_path = output_root / str(manifest["trace_file"])
        if _valid_episode_trace(
            trace_path,
            str(job["run_fingerprint"]),
            expected_episode_id=episode_id,
            expected_policy=str(manifest["policy"]),
            expected_solver_seed=int(manifest["solver_seed"]),
        ) is None:
            raise ValueError(
                "counterfactual source trace is legacy or timing-incompatible"
            )
        events = _read_jsonl(trace_path)
        decisions = _select_evenly(
            _decision_states(events), int(job["counterfactual"]["max_states_per_episode"])
        )
        states: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        row = job["row"]
        solver_seed = int(manifest["solver_seed"])
        horizons = sorted(int(value) for value in job["counterfactual"]["horizons"])
        maximum_horizon = max(horizons)
        for decision in decisions:
            state = decision["state"]
            fingerprint = state_fingerprint(state)
            state_id = f"{episode_id}__decision_{int(decision['decision_index']):04d}"
            actions = candidate_actions(
                state,
                int(job["counterfactual"]["max_seed_agents"]),
                list(job["counterfactual"]["heuristics"]),
                [int(value) for value in job["counterfactual"]["neighborhood_sizes"]],
            )
            states.append(
                {
                    **_artifact_fields(COUNTERFACTUAL_SCHEMA),
                    "run_fingerprint": job["run_fingerprint"],
                    "episode_id": episode_id,
                    "state_id": state_id,
                    "decision_index": int(decision["decision_index"]),
                    "state_fingerprint": fingerprint,
                    "prefix_actions": decision["prefix_actions"],
                    "candidate_count": len(actions),
                    "state": state,
                }
            )
            for candidate_index, candidate in enumerate(actions):
                for trial_index in range(int(job["counterfactual"]["trials"])):
                    branch_seed = _trial_seed(
                        episode_id, state_id, candidate, trial_index
                    )
                    try:
                        environment = _make_environment(
                            job["dataset_root"], row, job["environment"], "Adaptive"
                        )
                        replayed = _plain(environment.reset(seed=solver_seed))
                        for prefix_action in decision["prefix_actions"]:
                            if replayed["done"]:
                                raise RuntimeError("replay terminated before the decision state")
                            replayed = _plain(environment.step(prefix_action))["observation"]
                        replayed_fingerprint = state_fingerprint(replayed)
                        if replayed_fingerprint != fingerprint:
                            raise RuntimeError(
                                "replay fingerprint mismatch: "
                                f"expected {fingerprint}, got {replayed_fingerprint}"
                            )
                        action = dict(candidate)
                        action["random_seed"] = branch_seed
                        points = [
                            {
                                "step": 0,
                                "state": replayed,
                                "action": None,
                                "metrics": None,
                                "step_runtime": 0.0,
                                "low_level": dict(replayed["low_level"]),
                                "terminated": False,
                                "truncated": False,
                            }
                        ]
                        current = replayed
                        for step in range(1, maximum_horizon + 1):
                            if current["done"]:
                                break
                            requested = action if step == 1 else {"mode": "official"}
                            result = _plain(environment.step(requested))
                            if result["metrics"].get("step_applied") is False:
                                raise RuntimeError(
                                    "branch deadline expired before the repair step started"
                                )
                            current = result["observation"]
                            points.append(
                                {
                                    "step": step,
                                    "state": current,
                                    "action": requested,
                                    "metrics": result["metrics"],
                                    "step_runtime": _native_step_seconds(
                                        result["metrics"]
                                    ),
                                    "low_level": dict(current["low_level"]),
                                    "terminated": bool(result["terminated"]),
                                    "truncated": bool(result["truncated"]),
                                }
                            )
                        outcomes.append(
                            {
                                **_artifact_fields(COUNTERFACTUAL_SCHEMA),
                                "run_fingerprint": job["run_fingerprint"],
                                "episode_id": episode_id,
                                "state_id": state_id,
                                "state_fingerprint": fingerprint,
                                "candidate_index": candidate_index,
                                "candidate_action": action,
                                "trial_index": trial_index,
                                "trial_seed": branch_seed,
                                "step_runtime_label": REPAIR_TIME_LABEL,
                                "action_valid": bool(
                                    points[1]["metrics"]["action_valid"]
                                ),
                                "conflict_trajectory": [
                                    int(point["state"]["num_of_colliding_pairs"])
                                    for point in points
                                ],
                                "steps": [
                                    {
                                        key: value
                                        for key, value in point.items()
                                        if key != "state"
                                    }
                                    | {
                                        "state_fingerprint": state_fingerprint(
                                            point["state"]
                                        ),
                                        "conflicts": int(
                                            point["state"]["num_of_colliding_pairs"]
                                        ),
                                        "sum_of_costs": int(
                                            point["state"]["sum_of_costs"]
                                        ),
                                    }
                                    for point in points
                                ],
                                "horizon_outcomes": _horizon_outcomes(
                                    replayed, points, horizons
                                ),
                            }
                        )
                    except Exception as error:
                        errors.append(
                            {
                                **_artifact_fields(COUNTERFACTUAL_SCHEMA),
                                "run_fingerprint": job["run_fingerprint"],
                                "episode_id": episode_id,
                                "state_id": state_id,
                                "state_fingerprint": fingerprint,
                                "candidate_index": candidate_index,
                                "candidate_action": candidate,
                                "trial_index": trial_index,
                                "trial_seed": branch_seed,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
        states_path = episode_root / "states.jsonl"
        outcomes_path = episode_root / "outcomes.jsonl"
        errors_path = episode_root / "errors.jsonl"
        _write_jsonl(states_path, states)
        _write_jsonl(outcomes_path, outcomes)
        _write_jsonl(errors_path, errors)
        metadata = {
            **_artifact_fields(COUNTERFACTUAL_METADATA_SCHEMA),
            "run_fingerprint": job["run_fingerprint"],
            "episode_id": episode_id,
            "split": manifest["split"],
            "state_count": len(states),
            "outcome_count": len(outcomes),
            "error_count": len(errors),
            "states_file": states_path.relative_to(output_root).as_posix(),
            "outcomes_file": outcomes_path.relative_to(output_root).as_posix(),
            "errors_file": errors_path.relative_to(output_root).as_posix(),
            "metadata_file": relative_metadata,
            "counterfactual_config": (
                _normalized_counterfactual_validation_config(
                    dict(job["counterfactual"])
                )
            ),
            "complete": True,
            "status": "ok" if not errors else "error",
        }
        _write_json(metadata_path, metadata)
        return metadata
    except Exception as error:
        return {
            **_artifact_fields(COUNTERFACTUAL_METADATA_SCHEMA),
            "run_fingerprint": job["run_fingerprint"],
            "episode_id": episode_id,
            "split": manifest["split"],
            "state_count": 0,
            "outcome_count": 0,
            "error_count": 1,
            "metadata_file": relative_metadata,
            "complete": False,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def _job_label(job: dict[str, Any]) -> str:
    if "job_id" in job:
        return str(job["job_id"])
    if "manifest" in job:
        return str(job["manifest"]["episode_id"])
    row = job["row"]
    if "policy" in job:
        return _episode_id(row, int(job["solver_seed"]), str(job["policy"]))
    return f"{row['task_id']}__seed_{int(job['solver_seed']):04d}"


def _failed_job_result(
    job: dict[str, Any], status: str, message: str
) -> dict[str, Any]:
    row = job["row"]
    common = {
        **_artifact_fields(),
        "split": row["split"],
        "map_id": row["map_id"],
        "task_id": row["task_id"],
        "agent_count": int(row["agent_count"]),
        "solver_seed": int(job.get("solver_seed", job.get("manifest", {}).get("solver_seed", 0))),
        "status": status,
        "error": message,
    }
    for key in (
        "job_id",
        "state_id",
        "candidate_id",
        "evaluation_trial_index",
    ):
        if key in job:
            common[key] = job[key]
    if "manifest" in job:
        manifest = job["manifest"]
        output_root = Path(job["output_root"])
        metadata_path = (
            output_root
            / "counterfactual"
            / str(manifest["split"])
            / str(manifest["episode_id"])
            / "metadata.json"
        )
        return {
            **common,
            "run_fingerprint": job["run_fingerprint"],
            "episode_id": str(manifest["episode_id"]),
            "state_count": 0,
            "outcome_count": 0,
            "error_count": 1,
            "metadata_file": metadata_path.relative_to(output_root).as_posix(),
            "complete": False,
        }
    if "policy" in job:
        policy = str(job["policy"])
        return {
            **common,
            "episode_id": _episode_id(row, int(job["solver_seed"]), policy),
            "policy": policy,
            "trace_file": None,
            "summary": None,
        }
    return {
        **common,
        "layout_mode": row.get("layout_mode"),
        "task_variant": row.get("task_variant"),
    }


def _job_process_entry(
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    job: dict[str, Any],
    connection: Any,
) -> None:
    try:
        connection.send({"ok": True, "result": worker(job)})
    except BaseException as error:
        connection.send(
            {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    finally:
        connection.close()


def _stop_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join(timeout=0.1)
        return
    process.terminate()
    process.join(timeout=PROCESS_STOP_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=PROCESS_STOP_GRACE_SECONDS)


def _write_progress(
    path: Path,
    *,
    run_fingerprint: str,
    phase: str,
    status: str,
    total: int,
    completed: int,
    results: list[dict[str, Any]],
    active_labels: list[str],
    started_at: str,
) -> None:
    _write_json(
        path,
        {
            **_artifact_fields(),
            "run_fingerprint": run_fingerprint,
            "phase": phase,
            "status": status,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "total_jobs": total,
            "completed_jobs": completed,
            "pending_jobs": max(0, total - completed - len(active_labels)),
            "active_jobs": sorted(active_labels),
            "error_jobs": sum(
                row.get("status") in {"error", "timeout"} for row in results
            ),
            "timeout_jobs": sum(row.get("status") == "timeout" for row in results),
            "state_count": sum(int(row.get("state_count", 0)) for row in results),
            "outcome_count": sum(int(row.get("outcome_count", 0)) for row in results),
        },
    )


def _run_jobs(
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    jobs: list[dict[str, Any]],
    workers: int,
    *,
    phase: str = "jobs",
    output_root: Path | None = None,
    run_fingerprint: str = "untracked",
    timeout_seconds: float | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("job timeout must be positive")
    if not jobs:
        if output_root is not None:
            _write_progress(
                output_root / "collection_progress.json",
                run_fingerprint=run_fingerprint,
                phase=phase,
                status="complete",
                total=0,
                completed=0,
                results=[],
                active_labels=[],
                started_at=_utc_now(),
            )
        return []

    context = multiprocessing.get_context("spawn")
    pending = deque(enumerate(jobs))
    active: dict[int, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    started_at = _utc_now()
    progress_path = output_root / "collection_progress.json" if output_root else None
    interrupted_signal: int | None = None
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signum

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)

    def update_progress(status: str) -> None:
        if progress_path is None:
            return
        _write_progress(
            progress_path,
            run_fingerprint=run_fingerprint,
            phase=phase,
            status=status,
            total=len(jobs),
            completed=len(results),
            results=results,
            active_labels=[entry["label"] for entry in active.values()],
            started_at=started_at,
        )

    def record(result: dict[str, Any]) -> None:
        results.append(result)
        if on_result is not None:
            on_result(result)
        update_progress("running")

    update_progress("running")
    try:
        while pending or active:
            if interrupted_signal is not None:
                update_progress("interrupted")
                raise KeyboardInterrupt(f"received signal {interrupted_signal}")
            while pending and len(active) < min(workers, len(jobs)):
                index, job = pending.popleft()
                parent_connection, child_connection = context.Pipe(duplex=False)
                process = context.Process(
                    target=_job_process_entry,
                    args=(worker, job, child_connection),
                    name=f"lns2-{phase}-{index}",
                )
                process.start()
                child_connection.close()
                active[index] = {
                    "job": job,
                    "label": _job_label(job),
                    "process": process,
                    "connection": parent_connection,
                    "started": time.monotonic(),
                }
                update_progress("running")

            made_progress = False
            for index, entry in list(active.items()):
                process = entry["process"]
                connection = entry["connection"]
                payload: dict[str, Any] | None = None
                if connection.poll():
                    try:
                        payload = connection.recv()
                    except EOFError:
                        payload = {
                            "ok": False,
                            "error": f"worker exited with code {process.exitcode}",
                        }
                elif not process.is_alive():
                    if connection.poll():
                        try:
                            payload = connection.recv()
                        except EOFError:
                            payload = None
                    else:
                        payload = {
                            "ok": False,
                            "error": f"worker exited with code {process.exitcode}",
                        }
                    if payload is None:
                        payload = {
                            "ok": False,
                            "error": f"worker exited with code {process.exitcode}",
                        }
                elif (
                    timeout_seconds is not None
                    and time.monotonic() - float(entry["started"]) >= timeout_seconds
                ):
                    _stop_process(process)
                    payload = {
                        "ok": False,
                        "timeout": True,
                        "error": f"episode exceeded {timeout_seconds:.3f} seconds",
                    }
                if payload is None:
                    continue
                process.join(timeout=0.2)
                if process.is_alive():
                    _stop_process(process)
                connection.close()
                del active[index]
                if payload.get("ok"):
                    result = payload["result"]
                else:
                    result = _failed_job_result(
                        entry["job"],
                        "timeout" if payload.get("timeout") else "error",
                        str(payload.get("error", "worker failed")),
                    )
                record(result)
                made_progress = True
            if not made_progress and (pending or active):
                time.sleep(LOCK_POLL_SECONDS)
        update_progress("complete")
        return results
    except BaseException:
        update_progress("interrupted" if interrupted_signal is not None else "error")
        raise
    finally:
        for entry in active.values():
            _stop_process(entry["process"])
            entry["connection"].close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported repair collection config schema")
    seeds = [int(value) for value in config.get("solver_seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)) or any(value < 0 for value in seeds):
        raise ValueError("solver_seeds must be unique non-negative integers")
    policies = list(config.get("policies", []))
    if (
        not policies
        or len(policies) != len(set(policies))
        or any(value not in POLICY_DESTROY_STRATEGIES for value in policies)
    ):
        raise ValueError("collection config contains an unknown policy")
    environment = dict(config.get("environment", {}))
    required_environment = {
        "time_limit",
        "max_repair_iterations",
        "neighborhood_size",
        "replan_algorithm",
        "use_sipp",
    }
    if not required_environment.issubset(environment):
        raise ValueError("collection config omits environment settings")
    unlimited_time = bool(environment.get("unlimited_time", False))
    configured_time_limit = float(environment["time_limit"])
    if (
        configured_time_limit < 0
        or (configured_time_limit == 0 and not unlimited_time)
        or (configured_time_limit > 0 and unlimited_time)
        or int(environment["max_repair_iterations"]) < 0
        or (
            int(environment["max_repair_iterations"]) == 0
            and not unlimited_time
        )
        or int(environment["neighborhood_size"]) <= 0
    ):
        raise ValueError("collection environment limits must be positive")
    counterfactual = dict(config.get("counterfactual", {}))
    if counterfactual.get("source_policy") != "official_adaptive":
        raise ValueError("the first counterfactual source must be official_adaptive")
    horizons = [int(value) for value in counterfactual.get("horizons", [])]
    if (
        not horizons
        or len(horizons) != len(set(horizons))
        or any(value <= 0 for value in horizons)
    ):
        raise ValueError("counterfactual horizons must be unique and positive")
    if (
        int(counterfactual.get("max_states_per_episode", 0)) <= 0
        or int(counterfactual.get("trials", 0)) <= 0
    ):
        raise ValueError("counterfactual state and trial counts must be positive")
    minimum_conflicts = int(counterfactual.get("minimum_initial_conflicts", 1))
    maximum_conflicts = counterfactual.get("maximum_initial_conflicts")
    if minimum_conflicts <= 0 or (
        maximum_conflicts is not None
        and int(maximum_conflicts) < minimum_conflicts
    ):
        raise ValueError("counterfactual initial-conflict bounds are invalid")
    if not isinstance(counterfactual.get("require_source_success", False), bool):
        raise ValueError("counterfactual require_source_success must be boolean")
    maximum_agent_count = counterfactual.get("maximum_agent_count")
    if maximum_agent_count is not None and int(maximum_agent_count) <= 0:
        raise ValueError("counterfactual maximum_agent_count must be positive")
    episode_timeout = counterfactual.get("episode_wall_time_limit_seconds")
    if episode_timeout is not None and float(episode_timeout) <= 0:
        raise ValueError("counterfactual episode wall-time limit must be positive")
    candidate_actions(
        {
            "initialized": True,
            "initial_solution_complete": True,
            "feasible": False,
            "done": False,
            "iteration": 0,
            "rows": 1,
            "cols": 2,
            "sum_of_costs": 1,
            "num_of_colliding_pairs": 1,
            "low_level": {"expanded": 0, "generated": 0, "reopened": 0, "runs": 0},
            "obstacles": [0, 0],
            "conflict_edges": [[0, 1]],
            "agents": [
                {"id": 0, "delay": 0, "conflict_degree": 1, "path": [0]},
                {"id": 1, "delay": 0, "conflict_degree": 1, "path": [1]},
            ],
        },
        int(counterfactual["max_seed_agents"]),
        list(counterfactual["heuristics"]),
        [int(value) for value in counterfactual["neighborhood_sizes"]],
    )


def _counterfactual_source_eligible(
    row: dict[str, Any], counterfactual: dict[str, Any]
) -> bool:
    return _counterfactual_source_reason(row, counterfactual) == "eligible"


def _counterfactual_source_reason(
    row: dict[str, Any], counterfactual: dict[str, Any]
) -> str:
    summary = row.get("summary", {})
    if not bool(summary.get("repairable")):
        return "not_repairable"
    initial_conflicts = int(summary.get("initial_conflicts", 0))
    if initial_conflicts < int(counterfactual.get("minimum_initial_conflicts", 1)):
        return "below_minimum_initial_conflicts"
    maximum_conflicts = counterfactual.get("maximum_initial_conflicts")
    if maximum_conflicts is not None and initial_conflicts > int(maximum_conflicts):
        return "above_maximum_initial_conflicts"
    if bool(counterfactual.get("require_source_success", False)) and not bool(
        summary.get("success")
    ):
        return "source_policy_unsolved"
    maximum_agent_count = counterfactual.get("maximum_agent_count")
    if maximum_agent_count is not None and int(row.get("agent_count", 0)) > int(
        maximum_agent_count
    ):
        return "above_maximum_agent_count"
    return "eligible"


def recover_counterfactual_manifest(output: str | Path) -> dict[str, Any]:
    output_root = Path(output).resolve()
    if collection_status(output_root)["active"]:
        raise CollectionLockError(
            "cannot recover a manifest while its collection is active"
        )
    run_config = _read_json(output_root / "run_config.json")
    strict_v2 = (
        str(run_config.get("schema")) == REPAIR_COLLECTION_SCHEMA
        and run_config.get("schema_version")
        == REPAIR_COLLECTION_ARTIFACT_VERSION
        and str(run_config.get("repair_time_label")) == REPAIR_TIME_LABEL
    )
    legacy_identity = (
        run_config.get("schema") in (None, "", "lns2.repair_collection.v1")
        and run_config.get("schema_version") in (None, SCHEMA_VERSION)
        and run_config.get("repair_time_label") in (None, "")
    )
    if not strict_v2 and not legacy_identity:
        raise ValueError(
            "counterfactual recovery run_config has an incomplete or "
            "unsupported artifact identity"
        )
    counterfactual_config: dict[str, Any] = {}
    if strict_v2:
        configuration = run_config.get("configuration")
        counterfactual = (
            configuration.get("counterfactual")
            if isinstance(configuration, dict)
            else None
        )
        normalized_config = _normalized_counterfactual_validation_config(
            counterfactual
        )
        if normalized_config is None:
            raise ValueError(
                "counterfactual recovery run_config has missing or invalid "
                "counterfactual validation settings (horizons, trials, "
                "candidate generation)"
            )
        counterfactual_config = dict(counterfactual)
    run_fingerprint = str(run_config["run_fingerprint"])
    rows = []
    invalid = []
    pattern = "counterfactual/*/*/metadata.json"
    for metadata_path in sorted(output_root.glob(pattern)):
        try:
            metadata = _read_json(metadata_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            invalid.append(
                {
                    "metadata_file": metadata_path.relative_to(
                        output_root
                    ).as_posix(),
                    "reason": "invalid_metadata_json_object",
                }
            )
            continue
        reason = None
        if strict_v2 and (
            not _is_finite_json_tree(metadata)
            or str(metadata.get("schema")) != COUNTERFACTUAL_METADATA_SCHEMA
            or metadata.get("schema_version")
            != REPAIR_COLLECTION_ARTIFACT_VERSION
            or str(metadata.get("repair_time_label")) != REPAIR_TIME_LABEL
            or str(metadata.get("episode_id"))
            != metadata_path.parent.name
            or str(metadata.get("metadata_file"))
            != metadata_path.relative_to(output_root).as_posix()
            or any(
                str(metadata.get(field))
                != (metadata_path.parent / filename)
                .relative_to(output_root)
                .as_posix()
                for field, filename in (
                    ("states_file", "states.jsonl"),
                    ("outcomes_file", "outcomes.jsonl"),
                    ("errors_file", "errors.jsonl"),
                )
            )
            or _normalized_counterfactual_validation_config(
                metadata.get("counterfactual_config")
            )
            != _normalized_counterfactual_validation_config(
                counterfactual_config
            )
        ):
            reason = "artifact_schema_or_timing_mismatch"
        elif metadata.get("complete") is not True:
            reason = "incomplete_metadata"
        elif str(metadata.get("run_fingerprint")) != run_fingerprint:
            reason = "run_fingerprint_mismatch"
        else:
            rows_by_field: dict[str, list[dict[str, Any]]] = {}
            for key, count_key in (
                ("states_file", "state_count"),
                ("outcomes_file", "outcome_count"),
                ("errors_file", "error_count"),
            ):
                try:
                    path = contained_file(
                        output_root, metadata.get(key), field=key
                    )
                    artifact_rows = _read_jsonl(path)
                    expected_count = _strict_int(metadata.get(count_key))
                except (
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    reason = f"missing_{key}"
                    break
                if expected_count is None or expected_count < 0:
                    reason = f"invalid_{count_key}"
                    break
                if len(artifact_rows) != expected_count:
                    reason = f"count_mismatch_{key}"
                    break
                rows_by_field[key] = artifact_rows
            if (
                reason is None
                and strict_v2
                and not _counterfactual_artifact_rows_are_valid(
                    rows_by_field,
                    run_fingerprint=run_fingerprint,
                    episode_id=str(metadata.get("episode_id")),
                    counterfactual_config=counterfactual_config,
                )
            ):
                reason = "artifact_schema_or_timing_mismatch"
            if reason is None and strict_v2:
                expected_status = (
                    "ok" if not rows_by_field["errors_file"] else "error"
                )
                if str(metadata.get("status")) != expected_status:
                    reason = "artifact_schema_or_timing_mismatch"
        if reason is None:
            rows.append(metadata)
        else:
            invalid.append(
                {
                    "metadata_file": metadata_path.relative_to(output_root).as_posix(),
                    "reason": reason,
                }
            )
    rows.sort(key=lambda row: str(row["episode_id"]))
    _write_jsonl(output_root / "counterfactual_manifest.jsonl", rows)
    summary_identity = (
        _artifact_fields()
        if strict_v2
        else {
            key: run_config[key]
            for key in ("schema", "schema_version", "repair_time_label")
            if key in run_config
        }
    )
    if "schema_version" not in summary_identity:
        summary_identity["schema_version"] = SCHEMA_VERSION
    summary = _update_summary(
        output_root,
        artifact_identity=summary_identity,
    )
    expected = []
    source_path = output_root / "counterfactual_source_manifest.jsonl"
    if source_path.is_file():
        expected = [
            str(row["episode_id"])
            for row in _read_jsonl(source_path)
            if bool(row.get("eligible"))
        ]
    recovered_ids = {str(row["episode_id"]) for row in rows}
    return {
        "recovered_count": len(rows),
        "expected_eligible_count": len(expected),
        "missing_episode_ids": sorted(set(expected) - recovered_ids),
        "invalid_metadata": invalid,
        "summary": summary,
    }


def _dataset_fingerprint(dataset_root: Path) -> str:
    summary_path = dataset_root / "dataset_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing dataset summary: {summary_path}")
    resolved_root = dataset_root.resolve()
    paths = {summary_path.resolve()}
    for manifest_path in sorted(dataset_root.glob("*/manifest.jsonl")):
        paths.add(manifest_path.resolve())
        for row in _read_jsonl(manifest_path):
            for key in (
                "map_file",
                "scenario_file",
                "map_metadata_file",
                "task_file",
            ):
                path = (manifest_path.parent / str(row[key])).resolve()
                try:
                    path.relative_to(resolved_root)
                except ValueError as error:
                    raise ValueError(
                        f"dataset manifest path escapes its root: {path}"
                    ) from error
                if not path.is_file():
                    raise ValueError(f"dataset input is missing: {path}")
                paths.add(path)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(resolved_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_dataset_rows(dataset_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        manifest_path = dataset_root / split / "manifest.jsonl"
        if not manifest_path.is_file():
            raise ValueError(f"missing dataset split manifest: {manifest_path}")
        for row in _read_jsonl(manifest_path):
            if str(row.get("split")) != split:
                raise ValueError(f"manifest row crosses split boundary: {manifest_path}")
            rows.append(row)
    return rows


def _effective_config(
    config: dict[str, Any],
    max_states: int | None,
    max_seed_agents: int | None,
    neighborhood_sizes: list[int] | None,
    trials: int | None,
    horizons: list[int] | None,
    episode_time_limit: float | None,
) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    counterfactual = value["counterfactual"]
    if max_states is not None:
        counterfactual["max_states_per_episode"] = max_states
    if max_seed_agents is not None:
        counterfactual["max_seed_agents"] = max_seed_agents
    if neighborhood_sizes is not None:
        counterfactual["neighborhood_sizes"] = neighborhood_sizes
    if trials is not None:
        counterfactual["trials"] = trials
    if horizons is not None:
        counterfactual["horizons"] = horizons
    if episode_time_limit is not None:
        counterfactual["episode_wall_time_limit_seconds"] = episode_time_limit
    _validate_config(value)
    return value


def _run_metadata(
    dataset_root: Path,
    config: dict[str, Any],
    splits: list[str],
    task_ids: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    dataset_hash = _dataset_fingerprint(dataset_root)
    configuration_hash = _fingerprint(config)
    collection_identity = _collection_identity()
    fingerprint_payload: dict[str, Any] = {
        "dataset_fingerprint": dataset_hash,
        "configuration_fingerprint": configuration_hash,
        "splits": splits,
        "collection_identity": collection_identity,
    }
    if task_ids is not None:
        fingerprint_payload["task_ids"] = task_ids
    run_fingerprint = _fingerprint(fingerprint_payload)
    run_config = {
        **_artifact_fields(),
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_hash,
        "configuration": config,
        "configuration_fingerprint": configuration_hash,
        "splits": splits,
        "input_config_schema_version": SCHEMA_VERSION,
        "collection_identity": collection_identity,
        "repair_time_semantics": _repair_time_semantics(),
        "run_fingerprint": run_fingerprint,
        "collection_control": {
            "workspace_exclusive_lock": True,
            "incremental_manifests": True,
            "runtime_scope": "collector-exclusive; external system load is not controlled",
        },
    }
    if task_ids is not None:
        run_config["task_ids"] = task_ids
    return run_fingerprint, run_config


def _prepare_run(
    dataset_root: Path,
    output_root: Path,
    config: dict[str, Any],
    splits: list[str],
    resume: bool,
    task_ids: list[str] | None = None,
    metadata: tuple[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    run_fingerprint, run_config = metadata or _run_metadata(
        dataset_root, config, splits, task_ids
    )
    path = output_root / "run_config.json"
    if path.is_file():
        existing = _read_json(path)
        if (
            str(existing.get("schema")) != REPAIR_COLLECTION_SCHEMA
            or existing.get("schema_version")
            != REPAIR_COLLECTION_ARTIFACT_VERSION
            or str(existing.get("repair_time_label")) != REPAIR_TIME_LABEL
            or existing.get("collection_identity")
            != run_config["collection_identity"]
            or existing.get("repair_time_semantics")
            != run_config["repair_time_semantics"]
        ):
            raise ValueError(
                "output contains an incompatible repair collection "
                "artifact schema, timing semantics, or producer"
            )
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ValueError("output contains a different dataset or collection config")
        if not resume:
            raise ValueError("output already exists; pass --resume to continue it")
    else:
        existing_entries = [
            entry
            for entry in output_root.iterdir()
            if entry.name != ".collection.lock"
        ] if output_root.is_dir() else []
        if existing_entries:
            raise ValueError(
                "output is non-empty but has no run_config.json; refusing to "
                "adopt or overwrite existing artifacts"
            )
        _write_json(path, run_config)
    return run_fingerprint, run_config


def _select_task_rows(
    rows: list[dict[str, Any]], task_ids: list[str] | None
) -> list[dict[str, Any]]:
    if task_ids is None:
        return rows
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("task_ids must contain unique task identifiers")
    row_index = {str(row["task_id"]): row for row in rows}
    missing = sorted(set(task_ids) - set(row_index))
    if missing:
        raise ValueError(f"requested task_ids are absent from the dataset: {missing}")
    selected = set(task_ids)
    return [row for row in rows if str(row["task_id"]) in selected]


def _collection_estimate(
    output_root: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    max_episodes: int | None = None,
) -> dict[str, Any]:
    counterfactual = config["counterfactual"]
    solver_seeds = [int(value) for value in config["solver_seeds"]]
    available_source_pairs = len(rows) * len(solver_seeds)
    source_pairs = (
        min(available_source_pairs, max_episodes)
        if max_episodes is not None
        else available_source_pairs
    )
    branches_per_source_upper = (
        int(counterfactual["max_states_per_episode"])
        * int(counterfactual["max_seed_agents"])
        * len(counterfactual["heuristics"])
        * len(counterfactual["neighborhood_sizes"])
        * int(counterfactual["trials"])
    )
    result: dict[str, Any] = {
        "task_count": len(rows),
        "solver_seed_count": len(solver_seeds),
        "available_source_pairs": available_source_pairs,
        "selected_source_pair_upper_bound": source_pairs,
        "qualification_jobs": source_pairs,
        "baseline_jobs": source_pairs * len(config["policies"]),
        "counterfactual_source_upper_bound": source_pairs,
        "states_per_source_upper_bound": int(
            counterfactual["max_states_per_episode"]
        ),
        "branches_per_source_upper_bound": branches_per_source_upper,
        "environment_reset_upper_bound": source_pairs * branches_per_source_upper,
        "exact_from_baseline": False,
    }
    manifest_path = output_root / "collection_manifest.jsonl"
    if not manifest_path.is_file():
        return result
    task_ids = {str(row["task_id"]) for row in rows}
    source_policy = str(counterfactual["source_policy"])
    eligible_splits = set(counterfactual["eligible_splits"])
    baseline = [
        row
        for row in _read_jsonl(manifest_path)
        if str(row.get("task_id")) in task_ids
        and row.get("policy") == source_policy
        and row.get("split") in eligible_splits
        and row.get("status") not in {"error", "timeout"}
        and _counterfactual_source_eligible(row, counterfactual)
    ]
    exact_states = 0
    exact_branches = 0
    reset_cpu_lower_bound = 0.0
    for manifest in baseline:
        trace_path = output_root / str(manifest["trace_file"])
        if not trace_path.is_file():
            return result
        decisions = _select_evenly(
            _decision_states(_read_jsonl(trace_path)),
            int(counterfactual["max_states_per_episode"]),
        )
        branch_count = 0
        for decision in decisions:
            action_count = len(
                candidate_actions(
                    decision["state"],
                    int(counterfactual["max_seed_agents"]),
                    list(counterfactual["heuristics"]),
                    [int(value) for value in counterfactual["neighborhood_sizes"]],
                )
            )
            branch_count += action_count * int(counterfactual["trials"])
        exact_states += len(decisions)
        exact_branches += branch_count
        reset_cpu_lower_bound += float(
            manifest.get("summary", {}).get("initial_runtime", 0.0)
        ) * branch_count
    result.update(
        {
            "exact_from_baseline": True,
            "eligible_counterfactual_sources": len(baseline),
            "selected_state_count": exact_states,
            "environment_reset_count": exact_branches,
            "estimated_reset_cpu_seconds_lower_bound": reset_cpu_lower_bound,
        }
    )
    return result


def _manifest_accumulator(
    path: Path,
    initial_rows: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], Any],
) -> tuple[Callable[[dict[str, Any]], None], Callable[[], list[dict[str, Any]]]]:
    index = {key(row): row for row in initial_rows}

    def rows() -> list[dict[str, Any]]:
        return sorted(index.values(), key=lambda row: str(key(row)))

    def record(row: dict[str, Any]) -> None:
        index[key(row)] = row
        _write_jsonl(path, rows())

    return record, rows


def _qualification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "run_count": len(rows),
        "error_count": sum(row["status"] in {"error", "timeout"} for row in rows),
        "repairable_count": sum(bool(row.get("repairable")) for row in rows),
        "by_split": {},
    }
    for split in sorted({str(row["split"]) for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        valid = [row for row in selected if row["status"] == "ok"]
        repairable = sum(bool(row["repairable"]) for row in valid)
        result["by_split"][split] = {
            "run_count": len(selected),
            "valid_count": len(valid),
            "repairable_count": repairable,
            "repairable_rate": repairable / len(valid) if valid else 0.0,
            "mean_initial_conflicts": (
                sum(int(row["initial_conflicts"]) for row in valid) / len(valid)
                if valid
                else 0.0
            ),
        }
    return result


def _update_summary(
    output_root: Path,
    *,
    artifact_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = dict(
        artifact_identity if artifact_identity is not None else _artifact_fields()
    )
    qualification_path = output_root / "qualification_manifest.jsonl"
    if qualification_path.is_file():
        summary["qualification"] = _qualification_summary(
            _read_jsonl(qualification_path)
        )
    baseline_path = output_root / "collection_manifest.jsonl"
    if baseline_path.is_file():
        rows = _read_jsonl(baseline_path)
        summary["baseline"] = {
            "episode_count": len(rows),
            "error_count": sum(
                row["status"] in {"error", "timeout"} for row in rows
            ),
            "success_count": sum(
                bool(row.get("summary", {}).get("success"))
                for row in rows
                if row.get("summary")
            ),
            "repairable_count": sum(
                bool(row.get("summary", {}).get("repairable"))
                for row in rows
                if row.get("summary")
            ),
        }
    counterfactual_path = output_root / "counterfactual_manifest.jsonl"
    if counterfactual_path.is_file():
        rows = _read_jsonl(counterfactual_path)
        summary["counterfactual"] = {
            "episode_count": len(rows),
            "state_count": sum(int(row.get("state_count", 0)) for row in rows),
            "outcome_count": sum(int(row.get("outcome_count", 0)) for row in rows),
            "error_count": sum(int(row.get("error_count", 0)) for row in rows),
        }
    source_path = output_root / "counterfactual_source_manifest.jsonl"
    if source_path.is_file():
        rows = _read_jsonl(source_path)
        summary["counterfactual_sources"] = {
            "episode_count": len(rows),
            "eligible_count": sum(bool(row.get("eligible")) for row in rows),
            "by_reason": dict(
                sorted(collections.Counter(str(row["reason"]) for row in rows).items())
            ),
        }
    _write_json(output_root / "summary.json", summary)
    return summary


def run_collection(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    phase: str = "all",
    splits: list[str] | None = None,
    workers: int | None = None,
    resume: bool = False,
    max_episodes: int | None = None,
    max_states: int | None = None,
    max_seed_agents: int | None = None,
    neighborhood_sizes: list[int] | None = None,
    trials: int | None = None,
    horizons: list[int] | None = None,
    task_ids: list[str] | None = None,
    episode_time_limit: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if phase not in {"qualify", "baseline", "counterfactual", "all"}:
        raise ValueError("phase must be qualify, baseline, counterfactual, or all")
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    config = _effective_config(
        _read_json(Path(config_path)),
        max_states,
        max_seed_agents,
        neighborhood_sizes,
        trials,
        horizons,
        episode_time_limit,
    )
    dataset_summary = _read_json(dataset_root / "dataset_summary.json")
    available_splits = list(dataset_summary["splits"])
    requested_splits = splits or available_splits
    if not requested_splits or any(value not in available_splits for value in requested_splits):
        raise ValueError("requested split is not present in the dataset")
    worker_count = int(workers if workers is not None else config.get("workers", 4))
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if max_states is not None and max_states <= 0:
        raise ValueError("max_states must be positive")
    normalized_task_ids = sorted(task_ids) if task_ids is not None else None
    all_rows = _load_dataset_rows(dataset_root, requested_splits)
    rows = _select_task_rows(all_rows, normalized_task_ids)
    metadata = _run_metadata(
        dataset_root, config, requested_splits, normalized_task_ids
    )
    run_fingerprint = metadata[0]
    if dry_run:
        return {
            **_artifact_fields(),
            "dry_run": True,
            "run_fingerprint": run_fingerprint,
            "phase": phase,
            "workers": worker_count,
            "estimate": _collection_estimate(
                output_root, rows, config, max_episodes=max_episodes
            ),
        }

    environment = dict(config["environment"])
    solver_seeds = [int(value) for value in config["solver_seeds"]]
    run_config_path = output_root / "run_config.json"
    if (
        output_root.is_dir()
        and not run_config_path.is_file()
        and any(output_root.iterdir())
    ):
        raise ValueError(
            "output is non-empty but has no run_config.json; refusing to "
            "adopt or overwrite existing artifacts"
        )
    with _CollectionRunLock(output_root, run_fingerprint, phase):
        _prepare_run(
            dataset_root,
            output_root,
            config,
            requested_splits,
            resume,
            normalized_task_ids,
            metadata,
        )

        if phase in {"qualify", "all"}:
            qualification_path = output_root / "qualification_manifest.jsonl"
            existing_qualification = (
                _read_jsonl(qualification_path)
                if resume and qualification_path.is_file()
                else []
            )
            existing_index = {
                (str(row["task_id"]), int(row["solver_seed"])): row
                for row in existing_qualification
                if row.get("status") == "ok"
            }
            jobs = [
                {
                    "dataset_root": str(dataset_root),
                    "row": row,
                    "solver_seed": seed,
                    "environment": environment,
                }
                for row in rows
                for seed in solver_seeds
                if (str(row["task_id"]), seed) not in existing_index
            ]
            record, qualification_rows = _manifest_accumulator(
                qualification_path,
                existing_qualification,
                lambda row: (str(row["task_id"]), int(row["solver_seed"])),
            )
            _run_jobs(
                _qualification_worker,
                jobs,
                worker_count,
                phase="qualify",
                output_root=output_root,
                run_fingerprint=run_fingerprint,
                on_result=record,
            )
            _write_jsonl(qualification_path, qualification_rows())

        qualification_path = output_root / "qualification_manifest.jsonl"
        qualification = (
            _read_jsonl(qualification_path) if qualification_path.is_file() else []
        )
        qualification_index = {
            (str(row["task_id"]), int(row["solver_seed"])): row
            for row in qualification
            if row["status"] == "ok"
        }
        pairs = [(row, seed) for row in rows for seed in solver_seeds]
        if max_episodes is not None:
            pairs.sort(
                key=lambda item: (
                    not bool(
                        qualification_index.get(
                            (str(item[0]["task_id"]), int(item[1])), {}
                        ).get("repairable")
                    ),
                    str(item[0]["split"]),
                    str(item[0]["task_id"]),
                    int(item[1]),
                )
            )
            pairs = pairs[:max_episodes]

        if phase in {"baseline", "all"}:
            manifest_path = output_root / "collection_manifest.jsonl"
            existing_baseline = (
                _read_jsonl(manifest_path)
                if resume and manifest_path.is_file()
                else []
            )
            jobs = [
                {
                    "dataset_root": str(dataset_root),
                    "output_root": str(output_root),
                    "row": row,
                    "solver_seed": seed,
                    "policy": policy,
                    "environment": environment,
                    "run_fingerprint": run_fingerprint,
                    "resume": resume,
                }
                for row, seed in pairs
                for policy in config["policies"]
            ]
            record, baseline_rows = _manifest_accumulator(
                manifest_path,
                existing_baseline,
                lambda row: str(row["episode_id"]),
            )
            _run_jobs(
                _baseline_worker,
                jobs,
                worker_count,
                phase="baseline",
                output_root=output_root,
                run_fingerprint=run_fingerprint,
                on_result=record,
            )
            _write_jsonl(manifest_path, baseline_rows())

        if phase in {"counterfactual", "all"}:
            manifest_path = output_root / "collection_manifest.jsonl"
            if not manifest_path.is_file():
                raise ValueError(
                    "counterfactual phase requires a baseline collection manifest"
                )
            baseline = _read_jsonl(manifest_path)
            row_index = {str(row["task_id"]): row for row in rows}
            eligible_splits = set(config["counterfactual"]["eligible_splits"])
            source_policy = str(config["counterfactual"]["source_policy"])
            considered = [
                row
                for row in baseline
                if row["policy"] == source_policy
                and row["split"] in eligible_splits
                and row["status"] not in {"error", "timeout"}
                and str(row["task_id"]) in row_index
            ]
            source_selection = []
            for row in considered:
                reason = _counterfactual_source_reason(row, config["counterfactual"])
                source_selection.append(
                    {
                        **_artifact_fields(),
                        "episode_id": str(row["episode_id"]),
                        "split": str(row["split"]),
                        "map_id": str(row["map_id"]),
                        "task_id": str(row["task_id"]),
                        "solver_seed": int(row["solver_seed"]),
                        "initial_conflicts": int(
                            row.get("summary", {}).get("initial_conflicts", 0)
                        ),
                        "source_success": bool(
                            row.get("summary", {}).get("success")
                        ),
                        "eligible": reason == "eligible",
                        "reason": reason,
                    }
                )
            _write_jsonl(
                output_root / "counterfactual_source_manifest.jsonl",
                sorted(source_selection, key=lambda row: str(row["episode_id"])),
            )
            selected = [
                row
                for row in considered
                if _counterfactual_source_eligible(row, config["counterfactual"])
            ]
            jobs = [
                {
                    "dataset_root": str(dataset_root),
                    "output_root": str(output_root),
                    "row": row_index[str(manifest["task_id"])],
                    "manifest": manifest,
                    "environment": environment,
                    "counterfactual": config["counterfactual"],
                    "run_fingerprint": run_fingerprint,
                    "resume": resume,
                }
                for manifest in selected
            ]
            counterfactual_path = output_root / "counterfactual_manifest.jsonl"
            existing_counterfactual = (
                _read_jsonl(counterfactual_path)
                if resume and counterfactual_path.is_file()
                else []
            )
            record, counterfactual_rows = _manifest_accumulator(
                counterfactual_path,
                existing_counterfactual,
                lambda row: str(row["episode_id"]),
            )
            _run_jobs(
                _counterfactual_worker,
                jobs,
                worker_count,
                phase="counterfactual",
                output_root=output_root,
                run_fingerprint=run_fingerprint,
                timeout_seconds=config["counterfactual"].get(
                    "episode_wall_time_limit_seconds"
                ),
                on_result=record,
            )
            _write_jsonl(counterfactual_path, counterfactual_rows())

        return _update_summary(output_root)
