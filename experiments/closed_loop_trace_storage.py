from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, TextIO

from experiments.repair_collection import (
    STATE_FINGERPRINT_KEYS,
    _fingerprint,
    _plain,
    state_fingerprint,
)


TRACE_FORMAT_FULL_V1 = "full-v1"
TRACE_FORMAT_DELTA_GZIP_V2 = "delta-gzip-v2"
TRACE_FORMATS = (TRACE_FORMAT_DELTA_GZIP_V2, TRACE_FORMAT_FULL_V1)
EPISODE_SCHEMA_V1 = "lns2.closed_loop_episode.v1"
EPISODE_SCHEMA_V2 = "lns2.closed_loop_episode.v2"
TRACE_SCHEMA_VERSION_V1 = 1
TRACE_SCHEMA_VERSION_V2 = 2
STATE_DELTA_VERSION = 1
GZIP_COMPRESSLEVEL = 6


class TraceStorageError(ValueError):
    pass


def storage_fingerprint(trace_format: str) -> str:
    if trace_format not in TRACE_FORMATS:
        raise ValueError(f"unsupported trace format: {trace_format}")
    if trace_format == TRACE_FORMAT_FULL_V1:
        payload = {
            "trace_format": TRACE_FORMAT_FULL_V1,
            "episode_schema": EPISODE_SCHEMA_V1,
            "schema_version": TRACE_SCHEMA_VERSION_V1,
        }
    else:
        payload = {
            "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
            "episode_schema": EPISODE_SCHEMA_V2,
            "schema_version": TRACE_SCHEMA_VERSION_V2,
            "state_delta_version": STATE_DELTA_VERSION,
            "compression": "gzip",
            "compression_level": GZIP_COMPRESSLEVEL,
            "gzip_mtime": 0,
        }
    return _fingerprint(payload)


def trace_suffix(trace_format: str) -> str:
    if trace_format == TRACE_FORMAT_FULL_V1:
        return ".jsonl"
    if trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
        return ".jsonl.gz"
    raise ValueError(f"unsupported trace format: {trace_format}")


def partial_trace_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def state_core(state: dict[str, Any]) -> dict[str, Any]:
    try:
        return {key: _plain(state[key]) for key in STATE_FINGERPRINT_KEYS}
    except KeyError as error:
        raise TraceStorageError(f"state is missing fingerprint field: {error.args[0]}") from error


def _state_core_view(state: dict[str, Any]) -> dict[str, Any]:
    try:
        return {key: state[key] for key in STATE_FINGERPRINT_KEYS}
    except KeyError as error:
        raise TraceStorageError(f"state is missing fingerprint field: {error.args[0]}") from error


def state_extras(state: dict[str, Any]) -> dict[str, Any]:
    core = set(STATE_FINGERPRINT_KEYS)
    return {str(key): _plain(value) for key, value in state.items() if key not in core}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_deterministic_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=GZIP_COMPRESSLEVEL,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                compressed.write(payload)
        if path.is_file():
            temporary.unlink()
            return
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if path.is_file():
                    temporary.unlink()
                    return
                if attempt == 7:
                    raise
                time.sleep(min(0.025 * (2**attempt), 0.5))
    finally:
        temporary.unlink(missing_ok=True)


def write_state_blob(output_root: Path, state: dict[str, Any]) -> tuple[str, Path]:
    core = state_core(state)
    fingerprint = state_fingerprint(core)
    relative = Path("state_blobs") / f"{fingerprint}.json.gz"
    path = output_root / relative
    if path.is_file():
        existing = read_state_blob(path)
        if state_fingerprint(existing) != fingerprint or existing != core:
            raise TraceStorageError(f"state blob collision or corruption: {path}")
        return relative.as_posix(), path
    _write_deterministic_gzip(path, _canonical_json_bytes(core) + b"\n")
    existing = read_state_blob(path)
    if state_fingerprint(existing) != fingerprint or existing != core:
        raise TraceStorageError(f"state blob verification failed: {path}")
    return relative.as_posix(), path


def read_state_blob(path: Path) -> dict[str, Any]:
    value: Any = None
    for attempt in range(8):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                value = json.load(stream)
            break
        except PermissionError as error:
            if attempt == 7:
                raise TraceStorageError(f"cannot read state blob {path}: {error}") from error
            time.sleep(min(0.025 * (2**attempt), 0.5))
        except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as error:
            raise TraceStorageError(f"cannot read state blob {path}: {error}") from error
    if not isinstance(value, dict):
        raise TraceStorageError(f"state blob is not an object: {path}")
    return state_core(value)


def _is_sorted_unique_edges(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    try:
        edges = [tuple(map(int, edge)) for edge in value]
    except (TypeError, ValueError):
        return False
    return len(edges) == len(set(edges)) and edges == sorted(edges)


def encode_state_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    left = _state_core_view(before)
    right = _state_core_view(after)
    excluded = {"conflict_edges", "agents"}
    top_set = {
        key: copy.deepcopy(right[key])
        for key in STATE_FINGERPRINT_KEYS
        if key not in excluded and left[key] != right[key]
    }

    left_edges = left["conflict_edges"]
    right_edges = right["conflict_edges"]
    if _is_sorted_unique_edges(left_edges) and _is_sorted_unique_edges(right_edges):
        old = {tuple(map(int, edge)) for edge in left_edges}
        new = {tuple(map(int, edge)) for edge in right_edges}
        edge_change: dict[str, Any] = {
            "mode": "delta",
            "remove": [list(edge) for edge in sorted(old - new)],
            "add": [list(edge) for edge in sorted(new - old)],
        }
    else:
        edge_change = {"mode": "replace", "value": copy.deepcopy(right_edges)}

    left_agents = left["agents"]
    right_agents = right["agents"]
    can_patch = (
        isinstance(left_agents, list)
        and isinstance(right_agents, list)
        and len(left_agents) == len(right_agents)
        and all(isinstance(agent, dict) and "id" in agent for agent in left_agents)
        and all(isinstance(agent, dict) and "id" in agent for agent in right_agents)
        and [agent["id"] for agent in left_agents]
        == [agent["id"] for agent in right_agents]
    )
    if can_patch:
        patches = []
        for old, new in zip(left_agents, right_agents):
            changed = {
                key: copy.deepcopy(value)
                for key, value in new.items()
                if key != "id" and (key not in old or old[key] != value)
            }
            removed = sorted(key for key in old if key not in new and key != "id")
            if changed or removed:
                patches.append(
                    {
                        "id": int(new["id"]),
                        "set": changed,
                        "remove": removed,
                    }
                )
        agent_change: dict[str, Any] = {"mode": "patch", "patches": patches}
    else:
        agent_change = {"mode": "replace", "value": copy.deepcopy(right_agents)}

    return {
        "version": STATE_DELTA_VERSION,
        "top_set": top_set,
        "conflict_edges": edge_change,
        "agents": agent_change,
    }


def apply_state_delta(
    before: dict[str, Any], delta: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(delta, dict) or int(delta.get("version", -1)) != STATE_DELTA_VERSION:
        raise TraceStorageError("unsupported or missing state delta version")
    result = dict(_state_core_view(before))
    top_set = delta.get("top_set")
    if not isinstance(top_set, dict):
        raise TraceStorageError("state delta top_set must be an object")
    forbidden = {"conflict_edges", "agents"}
    if any(key not in STATE_FINGERPRINT_KEYS or key in forbidden for key in top_set):
        raise TraceStorageError("state delta contains an invalid top-level field")
    for key, value in top_set.items():
        result[key] = copy.deepcopy(value)

    edge_change = delta.get("conflict_edges")
    if not isinstance(edge_change, dict):
        raise TraceStorageError("state delta is missing conflict edge changes")
    edge_mode = str(edge_change.get("mode"))
    if edge_mode == "replace":
        value = edge_change.get("value")
        if not isinstance(value, list):
            raise TraceStorageError("replacement conflict edges must be a list")
        result["conflict_edges"] = value
    elif edge_mode == "delta":
        if not _is_sorted_unique_edges(result["conflict_edges"]):
            raise TraceStorageError("cannot apply edge delta to non-canonical edges")
        current = {tuple(map(int, edge)) for edge in result["conflict_edges"]}
        try:
            removed = [tuple(map(int, edge)) for edge in edge_change.get("remove", [])]
            added = [tuple(map(int, edge)) for edge in edge_change.get("add", [])]
        except (TypeError, ValueError) as error:
            raise TraceStorageError("invalid conflict edge delta") from error
        if len(removed) != len(set(removed)) or len(added) != len(set(added)):
            raise TraceStorageError("conflict edge delta contains duplicates")
        if not set(removed).issubset(current):
            raise TraceStorageError("conflict edge delta removes a missing edge")
        if set(added) & (current - set(removed)):
            raise TraceStorageError("conflict edge delta adds an existing edge")
        current.difference_update(removed)
        current.update(added)
        result["conflict_edges"] = [list(edge) for edge in sorted(current)]
    else:
        raise TraceStorageError(f"unsupported conflict edge mode: {edge_mode}")

    agent_change = delta.get("agents")
    if not isinstance(agent_change, dict):
        raise TraceStorageError("state delta is missing agent changes")
    agent_mode = str(agent_change.get("mode"))
    if agent_mode == "replace":
        value = agent_change.get("value")
        if not isinstance(value, list):
            raise TraceStorageError("replacement agents must be a list")
        result["agents"] = value
    elif agent_mode == "patch":
        source_agents = result["agents"]
        agents = list(source_agents) if isinstance(source_agents, list) else source_agents
        if not isinstance(agents, list) or not all(
            isinstance(agent, dict) and "id" in agent for agent in agents
        ):
            raise TraceStorageError("cannot patch invalid agents")
        by_id = {int(agent["id"]): index for index, agent in enumerate(agents)}
        if len(by_id) != len(agents):
            raise TraceStorageError("state contains duplicate agent ids")
        seen: set[int] = set()
        patches = agent_change.get("patches")
        if not isinstance(patches, list):
            raise TraceStorageError("agent patches must be a list")
        for patch in patches:
            if not isinstance(patch, dict):
                raise TraceStorageError("agent patch must be an object")
            identifier = int(patch.get("id", -1))
            if identifier in seen or identifier not in by_id:
                raise TraceStorageError("agent patch has a duplicate or unknown id")
            seen.add(identifier)
            changed = patch.get("set")
            removed = patch.get("remove", [])
            if not isinstance(changed, dict) or not isinstance(removed, list):
                raise TraceStorageError("agent patch set/remove fields are invalid")
            if "id" in changed or "id" in removed:
                raise TraceStorageError("agent patches cannot change ids")
            agent = dict(agents[by_id[identifier]])
            for key in removed:
                if key not in agent:
                    raise TraceStorageError("agent patch removes a missing field")
                del agent[key]
            for key, value in changed.items():
                agent[str(key)] = value
            agents[by_id[identifier]] = agent
        result["agents"] = agents
    else:
        raise TraceStorageError(f"unsupported agent mode: {agent_mode}")
    return result


def encode_extras_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    left = state_extras(before)
    right = state_extras(after)
    return {
        "set": {
            key: copy.deepcopy(value)
            for key, value in right.items()
            if key not in left or left[key] != value
        },
        "remove": sorted(key for key in left if key not in right),
    }


def apply_extras_delta(
    before: dict[str, Any], delta: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(delta, dict):
        raise TraceStorageError("state extras delta must be an object")
    changed = delta.get("set")
    removed = delta.get("remove")
    if not isinstance(changed, dict) or not isinstance(removed, list):
        raise TraceStorageError("state extras set/remove fields are invalid")
    if any(not isinstance(key, str) for key in changed) or any(
        not isinstance(key, str) for key in removed
    ):
        raise TraceStorageError("state extras keys must be strings")
    if len(removed) != len(set(removed)):
        raise TraceStorageError("state extras delta contains duplicate removals")
    if set(changed) & set(removed):
        raise TraceStorageError("state extras delta both sets and removes a field")
    core = set(STATE_FINGERPRINT_KEYS)
    if set(changed) & core or set(removed) & core:
        raise TraceStorageError("state extras delta changes a fingerprint field")
    result = state_extras(before)
    if not set(removed).issubset(result):
        raise TraceStorageError("state extras delta removes a missing field")
    for key in removed:
        del result[key]
    for key, value in changed.items():
        result[key] = copy.deepcopy(value)
    return result


def encode_initial_event(
    event: dict[str, Any], state: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], str]:
    reference, _ = write_state_blob(output_root, state)
    encoded = {key: copy.deepcopy(value) for key, value in event.items() if key != "state"}
    encoded.update(
        {
            "schema": EPISODE_SCHEMA_V2,
            "schema_version": TRACE_SCHEMA_VERSION_V2,
            "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
            "storage_fingerprint": storage_fingerprint(TRACE_FORMAT_DELTA_GZIP_V2),
            "state_blob": reference,
            "state_extras": state_extras(state),
        }
    )
    return encoded, reference


def encode_transition_event(
    event: dict[str, Any], before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    encoded = {key: copy.deepcopy(value) for key, value in event.items() if key != "after"}
    encoded.update(
        {
            "schema": EPISODE_SCHEMA_V2,
            "schema_version": TRACE_SCHEMA_VERSION_V2,
            "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
            "storage_fingerprint": storage_fingerprint(TRACE_FORMAT_DELTA_GZIP_V2),
            "state_delta": encode_state_delta(before, after),
            "state_extras_delta": encode_extras_delta(before, after),
        }
    )
    return encoded


def encode_finish_event(event: dict[str, Any]) -> dict[str, Any]:
    encoded = copy.deepcopy(event)
    encoded.update(
        {
            "schema": EPISODE_SCHEMA_V2,
            "schema_version": TRACE_SCHEMA_VERSION_V2,
            "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
            "storage_fingerprint": storage_fingerprint(TRACE_FORMAT_DELTA_GZIP_V2),
        }
    )
    return encoded


def _is_gzip_path(path: Path) -> bool:
    return path.name.endswith(".gz") or ".gz." in path.name


@contextlib.contextmanager
def open_trace_text(path: Path, mode: str) -> Iterator[TextIO]:
    if mode not in {"r", "w"}:
        raise ValueError("trace mode must be r or w")
    if mode == "r":
        if _is_gzip_path(path):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                yield stream
        else:
            with path.open("r", encoding="utf-8") as stream:
                yield stream
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_gzip_path(path):
        with path.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=GZIP_COMPRESSLEVEL,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                    yield stream
    else:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            yield stream


def iter_trace_events(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with open_trace_text(path, "r") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TraceStorageError(
                        f"trace row {line_number} is not an object: {path}"
                    )
                yield value
    except TraceStorageError:
        raise
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as error:
        raise TraceStorageError(f"cannot read trace {path}: {error}") from error


def read_trace_events(path: Path) -> list[dict[str, Any]]:
    return list(iter_trace_events(path))


def infer_collection_root(trace_path: Path) -> Path:
    resolved = trace_path.resolve()
    for parent in resolved.parents:
        if parent.name == "episodes":
            return parent.parent
    raise TraceStorageError(f"cannot infer collection root for trace: {trace_path}")


def resolve_state_blob(
    trace_path: Path, reference: str, collection_root: Path | None = None
) -> Path:
    root = (collection_root or infer_collection_root(trace_path)).resolve()
    path = (root / reference).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise TraceStorageError("state blob reference escapes collection root") from error
    return path


def trace_file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "trace_sha256": digest.hexdigest(),
        "trace_bytes": path.stat().st_size,
    }


def convert_v1_trace(
    source: Path,
    destination: Path,
    output_root: Path,
) -> dict[str, Any]:
    partial = partial_trace_path(destination)
    partial.unlink(missing_ok=True)
    event_count = 0
    initial_state_ref: str | None = None
    previous: dict[str, Any] | None = None
    try:
        with open_trace_text(partial, "w") as output:
            for event in iter_trace_events(source):
                event_count += 1
                kind = str(event.get("event"))
                if kind == "initial":
                    if previous is not None or not isinstance(event.get("state"), dict):
                        raise TraceStorageError("invalid v1 initial event")
                    previous = _plain(event["state"])
                    encoded, initial_state_ref = encode_initial_event(
                        event, previous, output_root
                    )
                elif kind == "transition":
                    after = event.get("after")
                    if previous is None or not isinstance(after, dict):
                        raise TraceStorageError("invalid v1 transition event")
                    encoded = encode_transition_event(event, previous, after)
                    reconstructed = apply_state_delta(previous, encoded["state_delta"])
                    reconstructed.update(
                        apply_extras_delta(previous, encoded["state_extras_delta"])
                    )
                    if reconstructed != after:
                        raise TraceStorageError("state delta round-trip mismatch")
                    previous = after
                elif kind == "finish":
                    if previous is None:
                        raise TraceStorageError("finish event precedes initial state")
                    encoded = encode_finish_event(event)
                else:
                    raise TraceStorageError(f"unsupported v1 event: {kind}")
                output.write(
                    json.dumps(
                        encoded,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    if initial_state_ref is None:
        raise TraceStorageError("trace did not contain an initial state")
    return {
        **trace_file_metadata(destination),
        "trace_event_count": event_count,
        "initial_state_ref": initial_state_ref,
    }


__all__ = [
    "EPISODE_SCHEMA_V1",
    "EPISODE_SCHEMA_V2",
    "TRACE_FORMAT_DELTA_GZIP_V2",
    "TRACE_FORMAT_FULL_V1",
    "TRACE_FORMATS",
    "TraceStorageError",
    "apply_extras_delta",
    "apply_state_delta",
    "convert_v1_trace",
    "encode_finish_event",
    "encode_initial_event",
    "encode_state_delta",
    "encode_transition_event",
    "infer_collection_root",
    "iter_trace_events",
    "open_trace_text",
    "partial_trace_path",
    "read_state_blob",
    "read_trace_events",
    "resolve_state_blob",
    "state_core",
    "storage_fingerprint",
    "trace_file_metadata",
    "trace_suffix",
    "write_state_blob",
]
