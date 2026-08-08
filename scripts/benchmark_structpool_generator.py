#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import contained_file  # noqa: E402
from experiments.closed_loop_trace_storage import (  # noqa: E402
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
)
from experiments.repair_collection import _read_jsonl, _write_json, state_fingerprint  # noqa: E402
from experiments.state_analysis import analyze_state, analyze_static_grid  # noqa: E402
from lns2_selector.runtime.topology_candidates import (  # noqa: E402
    generate_structpool_candidates,
)


SCHEMA = "lns2.stride.structpool_generator_benchmark.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


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
        raise ValueError("source trace has invalid initial extras")
    state.update(extras)
    return state


def _samples(collection_roots: list[Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for collection_root in collection_roots:
        manifest_path = collection_root / "realized_dynamic_manifest.jsonl"
        for manifest in _read_jsonl(manifest_path):
            if str(manifest.get("status")) not in {"ok", "resumed"}:
                continue
            trace_path = contained_file(
                collection_root, manifest.get("trace_file"), field="trace_file"
            )
            events = read_trace_events(trace_path)
            state = _initial_state(collection_root, trace_path, events[0])
            static_grid = analyze_static_grid(state)
            for event in events[1:-1]:
                before = state_fingerprint(state)
                if before != str(event.get("before_fingerprint")):
                    raise ValueError(f"source before fingerprint mismatch: {trace_path}")
                controller = event.get("controller")
                proposal = controller.get("proposal") if isinstance(controller, dict) else None
                if isinstance(proposal, dict) and bool(
                    proposal.get("structpool_gate_passed", False)
                ):
                    analysis = analyze_state(state, static_grid=static_grid)
                    samples.append(
                        {
                            "key": (
                                f"{collection_root.as_posix()}|"
                                f"{manifest['episode_id']}|"
                                f"{int(event['decision_index'])}|{before}"
                            ),
                            "state_fingerprint": before,
                            "state": state,
                            "analysis": analysis,
                        }
                    )
                if str(event.get("schema")) == EPISODE_SCHEMA_V2:
                    after = apply_state_delta(state, event.get("state_delta"))
                    after.update(
                        apply_extras_delta(state, event.get("state_extras_delta"))
                    )
                else:
                    after = dict(event["after"])
                state = after
    samples.sort(key=lambda row: str(row["key"]))
    if len({str(row["key"]) for row in samples}) != len(samples):
        raise ValueError("benchmark samples contain duplicate keys")
    return samples


def _candidate_records(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for sample in samples:
        candidates = generate_structpool_candidates(sample["state"], sample["analysis"])
        records.append(
            {
                "key": sample["key"],
                "state_fingerprint": sample["state_fingerprint"],
                "candidate_count": len(candidates),
                "candidate_sha256": _sha256(candidates),
                "candidates": candidates,
            }
        )
    return records


def _timings(samples: list[dict[str, Any]], repeats: int) -> list[float]:
    values = []
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            for sample in samples:
                generate_structpool_candidates(sample["state"], sample["analysis"])
            values.append(time.perf_counter() - started)
    finally:
        if enabled:
            gc.enable()
    return values


def _comparison(
    records: list[dict[str, Any]], expected_path: Path | None
) -> dict[str, Any] | None:
    if expected_path is None:
        return None
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    expected_by_key = {str(row["key"]): row for row in expected["records"]}
    actual_by_key = {str(row["key"]): row for row in records}
    missing = sorted(set(expected_by_key) - set(actual_by_key))
    unexpected = sorted(set(actual_by_key) - set(expected_by_key))
    changed = sorted(
        key
        for key in set(expected_by_key) & set(actual_by_key)
        if expected_by_key[key]["candidates"] != actual_by_key[key]["candidates"]
    )
    return {
        "expected": expected_path.as_posix(),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "changed_keys": changed,
        "exact_match": not missing and not unexpected and not changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark StructPool generation on stored activated decisions."
    )
    parser.add_argument("--collection", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected")
    parser.add_argument("--repeats", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        raise ValueError("repeats must be positive")

    roots = [Path(value).resolve() for value in arguments.collection]
    samples = _samples(roots)
    records = _candidate_records(samples)
    timings = _timings(samples, arguments.repeats)
    comparison = _comparison(
        records, Path(arguments.expected).resolve() if arguments.expected else None
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    report = {
        "schema": SCHEMA,
        "source_commit": commit,
        "collections": [root.as_posix() for root in roots],
        "activated_state_count": len(samples),
        "candidate_count": sum(int(row["candidate_count"]) for row in records),
        "records_sha256": _sha256(records),
        "batch_seconds": timings,
        "median_batch_seconds": statistics.median(timings),
        "minimum_batch_seconds": min(timings),
        "comparison": comparison,
        "records": records,
    }
    _write_json(Path(arguments.output), report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if comparison is None or comparison["exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
