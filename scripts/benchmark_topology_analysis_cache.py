#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
if NATIVE_BUILD.is_dir() and str(NATIVE_BUILD) not in sys.path:
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments._common import contained_file  # noqa: E402
from experiments.closed_loop_trace_storage import (  # noqa: E402
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_trace_events,
)
from experiments.online_feature_engine import (  # noqa: E402
    TopologyAnalysisCache,
)
from experiments.repair_collection import (  # noqa: E402
    _fingerprint,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)
from experiments.trace_replay import _initial_state  # noqa: E402


SCHEMA = "lns2.stride.topology_analysis_cache_benchmark.v1"


def _analysis_signature(analysis: Any) -> str:
    return _fingerprint(
        {
            "visit_heat": sorted(analysis.visit_heat.items()),
            "agent_heat": sorted(analysis.agent_heat.items()),
            "events": [
                [event.time, event.kind, event.left, event.right, list(event.cells)]
                for event in analysis.events
            ],
            "pair_set": [list(pair) for pair in sorted(analysis.pair_set)],
            "component_id": sorted(analysis.component_id.items()),
            "component_members": [
                [component, sorted(members)]
                for component, members in sorted(analysis.component_members.items())
            ],
        }
    )


def _advance_state(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    if str(event.get("schema")) == EPISODE_SCHEMA_V2:
        after = apply_state_delta(state, event.get("state_delta"))
        after.update(apply_extras_delta(state, event.get("state_extras_delta")))
        return after
    return dict(event["after"])


def _sequences(collection_roots: list[Path]) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for collection_root in collection_roots:
        for manifest in _read_jsonl(
            collection_root / "realized_dynamic_manifest.jsonl"
        ):
            if str(manifest.get("status")) not in {"ok", "resumed"}:
                continue
            trace_path = contained_file(
                collection_root, manifest.get("trace_file"), field="trace_file"
            )
            events = read_trace_events(trace_path)
            state = _initial_state(collection_root, trace_path, events[0])
            pending_changed: set[int] = set()
            samples: list[dict[str, Any]] = []
            for event in events[1:-1]:
                before = state_fingerprint(state)
                if before != str(event.get("before_fingerprint")):
                    raise ValueError(
                        f"source before fingerprint mismatch: {trace_path}"
                    )
                controller = event.get("controller")
                proposal = (
                    controller.get("proposal")
                    if isinstance(controller, dict)
                    else None
                )
                if isinstance(proposal, dict) and bool(
                    proposal.get("structpool_gate_passed", False)
                ):
                    samples.append(
                        {
                            "state": copy.deepcopy(state),
                            "changed_agents": sorted(pending_changed),
                        }
                    )
                    pending_changed.clear()
                metrics = dict(event.get("metrics") or {})
                pending_changed.update(map(int, metrics.get("neighborhood", [])))
                state = _advance_state(state, event)
            if samples:
                sequences.append(
                    {
                        "episode_id": str(manifest["episode_id"]),
                        "samples": samples,
                    }
                )
    sequences.sort(key=lambda row: str(row["episode_id"]))
    if len({str(row["episode_id"]) for row in sequences}) != len(sequences):
        raise ValueError("cache benchmark contains duplicate episode ids")
    return sequences


def _run_sequence(
    sequence: dict[str, Any], *, incremental: bool, collect_signatures: bool
) -> list[str]:
    samples = list(sequence["samples"])
    cache = TopologyAnalysisCache(samples[0]["state"], backend="native")
    signatures = (
        [_analysis_signature(cache.analysis)] if collect_signatures else []
    )
    for sample in samples[1:]:
        cache.prepare(
            sample["state"],
            changed_agents=(sample["changed_agents"] if incremental else None),
        )
        if collect_signatures:
            signatures.append(_analysis_signature(cache.analysis))
    return signatures


def _run_all(
    sequences: list[dict[str, Any]], *, incremental: bool
) -> None:
    for sequence in sequences:
        _run_sequence(
            sequence, incremental=incremental, collect_signatures=False
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact native topology analysis with full versus "
            "changed-agent path-heat maintenance."
        )
    )
    parser.add_argument("--collection", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=11)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    roots = [Path(value).resolve() for value in arguments.collection]
    sequences = _sequences(roots)
    reference_signatures = {
        str(sequence["episode_id"]): _run_sequence(
            sequence, incremental=False, collect_signatures=True
        )
        for sequence in sequences
    }
    incremental_signatures = {
        str(sequence["episode_id"]): _run_sequence(
            sequence, incremental=True, collect_signatures=True
        )
        for sequence in sequences
    }
    exact = reference_signatures == incremental_signatures
    for _ in range(arguments.warmup):
        _run_all(sequences, incremental=False)
        _run_all(sequences, incremental=True)

    reference_seconds: list[float] = []
    incremental_seconds: list[float] = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(arguments.repeats):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            measured: dict[bool, float] = {}
            for incremental in order:
                started = time.perf_counter()
                _run_all(sequences, incremental=incremental)
                measured[incremental] = time.perf_counter() - started
            reference_seconds.append(measured[False])
            incremental_seconds.append(measured[True])
    finally:
        if enabled:
            gc.enable()

    paired_faster_count = sum(
        candidate < reference
        for reference, candidate in zip(reference_seconds, incremental_seconds)
    )
    report = {
        "schema": SCHEMA,
        "collections": [root.as_posix() for root in roots],
        "episode_count": len(sequences),
        "activated_state_count": sum(
            len(sequence["samples"]) for sequence in sequences
        ),
        "analysis_exact": exact,
        "analysis_signature_sha256": _fingerprint(reference_signatures),
        "reference_seconds": reference_seconds,
        "incremental_seconds": incremental_seconds,
        "reference_median_seconds": statistics.median(reference_seconds),
        "incremental_median_seconds": statistics.median(incremental_seconds),
        "paired_faster_count": paired_faster_count,
        "repeat_count": arguments.repeats,
    }
    report["median_relative_improvement"] = 1.0 - (
        report["incremental_median_seconds"]
        / report["reference_median_seconds"]
    )
    report["passed"] = bool(
        exact
        and paired_faster_count >= (arguments.repeats // 2 + 1)
        and report["incremental_median_seconds"]
        < report["reference_median_seconds"]
    )
    _write_json(Path(arguments.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
