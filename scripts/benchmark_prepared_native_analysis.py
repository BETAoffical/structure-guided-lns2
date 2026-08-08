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
    read_trace_events,
)
from experiments.online_feature_engine import (  # noqa: E402
    OnlineFeatureEngine,
    TopologyAnalysisCache,
)
from experiments.repair_collection import (  # noqa: E402
    _fingerprint,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)
from experiments.trace_replay import _initial_state  # noqa: E402
from scripts.benchmark_topology_analysis_cache import _advance_state  # noqa: E402


SCHEMA = "lns2.stride.prepared_native_analysis_benchmark.v1"


def _retained_candidates(controller: dict[str, Any]) -> list[dict[str, Any]]:
    pool = controller.get("candidate_pool")
    if not isinstance(pool, list) or not pool:
        raise ValueError("activated StructPool transition lacks a candidate pool")
    retained = [row for row in pool if bool(row.get("retained", True))]
    if not retained:
        raise ValueError("activated StructPool transition retains no candidates")
    return [copy.deepcopy(dict(row)) for row in retained]


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
                if state_fingerprint(state) != str(event.get("before_fingerprint")):
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
                    assert isinstance(controller, dict)
                    samples.append(
                        {
                            "state": copy.deepcopy(state),
                            "changed_agents": sorted(pending_changed),
                            "candidates": _retained_candidates(controller),
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
        raise ValueError("prepared-analysis benchmark has duplicate episode ids")
    return sequences


def _row_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": str(row["candidate_id"]),
            "feature_names": list(map(str, row["feature_names"])),
            "feature_values": list(map(float, row["feature_values"])),
        }
        for row in rows
    ]


def _run_sequence(
    sequence: dict[str, Any], *, shared: bool, collect_signature: bool
) -> list[dict[str, Any]]:
    samples = list(sequence["samples"])
    first = samples[0]
    topology = TopologyAnalysisCache(first["state"], backend="native")
    features = OnlineFeatureEngine(
        first["state"], backend="native", dense_output=True
    )
    if shared:
        if topology.last_native_prepared is None:
            raise RuntimeError("prepared native analysis API is unavailable")
        features.prepare(
            first["state"],
            prepared_native_analysis=topology.last_native_prepared,
        )
    rows, metrics = features.realized_rows(
        first["candidates"], state_hash="benchmark"
    )
    if shared and float(metrics.get("prepared_analysis_reused", 0.0)) != 1.0:
        raise RuntimeError("prepared analysis was not reused")
    signatures = [_row_signature(rows)] if collect_signature else []

    for sample in samples[1:]:
        topology.prepare(
            sample["state"], changed_agents=sample["changed_agents"]
        )
        features.prepare(
            sample["state"],
            changed_agents=sample["changed_agents"],
            prepared_native_analysis=(
                topology.last_native_prepared if shared else None
            ),
        )
        rows, metrics = features.realized_rows(
            sample["candidates"], state_hash="benchmark"
        )
        if shared and float(metrics.get("prepared_analysis_reused", 0.0)) != 1.0:
            raise RuntimeError("prepared analysis was not reused")
        if collect_signature:
            signatures.append(_row_signature(rows))
    return signatures


def _run_all(sequences: list[dict[str, Any]], *, shared: bool) -> None:
    for sequence in sequences:
        _run_sequence(sequence, shared=shared, collect_signature=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark separate versus shared native StructPool topology and "
            "124-dimensional feature analysis."
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
    separate_signatures = {
        str(sequence["episode_id"]): _run_sequence(
            sequence, shared=False, collect_signature=True
        )
        for sequence in sequences
    }
    shared_signatures = {
        str(sequence["episode_id"]): _run_sequence(
            sequence, shared=True, collect_signature=True
        )
        for sequence in sequences
    }
    exact = separate_signatures == shared_signatures
    for _ in range(arguments.warmup):
        _run_all(sequences, shared=False)
        _run_all(sequences, shared=True)

    separate_seconds: list[float] = []
    shared_seconds: list[float] = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(arguments.repeats):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            measured: dict[bool, float] = {}
            for shared in order:
                started = time.perf_counter()
                _run_all(sequences, shared=shared)
                measured[shared] = time.perf_counter() - started
            separate_seconds.append(measured[False])
            shared_seconds.append(measured[True])
    finally:
        if enabled:
            gc.enable()

    paired_faster_count = sum(
        shared < separate
        for separate, shared in zip(separate_seconds, shared_seconds)
    )
    report = {
        "schema": SCHEMA,
        "collections": [root.as_posix() for root in roots],
        "episode_count": len(sequences),
        "activated_state_count": sum(
            len(sequence["samples"]) for sequence in sequences
        ),
        "feature_vector_exact": exact,
        "feature_signature_sha256": _fingerprint(separate_signatures),
        "separate_seconds": separate_seconds,
        "shared_seconds": shared_seconds,
        "separate_median_seconds": statistics.median(separate_seconds),
        "shared_median_seconds": statistics.median(shared_seconds),
        "paired_faster_count": paired_faster_count,
        "repeat_count": arguments.repeats,
    }
    report["median_relative_improvement"] = 1.0 - (
        report["shared_median_seconds"] / report["separate_median_seconds"]
    )
    report["passed"] = bool(
        exact
        and paired_faster_count >= (arguments.repeats // 2 + 1)
        and report["shared_median_seconds"]
        < report["separate_median_seconds"]
    )
    _write_json(Path(arguments.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
