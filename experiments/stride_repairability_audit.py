from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.stride_lns import FROZEN_FEATURE_SCHEMA_ID, STRIDE_TRIAL_SCHEMA
from experiments.stride_repairability_collection import (
    COLLECTION_SCHEMA,
    STATE_SCHEMA,
    repairability_pp_seed,
)
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT


AUDIT_SCHEMA = "lns2.stride.repairability_collection_audit.v1"


def _trial_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["state_id"]),
        str(row["candidate_id"]),
        int(row["trial_index"]),
    )


def audit_repairability_collection(
    *,
    collection: str | Path,
    output: str | Path,
    expected_state_count: int = 240,
) -> dict[str, Any]:
    collection = Path(collection).resolve()
    output = Path(output).resolve()
    run_path = collection / "run_config.json"
    selection_path = collection / "state_selection.jsonl"
    status_path = collection / "collection_status.json"
    collection_report_path = collection / "collection_report.json"
    trial_path = collection / "repair_trials.jsonl"
    run = _read_json(run_path)
    status = _read_json(status_path)
    collection_report = _read_json(collection_report_path)
    selection = _read_jsonl(selection_path)
    selected_by_id = {str(row["state_id"]): row for row in selection}
    errors: list[str] = []
    if len(selected_by_id) != len(selection):
        errors.append("state selection contains duplicate ids")

    run_fingerprint = str(run.get("run_fingerprint", ""))
    trial_indices = tuple(map(int, run.get("pp_trial_indices") or ()))
    required_features = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    state_files = sorted((collection / "states").glob("*.json"))
    observed_state_ids: set[str] = set()
    expected_trial_hashes: dict[tuple[str, str, int], str] = {}
    candidate_count = 0
    boundary_candidate_count = 0
    split_maps: defaultdict[str, set[str]] = defaultdict(set)
    candidate_count_distribution: Counter[int] = Counter()
    boundary_count_distribution: Counter[int] = Counter()

    for state_path in state_files:
        payload = _read_json(state_path)
        state_id = str(payload.get("state_id", ""))
        if state_id in observed_state_ids:
            errors.append(f"duplicate state artifact: {state_id}")
            continue
        observed_state_ids.add(state_id)
        decision = dict(payload.get("decision") or {})
        selected = selected_by_id.get(state_id)
        if (
            payload.get("schema") != STATE_SCHEMA
            or payload.get("complete") is not True
            or payload.get("run_fingerprint") != run_fingerprint
            or selected is None
        ):
            errors.append(f"invalid state artifact identity: {state_id}")
            continue
        if (
            str(decision.get("before_fingerprint"))
            != str(payload.get("before_fingerprint"))
            or str(selected.get("before_fingerprint"))
            != str(payload.get("before_fingerprint"))
            or int(selected.get("before_conflicts", -1))
            != int(payload.get("before_conflicts", -2))
        ):
            errors.append(f"selected source state changed: {state_id}")
        restore = dict(payload.get("state_restore") or {})
        if (
            restore.get("contract") != TARGET_STATE_RESTORE_CONTRACT
            or restore.get("source_full_fingerprint")
            != payload.get("before_fingerprint")
            or restore.get("repair_structure_fingerprint")
            != payload.get("before_repair_fingerprint")
        ):
            errors.append(f"invalid target-path restoration: {state_id}")

        candidates = list(payload.get("candidates") or ())
        trials = list(payload.get("trials") or ())
        candidate_ids = [str(row.get("candidate_id")) for row in candidates]
        if (
            not candidates
            or len(candidates) > 20
            or len(candidate_ids) != len(set(candidate_ids))
        ):
            errors.append(f"invalid candidate pool cardinality: {state_id}")
            continue
        base = [row for row in candidates if row.get("candidate_kind") == "base"]
        boundary = [
            row for row in candidates if row.get("candidate_kind") == "boundary_only"
        ]
        if (
            len(base) != int(payload.get("base_candidate_count", -1))
            or len(boundary) != int(payload.get("boundary_candidate_count", -1))
            or len(boundary) > 2
        ):
            errors.append(f"candidate kind counts changed: {state_id}")
        base_roots = {
            family.split(":", 1)[0]
            for row in base
            for family in map(str, row.get("selection_families") or ())
            if not family.startswith("topology-boundary-")
        }
        if base_roots != {"target", "collision", "random"}:
            errors.append(f"frozen base candidate families changed: {state_id}")
        if any(
            int(row.get("actual_size", -1)) != 16
            or not any(
                str(value).startswith("topology-boundary-")
                for value in row.get("selection_families") or ()
            )
            for row in boundary
        ):
            errors.append(f"invalid topology boundary candidate: {state_id}")

        by_candidate: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for trial in trials:
            by_candidate[str(trial.get("candidate_id"))].append(trial)
        if set(by_candidate) != set(candidate_ids):
            errors.append(f"candidate trial coverage differs: {state_id}")
        repair_fingerprint = str(payload.get("before_repair_fingerprint", ""))
        seeds_by_index: defaultdict[int, set[int]] = defaultdict(set)
        features_by_candidate: dict[str, dict[str, float]] = {}
        for trial in trials:
            key = _trial_key(trial)
            candidate_id = key[1]
            index = key[2]
            features = trial.get("features")
            if (
                trial.get("schema") != STRIDE_TRIAL_SCHEMA
                or trial.get("feature_schema_id") != FROZEN_FEATURE_SCHEMA_ID
                or index not in trial_indices
                or not isinstance(features, dict)
                or set(features) != required_features
            ):
                errors.append(f"invalid trial schema/features: {key}")
                continue
            normalized_features = {
                str(name): float(value) for name, value in features.items()
            }
            if (
                candidate_id in features_by_candidate
                and features_by_candidate[candidate_id] != normalized_features
            ):
                errors.append(f"candidate features changed across seeds: {key}")
            features_by_candidate[candidate_id] = normalized_features
            seed = int(trial.get("pp_seed", -1))
            seeds_by_index[index].add(seed)
            if seed != repairability_pp_seed(repair_fingerprint, index):
                errors.append(f"unexpected paired PP seed: {key}")
            if key in expected_trial_hashes:
                errors.append(f"duplicate state trial: {key}")
            expected_trial_hashes[key] = _fingerprint(trial)
        expected_keys = {
            (state_id, candidate_id, index)
            for candidate_id in candidate_ids
            for index in trial_indices
        }
        if set(_trial_key(row) for row in trials) != expected_keys:
            errors.append(f"incomplete 16-seed product: {state_id}")
        if any(len(seeds_by_index[index]) != 1 for index in trial_indices):
            errors.append(f"PP seeds are not paired by trial: {state_id}")
        if len({next(iter(seeds_by_index[index]), -1) for index in trial_indices}) != len(
            trial_indices
        ):
            errors.append(f"PP seeds are not distinct across trials: {state_id}")

        split = str(selected.get("research_split"))
        split_maps[str(selected.get("map_id"))].add(split)
        candidate_count += len(candidates)
        boundary_candidate_count += len(boundary)
        candidate_count_distribution[len(candidates)] += 1
        boundary_count_distribution[len(boundary)] += 1

    actual_trial_hashes: dict[tuple[str, str, int], str] = {}
    if trial_path.is_file():
        with trial_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                key = _trial_key(row)
                if key in actual_trial_hashes:
                    errors.append(f"duplicate aggregate trial at line {line_number}: {key}")
                actual_trial_hashes[key] = _fingerprint(row)
    else:
        errors.append("repair_trials.jsonl is missing")
    if actual_trial_hashes != expected_trial_hashes:
        errors.append("aggregate repair trials differ from state artifacts")

    leaking_maps = sorted(map_id for map_id, splits in split_maps.items() if len(splits) != 1)
    gates = {
        "registered_collection_schema": run.get("schema") == COLLECTION_SCHEMA,
        "consistent_run_fingerprint": status.get("run_fingerprint")
        == run_fingerprint
        and collection_report.get("run_fingerprint") == run_fingerprint,
        "registered_restore_contract": run.get("target_state_restore_contract")
        == TARGET_STATE_RESTORE_CONTRACT,
        "registered_selection_identity": list(
            map(str, run.get("selected_state_ids") or ())
        )
        == [str(row["state_id"]) for row in selection]
        and collection_report.get("selection_sha256")
        == sha256_file(selection_path),
        "complete_status": status.get("status") == "complete",
        "complete_report": collection_report.get("complete") is True,
        "zero_collection_errors": int(status.get("error_state_count", -1)) == 0
        and int(collection_report.get("error_state_count", -1)) == 0,
        "registered_state_count": len(selection) == expected_state_count,
        "complete_state_coverage": observed_state_ids == set(selected_by_id)
        and len(observed_state_ids) == expected_state_count,
        "trial_indices_0_15": trial_indices == tuple(range(16)),
        "complete_trial_aggregation": actual_trial_hashes == expected_trial_hashes,
        "reported_counts_match": int(
            collection_report.get("requested_state_count", -1)
        )
        == expected_state_count
        and int(collection_report.get("completed_state_count", -1))
        == expected_state_count
        and int(collection_report.get("trial_count", -1))
        == len(actual_trial_hashes)
        and int(collection_report.get("expected_trial_count", -1))
        == len(expected_trial_hashes),
        "reported_candidate_distributions_match": dict(
            collection_report.get("candidate_count_distribution") or {}
        )
        == {
            str(key): value
            for key, value in sorted(candidate_count_distribution.items())
        }
        and dict(
            collection_report.get("boundary_candidate_count_distribution")
            or {}
        )
        == {
            str(key): value
            for key, value in sorted(boundary_count_distribution.items())
        },
        "map_held_out_split": not leaking_maps,
        "no_validation_errors": not errors,
    }
    report = {
        "schema": AUDIT_SCHEMA,
        "collection": str(collection),
        "run_fingerprint": run_fingerprint,
        "passed": all(gates.values()),
        "gates": gates,
        "errors": errors,
        "state_count": len(observed_state_ids),
        "candidate_count": candidate_count,
        "boundary_candidate_count": boundary_candidate_count,
        "trial_count": len(actual_trial_hashes),
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_count_distribution.items())
        },
        "boundary_candidate_count_distribution": {
            str(key): value for key, value in sorted(boundary_count_distribution.items())
        },
        "map_count_by_split": {
            split: len({map_id for map_id, splits in split_maps.items() if split in splits})
            for split in ("train", "validation")
        },
        "sha256": {
            "run_config": sha256_file(run_path),
            "state_selection": sha256_file(selection_path),
            "collection_status": sha256_file(status_path),
            "collection_report": sha256_file(collection_report_path),
            "repair_trials": sha256_file(trial_path) if trial_path.is_file() else None,
            "state_artifact_manifest": _fingerprint(
                [
                    {
                        "file": path.relative_to(collection).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for path in state_files
                ]
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "repairability_audit_report.json", report)
    return report


__all__ = ["AUDIT_SCHEMA", "audit_repairability_collection"]
