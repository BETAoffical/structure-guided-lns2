"""Result-blind H1 state selection from the compact-flow source product.

The selector verifies the complete balanced source product before extracting
any pre-action state.  It never executes a solver, inspects a target action
outcome, filters on an outcome, or backfills a failed supply gate.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from experiments import trace_replay
from experiments._common import contained_file, sha256_file


CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_h1_state_selection_config.v1"
)
SELECTED_STATE_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_h1_selected_state.v1"
)
TRUST_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_h1_state_selection_trust.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_h1_state_selection_v1"
DEFAULT_OUTPUT_NAME = "stride-hierarchical-ch-compact-flow-h1-state-selection-v1"
DEFAULT_WORKERS = 16
MAXIMUM_WORKERS = 20
ALLOWED_SPLITS = ("train", "development")
SOURCE_EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_balanced_source_v1"
SOURCE_TRUST_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_balanced_source_collection_trust.v1"
)
SOURCE_STATUS = "COLLECTED_EXACT_PRODUCT"
SOURCE_POLICY = "realized_dynamic"
SOURCE_CONTROLLER = "v2-full"
SOLVER_SEEDS = (41, 42)
EXPECTED_EPISODES_PER_SPLIT = 192
EXPECTED_EPISODE_COUNT = 384
EXPECTED_SELECTED_PER_SPLIT = 96
EXPECTED_SELECTED_COUNT = 192
TRAIN_MAPS = (
    "den404d",
    "lak101d",
    "den201d",
    "hrt002d",
    "den009d",
    "den101d",
    "den203d",
    "den308d",
)
DEVELOPMENT_MAPS = (
    "lak108d",
    "lak110d",
    "ost102d",
    "den408d",
    "den202d",
    "den207d",
    "den998d",
    "den020d",
)
TRAIN_MAP_FOLDS = {
    "fold0": ("den404d", "den009d"),
    "fold1": ("lak101d", "den101d"),
    "fold2": ("den201d", "den203d"),
    "fold3": ("hrt002d", "den308d"),
}
MAP_LOADS = {
    "den404d": (80, 180, 280),
    "lak101d": (70, 160, 240),
    "den201d": (40, 100, 180),
    "hrt002d": (50, 100, 180),
    "den009d": (60, 120, 200),
    "den101d": (80, 160, 240),
    "den203d": (100, 200, 320),
    "den308d": (100, 200, 320),
    "lak108d": (60, 140, 220),
    "lak110d": (40, 90, 140),
    "ost102d": (50, 120, 200),
    "den408d": (100, 240, 400),
    "den202d": (40, 100, 180),
    "den207d": (60, 120, 200),
    "den998d": (80, 160, 240),
    "den020d": (100, 200, 320),
}
DEPTH_BANDS = {"d0": (0, 0), "d1_3": (1, 3), "d4plus": (4, 11)}


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL object required: {path}:{line_number}")
        rows.append(row)
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("compact-flow H1 selection identity changed")
    expected_source = {
        "root": "build/stride-hierarchical-ch-compact-flow-balanced-source-v1",
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "collection_trust_schema": SOURCE_TRUST_SCHEMA,
        "collection_trust_file": "collection_trust_report__all.json",
        "schedule_file": "source_schedule.jsonl",
        "realized_manifest_by_split": {
            "train": "collection/train/realized_dynamic_manifest.jsonl",
            "development": "collection/development/realized_dynamic_manifest.jsonl",
        },
        "required_status": SOURCE_STATUS,
        "required_policy": SOURCE_POLICY,
        "solver_seeds": list(SOLVER_SEEDS),
        "expected_episode_count": EXPECTED_EPISODE_COUNT,
        "expected_episodes_per_split": EXPECTED_EPISODES_PER_SPLIT,
    }
    if config.get("source") != expected_source:
        raise ValueError("compact-flow H1 source contract changed")
    if config.get("map_splits") != {
        "train": list(TRAIN_MAPS),
        "development": list(DEVELOPMENT_MAPS),
    }:
        raise ValueError("compact-flow H1 map splits changed")
    if set(TRAIN_MAPS) & set(DEVELOPMENT_MAPS):
        raise ValueError("train and development maps overlap")
    if config.get("map_loads") != {
        key: list(value) for key, value in MAP_LOADS.items()
    }:
        raise ValueError("compact-flow H1 map/load product changed")
    folds = config.get("train_map_folds")
    if folds != {key: list(value) for key, value in TRAIN_MAP_FOLDS.items()}:
        raise ValueError("outcome-blind train map folds changed")
    folded = [map_id for values in folds.values() for map_id in values]
    if len(folded) != len(set(folded)) or set(folded) != set(TRAIN_MAPS):
        raise ValueError("train folds must partition all train maps exactly once")
    if config.get("selection") != {
        "depth_bands": {key: list(value) for key, value in DEPTH_BANDS.items()},
        "target_states_per_depth_band_per_split": 32,
        "minimum_active_maps_per_depth_band_per_split": 7,
        "minimum_states_per_map": 8,
        "maximum_states_per_episode": 2,
        "maximum_states_per_map_load": 8,
        "maximum_states_per_map": 16,
        "expected_selected_states_per_split": EXPECTED_SELECTED_PER_SPLIT,
        "expected_selected_state_count": EXPECTED_SELECTED_COUNT,
        "rank_rule": (
            "sha256_canonical_json_array_of_split_task_seed_decision_index_"
            "before_fingerprint"
        ),
        "result_blind_extractor": (
            "experiments.trace_replay.result_blind_decision_rows"
        ),
        "outcome_filtering_allowed": False,
        "reserve_or_replacement_backfill_allowed": False,
        "under_supply_action": "STATE_SUPPLY_FAIL_NO_BACKFILL",
    }:
        raise ValueError("compact-flow H1 selection contract changed")
    if config.get("execution") != {
        "workers": DEFAULT_WORKERS,
        "maximum_workers": MAXIMUM_WORKERS,
        "workers_enter_run_fingerprint": True,
        "solver_invocation_allowed": False,
    }:
        raise ValueError("compact-flow H1 worker or no-solver contract changed")
    if config.get("sealed_final") != {
        "status": "SEALED",
        "source_registered": False,
        "semantic_access_allowed": False,
        "state_selection_allowed": False,
        "access_requires_frozen_model_and_threshold": True,
    }:
        raise ValueError("sealed-final boundary changed")
    if config.get("claim_boundary") != {
        "train_and_development_usage": "sequential_design_only",
        "map_disjoint_final_confirmation_required": True,
        "target_outcomes_read_for_selection": False,
        "h1_outcomes_collected": False,
        "training_authorized": False,
        "model_fit_executed": False,
        "runtime_or_ttf_claim_authorized": False,
    }:
        raise ValueError("compact-flow H1 claim boundary changed")


def load_config(config_path: str | Path) -> dict[str, Any]:
    config = _read_json(Path(config_path).resolve())
    _validate_config(config)
    return config


def _effective_workers(config: dict[str, Any], workers: int | None) -> int:
    value = int(config["execution"]["workers"] if workers is None else workers)
    maximum = int(config["execution"]["maximum_workers"])
    if value < 1 or value > maximum:
        raise ValueError(f"workers must be between 1 and {maximum}")
    return value


def _depth_band(decision_index: int) -> str:
    for name, (lower, upper) in DEPTH_BANDS.items():
        if lower <= decision_index <= upper:
            return name
    raise ValueError(f"decision index is outside the frozen depth bands: {decision_index}")


def selection_rank(
    split: str,
    task_id: str,
    solver_seed: int,
    decision_index: int,
    before_fingerprint: str,
) -> str:
    """Rank exactly the five registered pre-action identity fields."""

    payload = [
        str(split),
        str(task_id),
        int(solver_seed),
        int(decision_index),
        str(before_fingerprint),
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _validate_source_trust(trust: dict[str, Any]) -> None:
    if (
        trust.get("schema") != SOURCE_TRUST_SCHEMA
        or trust.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or trust.get("status") != SOURCE_STATUS
    ):
        raise ValueError("balanced source trust is not COLLECTED_EXACT_PRODUCT")
    if (
        int(trust.get("planned_episode_count", -1)) != EXPECTED_EPISODE_COUNT
        or int(trust.get("realized_episode_count", -1)) != EXPECTED_EPISODE_COUNT
        or int(trust.get("qualification_job_count", -1)) != EXPECTED_EPISODE_COUNT
        or trust.get("qualification_passed")
        != {"train": True, "development": True}
        or int(trust.get("source_v2_qualification_rows_imported", -1)) != 0
        or trust.get("outcome_filtering") is not False
        or trust.get("sealed_final_semantic_access") is not False
        or trust.get("training_authorized") is not False
    ):
        raise ValueError("balanced source trust boundary or exact count changed")
    source_workers = trust.get("workers")
    if type(source_workers) is not int or not 1 <= source_workers <= MAXIMUM_WORKERS:
        raise ValueError("balanced source trust has invalid workers")
    schedule_sha = str(trust.get("source_schedule_sha256") or "")
    if len(schedule_sha) != 64:
        raise ValueError("balanced source trust lacks source schedule SHA256")


def _validate_schedule(
    rows: list[dict[str, Any]], trust: dict[str, Any]
) -> dict[tuple[str, str, int], dict[str, Any]]:
    if len(rows) != EXPECTED_EPISODE_COUNT:
        raise ValueError("source schedule is not exactly 384 rows")
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_workers = int(trust["workers"])
    split_maps = {"train": set(TRAIN_MAPS), "development": set(DEVELOPMENT_MAPS)}
    for row in rows:
        split = str(row.get("split") or "")
        task_id = str(row.get("task_id") or "")
        seed = row.get("solver_seed")
        map_id = str(row.get("map_id") or "")
        load = row.get("agent_count")
        if (
            split not in ALLOWED_SPLITS
            or not task_id
            or type(seed) is not int
            or seed not in SOLVER_SEEDS
            or map_id not in split_maps[split]
            or type(load) is not int
            or load not in MAP_LOADS[map_id]
            or row.get("controller") != SOURCE_CONTROLLER
            or row.get("policy") != SOURCE_POLICY
            or row.get("qualification_required") is not True
            or row.get("source_v2_qualification_imported") is not False
            or row.get("training_authorized") is not False
            or row.get("workers") != source_workers
        ):
            raise ValueError("source schedule metadata changed")
        key = (split, task_id, seed)
        if key in index:
            raise ValueError("source schedule contains a duplicate task/seed")
        index[key] = dict(row)
    split_counts = Counter(key[0] for key in index)
    if split_counts != Counter({"train": 192, "development": 192}):
        raise ValueError("source schedule split counts changed")
    for split, maps in (("train", TRAIN_MAPS), ("development", DEVELOPMENT_MAPS)):
        for map_id in maps:
            map_rows = [
                row
                for (row_split, _task, _seed), row in index.items()
                if row_split == split and row["map_id"] == map_id
            ]
            if len(map_rows) != 24:
                raise ValueError(f"source schedule map product changed: {split}/{map_id}")
            per_load = Counter(int(row["agent_count"]) for row in map_rows)
            if per_load != Counter({load: 8 for load in MAP_LOADS[map_id]}):
                raise ValueError(f"source schedule map/load product changed: {map_id}")
    return index


def _validate_manifest(
    split: str,
    rows: list[dict[str, Any]],
    schedule: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_EPISODES_PER_SPLIT:
        raise ValueError(f"{split} realized manifest is not exactly 192 rows")
    sanitized: list[dict[str, Any]] = []
    observed: set[tuple[str, str, int]] = set()
    trace_files: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        seed = row.get("solver_seed")
        key = (split, task_id, seed) if type(seed) is int else (split, task_id, -1)
        expected = schedule.get(key)
        trace_file = str(row.get("trace_file") or "")
        trace_sha = str(row.get("trace_sha256") or "")
        if (
            expected is None
            or key in observed
            or str(row.get("status")) != "ok"
            or row.get("error") not in (None, "")
            or str(row.get("split")) != split
            or str(row.get("map_id")) != str(expected["map_id"])
            or row.get("agent_count") != expected["agent_count"]
            or not str(row.get("layout_mode") or "")
            or row.get("policy") != SOURCE_POLICY
            or not str(row.get("episode_id") or "")
            or not trace_file
            or len(trace_sha) != 64
            or trace_file in trace_files
        ):
            raise ValueError(f"{split} realized row is incomplete or differs from schedule")
        observed.add(key)
        trace_files.add(trace_file)
        sanitized.append(
            {
                "split": split,
                "task_id": task_id,
                "solver_seed": int(seed),
                "episode_id": str(row["episode_id"]),
                "map_id": str(expected["map_id"]),
                "map_family": str(row["layout_mode"]),
                "agent_count": int(expected["agent_count"]),
                "od_variant": str(expected.get("od_variant") or ""),
                "task_seed": int(expected.get("task_seed", -1)),
                "trace_file": trace_file,
                "trace_sha256": trace_sha,
            }
        )
    expected_keys = {key for key in schedule if key[0] == split}
    if observed != expected_keys:
        raise ValueError(f"{split} realized manifest differs from exact schedule product")
    sanitized.sort(key=lambda row: (row["task_id"], row["solver_seed"]))
    return sanitized


def _extract_episode(item: tuple[Path, dict[str, Any]]) -> list[dict[str, Any]]:
    source_root, episode = item
    trace_path = contained_file(
        source_root, episode["trace_file"], field="source trace_file"
    )
    if sha256_file(trace_path) != episode["trace_sha256"]:
        raise ValueError(f"source trace SHA256 mismatch: {episode['episode_id']}")
    # Pass a deliberately sanitized manifest.  The result-blind replay helper
    # cannot observe the source manifest summary or any other outcome field.
    preactions, _discarded_events = trace_replay.result_blind_decision_rows(
        source_root, {"trace_file": episode["trace_file"]}
    )
    selected_fields: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in preactions:
        decision_index = row.get("decision_index")
        before_fingerprint = str(row.get("before_fingerprint") or "")
        if (
            type(decision_index) is not int
            or not 0 <= decision_index <= 11
            or not before_fingerprint
            or type(row.get("before_conflicts")) is not int
            or not isinstance(row.get("prefix_actions"), list)
            or (decision_index, before_fingerprint) in seen
        ):
            raise ValueError(f"invalid result-blind pre-action row: {episode['episode_id']}")
        seen.add((decision_index, before_fingerprint))
        selected_fields.append(
            {
                **episode,
                "decision_index": decision_index,
                "before_fingerprint": before_fingerprint,
                "before_conflicts": int(row["before_conflicts"]),
                "prefix_actions": [dict(action) for action in row["prefix_actions"]],
            }
        )
    return selected_fields


def _selection_candidates(
    source_roots: dict[str, Path],
    episodes: dict[str, list[dict[str, Any]]],
    workers: int,
) -> list[dict[str, Any]]:
    items = [
        (source_roots[split], episode)
        for split in ALLOWED_SPLITS
        for episode in episodes[split]
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        nested = list(executor.map(_extract_episode, items))
    rows = [row for episode_rows in nested for row in episode_rows]
    identities = [
        (
            row["split"],
            row["task_id"],
            row["solver_seed"],
            row["decision_index"],
            row["before_fingerprint"],
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate pre-action identity in source traces")
    for row in rows:
        row["depth_band"] = _depth_band(int(row["decision_index"]))
        row["selection_rank_sha256"] = selection_rank(
            row["split"],
            row["task_id"],
            row["solver_seed"],
            row["decision_index"],
            row["before_fingerprint"],
        )
    return rows


def _select(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_per_band = 32
    minimum_per_map = 8
    maximum_per_map = 16
    episode_cap = 2
    map_load_cap = 8
    episode_counts: Counter[str] = Counter()
    map_load_counts: Counter[tuple[str, str, int]] = Counter()
    map_counts: Counter[tuple[str, str]] = Counter()
    band_counts: Counter[tuple[str, str]] = Counter()
    provisional: list[dict[str, Any]] = []
    failures: list[str] = []
    supply_counts: dict[str, dict[str, dict[str, int]]] = {}

    def can_add(row: dict[str, Any]) -> bool:
        return (
            episode_counts[str(row["episode_id"])] < episode_cap
            and map_load_counts[
                (row["split"], row["map_id"], int(row["agent_count"]))
            ]
            < map_load_cap
            and map_counts[(row["split"], row["map_id"])] < maximum_per_map
            and band_counts[(row["split"], row["depth_band"])] < target_per_band
            and row not in provisional
        )

    def add(row: dict[str, Any]) -> None:
        provisional.append(row)
        episode_counts[str(row["episode_id"])] += 1
        map_load_counts[
            (row["split"], row["map_id"], int(row["agent_count"]))
        ] += 1
        map_counts[(row["split"], row["map_id"])] += 1
        band_counts[(row["split"], row["depth_band"])] += 1

    split_maps = {"train": TRAIN_MAPS, "development": DEVELOPMENT_MAPS}
    for split in ALLOWED_SPLITS:
        supply_counts[split] = {}
        candidates_by_map: dict[str, list[dict[str, Any]]] = {}
        for map_id in split_maps[split]:
            candidates = sorted(
                (
                    row
                    for row in rows
                    if row["split"] == split and row["map_id"] == map_id
                ),
                key=lambda row: row["selection_rank_sha256"],
            )
            candidates_by_map[map_id] = candidates
            supply_counts[split][map_id] = dict(
                Counter(str(row["depth_band"]) for row in candidates)
            )
        # Pre-registered map floor.  Cycle the depth bands, but require only a
        # split-level depth gate: an unavailable map/depth cell is not a hard
        # failure.  SHA order is used within each eligible cell.
        for map_id in split_maps[split]:
            candidates = candidates_by_map[map_id]
            while map_counts[(split, map_id)] < minimum_per_map:
                preferred = tuple(DEPTH_BANDS)[
                    map_counts[(split, map_id)] % len(DEPTH_BANDS)
                ]
                band_order = (preferred,) + tuple(
                    band for band in DEPTH_BANDS if band != preferred
                )
                winner = next(
                    (
                        row
                        for band in band_order
                        for row in candidates
                        if row["depth_band"] == band and can_add(row)
                    ),
                    None,
                )
                if winner is None:
                    failures.append(f"{split}/{map_id}:minimum_8")
                    break
                add(winner)
        # Complete the three exact split-level depth quotas in fixed band
        # order.  This is part of the original selection, never a reserve or
        # outcome-triggered backfill pass.
        for band in DEPTH_BANDS:
            candidates = sorted(
                (
                    row
                    for row in rows
                    if row["split"] == split and row["depth_band"] == band
                ),
                key=lambda row: row["selection_rank_sha256"],
            )
            for row in candidates:
                if band_counts[(split, band)] >= target_per_band:
                    break
                if can_add(row):
                    add(row)
            if band_counts[(split, band)] != target_per_band:
                failures.append(f"{split}/{band}:exact_32")
        active_maps = {
            band: {
                row["map_id"]
                for row in provisional
                if row["split"] == split and row["depth_band"] == band
            }
            for band in DEPTH_BANDS
        }
        for band, maps in active_maps.items():
            if len(maps) < 7:
                failures.append(f"{split}/{band}:minimum_7_active_maps")

    if failures:
        return [], {
            "status": "STATE_SUPPLY_FAIL_NO_BACKFILL",
            "failure_reasons": sorted(set(failures)),
            "preaction_state_count": len(rows),
            "provisional_selected_state_count": len(provisional),
            "selected_state_count": 0,
            "supply_counts": supply_counts,
        }

    fold_by_map = {
        map_id: fold for fold, maps in TRAIN_MAP_FOLDS.items() for map_id in maps
    }
    frozen: list[dict[str, Any]] = []
    for row in provisional:
        occurrence_id = "compact-flow-h1-state-" + _fingerprint(
            [
                row["split"],
                row["task_id"],
                row["solver_seed"],
                row["decision_index"],
                row["before_fingerprint"],
            ]
        )[:24]
        frozen.append(
            {
                "schema": SELECTED_STATE_SCHEMA,
                "state_id": occurrence_id,
                "state_occurrence_id": occurrence_id,
                "split": row["split"],
                "map_id": row["map_id"],
                "map_family": row["map_family"],
                "agent_count": row["agent_count"],
                "load": row["agent_count"],
                "od_variant": row["od_variant"],
                "task_seed": row["task_seed"],
                "task_id": row["task_id"],
                "episode_id": row["episode_id"],
                "solver_seed": row["solver_seed"],
                "source_policy": SOURCE_CONTROLLER,
                "decision_index": row["decision_index"],
                "before_fingerprint": row["before_fingerprint"],
                "before_conflicts": row["before_conflicts"],
                "depth_band": row["depth_band"],
                "prefix_actions": row["prefix_actions"],
                "source_root": row["source_root"],
                "source_trace_file": row["trace_file"],
                "source_trace_sha256": row["trace_sha256"],
                "selection_rank_sha256": row["selection_rank_sha256"],
                "train_fold": fold_by_map.get(row["map_id"]),
                "target_outcome_fields_read": False,
                "outcome_filtering": False,
                "reserve_or_replacement_backfill": False,
                "sequential_design_only": True,
                "training_authorized": False,
            }
        )
    frozen.sort(key=lambda row: (row["split"], row["map_id"], row["selection_rank_sha256"]))
    split_counts = Counter(row["split"] for row in frozen)
    frozen_map_counts = Counter((row["split"], row["map_id"]) for row in frozen)
    frozen_band_counts = Counter(
        (row["split"], row["map_id"], row["depth_band"]) for row in frozen
    )
    failures = []
    if len(frozen) != EXPECTED_SELECTED_COUNT:
        failures.append("exact_192")
    if split_counts != Counter({"train": 96, "development": 96}):
        failures.append("exact_96_per_split")
    if (
        len(frozen_map_counts) != 16
        or any(not 8 <= value <= 16 for value in frozen_map_counts.values())
    ):
        failures.append("per_map_8_to_16")
    if any(
        sum(
            frozen_band_counts[(split, map_id, band)]
            for map_id in split_maps[split]
        )
        != 32
        for split in ALLOWED_SPLITS
        for band in DEPTH_BANDS
    ):
        failures.append("exact_32_per_split_depth_band")
    if any(
        sum(
            frozen_band_counts[(split, map_id, band)] > 0
            for map_id in split_maps[split]
        )
        < 7
        for split in ALLOWED_SPLITS
        for band in DEPTH_BANDS
    ):
        failures.append("minimum_7_active_maps_per_split_depth_band")
    if max(episode_counts.values(), default=0) > episode_cap:
        failures.append("episode_cap")
    if max(map_load_counts.values(), default=0) > map_load_cap:
        failures.append("map_load_cap")
    if failures:
        raise AssertionError(f"post-selection invariant failed: {failures}")
    return frozen, {
        "status": "SELECTED_RESULT_BLIND_EXACT_PRODUCT",
        "failure_reasons": [],
        "preaction_state_count": len(rows),
        "selected_state_count": len(frozen),
        "selected_counts_by_split": dict(sorted(split_counts.items())),
        "selected_counts_by_split_map": {
            f"{split}/{map_id}": map_counts[(split, map_id)]
            for split in ALLOWED_SPLITS
            for map_id in split_maps[split]
        },
        "selected_counts_by_split_map_depth_band": {
            f"{split}/{map_id}/{band}": frozen_band_counts[(split, map_id, band)]
            for split in ALLOWED_SPLITS
            for map_id in split_maps[split]
            for band in DEPTH_BANDS
        },
        "maximum_selected_per_episode": max(episode_counts.values(), default=0),
        "maximum_selected_per_map_load": max(map_load_counts.values(), default=0),
        "supply_counts": supply_counts,
    }


def build_state_selection(
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int | None = None,
    project_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Verify the source and write a deterministic result-blind selection."""

    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    effective_workers = _effective_workers(config, workers)
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    source_root = (root / config["source"]["root"]).resolve()
    trust_path = source_root / config["source"]["collection_trust_file"]
    schedule_path = source_root / config["source"]["schedule_file"]
    trust = _read_json(trust_path)
    _validate_source_trust(trust)
    if sha256_file(schedule_path) != str(trust["source_schedule_sha256"]):
        raise ValueError("source schedule SHA256 differs from collection trust")
    schedule_rows = _read_jsonl(schedule_path)
    schedule = _validate_schedule(schedule_rows, trust)

    manifest_paths = {
        split: source_root / config["source"]["realized_manifest_by_split"][split]
        for split in ALLOWED_SPLITS
    }
    episodes: dict[str, list[dict[str, Any]]] = {}
    source_roots: dict[str, Path] = {}
    for split in ALLOWED_SPLITS:
        manifest_path = manifest_paths[split]
        source_roots[split] = manifest_path.parent.resolve()
        episodes[split] = _validate_manifest(
            split, _read_jsonl(manifest_path), schedule
        )
        for episode in episodes[split]:
            episode["source_root"] = str(source_roots[split])

    fingerprint_payload = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "source_collection_trust_sha256": sha256_file(trust_path),
        "source_schedule_sha256": sha256_file(schedule_path),
        "source_realized_manifest_sha256_by_split": {
            split: sha256_file(manifest_paths[split]) for split in ALLOWED_SPLITS
        },
        "workers": effective_workers,
        "train_map_folds": {key: list(value) for key, value in TRAIN_MAP_FOLDS.items()},
        "selection": config["selection"],
    }
    run_fingerprint = _fingerprint(fingerprint_payload)
    candidates = _selection_candidates(source_roots, episodes, effective_workers)
    selected, selection_audit = _select(candidates)
    report = {
        "schema": TRUST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": selection_audit["status"],
        "complete": bool(selected),
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_payload": fingerprint_payload,
        "workers": effective_workers,
        "source_verified": True,
        "source_collection_status": SOURCE_STATUS,
        "source_episode_count": EXPECTED_EPISODE_COUNT,
        "source_episode_counts_by_split": {
            "train": EXPECTED_EPISODES_PER_SPLIT,
            "development": EXPECTED_EPISODES_PER_SPLIT,
        },
        "source_trace_count": EXPECTED_EPISODE_COUNT,
        "source_trace_sha256_verified": True,
        "result_blind_extractor": (
            "experiments.trace_replay.result_blind_decision_rows"
        ),
        "target_outcome_fields_read": False,
        "outcome_fields_used_for_selection": [],
        "outcome_filtering": False,
        "reserve_or_replacement_backfill": False,
        "train_map_folds_fixed_before_manifest_read": {
            key: list(value) for key, value in TRAIN_MAP_FOLDS.items()
        },
        "sealed_final_status": "SEALED",
        "sealed_final_semantic_access": False,
        "sequential_design_only": True,
        "map_disjoint_final_confirmation_required": True,
        "h1_executed": False,
        "training_authorized": False,
        "model_fit_executed": False,
        "runtime_or_ttf_claim_authorized": False,
        "dry_run": bool(dry_run),
        "artifacts_written": not dry_run,
        **selection_audit,
    }
    output_root = Path(output).resolve()
    report_path = output_root / "state_selection_trust_report.json"
    selected_path = output_root / "selected_states.jsonl"
    if dry_run:
        report["selected_states_file"] = None
        report["selected_states_sha256"] = None
        report["trust_report_file"] = None
        report["trust_report_sha256"] = None
        return report
    if report_path.exists() or selected_path.exists():
        raise FileExistsError("state-selection output already exists; use a fresh output")
    if selected:
        for row in selected:
            row["selection_run_fingerprint"] = run_fingerprint
        _atomic_write(selected_path, _jsonl_bytes(selected))
        report["selected_states_file"] = str(selected_path)
        report["selected_states_sha256"] = sha256_file(selected_path)
    else:
        report["selected_states_file"] = None
        report["selected_states_sha256"] = None
    _atomic_write(report_path, _json_bytes(report))
    report["trust_report_file"] = str(report_path)
    report["trust_report_sha256"] = sha256_file(report_path)
    return report


__all__ = [
    "ALLOWED_SPLITS",
    "CONFIG_SCHEMA",
    "DEFAULT_OUTPUT_NAME",
    "DEFAULT_WORKERS",
    "DEVELOPMENT_MAPS",
    "EXPERIMENT_ID",
    "EXPECTED_EPISODE_COUNT",
    "EXPECTED_SELECTED_COUNT",
    "MAXIMUM_WORKERS",
    "SELECTED_STATE_SCHEMA",
    "TRAIN_MAP_FOLDS",
    "TRAIN_MAPS",
    "TRUST_SCHEMA",
    "build_state_selection",
    "load_config",
    "selection_rank",
]
