"""Outcome-blind compact-flow H1 action collection.

The selected pre-action states are fixed by a separate result-blind selector.
This module regenerates V2/Component16/Hotspot16 at each restored state,
canonicalizes exact agent tuples, and evaluates each unique action under the
same sixteen paired PP seeds.  It deliberately does not fit a model or make an
evaluation/runtime claim.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_collection import (
    _paired_action,
    _replay_job,
    _validate_native_repair,
)
from experiments.stride_fresh_matched_v2_c16_h16_v1 import _preflight_decision
from experiments.stride_repairability_collection import (
    _source_target_state,
    repairability_restore_seed,
)
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.unique_action_hierarchy import (
    ALL_SHARED,
    COMPONENT_ROLE,
    HOTSPOT_ROLE,
    ROLE_ORDER,
    V2_ANCHOR_ROLE,
    canonicalize_role_actions,
)


CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_h1_collection_config.v1"
SELECTED_STATE_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_h1_selected_state.v1"
SELECTION_TRUST_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_h1_state_selection_trust.v1"
)
PREFLIGHT_STATE_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_h1_preflight_state.v1"
)
PREFLIGHT_REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_h1_preflight_report.v1"
)
H1_TRIAL_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_h1_trial.v1"
H1_STATE_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_h1_state.v1"
H1_MANIFEST_ROW_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_h1_manifest_row.v1"
)
H1_COLLECTION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_h1_collection_report.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_h1_collection_v1"
DEFAULT_OUTPUT = "build/stride-hierarchical-ch-compact-flow-h1-collection-v1"
TRIAL_INDICES = tuple(range(16))
DEFAULT_WORKERS = 16
MAXIMUM_WORKERS = 20

PARTITION_FOR_LABELS = {
    "consensus_structural": "consensus_structural",
    "anchor_component_shared": "anchor_component_shared",
    "anchor_hotspot_shared": "anchor_hotspot_shared",
    "three_unique": "three_unique",
    ALL_SHARED: "single_unique",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, path)


def _contained(root: Path, value: str, *, field: str) -> Path:
    portable = PurePosixPath(str(value).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"{field} must be a contained project-relative path")
    resolved = (root / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes project root") from error
    return resolved


def _strict_h1_contract() -> dict[str, Any]:
    return {
        "trial_indices": list(TRIAL_INDICES),
        "workers": DEFAULT_WORKERS,
        "maximum_workers": MAXIMUM_WORKERS,
        "per_action_time_limit_seconds": 5.0,
        "per_state_process_fuse_seconds": 420.0,
        "same_state_trial_seed_across_unique_actions": True,
        "execute_each_unique_set_once_per_paired_seed": True,
        "failure_backfill_allowed": False,
    }


def _strict_hierarchy_contract() -> dict[str, Any]:
    return {
        "exact_agent_tuple_deduplication": True,
        "stage1": "v2_vs_each_distinct_structural_action",
        "stage2": "component_vs_hotspot_only_when_exact_sets_differ",
        "all_equal_excluded": True,
    }


def _strict_claim_boundary() -> dict[str, Any]:
    return {
        "sequential_design_only": True,
        "outcome_based_selection_allowed": False,
        "reserve_or_replacement_backfill_allowed": False,
        "training_authorized": False,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_confirmation_required": True,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("compact-flow H1 collection identity changed")
    inputs = dict(config.get("inputs") or {})
    if set(inputs) != {
        "selected_states",
        "selection_trust_report",
        "selected_state_schema",
        "selection_trust_schema",
        "controller_bundle",
    }:
        raise ValueError("compact-flow H1 input contract changed")
    if (
        inputs.get("selected_state_schema") != SELECTED_STATE_SCHEMA
        or inputs.get("selection_trust_schema") != SELECTION_TRUST_SCHEMA
    ):
        raise ValueError("compact-flow H1 selected-state schemas changed")
    controller = dict(inputs.get("controller_bundle") or {})
    if set(controller) != {"path", "manifest_sha256"}:
        raise ValueError("compact-flow H1 controller pin changed")
    digest = str(controller.get("manifest_sha256", "")).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("compact-flow H1 controller SHA256 is invalid")
    if dict(config.get("h1") or {}) != _strict_h1_contract():
        raise ValueError("compact-flow H1 paired execution contract changed")
    if dict(config.get("hierarchy") or {}) != _strict_hierarchy_contract():
        raise ValueError("compact-flow H1 hierarchy changed")
    if dict(config.get("claim_boundary") or {}) != _strict_claim_boundary():
        raise ValueError("compact-flow H1 claim boundary changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    inputs = dict(config["inputs"])
    controller = dict(inputs["controller_bundle"])
    controller_root = _contained(root, str(controller["path"]), field="controller.path")
    manifest = controller_root / "controller_manifest.json"
    if not manifest.is_file() or sha256_file(manifest) != str(
        controller["manifest_sha256"]
    ).lower():
        raise ValueError("compact-flow H1 v2-full controller bundle changed")
    return config_path, root, config


def _workers(config: Mapping[str, Any], workers: int | None) -> int:
    value = int(config["h1"]["workers"] if workers is None else workers)
    maximum = int(config["h1"]["maximum_workers"])
    if value <= 0 or value > maximum:
        raise ValueError(f"workers must be in 1..{maximum}")
    return value


def _selection_inputs(
    root: Path, config: Mapping[str, Any]
) -> tuple[Path, Path, list[dict[str, Any]], dict[str, Any]]:
    inputs = dict(config["inputs"])
    selected_path = _contained(
        root, str(inputs["selected_states"]), field="inputs.selected_states"
    )
    trust_path = _contained(
        root,
        str(inputs["selection_trust_report"]),
        field="inputs.selection_trust_report",
    )
    if not selected_path.is_file() or not trust_path.is_file():
        raise ValueError("compact-flow H1 requires frozen selected-state artifacts")
    rows = _read_jsonl(selected_path)
    trust = _read_json(trust_path)
    if (
        trust.get("schema") != SELECTION_TRUST_SCHEMA
        or trust.get("complete") is not True
        or trust.get("target_outcome_fields_read") is not False
        or trust.get("outcome_filtering") is not False
        or trust.get("reserve_or_replacement_backfill") is not False
        or trust.get("training_authorized") is not False
        or str(trust.get("selected_states_sha256", "")) != sha256_file(selected_path)
    ):
        raise ValueError("compact-flow H1 selection trust report is invalid")
    seen: set[str] = set()
    required = {
        "split",
        "map_id",
        "task_id",
        "episode_id",
        "solver_seed",
        "decision_index",
        "before_fingerprint",
        "before_conflicts",
        "source_root",
        "source_trace_file",
        "source_trace_sha256",
        "prefix_actions",
    }
    for row in rows:
        state_id = str(row.get("state_occurrence_id", ""))
        if (
            row.get("schema") != SELECTED_STATE_SCHEMA
            or not state_id
            or state_id in seen
            or any(field not in row for field in required)
            or row.get("target_outcome_fields_read") is not False
            or row.get("outcome_filtering") is not False
            or row.get("reserve_or_replacement_backfill") is not False
            or row.get("sequential_design_only") is not True
            or row.get("training_authorized") is not False
        ):
            raise ValueError("compact-flow H1 selected-state row is invalid")
        if "state_id" in row and str(row["state_id"]) != state_id:
            raise ValueError("compact-flow H1 selected state aliases disagree")
        seen.add(state_id)
    if int(trust.get("selected_state_count", -1)) != len(rows):
        raise ValueError("compact-flow H1 selected-state count changed")
    return selected_path, trust_path, rows, trust


def paired_pp_seed(state_occurrence_id: str, trial_index: int) -> int:
    if trial_index not in TRIAL_INDICES:
        raise ValueError("compact-flow H1 trial index must be in 0..15")
    return int(
        _fingerprint(
            {
                "namespace": "stride-hierarchical-ch-compact-flow-h1-paired-pp-v1",
                "state_occurrence_id": str(state_occurrence_id),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def _action_product(
    state_id: str, canonical: Any
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    actions: list[dict[str, Any]] = []
    action_ids: list[str] = []
    for action in canonical.actions:
        action_id = "compact-flow-h1-action-" + _fingerprint(
            {
                "namespace": "stride-hierarchical-ch-compact-flow-h1-action-v1",
                "state_occurrence_id": state_id,
                "agents": list(action.agents),
            }
        )[:24]
        action_ids.append(action_id)
        aliases = list(action.role_aliases)
        actions.append(
            {
                "action_id": action_id,
                "agents": list(action.agents),
                "actual_size": len(action.agents),
                "role_aliases": aliases,
                "candidate_ids_by_role": dict(action.candidate_ids_by_role),
                "contains_v2_anchor_role": V2_ANCHOR_ROLE in aliases,
                "structural_role_aliases": [
                    role for role in aliases if role in {COMPONENT_ROLE, HOTSPOT_ROLE}
                ],
            }
        )
    role_map = {
        role: action_ids[canonical.action_index(role)] for role in ROLE_ORDER
    }
    return actions, role_map


def prepare_selected_state(
    selected: Mapping[str, Any],
    *,
    bundle_root: str | Path,
    preflight_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Restore one selected state, generate V/C/H, and exact-deduplicate it."""

    state_id = str(selected["state_occurrence_id"])
    decision = dict(selected)
    decision.setdefault("state_id", state_id)
    decision.setdefault("source_policy", "v2-full")
    fn = _preflight_decision if preflight_fn is None else preflight_fn
    preflight = dict(fn(decision, bundle_root=str(bundle_root)))
    arms = dict(preflight.get("arms") or {})
    if set(arms) != set(ROLE_ORDER) or any(arms.get(role) is None for role in ROLE_ORDER):
        raise RuntimeError("compact-flow H1 preflight did not produce V2/C16/H16")
    fatal_reasons = set(map(str, preflight.get("ineligibility_reasons") or ())) - {
        "arm_agent_sets_not_pairwise_distinct"
    }
    if fatal_reasons:
        raise RuntimeError(
            f"compact-flow H1 preflight state is unusable: {sorted(fatal_reasons)}"
        )
    agent_count = int(preflight.get("agent_count", selected.get("agent_count", 0)))
    canonical = canonicalize_role_actions(
        arms, agent_count=agent_count, require_shared_candidate_id=False
    )
    if (
        len(canonical.agents_for_role(COMPONENT_ROLE)) != 16
        or len(canonical.agents_for_role(HOTSPOT_ROLE)) != 16
    ):
        raise RuntimeError("compact-flow H1 structural action is not size16")
    actions, role_map = _action_product(state_id, canonical)
    partition = PARTITION_FOR_LABELS[canonical.partition]
    all_equal = canonical.partition == ALL_SHARED
    before_conflicts = int(preflight.get("before_conflicts", -1))
    if before_conflicts <= 0:
        raise RuntimeError("compact-flow H1 selected state is conflict-free")
    return {
        **preflight,
        "schema": PREFLIGHT_STATE_SCHEMA,
        "state_occurrence_id": state_id,
        "state_id": state_id,
        "source_policy": str(decision["source_policy"]),
        "canonical_partition": partition,
        "hierarchical_stratum": partition,
        "unique_actions": actions,
        "unique_action_count": len(actions),
        "role_to_action_id": role_map,
        "stage1_structural_action_ids": sorted(
            {
                role_map[role]
                for role in (COMPONENT_ROLE, HOTSPOT_ROLE)
                if role_map[role] != role_map[V2_ANCHOR_ROLE]
            }
        ),
        "stage2_eligible": role_map[COMPONENT_ROLE] != role_map[HOTSPOT_ROLE],
        "all_equal_excluded": all_equal,
        "target_outcome_fields_read": False,
        "outcome_filtering": False,
        "candidate_repair_actions_executed": False,
        "reserve_or_replacement_backfill": False,
        "sequential_design_only": True,
        "training_authorized": False,
        "final_claim_authorized": False,
        "runtime_claim_authorized": False,
    }


def _preflight_payload_valid(
    payload: Mapping[str, Any], *, identity: str, selected: Mapping[str, Any]
) -> bool:
    row = payload.get("state_row")
    if not isinstance(row, Mapping):
        return False
    state_id = str(selected.get("state_occurrence_id", ""))
    try:
        canonical = canonicalize_role_actions(
            dict(row.get("arms") or {}),
            agent_count=int(row.get("agent_count", 0)),
            require_shared_candidate_id=False,
        )
        actions, role_map = _action_product(state_id, canonical)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        payload.get("schema") == PREFLIGHT_STATE_SCHEMA
        and payload.get("identity") == identity
        and payload.get("complete") is True
        and row.get("schema") == PREFLIGHT_STATE_SCHEMA
        and row.get("state_occurrence_id") == state_id
        and row.get("unique_actions") == actions
        and row.get("role_to_action_id") == role_map
        and row.get("canonical_partition") == PARTITION_FOR_LABELS[canonical.partition]
        and row.get("all_equal_excluded") is (canonical.partition == ALL_SHARED)
        and row.get("target_outcome_fields_read") is False
        and row.get("candidate_repair_actions_executed") is False
        and row.get("training_authorized") is False
    )


def _preflight_worker(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(job["output_path"]))
    selected = dict(job["selected"])
    identity = str(job["identity"])
    if path.exists() and bool(job["resume"]):
        payload = _read_json(path)
        if _preflight_payload_valid(payload, identity=identity, selected=selected):
            return {
                "status": "resumed",
                "job_id": str(job["job_id"]),
                "state_occurrence_id": str(selected["state_occurrence_id"]),
                "output_path": str(path),
            }
        raise ValueError(f"invalid compact-flow H1 preflight artifact: {path}")
    if path.exists():
        raise ValueError(f"compact-flow H1 preflight output exists; pass --resume: {path}")
    row = prepare_selected_state(
        selected, bundle_root=str(job["bundle_root"])
    )
    payload = {
        "schema": PREFLIGHT_STATE_SCHEMA,
        "identity": identity,
        "complete": True,
        "state_occurrence_id": str(selected["state_occurrence_id"]),
        "state_row": row,
        "target_outcome_fields_read": False,
        "candidate_repair_actions_executed": False,
        "training_authorized": False,
    }
    if not _preflight_payload_valid(payload, identity=identity, selected=selected):
        raise RuntimeError("compact-flow H1 preflight payload is invalid")
    _atomic_json(path, payload)
    return {
        "status": "ok",
        "job_id": str(job["job_id"]),
        "state_occurrence_id": str(selected["state_occurrence_id"]),
        "output_path": str(path),
    }


def _failure(job: Mapping[str, Any], status: str, message: str) -> dict[str, Any]:
    return {
        "status": status,
        "error": message,
        "job_id": str(job["job_id"]),
        "state_occurrence_id": str(
            dict(job.get("selected") or job.get("state_row") or {}).get(
                "state_occurrence_id", ""
            )
        ),
        "output_path": str(job["output_path"]),
    }


def _run_identity(
    *,
    config_sha256: str,
    selected_sha256: str,
    trust_sha256: str,
    workers: int,
    h1: Mapping[str, Any],
) -> str:
    return _fingerprint(
        {
            "namespace": "stride-hierarchical-ch-compact-flow-h1-run-v1",
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": config_sha256,
            "selected_states_sha256": selected_sha256,
            "selection_trust_sha256": trust_sha256,
            "workers": workers,
            "h1": dict(h1),
        }
    )


def run_preflight(
    config_path: str | Path,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    worker_count = _workers(config, workers)
    selected_path, trust_path, selected, _trust = _selection_inputs(root, config)
    output_root = Path(output).resolve()
    controller_root = _contained(
        root,
        str(config["inputs"]["controller_bundle"]["path"]),
        field="controller.path",
    )
    identity = _run_identity(
        config_sha256=sha256_file(path),
        selected_sha256=sha256_file(selected_path),
        trust_sha256=sha256_file(trust_path),
        workers=worker_count,
        h1=dict(config["h1"]),
    )
    jobs = [
        {
            "job_id": str(row["state_occurrence_id"]),
            "selected": row,
            "bundle_root": str(controller_root),
            "identity": identity,
            "output_path": str(
                output_root
                / "preflight_states"
                / f"{row['state_occurrence_id']}.json"
            ),
            "resume": bool(resume),
        }
        for row in selected
    ]
    if dry_run:
        return {
            "schema": PREFLIGHT_REPORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "dry_run",
            "run_identity": identity,
            "selected_state_count": len(selected),
            "preflight_job_count": len(jobs),
            "workers": worker_count,
            "workers_in_fingerprint": True,
            "candidate_repair_actions_executed": False,
            "outcome_fields_read": False,
            "backfill_allowed": False,
            "training_authorized": False,
            "final_claim_authorized": False,
            "runtime_claim_authorized": False,
        }
    results = _run_jobs(
        _preflight_worker,
        jobs,
        worker_count,
        phase="compact-flow-h1-preflight",
        output_root=output_root / "preflight_progress",
        run_fingerprint=identity,
        timeout_seconds=float(config["h1"]["per_state_process_fuse_seconds"]),
        failure_result=_failure,
    )
    result_by_id = {
        str(row.get("state_occurrence_id")): row
        for row in results
        if row.get("status") in {"ok", "resumed"}
    }
    prepared: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    invalid = 0
    for selected_row in selected:
        state_id = str(selected_row["state_occurrence_id"])
        result = result_by_id.get(state_id)
        if result is None:
            invalid += 1
            continue
        try:
            payload = _read_json(Path(str(result["output_path"])))
        except (OSError, ValueError):
            invalid += 1
            continue
        if not _preflight_payload_valid(
            payload, identity=identity, selected=selected_row
        ):
            invalid += 1
            continue
        row = dict(payload["state_row"])
        (excluded if row["all_equal_excluded"] else prepared).append(row)
    prepared.sort(key=lambda row: str(row["state_occurrence_id"]))
    excluded.sort(key=lambda row: str(row["state_occurrence_id"]))
    complete = len(results) == len(selected) and invalid == 0
    if complete:
        _write_jsonl(output_root / "preflight_state_records.jsonl", prepared)
        _write_jsonl(output_root / "excluded_all_equal_states.jsonl", excluded)
    report = {
        "schema": PREFLIGHT_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if complete else "STATE_SUPPLY_FAIL_NO_BACKFILL",
        "complete": complete,
        "run_identity": identity,
        "config_sha256": sha256_file(path),
        "selected_states_sha256": sha256_file(selected_path),
        "selection_trust_sha256": sha256_file(trust_path),
        "selected_state_count": len(selected),
        "eligible_state_count": len(prepared) if complete else 0,
        "excluded_all_equal_state_count": len(excluded) if complete else 0,
        "invalid_or_failed_state_count": invalid,
        "unique_action_count": (
            sum(int(row["unique_action_count"]) for row in prepared) if complete else 0
        ),
        "logical_trial_count": (
            sum(int(row["unique_action_count"]) for row in prepared) * 16
            if complete
            else 0
        ),
        "workers": worker_count,
        "workers_in_fingerprint": True,
        "worker_results": results,
        "candidate_repair_actions_executed": False,
        "outcome_fields_read": False,
        "backfill_allowed": False,
        "training_authorized": False,
        "final_claim_authorized": False,
        "runtime_claim_authorized": False,
    }
    _atomic_json(output_root / "preflight_report.json", report)
    return report


def _state_payload_valid(
    payload: Mapping[str, Any], *, identity: str, state_row: Mapping[str, Any]
) -> bool:
    state_id = str(state_row["state_occurrence_id"])
    actions = {
        str(row["action_id"]): dict(row) for row in state_row["unique_actions"]
    }
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return False
    product = Counter(
        (str(row.get("action_id", "")), int(row.get("trial_index", -1)))
        for row in trials
    )
    expected = {
        (action_id, trial_index)
        for action_id in actions
        for trial_index in TRIAL_INDICES
    }
    complete = payload.get("complete") is True
    observed_product = set(product)
    product_ok = (
        observed_product == expected
        if complete
        else observed_product <= expected
    )
    if not product_ok:
        return False
    if any(count != 1 for count in product.values()):
        return False
    if (
        payload.get("schema") != H1_STATE_SCHEMA
        or payload.get("identity") != identity
        or payload.get("state_occurrence_id") != state_id
        or payload.get("state_row") != state_row
        or payload.get("runtime_used_in_label") is not False
        or payload.get("training_authorized") is not False
        or payload.get("final_claim_authorized") is not False
        or payload.get("runtime_claim_authorized") is not False
    ):
        return False
    for row in trials:
        action_id = str(row.get("action_id", ""))
        trial_index = int(row.get("trial_index", -1))
        if (
            row.get("schema") != H1_TRIAL_SCHEMA
            or action_id not in actions
            or trial_index not in TRIAL_INDICES
            or int(row.get("pp_seed", -1)) != paired_pp_seed(state_id, trial_index)
            or row.get("runtime_used_in_label") is not False
            or row.get("training_authorized") is not False
            or row.get("final_claim_authorized") is not False
            or row.get("runtime_claim_authorized") is not False
            or not math.isfinite(
                float(row.get("normalized_conflict_reduction", math.nan))
            )
        ):
            return False
    for trial_index in TRIAL_INDICES:
        observed = {
            int(row["pp_seed"])
            for row in trials
            if int(row["trial_index"]) == trial_index
        }
        if complete and observed != {paired_pp_seed(state_id, trial_index)}:
            return False
    return True


def _source_snapshot(state_row: Mapping[str, Any]) -> dict[str, Any]:
    source_state, source_manifest, source_trace_path = _source_target_state(dict(state_row))
    observed_sha = sha256_file(source_trace_path)
    repair_fingerprint = repair_structure_fingerprint(source_state)
    snapshot = {
        "source_state": source_state,
        "source_trace_file": str(source_manifest["trace_file"]),
        "source_trace_path": str(source_trace_path),
        "source_trace_sha256": observed_sha,
        "before_fingerprint": state_fingerprint(source_state),
        "before_repair_fingerprint": repair_fingerprint,
        "before_conflicts": int(source_state["num_of_colliding_pairs"]),
        "before_sum_of_costs": int(source_state["sum_of_costs"]),
        "restore_seed": repairability_restore_seed(repair_fingerprint),
    }
    if (
        snapshot["source_trace_file"] != state_row.get("source_trace_file")
        or observed_sha != state_row.get("source_trace_sha256")
        or snapshot["before_fingerprint"] != state_row.get("before_fingerprint")
        or repair_fingerprint != state_row.get("before_repair_fingerprint")
        or snapshot["before_conflicts"] != int(state_row.get("before_conflicts", -1))
        or snapshot["before_conflicts"] <= 0
    ):
        raise RuntimeError("compact-flow H1 source state changed")
    return snapshot


def _state_worker(job: dict[str, Any]) -> dict[str, Any]:
    state_row = dict(job["state_row"])
    state_id = str(state_row["state_occurrence_id"])
    path = Path(str(job["output_path"]))
    partial_path = path.with_name(path.name + ".partial")
    identity = str(job["identity"])
    if state_row.get("all_equal_excluded") is not False:
        raise RuntimeError("compact-flow H1 all-equal state reached execution")
    if path.exists() and not bool(job["resume"]):
        raise ValueError(f"compact-flow H1 state output exists; pass --resume: {path}")
    if partial_path.exists() and not bool(job["resume"]):
        raise ValueError(f"compact-flow H1 partial exists; pass --resume: {partial_path}")
    snapshot = _source_snapshot(state_row)
    if path.exists() and bool(job["resume"]):
        payload = _read_json(path)
        if _state_payload_valid(payload, identity=identity, state_row=state_row):
            return {
                "status": "resumed",
                "job_id": state_id,
                "state_occurrence_id": state_id,
                "output_path": str(path),
                "trial_count": len(payload["trials"]),
            }
        raise ValueError(f"invalid compact-flow H1 state artifact: {path}")
    source_state = dict(snapshot["source_state"])
    before_repair = str(snapshot["before_repair_fingerprint"])
    before_conflicts = int(snapshot["before_conflicts"])
    before_soc = int(snapshot["before_sum_of_costs"])
    restore_seed = int(snapshot["restore_seed"])
    payload = {
        "schema": H1_STATE_SCHEMA,
        "identity": identity,
        "complete": False,
        "state_occurrence_id": state_id,
        "state_row": state_row,
        "source_trace_file": str(snapshot["source_trace_file"]),
        "source_trace_path": str(snapshot["source_trace_path"]),
        "source_trace_sha256": str(snapshot["source_trace_sha256"]),
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "before_sum_of_costs": before_soc,
        "restore_seed": restore_seed,
        "trials": [],
        "runtime_used_in_label": False,
        "sequential_design_only": True,
        "training_authorized": False,
        "final_claim_authorized": False,
        "runtime_claim_authorized": False,
    }
    if partial_path.exists() and bool(job["resume"]):
        payload = _read_json(partial_path)
        if not _state_payload_valid(payload, identity=identity, state_row=state_row):
            raise ValueError(f"invalid compact-flow H1 partial: {partial_path}")
    replay = _replay_job(state_row)
    trials = list(payload["trials"])
    completed = {
        (str(row["action_id"]), int(row["trial_index"])) for row in trials
    }
    for action in list(state_row["unique_actions"]):
        action_id = str(action["action_id"])
        agents = list(map(int, action["agents"]))
        for trial_index in TRIAL_INDICES:
            if (action_id, trial_index) in completed:
                continue
            environment, branch = restore_repair_state(
                replay, source_state, seed=restore_seed
            )
            if (
                repair_structure_fingerprint(branch) != before_repair
                or int(branch["num_of_colliding_pairs"]) != before_conflicts
                or int(branch["sum_of_costs"]) != before_soc
            ):
                raise RuntimeError("compact-flow H1 branch restore changed")
            pp_seed = paired_pp_seed(state_id, trial_index)
            result = _plain(
                environment.step_with_time_limit(
                    _paired_action(agents, pp_seed),
                    float(job["per_action_time_limit_seconds"]),
                )
            )
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            after_repair = repair_structure_fingerprint(after)
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_soc = int(after["sum_of_costs"])
            success = bool(metrics["replan_success"])
            rolled_back = bool(metrics.get("pp_rolled_back", False))
            failure_reason = str(metrics.get("pp_failure_reason", ""))
            atomic_rollback = bool(
                rolled_back
                and after_repair == before_repair
                and conflicts_after == before_conflicts
                and after_soc == before_soc
            )
            if not success and not atomic_rollback:
                raise RuntimeError("compact-flow H1 PP failure was not atomic")
            if success and conflicts_after > before_conflicts:
                raise RuntimeError("compact-flow H1 successful PP increased conflicts")
            repair_order = metrics.get("repair_order")
            if not isinstance(repair_order, list):
                raise RuntimeError("compact-flow H1 PP repair order is missing")
            trial = {
                "schema": H1_TRIAL_SCHEMA,
                "trial_identity": _fingerprint(
                    {
                        "namespace": "stride-hierarchical-ch-compact-flow-h1-trial-v1",
                        "state_occurrence_id": state_id,
                        "action_id": action_id,
                        "trial_index": trial_index,
                    }
                ),
                "state_occurrence_id": state_id,
                "action_id": action_id,
                "role_aliases": list(action["role_aliases"]),
                "candidate_ids_by_role": dict(action["candidate_ids_by_role"]),
                "agents": agents,
                "actual_size": len(agents),
                "trial_index": trial_index,
                "step_index": 0,
                "restore_seed": restore_seed,
                "fresh_independent_environment_restore": True,
                "pp_seed": pp_seed,
                "requested_random_seed": int(metrics["requested_random_seed"]),
                "requested_pp_random_seed": int(
                    metrics.get("requested_pp_random_seed", -1)
                ),
                "applied_pp_random_seed": int(
                    metrics.get("applied_pp_random_seed", -2)
                ),
                "repair_order_count": len(repair_order),
                "before_conflicts": before_conflicts,
                "conflicts_after": conflicts_after,
                "before_sum_of_costs": before_soc,
                "after_sum_of_costs": after_soc,
                "normalized_conflict_reduction": (
                    before_conflicts - conflicts_after
                )
                / max(1, before_conflicts),
                "replan_success": success,
                "rollback": rolled_back,
                "atomic_rollback": atomic_rollback if not success else False,
                "failure_reason": failure_reason,
                "time_limit": bool(
                    failure_reason == "time_limit" and not success and atomic_rollback
                ),
                "no_progress": bool(
                    failure_reason == "time_limit"
                    or conflicts_after >= before_conflicts
                ),
                "before_repair_fingerprint": before_repair,
                "after_repair_fingerprint": after_repair,
                "native_step_seconds": float(metrics.get("native_step_seconds", 0.0)),
                "pp_seconds": float(
                    metrics.get(
                        "pp_replan_seconds", metrics.get("native_replan_seconds", 0.0)
                    )
                ),
                "requested_pp_time_limit_seconds": float(
                    job["per_action_time_limit_seconds"]
                ),
                "action_valid": True,
                "generated": True,
                "integrity_ok": True,
                "runtime_used_in_label": False,
                "sequential_design_only": True,
                "training_authorized": False,
                "final_claim_authorized": False,
                "runtime_claim_authorized": False,
            }
            trials.append(trial)
            payload["trials"] = trials
            _atomic_json(partial_path, payload)
    payload = {**payload, "complete": True, "trials": trials}
    if not _state_payload_valid(payload, identity=identity, state_row=state_row):
        raise RuntimeError("compact-flow H1 completed state payload is invalid")
    _atomic_json(path, payload)
    partial_path.unlink(missing_ok=True)
    return {
        "status": "ok",
        "job_id": state_id,
        "state_occurrence_id": state_id,
        "output_path": str(path),
        "trial_count": len(trials),
    }


def _load_frozen_preflight(
    output_root: Path,
    *,
    identity: str,
    selected_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report_path = output_root / "preflight_report.json"
    rows_path = output_root / "preflight_state_records.jsonl"
    if not report_path.is_file() or not rows_path.is_file():
        raise ValueError("compact-flow H1 collection requires completed preflight")
    report = _read_json(report_path)
    rows = _read_jsonl(rows_path)
    if (
        report.get("schema") != PREFLIGHT_REPORT_SCHEMA
        or report.get("complete") is not True
        or report.get("status") != "complete"
        or report.get("run_identity") != identity
        or int(report.get("selected_state_count", -1)) != selected_count
        or int(report.get("eligible_state_count", -1)) != len(rows)
        or report.get("candidate_repair_actions_executed") is not False
        or report.get("outcome_fields_read") is not False
        or report.get("backfill_allowed") is not False
    ):
        raise ValueError("compact-flow H1 preflight trust product changed")
    state_ids = [str(row.get("state_occurrence_id", "")) for row in rows]
    if not all(state_ids) or len(state_ids) != len(set(state_ids)):
        raise ValueError("compact-flow H1 preflight states are invalid")
    return report, rows


def _state_id_inventory_complete(
    expected_state_ids: list[str], manifest_state_ids: list[str]
) -> bool:
    expected_ids = Counter(expected_state_ids)
    manifest_ids = Counter(manifest_state_ids)
    return bool(
        len(expected_ids) == len(expected_state_ids)
        and len(manifest_ids) == len(manifest_state_ids)
        and manifest_ids == expected_ids
    )


def run_collection(
    config_path: str | Path,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    worker_count = _workers(config, workers)
    selected_path, trust_path, selected, _trust = _selection_inputs(root, config)
    output_root = Path(output).resolve()
    identity = _run_identity(
        config_sha256=sha256_file(path),
        selected_sha256=sha256_file(selected_path),
        trust_sha256=sha256_file(trust_path),
        workers=worker_count,
        h1=dict(config["h1"]),
    )
    preflight_report, states = _load_frozen_preflight(
        output_root, identity=identity, selected_count=len(selected)
    )
    expected_action_count = sum(int(row["unique_action_count"]) for row in states)
    expected_trial_count = expected_action_count * len(TRIAL_INDICES)
    jobs = [
        {
            "job_id": str(row["state_occurrence_id"]),
            "state_row": row,
            "identity": identity,
            "per_action_time_limit_seconds": float(
                config["h1"]["per_action_time_limit_seconds"]
            ),
            "output_path": str(
                output_root / "h1_states" / f"{row['state_occurrence_id']}.json"
            ),
            "resume": bool(resume),
        }
        for row in states
    ]
    if dry_run:
        return {
            "schema": H1_COLLECTION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "dry_run",
            "run_identity": identity,
            "state_job_count": len(jobs),
            "unique_action_count": expected_action_count,
            "logical_trial_count": expected_trial_count,
            "workers": worker_count,
            "workers_in_fingerprint": True,
            "collection_executed": False,
            "backfill_allowed": False,
            "training_authorized": False,
            "final_claim_authorized": False,
            "runtime_claim_authorized": False,
        }
    results = _run_jobs(
        _state_worker,
        jobs,
        worker_count,
        phase="compact-flow-h1-collection",
        output_root=output_root / "collection_progress",
        run_fingerprint=identity,
        timeout_seconds=float(config["h1"]["per_state_process_fuse_seconds"]),
        failure_result=_failure,
    )
    state_by_id = {str(row["state_occurrence_id"]): row for row in states}
    manifest: list[dict[str, Any]] = []
    raw_outcomes: list[dict[str, Any]] = []
    invalid = 0
    for result in results:
        if result.get("status") not in {"ok", "resumed"}:
            continue
        state_id = str(result.get("state_occurrence_id", ""))
        state_row = state_by_id.get(state_id)
        if state_row is None:
            invalid += 1
            continue
        try:
            state_path = Path(str(result["output_path"]))
            payload = _read_json(state_path)
        except (OSError, ValueError):
            invalid += 1
            continue
        if not _state_payload_valid(payload, identity=identity, state_row=state_row):
            invalid += 1
            continue
        trials = list(payload["trials"])
        raw_outcomes.extend(trials)
        manifest.append(
            {
                "schema": H1_MANIFEST_ROW_SCHEMA,
                "state_occurrence_id": state_id,
                "state_file": str(state_path.relative_to(output_root)).replace("\\", "/"),
                "state_sha256": sha256_file(state_path),
                "unique_action_count": int(state_row["unique_action_count"]),
                "trial_count": len(trials),
                "sequential_design_only": True,
                "training_authorized": False,
                "final_claim_authorized": False,
                "runtime_claim_authorized": False,
            }
        )
    manifest.sort(key=lambda row: str(row["state_occurrence_id"]))
    raw_outcomes.sort(
        key=lambda row: (
            str(row["state_occurrence_id"]),
            str(row["action_id"]),
            int(row["trial_index"]),
        )
    )
    expected_state_ids = [str(row["state_occurrence_id"]) for row in states]
    manifest_state_ids = [str(row["state_occurrence_id"]) for row in manifest]
    state_ids_complete = _state_id_inventory_complete(
        expected_state_ids, manifest_state_ids
    )
    complete = bool(
        len(results) == len(states)
        and invalid == 0
        and state_ids_complete
        and sum(int(row["unique_action_count"]) for row in manifest)
        == expected_action_count
        and len(raw_outcomes) == expected_trial_count
    )
    manifest_path = output_root / "h1_state_manifest.jsonl"
    raw_path = output_root / "raw_unique_action_outcomes.jsonl"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(raw_path, raw_outcomes)
    report = {
        "schema": H1_COLLECTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if complete else "INTEGRITY_FAIL_NO_BACKFILL",
        "complete": complete,
        "run_identity": identity,
        "config_sha256": sha256_file(path),
        "selected_states_sha256": sha256_file(selected_path),
        "selection_trust_sha256": sha256_file(trust_path),
        "preflight_report_sha256": sha256_file(output_root / "preflight_report.json"),
        "selected_state_count": len(selected),
        "excluded_all_equal_state_count": int(
            preflight_report["excluded_all_equal_state_count"]
        ),
        "expected_state_count": len(states),
        "observed_valid_state_count": len(manifest),
        "expected_unique_action_count": expected_action_count,
        "observed_unique_action_count": sum(
            int(row["unique_action_count"]) for row in manifest
        ),
        "expected_trial_count": expected_trial_count,
        "observed_trial_count": len(raw_outcomes),
        "invalid_state_count": invalid,
        "workers": worker_count,
        "workers_in_fingerprint": True,
        "worker_results": results,
        "artifacts": {
            "label_source_state_manifest": {
                "file": manifest_path.name,
                "sha256": sha256_file(manifest_path),
                "state_schema": H1_STATE_SCHEMA,
            },
            "raw_unique_action_outcomes": {
                "file": raw_path.name,
                "sha256": sha256_file(raw_path),
            },
        },
        "exact_agent_tuple_deduplication": True,
        "stage1_population": "v2_vs_each_distinct_structural_action",
        "stage2_population": "component_vs_hotspot_only_when_exact_sets_differ",
        "all_equal_excluded": True,
        "outcome_based_selection": False,
        "backfill_allowed": False,
        "sequential_design_only": True,
        "training_authorized": False,
        "model_fit_executed": False,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_confirmation_required": True,
    }
    _atomic_json(output_root / "h1_collection_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_OUTPUT",
    "EXPERIMENT_ID",
    "H1_COLLECTION_SCHEMA",
    "H1_MANIFEST_ROW_SCHEMA",
    "H1_STATE_SCHEMA",
    "H1_TRIAL_SCHEMA",
    "MAXIMUM_WORKERS",
    "PREFLIGHT_REPORT_SCHEMA",
    "PREFLIGHT_STATE_SCHEMA",
    "SELECTED_STATE_SCHEMA",
    "SELECTION_TRUST_SCHEMA",
    "TRIAL_INDICES",
    "load_config",
    "paired_pp_seed",
    "prepare_selected_state",
    "run_collection",
    "run_preflight",
    "validate_config",
]
