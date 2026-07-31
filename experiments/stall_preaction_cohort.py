from __future__ import annotations

import collections
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file, strict_int
from experiments.lns2_bottleneck import validate_manifest_trace
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.run_output_guard import prepare_run_output
from experiments.stall_escape_qualification import normalize_episode_decisions
from experiments.stall_shadow import neighborhood_key
from experiments.stall_trigger_policy_audit import wilson_upper
from experiments.trace_replay import decision_rows


STALL_PREACTION_COHORT_SCHEMA = "lns2.stall_preaction_cohort.v1"
STALL_PREACTION_COHORT_VERSION = 1
NO_PROGRESS_CLASSES = {
    "single_no_progress_recovery",
    "multi_no_progress_recovery",
    "long_no_progress_recovery",
    "confirmed_long_stall",
    "unresolved_no_progress",
}
TRAINABLE_OBSERVATIONAL_CLASSES = {
    "immediate_progress",
    "state_changed_no_reduction",
    "single_no_progress_recovery",
    "multi_no_progress_recovery",
    "long_no_progress_recovery",
    "confirmed_long_stall",
}


def _stable_key(namespace: str, seed: int, *values: Any) -> str:
    payload = ":".join([namespace, str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_preaction_decisions(
    decisions: list[dict[str, Any]],
    *,
    long_stall_threshold: int = 9,
    future_observation_decisions: int = 3,
) -> list[dict[str, Any]]:
    """Classify one independent pre-action state per unchanged-state run.

    The class is a retrospective sampling stratum, never an online feature or a
    selector-failure label.  A selector-failure target is only available after
    paired same-state counterfactual trials of rank 1 and its alternatives.
    """

    threshold = strict_int(
        long_stall_threshold, field="long stall threshold", minimum=1
    )
    future = strict_int(
        future_observation_decisions,
        field="future observation decisions",
        minimum=1,
    )
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(decisions):
        decision = decisions[index]
        if not bool(decision.get("no_progress")):
            outcome = str(decision.get("repair_outcome") or "")
            if outcome == "state_changed_no_reduction":
                observational_class = "state_changed_no_reduction"
            elif outcome in {"conflict_reduced", "feasible"}:
                observational_class = "immediate_progress"
            else:
                raise ValueError(
                    f"unexpected progress repair outcome: {outcome or '<missing>'}"
                )
            rows.append(
                {
                    "anchor": decision,
                    "observational_class": observational_class,
                    "no_progress_run_length": 0,
                    "eventual_state_change": True,
                    "recovery_delay_decisions": 0,
                    "future_window_complete": True,
                    "run_outcomes": [],
                }
            )
            index += 1
            continue

        state_anchor = str(decision.get("before_repair_fingerprint") or "")
        if not state_anchor:
            raise ValueError("no-progress decision is missing its repair fingerprint")
        start = index
        while (
            index < len(decisions)
            and bool(decisions[index].get("no_progress"))
            and str(decisions[index].get("before_repair_fingerprint") or "")
            == state_anchor
        ):
            index += 1
        run = decisions[start:index]
        following = decisions[index] if index < len(decisions) else None
        recovered = bool(
            following is not None
            and not bool(following.get("no_progress"))
            and str(following.get("before_repair_fingerprint") or "") == state_anchor
        )
        run_length = len(run)
        if recovered and run_length == 1:
            observational_class = "single_no_progress_recovery"
        elif recovered and run_length < threshold:
            observational_class = "multi_no_progress_recovery"
        elif recovered:
            observational_class = "long_no_progress_recovery"
        elif run_length >= threshold + future:
            observational_class = "confirmed_long_stall"
        else:
            observational_class = "unresolved_no_progress"
        rows.append(
            {
                "anchor": run[0],
                "observational_class": observational_class,
                "no_progress_run_length": run_length,
                "eventual_state_change": recovered,
                "recovery_delay_decisions": run_length if recovered else None,
                "future_window_complete": bool(
                    recovered or run_length >= threshold + future
                ),
                "run_outcomes": [str(row["repair_outcome"]) for row in run],
            }
        )
    return rows


def assign_map_folds(
    map_ids: Iterable[str], *, master_seed: int, fold_count: int = 4
) -> dict[str, int]:
    folds = strict_int(fold_count, field="map fold count", minimum=1)
    unique = sorted({str(value) for value in map_ids if str(value)})
    if len(unique) < folds:
        raise ValueError("pre-action cohort has fewer maps than requested folds")
    ordered = sorted(
        unique,
        key=lambda value: _stable_key("stall-preaction-map-fold-v1", master_seed, value),
    )
    return {map_id: index % folds for index, map_id in enumerate(ordered)}


def outcome_blind_map_sample(
    rows: Iterable[dict[str, Any]],
    count: int,
    *,
    master_seed: int,
    excluded_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a map-balanced control without inspecting future outcome class."""

    target = strict_int(
        count, field="outcome-blind control count", minimum=1
    )
    excluded = set(excluded_keys or ())
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        key = str(row.get("state_key") or "")
        map_id = str(row.get("map_id") or "")
        if not key or not map_id or key in excluded:
            continue
        grouped[map_id].append(dict(row))
    for map_id, values in grouped.items():
        values.sort(
            key=lambda row: _stable_key(
                "stall-preaction-control-v1", master_seed, map_id, row["state_key"]
            )
        )
    map_order = sorted(
        grouped,
        key=lambda map_id: _stable_key(
            "stall-preaction-control-map-v1", master_seed, map_id
        ),
    )
    selected: list[dict[str, Any]] = []
    position = 0
    while len(selected) < target and map_order:
        map_id = map_order[position % len(map_order)]
        if grouped[map_id]:
            selected.append(grouped[map_id].pop(0))
        if not grouped[map_id]:
            map_order.remove(map_id)
            position = 0
        else:
            position += 1
    return selected


def _enriched_sample(
    rows: list[dict[str, Any]],
    targets: dict[str, int],
    *,
    master_seed: int,
    excluded_keys: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for observational_class, raw_count in sorted(targets.items()):
        if observational_class not in TRAINABLE_OBSERVATIONAL_CLASSES:
            raise ValueError(
                f"unsupported enriched observational class: {observational_class}"
            )
        count = strict_int(
            raw_count,
            field=f"enriched target {observational_class}",
            minimum=1,
        )
        candidates = [
            row
            for row in rows
            if str(row["observational_class"]) == observational_class
        ]
        chosen = outcome_blind_map_sample(
            candidates,
            count,
            master_seed=int(
                _stable_key(
                    "stall-preaction-enriched-v1", master_seed, observational_class
                )[:12],
                16,
            ),
            excluded_keys=excluded_keys,
        )
        selected.extend(chosen)
        excluded_keys.update(str(row["state_key"]) for row in chosen)
    return selected


def _zero_failure_requirement(maximum_rate: float) -> int:
    if not math.isfinite(maximum_rate) or not 0.0 < maximum_rate < 1.0:
        raise ValueError("maximum false-trigger rate must be between zero and one")
    count = 1
    while float(wilson_upper(0, count)) > maximum_rate:
        count += 1
    return count


def _candidate_summary(
    anchor: dict[str, Any], *, candidate_rank_limit: int
) -> list[dict[str, Any]]:
    pool = list(anchor.get("candidate_pool") or ())
    selected_id = str(anchor.get("selected_candidate_id") or "")
    if len(pool) < candidate_rank_limit:
        raise ValueError("pre-action state has too few frozen v2 candidates")
    if int(pool[0].get("rank", -1)) != 1 or str(pool[0].get("candidate_id")) != selected_id:
        raise ValueError("pre-action v2 selected action is not frozen rank 1")
    rows: list[dict[str, Any]] = []
    for candidate in pool[:candidate_rank_limit]:
        agents = list(map(int, candidate["agents"]))
        rows.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "rank": int(candidate["rank"]),
                "actual_size": int(candidate["actual_size"]),
                "score": float(candidate["score"]),
                "neighborhood_key": neighborhood_key(agents),
            }
        )
    return rows


def _source_inventory(
    source_root: Path,
    *,
    long_stall_threshold: int,
    future_observation_decisions: int,
    candidate_rank_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_path = source_root / "run_config.json"
    manifest_path = source_root / "realized_dynamic_manifest.jsonl"
    run = _read_json(run_path)
    configuration = dict(run.get("configuration") or {})
    dataset_text = str(run.get("dataset") or "").lower()
    if (
        str(run.get("controller")) != "v2-full"
        or str(configuration.get("split")) != "policy_train"
        or configuration.get("formal") is not False
        or any(token in dataset_text for token in ("movingai", "ood", "formal"))
    ):
        raise ValueError(
            "pre-action source must be non-formal synthetic policy_train v2-full"
        )
    manifests = [
        dict(row)
        for row in _read_jsonl(manifest_path)
        if str(row.get("status")) in {"ok", "resumed"}
    ]
    if not manifests:
        raise ValueError("pre-action source has no complete v2 manifests")
    result: list[dict[str, Any]] = []
    for manifest in manifests:
        validate_manifest_trace(
            source_root,
            manifest,
            run_fingerprint=str(run["run_fingerprint"]),
            expected_policy="realized_dynamic",
        )
        raw_decisions, events = decision_rows(source_root, manifest)
        decisions = normalize_episode_decisions(raw_decisions, events)
        classes = classify_preaction_decisions(
            decisions,
            long_stall_threshold=long_stall_threshold,
            future_observation_decisions=future_observation_decisions,
        )
        for classified in classes:
            anchor = dict(classified["anchor"])
            repair_fingerprint = str(anchor["before_repair_fingerprint"])
            state_key = _fingerprint(
                {
                    "map_id": str(manifest["map_id"]),
                    "repair_fingerprint": repair_fingerprint,
                }
            )
            candidate_rows = _candidate_summary(
                anchor, candidate_rank_limit=candidate_rank_limit
            )
            result.append(
                {
                    "state_key": state_key,
                    "source": str(source_root),
                    "source_run_fingerprint": str(run["run_fingerprint"]),
                    "source_trace_sha256": str(manifest["trace_sha256"]),
                    "split": "policy_train",
                    "map_id": str(manifest["map_id"]),
                    "layout_mode": str(manifest.get("layout_mode", "unknown")),
                    "agent_count": int(manifest.get("agent_count", 0)),
                    "task_id": str(manifest["task_id"]),
                    "solver_seed": int(manifest["solver_seed"]),
                    "decision_index": int(anchor["decision_index"]),
                    "before_fingerprint": str(anchor["before_fingerprint"]),
                    "before_repair_fingerprint": repair_fingerprint,
                    "observational_class": str(
                        classified["observational_class"]
                    ),
                    "no_progress_run_length": int(
                        classified["no_progress_run_length"]
                    ),
                    "eventual_state_change": bool(
                        classified["eventual_state_change"]
                    ),
                    "recovery_delay_decisions": classified[
                        "recovery_delay_decisions"
                    ],
                    "future_window_complete": bool(
                        classified["future_window_complete"]
                    ),
                    "trajectory_lookahead_is_label_only": True,
                    "candidate_rank_limit": candidate_rank_limit,
                    "candidate_rows": candidate_rows,
                    "unique_neighborhood_count": len(
                        {row["neighborhood_key"] for row in candidate_rows}
                    ),
                    "v2_rank1_candidate_id": candidate_rows[0]["candidate_id"],
                }
            )
    return result, {
        "source": str(source_root),
        "run_config_sha256": sha256_file(run_path),
        "manifest_sha256": sha256_file(manifest_path),
        "episode_count": len(manifests),
        "state_count": len(result),
    }


def prepare_stall_preaction_cohort(
    config: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config).resolve()
    output_root = Path(output).resolve()
    raw = _read_json(config_path)
    if str(raw.get("schema")) != STALL_PREACTION_COHORT_SCHEMA:
        raise ValueError("pre-action cohort config schema mismatch")
    master_seed = strict_int(
        raw.get("master_seed"), field="master seed", minimum=1
    )
    long_threshold = strict_int(
        raw.get("long_stall_threshold"),
        field="long stall threshold",
        minimum=1,
    )
    future = strict_int(
        raw.get("future_observation_decisions"),
        field="future observation decisions",
        minimum=1,
    )
    rank_limit = strict_int(
        raw.get("candidate_rank_limit"),
        field="candidate rank limit",
        minimum=1,
    )
    trials = strict_int(
        raw.get("paired_trials_per_candidate"),
        field="paired trials per candidate",
        minimum=1,
    )
    if trials < 4 or rank_limit < 2:
        raise ValueError("pre-action cohort requires >=4 trials and >=2 ranks")
    project_root = Path(__file__).resolve().parents[1]
    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("pre-action cohort config requires source collections")
    sources = [
        (project_root / str(value)).resolve()
        if not Path(str(value)).is_absolute()
        else Path(str(value)).resolve()
        for value in raw_sources
    ]
    identity = {
        "schema": STALL_PREACTION_COHORT_SCHEMA,
        "schema_version": STALL_PREACTION_COHORT_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_paths": list(map(str, sources)),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    prepare_run_output(output_root, resume=resume, identity=identity)

    inventory: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    episode_keys: set[tuple[str, int]] = set()
    for source in sources:
        rows, source_report = _source_inventory(
            source,
            long_stall_threshold=long_threshold,
            future_observation_decisions=future,
            candidate_rank_limit=rank_limit,
        )
        source_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in rows}
        overlap = episode_keys & source_keys
        if overlap:
            raise ValueError("pre-action sources contain duplicate task/seed episodes")
        episode_keys.update(source_keys)
        inventory.extend(rows)
        source_reports.append(source_report)
    by_state: dict[str, dict[str, Any]] = {}
    duplicate_states = 0
    for row in inventory:
        state_key = str(row["state_key"])
        if state_key in by_state:
            duplicate_states += 1
            continue
        by_state[state_key] = row
    inventory = sorted(
        by_state.values(),
        key=lambda row: (
            str(row["map_id"]),
            str(row["task_id"]),
            int(row["solver_seed"]),
            int(row["decision_index"]),
        ),
    )
    folds = assign_map_folds(
        (str(row["map_id"]) for row in inventory),
        master_seed=master_seed,
        fold_count=int(raw.get("map_fold_count", 4)),
    )
    for row in inventory:
        row["map_fold"] = folds[str(row["map_id"])]

    trainable = [
        row
        for row in inventory
        if str(row["observational_class"]) in TRAINABLE_OBSERVATIONAL_CLASSES
    ]
    control = outcome_blind_map_sample(
        inventory,
        strict_int(
            raw.get("outcome_blind_control_count"),
            field="outcome-blind control count",
            minimum=1,
        ),
        master_seed=master_seed,
    )
    selected_keys = {str(row["state_key"]) for row in control}
    enriched_raw = raw.get("enriched_training_targets")
    if not isinstance(enriched_raw, dict) or not enriched_raw:
        raise ValueError("pre-action cohort requires enriched training targets")
    enriched = _enriched_sample(
        trainable,
        {str(key): value for key, value in enriched_raw.items()},
        master_seed=master_seed,
        excluded_keys=selected_keys,
    )
    for row in control:
        row["cohort_role"] = "outcome_blind_control"
    for row in enriched:
        row["cohort_role"] = "enriched_training"
    selected = control + enriched
    selected_state_keys = [str(row["state_key"]) for row in selected]
    if len(selected_state_keys) != len(set(selected_state_keys)):
        raise RuntimeError("pre-action cohort selected duplicate states")

    inventory_counts = collections.Counter(
        str(row["observational_class"]) for row in inventory
    )
    selected_counts = collections.Counter(
        (str(row["cohort_role"]), str(row["observational_class"]))
        for row in selected
    )
    index_rows = [
        {
            key: row.get(key)
            for key in (
                "state_key",
                "source",
                "map_id",
                "layout_mode",
                "agent_count",
                "task_id",
                "solver_seed",
                "decision_index",
                "before_repair_fingerprint",
                "observational_class",
                "no_progress_run_length",
                "eventual_state_change",
                "future_window_complete",
                "unique_neighborhood_count",
                "map_fold",
            )
        }
        for row in inventory
    ]
    plan = [
        {
            "source": str(row["source"]),
            "task_id": str(row["task_id"]),
            "solver_seed": int(row["solver_seed"]),
            "decision_index": int(row["decision_index"]),
            "before_repair_fingerprint": str(row["before_repair_fingerprint"]),
            "trials_per_branch": trials,
            "all_candidates": True,
        }
        for row in selected
    ]
    zero_requirement = _zero_failure_requirement(
        float(raw.get("maximum_false_trigger_rate", 0.01))
    )
    requested_enriched = {
        str(key): int(value) for key, value in enriched_raw.items()
    }
    realized_enriched = collections.Counter(
        str(row["observational_class"]) for row in enriched
    )
    unfilled_enriched = {
        key: max(0, requested - int(realized_enriched[key]))
        for key, requested in sorted(requested_enriched.items())
        if int(realized_enriched[key]) < requested
    }
    total_source_episodes = sum(
        int(row["episode_count"]) for row in source_reports
    )
    censored_controls = sum(
        str(row["observational_class"]) == "unresolved_no_progress"
        for row in control
    )
    report = {
        "schema": STALL_PREACTION_COHORT_SCHEMA,
        "schema_version": STALL_PREACTION_COHORT_VERSION,
        "complete": True,
        "evidence_level": "pre-action cohort design; no counterfactual labels yet",
        "source_reports": source_reports,
        "source_episode_count": total_source_episodes,
        "eligible_source_episode_count": len(episode_keys),
        "map_count": len(folds),
        "inventory_state_count": len(inventory),
        "duplicate_state_count": duplicate_states,
        "inventory_class_counts": dict(sorted(inventory_counts.items())),
        "outcome_blind_control_count": len(control),
        "enriched_training_count": len(enriched),
        "unfilled_enriched_targets": unfilled_enriched,
        "selected_state_count": len(selected),
        "selected_class_counts": {
            f"{role}:{label}": count
            for (role, label), count in sorted(selected_counts.items())
        },
        "map_fold_counts": dict(
            sorted(collections.Counter(folds.values()).items())
        ),
        "candidate_rank_limit": rank_limit,
        "paired_trials_per_candidate": trials,
        "oracle_plan_uses_complete_candidate_pool": True,
        "minimum_zero_false_controls_for_one_percent_wilson_gate": zero_requirement,
        "current_control_shortfall": max(0, zero_requirement - len(control)),
        "control_selection_uses_future_outcome": False,
        "outcome_blind_control_censored_count": censored_controls,
        "control_estimand": "map-balanced pre-action states",
        "enriched_selection_uses_future_outcome_strata": True,
        "counterfactual_collection_started": False,
        "training_started": False,
        "controller_actions_changed": False,
        "deployment_promoted": False,
        "decision": "cohort_ready_for_paired_oracle_smoke_only",
    }
    atomic_write_csv(output_root / "eligible_state_index.csv", index_rows)
    _write_jsonl(output_root / "selected_states.jsonl", selected)
    _write_json(output_root / "oracle_plan.json", plan)
    _write_json(output_root / "stall_preaction_cohort_report.json", report)
    lines = [
        "# Stall pre-action cohort",
        "",
        f"- Sources: `{len(sources)}`; source episodes: `{total_source_episodes}`; episodes with eligible decisions: `{len(episode_keys)}`; maps: `{len(folds)}`.",
        f"- Eligible unique states: `{len(inventory)}`; selected: `{len(selected)}`.",
        f"- Outcome-blind controls: `{len(control)}`; enriched training: `{len(enriched)}`.",
        f"- One-percent zero-error Wilson requirement: `{zero_requirement}` controls; current shortfall: `{max(0, zero_requirement - len(control))}`.",
        "- No PP counterfactual, training, action override, or promotion has started.",
        "",
        "The observational class uses future trajectory evidence only for sampling and must never enter online features. Selector-failure labels require the paired full-pool Oracle plan.",
        "",
    ]
    (output_root / "stall_preaction_cohort_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "STALL_PREACTION_COHORT_SCHEMA",
    "assign_map_folds",
    "classify_preaction_decisions",
    "outcome_blind_map_sample",
    "prepare_stall_preaction_cohort",
]
