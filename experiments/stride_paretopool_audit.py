from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from experiments._common import contained_file, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.state_analysis import analyze_state, analyze_static_grid
from lns2_selector.runtime.paretopool import (
    PARETOPOOL_BUDGETS,
    generate_paretopool_candidates,
)


CONFIG_SCHEMA = "lns2.stride.paretopool_gap_audit_registration.v1"
REPORT_SCHEMA = "lns2.stride.paretopool_gap_audit_report.v1"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="ParetoPool gap audit")


def load_paretopool_audit_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_candidate_gap_audit_only"
        or config.get("experiment_id") != "stride-paretopool-gap-audit-v1"
    ):
        raise ValueError("ParetoPool audit identity changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {"root_checkpoints", "candidate_aggregates"}:
        raise ValueError("ParetoPool audit input registry changed")
    if tuple(map(int, config.get("candidate_budgets") or ())) != PARETOPOOL_BUDGETS:
        raise ValueError("ParetoPool audit budgets changed")
    boundary = dict(config.get("claim_boundary") or {})
    if not all(
        boundary.get(name) is expected
        for name, expected in {
            "candidate_gap_audit_only": True,
            "model_training_allowed": False,
            "ttf_improvement_claim": False,
            "one_step_labels_are_not_future_value": True,
            "novel_candidates_are_not_scored_as_failures": True,
        }.items()
    ):
        raise ValueError("ParetoPool audit claim boundary changed")
    return path, root, config, inputs


def _base_anchor(checkpoint: dict[str, Any]) -> dict[str, Any]:
    base = [
        dict(candidate)
        for candidate in checkpoint["candidate_pool"]
        if not candidate.get("structpool_family_groups")
    ]
    if not base:
        raise ValueError("ParetoPool audit checkpoint has no V2 base candidate")
    if any(candidate.get("score") is None for candidate in base):
        raise ValueError("ParetoPool audit base candidate has no frozen V2 score")
    return max(
        base,
        key=lambda candidate: (
            float(candidate["score"]),
            str(candidate["candidate_id"]),
        ),
    )


def run_paretopool_gap_audit(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    _path, _root, config, inputs = load_paretopool_audit_config(config_path)
    checkpoints = _read_jsonl(inputs["root_checkpoints"])
    aggregates = {
        (str(row["state_fingerprint"]), str(row["candidate_id"])): dict(row)
        for row in _read_jsonl(inputs["candidate_aggregates"])
    }
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, int], Any] = {}
    state_cache: dict[str, tuple[dict[str, Any], Any, str]] = {}
    static_cache: dict[str, Any] = {}
    for checkpoint in checkpoints:
        fingerprint = str(checkpoint["state_fingerprint"])
        state_path = contained_file(
            inputs["root_checkpoints"].parent,
            checkpoint["state_blob"],
            field="ParetoPool audit state blob",
        )
        blob_sha256 = str(checkpoint["state_blob_sha256"])
        if sha256_file(state_path) != blob_sha256:
            raise ValueError("ParetoPool audit state blob hash changed")
        cached_state = state_cache.get(fingerprint)
        if cached_state is None:
            state = read_state_blob(state_path)
            map_id = str(checkpoint["map_id"])
            static_grid = static_cache.get(map_id)
            if static_grid is None:
                static_grid = analyze_static_grid(state)
                static_cache[map_id] = static_grid
            analysis = analyze_state(state, static_grid=static_grid)
            state_cache[fingerprint] = (state, analysis, blob_sha256)
        else:
            state, analysis, cached_sha256 = cached_state
            if cached_sha256 != blob_sha256:
                raise ValueError("duplicate ParetoPool state blob identity changed")
        anchor = _base_anchor(checkpoint)
        structural = [
            dict(candidate)
            for candidate in checkpoint["candidate_pool"]
            if candidate.get("structpool_family_groups")
        ]
        labeled = [
            aggregates[(fingerprint, str(candidate["candidate_id"]))]
            for candidate in structural
            if (fingerprint, str(candidate["candidate_id"])) in aggregates
        ]
        if len(labeled) != len(structural):
            raise ValueError("ParetoPool audit structural label coverage changed")
        old_best = max(
            labeled,
            key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"])),
        )
        for budget in PARETOPOOL_BUDGETS:
            cache_key = (fingerprint, str(anchor["candidate_id"]), budget)
            result = cache.get(cache_key)
            if result is None:
                result = generate_paretopool_candidates(
                    state,
                    analysis,
                    v2_anchor_agents=anchor["agents"],
                    maximum_candidates=budget,
                    maximum_neighborhood_size=int(
                        config["maximum_neighborhood_size"]
                    ),
                )
                cache[cache_key] = result
            selected_ids = {
                str(candidate["candidate_id"]) for candidate in result.candidates
            }
            known_selected = [
                row for row in labeled if str(row["candidate_id"]) in selected_ids
            ]
            known_best = (
                max(
                    known_selected,
                    key=lambda row: (
                        float(row["seed_mean"]),
                        str(row["candidate_id"]),
                    ),
                )
                if known_selected
                else None
            )
            rows.append(
                {
                    "case_id": str(checkpoint["case_id"]),
                    "state_occurrence": (
                        f"{checkpoint['case_id']}::decision_"
                        f"{int(checkpoint['decision_index']):04d}"
                    ),
                    "state_fingerprint": fingerprint,
                    "map_id": str(checkpoint["map_id"]),
                    "task_id": str(checkpoint["task_id"]),
                    "decision_index": int(checkpoint["decision_index"]),
                    "budget": budget,
                    "v2_anchor_candidate_id": str(anchor["candidate_id"]),
                    "old_structural_candidate_count": len(structural),
                    "paretopool_raw_candidate_count": result.raw_candidate_count,
                    "paretopool_selected_candidate_count": len(result.candidates),
                    "paretopool_novel_candidate_count": sum(
                        candidate_id
                        not in {str(row["candidate_id"]) for row in labeled}
                        for candidate_id in selected_ids
                    ),
                    "old_one_step_best_candidate_id": str(old_best["candidate_id"]),
                    "old_one_step_best_retained": str(old_best["candidate_id"])
                    in selected_ids,
                    "known_overlap_candidate_count": len(known_selected),
                    "known_overlap_normalized_regret": (
                        (
                            float(old_best["seed_mean"])
                            - float(known_best["seed_mean"])
                        )
                        / max(1, int(old_best["before_conflicts"]))
                        if known_best is not None
                        else None
                    ),
                }
            )
    expected_cases = int(config["cohort"]["state_occurrence_count"])
    budget_summary: dict[str, dict[str, Any]] = {}
    for budget in PARETOPOOL_BUDGETS:
        selected = [row for row in rows if int(row["budget"]) == budget]
        regrets = [
            float(row["known_overlap_normalized_regret"])
            for row in selected
            if row["known_overlap_normalized_regret"] is not None
        ]
        budget_summary[str(budget)] = {
            "state_occurrence_count": len(selected),
            "mean_raw_candidate_count": sum(
                int(row["paretopool_raw_candidate_count"]) for row in selected
            )
            / len(selected),
            "mean_selected_candidate_count": sum(
                int(row["paretopool_selected_candidate_count"]) for row in selected
            )
            / len(selected),
            "mean_novel_candidate_count": sum(
                int(row["paretopool_novel_candidate_count"]) for row in selected
            )
            / len(selected),
            "old_one_step_best_retained_fraction": sum(
                bool(row["old_one_step_best_retained"]) for row in selected
            )
            / len(selected),
            "known_overlap_state_count": len(regrets),
            "mean_known_overlap_normalized_regret": (
                sum(regrets) / len(regrets) if regrets else None
            ),
        }
    integrity = {
        "state_occurrence_count": len(checkpoints) == expected_cases,
        "complete_budget_matrix": len(rows)
        == expected_cases * len(PARETOPOOL_BUDGETS),
        "all_states_have_structural_candidates": all(
            int(row["paretopool_raw_candidate_count"]) > 0 for row in rows
        ),
        "budget_respected": all(
            int(row["paretopool_selected_candidate_count"]) <= int(row["budget"])
            for row in rows
        ),
        "v2_anchor_always_present_separately": all(
            bool(row["v2_anchor_candidate_id"]) for row in rows
        ),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_candidate_gap_audit_only",
        "state_occurrence_count": len(checkpoints),
        "unique_state_fingerprint_count": len(
            {str(row["state_fingerprint"]) for row in rows}
        ),
        "budget_summary": budget_summary,
        "rows": rows,
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "interpretation": {
            "one_step_retention_is_diagnostic_only": True,
            "novel_candidates_have_no_historical_outcome": True,
            "budget_selection_requires_multihorizon_labels": True,
        },
        "claim_boundary": dict(config["claim_boundary"]),
        "artifact_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "paretopool_gap_audit_report.json", report)
    return report


__all__ = [
    "load_paretopool_audit_config",
    "run_paretopool_gap_audit",
]
