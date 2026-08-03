from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.stride_maprank import (
    CONTROLLER_ID,
    TRAINING_REPORT_SCHEMA,
    validate_maprank_evaluation_config,
)
from experiments.stride_augcontrol_evaluation import _dataset_tasks
from experiments.stride_stage3 import _project_path


SHADOW_REPORT_SCHEMA = "lns2.stride.maprank_shadow_report.v1"


def _load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    validate_maprank_evaluation_config(config)
    return path, root, config


def _training_evidence(
    root: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = _project_path(root, str(config["training_report"]))
    report = _read_json(path)
    if (
        report.get("schema") != TRAINING_REPORT_SCHEMA
        or report.get("controller_id") != CONTROLLER_ID
        or report.get("fresh_development_eligible") is not True
        or not all(dict(report.get("promotion_gates") or {}).values())
        or not all(dict(report.get("integrity_gates") or {}).values())
        or bool(report.get("formal_ood_data_read"))
        or bool(report.get("test_data_read"))
    ):
        raise ValueError("MapRank did not pass its preregistered offline gate")
    return path, report


def _bundle_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    bundles = dict(config["controller_bundles"])
    paths = {
        "v2-full": _project_path(root, str(bundles["v2-full"])),
        CONTROLLER_ID: _project_path(root, str(bundles[CONTROLLER_ID])),
    }
    for controller, path in paths.items():
        manifest = _read_json(path / "controller_manifest.json")
        actual = str(manifest.get("controller_id", manifest.get("default_controller", "")))
        if actual != controller:
            raise ValueError(
                f"unexpected MapRank evaluation bundle identity: {controller}/{actual}"
            )
        if controller == CONTROLLER_ID and (
            manifest.get("scientific_status") != "diagnostic_only"
            or manifest.get("default_replacement_allowed") is not False
        ):
            raise ValueError("MapRank Shadow requires a diagnostic-only bundle")
    return paths


def run_maprank_shadow(
    config_path: str | Path, output: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    path, root, config = _load_config(config_path)
    training_path, _ = _training_evidence(root, config)
    bundles = _bundle_paths(root, config)
    shadow = dict(config["legacy_shadow"])
    dataset = _project_path(root, str(shadow["dataset"]))
    runtime = _project_path(root, str(shadow["runtime_config"]))
    historical_qualification = _project_path(
        root, str(shadow["qualification_source"])
    )
    if not (historical_qualification / "qualification_report.json").is_file():
        raise ValueError("MapRank historical qualification reference is incomplete")
    rows = _dataset_tasks(dataset, "balanced_wall_clock")
    maps = set(map(str, shadow["validation_maps"]))
    task_ids = sorted(
        task_id for task_id, row in rows.items() if str(row["map_id"]) in maps
    )
    if len(task_ids) != 12 or {str(rows[task]["map_id"]) for task in task_ids} != maps:
        raise ValueError("MapRank Shadow validation-map task coverage changed")
    keys = {
        (task_id, seed)
        for task_id in task_ids
        for seed in map(int, shadow["solver_seeds"])
    }
    output = Path(output).resolve()
    collection_kwargs = {
        "workers": 1,
        "cohort_job_keys": keys,
        "job_keys": keys,
        # The registered reference used a wall-clock reset protocol.  The
        # run-to-completion Shadow must qualify the same keys afresh instead
        # of bypassing the reset-protocol fingerprint check.
        "qualification_source": None,
        "controller": "v2-full",
        "controller_bundle": str(bundles["v2-full"]),
        "diagnostic_shadow_bundles": {CONTROLLER_ID: bundles[CONTROLLER_ID]},
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "topology_boundary_augmentation": dict(
            config["topology_boundary_augmentation"]
        ),
        "stopping_rule": "run-to-completion",
    }
    run_closed_loop_collection(
        dataset,
        runtime,
        output,
        phase="qualify",
        resume=resume,
        **collection_kwargs,
    )
    run_closed_loop_collection(
        dataset,
        runtime,
        output,
        phase="realized_dynamic",
        resume=True,
        **collection_kwargs,
    )
    return analyze_maprank_shadow(path, output, training_path=training_path)


def analyze_maprank_shadow(
    config_path: str | Path,
    output: str | Path,
    *,
    training_path: Path | None = None,
) -> dict[str, Any]:
    path, root, config = _load_config(config_path)
    if training_path is None:
        training_path, _ = _training_evidence(root, config)
    output = Path(output).resolve()
    manifest = output / "realized_dynamic_manifest.jsonl"
    qualification_path = output / "qualification_report.json"
    qualification = _read_json(qualification_path)
    rows = _read_jsonl(manifest)
    expected = 12 * len(config["legacy_shadow"]["solver_seeds"])
    totals = [
        dict(dict(row.get("summary") or {}).get("controller_totals") or {})
        for row in rows
    ]
    key = f"diagnostic_shadow_decision_count:{CONTROLLER_ID}"
    disagreements = f"diagnostic_shadow_disagreement_count:{CONTROLLER_ID}"
    decisions = sum(int(row.get(key, 0)) for row in totals)
    disagreement_count = sum(int(row.get(disagreements, 0)) for row in totals)
    gates = {
        "complete_episode_coverage": len(rows) == expected,
        "fresh_qualification_passed": qualification.get("passed") is True,
        "zero_execution_errors": all(row.get("status") == "ok" for row in rows),
        "minimum_shadow_decisions": decisions
        >= int(config["legacy_shadow"]["minimum_decisions"]),
        "zero_action_overrides": sum(
            int(row.get("diagnostic_shadow_action_override_count", 0))
            for row in totals
        )
        == 0,
        "zero_semantic_mismatches": sum(
            int(row.get("diagnostic_shadow_semantic_mismatch_count", 0))
            for row in totals
        )
        == 0,
        "zero_invalid_actions": sum(
            int(dict(row.get("summary") or {}).get("invalid_action_count", 0))
            for row in rows
        )
        == 0,
    }
    report = {
        "schema": SHADOW_REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "active_controller": "v2-full",
        "active_comparator": "v2-augmented-pool",
        "stopping_rule": "run-to-completion",
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "episode_count": len(rows),
        "shadow_decision_count": decisions,
        "shadow_disagreement_count": disagreement_count,
        "shadow_disagreement_fraction": (
            disagreement_count / decisions if decisions else 0.0
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "inputs": {
            "config_sha256": sha256_file(path),
            "training_report_sha256": sha256_file(training_path),
            "qualification_report_sha256": sha256_file(qualification_path),
            "manifest_sha256": sha256_file(manifest),
        },
    }
    _write_json(output / "maprank_shadow_report.json", report)
    return report


__all__ = [
    "analyze_maprank_shadow",
    "run_maprank_shadow",
]
