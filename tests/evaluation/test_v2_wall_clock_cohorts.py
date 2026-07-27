from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from experiments import v2_wall_clock_cohorts as cohorts
from experiments.v2_wall_clock_cohorts import (
    CONTROLLERS,
    EXPECTED_TASKS,
    generate_v2_wall_clock_cohort_report,
)
from experiments.tradeoff_evaluation import _manifest_path as controller_manifest_path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_source(
    root: Path,
    tasks: tuple[str, ...],
    *,
    extra_controller: bool = False,
) -> None:
    episode_count = len(tasks) * len(CONTROLLERS) * 3
    _write_json(
        root / "status.json",
        {
            "status": "complete",
            "phase": "complete",
            "episode_count": episode_count,
            "bottleneck_validation": {"error_episode_count": 0},
        },
    )
    _write_json(
        root / "collection_progress.json",
        {
            "status": "complete",
            "phase": "complete",
            "error_jobs": 0,
        },
    )
    runner_controllers = [*CONTROLLERS]
    if extra_controller:
        runner_controllers.append("v2-critical")
    _write_json(
        root / "runner_config.json",
        {
            "identity_fingerprint": "runner-identity",
            "identity": {
                "controllers": runner_controllers,
                "controller_runtime": "optimized",
                "feature_backend": "native",
                "verification_profile": "deployment",
                "paired_execution": "strict",
                "wall_clock_seconds": 600.0,
                "tracks": ["wall-clock"],
                "model_semantic_fingerprint": "model-semantic",
                "dataset": "/dataset",
                "controller_bundle": "/bundle",
                "task_ids": list(tasks),
            },
        },
    )
    schedule = root / "tracks" / cohorts.TRACK / "execution_schedule.json"
    _write_json(schedule, {"schema": "lns2.controller_execution_schedule.v1"})
    for controller in CONTROLLERS:
        collection = (
            root
            / "tracks"
            / cohorts.TRACK
            / "collections"
            / controller
        )
        _write_json(
            collection / "run_config.json",
            {
                "controller": cohorts.RUN_CONFIG_CONTROLLERS[controller],
                "run_fingerprint": f"run-{root.name}-{controller}",
                "dataset_fingerprint": "dataset-fingerprint",
                "controller_implementation": {
                    "sha256": "controller-implementation",
                    "native_module": {"sha256": "native-module"},
                },
                "controller_bundle": {"semantic": "frozen-v2"},
                "configuration": {
                    "stopping_rule": "wall-clock",
                    "controller_runtime": "optimized",
                    "feature_backend": "native",
                    "verification_profile": "deployment",
                    "wall_time_budget_seconds": 600.0,
                    "environment": {
                        "replan_algorithm": "PP",
                        "use_sipp": True,
                        "max_repair_iterations": 0,
                        "time_limit": 600.0,
                    },
                },
            },
        )
        controller_manifest_path(collection, controller).write_text(
            "{}\n", encoding="utf-8"
        )


def _episode(task_id: str, seed: int, controller: str) -> dict[str, Any]:
    task_index = EXPECTED_TASKS.index(task_id)
    initial_conflicts = (0, 2, 5)[task_index % 3]
    repairable = initial_conflicts > 0
    lns2 = controller == "official_adaptive"
    capped_ttf = 0.1 if not repairable else (10.0 if lns2 else 8.0)
    repairs = 0 if not repairable else (5 if lns2 else 4)
    return {
        "track": cohorts.TRACK,
        "controller": controller,
        "task_id": task_id,
        "map_id": task_id.split("__", 1)[0],
        "layout_family": task_id.split("-", 1)[0],
        "agent_count": int(task_id.rsplit("_", 1)[-1]),
        "solver_seed": seed,
        "status": "ok",
        "initial_fingerprint": f"initial-{task_id}-{seed}",
        "initial_conflicts": initial_conflicts,
        "repairable": repairable,
        "success": True,
        "stopping_rule": "wall-clock",
        "wall_time_budget_seconds": 600.0,
        "restricted_time_to_feasible": capped_ttf,
        "normalized_wall_clock_conflict_auc": (
            0.0 if not repairable else (0.10 if lns2 else 0.09)
        ),
        "budget_final_sum_of_costs": 100,
        "environment_construct_seconds": 0.01,
        "reset_wall_seconds": 0.02,
        "neighborhood_selection_seconds": 0.1 if lns2 else 0.4,
        "pp_replan_seconds": 4.0 if lns2 else 3.0,
        "repair_iterations": repairs,
        "no_improvement_repair_count": 0,
        "timing_instrumentation_complete": True,
        "finalization_timing_instrumented": True,
        "process_timing_closure_error_seconds": 0.0,
        "episode_process_wall_seconds": capped_ttf,
    }


def _install_loader(
    monkeypatch: pytest.MonkeyPatch,
    source_tasks: dict[Path, tuple[str, ...]],
    *,
    mutate: Any = None,
) -> list[set[str]]:
    calls: list[set[str]] = []

    def fake_load(track: str, roots: dict[str, Path]):
        assert track == cohorts.TRACK
        calls.append(set(roots))
        source = next(iter(roots.values())).parents[3]
        rows = [
            _episode(task, seed, controller)
            for task in source_tasks[source]
            for seed in (1, 2, 3)
            for controller in CONTROLLERS
        ]
        if mutate is not None:
            rows = mutate(source, rows)
        return rows, [], {}

    monkeypatch.setattr(cohorts, "load_track", fake_load)
    return calls


def _complete_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = (tmp_path / "base").resolve()
    supplement = (tmp_path / "supplement").resolve()
    _make_source(base, EXPECTED_TASKS[:4], extra_controller=True)
    _make_source(supplement, EXPECTED_TASKS[4:])
    calls = _install_loader(
        monkeypatch,
        {base: EXPECTED_TASKS[:4], supplement: EXPECTED_TASKS[4:]},
    )
    return base, supplement, calls


def test_generates_complete_report_and_filters_extra_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, supplement, calls = _complete_fixture(tmp_path, monkeypatch)
    output = tmp_path / "report"
    report = generate_v2_wall_clock_cohort_report(
        [base, supplement], output
    )
    assert report["validation"]["passed"] is True
    assert report["episode_count"] == 42
    assert report["paired_episode_count"] == 21
    assert report["decision"] == "v2_wall_clock_supported"
    assert calls == [set(CONTROLLERS), set(CONTROLLERS)]
    assert (output / "paired_episodes.csv").is_file()
    assert (output / "controller_summary.csv").is_file()
    assert (output / "per_map_results.csv").is_file()
    assert (output / "conflict_strata.csv").is_file()
    assert (output / "v2_lns2_seven_map_report.json").is_file()
    assert (output / "v2_lns2_seven_map_report.md").is_file()
    with (output / "per_map_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        per_map = list(csv.DictReader(stream))
    assert len(per_map) == 7
    assert float(per_map[0]["lns2_mean_repair_iterations"]) >= 0.0
    assert float(per_map[0]["v2_mean_selection_seconds"]) >= 0.0
    assert float(per_map[0]["v2_mean_pp_seconds"]) >= 0.0
    with (output / "conflict_strata.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        strata = {row["group"]: row for row in csv.DictReader(stream)}
    initially_feasible = strata["initially_feasible"]
    assert initially_feasible["lns2_mean_capped_ttf"] == ""
    assert float(initially_feasible["lns2_mean_initialization_seconds"]) > 0.0


def test_rejects_overlapping_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, supplement, _calls = _complete_fixture(tmp_path, monkeypatch)

    def overlap(source: Path, rows: list[dict[str, Any]]):
        if source == supplement:
            rows[0] = _episode(EXPECTED_TASKS[0], 1, "official_adaptive")
        return rows

    _install_loader(
        monkeypatch,
        {base: EXPECTED_TASKS[:4], supplement: EXPECTED_TASKS[4:]},
        mutate=overlap,
    )
    with pytest.raises(ValueError, match="overlap"):
        generate_v2_wall_clock_cohort_report(
            [base, supplement], tmp_path / "report"
        )


def test_rejects_missing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, supplement, _calls = _complete_fixture(tmp_path, monkeypatch)

    def remove_pair(source: Path, rows: list[dict[str, Any]]):
        return rows[:-1] if source == supplement else rows

    _install_loader(
        monkeypatch,
        {base: EXPECTED_TASKS[:4], supplement: EXPECTED_TASKS[4:]},
        mutate=remove_pair,
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        generate_v2_wall_clock_cohort_report(
            [base, supplement], tmp_path / "report"
        )


@pytest.mark.parametrize(
    ("relative", "field", "value", "message"),
    (
        (
            "tracks/wall-clock-600/collections/v2-full/run_config.json",
            "dataset_fingerprint",
            "different-dataset",
            "dataset_fingerprint mismatch",
        ),
        (
            "tracks/wall-clock-600/collections/v2-full/run_config.json",
            "controller_implementation.sha256",
            "different-implementation",
            "controller_implementation_sha256 mismatch",
        ),
        (
            "tracks/wall-clock-600/collections/v2-full/run_config.json",
            "configuration.wall_time_budget_seconds",
            300.0,
            "runtime/PP configuration mismatch",
        ),
    ),
)
def test_rejects_cross_source_identity_or_budget_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    field: str,
    value: Any,
    message: str,
) -> None:
    base, supplement, _calls = _complete_fixture(tmp_path, monkeypatch)
    path = supplement / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    target = document
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    _write_json(path, document)
    with pytest.raises(ValueError, match=message):
        generate_v2_wall_clock_cohort_report(
            [base, supplement], tmp_path / "report"
        )


def test_propagates_corrupt_trace_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, supplement, _calls = _complete_fixture(tmp_path, monkeypatch)

    def corrupt_trace(_track: str, _roots: dict[str, Path]):
        raise ValueError("trace sha256 mismatch")

    monkeypatch.setattr(cohorts, "load_track", corrupt_trace)
    with pytest.raises(ValueError, match="trace sha256 mismatch"):
        generate_v2_wall_clock_cohort_report(
            [base, supplement], tmp_path / "report"
        )


def test_refuses_nonempty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, supplement, _calls = _complete_fixture(tmp_path, monkeypatch)
    output = tmp_path / "report"
    output.mkdir()
    (output / "old.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        generate_v2_wall_clock_cohort_report([base, supplement], output)
    assert (output / "old.txt").read_text(encoding="utf-8") == "preserve"
