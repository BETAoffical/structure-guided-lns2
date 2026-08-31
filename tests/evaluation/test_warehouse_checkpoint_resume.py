from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import experiments.stride_warehouse_disruption_recovery as recovery
import experiments.stride_warehouse_disruption_recovery_boundary as boundary
import experiments.stride_warehouse_disruption_recovery_heldout as heldout
import experiments.stride_warehouse_disruption_recovery_load_extension_v1 as load_extension
from experiments._common import sha256_file
from experiments.warehouse_checkpoint_resume import reusable_checkpoint_report
from experiments.warehouse_disruption_checkpoints import (
    compute_checkpoint_identity_sha256,
)


SCHEMA = "test.warehouse.checkpoint_report.v1"
EXPERIMENT_ID = "test-warehouse-checkpoints"
CONFIG_SHA256 = "a" * 64


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _schedule() -> list[dict[str, Any]]:
    return [
        {
            "key_index": index,
            "task_index": index,
            "key_id": f"job-{index}",
            "checkpoint_id": f"checkpoint-{index}",
            "map_id": f"map-{index}",
            "task_id": f"task-{index}",
            "task_variant": "balanced_od_d125",
            "split": "test",
            "agent_count": 32,
            "incumbent_solver_seed": 200 + index,
            "selection_seed": 300 + index,
            "screen_solver_seed": 100 + index,
        }
        for index in range(2)
    ]


def _report(
    rows: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> dict[str, Any]:
    qualified = [row for row in rows if row["qualification"]["passed"]]
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": CONFIG_SHA256,
        "attempted_candidate_count": len(rows) + len(failures),
        "completed_checkpoint_count": len(rows),
        "qualified_checkpoint_count": len(qualified),
        "qualified_checkpoint_ids": [row["checkpoint_id"] for row in qualified],
        "passed": len(rows) == 2 and not failures,
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    checkpoint_root = tmp_path / "checkpoints"
    schedule = _schedule()
    rows = []
    for item in schedule:
        blob = checkpoint_root / "states" / f"{item['key_index']}.bin"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(f"state-{item['key_index']}".encode())
        row = {
            **{
                key: value
                for key, value in item.items()
                if key
                not in {"task_index", "incumbent_solver_seed", "selection_seed"}
            },
            "source_kind": "checkpoint_blob_v1",
            "state_blob": f"states/{item['key_index']}.bin",
            "state_blob_sha256": sha256_file(blob),
            "qualification": {"passed": True},
            "expected_conflicts": 20 + int(item["key_index"]),
            "incumbent": {"solver_seed": item["incumbent_solver_seed"]},
            "restore_seed": item["selection_seed"],
        }
        row["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(row)
        rows.append(row)
    plan = {
        "schema": "test.warehouse.checkpoint_plan.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": CONFIG_SHA256,
        "schedule": schedule,
    }
    paths = {
        "report_path": checkpoint_root / "report.json",
        "plan_path": tmp_path / "plan.json",
        "manifest_path": checkpoint_root / "manifest.jsonl",
        "worker_path": checkpoint_root / "worker.json",
    }
    _write_json(paths["plan_path"], plan)
    _write_jsonl(paths["manifest_path"], rows)
    _write_json(paths["worker_path"], {"completed_count": 2, "failures": []})
    _write_json(paths["report_path"], _report(rows, []))
    return {**paths, "expected_plan": plan, "schedule": schedule, "rows": rows}


def _validate(fixture: dict[str, Any]) -> dict[str, Any] | None:
    return reusable_checkpoint_report(
        report_path=fixture["report_path"],
        plan_path=fixture["plan_path"],
        manifest_path=fixture["manifest_path"],
        worker_path=fixture["worker_path"],
        expected_schema=SCHEMA,
        experiment_id=EXPERIMENT_ID,
        config_sha256=CONFIG_SHA256,
        expected_plan=fixture["expected_plan"],
        schedule=fixture["schedule"],
        build_report=_report,
    )


def test_valid_complete_report_is_reused(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    assert _validate(fixture) == _report(fixture["rows"], [])


@pytest.mark.parametrize("target", ("manifest", "report"))
def test_completed_tampering_is_rejected_and_preserved(
    tmp_path: Path, target: str
) -> None:
    fixture = _fixture(tmp_path)
    original_report = fixture["report_path"].read_bytes()
    if target == "manifest":
        rows = copy.deepcopy(fixture["rows"])
        rows[0]["expected_conflicts"] += 1
        _write_jsonl(fixture["manifest_path"], rows)
    else:
        report = _report(fixture["rows"], [])
        report["qualified_checkpoint_count"] = 1
        _write_json(fixture["report_path"], report)
        original_report = fixture["report_path"].read_bytes()

    with pytest.raises(ValueError, match="changed|does not match"):
        _validate(fixture)

    assert fixture["report_path"].read_bytes() == original_report


def test_completed_failed_report_is_not_a_success_cache(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    failed = _report(fixture["rows"], [])
    failed["passed"] = False
    _write_json(fixture["report_path"], failed)
    original = fixture["report_path"].read_bytes()

    with pytest.raises(ValueError, match="did not pass"):
        _validate(fixture)

    assert fixture["report_path"].read_bytes() == original


def test_incomplete_report_requests_safe_continuation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    incomplete = _report(fixture["rows"][:1], [])
    incomplete["complete"] = False
    _write_json(fixture["report_path"], incomplete)
    _write_jsonl(fixture["manifest_path"], fixture["rows"][:1])
    _write_json(fixture["worker_path"], {"completed_count": 1, "failures": []})

    assert _validate(fixture) is None


def test_completed_report_rejects_missing_schedule_row(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_jsonl(fixture["manifest_path"], fixture["rows"][:1])
    _write_json(fixture["worker_path"], {"completed_count": 1, "failures": []})

    with pytest.raises(ValueError, match="do not cover the schedule"):
        _validate(fixture)


def test_completed_report_rejects_duplicate_schedule_row(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    duplicated = [fixture["rows"][0], fixture["rows"][0]]
    _write_jsonl(fixture["manifest_path"], duplicated)

    with pytest.raises(ValueError, match="duplicate or unknown"):
        _validate(fixture)


@pytest.mark.parametrize(
    "field",
    ("screen_solver_seed", "incumbent_solver_seed", "selection_seed"),
)
def test_completed_success_manifest_binds_every_materialized_schedule_field(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)
    rows = copy.deepcopy(fixture["rows"])
    if field == "incumbent_solver_seed":
        rows[0]["incumbent"]["solver_seed"] += 1
    elif field == "selection_seed":
        rows[0]["restore_seed"] += 1
    else:
        rows[0][field] += 1
    # Rebind both self-hash and report so this reaches the schedule contract,
    # rather than being rejected by the earlier generic tamper checks.
    rows[0]["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(
        rows[0]
    )
    _write_jsonl(fixture["manifest_path"], rows)
    _write_json(fixture["report_path"], _report(rows, []))

    with pytest.raises(
        ValueError, match=f"checkpoint manifest {field} differs from its schedule"
    ):
        _validate(fixture)


def test_completed_success_manifest_binds_schedule_only_task_index(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["schedule"][0]["task_index"] = 7
    _write_json(fixture["plan_path"], fixture["expected_plan"])

    with pytest.raises(
        ValueError, match="checkpoint manifest task_index differs from its schedule"
    ):
        _validate(fixture)


@pytest.mark.parametrize("tamper", ("change", "remove"))
def test_completed_report_binds_every_failure_identity_field(
    tmp_path: Path, tamper: str
) -> None:
    fixture = _fixture(tmp_path)
    rows = fixture["rows"][:1]
    failure = {
        **fixture["schedule"][1],
        "status": "state_supply_unavailable",
        "error": "incumbent unavailable",
    }

    def supply_report(
        current_rows: list[dict[str, Any]],
        failures: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": CONFIG_SHA256,
            "attempted_candidate_count": len(current_rows) + len(failures),
            "completed_checkpoint_count": len(current_rows),
            "failures": failures,
            "passed": len(current_rows) + len(failures) == 2,
        }

    if tamper == "change":
        failure["task_variant"] = "tampered"
    else:
        failure.pop("screen_solver_seed")
    failures = [failure]
    _write_jsonl(fixture["manifest_path"], rows)
    _write_json(
        fixture["worker_path"],
        {"completed_count": len(rows), "failures": failures},
    )
    _write_json(fixture["report_path"], supply_report(rows, failures))

    with pytest.raises(ValueError, match="worker failure .* differs"):
        reusable_checkpoint_report(
            report_path=fixture["report_path"],
            plan_path=fixture["plan_path"],
            manifest_path=fixture["manifest_path"],
            worker_path=fixture["worker_path"],
            expected_schema=SCHEMA,
            experiment_id=EXPERIMENT_ID,
            config_sha256=CONFIG_SHA256,
            expected_plan=fixture["expected_plan"],
            schedule=fixture["schedule"],
            build_report=supply_report,
        )


@pytest.mark.parametrize(
    ("subject", "schedule_name", "dataset_name", "report_schema"),
    (
        (recovery, "_screen_schedule", "_ensure_dataset", recovery.REPORT_SCHEMA),
        (heldout, "_heldout_schedule", "_ensure_dataset", heldout.REPORT_SCHEMA),
        (boundary, "_boundary_schedule", "_ensure_datasets", boundary.REPORT_SCHEMA),
        (
            load_extension,
            "_checkpoint_schedule",
            "_ensure_dataset",
            load_extension.CHECKPOINT_REPORT_SCHEMA,
        ),
    ),
)
def test_all_retained_checkpoint_entrypoints_authenticate_resume_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subject: Any,
    schedule_name: str,
    dataset_name: str,
    report_schema: str,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    schedule = _schedule()
    plan = {
        "schema": "test.plan.v1",
        "experiment_id": subject.EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "schedule": schedule,
    }
    monkeypatch.setattr(
        subject, "load_config", lambda *_args: (config_path, tmp_path, {})
    )
    monkeypatch.setattr(subject, dataset_name, lambda *_args: tmp_path / "dataset")
    monkeypatch.setattr(subject, schedule_name, lambda *_args: schedule)
    monkeypatch.setattr(subject, "plan", lambda *_args: plan)
    calls: list[dict[str, Any]] = []
    cached = {"passed": True, "authenticated": True}

    def authenticate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return cached

    monkeypatch.setattr(subject, "reusable_checkpoint_report", authenticate)
    monkeypatch.setattr(
        subject,
        "_run_jobs",
        lambda *_args, **_kwargs: pytest.fail("resume reuse executed checkpoint jobs"),
    )

    assert subject.prepare_checkpoints(config_path, tmp_path / "output", resume=True) is cached
    assert len(calls) == 1
    assert calls[0]["expected_schema"] == report_schema
    assert calls[0]["experiment_id"] == subject.EXPERIMENT_ID
    assert calls[0]["schedule"] == schedule
