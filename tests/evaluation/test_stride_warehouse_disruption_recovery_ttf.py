from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import experiments.stride_warehouse_disruption_recovery_ttf as subject
from tests.evaluation.ttf_fixtures import mocked_validated_lane, synthetic_summary
from experiments.warehouse_disruption_checkpoints import (
    compute_checkpoint_identity_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_warehouse_disruption_recovery_screen_v1.json"
)


@pytest.fixture(autouse=True)
def authenticated_report_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "load_ttf_lane", mocked_validated_lane)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _checkpoint_fixture(output: Path) -> list[dict[str, Any]]:
    checkpoint_root = output / "checkpoints"
    rows: list[dict[str, Any]] = []
    for index in range(7):
        blob_relative = f"states/checkpoint_{index:02d}.json.gz"
        blob = checkpoint_root / blob_relative
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(f"checkpoint-state-{index}".encode())
        row = {
            "schema": "lns2.warehouse_disruption_checkpoint.native.v1",
            "key_index": index,
            "checkpoint_id": f"warehouse-disruption-dev-{index:02d}",
            "source_kind": "checkpoint_blob_v1",
            "state_blob": blob_relative,
            "state_blob_sha256": _sha256(blob),
            "map_id": f"development_station_centric_{index % 4:04d}",
            "task_id": f"warehouse-task-{index:02d}",
            "task_variant": (
                "station_rush_d10"
                if index % 2 == 0
                else "balanced_od_d125"
            ),
            "agent_count": 180 + index,
            "screen_solver_seed": 31 + index,
            "expected_fingerprint": f"fingerprint-{index:02d}",
            "expected_conflicts": 20 + index,
            "qualification": {"passed": True},
        }
        row["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(
            row
        )
        rows.append(row)

    _write_jsonl(checkpoint_root / "checkpoint_manifest.jsonl", rows)
    _write_json(
        checkpoint_root / "checkpoint_qualification_report.json",
        {
            "experiment_id": subject.EXPERIMENT_ID,
            "passed": True,
            "config_sha256": _sha256(CONFIG),
            "attempted_candidate_count": 8,
            "qualified_checkpoint_count": 7,
            "state_supply_unavailable_count": 1,
            "execution_failure_count": 0,
            "qualified_checkpoint_ids": [
                row["checkpoint_id"] for row in rows
            ],
            "checkpoint_selection_controller_outcomes_consulted": False,
            "failed_candidate_replacement": False,
        },
    )
    return rows


def test_invalid_schedule_does_not_overwrite_existing_report(tmp_path: Path) -> None:
    _checkpoint_fixture(tmp_path)
    report_path = tmp_path / "ttf" / "ttf_report.json"
    _write_json(report_path, {"existing": True})
    before = report_path.read_bytes()
    _write_jsonl(tmp_path / "ttf" / "execution_schedule.jsonl", [{"wrong": True}])
    with pytest.raises(ValueError, match="schedule mismatch"):
        subject.analyze_ttf(CONFIG, tmp_path)
    assert report_path.read_bytes() == before


def test_plan_has_seven_by_four_rotating_serial_schedule(tmp_path: Path) -> None:
    rows = _checkpoint_fixture(tmp_path)

    plan = subject.plan_ttf(CONFIG, tmp_path)
    schedule = plan["schedule"]

    assert plan["schema"] == subject.TTF_PLAN_SCHEMA
    assert plan["checkpoint_count"] == 7
    assert plan["controller_count"] == 4
    assert plan["episode_count"] == 28
    assert plan["timed_workers"] == 1
    assert len(schedule) == 28
    assert [row["schedule_index"] for row in schedule] == list(range(28))

    for checkpoint in rows:
        key_index = checkpoint["key_index"]
        key_schedule = [
            row for row in schedule if row["key_index"] == key_index
        ]
        expected = [
            subject.CONTROLLERS[
                (key_index % len(subject.CONTROLLERS) + position)
                % len(subject.CONTROLLERS)
            ]
            for position in range(4)
        ]
        assert [row["controller"] for row in key_schedule] == expected
        assert [row["within_key_position"] for row in key_schedule] == [
            0,
            1,
            2,
            3,
        ]
        assert {row["checkpoint_id"] for row in key_schedule} == {
            checkpoint["checkpoint_id"]
        }
        assert {row["solver_seed"] for row in key_schedule} == {
            checkpoint["screen_solver_seed"]
        }


def test_plan_rejects_checkpoint_identity_change(tmp_path: Path) -> None:
    rows = _checkpoint_fixture(tmp_path)
    rows[0]["expected_conflicts"] += 1
    _write_jsonl(
        tmp_path / "checkpoints" / "checkpoint_manifest.jsonl",
        rows,
    )

    with pytest.raises(ValueError, match="checkpoint identity changed"):
        subject.plan_ttf(CONFIG, tmp_path)


def _profile(
    controller: str,
    key_index: int,
    case: str,
) -> tuple[float, float, bool, bool]:
    if controller != "v2_only":
        return 20.0, 72.0, True, False

    capped = 15.8 if key_index < 5 else 20.0
    observed = 60.0
    success = True
    timeout = False
    if case == "win_rate_below":
        capped = 14.75 if key_index < 4 else 20.0
    elif case == "ttf_below":
        capped = 16.0 if key_index < 5 else 21.0
    elif case == "throughput_below":
        observed = 61.0
    elif case == "success_loss_above" and key_index == 0:
        success = False
    elif case == "additional_timeout" and key_index == 0:
        success = False
        timeout = True
    return capped, observed, success, timeout


def _write_ttf_manifests(
    output: Path,
    checkpoints: list[dict[str, Any]],
    *,
    case: str,
) -> None:
    ttf_root = output / "ttf"
    _write_jsonl(ttf_root / "execution_schedule.jsonl", subject._schedule(checkpoints))
    for item in subject._schedule(checkpoints):
        checkpoint = checkpoints[int(item["key_index"])]
        capped, observed, success, timeout = _profile(
            str(item["controller"]), int(item["key_index"]), case
        )
        manifest = (
            subject._lane_root(ttf_root, item)
            / subject.TIMED_MANIFESTS[str(item["controller"])]
        )
        _write_jsonl(
            manifest,
            [
                {
                    "task_id": item["task_id"],
                    "solver_seed": item["solver_seed"],
                    "status": "ok",
                    "summary": synthetic_summary(
                        checkpoint, success=success, capped=capped,
                        observed=observed, budget=120.0, timeout=timeout,
                    ),
                }
            ],
        )


def test_screen_gate_accepts_exact_registered_boundaries(tmp_path: Path) -> None:
    checkpoints = _checkpoint_fixture(tmp_path)
    _write_ttf_manifests(tmp_path, checkpoints, case="boundary_pass")

    report = subject.analyze_ttf(CONFIG, tmp_path)
    comparison = report["comparisons_vs_official"]["v2_only"]

    assert report["complete_pairing_and_initial_state_identity"] is True
    assert comparison["paired_key_count"] == 7
    assert comparison["paired_win_count"] == 5
    assert comparison["paired_win_rate"] == pytest.approx(5 / 7)
    assert comparison["mean_capped_ttf_improvement"] == pytest.approx(0.15)
    assert comparison["throughput_improvement"] == pytest.approx(0.20)
    assert comparison["success_rate_loss"] == 0.0
    assert comparison["additional_timeout_count"] == 0
    assert comparison["additional_censor_count"] == 0
    assert comparison["screen_gate_passed"] is True
    assert report["screen_gate_passed_controllers"] == ["v2_only"]


def test_analysis_propagates_completed_lane_authentication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _checkpoint_fixture(tmp_path)

    def reject_lane(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("completed TTF lane trace hash mismatch")

    monkeypatch.setattr(subject, "load_ttf_lane", reject_lane)
    with pytest.raises(ValueError, match="trace hash mismatch"):
        subject.analyze_ttf(CONFIG, tmp_path)


@pytest.mark.parametrize(
    "case",
    (
        "win_rate_below",
        "ttf_below",
        "throughput_below",
        "success_loss_above",
        "additional_timeout",
    ),
)
def test_screen_gate_rejects_each_below_boundary_case(
    tmp_path: Path,
    case: str,
) -> None:
    checkpoints = _checkpoint_fixture(tmp_path)
    _write_ttf_manifests(tmp_path, checkpoints, case=case)

    report = subject.analyze_ttf(CONFIG, tmp_path)

    assert report["complete_pairing_and_initial_state_identity"] is True
    assert report["comparisons_vs_official"]["v2_only"][
        "screen_gate_passed"
    ] is False
