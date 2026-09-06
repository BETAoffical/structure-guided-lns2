from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import experiments.stride_warehouse_component16_dual16_independent_ttf_v1 as subject
from tests.evaluation.ttf_fixtures import mocked_validated_lane, synthetic_summary


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_warehouse_component16_dual16_independent_ttf_v1.json"
)


@pytest.fixture(autouse=True)
def authenticated_report_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "load_ttf_lane", mocked_validated_lane)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _preflight() -> dict[str, Any]:
    return subject.run_preflight(CONFIG)


def _checkpoint_rows() -> list[dict[str, Any]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    relative = config["inputs"]["source_checkpoint_manifest"]["path"]
    return [
        json.loads(line)
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_schedule(output: Path, schedule: list[dict[str, Any]]) -> None:
    _write_jsonl(output / "ttf" / "execution_schedule.jsonl", schedule)


def _write_new_lane_manifests(
    output: Path,
    schedule: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> None:
    by_id = {str(row["checkpoint_id"]): row for row in checkpoints}
    profiles = {
        "official_adaptive": (20.0, 60.0),
        "component16": (12.0, 40.0),
        "dual16": (16.0, 50.0),
    }
    for item in schedule:
        checkpoint = by_id[str(item["checkpoint_id"])]
        capped, observed = profiles[str(item["controller"])]
        manifest = (
            subject._lane_root(output / "ttf", item)
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
                        checkpoint, capped=capped, observed=observed, budget=60.0,
                    ),
                }
            ],
        )


def test_invalid_schedule_does_not_overwrite_existing_report(tmp_path: Path) -> None:
    report_path = tmp_path / "ttf" / subject.REPORT_FILENAME
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"existing": true}', encoding="utf-8")
    before = report_path.read_bytes()
    _write_schedule(tmp_path, [{"wrong": True}])
    with pytest.raises(ValueError, match="schedule mismatch"):
        subject.analyze_collection(CONFIG, tmp_path)
    assert report_path.read_bytes() == before


def test_preflight_has_16_by_3_key_mod_rotating_strict_schedule() -> None:
    preflight = _preflight()
    schedule = preflight["schedule"]

    assert preflight["schema"] == subject.PREFLIGHT_SCHEMA
    assert preflight["checkpoint_count"] == 16
    assert preflight["map_count"] == 8
    assert preflight["controller_count"] == 3
    assert preflight["episode_count"] == 48
    assert preflight["timed_workers"] == 1
    assert len(schedule) == 48
    assert [row["schedule_index"] for row in schedule] == list(range(48))

    for key_index in range(16):
        group = [row for row in schedule if row["key_index"] == key_index]
        expected = [
            subject.CONTROLLERS[(key_index + position) % 3]
            for position in range(3)
        ]
        assert [row["controller"] for row in group] == expected
        assert [row["within_key_position"] for row in group] == [0, 1, 2]
        assert len({row["checkpoint_id"] for row in group}) == 1
        assert len({row["checkpoint_identity_sha256"] for row in group}) == 1
        assert len({row["solver_seed"] for row in group}) == 1


@pytest.mark.parametrize(
    ("field", "error"),
    (
        ("map_sha256", "SHA-disjoint from H1"),
        ("task_sha256", "SHA-disjoint from H1"),
        ("checkpoint_identity_sha256", "SHA-disjoint from H1"),
    ),
)
def test_preflight_rejects_h1_identity_sha_overlap(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    error: str,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_path = (
        ROOT / config["inputs"]["source_checkpoint_manifest"]["path"]
    ).resolve()
    h1_path = (ROOT / config["inputs"]["h1_source_manifest"]["path"]).resolve()
    original = subject.read_jsonl
    source_rows = [dict(row) for row in original(source_path)]

    def tampered_read_jsonl(path: str | Path) -> list[dict[str, Any]]:
        rows = [dict(row) for row in original(path)]
        if Path(path).resolve() == h1_path:
            rows[0][field] = source_rows[0][field]
        return rows

    monkeypatch.setattr(subject, "read_jsonl", tampered_read_jsonl)
    with pytest.raises(ValueError, match=error):
        subject.run_preflight(CONFIG)


def test_three_arm_analysis_requires_complete_new_outputs_and_summarizes_pairs(
    tmp_path: Path,
) -> None:
    preflight = _preflight()
    schedule = list(preflight["schedule"])
    checkpoints = _checkpoint_rows()
    _write_schedule(tmp_path, schedule)
    _write_new_lane_manifests(tmp_path, schedule, checkpoints)

    report = subject.analyze_collection(CONFIG, tmp_path)

    assert report["completed_episode_count"] == 48
    assert report["missing_or_failed_schedule_rows"] == []
    assert report["complete_pairing_and_initial_state_identity"] is True
    assert all(report["integrity_gates"].values())
    assert set(report["controllers"]) == set(subject.CONTROLLERS)
    assert all(
        report["controllers"][controller]["episode_count"] == 16
        for controller in subject.CONTROLLERS
    )
    assert set(report["comparisons"]) == {
        "component16_vs_official",
        "dual16_vs_official",
        "component16_vs_dual16",
    }
    assert all(
        comparison["paired_key_count"] == 16
        for comparison in report["comparisons"].values()
    )
    assert report["comparisons"]["component16_vs_official"][
        "diagnostic_screen_gate_passed"
    ] is True
    assert report["comparisons"]["dual16_vs_official"][
        "diagnostic_screen_gate_passed"
    ] is True
    assert report["diagnostic_gate_only"] is True
    assert report["formal_promotion_allowed"] is False
    assert report["default_replacement_allowed"] is False


def test_old_confirmation_ttf_outputs_are_not_counted_as_new_results(
    tmp_path: Path,
) -> None:
    preflight = _preflight()
    _write_schedule(tmp_path, list(preflight["schedule"]))

    report = subject.analyze_collection(CONFIG, tmp_path)

    assert report["completed_episode_count"] == 0
    assert len(report["missing_or_failed_schedule_rows"]) == 48
    assert report["complete_pairing_and_initial_state_identity"] is False
    assert report["integrity_gates"][
        "complete_key_mod_3_rotating_strict_serial_schedule"
    ] is False
    assert report["prior_dual16_confirmation_cohort_reused"] is True
    assert report["promotion_decision"] == (
        "development_evidence_only_no_default_change"
    )


def test_analysis_propagates_completed_lane_authentication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_schedule(tmp_path, list(_preflight()["schedule"]))

    def reject_lane(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("completed TTF lane controller mismatch")

    monkeypatch.setattr(subject, "load_ttf_lane", reject_lane)
    with pytest.raises(ValueError, match="controller mismatch"):
        subject.analyze_collection(CONFIG, tmp_path)


def test_local_qualification_anchor_uses_current_native_then_rebinds_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = subject._evidence(CONFIG)
    checkpoints = list(evidence["checkpoints"])
    calls: list[dict[str, Any]] = []

    def fake_collection(
        _dataset_root: Path,
        _runtime_config: Path,
        output: Path,
        *,
        phase: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append({"output": Path(output), "phase": phase, **kwargs})
        if len(calls) == 1:
            anchor = Path(output)
            anchor.mkdir(parents=True, exist_ok=True)
            (anchor / "run_config.json").write_text("{}\n", encoding="utf-8")
            _write_jsonl(
                anchor / "qualification_manifest.jsonl",
                [
                    {
                        "task_id": checkpoint["task_id"],
                        "map_id": checkpoint["map_id"],
                        "agent_count": checkpoint["agent_count"],
                        "solver_seed": checkpoint["screen_solver_seed"],
                        "status": "ok",
                        "current_native_probe": True,
                    }
                    for checkpoint in checkpoints
                ],
            )
        return {"status": "mocked"}

    monkeypatch.setattr(subject, "run_closed_loop_collection", fake_collection)
    anchor = subject._prepare_local_qualification_anchor(
        evidence,
        tmp_path / "ttf",
        resume=False,
    )

    assert anchor == tmp_path / "ttf" / "qualification_anchor"
    assert len(calls) == 2
    assert calls[0]["phase"] == "qualify"
    assert calls[0]["workers"] == 8
    assert calls[0]["resume"] is False
    assert "qualification_source" not in calls[0]
    assert len(calls[0]["job_keys"]) == 16
    assert calls[1]["phase"] == "qualify"
    assert calls[1]["workers"] == 1
    assert calls[1]["resume"] is True
    assert calls[1]["qualification_source"] == anchor
    assert len(calls[1]["job_keys"]) == 16
    assert not {
        "controller",
        "controller_bundle",
        "hybridstructpool_augmentation",
        "episode_overrides",
    } & set(calls[0])

    rebound = subject.read_jsonl(anchor / "qualification_manifest.jsonl")
    by_key = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in rebound
    }
    assert len(by_key) == 16
    source_sha = subject.sha256_file(
        Path(evidence["config"]["_registered_inputs"]["source_qualification_manifest"])
    )
    for checkpoint in checkpoints:
        key = (str(checkpoint["task_id"]), int(checkpoint["screen_solver_seed"]))
        row = by_key[key]
        assert row["current_native_probe"] is True
        assert row["qualification_source_kind"] == (
            "authenticated_checkpoint_blob_v1"
        )
        assert row["checkpoint_id"] == checkpoint["checkpoint_id"]
        assert row["checkpoint_identity_sha256"] == (
            checkpoint["checkpoint_identity_sha256"]
        )
        assert row["state_fingerprint"] == checkpoint["expected_fingerprint"]
        assert row["initial_conflicts"] == checkpoint["expected_conflicts"]
        assert row["authenticated_source_qualification_manifest_sha256"] == (
            source_sha
        )


def test_resume_accepts_existing_schedule_and_key_zero_lane_run_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = subject._evidence(CONFIG)
    schedule = subject._schedule(list(evidence["checkpoints"]))
    _write_schedule(tmp_path, schedule)
    first_lane = subject._lane_root(tmp_path / "ttf", schedule[0])
    first_lane.mkdir(parents=True, exist_ok=True)
    (first_lane / "run_config.json").write_text("{}\n", encoding="utf-8")
    local_anchor = tmp_path / "ttf" / "qualification_anchor"
    prepare_resume: list[bool] = []
    lane_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(subject, "_evidence", lambda _path: evidence)
    monkeypatch.setattr(
        subject,
        "run_preflight",
        lambda _path, _output=None: {"schema": subject.PREFLIGHT_SCHEMA},
    )

    def fake_prepare(
        _evidence: dict[str, Any],
        _ttf_root: Path,
        *,
        resume: bool,
    ) -> Path:
        prepare_resume.append(resume)
        return local_anchor

    monkeypatch.setattr(subject, "_prepare_local_qualification_anchor", fake_prepare)
    monkeypatch.setattr(
        subject,
        "_controller_kwargs",
        lambda _root, _config, controller: (controller, {}),
    )

    def fake_collection(
        _dataset_root: Path,
        _runtime_config: Path,
        output: Path,
        *,
        phase: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        lane_calls.append({"output": Path(output), "phase": phase, **kwargs})
        return {"status": "mocked"}

    monkeypatch.setattr(subject, "run_closed_loop_collection", fake_collection)
    monkeypatch.setattr(
        subject,
        "analyze_collection",
        lambda _path, _output: {"status": "analyzed"},
    )

    result = subject.run_collection(CONFIG, tmp_path, resume=True)

    assert result == {"status": "analyzed"}
    assert prepare_resume == [True]
    assert len(lane_calls) == 96
    assert lane_calls[0]["output"] == first_lane
    assert lane_calls[0]["phase"] == "qualify"
    assert lane_calls[0]["resume"] is True
    assert lane_calls[1]["output"] == first_lane
    assert lane_calls[1]["resume"] is True
    assert all(call["qualification_source"] == local_anchor for call in lane_calls)
