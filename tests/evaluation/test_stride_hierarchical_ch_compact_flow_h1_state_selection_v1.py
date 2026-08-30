from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_compact_flow_h1_state_selection_v1 as selection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_h1_state_selection_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture()
def source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    config_path = tmp_path / "selection-config.json"
    config_path.write_bytes(CONFIG.read_bytes())
    source_root = (
        tmp_path
        / "build"
        / "stride-hierarchical-ch-compact-flow-balanced-source-v1"
    )
    schedule: list[dict] = []
    manifests: dict[str, list[dict]] = {"train": [], "development": []}
    blind_rows_by_trace: dict[tuple[str, str], list[dict]] = {}
    split_maps = {
        "train": selection.TRAIN_MAPS,
        "development": selection.DEVELOPMENT_MAPS,
    }
    for split, maps in split_maps.items():
        collection_root = source_root / "collection" / split
        _write_json(collection_root / "run_config.json", {"run_fingerprint": split})
        for map_id in maps:
            episode_number = 0
            for load in selection.MAP_LOADS[map_id]:
                for od_variant in ("opposite_exchange", "uniform_random"):
                    for task_seed in (11, 17):
                        task_id = (
                            f"{map_id}__{od_variant}__seed_{task_seed}__agents_{load}"
                        )
                        for solver_seed in selection.SOLVER_SEEDS:
                            episode_id = f"{task_id}__solver_{solver_seed}"
                            trace_file = f"episodes/{episode_id}.trace"
                            trace_path = collection_root / trace_file
                            trace_path.parent.mkdir(parents=True, exist_ok=True)
                            trace_path.write_text(
                                f"synthetic {split} {episode_id}\n", encoding="utf-8"
                            )
                            trace_sha = _sha256(trace_path)
                            schedule.append(
                                {
                                    "split": split,
                                    "task_id": task_id,
                                    "map_id": map_id,
                                    "layout_mode": "dao_compact_game",
                                    "od_variant": od_variant,
                                    "capacity_stratum": "fixture",
                                    "agent_count": load,
                                    "task_seed": task_seed,
                                    "solver_seed": solver_seed,
                                    "workers": 16,
                                    "controller": "v2-full",
                                    "policy": "realized_dynamic",
                                    "qualification_required": True,
                                    "source_v2_qualification_imported": False,
                                    "training_authorized": False,
                                }
                            )
                            manifests[split].append(
                                {
                                    "split": split,
                                    "task_id": task_id,
                                    "map_id": map_id,
                                    "layout_mode": "dao_compact_game",
                                    "agent_count": load,
                                    "solver_seed": solver_seed,
                                    "episode_id": episode_id,
                                    "policy": "realized_dynamic",
                                    "trace_file": trace_file,
                                    "trace_sha256": trace_sha,
                                    "status": "ok",
                                    "error": None,
                                    "summary": {
                                        "success": episode_number % 2 == 0,
                                        "forbidden_outcome_poison": episode_number,
                                    },
                                }
                            )
                            maximum_decision = 3 if map_id == "den207d" else 11
                            blind_rows_by_trace[(split, trace_file)] = [
                                {
                                    "decision_index": decision_index,
                                    "before_fingerprint": (
                                        f"before/{split}/{task_id}/{solver_seed}/"
                                        f"{decision_index}"
                                    ),
                                    "before_conflicts": 100 - decision_index,
                                    "prefix_actions": [
                                        {"mode": "replay_neighborhood", "agents": [0]}
                                        for _ in range(decision_index)
                                    ],
                                }
                                for decision_index in range(maximum_decision + 1)
                            ]
                            episode_number += 1
        manifests[split].sort(key=lambda row: (row["task_id"], row["solver_seed"]))
        _write_jsonl(
            collection_root / "realized_dynamic_manifest.jsonl", manifests[split]
        )
    schedule.sort(key=lambda row: (row["split"], row["task_id"], row["solver_seed"]))
    schedule_path = source_root / "source_schedule.jsonl"
    _write_jsonl(schedule_path, schedule)
    trust = {
        "schema": selection.SOURCE_TRUST_SCHEMA,
        "experiment_id": selection.SOURCE_EXPERIMENT_ID,
        "status": "COLLECTED_EXACT_PRODUCT",
        "workers": 16,
        "planned_episode_count": 384,
        "realized_episode_count": 384,
        "qualification_job_count": 384,
        "qualification_passed": {"train": True, "development": True},
        "source_v2_qualification_rows_imported": 0,
        "outcome_filtering": False,
        "sealed_final_semantic_access": False,
        "training_authorized": False,
        "source_schedule_sha256": _sha256(schedule_path),
    }
    trust_path = source_root / "collection_trust_report__all.json"
    _write_json(trust_path, trust)
    calls: list[tuple[str, dict]] = []

    def result_blind_stub(collection_root: Path, manifest: dict) -> tuple[list[dict], list]:
        assert set(manifest) == {"trace_file"}
        split = Path(collection_root).name
        calls.append((split, dict(manifest)))
        return [
            dict(row) for row in blind_rows_by_trace[(split, manifest["trace_file"])]
        ], [{"forbidden_outcome_event": True}]

    monkeypatch.setattr(
        selection.trace_replay, "result_blind_decision_rows", result_blind_stub
    )
    return {
        "root": tmp_path,
        "source_root": source_root,
        "config": config_path,
        "trust": trust,
        "trust_path": trust_path,
        "schedule": schedule,
        "manifests": manifests,
        "blind_rows": blind_rows_by_trace,
        "calls": calls,
    }


def test_repository_config_freezes_folds_workers_and_claim_boundary() -> None:
    config = selection.load_config(CONFIG)
    assert config["train_map_folds"] == {
        key: list(value) for key, value in selection.TRAIN_MAP_FOLDS.items()
    }
    assert config["execution"] == {
        "workers": 16,
        "maximum_workers": 20,
        "workers_enter_run_fingerprint": True,
        "solver_invocation_allowed": False,
    }
    assert config["sealed_final"]["status"] == "SEALED"
    assert config["claim_boundary"]["train_and_development_usage"] == (
        "sequential_design_only"
    )
    assert config["claim_boundary"]["training_authorized"] is False
    assert config["claim_boundary"]["runtime_or_ttf_claim_authorized"] is False


def test_exact_result_blind_selection_allows_one_missing_map_depth_cell(
    source_fixture: dict, tmp_path: Path
) -> None:
    output = tmp_path / "selected"
    report = selection.build_state_selection(
        source_fixture["config"],
        output,
        project_root=source_fixture["root"],
    )
    rows = _read_jsonl(output / "selected_states.jsonl")
    assert report["status"] == "SELECTED_RESULT_BLIND_EXACT_PRODUCT"
    assert report["complete"] is True
    assert report["selected_state_count"] == 192
    assert len(rows) == 192
    assert len(source_fixture["calls"]) == 384
    assert report["workers"] == 16
    assert report["maximum_selected_per_episode"] <= 2
    assert report["maximum_selected_per_map_load"] <= 8
    split_counts = Counter(row["split"] for row in rows)
    assert split_counts == Counter({"train": 96, "development": 96})
    map_counts = Counter((row["split"], row["map_id"]) for row in rows)
    assert len(map_counts) == 16
    assert all(8 <= count <= 16 for count in map_counts.values())
    depth_counts = Counter((row["split"], row["depth_band"]) for row in rows)
    assert all(
        depth_counts[(split, band)] == 32
        for split in selection.ALLOWED_SPLITS
        for band in selection.DEPTH_BANDS
    )
    active_maps = {
        (split, band): {
            row["map_id"]
            for row in rows
            if row["split"] == split and row["depth_band"] == band
        }
        for split in selection.ALLOWED_SPLITS
        for band in selection.DEPTH_BANDS
    }
    assert all(len(maps) >= 7 for maps in active_maps.values())
    assert active_maps[("development", "d4plus")] == set(
        selection.DEVELOPMENT_MAPS
    ) - {"den207d"}
    assert all(row["state_id"] == row["state_occurrence_id"] for row in rows)
    assert {row["source_policy"] for row in rows} == {"v2-full"}
    assert {row["map_family"] for row in rows} == {"dao_compact_game"}
    assert all(row["target_outcome_fields_read"] is False for row in rows)
    assert all(row["outcome_filtering"] is False for row in rows)
    assert all("summary" not in row for row in rows)
    assert all("forbidden_outcome_event" not in row for row in rows)
    assert all(row["train_fold"] is not None for row in rows if row["split"] == "train")
    assert all(row["train_fold"] is None for row in rows if row["split"] == "development")


def test_workers_enter_fingerprint_but_do_not_change_deterministic_selection(
    source_fixture: dict, tmp_path: Path
) -> None:
    report16 = selection.build_state_selection(
        source_fixture["config"],
        tmp_path / "workers16",
        workers=16,
        project_root=source_fixture["root"],
    )
    report20 = selection.build_state_selection(
        source_fixture["config"],
        tmp_path / "workers20",
        workers=20,
        project_root=source_fixture["root"],
    )
    rows16 = _read_jsonl(tmp_path / "workers16" / "selected_states.jsonl")
    rows20 = _read_jsonl(tmp_path / "workers20" / "selected_states.jsonl")
    assert report16["run_fingerprint"] != report20["run_fingerprint"]
    assert report16["run_fingerprint_payload"]["workers"] == 16
    assert report20["run_fingerprint_payload"]["workers"] == 20
    assert [row["state_id"] for row in rows16] == [row["state_id"] for row in rows20]
    with pytest.raises(ValueError, match="between 1 and 20"):
        selection.build_state_selection(
            source_fixture["config"],
            tmp_path / "workers21",
            workers=21,
            project_root=source_fixture["root"],
        )


def test_rank_is_exact_five_field_canonical_sha256() -> None:
    payload = ["train", "task", 41, 3, "before"]
    expected = hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert selection.selection_rank("train", "task", 41, 3, "before") == expected


def test_dry_run_verifies_and_selects_without_writing(
    source_fixture: dict, tmp_path: Path
) -> None:
    output = tmp_path / "dry-run"
    report = selection.build_state_selection(
        source_fixture["config"],
        output,
        project_root=source_fixture["root"],
        dry_run=True,
    )
    assert report["status"] == "SELECTED_RESULT_BLIND_EXACT_PRODUCT"
    assert report["selected_state_count"] == 192
    assert report["dry_run"] is True
    assert report["artifacts_written"] is False
    assert not output.exists()


def test_source_trust_must_be_collected_before_any_trace_read(
    source_fixture: dict, tmp_path: Path
) -> None:
    trust = dict(source_fixture["trust"])
    trust["status"] = "STATE_SUPPLY_FAIL_NO_BACKFILL"
    _write_json(source_fixture["trust_path"], trust)
    with pytest.raises(ValueError, match="not COLLECTED_EXACT_PRODUCT"):
        selection.build_state_selection(
            source_fixture["config"],
            tmp_path / "bad-trust",
            project_root=source_fixture["root"],
        )
    assert source_fixture["calls"] == []


def test_each_realized_manifest_requires_exactly_192_ok_rows(
    source_fixture: dict, tmp_path: Path
) -> None:
    path = (
        source_fixture["source_root"]
        / "collection"
        / "train"
        / "realized_dynamic_manifest.jsonl"
    )
    rows = _read_jsonl(path)
    rows[0]["status"] = "resumed"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="incomplete or differs"):
        selection.build_state_selection(
            source_fixture["config"],
            tmp_path / "bad-manifest",
            project_root=source_fixture["root"],
        )
    assert source_fixture["calls"] == []


def test_trace_sha_mismatch_is_rejected(
    source_fixture: dict, tmp_path: Path
) -> None:
    first = source_fixture["manifests"]["train"][0]
    trace = (
        source_fixture["source_root"]
        / "collection"
        / "train"
        / first["trace_file"]
    )
    trace.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trace SHA256 mismatch"):
        selection.build_state_selection(
            source_fixture["config"],
            tmp_path / "bad-trace",
            project_root=source_fixture["root"],
        )


def test_depth_under_supply_fails_without_selection_or_backfill(
    source_fixture: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def shallow_stub(_root: Path, manifest: dict) -> tuple[list[dict], list]:
        return [
            {
                "decision_index": 0,
                "before_fingerprint": f"only-d0/{manifest['trace_file']}",
                "before_conflicts": 1,
                "prefix_actions": [],
            }
        ], []

    monkeypatch.setattr(
        selection.trace_replay, "result_blind_decision_rows", shallow_stub
    )
    output = tmp_path / "under-supply"
    report = selection.build_state_selection(
        source_fixture["config"],
        output,
        project_root=source_fixture["root"],
    )
    assert report["status"] == "STATE_SUPPLY_FAIL_NO_BACKFILL"
    assert report["complete"] is False
    assert report["selected_state_count"] == 0
    assert report["reserve_or_replacement_backfill"] is False
    assert report["outcome_filtering"] is False
    assert not (output / "selected_states.jsonl").exists()
    stored = json.loads(
        (output / "state_selection_trust_report.json").read_text(encoding="utf-8")
    )
    assert stored["status"] == "STATE_SUPPLY_FAIL_NO_BACKFILL"
