from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_source_v1 as source
from experiments.stride_hierarchical_ch_h1_v2 import SOURCE_LOAD_GRID


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_hierarchical_ch_source_v1.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture()
def materialized(tmp_path: Path) -> Path:
    output = tmp_path / "source"
    report = source.materialize_source_dataset(CONFIG, output)
    assert report["status"] == "MATERIALIZED"
    return output


def test_registered_context_is_exact_train_development_no_train() -> None:
    ctx = source.load_registered_source_context(CONFIG)
    assert ctx["report"]["status"] == "REGISTERED"
    assert ctx["report"]["training_authorized"] is False
    assert set(ctx["load_grid"]) == set(SOURCE_LOAD_GRID)
    assert ctx["load_grid"] == {key: list(value) for key, value in SOURCE_LOAD_GRID.items()}
    assert set(ctx["allowed_ids"]) == set(ctx["map_splits"]["train"]) | set(
        ctx["map_splits"]["development"]
    )
    assert not ctx["allowed_ids"] & ctx["sealed_ids"]


def test_materialize_copies_exact_registered_raw_files_and_prefix_tasks(
    materialized: Path,
) -> None:
    trust = _json(materialized / "materialization_trust_report.json")
    assert trust["task_count"] == 32
    assert trust["raw_file_count"] == 32
    assert trust["training_authorized"] is False
    dataset = materialized / "dataset"
    source_rows = _jsonl(dataset / "source_manifest.jsonl")
    assert len(source_rows) == 16
    for split in source.ALLOWED_SPLITS:
        rows = _jsonl(dataset / split / "manifest.jsonl")
        assert len(rows) == 16
        assert {row["map_id"] for row in rows}.isdisjoint(
            source.load_registered_source_context(CONFIG)["sealed_ids"]
        )
        expected = {
            (map_id, load)
            for map_id in source.load_registered_source_context(CONFIG)["map_splits"][split]
            for load in SOURCE_LOAD_GRID[map_id]
        }
        assert {(row["map_id"], row["agent_count"]) for row in rows} == expected
        for row in rows:
            map_file = dataset / split / row["map_file"]
            scenario_file = dataset / split / row["scenario_file"]
            assert hashlib.sha256(map_file.read_bytes()).hexdigest() == row["map_sha256"]
            assert hashlib.sha256(scenario_file.read_bytes()).hexdigest() == row["scenario_sha256"]
            task = _json(dataset / split / row["task_file"])
            assert task["unique_start_count"] == row["agent_count"]
            assert task["unique_goal_count"] == row["agent_count"]
            assert task["map_sha256"] == row["map_sha256"]
            assert task["scenario_sha256"] == row["scenario_sha256"]


def test_plan_and_dry_collect_are_exact_products_and_call_only_registered_phases(
    materialized: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = source.plan_source_collection(CONFIG, materialized)
    assert plan["episode_count"] == 64
    assert plan["split_episode_counts"] == {"train": 32, "development": 32}
    schedule = _jsonl(materialized / "source_schedule.jsonl")
    assert len(schedule) == 64
    assert len({(row["split"], row["task_id"], row["solver_seed"]) for row in schedule}) == 64
    calls: list[dict] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return {"status": "dry-run"}

    monkeypatch.setattr(source, "run_closed_loop_collection", fake_runner)
    report = source.collect_source_episodes(CONFIG, materialized, dry_run=True)
    assert report["status"] == "DRY_RUN_DISPATCHED"
    assert report["episode_count"] == 64
    assert not (materialized / "collection_trust_report__all.json").exists()
    assert [call["phase"] for call in calls] == [
        "qualify",
        "qualify",
        "realized_dynamic",
        "realized_dynamic",
    ]
    assert [call["resume"] for call in calls] == [False, False, True, True]
    for call in calls:
        assert call["controller"] == "v2-full"
        assert call["feature_backend"] == "native"
        assert call["deterministic_pp_replay"] is True
        assert call["dry_run"] is True
        assert len(call["job_keys"]) == 32
        assert call["job_keys"] == call["cohort_job_keys"]
        assert call["wall_time_budget_seconds"] == 200.0
        assert call["episode_process_timeout_seconds"] == 240.0
    for split in source.ALLOWED_SPLITS:
        config = _json(materialized / "collection_configs" / f"{split}.json")
        assert config["split"] == split
        assert config["solver_seeds"] == [41, 42]
        assert config["max_decisions"] == 12
        assert config["workers"] == 16
        assert config["dataset_design"]["task_count"] == 16
        assert config["dataset_design"]["scenario_indices"] == [0]
        assert sum(config["dataset_design"]["layout_family_counts"].values()) == 8
        assert config["dataset_design"]["layout_family_counts"] == dict(
            sorted(
                collections.Counter(
                    source.load_registered_source_context(CONFIG)["design"]["map_groups"][map_id]
                    for map_id in source.load_registered_source_context(CONFIG)["map_splits"][split]
                ).items()
            )
        )
        assert config["training_authorized"] is False


def test_tamper_and_sealed_final_access_fail_closed(
    materialized: Path, tmp_path: Path
) -> None:
    tampered_config = tmp_path / "tampered.json"
    config = _json(CONFIG)
    config["registered_design"]["registration_report"]["sha256"] = "0" * 64
    tampered_config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="registration report SHA256 mismatch"):
        source.load_registered_source_context(tampered_config)
    source.plan_source_collection(CONFIG, materialized)
    with pytest.raises(ValueError, match="sealed-final"):
        source.collect_source_episodes(CONFIG, materialized, split="sealed_final", dry_run=True)
    train_manifest = materialized / "dataset" / "train" / "manifest.jsonl"
    rows = _jsonl(train_manifest)
    map_file = materialized / "dataset" / "train" / rows[0]["map_file"]
    map_file.write_bytes(map_file.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        source.plan_source_collection(CONFIG, materialized, resume=True)


def test_collect_dry_run_then_real_collection_uses_same_output(
    materialized: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source.plan_source_collection(CONFIG, materialized)

    def fake_runner(**kwargs):
        if not kwargs["dry_run"]:
            root = Path(kwargs["output"])
            root.mkdir(parents=True, exist_ok=True)
            name = (
                "qualification_manifest.jsonl"
                if kwargs["phase"] == "qualify"
                else "realized_dynamic_manifest.jsonl"
            )
            rows = [
                {"task_id": task_id, "solver_seed": seed}
                for task_id, seed in sorted(kwargs["job_keys"])
            ]
            (root / name).write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            if kwargs["phase"] == "qualify":
                (root / "qualification_report.json").write_text(
                    json.dumps({"passed": True}), encoding="utf-8"
                )
        return {"status": "dry-run" if kwargs["dry_run"] else "ok"}

    monkeypatch.setattr(source, "run_closed_loop_collection", fake_runner)
    dry_report = source.collect_source_episodes(CONFIG, materialized, dry_run=True)
    trust_path = materialized / "collection_trust_report__all.json"
    assert dry_report["status"] == "DRY_RUN_DISPATCHED"
    assert not trust_path.exists()

    collected = source.collect_source_episodes(CONFIG, materialized)
    assert collected["status"] == "COLLECTED_EXACT_PRODUCT"
    assert _json(trust_path)["status"] == "COLLECTED_EXACT_PRODUCT"


def test_failed_qualification_suppresses_all_policy_calls_and_reuses_resume_source(
    materialized: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source.plan_source_collection(CONFIG, materialized)
    ctx = source.load_registered_source_context(CONFIG)
    seeds = ctx["source_config"]["collection"]["solver_seeds"]
    for split in source.ALLOWED_SPLITS:
        collection_root = materialized / "collection" / split
        collection_root.mkdir(parents=True)
        task_ids = [
            row["task_id"] for row in _jsonl(materialized / "dataset" / split / "manifest.jsonl")
        ]
        rows = [
            {"task_id": task_id, "solver_seed": seed}
            for task_id in task_ids
            for seed in seeds
        ]
        (collection_root / "qualification_manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        (collection_root / "run_config.json").write_text("{}\n", encoding="utf-8")

    calls: list[dict] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        root = Path(kwargs["output"])
        assert kwargs["phase"] == "qualify"
        assert Path(kwargs["qualification_source"]) == root
        (root / "qualification_report.json").write_text(
            json.dumps({"passed": kwargs["config_path"].stem == "train"}),
            encoding="utf-8",
        )
        return {"status": "resumed"}

    monkeypatch.setattr(source, "run_closed_loop_collection", fake_runner)
    report = source.collect_source_episodes(CONFIG, materialized, resume=True)
    assert [call["phase"] for call in calls] == ["qualify", "qualify"]
    assert report["status"] == "STATE_SUPPLY_FAIL_NO_BACKFILL"
    assert report["planned_episode_count"] == 64
    assert report["realized_episode_count"] == 0
    assert report["qualification_passed"] == {"train": True, "development": False}
    assert not list((materialized / "collection").rglob("realized_dynamic_manifest.jsonl"))
    assert _json(materialized / "collection_trust_report__all.json")["status"] == (
        "STATE_SUPPLY_FAIL_NO_BACKFILL"
    )


def test_dry_materialize_does_not_create_or_read_raw_output(tmp_path: Path) -> None:
    output = tmp_path / "dry"
    report = source.materialize_source_dataset(CONFIG, output, dry_run=True)
    assert report["status"] == "DRY_RUN"
    assert report["expected_task_count"] == 32
    assert not output.exists()
