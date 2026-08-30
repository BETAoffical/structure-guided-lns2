from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_source_v2 as source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_hierarchical_ch_source_v2.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture()
def materialized(tmp_path: Path) -> Path:
    output = tmp_path / "source-v2"
    report = source.materialize_source_dataset(CONFIG, output)
    assert report["status"] == "MATERIALIZED"
    assert report["task_count"] == 64
    return output


@pytest.fixture()
def planned(materialized: Path) -> Path:
    report = source.plan_source_collection(CONFIG, materialized)
    assert report["status"] == "PLANNED"
    return materialized


def _write_qualification(
    *,
    output: Path,
    dataset: Path,
    split: str,
    jobs: set[tuple[str, int]],
    passing: bool,
) -> None:
    split_rows = {
        str(row["task_id"]): row
        for row in _jsonl(dataset / split / "manifest.jsonl")
    }
    rows = []
    for task_id, seed in sorted(jobs):
        task = split_rows[task_id]
        conflicts = 16 if passing else 0
        rows.append(
            {
                "task_id": task_id,
                "solver_seed": seed,
                "map_id": task["map_id"],
                "split": split,
                "layout_mode": task["layout_mode"],
                "task_variant": task["task_variant"],
                "agent_count": task["agent_count"],
                "status": "ok",
                "error": None,
                "initial_complete": True,
                "initial_feasible": conflicts == 0,
                "initial_conflicts": conflicts,
                "state_fingerprint": f"{task_id}/{seed}",
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text("{}\n", encoding="utf-8")
    (output / "qualification_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "qualification_report.json").write_text(
        json.dumps({"passed": passing}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_context_pins_new_identity_exact_product_and_no_final() -> None:
    ctx = source.load_registered_source_context(CONFIG)
    assert ctx["task_registration"]["status"] == "REGISTERED"
    assert ctx["task_registration_report"]["passed"] is True
    assert ctx["variant_ids"] == source.VARIANT_IDS
    assert len(ctx["allowed_ids"]) == 16
    assert not ctx["allowed_ids"] & ctx["sealed_ids"]
    assert all(len(loads) == 2 for loads in ctx["load_grid"].values())
    assert ctx["source_config"]["collection"]["workers"] == 20
    assert ctx["source_config"]["collection"]["expected_episode_count"] == 128


def test_materializer_emits_exact_bucket_mix_product_and_map_family_counts(
    materialized: Path,
) -> None:
    dataset = materialized / "dataset"
    assert len(_jsonl(dataset / "source_manifest.jsonl")) == 16
    ctx = source.load_registered_source_context(CONFIG)
    for split in source.ALLOWED_SPLITS:
        rows = _jsonl(dataset / split / "manifest.jsonl")
        assert len(rows) == 32
        assert {
            (row["map_id"], row["variant_id"], row["agent_count"]) for row in rows
        } == {
            (map_id, variant, load)
            for map_id in ctx["map_splits"][split]
            for variant in source.VARIANT_IDS
            for load in ctx["load_grid"][map_id]
        }
        assert all(row["training_authorized"] is False for row in rows)
        for row in rows:
            assert _json(dataset / split / row["task_file"])["agent_count"] == row["agent_count"]

    source.plan_source_collection(CONFIG, materialized)
    for split in source.ALLOWED_SPLITS:
        config = _json(materialized / "collection_configs" / f"{split}.json")
        assert config["workers"] == 20
        assert config["dataset_design"]["scenario_indices"] == [0, 1]
        # Counts are maps per family, not four materialized tasks per map.
        assert sum(config["dataset_design"]["layout_family_counts"].values()) == 8
        assert config["dataset_design"]["task_count"] == 32


def test_plan_is_128_jobs_workers_are_overridable_and_dry_collect_is_pure(
    planned: Path,
) -> None:
    schedule = _jsonl(planned / "source_schedule.jsonl")
    assert len(schedule) == 128
    assert len({(row["split"], row["task_id"], row["solver_seed"]) for row in schedule}) == 128
    preview = source.plan_source_collection(CONFIG, planned, dry_run=True, workers=22)
    assert preview["workers"] == 22
    calls = []

    def forbidden_runner(**kwargs):
        calls.append(kwargs)
        raise AssertionError("dry-run must not invoke the production runner")

    trust_path = planned / "collection_trust_report__all.json"
    report = source.collect_source_episodes(CONFIG, planned, dry_run=True, runner=forbidden_runner)
    assert report["status"] == "DRY_RUN_PREVIEW"
    assert report["runner_invoked"] is False
    assert report["planned_episode_count"] == 128
    assert calls == []
    assert not trust_path.exists()


def test_custom_supply_gate_failure_runs_both_qualifications_and_no_policy(
    planned: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_runner(**kwargs):
        split = Path(kwargs["config_path"]).stem
        calls.append((split, kwargs["phase"]))
        assert kwargs["phase"] == "qualify"
        _write_qualification(
            output=Path(kwargs["output"]),
            dataset=Path(kwargs["dataset"]),
            split=split,
            jobs=kwargs["job_keys"],
            passing=split == "train",
        )
        return {"status": "ok"}

    report = source.collect_source_episodes(CONFIG, planned, runner=fake_runner)
    assert calls == [("train", "qualify"), ("development", "qualify")]
    assert report["status"] == "STATE_SUPPLY_FAIL_NO_BACKFILL"
    assert report["realized_episode_count"] == 0
    assert report["qualification_passed"] == {"train": True, "development": False}
    assert not list((planned / "collection").rglob("realized_dynamic_manifest.jsonl"))
    dev_audit = _json(
        planned / "collection" / "development" / "source_v2_qualification_audit.json"
    )
    assert dev_audit["gates"]["minimum_nonzero_resets_per_map_variant"] is False
    assert dev_audit["gates"]["minimum_maps_with_a_reset_at_least_16_conflicts"] is False


def test_resume_reuses_complete_qualification_then_runs_only_v2_full_policy(
    planned: Path,
) -> None:
    ctx = source.load_registered_source_context(CONFIG)
    dataset = planned / "dataset"
    for split in source.ALLOWED_SPLITS:
        task_ids = [row["task_id"] for row in _jsonl(dataset / split / "manifest.jsonl")]
        jobs = {
            (str(task_id), int(seed))
            for task_id in task_ids
            for seed in ctx["source_config"]["collection"]["solver_seeds"]
        }
        _write_qualification(
            output=planned / "collection" / split,
            dataset=dataset,
            split=split,
            jobs=jobs,
            passing=True,
        )

    calls: list[tuple[str, str, int]] = []

    def fake_runner(**kwargs):
        split = Path(kwargs["config_path"]).stem
        calls.append((split, kwargs["phase"], kwargs["workers"]))
        assert kwargs["phase"] == "realized_dynamic"
        root = Path(kwargs["output"])
        rows = [
            {"task_id": task_id, "solver_seed": seed}
            for task_id, seed in sorted(kwargs["job_keys"])
        ]
        (root / "realized_dynamic_manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return {"status": "ok"}

    report = source.collect_source_episodes(CONFIG, planned, resume=True, runner=fake_runner)
    assert calls == [
        ("train", "realized_dynamic", 20),
        ("development", "realized_dynamic", 20),
    ]
    assert report["status"] == "COLLECTED_EXACT_PRODUCT"
    assert report["qualification_reused"] == {"train": True, "development": True}
    assert report["qualification_passed"] == {"train": True, "development": True}
    assert report["realized_episode_count"] == 128


def test_registration_or_materializer_tamper_fails_closed(tmp_path: Path) -> None:
    config = _json(CONFIG)
    config["task_registration"]["sha256"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="task registration SHA256 mismatch"):
        source.load_registered_source_context(tampered)
