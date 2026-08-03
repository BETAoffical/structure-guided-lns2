from __future__ import annotations

import json
from pathlib import Path

from experiments._common import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _read(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_guardrank_expansion_is_outcome_blind_map_disjoint_preflight() -> None:
    source = _read("stride_guardrank_map_expansion_preflight_source.json")
    runtime = _read("stride_guardrank_map_expansion_preflight_runtime.json")
    maps = {str(row["id"]) for row in source["benchmarks"]}
    forbidden = set(source["freshness"]["forbid_current_repairability_maps"])
    formal_path = ROOT / source["formal_ood_config"]
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    formal_maps = {str(row["benchmark_id"]) for row in formal["cases"]}
    assert source["scientific_status"] == (
        "fresh_map_train_extension_preflight_before_candidate_outcomes"
    )
    assert len(source["freshness"]["pre_registration_git_commit"]) == 40
    assert len(maps) == source["expected_map_count"] == 8
    assert not maps & forbidden
    assert not maps & formal_maps
    assert sha256_file(formal_path) == source["formal_ood_config_sha256"]
    assert source["selection_inputs_forbidden"] == [
        "v2_relative_ttf",
        "guardrank_relative_ttf",
        "controller_action",
        "controller_repair_outcome",
        "candidate_repair_outcome",
    ]
    expected_tasks = sum(
        len(row["agent_counts"])
        * len(source["task_seeds"])
        * len(source["task_variants"])
        for row in source["benchmarks"]
    )
    assert expected_tasks == source["expected_task_count"] == 48
    assert all(
        agent_count % 2 == 0
        for benchmark in source["benchmarks"]
        for agent_count in benchmark["agent_counts"]
    ), "opposite_exchange requires even loads under the frozen endpoint generator"
    assert runtime["max_decisions"] == 0
    assert runtime["environment"]["max_repair_iterations"] == 0
    assert runtime["dataset_design"]["map_count"] == 8
    assert runtime["dataset_design"]["instance_count"] == 48
    assert maps.isdisjoint(runtime["dataset_design"]["historical_map_ids"])


def test_guardrank_expansion_archive_members_are_checksum_pinned_when_available() -> None:
    source = _read("stride_guardrank_map_expansion_preflight_source.json")
    archive = ROOT / "build" / "movingai-dao-compact-source-v1" / "_archives" / "dao-map.zip"
    if not archive.is_file():
        return
    import hashlib
    import zipfile

    assert sha256_file(archive) == source["map_archive"]["sha256"]
    with zipfile.ZipFile(archive) as bundle:
        for row in source["benchmarks"]:
            payload = bundle.read(str(row["member"]))
            assert hashlib.sha256(payload).hexdigest() == row["member_sha256"]
