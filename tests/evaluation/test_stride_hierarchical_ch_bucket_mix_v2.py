from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_bucket_mix_v2 as bucket_mix


def _scenario_bytes(map_id: str = "toy", *, buckets: int = 4, rows_per_bucket: int = 8) -> bytes:
    rows = ["version 1"]
    for bucket in range(buckets):
        for offset in range(rows_per_bucket):
            index = bucket * rows_per_bucket + offset
            rows.append(
                "\t".join(
                    (
                        str(bucket),
                        f"{map_id}.map",
                        "128",
                        "4",
                        str(index),
                        "0",
                        str(index),
                        "1",
                        f"{bucket + 1}.{offset:02d}",
                    )
                )
            )
    return ("\n".join(rows) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_fixed_variants_are_deterministic_round_robin_exact_prefixes() -> None:
    payload = _scenario_bytes()
    kwargs = {
        "expected_source_sha256": _sha(payload),
        "map_id": "toy",
        "registered_loads": [8, 16],
    }
    first = bucket_mix.build_bucket_mix_variants_from_bytes(payload, **kwargs)
    second = bucket_mix.build_bucket_mix_variants_from_bytes(payload, **kwargs)

    assert tuple(variant.variant_id for variant in first.variants) == bucket_mix.VARIANT_IDS
    assert first.capacity_by_variant == {"bucket_mix_00": 32, "bucket_mix_01": 32}
    assert first == second
    assert first.variants[0].task_for_load(16).selection_sha256 != (
        first.variants[1].task_for_load(16).selection_sha256
    )
    for variant in first.variants:
        short = variant.task_for_load(8)
        long = variant.task_for_load(16)
        assert short.rows == long.rows[:8]
        assert Counter(row.bucket for row in short.rows) == {0: 2, 1: 2, 2: 2, 3: 2}
        assert len({row.start for row in long.rows}) == 16
        assert len({row.goal for row in long.rows}) == 16
        assert short.scenario_payload.count(b"\n") == 9


def test_greedy_uniqueness_exposes_capacity_and_fails_below_registered_load() -> None:
    payload = (
        "version 1\n"
        "0\ttoy.map\t8\t8\t0\t0\t4\t4\t1.0\n"
        "1\ttoy.map\t8\t8\t0\t0\t5\t5\t2.0\n"
        "2\ttoy.map\t8\t8\t1\t1\t4\t4\t3.0\n"
        "3\ttoy.map\t8\t8\t2\t2\t6\t6\t4.0\n"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="unique capacity is below registered load"):
        bucket_mix.build_bucket_mix_variants_from_bytes(
            payload,
            expected_source_sha256=_sha(payload),
            map_id="toy",
            registered_loads=[3],
        )


def test_full_file_hash_tamper_and_sealed_access_fail_before_read(tmp_path: Path) -> None:
    payload = _scenario_bytes()
    with pytest.raises(ValueError, match="full-file SHA256 mismatch"):
        bucket_mix.build_bucket_mix_variants_from_bytes(
            payload + b"\n",
            expected_source_sha256=_sha(payload),
            map_id="toy",
            registered_loads=[8],
        )

    missing_sealed_path = tmp_path / "secret.map.scen"
    with pytest.raises(ValueError, match="sealed-final raw scenario access"):
        bucket_mix.build_registered_bucket_mix_variants(
            missing_sealed_path,
            expected_source_sha256="0" * 64,
            map_id="secret",
            registered_loads=[1],
            allowed_map_ids={"toy"},
            sealed_map_ids={"secret"},
        )


def test_materializer_emits_two_variants_at_each_registered_load(tmp_path: Path) -> None:
    payload = _scenario_bytes()
    source = tmp_path / "toy.map.scen"
    source.write_bytes(payload)
    split_root = tmp_path / "dataset" / "train"
    kwargs = {
        "expected_source_sha256": _sha(payload),
        "map_id": "toy",
        "registered_loads": [8, 16],
        "split": "train",
        "output_split_root": split_root,
        "allowed_map_ids": {"toy"},
        "sealed_map_ids": {"secret"},
        "map_file": "maps/toy.map",
        "map_sha256": "a" * 64,
        "layout_family": "toy-family",
    }
    rows = bucket_mix.materialize_registered_bucket_mix_map(source, **kwargs)
    assert len(rows) == 4
    assert {(row["bucket_mix_variant"], row["agent_count"]) for row in rows} == {
        ("bucket_mix_00", 8),
        ("bucket_mix_00", 16),
        ("bucket_mix_01", 8),
        ("bucket_mix_01", 16),
    }
    for row in rows:
        assert row["layout_mode"] == "toy-family"
        scenario_path = split_root / row["scenario_file"]
        task_path = split_root / row["task_file"]
        assert _sha(scenario_path.read_bytes()) == row["scenario_sha256"]
        assert _sha(task_path.read_bytes()) == row["task_sha256"]
        task = _json(task_path)
        assert task["unique_start_count"] == row["agent_count"]
        assert task["unique_goal_count"] == row["agent_count"]
        assert len(task["ordered_source_line_numbers"]) == row["agent_count"]
        assert task["outcome_filtering_used"] is False
        assert task["sealed_final_semantic_access"] is False
    for variant_id in bucket_mix.VARIANT_IDS:
        tasks = sorted(
            (_json(split_root / row["task_file"]) for row in rows if row["bucket_mix_variant"] == variant_id),
            key=lambda task: task["agent_count"],
        )
        assert tasks[0]["ordered_source_line_numbers"] == tasks[1][
            "ordered_source_line_numbers"
        ][:8]

    assert bucket_mix.materialize_registered_bucket_mix_map(
        source, resume=True, **kwargs
    ) == rows


def test_parse_rejects_wrong_map_and_materializer_rejects_final_split(tmp_path: Path) -> None:
    payload = _scenario_bytes("other")
    with pytest.raises(ValueError, match="different map"):
        bucket_mix.build_bucket_mix_variants_from_bytes(
            payload,
            expected_source_sha256=_sha(payload),
            map_id="toy",
            registered_loads=[8],
        )

    source = tmp_path / "toy.map.scen"
    source.write_bytes(_scenario_bytes())
    with pytest.raises(ValueError, match="sealed-final or unregistered source split"):
        bucket_mix.materialize_registered_bucket_mix_map(
            source,
            expected_source_sha256=_sha(source.read_bytes()),
            map_id="toy",
            registered_loads=[8],
            split="sealed_final",
            output_split_root=tmp_path / "out",
            allowed_map_ids={"toy"},
            sealed_map_ids={"secret"},
            map_file="maps/toy.map",
            map_sha256="a" * 64,
            layout_family="toy-family",
        )
