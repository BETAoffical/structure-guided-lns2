from __future__ import annotations

from pathlib import Path
from typing import Any

from generators.config import load_json
from generators.dataset import generate_dataset

from experiments.repair_collection import _read_json


def validate_confirmation_dataset_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != 2:
        raise ValueError("unsupported ranking-objective confirmation dataset config")
    if int(config.get("master_seed", -1)) != 20270421:
        raise ValueError("ranking-objective confirmation master seed must be 20270421")
    splits = dict(config.get("splits", {}))
    if set(splits) != {"policy_confirmation"}:
        raise ValueError("ranking-objective confirmation must use its own split")
    counts = dict(splits["policy_confirmation"].get("layout_counts", {}))
    expected = {
        "regular_beltway": 4,
        "compartmentalized": 4,
        "dead_end_aisles": 4,
    }
    if counts != expected or sum(counts.values()) != 12:
        raise ValueError("ranking-objective confirmation requires four maps per layout")
    if int(config.get("tasks_per_map", 0)) != 4:
        raise ValueError("ranking-objective confirmation requires four tasks per map")


def generate_gated_confirmation_dataset(
    audit_report: str | Path,
    config_path: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    report = _read_json(Path(audit_report).resolve())
    if (
        str(report.get("decision")) != "eligible_for_independent_confirmation"
        or not bool(report.get("confirmation_generation_allowed"))
        or not bool(
            report.get("development_validation", {})
            .get("acceptance", {})
            .get("passed")
        )
    ):
        raise ValueError(
            "development objective gate did not pass; confirmation generation is forbidden"
        )
    config = load_json(config_path)
    validate_confirmation_dataset_config(config)
    return generate_dataset(config, output)


__all__ = [
    "generate_gated_confirmation_dataset",
    "validate_confirmation_dataset_config",
]
