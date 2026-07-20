from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.repair_collection import _fingerprint


BALANCED_CONTROLLER_SCHEMA = "lns2.balanced_controller.v1"
BALANCED_CONTROLLER_VERSION = 1
REGISTERED_CONFLICT_THRESHOLDS = (0, 1, 2, 4, 8, 16)
REGISTERED_PRUNER_THRESHOLDS = (0.50, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80)
DEFAULT_BALANCED_CONFIG = (
    "build/initlns-lns2-speed-quality-calibration/balanced_controller.json"
)


@dataclass(frozen=True)
class BalancedControllerConfig:
    conflict_threshold: int
    pruner_threshold: float | None
    source: dict[str, Any]
    quality_gain_retention: float = 0.50
    minimum_speedup_over_full: float = 0.10
    maximum_lns2_wall_ratio: float = 1.0
    promoted: bool = False

    def route(self, conflicts: int) -> str:
        if conflicts <= 0:
            raise ValueError("balanced routing requires a repairable state")
        return "official_adaptive" if conflicts <= self.conflict_threshold else "model"

    def payload(self) -> dict[str, Any]:
        core = {
            "schema": BALANCED_CONTROLLER_SCHEMA,
            "schema_version": BALANCED_CONTROLLER_VERSION,
            "conflict_threshold": self.conflict_threshold,
            "pruner_threshold": self.pruner_threshold,
            "quality_gain_retention": self.quality_gain_retention,
            "minimum_speedup_over_full": self.minimum_speedup_over_full,
            "maximum_lns2_wall_ratio": self.maximum_lns2_wall_ratio,
            "promoted": self.promoted,
            "source": self.source,
        }
        return {**core, "configuration_fingerprint": _fingerprint(core)}


def validate_balanced_payload(value: dict[str, Any]) -> BalancedControllerConfig:
    if str(value.get("schema")) != BALANCED_CONTROLLER_SCHEMA:
        raise ValueError("unexpected balanced controller schema")
    if int(value.get("schema_version", -1)) != BALANCED_CONTROLLER_VERSION:
        raise ValueError("unsupported balanced controller version")
    conflict_threshold = int(value.get("conflict_threshold", -1))
    if conflict_threshold not in REGISTERED_CONFLICT_THRESHOLDS:
        raise ValueError("balanced conflict threshold is not registered")
    raw_pruner = value.get("pruner_threshold")
    pruner_threshold = None if raw_pruner is None else float(raw_pruner)
    if pruner_threshold is not None and not any(
        abs(pruner_threshold - registered) <= 1e-12
        for registered in REGISTERED_PRUNER_THRESHOLDS
    ):
        raise ValueError("balanced pruner threshold is not registered")
    quality = float(value.get("quality_gain_retention", 0.50))
    speed = float(value.get("minimum_speedup_over_full", 0.10))
    wall_ratio = float(value.get("maximum_lns2_wall_ratio", 1.0))
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality gain retention must be in [0, 1]")
    if not 0.0 <= speed < 1.0:
        raise ValueError("minimum speedup over full must be in [0, 1)")
    if wall_ratio <= 0.0:
        raise ValueError("maximum LNS2 wall ratio must be positive")
    source = value.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("balanced controller source must be an object")
    core = {
        "schema": BALANCED_CONTROLLER_SCHEMA,
        "schema_version": BALANCED_CONTROLLER_VERSION,
        "conflict_threshold": conflict_threshold,
        "pruner_threshold": pruner_threshold,
        "quality_gain_retention": quality,
        "minimum_speedup_over_full": speed,
        "maximum_lns2_wall_ratio": wall_ratio,
        "promoted": bool(value.get("promoted", False)),
        "source": source,
    }
    expected = value.get("configuration_fingerprint")
    if expected is not None and str(expected) != _fingerprint(core):
        raise ValueError("balanced controller configuration fingerprint mismatch")
    return BalancedControllerConfig(
        conflict_threshold=conflict_threshold,
        pruner_threshold=pruner_threshold,
        source=dict(source),
        quality_gain_retention=quality,
        minimum_speedup_over_full=speed,
        maximum_lns2_wall_ratio=wall_ratio,
        promoted=bool(value.get("promoted", False)),
    )


def load_balanced_controller(path_or_value: str | Path | dict[str, Any]) -> BalancedControllerConfig:
    if isinstance(path_or_value, dict):
        value = dict(path_or_value)
    else:
        path = Path(path_or_value).resolve()
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("balanced controller configuration must be an object")
    return validate_balanced_payload(value)


def write_balanced_controller(path: str | Path, config: BalancedControllerConfig) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(
        json.dumps(config.payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


__all__ = [
    "BALANCED_CONTROLLER_SCHEMA",
    "BalancedControllerConfig",
    "DEFAULT_BALANCED_CONFIG",
    "REGISTERED_CONFLICT_THRESHOLDS",
    "REGISTERED_PRUNER_THRESHOLDS",
    "load_balanced_controller",
    "validate_balanced_payload",
    "write_balanced_controller",
]
