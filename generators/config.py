from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def merge_dicts(
    base: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def sample_int(
    value: int | list[int], rng: random.Random, name: str
) -> int:
    if isinstance(value, int):
        return value
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be an integer or [minimum, maximum]")
    minimum, maximum = value
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        raise ValueError(f"{name} range values must be integers")
    if minimum > maximum:
        raise ValueError(f"{name} minimum exceeds maximum")
    return rng.randint(minimum, maximum)


def sample_float(
    value: float | int | list[float],
    rng: random.Random,
    name: str,
) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a number or [minimum, maximum]")
    minimum, maximum = map(float, value)
    if minimum > maximum:
        raise ValueError(f"{name} minimum exceeds maximum")
    return rng.uniform(minimum, maximum)


def sample_choice(
    value: str | list[str], rng: random.Random, name: str
) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a string or non-empty string list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} choices must be strings")
    return rng.choice(value)


def weighted_choice(
    weights: dict[str, float], rng: random.Random, name: str
) -> str:
    if not weights:
        raise ValueError(f"{name} must not be empty")
    choices = list(weights)
    values = [float(weights[choice]) for choice in choices]
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError(f"{name} must contain positive weights")
    return rng.choices(choices, weights=values, k=1)[0]
