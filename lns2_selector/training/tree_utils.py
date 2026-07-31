from __future__ import annotations

import collections
from typing import Any


def balanced_map_folds(
    rows: list[dict[str, Any]], count: int = 4
) -> list[dict[str, Any]]:
    """Build deterministic layout-balanced train/validation map folds."""

    map_layout: dict[str, str] = {}
    for row in rows:
        map_id = str(row["map_id"])
        layout = str(row.get("layout_mode", "unknown"))
        previous = map_layout.setdefault(map_id, layout)
        if previous != layout:
            raise ValueError(f"map {map_id} has inconsistent layouts")
    if len(map_layout) < count:
        raise ValueError(f"OOF requires at least {count} training maps")
    by_layout: dict[str, list[str]] = collections.defaultdict(list)
    for map_id, layout in map_layout.items():
        by_layout[layout].append(map_id)
    validation: list[list[str]] = [[] for _ in range(count)]
    for layout in sorted(by_layout):
        for index, map_id in enumerate(sorted(by_layout[layout])):
            validation[index % count].append(map_id)
    all_maps = set(map_layout)
    folds = []
    for index, maps in enumerate(validation):
        if not maps:
            raise ValueError("OOF produced an empty validation fold")
        folds.append(
            {
                "fold": index,
                "train_maps": sorted(all_maps - set(maps)),
                "validation_maps": sorted(maps),
            }
        )
    return folds


def histogram_trees(estimator: Any) -> list[list[dict[str, Any]]]:
    """Convert a fitted sklearn histogram ensemble to the portable schema."""

    trees = []
    for stage in estimator._predictors:
        if len(stage) != 1:
            raise ValueError("portable scalar model requires one tree per stage")
        nodes = []
        for node in stage[0].nodes:
            if bool(node["is_categorical"]):
                raise ValueError("portable scalar model does not support categorical nodes")
            nodes.append(
                {
                    "value": float(node["value"]),
                    "feature_idx": int(node["feature_idx"]),
                    "num_threshold": float(node["num_threshold"]),
                    "missing_go_to_left": bool(node["missing_go_to_left"]),
                    "left": int(node["left"]),
                    "right": int(node["right"]),
                    "is_leaf": bool(node["is_leaf"]),
                }
            )
        trees.append(nodes)
    return trees
