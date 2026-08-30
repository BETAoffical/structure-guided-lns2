from __future__ import annotations

from pathlib import Path

from experiments.mixed_full_v2_high_load_multiseed_v1 import (
    CONTROLLERS,
    _common_collection_kwargs,
)


def test_collection_kwargs_preserve_each_controller_identity() -> None:
    for controller in CONTROLLERS:
        kwargs = _common_collection_kwargs(
            controller=controller,
            dataset=Path("dataset"),
            config=Path("config.json"),
            output=Path("output"),
            task_ids=["task"],
            keys={("task", 1)},
            bundle=Path("bundle"),
        )
        assert kwargs["controller"] == controller
