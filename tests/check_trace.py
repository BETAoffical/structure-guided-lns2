from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 3
    assert rows[0]["event"] == "initial"
    assert rows[-1]["event"] == "finish"
    assert all(row["schema_version"] == 1 for row in rows)
    transitions = [row for row in rows if row["event"] == "transition"]
    assert transitions
    for row in transitions:
        assert row["action"]["mode"] == "official"
        assert row["action_valid"] is True
        assert row["metrics"]["conflicts_before"] >= 0
        assert row["metrics"]["conflicts_after"] >= 0
        metrics = row["metrics"]
        timing_names = (
            "native_step_seconds",
            "native_neighborhood_generation_seconds",
            "native_replan_seconds",
            "native_state_snapshot_seconds",
            "native_repair_bookkeeping_seconds",
            "native_residual_seconds",
        )
        assert all(
            math.isfinite(float(metrics[name])) and float(metrics[name]) >= 0.0
            for name in timing_names
        )
        partition = sum(float(metrics[name]) for name in timing_names[1:])
        assert math.isclose(
            partition,
            float(metrics["native_step_seconds"]),
            rel_tol=0.01,
            abs_tol=max(1e-6, 0.01 * float(metrics["native_step_seconds"])),
        )
        assert math.isclose(
            float(metrics["pp_replan_seconds"]),
            float(metrics["native_replan_seconds"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        assert len(row["after"]["conflict_edges"]) == row["after"][
            "num_of_colliding_pairs"
        ]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
