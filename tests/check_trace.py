from __future__ import annotations

import json
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
        assert len(row["after"]["conflict_edges"]) == row["after"][
            "num_of_colliding_pairs"
        ]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
