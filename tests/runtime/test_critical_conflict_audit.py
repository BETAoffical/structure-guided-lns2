from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.critical_conflict_audit import run_critical_conflict_audit


def _state_row(state_id: str, split: str) -> dict:
    state = {
        "rows": 2,
        "cols": 3,
        "obstacles": [0, 0, 0, 0, 0, 0],
        "conflict_edges": [[0, 1], [1, 2], [2, 3]],
        "agents": [
            {
                "id": index,
                "conflict_degree": 2 if index in {1, 2} else 1,
                "delay": index,
                "path": [index, index + 1],
            }
            for index in range(4)
        ],
    }
    return {
        "state_id": state_id,
        "split": split,
        "map_id": f"map-{split}",
        "layout_mode": "synthetic",
        "agent_count": 4,
        "state_hash": "0" * 64,
        "state": state,
        "edge_ages": {(0, 1): 1, (1, 2): 1, (2, 3): 1},
        "candidate_pool": [
            {"candidate_id": "winner", "seed_agents": [99]},
        ],
        "v2_winner": "winner",
        "outcome_winner": "winner",
    }


class CriticalConflictAuditTests(unittest.TestCase):
    def test_failed_gate_writes_evidence_without_runtime_config(self) -> None:
        states = [
            _state_row("train", "policy_train"),
            _state_row("diagnostic", "policy_validation"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit"
            output.mkdir()
            stale_config = output / "v2_critical_config.json"
            stale_config.write_text("{}", encoding="utf-8")
            with patch(
                "experiments.critical_conflict_audit._load_states",
                return_value=(states, {}),
            ):
                report = run_critical_conflict_audit(
                    Path(temporary) / "source", output
                )
            self.assertEqual(report["decision"], "keep_v2_full")
            self.assertFalse(report["deployment_promoted"])
            self.assertTrue((output / "critical_config_grid.csv").is_file())
            self.assertTrue((output / "critical_state_rows.csv").is_file())
            self.assertTrue(
                (output / "critical_conflict_audit_report.json").is_file()
            )
            self.assertFalse(stale_config.exists())


if __name__ == "__main__":
    unittest.main()
