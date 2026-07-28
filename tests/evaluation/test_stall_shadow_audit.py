from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.stall_shadow import StallShadowState, load_stall_shadow_config
from experiments.stall_shadow_audit import audit_stall_shadow_collection


RAW_CONFIG = {
    "schema": "lns2.stall_shadow.v2",
    "schema_version": 2,
    "mode": "shadow",
    "unchanged_attempt_thresholds": [1],
    "minimum_distinct_pp_attempts": 2,
    "future_observation_decisions": 1,
    "maximum_false_trigger_rate": 1.0,
    "post_state_change_cooldown_decisions": 2,
    "deployment_enabled": False,
}
CANDIDATES = [
    {"candidate_id": "winner", "agents": [1, 2], "actual_size": 2},
    {"candidate_id": "backup", "agents": [3, 4], "actual_size": 2},
]


def _events() -> list[dict]:
    state = StallShadowState(load_stall_shadow_config(RAW_CONFIG))
    events = []
    for decision_index, seed in enumerate((11, 12, 13)):
        _selected, before = state.before_selection(
            CANDIDATES,
            [2.0, 1.0],
            0,
            before_fingerprint="same",
            decision_index=decision_index,
        )
        observed = state.observe(
            before_fingerprint="same",
            after_fingerprint="same",
            replan_success=False,
            conflicts_before=10,
            conflicts_after=10,
            feasible=False,
            candidate_id="winner",
            actual_agents=[1, 2],
            step_random_seed=seed,
            requested_pp_seed=seed,
            applied_pp_seed=seed,
            repair_order=[1, 2],
        )
        events.append(
            {
                "event": "transition",
                "decision_index": decision_index,
                "action": {"random_seed": seed},
                "metrics": {
                    "requested_pp_random_seed": seed,
                    "applied_pp_random_seed": seed,
                    "repair_order": [1, 2],
                },
                "controller": {"stall_shadow": {**before, **observed}},
            }
        )
    events.append(
        {"event": "finish", "summary": {"stall_shadow": state.summary()}}
    )
    return events


class StallShadowAuditTests(unittest.TestCase):
    def _source(self, directory: str) -> tuple[Path, Path]:
        root = Path(directory) / "source"
        output = Path(directory) / "report"
        root.mkdir()
        (root / "run_config.json").write_text(
            json.dumps(
                {
                    "controller": "v2-stall-shadow",
                    "run_fingerprint": "run",
                    "configuration": {
                        "cohort_job_keys_override": [["task", 1]],
                        "stall_shadow_config": RAW_CONFIG,
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest = {
            "status": "ok",
            "policy": "realized_dynamic",
            "task_id": "task",
            "solver_seed": 1,
            "agent_count": 600,
        }
        (root / "realized_dynamic_manifest.jsonl").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        return root, output

    def test_recomputes_complete_threshold_evidence_without_promoting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, output = self._source(directory)
            with patch(
                "experiments.stall_shadow_audit.validate_manifest_trace",
                return_value=(root / "trace", _events(), None),
            ):
                report = audit_stall_shadow_collection(root, output)
        self.assertEqual(report["most_conservative_passing_threshold"], 1)
        self.assertTrue(report["shadow_integrity_passed"])
        self.assertFalse(report["deployment_promoted"])
        self.assertEqual(report["decision"], "shadow_candidate_threshold_found")
        self.assertEqual(report["expected_episode_count"], 1)

    def test_rejects_tampered_runtime_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, output = self._source(directory)
            events = _events()
            events[-1]["summary"]["stall_shadow"]["decision_count"] = 2
            with patch(
                "experiments.stall_shadow_audit.validate_manifest_trace",
                return_value=(root / "trace", events, None),
            ):
                with self.assertRaisesRegex(ValueError, "decision count"):
                    audit_stall_shadow_collection(root, output)

    def test_rejects_incomplete_registered_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, output = self._source(directory)
            run = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
            run["configuration"]["cohort_job_keys_override"].append(["missing", 1])
            (root / "run_config.json").write_text(json.dumps(run), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registered cohort"):
                audit_stall_shadow_collection(root, output)


if __name__ == "__main__":
    unittest.main()
