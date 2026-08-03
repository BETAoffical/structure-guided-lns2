from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.repair_collection import _fingerprint
from experiments.stride_stage4r_pp_replay import (
    ACTION_LABELS,
    PP_TRIAL_INDICES,
    STRIDE_STAGE4R_PP_REPLAY_SELECTION_SCHEMA,
    STRIDE_STAGE4R_PP_REPLAY_STATE_SCHEMA,
    analyze_pp_replay,
    stage4r_paired_pp_seed,
    validate_pp_replay_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_stage4r_pp_replay.json"


def _synthetic_collection(root: Path) -> tuple[Path, Path]:
    selection_path = root / "selection.jsonl"
    output = root / "output"
    (output / "states").mkdir(parents=True)
    config_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    selection = []
    for state_index in range(16):
        repair_fingerprint = f"{state_index + 1:064x}"
        same_action = state_index < 10
        actions = {
            "v2-full": {
                "action_id": f"v2-{state_index}",
                "agents": [0, 1],
                "size": 2,
            },
            "stride-quality-v1": {
                "action_id": (
                    f"v2-{state_index}" if same_action else f"quality-{state_index}"
                ),
                "agents": [0, 1] if same_action else [2, 3],
                "size": 2,
            },
        }
        selection.append(
            {
                "schema": STRIDE_STAGE4R_PP_REPLAY_SELECTION_SCHEMA,
                "state_id": f"state-{state_index}",
                "artifact_file": f"state-{state_index}.json",
                "task_id": f"task-{state_index // 4}",
                "map_id": f"map-{state_index // 4}",
                "split": "balanced_wall_clock",
                "layout_mode": "synthetic",
                "agent_count": 100,
                "solver_seed": state_index % 4 + 1,
                "decision_index": 0,
                "before_fingerprint": f"{1000 + state_index:064x}",
                "before_repair_fingerprint": repair_fingerprint,
                "before_conflicts": 10,
                "actions": actions,
                "same_action": same_action,
                "source_trace": {},
                "source_outcomes_used_for_selection": False,
            }
        )
    selection_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selection),
        encoding="utf-8",
    )
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    identity = _fingerprint(
        {
            "schema": STRIDE_STAGE4R_PP_REPLAY_STATE_SCHEMA,
            "config_sha256": config_sha,
            "selection_sha256": selection_sha,
        }
    )
    for state_index, selected in enumerate(selection):
        trials = []
        for trial_index in PP_TRIAL_INDICES:
            seed = stage4r_paired_pp_seed(
                selected["before_repair_fingerprint"], trial_index
            )
            for label in ACTION_LABELS:
                if selected["same_action"]:
                    reduction = 4 + trial_index % 2
                    after_fingerprint = f"same-{state_index}-{trial_index}"
                    repair_order = [0, 1]
                    low_level = {"runs": 2, "generated": 10}
                else:
                    v2_wins = trial_index < 8
                    reduction = 6 if (label == "v2-full") == v2_wins else 3
                    after_fingerprint = f"{label}-{state_index}-{trial_index}"
                    repair_order = [0, 1] if label == "v2-full" else [2, 3]
                    low_level = {"runs": 2, "generated": 10 + reduction}
                trials.append(
                    {
                        "ordinal": len(trials),
                        "action_position": 0,
                        "action_label": label,
                        "action_id": selected["actions"][label]["action_id"],
                        "trial_index": trial_index,
                        "pp_seed": seed,
                        "before_repair_fingerprint": selected[
                            "before_repair_fingerprint"
                        ],
                        "replan_success": True,
                        "repair_outcome": "progress",
                        "feasible": False,
                        "conflicts_before": 10,
                        "conflicts_after": 10 - reduction,
                        "conflict_reduction": reduction,
                        "no_progress": False,
                        "after_repair_fingerprint": after_fingerprint,
                        "repair_order": repair_order,
                        "post_structure": {},
                        "pp_replan_seconds": 0.1,
                        "native_step_seconds": 0.11,
                        "low_level": low_level,
                    }
                )
        payload = {
            "schema": STRIDE_STAGE4R_PP_REPLAY_STATE_SCHEMA,
            "identity": identity,
            "complete": True,
            "state_id": selected["state_id"],
            "task_id": selected["task_id"],
            "map_id": selected["map_id"],
            "solver_seed": selected["solver_seed"],
            "before_fingerprint": selected["before_fingerprint"],
            "before_repair_fingerprint": selected["before_repair_fingerprint"],
            "before_conflicts": 10,
            "actions": selected["actions"],
            "same_action": selected["same_action"],
            "trials": trials,
        }
        (output / "states" / selected["artifact_file"]).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return selection_path, output


class StrideStage4RPPReplayTest(unittest.TestCase):
    def test_seed_is_state_paired_and_trial_distinct(self) -> None:
        first = [stage4r_paired_pp_seed("a" * 64, index) for index in PP_TRIAL_INDICES]
        second = [stage4r_paired_pp_seed("a" * 64, index) for index in PP_TRIAL_INDICES]
        other = [stage4r_paired_pp_seed("b" * 64, index) for index in PP_TRIAL_INDICES]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 16)
        self.assertNotEqual(first, other)

    def test_registration_rejects_changed_contract(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_pp_replay_config(config)
        for field, value in (
            ("formal_speed_claim", True),
            ("pp_trial_count", 8),
            ("workers", 2),
            ("decision_index", 1),
        ):
            changed = copy.deepcopy(config)
            changed[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_pp_replay_config(changed)

    def test_analysis_detects_material_pp_seed_winner_flips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, output = _synthetic_collection(Path(directory))
            report = analyze_pp_replay(CONFIG, selection, output)

        self.assertTrue(report["passed"])
        self.assertEqual(report["state_count"], 16)
        self.assertEqual(report["trial_count"], 512)
        self.assertEqual(report["same_action_state_count"], 10)
        self.assertEqual(report["different_action_state_count"], 6)
        self.assertEqual(report["diagnosis"]["winner_flip_state_count"], 6)
        self.assertTrue(report["diagnosis"]["pp_seed_material"])
        self.assertEqual(
            report["next_decision"],
            "retain_multiseed_label_and_model_action_uncertainty",
        )

    def test_analysis_rejects_nondeterministic_identical_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, output = _synthetic_collection(Path(directory))
            path = output / "states" / "state-0.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            quality = next(
                row
                for row in payload["trials"]
                if row["action_label"] == "stride-quality-v1"
                and row["trial_index"] == 0
            )
            quality["conflicts_after"] += 1
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = analyze_pp_replay(CONFIG, selection, output)

        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["same_action_exact_outcomes"])
        self.assertEqual(
            report["next_decision"],
            "repair_pp_replay_before_scientific_conclusion",
        )


if __name__ == "__main__":
    unittest.main()
