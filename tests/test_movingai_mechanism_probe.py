from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.movingai_mechanism_probe import (
    prepare_probe_dataset,
    summarize_probe_records,
)


def _settings() -> dict:
    return {
        "schema_version": 1,
        "cases": [
            {"benchmark_id": "map-a", "agent_counts": [2]},
            {"benchmark_id": "map-b", "agent_counts": [2]},
        ],
        "analysis": {
            "permutations": 20,
            "minimum_repairable_rate": 0.5,
            "minimum_state_count": 2,
            "minimum_action_diversity_rate": 0.5,
            "minimum_neighborhood_diversity_rate": 0.5,
            "minimum_action_effect_ratio": 0.5,
            "maximum_fixed_unique_pareto_share": 0.8,
            "context_permutation_percentile": 0.95,
        },
    }


def _collection_config() -> dict:
    return {
        "solver_seeds": [0],
        "policies": ["official_adaptive"],
        "counterfactual": {
            "source_policy": "official_adaptive",
            "heuristics": ["target", "collision"],
            "neighborhood_sizes": [4],
            "trials": 2,
        },
    }


def _state(map_id: str) -> dict:
    episode_id = f"{map_id}__episode"
    return {
        "state_id": f"{episode_id}__decision_0000",
        "state_fingerprint": f"fingerprint-{map_id}",
        "episode_id": episode_id,
        "candidate_count": 2,
        "state": {
            "context": {
                "map_id": map_id,
                "task_id": f"{map_id}__agents_0002",
                "agent_count": 2,
            }
        },
    }


def _outcome(state: dict, heuristic: str, trial: int, conflicts: int) -> dict:
    neighborhood = [0] if heuristic == "target" else [1]
    return {
        "state_id": state["state_id"],
        "candidate_action": {
            "seed_agent": 0,
            "heuristic": heuristic,
            "neighborhood_size": 4,
        },
        "trial_index": trial,
        "action_valid": True,
        "steps": [
            {"step": 0},
            {"step": 1, "metrics": {"neighborhood": neighborhood}},
        ],
        "horizon_outcomes": [
            {
                "horizon": 1,
                "available": True,
                "solved": conflicts == 0,
                "conflicts_after": conflicts,
                "conflict_auc": float(conflicts),
                "low_level_delta": {"generated": 10 + conflicts},
            }
        ],
    }


def _probe_records() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    qualification = []
    baseline = []
    states = [_state("map-a"), _state("map-b")]
    outcomes = []
    for state in states:
        map_id = state["state"]["context"]["map_id"]
        task_id = state["state"]["context"]["task_id"]
        qualification.append(
            {
                "status": "ok",
                "map_id": map_id,
                "task_id": task_id,
                "repairable": True,
            }
        )
        baseline.append(
            {
                "status": "ok",
                "policy": "official_adaptive",
                "summary": {"repairable": True},
            }
        )
        target_conflicts = 0 if map_id == "map-a" else 2
        collision_conflicts = 2 if map_id == "map-a" else 0
        for trial in (0, 1):
            outcomes.append(_outcome(state, "target", trial, target_conflicts))
            outcomes.append(
                _outcome(state, "collision", trial, collision_conflicts)
            )
    return qualification, baseline, states, outcomes


class MovingAIProbeDatasetTests(unittest.TestCase):
    def test_preparation_is_deterministic_and_validates_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "maps").mkdir(parents=True)
            (source / "scenarios").mkdir(parents=True)
            map_path = source / "maps" / "tiny.map"
            map_path.write_text(
                "type octile\nheight 1\nwidth 2\nmap\n..\n", encoding="utf-8"
            )
            scenario_path = source / "scenarios" / "tiny-random-1.scen"
            scenario_path.write_text(
                "version 1\n"
                "0\ttiny.map\t2\t1\t0\t0\t1\t0\t1\n"
                "0\ttiny.map\t2\t1\t1\t0\t0\t0\t1\n",
                encoding="utf-8",
            )
            manifest = {
                "id": "tiny",
                "map_file": "maps/tiny.map",
                "scenario_file": "scenarios/tiny-random-1.scen",
                "agent_counts": [1, 2],
                "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
                "scenario_sha256": hashlib.sha256(
                    scenario_path.read_bytes()
                ).hexdigest(),
            }
            (source / "manifest.jsonl").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cases": [
                            {"benchmark_id": "tiny", "agent_counts": [1, 2]}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "probe"
            first = prepare_probe_dataset(source, config, output)
            second = prepare_probe_dataset(source, config, output)
            self.assertEqual(first, second)
            rows = [
                json.loads(line)
                for line in (output / "probe" / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["agent_count"], 2)
            self.assertEqual(rows[0]["topology_metrics"]["free_cell_count"], 2)
            manifest["map_sha256"] = "0" * 64
            (source / "manifest.jsonl").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                prepare_probe_dataset(source, config, root / "corrupt")

    def test_preparation_expands_scenario_indices_without_task_id_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "maps").mkdir(parents=True)
            (source / "scenarios").mkdir(parents=True)
            map_path = source / "maps" / "tiny.map"
            map_path.write_text(
                "type octile\nheight 1\nwidth 2\nmap\n..\n", encoding="utf-8"
            )
            scenarios = []
            for index in (1, 2):
                scenario = source / "scenarios" / f"tiny-random-{index}.scen"
                scenario.write_text(
                    "version 1\n0\ttiny.map\t2\t1\t0\t0\t1\t0\t1\n",
                    encoding="utf-8",
                )
                scenarios.append(
                    {
                        "index": index,
                        "file": scenario.relative_to(source).as_posix(),
                        "sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
                    }
                )
            manifest = {
                "id": "tiny",
                "map_file": "maps/tiny.map",
                "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
                "scenario_file": scenarios[0]["file"],
                "scenario_sha256": scenarios[0]["sha256"],
                "scenarios": scenarios,
                "agent_counts": [1],
            }
            (source / "manifest.jsonl").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scenario_indices": [1, 2],
                        "cases": [{"benchmark_id": "tiny", "agent_counts": [1]}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "probe"
            summary = prepare_probe_dataset(source, config, output)
            rows = [
                json.loads(line)
                for line in (output / "probe" / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(summary["splits"]["probe"]["scenario_count"], 2)
            self.assertEqual(len({row["task_id"] for row in rows}), 2)
            self.assertEqual(
                {row["scenario_type"] for row in rows},
                {"movingai_random_1", "movingai_random_2"},
            )


class MovingAIMechanismAnalysisTests(unittest.TestCase):
    def test_mechanism_signal_separates_actions_and_avoids_fixed_dominance(self) -> None:
        qualification, baseline, states, outcomes = _probe_records()
        first = summarize_probe_records(
            qualification,
            baseline,
            states,
            outcomes,
            _settings(),
            _collection_config(),
            require_complete=True,
        )
        second = summarize_probe_records(
            qualification,
            baseline,
            states,
            outcomes,
            _settings(),
            _collection_config(),
            require_complete=True,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["integrity"]["passed"])
        self.assertEqual(first["mechanism"]["action_diversity_rate"], 1.0)
        self.assertEqual(first["mechanism"]["neighborhood_diversity_rate"], 1.0)
        self.assertEqual(first["mechanism"]["action_effect_ratio"], 1.0)
        self.assertEqual(
            first["mechanism"]["maximum_fixed_unique_pareto_share"], 0.5
        )

    def test_missing_trial_fails_integrity(self) -> None:
        qualification, baseline, states, outcomes = _probe_records()
        report = summarize_probe_records(
            qualification,
            baseline,
            states,
            outcomes[:-1],
            _settings(),
            _collection_config(),
            require_complete=True,
        )
        self.assertFalse(report["integrity"]["passed"])
        self.assertEqual(report["integrity"]["trial_mismatches"], 1)

    def test_duplicate_solver_states_are_pooled_as_trials(self) -> None:
        qualification, baseline, states, outcomes = _probe_records()
        duplicate = copy.deepcopy(states[0])
        duplicate["state_id"] = "map-a__duplicate__decision_0000"
        duplicate["episode_id"] = "map-a__duplicate"
        states.append(duplicate)
        baseline.append(
            {
                "status": "ok",
                "policy": "official_adaptive",
                "summary": {"repairable": True},
            }
        )
        for heuristic, conflicts in (("target", 0), ("collision", 2)):
            for trial in (0, 1):
                outcomes.append(_outcome(duplicate, heuristic, trial, conflicts))
        report = summarize_probe_records(
            qualification,
            baseline,
            states,
            outcomes,
            _settings(),
            _collection_config(),
            require_complete=False,
        )
        self.assertTrue(report["integrity"]["passed"])
        self.assertEqual(report["integrity"]["state_rows"], 3)
        self.assertEqual(report["integrity"]["unique_state_rows"], 2)
        self.assertEqual(report["integrity"]["duplicate_fingerprint_groups"], 1)
        self.assertEqual(report["mechanism"]["state_count"], 2)


if __name__ == "__main__":
    unittest.main()
