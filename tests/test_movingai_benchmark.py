from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from experiments.feasibility_benchmark import (
    BenchmarkCase,
    parse_gpbs_result,
    parse_lns2_result,
    run_benchmark,
    solver_command,
)
from experiments.movingai_devset import fetch_devset


def _archive(path: Path, member: str, content: str) -> str:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(member, content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MovingAIDevsetTests(unittest.TestCase):
    def test_fetch_checks_and_extracts_exact_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_zip = root / "map.zip"
            scenario_zip = root / "scenario.zip"
            map_hash = _archive(map_zip, "tiny.map", "type octile\nheight 1\nwidth 1\nmap\n.\n")
            scenario_hash = _archive(
                scenario_zip,
                "scen-random/tiny-random-1.scen",
                "version 1\n0\ttiny.map\t1\t1\t0\t0\t0\t0\t0\n",
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": "test",
                        "benchmarks": [
                            {
                                "id": "tiny",
                                "agent_counts": [1],
                                "map_archive": {
                                    "url": map_zip.as_uri(),
                                    "sha256": map_hash,
                                    "member": "tiny.map",
                                },
                                "scenario_archive": {
                                    "url": scenario_zip.as_uri(),
                                    "sha256": scenario_hash,
                                    "member": "scen-random/tiny-random-1.scen",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = fetch_devset(config, root / "output")
            self.assertEqual(manifest[0]["id"], "tiny")
            self.assertTrue((root / "output" / manifest[0]["map_file"]).is_file())
            self.assertTrue((root / "output" / manifest[0]["scenario_file"]).is_file())

    def test_fetch_extracts_multiple_pinned_scenario_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_zip = root / "map.zip"
            scenario_zip = root / "scenario.zip"
            map_hash = _archive(
                map_zip,
                "tiny.map",
                "type octile\nheight 1\nwidth 2\nmap\n..\n",
            )
            with zipfile.ZipFile(scenario_zip, "w") as bundle:
                for index in (1, 2):
                    bundle.writestr(
                        f"scen-random/tiny-random-{index}.scen",
                        "version 1\n0\ttiny.map\t2\t1\t0\t0\t1\t0\t1\n",
                    )
            scenario_hash = hashlib.sha256(scenario_zip.read_bytes()).hexdigest()
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": "test",
                        "scenario_indices": [1, 2],
                        "benchmarks": [
                            {
                                "id": "tiny",
                                "agent_counts": [1],
                                "map_archive": {
                                    "url": map_zip.as_uri(),
                                    "sha256": map_hash,
                                    "member": "tiny.map",
                                },
                                "scenario_archive": {
                                    "url": scenario_zip.as_uri(),
                                    "sha256": scenario_hash,
                                    "member": "scen-random/tiny-random-1.scen",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = fetch_devset(config, root / "output")
            self.assertEqual(
                [row["index"] for row in manifest[0]["scenarios"]], [1, 2]
            )
            for row in manifest[0]["scenarios"]:
                self.assertTrue((root / "output" / row["file"]).is_file())


class FeasibilityRunnerTests(unittest.TestCase):
    def test_parses_lns2_and_gpbs_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "lns"
            init_path = Path(str(prefix) + "-initLNS.csv")
            with init_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "runtime",
                        "num of collisions",
                        "solution cost",
                        "initial collisions",
                        "area under curve",
                        "preprocessing runtime",
                        "LL expanded nodes",
                        "LL generated",
                        "LL runs",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "runtime": "1.5",
                        "num of collisions": "0",
                        "solution cost": "42",
                        "initial collisions": "7",
                        "area under curve": "9.5",
                        "preprocessing runtime": "0.25",
                        "LL expanded nodes": "10",
                        "LL generated": "20",
                        "LL runs": "3",
                    }
                )
            lns2 = parse_lns2_result(prefix, 0)
            self.assertTrue(lns2["success"])
            self.assertEqual(lns2["time_to_feasible"], 1.75)

            gpbs_path = root / "gpbs.csv"
            with gpbs_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "runtime",
                        "solution cost",
                        "#low-level expanded",
                        "#low-level generated",
                        "#low-level search calls",
                        "#high-level expanded",
                        "#high-level generated",
                        "preprocessing runtime",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "runtime": "2",
                        "solution cost": "-1",
                        "#low-level expanded": "11",
                        "#low-level generated": "21",
                        "#low-level search calls": "4",
                        "#high-level expanded": "5",
                        "#high-level generated": "6",
                        "preprocessing runtime": "0.5",
                    }
                )
            self.assertFalse(parse_gpbs_result(gpbs_path)["success"])

    def test_common_case_parameters_reach_both_commands(self) -> None:
        case = BenchmarkCase("tiny", Path("map.map"), Path("test.scen"), 12, 3)
        for solver in ("lns2_repair", "gpbs"):
            command, _ = solver_command(
                solver, Path("solver"), case, 17, Path("output")
            )
            self.assertIn(str(case.map_path), command)
            self.assertIn(str(case.scenario_path), command)
            self.assertIn("12", command)
            self.assertIn("17", command)
            self.assertIn("3", command)

    def test_resume_rejects_configuration_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "unused.map").write_text("map", encoding="utf-8")
            (dataset / "unused.scen").write_text("version 1", encoding="utf-8")
            (dataset / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "id": "empty",
                        "map_file": "unused.map",
                        "scenario_file": "unused.scen",
                        "agent_counts": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            binary = root / "solver"
            binary.write_bytes(b"placeholder")
            output = root / "output"
            run_benchmark(
                dataset,
                output,
                {"lns2_repair": binary},
                [0],
                10,
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                run_benchmark(
                    dataset,
                    output,
                    {"lns2_repair": binary},
                    [0],
                    20,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
