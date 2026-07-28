from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    executable, map_path, scenario_path, trace_path = sys.argv[1:]
    trace = Path(trace_path)
    trace.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            executable,
            "--map",
            map_path,
            "--agents",
            scenario_path,
            "--agentNum",
            "80",
            "--cutoffTime",
            "0",
            "--seed",
            "29",
            "--trace",
            str(trace),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1, completed
    rows = [
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["event"] for row in rows] == ["initial", "finish"]
    assert rows[-1]["success"] is False
    for row in rows:
        state = row["state"]
        assert state["initial_solution_complete"] is False
        assert state["feasible"] is False
        assert state["done"] is True

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        partial_trace = temporary / "partial-initial-solution.jsonl"
        partial_result = subprocess.run(
            [
                executable,
                "--map",
                map_path,
                "--agents",
                scenario_path,
                "--agentNum",
                "200",
                "--cutoffTime",
                "0.01",
                "--seed",
                "29",
                "--trace",
                str(partial_trace),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert partial_result.returncode == 1, partial_result
        partial_rows = [
            json.loads(line)
            for line in partial_trace.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        partial_state = partial_rows[-1]["state"]
        path_costs = [
            int(agent["path_cost"])
            for agent in partial_state["agents"]
            if int(agent["path_cost"]) >= 0
        ]
        assert path_costs, "partial-timeout fixture produced no paths"
        assert partial_state["sum_of_costs"] == sum(path_costs)

        missing_scenario = temporary / "must-not-be-created.scen"
        missing_result = subprocess.run(
            [
                executable,
                "--map",
                map_path,
                "--agents",
                str(missing_scenario),
                "--agentNum",
                "2",
                "--cutoffTime",
                "0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert missing_result.returncode == 2, missing_result
        assert "scenario_path" in missing_result.stderr, missing_result
        assert not missing_scenario.exists()

        tab_header_map = temporary / "tab-header.map"
        tab_header_map.write_text(
            "type octile\n"
            "height\t3\n"
            "width 3\n"
            "map\n"
            "...\n"
            "...\n"
            "...\n",
            encoding="utf-8",
        )
        malformed_result = subprocess.run(
            [
                executable,
                "--map",
                str(tab_header_map),
                "--agents",
                scenario_path,
                "--agentNum",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert malformed_result.returncode == 2, malformed_result
        assert "map header is malformed" in malformed_result.stderr, (
            malformed_result
        )

        missing_agent_count = subprocess.run(
            [
                executable,
                "--map",
                map_path,
                "--agents",
                scenario_path,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert missing_agent_count.returncode == 2, missing_agent_count
        assert "agent_count must be greater than zero" in (
            missing_agent_count.stderr
        ), missing_agent_count

        invalid_cutoff = subprocess.run(
            [
                executable,
                "--map",
                map_path,
                "--agents",
                scenario_path,
                "--agentNum",
                "1",
                "--cutoffTime",
                "-1",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert invalid_cutoff.returncode == 2, invalid_cutoff
        assert "cutoffTime must be finite and non-negative" in (
            invalid_cutoff.stderr
        ), invalid_cutoff

        invalid_algorithm = subprocess.run(
            [
                executable,
                "--map",
                map_path,
                "--agents",
                scenario_path,
                "--agentNum",
                "1",
                "--replanAlgo",
                "not-an-algorithm",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert invalid_algorithm.returncode == 2, invalid_algorithm
        assert "replanAlgo must be" in invalid_algorithm.stderr, (
            invalid_algorithm
        )

        unavailable_trace = temporary / "missing-parent" / "trace.jsonl"
        trace_error = subprocess.run(
            [
                executable,
                "--map",
                map_path,
                "--agents",
                scenario_path,
                "--agentNum",
                "1",
                "--trace",
                str(unavailable_trace),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert trace_error.returncode == 2, trace_error
        assert "failed to open repair trace" in trace_error.stderr, trace_error

        full_device = Path("/dev/full")
        if full_device.exists():
            write_error = subprocess.run(
                [
                    executable,
                    "--map",
                    map_path,
                    "--agents",
                    scenario_path,
                    "--agentNum",
                    "1",
                    "--trace",
                    str(full_device),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert write_error.returncode == 2, write_error
            assert "failed to write repair trace" in write_error.stderr, write_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
