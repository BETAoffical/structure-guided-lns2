from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run(executable: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    executables = (Path(sys.argv[1]), Path(sys.argv[2]))
    official = executables[1]
    map_path = Path(sys.argv[3])
    scenario_path = Path(sys.argv[4])
    for executable in executables:
        completed = run(
            executable,
            "--map",
            str(map_path),
            "--agents",
            str(scenario_path),
            "--agentNum",
            "1",
            "--seed",
            "-1",
        )
        assert completed.returncode == 2, (executable, completed)
        assert "seed must be non-negative" in completed.stderr, (
            executable,
            completed.stderr,
        )

    missing_moving_ai_count = run(
        official,
        "--map",
        str(map_path),
        "--agents",
        str(scenario_path),
        "--seed",
        "0",
    )
    assert missing_moving_ai_count.returncode == 2, missing_moving_ai_count
    assert (
        "greater than zero for MovingAI" in missing_moving_ai_count.stderr
    ), missing_moving_ai_count.stderr

    data_root = Path(__file__).resolve().parent / "data"
    custom_map = data_root / "corrected_native_target.map"
    custom_scenario = data_root / "corrected_native_target.scen"
    resolved_custom = run(
        official,
        "--map",
        str(custom_map),
        "--agents",
        str(custom_scenario),
        "--cutoffTime",
        "5",
        "--maxIterations",
        "0",
        "--repairOnly",
        "true",
        "--seed",
        "0",
    )
    assert resolved_custom.returncode == 0, resolved_custom

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        invalid_headers = ("0\n", "-1\n", "3junk\n", "3,garbage\n")
        for index, header in enumerate(invalid_headers):
            invalid_scenario = root / f"invalid-header-{index}.scen"
            invalid_scenario.write_text(header, encoding="utf-8")
            rejected = run(
                official,
                "--map",
                str(custom_map),
                "--agents",
                str(invalid_scenario),
                "--seed",
                "0",
            )
            assert rejected.returncode == 2, (header, rejected)
            assert "custom scenario" in rejected.stderr, (
                header,
                rejected.stderr,
            )


if __name__ == "__main__":
    main()
