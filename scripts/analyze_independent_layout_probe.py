from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.independent_layout_probe import (  # noqa: E402
    run_independent_probe_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the preregistered independent-layout InitLNS probe."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--quality-config",
        default="configs/independent_layout_probe_quality.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument(
        "--reference-dataset",
        action="append",
        default=[],
        help="Dataset root whose generated map/task seeds must not overlap.",
    )
    arguments = parser.parse_args()
    report = run_independent_probe_analysis(
        arguments.dataset,
        arguments.collection,
        arguments.quality_config,
        arguments.output,
        qualification_only=arguments.qualification_only,
        reference_datasets=arguments.reference_dataset,
    )
    print(
        json.dumps(
            {
                "decision": report.get("decision", "qualification_gate"),
                "passed": report["passed"],
                "output": str(Path(arguments.output).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
