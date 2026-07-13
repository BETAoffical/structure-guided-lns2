from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.repair_quality import write_quality_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze repair calibration coverage and outcome quality."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    parser.add_argument("--expected-qualification-runs", type=int, default=72)
    parser.add_argument("--expected-baseline-episodes", type=int, default=288)
    parser.add_argument("--expected-source-episodes", type=int, default=72)
    parser.add_argument("--minimum-states", type=int, default=72)
    parser.add_argument("--minimum-outcomes", type=int, default=1296)
    parser.add_argument("--minimum-informative-rate", type=float, default=0.6)
    parser.add_argument("--maximum-family-dominance", type=float, default=0.8)
    arguments = parser.parse_args()
    report = write_quality_report(
        arguments.collection,
        json_path=arguments.output_json,
        markdown_path=arguments.output_markdown,
        expected_qualification_runs=arguments.expected_qualification_runs,
        expected_baseline_episodes=arguments.expected_baseline_episodes,
        expected_source_episodes=arguments.expected_source_episodes,
        minimum_states=arguments.minimum_states,
        minimum_outcomes=arguments.minimum_outcomes,
        minimum_informative_rate=arguments.minimum_informative_rate,
        maximum_family_dominance=arguments.maximum_family_dominance,
    )
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
