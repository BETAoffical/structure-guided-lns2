from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.result_consolidation import (  # noqa: E402
    EvidenceVerificationError,
    run_result_consolidation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the frozen InitLNS evidence snapshot and Chinese report."
    )
    parser.add_argument(
        "--config", default="configs/result_consolidation.json"
    )
    parser.add_argument(
        "--output", default="artifacts/initlns-research-evidence-v1"
    )
    parser.add_argument(
        "--report", default="docs/INITLNS_RESEARCH_REPORT_ZH.md"
    )
    parser.add_argument(
        "--verify-build",
        action="store_true",
        help="Require every ignored formal JSON, SHA, metric, and Git commit to match.",
    )
    arguments = parser.parse_args()
    try:
        result = run_result_consolidation(
            arguments.config,
            arguments.output,
            arguments.report,
            repository_root=PROJECT_ROOT,
            verify_build=arguments.verify_build,
        )
    except EvidenceVerificationError as error:
        print(f"evidence verification failed:\n{error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
