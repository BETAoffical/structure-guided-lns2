from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.neighborhood.movingai_probe_quality import audit_probe_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit effective sample size and label stability in the MovingAI probe."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--probe-config", default="research/configs/neighborhood/movingai_mechanism_probe_dataset.json"
    )
    parser.add_argument(
        "--quality-config", default="research/configs/neighborhood/movingai_probe_quality_audit.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Audit recovered complete episodes without requiring registered volume.",
    )
    parser.add_argument(
        "--invalidate-runtime",
        action="store_true",
        help="Mark runtime sensitivity invalid while retaining hardware-independent outcomes.",
    )
    arguments = parser.parse_args()
    report = audit_probe_quality(
        arguments.collection,
        arguments.probe_config,
        arguments.quality_config,
        arguments.output,
        require_complete=not arguments.allow_partial,
        runtime_metrics_valid=not arguments.invalidate_runtime,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
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
