from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_structshell_crossmap_offline_profile import (  # noqa: E402
    REPORT_FILENAME,
    analyze,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile three checkpoints from each frozen V2 cross-map trace "
            "without running a solver, controller, or PP."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    report = analyze(Path(args.source), output)
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "profile_state_count": report["contract"]["profile_state_count"],
                "decision": report["source_count_evidence"]["decision"],
                "additional_collection_allowed": report["source_count_evidence"][
                    "additional_collection_allowed"
                ],
                "router_training_allowed": report["router_training_allowed"],
                "report": str(output / REPORT_FILENAME),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
