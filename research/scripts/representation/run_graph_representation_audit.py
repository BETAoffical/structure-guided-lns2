from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.representation.graph_representation_audit import (  # noqa: E402
    run_graph_representation_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit InitLNS graph and agent-set representations.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default="research/configs/representation/graph_representation_audit.json")
    parser.add_argument("--output", default="build/initlns-graph-representation-audit-v2")
    parser.add_argument(
        "--phase", choices=("index", "cross_validate", "validate", "all"), default="all"
    )
    parser.add_argument(
        "--equivalence-reference",
        default=None,
        help="Optional completed graph-audit output to compare against after the run.",
    )
    args = parser.parse_args()
    result = run_graph_representation_audit(
        args.project_root,
        args.config,
        args.output,
        phase=args.phase,
        equivalence_reference=args.equivalence_reference,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
