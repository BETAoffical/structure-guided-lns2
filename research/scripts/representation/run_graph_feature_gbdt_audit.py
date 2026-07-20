from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.representation.graph_feature_gbdt_audit import (  # noqa: E402
    run_graph_feature_gbdt_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit deterministic conflict-graph features with the registered GBDT."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default="research/configs/representation/graph_feature_gbdt_audit.json")
    parser.add_argument("--output", default="build/initlns-graph-feature-gbdt-audit-v1")
    parser.add_argument("--phase", choices=("index", "train", "all"), default="all")
    args = parser.parse_args()
    result = run_graph_feature_gbdt_audit(
        args.project_root, args.config, args.output, phase=args.phase
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
