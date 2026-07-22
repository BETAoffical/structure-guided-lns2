from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v3_feature_audit import audit_v3_features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit v3 feature stability without rerunning PP or the solver."
    )
    parser.add_argument(
        "--source", default="build/initlns-v3-pilot-v1"
    )
    parser.add_argument(
        "--output", default="build/initlns-v3-feature-audit-v3"
    )
    arguments = parser.parse_args()
    report = audit_v3_features(arguments.source, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
