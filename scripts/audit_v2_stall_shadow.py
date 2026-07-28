from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stall_shadow_audit import (  # noqa: E402
    audit_stall_shadow_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate a diagnostic-only v2 stall-shadow collection."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = audit_stall_shadow_collection(arguments.source, arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
