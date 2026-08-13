from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_structshell_audit import analyze_structshell  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit StructPool size shells and outcome-blind cutpoints"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "stride_structshell_audit_v1_registration.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    report = analyze_structshell(
        arguments.config,
        arguments.output,
        workers=arguments.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
