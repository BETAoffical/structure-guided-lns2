from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_shellbudget_audit import analyze_shellbudget  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ranker-free budgets over the equal four-size pool"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "stride_shellbudget_audit_v1_registration.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    report = analyze_shellbudget(
        arguments.config,
        arguments.output,
        workers=arguments.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
