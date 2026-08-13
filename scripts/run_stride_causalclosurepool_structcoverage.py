from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_causalclosurepool_structcoverage import (  # noqa: E402
    analyze_structural_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit legacy StructPool coverage in CausalClosurePool"
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs" / "stride_causalclosurepool_structcoverage_v1.json"
        ),
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = analyze_structural_coverage(arguments.config, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
