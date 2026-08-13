from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_causaltopopool import (  # noqa: E402
    materialize_causaltopopool_cohort,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STRIDE CausalTopoPool v1")
    parser.add_argument(
        "--design",
        default=str(ROOT / "configs" / "stride_causaltopopool_v1_design.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "build" / "stride-causaltopopool-v1-compactness"),
    )
    arguments = parser.parse_args()
    report = materialize_causaltopopool_cohort(arguments.design, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
