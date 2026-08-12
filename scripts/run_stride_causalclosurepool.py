from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_causalclosurepool import materialize_causalclosure_cohort  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STRIDE CausalClosurePool v2")
    parser.add_argument(
        "--design",
        default=str(ROOT / "configs" / "stride_causalclosurepool_v2_r3_design.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "build" / "stride-causalclosurepool-v2-r3-compactness"),
    )
    arguments = parser.parse_args()
    report = materialize_causalclosure_cohort(arguments.design, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
