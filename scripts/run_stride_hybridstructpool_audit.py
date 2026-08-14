from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hybridstructpool_audit import (  # noqa: E402
    analyze_hybridstructpool,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the zero-solver HybridStructPool v1 candidate contract"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "stride_hybridstructpool_v1_registration.json"),
    )
    parser.add_argument(
        "--output", default=str(ROOT / "build" / "stride-hybridstructpool-v1-audit")
    )
    args = parser.parse_args()
    report = analyze_hybridstructpool(Path(args.config), Path(args.output))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
