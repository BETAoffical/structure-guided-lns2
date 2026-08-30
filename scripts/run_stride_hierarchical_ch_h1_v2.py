from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hierarchical_ch_h1_v2 import run_preflight  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "stride_hierarchical_ch_h1_v2.json"
DEFAULT_OUTPUT = ROOT / "build" / "stride-hierarchical-ch-h1-v2"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Register and checksum-verify the outcome-blind, map-disjoint "
            "hierarchical C/H H1 split. No episodes, H1 outcomes, or models "
            "are produced."
        )
    )
    parser.add_argument("phase", choices=("preflight",))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = run_preflight(args.config, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
