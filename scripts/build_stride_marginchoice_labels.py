from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_marginchoice import build_marginchoice_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build signed seed-half-stable STRIDE-MarginChoice labels."
    )
    parser.add_argument(
        "--config", default="configs/stride_marginchoice_labels.json"
    )
    parser.add_argument(
        "--output", default="build/stride-marginchoice-labels-v1"
    )
    args = parser.parse_args()
    report = build_marginchoice_labels(config_path=args.config, output=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
