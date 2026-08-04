from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_certguard_training import run_certguard_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the map-grouped STRIDE-CertGuard uncertainty gate."
    )
    parser.add_argument(
        "--config", default="configs/stride_certguard_training.json"
    )
    parser.add_argument(
        "--output", default="build/stride-certguard-training-v1"
    )
    args = parser.parse_args()
    report = run_certguard_training(config_path=args.config, output=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
