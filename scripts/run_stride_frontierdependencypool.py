from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_frontierdependencypool import (  # noqa: E402
    materialize_frontierdependencypool,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs/stride_frontierdependencypool_v1_registration.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "build/stride-frontierdependencypool-v1"),
    )
    arguments = parser.parse_args()
    report = materialize_frontierdependencypool(arguments.config, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
