from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.movingai_devset import fetch_devset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the pinned MovingAI dev set.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "movingai_devset.json"),
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    manifest = fetch_devset(arguments.config, arguments.output)
    print(json.dumps({"benchmarks": len(manifest), "output": arguments.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
