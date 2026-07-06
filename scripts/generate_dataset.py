from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.config import load_json  # noqa: E402
from generators.dataset import generate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a split structured warehouse MAPF dataset."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    summary = generate_dataset(
        load_json(arguments.config), arguments.output
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
