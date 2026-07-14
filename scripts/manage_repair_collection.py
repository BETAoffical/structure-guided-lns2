from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.repair_collection import (  # noqa: E402
    cancel_collection,
    collection_status,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or cancel one fingerprinted repair collection."
    )
    parser.add_argument("action", choices=("status", "cancel"))
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = (
        collection_status(arguments.output)
        if arguments.action == "status"
        else cancel_collection(arguments.output)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
