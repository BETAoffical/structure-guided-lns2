from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hybridstructpool_runtime_pilot import (  # noqa: E402
    analyze,
    collect,
)


DEFAULT_CONFIG = ROOT / "configs/stride_hybridstructpool_runtime_pilot_v1_registration.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "analyze"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "collect":
        result = collect(
            arguments.config.resolve(), arguments.output.resolve(), resume=arguments.resume
        )
    else:
        result = analyze(arguments.config.resolve(), arguments.output.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

