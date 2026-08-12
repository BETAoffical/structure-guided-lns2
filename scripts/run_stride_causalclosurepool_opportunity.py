from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_causalclosurepool_opportunity import (  # noqa: E402
    analyze_causalclosure_trials,
    collect_causalclosure_trials,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CausalClosurePool opportunity audit")
    parser.add_argument("command", choices=("preflight", "collect", "analyze"))
    parser.add_argument(
        "--execution",
        default=str(ROOT / "configs" / "stride_causalclosurepool_v2_opportunity_execution.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-output")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "preflight":
        result = collect_causalclosure_trials(
            execution_path=arguments.execution,
            output=arguments.output,
            mode="preflight",
            resume=arguments.resume,
        )
    elif arguments.command == "collect":
        result = collect_causalclosure_trials(
            execution_path=arguments.execution,
            output=arguments.output,
            mode="full",
            resume=arguments.resume,
            preflight_output=arguments.preflight_output,
        )
    else:
        result = analyze_causalclosure_trials(
            execution_path=arguments.execution,
            collection=arguments.output,
            output=arguments.output,
            mode="full",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
