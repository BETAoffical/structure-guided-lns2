from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NATIVE = ROOT / "build" / "linux" / "project"
if NATIVE.is_dir() and str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from experiments.stride_warehouse_disruption_recovery import (  # noqa: E402
    analyze_checkpoints,
    plan,
    prepare_checkpoints,
)
from experiments.stride_warehouse_disruption_recovery_ttf import (  # noqa: E402
    analyze_ttf,
    collect_ttf,
    plan_ttf,
)
from experiments.stride_warehouse_disruption_recovery_heldout import (  # noqa: E402
    analyze_checkpoints as analyze_heldout_checkpoints,
    plan as plan_heldout,
    prepare_checkpoints as prepare_heldout_checkpoints,
)
from experiments.stride_warehouse_disruption_recovery_heldout_ttf import (  # noqa: E402
    analyze_ttf as analyze_heldout_ttf,
    collect_ttf as collect_heldout_ttf,
    plan_ttf as plan_heldout_ttf,
)
from experiments.stride_warehouse_disruption_recovery_boundary import (  # noqa: E402
    analyze_checkpoints as analyze_boundary_checkpoints,
    plan as plan_boundary,
    prepare_checkpoints as prepare_boundary_checkpoints,
)
from experiments.stride_warehouse_disruption_recovery_boundary_ttf import (  # noqa: E402
    analyze_ttf as analyze_boundary_ttf,
    collect_ttf as collect_boundary_ttf,
    plan_ttf as plan_boundary_ttf,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, materialize, or analyze frozen warehouse disruption "
            "checkpoints without running controller outcomes."
        )
    )
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "prepare-checkpoints",
            "analyze-checkpoints",
            "plan-ttf",
            "collect-ttf",
            "analyze-ttf",
            "plan-heldout",
            "prepare-heldout-checkpoints",
            "analyze-heldout-checkpoints",
            "plan-heldout-ttf",
            "collect-heldout-ttf",
            "analyze-heldout-ttf",
            "plan-boundary",
            "prepare-boundary-checkpoints",
            "analyze-boundary-checkpoints",
            "plan-boundary-ttf",
            "collect-boundary-ttf",
            "analyze-boundary-ttf",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    if args.command in {"plan", "plan-heldout", "plan-boundary"}:
        if args.output or args.workers is not None or args.resume:
            parser.error(f"{args.command} accepts only --config")
        if args.command == "plan":
            result = plan(config)
        elif args.command == "plan-heldout":
            result = plan_heldout(config)
        else:
            result = plan_boundary(config)
    else:
        if not args.output:
            parser.error(f"--output is required for {args.command}")
        output = Path(args.output)
        if args.command == "prepare-checkpoints":
            if args.workers is not None and args.workers <= 0:
                parser.error("--workers must be positive")
            result = prepare_checkpoints(
                config,
                output,
                workers=args.workers,
                resume=args.resume,
            )
        elif args.command == "analyze-checkpoints":
            if args.workers is not None or args.resume:
                parser.error(
                    "analyze-checkpoints does not accept --workers/--resume"
                )
            result = analyze_checkpoints(config, output)
        elif args.command == "plan-ttf":
            if args.workers is not None or args.resume:
                parser.error("plan-ttf does not accept --workers/--resume")
            result = plan_ttf(config, output)
        elif args.command == "collect-ttf":
            if args.workers is not None:
                parser.error("collect-ttf does not accept --workers")
            result = collect_ttf(config, output, resume=args.resume)
        elif args.command == "analyze-ttf":
            if args.workers is not None or args.resume:
                parser.error("analyze-ttf does not accept --workers/--resume")
            result = analyze_ttf(config, output)
        elif args.command == "prepare-heldout-checkpoints":
            if args.workers is not None and args.workers <= 0:
                parser.error("--workers must be positive")
            result = prepare_heldout_checkpoints(
                config,
                output,
                workers=args.workers,
                resume=args.resume,
            )
        elif args.command == "analyze-heldout-checkpoints":
            if args.workers is not None or args.resume:
                parser.error(
                    "analyze-heldout-checkpoints does not accept --workers/--resume"
                )
            result = analyze_heldout_checkpoints(config, output)
        elif args.command == "plan-heldout-ttf":
            if args.workers is not None or args.resume:
                parser.error("plan-heldout-ttf does not accept --workers/--resume")
            result = plan_heldout_ttf(config, output)
        elif args.command == "collect-heldout-ttf":
            if args.workers is not None:
                parser.error("collect-heldout-ttf does not accept --workers")
            result = collect_heldout_ttf(config, output, resume=args.resume)
        elif args.command == "analyze-heldout-ttf":
            if args.workers is not None or args.resume:
                parser.error("analyze-heldout-ttf does not accept --workers/--resume")
            result = analyze_heldout_ttf(config, output)
        elif args.command == "prepare-boundary-checkpoints":
            if args.workers is not None and args.workers <= 0:
                parser.error("--workers must be positive")
            result = prepare_boundary_checkpoints(
                config,
                output,
                workers=args.workers,
                resume=args.resume,
            )
        elif args.command == "analyze-boundary-checkpoints":
            if args.workers is not None or args.resume:
                parser.error(
                    "analyze-boundary-checkpoints does not accept --workers/--resume"
                )
            result = analyze_boundary_checkpoints(config, output)
        elif args.command == "plan-boundary-ttf":
            if args.workers is not None or args.resume:
                parser.error("plan-boundary-ttf does not accept --workers/--resume")
            result = plan_boundary_ttf(config, output)
        elif args.command == "collect-boundary-ttf":
            if args.workers is not None:
                parser.error("collect-boundary-ttf does not accept --workers")
            result = collect_boundary_ttf(config, output, resume=args.resume)
        else:
            if args.workers is not None or args.resume:
                parser.error("analyze-boundary-ttf does not accept --workers/--resume")
            result = analyze_boundary_ttf(config, output)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
