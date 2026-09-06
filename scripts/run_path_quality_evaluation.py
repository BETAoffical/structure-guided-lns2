"""Prepare, reset-validate, explicitly collect, or analyze path-quality episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lns2_selector.evaluation.path_quality_cohort import analyze, collect, static_prepare, validate_resets
from lns2_selector.evaluation.path_quality_preflight import contained


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "validate", "collect", "analyze"))
    parser.add_argument("--config", default="configs/path_quality_evaluation_v1.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--authorize-timing", action="store_true")
    parser.add_argument("--quiet-machine", action="store_true")
    parser.add_argument("--on-ac", action="store_true")
    args = parser.parse_args(argv)
    if args.phase != "collect" and (args.authorize_timing or args.quiet_machine or args.on_ac):
        parser.error("timing authorization flags are only valid for collect")
    config = contained(ROOT, args.config)
    if args.phase == "prepare":
        result = static_prepare(ROOT, config)
    elif args.phase == "validate":
        result = validate_resets(ROOT, config, resume=args.resume)
    elif args.phase == "collect":
        result = collect(ROOT, config, authorize=args.authorize_timing, quiet_machine=args.quiet_machine, on_ac=args.on_ac, resume=args.resume)
    else:
        result = analyze(ROOT, config, partial=args.partial)
        result = {key: result[key] for key in ("complete", "scheduled", "promotion_allowed")}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
