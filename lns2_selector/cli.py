from __future__ import annotations

import argparse
from collections.abc import Sequence

from lns2_selector.controllers import CONTROLLER_IDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lns2_selector",
        description="Active MAPF-LNS2 selector workflows",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser(
        "list-controllers", help="list active controller identifiers"
    )
    list_parser.set_defaults(handler=_list_controllers)
    return parser


def _list_controllers(_: argparse.Namespace) -> int:
    for controller_id in CONTROLLER_IDS:
        print(controller_id)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))
