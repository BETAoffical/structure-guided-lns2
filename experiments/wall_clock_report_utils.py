from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_boolean(value: Any) -> bool:
    return str(value).lower() == "true"


__all__ = ["csv_boolean", "read_csv_rows"]
