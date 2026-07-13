from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    expected = sys.argv[2].lower()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"path hash mismatch: expected {expected}, got {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
