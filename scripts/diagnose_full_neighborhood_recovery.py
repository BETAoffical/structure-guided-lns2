"""Explicit full-neighborhood mechanism stages; production controllers remain frozen."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.full_recovery_collection import main

if __name__ == '__main__':
    main()
