"""CLI for reference-only path compatibility diagnosis."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.local_path_compatibility import main

if __name__ == '__main__':
    main()
