from pathlib import Path


def _manifest_path(root: Path, controller: str) -> Path:
    policy = "official_adaptive" if controller == "official_adaptive" else "realized_dynamic"
    return root / f"{policy}_manifest.jsonl"
