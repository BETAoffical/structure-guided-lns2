from __future__ import annotations

import lns2_features_native


def main() -> int:
    for name in ("batch_online_features", "batch_online_feature_vectors"):
        if not callable(getattr(lns2_features_native, name, None)):
            raise RuntimeError(f"features-only module is missing callable {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
