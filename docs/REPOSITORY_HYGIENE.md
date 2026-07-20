# Repository Hygiene

The active branch contains only the official solver wrapper, frozen controllers,
current evaluation runtime, generators, tests, and frozen evidence. Historical
study code was removed after creating and pushing tag
`pre-minimal-runtime-2026-07-20`.

## Ownership

- `experiments/`, `generators/`, `scripts/`: active Python code.
- `src/`, `include/`: C++ runtime and bindings.
- `configs/`, `docs/`, `tests/`: active protocols, documentation, and regression coverage.
- `artifacts/`: frozen Git-tracked models and compact evidence.
- `third_party/`: pinned upstream solvers and licenses.
- `build/`: ignored local data, environments, and build output.

The single-file `native_features/` and `requirements/` directories were removed.
Feature-only compilation is now controlled by `LNS2_FEATURES_ONLY` in the root
`CMakeLists.txt`; the policy environment lock is
`requirements-policy-training-wsl.lock`.

## Read-Only Audit

```powershell
python scripts/audit_repository_hygiene.py --check
python scripts/audit_repository_hygiene.py --emit-build-plan build/repository-hygiene
```

The audit reports ownership, duplicate code, caches, large tracked files, absolute
machine paths, formal evidence hashes, and protected/temporary build directories.
It intentionally has no automatic deletion option.

## Safety Rules

- Never delete a `build/` directory solely because its name looks temporary.
- Resolve and verify every deletion target under the repository `build/` root.
- Preserve formal result SHA256 values and run result consolidation in strict mode.
- Preserve `build/venv-graph`, Linux/Windows builds, MovingAI data, frozen models,
  official third-party code, and all registered evidence.
- Restore historical experiments from the safety tag in a separate branch instead
  of mixing them back into the runtime branch.
