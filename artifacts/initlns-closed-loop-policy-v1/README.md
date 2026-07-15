# Frozen InitLNS closed-loop policy

This directory contains the dependency-free JSON export used by the frozen
`proposal_dynamic` and `realized_dynamic` closed-loop policies. The bundle records the
source sklearn model hashes, development-index hash, feature ranges and portable model
hashes. It contains no confirmation labels and requires no sklearn installation for
inference.

Regenerate and verify it with:

```powershell
python scripts/export_closed_loop_models.py
```

The source models and development index remain research inputs under ignored `build/`
directories. The portable files in this directory are the versioned deployment artifact.
