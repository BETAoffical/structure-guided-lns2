# Active-code retention policy

Every production Python module and pytest file is assigned one of these roles
in the generated retention manifest:

- `active`: directly required by an active controller or the V3-S3 pipeline;
- `shared`: algorithm-neutral code used by at least two retained subsystems;
- `reproducibility`: required to rebuild a retained bundle or reproduce a
  checksum-bound current result, including a preregistered negative result
  whose report freezes a hard-stop decision;
- `compatibility`: the minimum read-only support for a frozen artifact schema;
- `evidence-only`: scientific evidence is retained but executable code is not;
- `obsolete`: no active caller, current reproduction role, or shared contract.

Only the first four roles remain executable on the active branch. A utility is
not retained merely because it might be useful someday: it must have at least
two retained callers or a named reproduction obligation.

Tests protect public behavior, known bug boundaries, artifact identity, or a
retained reproduction contract. Tests that only pin the iteration at which an
unpromoted controller falls back, its private attempt limit, cooldown, or
promotion threshold are removed together with that controller.

Before every pruning phase, the clean predecessor commit is pushed and tagged.
The generated manifest must contain no unclassified production module or test.

Failed or diagnostic-only research chains are never labeled `active`. They may
remain executable as `reproducibility` only when a retained report identifies
the exact negative result and the code is not imported by an active controller.
