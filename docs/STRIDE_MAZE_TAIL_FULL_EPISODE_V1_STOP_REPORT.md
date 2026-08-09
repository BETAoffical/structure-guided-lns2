# STRIDE Maze Tail Full-Episode v1 Stop Report

## Status

The run-to-completion v1 block was stopped at the user's request after it
demonstrated that unbounded solving was incompatible with the actual purpose:
collecting auditable long-tail state conditions.  The stop was deliberate and
is not an execution failure of a completed episode.

The runner completed 17 of 99 scheduled entries before receiving `SIGTERM`.
All 17 completed entries have status `ok`, report solver success, and have zero
invalid actions or stored execution errors.  The interrupted StructPool entry
was not appended to its manifest and is not treated as a completed outcome.

## Triggering state

- map: `maze-128-128-1`;
- task: `opposite_exchange`, task seed `853`;
- agents: `120`;
- solver seed: `43`;
- initial conflicting pairs: `208`.

The paired V2 episode completed in 1,596 repair iterations and 303.063 seconds.
The paired SlotPool episode completed in 32 iterations and 32.416 seconds.  The
StructPool episode was still consuming approximately one CPU core after more
than 46 minutes.  Its incomplete execution is right-censored operational
evidence only; no repair-iteration or final-conflict value is imputed.

## Scientific boundary

The v1 block is incomplete and must not be used for a paired TTF comparison,
tail-incidence rate, promotion decision, or training.  Its completed traces are
retained as diagnostics.  No remaining v1 episode will be resumed.

The replacement v2 block must start in a new output directory and use a normal
trace-finalizing evidence fuse.  Each episode stops after at most 200 repair
decisions or 300 reset-inclusive wall seconds.  A fused episode is a valid
right-censored state trajectory, not an execution error and not a failed
solution claim.  Analysis must use fixed-budget conflict progress and explicit
censoring rather than pretending that all episodes ran to feasibility.

## Frozen hashes

- status: `7b6d81a7a07ec20c94e75d2cf89bf2bb0dd2b54a7326f075549b9f093d7e3bf4`;
- V2 manifest: `43aa50c641a822d1fb5671be25f151ad956e9c26ca150e3864b7d8de496b8058`;
- StructPool manifest: `9b6c34b6c35c519377e246360d67f990596e8b247456b4f72ac5261c009a51b1`;
- SlotPool manifest: `022a064a639e15d50eb653a84fd94f034eaacd8d40f61aab2c6d1505a32450dc`;
- termination stderr: `5bd4fb724740973319a0914f61a8960528330c3389dc14bacce2f38e6e2ddb10`.
