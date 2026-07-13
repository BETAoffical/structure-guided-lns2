# MAPF-LNS2 upstream

- Repository: https://github.com/Jiaoyang-Li/MAPF-LNS2
- Branch: `init-LNS`
- Commit: `1369823985a15944f9a339226d521f61605a6d17`
- Imported: 2026-07-14
- License: USC Research License in `license.txt`; the PIBT subtree also carries the MIT terms listed there.

The complete upstream source is vendored so experiments do not silently drift with a remote branch.
Project-specific changes are intentionally limited to:

- a step-wise `InitLNS` API;
- neighborhood policy and observer hooks with an unchanged default path;
- a `repairOnly` switch in `LNS` and the official CLI;
- read-only low-level search counters used by traces.

The `official` policy mode delegates to the original ALNS and neighborhood generation code. External
actions are validated and fall back to the official path when invalid.
