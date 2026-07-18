# MovingAI compact migration evidence

This artifact preserves the machine-independent evidence for converting the 720 registered
`full-v1` episodes to `delta-gzip-v2`. All five policy manifests matched with zero scientific
mismatches, and total trace storage fell from 16,242,547,808 to 50,329,794 bytes (99.6901%).

The cleanup manifest is deliberately not an authorization. Only the legacy `episodes/` directory
is eligible for deletion after a fresh quick run passes and the user separately confirms the exact
resolved path. The v1 model, manifests, run metadata, reports, compact collection, and archive remain.
