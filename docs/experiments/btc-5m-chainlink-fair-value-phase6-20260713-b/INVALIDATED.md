# Invalidated realtime-paper cohort

This cohort started at `2026-07-13T20:54:43Z` and was stopped after its first
daily audit reported rejected `price_change` events as crossed-book integrity
gaps. The stored websocket evidence showed that these were fragmented venue
updates whose advertised top of book was confirmed by a matching full snapshot
milliseconds later. The runtime had applied and validated each fragment too
early instead of reconciling the complete per-token update against the venue's
advertised top.

The database rows and checksummed audit evidence are retained unchanged. This
experiment key must never be resumed or used for an expectancy claim. The
corrected implementation also decouples the reusable, API-managed trading
process from each immutable experiment run; its replacement therefore uses a
new phase-neutral experiment key and image tag.
