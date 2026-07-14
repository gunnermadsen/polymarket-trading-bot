# Preliminary cohort — not Phase 6 evidence

This cohort ran from `2026-07-13T20:22:24Z` to `2026-07-13T20:25:10Z` and was
stopped through Docker Compose. Its first audit exposed that the authoritative
report treated an expected startup-partial window, which cannot have the
opening Chainlink tick, as a fatal full-window lineage error.

The database rows and archived audit evidence are retained. This key must not
be resumed or used for an expectancy claim. The corrected experiment uses a
new immutable key and preregistration.
