# Post-Training Audit

Run: `20260901T163321Z`

Producing commit: `42e540c0b5ec044cf3885d390190ca8c60ec655a`

Artifact SHA-256: `3d2a66916a1fd18f219f55c202acca921428e50ed198f0dd79c29fe84b8543c9`

## Integrity result

- The artifact hash and completion hashes match.
- Five learned base models, seven candidate-specific time-band calibrator sets, and twenty-one admission models load successfully.
- The fit set ends before `2026-08-01T00:00:00Z`; the holdout is `[2026-08-01, 2026-09-01)` and contains no fit-market overlap.
- All model, calibration, admission, and policy selection evidence is pre-August.
- Directional candidate contracts contain no VWAP, Polymarket execution, TWAP, settlement, or label features.
- Admission contracts contain no outcome labels, realized payoff targets, TWAP, or settlement fields.
- The heldout trade ledger has no duplicate candidate/mode/period/replay/market keys.
- No database writes, tables, schemas, ingesters, sources, runtime exports, deployments, or image rebuilds occurred.

## Material post-training finding

The artifact is valid, but it is not recommended for deployment from this tournament:

- Every preferred learned admission policy abstains in the independently replayed `60–89`, `90–119`, and `120–149` bands. Its combined result is therefore entirely a `150–180` result.
- Four preferred modes have positive full-August nominal and stressed PnL, but every preferred mode is negative over the August 14–31 post-cutover subperiod.
- Directional Brier score and accuracy remain broadly stable across the cutover. The economic degradation is concentrated in admission/execution conversion, not a collapse in raw directional prediction.
- The simpler VWAP5 admission is more durable than the full-curve admission in this holdout. Middle specialist VWAP5 records `+$7.58` nominal / `+$3.33` stressed post-cutover, and price-time VWAP5 records `+$10.48` nominal / `+$1.73` stressed, while their preferred full-curve modes are negative post-cutover.
- Usable complete VWAP curves end at `2026-08-25T23:59:00Z`. The August 27–31 capacity artifacts contain missing-book flags and null curves, so those markets are retained in coverage but cannot generate an execution.

Training stopped after this completed run. No retraining was performed.

The OOF evidence ledger was losslessly partitioned by candidate after completion for source-control file-size compatibility; no predictions or metrics were changed.
