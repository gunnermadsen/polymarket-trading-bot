# UMR implementation verification

Status: implementation verified in the isolated feature worktree; production provisioning, activation and live paper observations have not been performed.

| Check | Result |
| --- | --- |
| Rust suite | 387 passed, one existing ignored test |
| Immutable catalog | All 23 mounted packages loaded and passed registration reference cases |
| New frozen champions | 320 Python-to-Rust probability, action and admission reference cases |
| Feature parity | Six candidate seconds; 67 core plus four causal oracle dimensions checked from raw inputs |
| Process integration | Five paper definitions pass existing resolution/start contracts; live authorization rejected; repeat preparation preserves run/config identity |
| Recovery and isolation | Causal book epoch/future rejection, process-local history and next-market recovery, outcome deduplication |
| Native concurrency | 640 inferences across ten models; observed p95 approximately 0.22 ms; not container capacity qualification |
| Dashboard queries | 72 Prometheus dashboard/alert expressions evaluated successfully; no PostgreSQL datasource or queries |
| Dashboard structure | 73 panels including section rows, unique IDs, no grid overlaps |
| Formatting | Rust formatting and diff whitespace checks pass |
| Strict Clippy | Fails on 25 pre-existing diagnostics outside new UMR code; unrelated lint cleanup intentionally excluded |

## Immutable exports

| Package | Runtime artifact SHA-256 |
| --- | --- |
| btc-5m-extended-specialist-official-umr-20260902 | `7cce65dd205578463128acb5c60e86a03f99e94fb6e7b1fd23b701c24c48b8dd` |
| btc-5m-bridge-aware-specialist-umr-20260902 | `c9e7ef4bef348554e1ab16fc2f868998668652620f0b93b8f30a3beffb23fb82` |
| btc-5m-official-vwap-admission-umr-20260902 | `50cde2b43532409c0045bb426357ed80d175f8355364b4e13766ffffad49e40c` |
| btc-5m-official-temporal-consensus-umr-20260902 | `107be64fa68452f3915a508fd678f2d3d874b5954a20267e55bab17a38922229` |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | `28b0a12d847e32e85dbbca4049f4d3b0f279d9627c7570115c331fc937abe6a4` |

## Qualification limits

- Canonical Chainlink OHLC and compatible optional L2 inputs are unavailable in the current adapter. Their declared native missing behavior is preserved; historical fully populated economics are not a live performance guarantee.
- Learned admission started after the first candidate slot waits for the next complete market when prior history is unknown. Enabled intent is preserved.
- Operational calibration and economic aggregates are session-scoped. Durable existing feature, decision, order and accounting records remain authoritative.
- Native tests do not establish deployed container latency, feed readiness, fills or settlement outcomes. Those require the actual candidate deployment.
- Training weights, thresholds, trade size and all five existing model packages/configurations remain unchanged. No training, database migration, process creation or source-data mutation was performed.

Detailed local logs are retained under the worktree’s ignored `target/umr-evidence/` directory. The dashboard rendering check and release provenance are recorded separately when executed.
