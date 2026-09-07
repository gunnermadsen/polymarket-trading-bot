# UMR implementation verification

Status: deployed directly from `feature/unified-model-runtime` on 2026-09-07. All five new paper processes and five existing controls are enabled and running. No integration merge or golden promotion was performed. The first full market window after the causal-history correction produced six successful predictions per new model, zero feature/inference errors, and no admitted trades.

| Check | Result |
| --- | --- |
| Rust suite | Latest repository-cutover suite: 394 passed, one existing ignored test |
| Immutable catalog | All 23 mounted packages loaded and passed registration reference cases |
| New frozen champions | 320 Python-to-Rust probability, action and admission reference cases |
| Feature parity | Six candidate seconds; 67 core plus four causal oracle dimensions checked from raw inputs |
| Process integration | Five paper definitions pass existing resolution/start contracts; live authorization rejected; repeat preparation preserves run/config identity |
| Recovery and isolation | Causal book epoch/future rejection, process-local history and next-market recovery, outcome deduplication |
| Native concurrency | 640 inferences across ten models; observed p95 approximately 0.22 ms; not container capacity qualification |
| Dashboard queries | 72 Prometheus dashboard/alert expressions evaluated successfully; no PostgreSQL datasource or queries |
| Dashboard structure | 73 panels including section rows, unique IDs, no grid overlaps |
| Grafana rendering | Exact committed dashboard provisioned in Grafana 12.4.2; introduction, summary, funnel and diagnostic layouts inspected with empty synthetic responses |
| Grafana alerts | Nine definitions accepted by the isolated preview; alert execution disabled |
| Alloy | Configuration validation passes with networking disabled |
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
- The initial live inference window establishes functioning feature construction and frozen admission evaluation, not profitable qualification. No new-model fill or settlement was observed in that window; those portions of live behavior remain unverified.
- Training weights, thresholds, trade size and all five existing model packages/configurations remain unchanged. Five new paper processes were created and started through existing APIs. No training, database migration or source-data mutation was performed.

Detailed local logs are retained under the worktree’s ignored `target/umr-evidence/` directory. Isolated Grafana provisioning is recorded by `provisioned/observability/umr-review/20260907T204438Z`, pointing to dashboard source `f525bc9`. Temporary preview services were removed.

## Feature deployment evidence

### Single RTDS repository cutover, 2026-09-07 22:39 UTC

Deployed source `802d582ba6043b188f5a0e004fbe749b93c39662`, immutable image `sha256:c7b046d1d9863928e8ac94d6914ba5deb6e4acc8c7c0df289a068685d7358680`, with matching embedded provenance and annotated image tag. Retained preceding image `sha256:71364fd5c8f5e0cdbd21eda79622a6d512abff449ac536b03d34a5885093bba4` as the immediate rollback reference. No integration merge or golden promotion occurred.

One repository now owns RTDS point history and the extracted candle calculation. Startup seeds it unconditionally through the existing bounded database query and retry path; the former dynamic RTDS hydration task/flag is removed. Model readers and existing global candle metrics use that repository. The existing gRPC source remains the only live RTDS source; the default shared selector is optional and explicit consumer selectors remain unchanged. See [the repository contract](rtds-repository.md).

All 394 tests passed, including exact OHLC/availability fixtures, causal gaps, duplicate/out-of-order hydration, immutable snapshot sharing, source uniqueness, model reference vectors and lifecycle tests. All ten process configurations compared exactly equal before/after deployment, all ten resumed enabled/running, all model manifest hashes were unchanged, and the catalog accepted all 23 packages.

Live checks confirmed 61 complete RTDS minutes and readiness 1 both directly and through Grafana's existing Prometheus datasource. By 22:41:42 UTC all ten models had successful inferences with zero inference errors. New-model counts were 6/6/6/5/5 across the first six scheduled opportunities; the two misses were unavailable UP books. Existing models also had orderbook-related feature skips, with no missing RTDS candle errors in the bounded post-deployment log inspection. No threshold or model-input substitution was made to bypass these checks.

The container had zero restarts; a single resource snapshot showed 3.80% CPU and 32.19 MiB memory, not a sustained capacity qualification. Existing monitoring definitions were not changed or reprovisioned, and no database migration or diagnostic database scan was used. Evidence is retained under `target/umr-evidence/` in the feature worktree (`before-rtds-repository.json`, `after-rtds-repository.json`, `rtds-repository-tests-final.log`, `deployed-image-rtds-repository.json`, and live metrics/diagnostics).

### Shared RTDS hydration correction, 2026-09-07 22:06 UTC

The first monitoring baseline exposed startup-order-dependent RTDS seeding: a new model could start the shared runtime without RTDS, and adding an existing RTDS consumer later subscribed to live ticks without restoring history. The dynamic source lifecycle now invokes the existing bounded RTDS hydration/recovery path exactly once, matching the established Binance/open-interest approach. No per-model history or new feed was introduced.

Deployed source `7c6c81e97670c8908dee9295eace29a2b930339f`, image `sha256:71364fd5c8f5e0cdbd21eda79622a6d512abff449ac536b03d34a5885093bba4`, with matching embedded provenance and annotated image tag. The full suite passed 390 tests, one existing ignored. At 22:06:32 UTC the live hydration log recorded 3,558 restored ticks and 61 complete minutes. The global candle-ready metric returned to 1; regime calibrated, full combined, stratified payoff and distilled fair value all resumed successful inference. All ten enabled processes resumed and their configurations compared exactly equal before/after deployment. Model artifacts and thresholds were unchanged; no integration merge, migration, or monitoring configuration deployment occurred.

The image and evidence below describe the preceding deployment, retained for history.

- Bot source: `d6d48e4b8fe484370d6bd5be93c6891955de32cd`.
- Immutable deployed image: `sha256:3bbe49ca1f6d5f43443a3b1dd9c399d0487a1a429c589d31755a5c6a76a79fcf`; matching embedded source revision and annotated `image/polymarket-bot/sha256-3bbe49ca1f6d5f43443a3b1dd9c399d0487a1a429c589d31755a5c6a76a79fcf` tag. Built from clean committed feature source.
- Previous accepted rollback image: `sha256:64268bd4762d03ff4ec812a2a8a9019b12c2b404bbef290ba472dde7dea7730d`. This image predates UMR; new UMR processes require a compatible runtime to infer. Existing process identities and durable records are retained.
- Runtime model directory is the feature worktree's read-only mounted `packages/btc-directional-model/runtime-models`. All five mounted manifests matched host files; catalog accepts all 23 packages. No database migrations or schema changes.
- All ten enabled processes resumed automatically after feature container replacement. The five original process configurations compare exactly equal before and after deployment.
- Live Grafana API verifies 73 dashboard panels and nine provisioned UMR alerts. Prometheus sees ten model identities; the dashboard's process-scoped Loki query returns real runtime logs. No PostgreSQL visual queries are used.
- Monitoring provisioning events: `provisioned/observability/development/20260907T212053Z` records the initial Grafana/Alloy deployment and missing alert-copy diagnosis; `20260907T212136Z` records corrected alert provisioning; `20260907T212931Z` records the corrected first-inference age display. Each full tag uses the same `provisioned/observability/development/` prefix.
- Live deployment exposed and corrected optional-versus-required selector matching and dense book-publication eviction. Required subscription settings remain unchanged. A 9,600-publication regression verifies causal whole-second boundaries survive bursts; eight UMR tests and 26 lifecycle tests passed.
- Final full-suite run passed 389 tests, with one existing ignored test. Local HTTP fixtures required socket access. A single live resource snapshot showed 4.60% CPU and 25.45 MiB resident container memory; this is not a sustained capacity benchmark.
- In the 21:30 UTC market, candidate seconds 60, 65, 70, 75, 80 and 85 produced 30 successful new-model inferences. All 30 were rejected by frozen model admission; zero new-model feature/inference errors or paper fills in this observed window. Existing controls continue using their existing feature requirements, including their original transient missing-data handling.

| New paper model | Process ID |
| --- | --- |
| Extended official | `4669169b-75b5-41e0-a08f-790d049a84da` |
| Bridge aware | `980a140e-3823-4548-b862-475c205d0e2f` |
| VWAP admission | `4ec890d3-720e-49fb-9b8c-b579b2925091` |
| Temporal consensus | `c3d19b13-16dc-4d20-9afb-d67814f84d38` |
| High-precision loss veto | `96c89496-da28-45fd-b18a-bad865352e9a` |
