# Unified Model Runtime

UMR is the model integration boundary inside the existing BTC trading runtime. It does not own another trading process system, service, feed connection, settlement ledger, or scheduler. `trading_processes.process_id` remains the owner of configuration, decisions, execution and outcomes.

## Architecture and ownership

The existing process runner resolves an immutable package, obtains a causal feature snapshot, evaluates a model adapter, applies the existing execution controls, and persists the decision through the existing repository. UMR supplies reusable contracts, adapters, catalog validation and instrumentation around that lifecycle.

Code lives in `packages/polymarket-bot/src/btc/unified_model_runtime/`:

| Module | Responsibility |
| --- | --- |
| `contract.rs` | Versioned mathematical input requirements and process binding validation |
| `catalog.rs` | Read-only package discovery and compatibility reporting |
| `adapters/mod.rs` | `ModelAdapter` capability registration and process-owned `FeatureSession` interface |
| `adapters/legacy.rs` | Existing schema-driven contract description, preserving the established feature/scoring path |
| `adapters/validation.rs` | Mandatory reference-vector parity at registration, before immutable model caching |
| `adapters/feature_names.rs` | Exact supported feature vocabulary for the frozen early-entry capability |
| `adapters/frozen_early_entry.rs` | Frozen distilled prediction, consensus and learned admission composition |
| `adapters/features.rs` | Bind canonical data products to the frozen early-entry feature recipe |
| `adapters/data.rs` | Bounded causal book history using immutable shared observations |
| `adapters/history.rs` | Process-owned, market-local probability history |
| `telemetry.rs` | Stable process-scoped metrics and prediction evidence |
| `tests.rs` | Contract, parity, causal input and recovery checks |

Existing directional, payoff and asymmetric implementations remain supported through the existing loader and scoring interface. Their artifacts, feature definitions, thresholds and process identities are not rewritten. The common process lifecycle instruments them too.

## Contracts

The UMR contract version is `capitonic-unified-model-runtime-v1`. The durable prediction envelope version is `capitonic-model-evaluation-v1`. Existing runtime manifest and feature snapshot contracts remain valid.

A new package embeds its UMR contract in the checksummed model payload. It declares its adapter/version, input products and semantics, required/optional status, lookback, age bounds, feature clock, missing-value policy, probability meaning and qualified trade size. Ordered feature names and numerical representation remain in the existing checksummed feature schema. Prediction cadence and frozen admission components remain in the model artifact.

`strategy.unified_model` supplies the version, source bindings and effective policy in the process playbook. For the frozen champions the policy must equal the exported qualification policy exactly, and size must equal five shares. Process configuration cannot silently reinterpret an input, change an estimator, or alter thresholds. Unknown adapters, versions, slots or unsupported bindings fail compatibility validation. Existing processes omit this additive field and preserve their legacy path.

Global credentials and stream connections remain infrastructure configuration. A binding consumes shared gRPC state; it does not create a socket or database poller. All required source history and integrity must be available before the dependent action is eligible.

## Model adapter standard

An adapter is a reusable capability, never a switch on champion name. It must:

1. Validate its complete mathematical and input contract before use.
2. Consume immutable, ordered float64 inputs with declared missing semantics.
3. Produce probability-up, confidence, admission status and bounded reasons without submitting orders.
4. Preserve the exported estimator, calibration, clipping, comparison and tie rules.
5. Return errors for unsupported or invalid inputs; never silently switch models or data products.
6. Keep mutable history owned by the process, bounded, market-scoped and explicitly recoverable.
7. Use common telemetry and durable prediction evidence rather than create model-specific dashboards or ledgers.
8. Include Python-to-Rust reference cases and input-construction parity evidence.

The process runner calls `FeatureSession::prepare` and `ModelAdapter::evaluate`; it does not branch on champion identity or admission architecture. Immutable estimators are shared; each process owns a separate session. The adapter declares the products it can actually bind, and catalog discovery reports unsupported optional inputs separately from package compatibility.

A model using an existing capability requires export, mounting and process selection only. A new mathematical feature or engine may add a thin adapter and capability version in this directory; it must not require changes to order submission, accounting, existing model behavior or dashboard queries. New adapters use existing tree primitives where applicable. Arbitrary model-supplied executable code is unsupported.

Contract changes that alter meaning require a new version. Backward-compatible additions must retain existing meanings and defaults. Remove an adapter/version only after proving no enabled or resumable process depends on it. A successful compile is not model-behavior parity evidence.

## Frozen early-entry semantics

The five champions use the frozen distilled students, not their training teachers. Admission components and temporal snapshots are exported without fitting or calibration. Entry opportunities are seconds 60, 65, 70, 75, 80 and 85; size is five shares.

The opening boundary is the open of the Binance second beginning one second before the market window. Core features follow the training recipe. Book features use a snapshot available at or before the model timestamp and within the frozen two-second age bound. Order execution independently uses the existing current-book identity, freshness, capital and accounting controls. The feature adapter does not impose the legacy payoff builder's 800-share ladder requirement.

Candle semantics are `chainlink_ohlc_close_available_120s_v1`. RTDS midpoint candles are not interchangeable with that product. Until a compatible product adapter is available, this optional input is absent and the trained native-missing branches receive NaNs. This absence is visible in missing-feature telemetry and must not be presented as reproducing the historical fully populated input distribution. Direct RefPrice and TWAP ticks are not inference dependencies of these packages.

Oracle features use causal shared oracle observations and the frozen optional-input behavior. L2 dimensions in the loss veto remain absent unless a compatible feature adapter is implemented and qualified; arbitrary similarly named L2 values cannot be substituted.

Learned admission uses prior eligible opportunity probabilities. History is isolated by process and market. On cold start or restart after the first scheduled opportunity, an unknown history cannot be replaced by fabricated zeros. The affected action waits until the next complete market; durable enabled intent remains intact. This runtime availability difference is separate from model numerical parity.

## Liveness and safety

Package detection never changes a running process's selection. Artifact hashes pin the process. Partial exports are hidden and published atomically. Unsupported or corrupt packages report a catalog error without changing other registrations.

The runtime caches immutable model components. Inference uses bounded histories and never scans training data or the database. Causal book history uses shared immutable references, retains five seconds and at most 64 snapshots, and forbids epoch crossover. For these whole-second model candidates, it retains the latest publication per token per second, including publications exactly at the boundary. Dense intra-second bursts cannot evict the preceding causal boundary. This compressed history is not a subsecond-query contract; a future adapter requiring subsecond history must declare and validate that capability.

Shared subscriptions preserve an existing required consumer's selector when an optional consumer uses the same product and contract version. Differing required selectors and differing contract versions still conflict. Optional adapter freshness rules remain process-local; adding an optional consumer cannot tighten or weaken an existing required subscription.

RTDS midpoint candle history is shared runtime state, restored from persisted RTDS ticks rather than baked into an image. Restore it when RTDS is present at startup or first added to the source union, once per shared runtime, using the existing bounded hydration/retry path. Every compatible model receives that same cache automatically; no process-specific seed is needed. `polymarket_btc_rtds_chainlink_candle_complete_minutes` reports its current 61-minute coverage. A model still validates history at its causal decision timestamp. RTDS midpoint candles must not replace canonical Chainlink OHLC for an adapter trained on a different product.

Transient feed, history, persistence and transport failures block only affected actions. They must not permanently disable a process or unrelated processes. Existing reconciliation, order reservation, execution safety, settlement and accounting are authoritative. Telemetry must not become an additional durable trading authorization gate.

## Related standards

- [Instrumentation and dashboard contract](instrumentation.md)
- [Model integration procedure and acceptance](integration.md)

- [Implementation verification and qualification limits](verification.md)
