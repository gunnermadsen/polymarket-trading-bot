# UMR model integration procedure

## Frozen collection and controls

New paper packages: Extended specialist official, Bridge-aware specialist, Official VWAP admission, Official temporal consensus, Official high-precision loss veto.

The five existing processes are retained: Chainlink regime calibrated, payoff-aware Q5, Chainlink full combined, Chainlink stratified payoff, specialist distilled fair value. Preserve their process IDs, enabled intent, model checksums, execution mode, run identity and history. Never complete or replace them as an implicit migration step.

## Reproducible export

Use `btc_directional_model.unified_runtime_export` with explicit `--source-root`, `--output` and `--panel`. The source root contains the frozen collection manifest and its referenced artifacts. The panel is offline Parquet, preferably on the external SSD. The exporter verifies source checksums, exports existing students/admission estimators, copies frozen policies and produces the existing immutable manifest/model/golden-vector files. It does not train, select thresholds or confer live qualification.

The exported package preserves the original source artifact and run/commit provenance and adds its own runtime checksum. Exporting an existing model must not mint a new `model/...` training tag.

## Deployment procedure

1. Export into a staging directory and run source-to-runtime parity checks.
2. Publish the complete immutable directory under the configured model mount. Never overwrite an active package's contents.
3. Inspect authenticated `GET /admin/models`. Compatibility describes package capability; process readiness also requires valid bindings and actual data.
4. Select the returned model key, artifact checksum and feature checksum in the existing trading-process configuration.
5. Supply `strategy.unified_model` with the contract version, compatible source bindings and the frozen qualified policy. `scripts/prepare-umr-process.py` consumes a saved catalog response, an existing paper playbook template, the selected model/process keys, display name and reviewed preregistration hash. It writes a new disabled definition without API mutations. The same procedure handles every supported capability; it cannot silently substitute unsupported streams. The five supplied definitions are already prepared.
6. Run the existing start-preview and inspect configuration, model, feed, history and execution eligibility.
7. Create/update/start the process only through existing trading-process APIs. New champions are paper-only.
8. Verify inference, admission, execution and official outcomes through the standard UMR dashboard and durable decision records. Lack of a qualifying trade is not permission to loosen frozen thresholds.

The configured read-only model directory is a bind mount independent of container replacement. A newly published compatible package is discovered through the catalog and loaded when selected. No per-tick filesystem scan, per-model service or new image is needed for supported capabilities.

The selector is the model key plus immutable artifact and feature checksums. Mounting a directory alone does not grant trading authorization. A genuinely new mathematical capability still requires a thin compiled adapter; arbitrary mounted Python or native code is never loaded dynamically. The stable engine/session interface keeps that change inside the adapter directory.

## Acceptance evidence

- Hashes match frozen training artifacts and runtime manifests.
- All pre-existing packaged model golden cases still pass.
- All five new packages match raw probabilities, actions and learned/consensus outputs in Python reference cases, including native missing values.
- Raw-second feature construction matches training calculations at every frozen candidate second.
- Causal books reject future evidence, stale observations and wrong epochs.
- Policy/source substitutions and incorrect size fail binding validation.
- History is process-isolated and automatically recovers at a complete market after restart.
- New processes use the existing execution safety, accounting and settlement path.
- Dashboard provisioning and metric/alert contracts validate; any PostgreSQL visual requires retained `EXPLAIN ANALYZE` evidence.
- Concurrency and bounded-state checks demonstrate acceptable runtime overhead before deployment.
- Candidate source/image/config/model identities and rollback references are recorded.

Numerical export parity, feature parity, deployment readiness and profitable live qualification are separate claims. Report actual evidence for each. In particular, absent canonical candles or L2 must not be concealed behind historical tournament statistics.

## Release and rollback

Follow repository branch/integration/image rules. Merge requires explicit authorization naming the feature branch. Record source provenance for the bot image; only rebuild changed components. Observability deployment is provisioned and tagged with its configuration hashes. Do not apply a database migration without its separately authorized narrow specification.

Model rollback selects a previously validated immutable package via process configuration. Service rollback deploys the previous accepted immutable image. Never rebuild an old image as a substitute for the retained rollback image, overwrite artifacts, reset process history, or remove durable trading intent during recovery.
