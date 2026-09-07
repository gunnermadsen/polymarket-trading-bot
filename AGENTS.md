# North Star
Think of Capitonic as a vision to generate income through systems with automation and algorithms, that require no customers. The polymarket trading bot is an extension of that vision, to systemically generate an  income stream based on a reproducable strategy that can trade with positive net expectancy through a narrow and simplified system. the key is positive net expectancy and a simplified system. Capitonic was not envisioned to live in a labatory forever. Capitonic was envisioned to prove a concept, that an automated trading bot can become a real product, a real business. Captitonic is NOT here to be reduced to a pet project or a labratory, or an experiment platform. Capitonic exists to discover an edge, exploited with a strategy, and repeated to generate net expectancy. Do not make the mistake of seeing Capitonic as a "research platform", because it is not that. Capitonic is not a school, or a university; it is a business. Businesses rely on income to survive. If Capitonic produces income, Capitonic WILL thrive.

## Golden Rules
- All Rust features and systems are implemented and optimized for latency, memory usage efficiency, CPU performance, zero downtime, and database resource usage, ensuring a performant and resiliant system.
- Do not name artifacts, branches, files or code comments based on phases or stages of an implementation. use domain specific naming for the issue or feature being addresses. I dont' want to see 'phase X' in git history, code files, branches, or file names or code comments. 

## Always-On Trading Liveness
- Do not create blunt or pointless kill switches that stop trading processes, revoke durable trading authorization, or require manual re-enablement merely because a container, database connection, reconciliation loop, websocket, or other dependency restarted or experienced a transient failure.
- A normal container deployment or service restart must preserve the configured intent of an enabled trading process and recover back to trading automatically once its required runtime evidence is healthy. Ephemeral in-memory gates must not silently override durable process configuration after restart.
- Transient unavailability, stale evidence, crossed or one-sided books, temporary reconciliation errors, database reconnects, and transport reconnects may block only the specific unsafe action while the condition exists. They must not terminate the process, permanently disable entries, or poison unrelated processes.
- Recovery must be automatic and narrowly scoped. When healthy evidence returns, the affected action must become eligible again without a manual operator ritual unless the operator explicitly disabled it in durable configuration.
- Reserve process termination or persistent trading disablement for explicit operator intent or a proven unrecoverable integrity condition. Do not treat ordinary infrastructure lifecycle events as unrecoverable integrity failures.
- Shared infrastructure faults must not allow one trading process to disable other independent processes. Scope runtime readiness and failure handling by `process_id` wherever process behavior is involved.
- When changing lifecycle or readiness code, verify restart recovery and confirm that configured live processes resume eligibility without bypassing the existing per-order capital, identity, accounting, and market-safety checks.

## Code
- Rebuild an image only when that component's code changes: Polymarket bot changes rebuild `polymarket-bot`, and ingester changes rebuild `ingester`.
- build the polymarket bot with provenance env var: POLYMARKET_GIT_REVISION=<GIT_COMMIT_HASH> docker compose build polymarket-bot
- Do not rebuild an image when its component's code did not change.
- put sensitive secrets in .env files
- Treat the main worktree's `.env` file as the source of truth for API-key secrets. Worktrees and services must reference or inherit those secrets without creating independent secret values.
- put non-sensitive runtime configuration in docker-compose files
- .env example files are templates, do not put plaintext env vars in the example env files.

## Rust file organization
- Organize Rust code by domain responsibility, with one clear purpose per module.
- Keep files narrowly scoped; split files that mix unrelated responsibilities or become difficult to navigate.
- Place shared types and behavior in the nearest common domain module. Do not create generic dumping-ground modules such as `utils` or `common`.
- Keep public module interfaces minimal and expose implementation details only when required by another module.
- Follow the existing crate and module structure unless the requested change requires a focused reorganization.

# Source Control and Worktrees

## Integration Cycle and Trading Release Roles

- `development` is the settlement branch for accepted trading-capable releases. Do not implement features, fixes, experiments, or integration corrections directly on `development`.
- A `development` tip is golden only when an annotated `golden/<image_name>/sha256-<docker-sha256-hash>` tag identifies the exact accepted image. Branch position or an `image/...` build tag alone is not golden evidence.
- Feature verification proves readiness for integration; it does not make a feature branch or its image golden.
- Use exactly one active integration branch for each golden-image build cycle. Name it `integration-<YYYY-MM-DD>`.
- Create the integration branch once from the current accepted `development` tip at the beginning of the cycle. An annotated `integration-cycle/<YYYY-MM-DD>` tag may record the cycle boundary for ancestry inspection, but its absence does not block explicitly authorized branch creation or integration.
- The active integration branch is always checked out in the main repository worktree. Never create or keep it in a disposable worktree under `target/worktrees`.
- The integration branch is the single collection point for the cycle. Do not create feature-specific, defect-specific, candidate-specific, or secondary integration branches.
- Creating the integration branch is the only point where the cycle branches from `development`. After it exists, every new feature or defect intended for that cycle starts from the latest integration tip and merges back into that same integration branch.
- A narrowly scoped integration-policy or coordination correction may be committed directly on the integration branch when the user explicitly requests it. Feature and defect implementation still use branches rooted in the active integration lineage.
- Merge a selected feature or defect branch into the integration branch only after the user explicitly authorizes merging that exact branch. Verification findings must be reported but do not create an additional authorization gate. Never merge a feature or defect branch directly into `development`.
- Abandon a rejected candidate branch rather than repairing its integration history with merge reverts. Preserve the rejected branch until its result and any reusable commits are accounted for.

## Abandoned Lineages

- Before deleting or otherwise retiring an intentionally discarded branch, divergent commit, rejected candidate, or superseded release snapshot, create an annotated tag on its final retained commit using `abandoned/<domain>/git-<full-git-commit-id>`.
- The abandoned tag annotation records the original branch or ref when known, the reason for abandonment, the replacement or superseding commit when one exists, any image identity built from it, and whether it was ever deployed.
- An `abandoned/...` tag excludes that lineage and its images from implicit integration, candidate admission, golden promotion, and rollback selection. The user may explicitly restore or use an exact abandoned branch, commit, or image without creating a new lineage.
- Preserve abandoned tags when removing worktrees or branches. An image build tag may remain for provenance, but it does not override abandoned status.
- Release and integration tasks must inspect `abandoned/...` tags before selecting branches, commits, or images and must fail closed rather than merge or deploy an abandoned lineage implicitly.

## Golden Image Admission and Promotion

- The user may explicitly authorize minting a golden image from an exact integration commit and immutable image at any time. That instruction is sufficient promotion authorization and must not be delayed or refused because of an undefined waiting period, elapsed-time requirement, or missing operational evidence.
- always deploy the golden images to containers after minting.
- Operational validation may be performed and recorded when requested, but it is not a mandatory time-based gate unless the user explicitly defines one.
- Before minting, verify the selected Git commit, immutable Docker image ID or registry digest, embedded source revision, and clean committed state. Record which tests and operational checks were performed and disclose known limitations.
- Track the exact candidate tuple that was evaluated: Git commit, immutable Docker image ID or registry digest, embedded source revision, migration state, material runtime configuration, and model identity when applicable.
- When operational validation is requested, evaluate whether the exact candidate image:
  - was built from a clean committed worktree with `POLYMARKET_GIT_REVISION` matching the candidate commit;
  - passed the required compilation, linting, focused tests, and migration checks;
  - preserved automatic container, database, reconciliation, and transport recovery where affected;
  - preserved enabled paper and live process eligibility without bypassing capital, identity, accounting, order, or market-safety controls;
  - produced healthy order, fill, reconciliation, settlement, and accounting evidence appropriate to the affected paths;
  - avoided sustained crash loops, resource exhaustion, systemic readiness poisoning, and cross-process failure propagation; and
  - remained rollback-compatible with the immediately preceding golden image and its database state.
- Missing operational evidence must be disclosed, but it does not override an explicit user instruction to mint the golden image unless the user explicitly made that evidence a requirement.
- A Codex task performing integration or release stewardship must inspect the active candidate and golden tags before acting. If the user explicitly requests minting a golden image, proceed using the exact selected candidate without another confirmation prompt. Without an explicit request, promotion is authorized only when exactly one candidate has complete admission evidence.
- Promote by fast-forwarding `development` with `--ff-only` to the exact selected candidate commit. Do not create a promotion merge commit, rewrite `development`, or force-push. Do not rebuild a user-selected immutable image unless the user explicitly authorizes building an image for the exact committed integration revision.
- Create an annotated `golden/<image_name>/sha256-<docker-sha256-hash>` tag recording the Git revision, immutable image identity, migration/config/model identity, checks performed, and accepted limitations.
- Promote or alias the selected already-built image manifest when one exists. If the user explicitly directs minting and no image exists for the exact committed integration revision, build it with provenance, verify its immutable identity, and mint that image golden. Do not substitute a rebuilt image for a user-selected immutable digest without explicit authorization. Verify the resulting branch, tag, image identity, and embedded provenance.
- Roll back by deploying a previously annotated immutable golden image. Do not rebuild the old commit and do not reset `development` merely to change the deployed image.

## Branching

- Commit changes in coherent groups organized by feature domain.
- After building an immutable container image, tag the exact source commit used for that build with `image/<image_name>/sha256-<docker-sha256-hash>`.
- A `model/<model-name-with-metadata>` tag records provenance for a new immutable model artifact produced by an executed model-build or training workflow. Create it only when that exact artifact exists, its identity can be verified, and the tag can point to the commit that first records or unambiguously references it.
- Deploying, activating, configuring, copying, exporting for runtime use, or packaging an existing model does not constitute a new model build and must not create a new `model/...` tag. Building a container that uses an existing model receives an `image/...` tag only; record the existing model tag, model key, or artifact SHA-256 in the image tag annotation.
- Starting a paper or live trading process with an existing model must not place a `model/...` tag on the process configuration commit, deployment commit, current branch tip, or latest repository commit. Reuse the existing model identity without moving or recreating its tag.
- Model tags must be annotated and record the model artifact SHA-256, artifact path or immutable URI, producing commit, executed model-build or training-run identity, source or input identity, qualification status, and deployment status at tagging time.
- Never infer model-tag eligibility from a model filename, manifest, process deployment, container build, branch name, or the fact that a commit is recent. If the task did not produce a new immutable model artifact, do not create a `model/...` tag.
- These rules govern Git tag creation and provenance only. They do not gate, delay, prohibit, prescribe, or otherwise interfere with model training, retraining, evaluation, export, or experimentation.
- Use these standard branch names for new cycle work: `integration-<YYYY-MM-DD>`, `feature/<feature-name>`, and `defect/<defect-name>`. Use another branch name only when the user explicitly requests it.
- By default, every new independent feature domain uses a dedicated `feature/<feature-name>` branch and every defect uses a dedicated `defect/<defect-name>` branch. An explicitly requested branch name may override this naming default.
- Create feature and defect branches from the latest tip of the active integration branch, never from `development` while an integration cycle is active.
- Follow-up work that must inherit an existing feature or training lineage starts from that lineage’s designated base or integration branch, not from `development`.
- Keep unrelated feature domains on separate branches.
- Do not merge a feature or defect branch directly into `development`.
- Creating a `feature/...` or `defect/...` branch from the active integration branch authorizes isolated work on that branch only; it does not authorize merging it back. Keep the branch unmerged until the user explicitly grants permission to merge that exact branch into integration. Completing implementation, committing, testing, reviewing, or declaring the branch ready does not imply merge permission. If permission is absent or ambiguous, stop before the merge and ask for authorization.
- Before carrying out an explicitly authorized merge, inspect and report:
  - the exact selected branch or commit;
  - its merge base and whether it descends from the active cycle marker when one exists;
  - whether it is marked by an `abandoned/...` tag;
  - its clean, committed, and test state;
  - whether it incorporates the latest integration tip; and
  - its commit log and diff against the active integration branch.
- These findings are disclosure requirements, not independent vetoes. Explicit user authorization naming the exact branch or commit is sufficient to proceed. Stop only when the target is ambiguous, uncommitted work would be lost, or the action requires destructive history rewriting that the user did not explicitly authorize.
- Merge an explicitly authorized feature or defect branch into the integration branch using `--ff-only` when the integration branch has not diverged from the feature branch's merge base.
- When the branches have diverged, merge the explicitly authorized feature or defect branch using `--no-ff`.
- Never rebase, rewrite, or discard either lineage merely to make a fast-forward merge possible.
- Never infer merge permission from recency, branch-name similarity, worktree existence, dirty state, or whether Git reports the branch as unmerged.
- Do not merge an old, pre-cycle, cross-cycle, abandoned, or otherwise unrelated branch implicitly. Explicit user authorization naming the exact branch or commit is sufficient authorization for that merge; disclose its lineage and status before proceeding.
- After integration collects new work, create subsequent feature and defect branches from the new integration tip so they begin with the complete collected code.
- Advance `development` only through the golden admission and fast-forward promotion rules above.
- Never discard, rewrite, or bypass an existing feature lineage merely to satisfy the “latest development” rule.

## When to Create a Worktree

- Create a new worktree only for a new, independent, overarching feature domain that requires isolation from the current checkout.
- Use one worktree for the entire feature domain, including its implementation, tests, fixes, review corrections, model variations, and follow-up iterations.
- Do not create additional worktrees for:
  - small fixes within the active feature;
  - test failures or review corrections;
  - configuration adjustments;
  - documentation changes;
  - model candidates or training variations belonging to the same training objective;
  - additional commits or temporary branches within the same feature;
  - read-only investigation or diagnostics.
- Reuse the existing feature worktree whenever the requested change belongs to that worktree’s overarching feature domain.
- A tiny unrelated change may be committed on its own branch without creating a worktree when isolation is unnecessary.
- Do not create multiple worktrees for the same feature domain.
- Do not create a new worktree while another agent-created feature worktree is active unless:
  - the existing worktree belongs to a materially different feature domain; and
  - parallel worktrees were explicitly requested or are strictly necessary.

## Worktree Location and Ownership

- Store all persistent project worktrees under `target/worktrees/<feature-domain>`.
- Do not create persistent worktrees under `/tmp`, `/private/tmp`, or arbitrary external directories.
- Each disposable worktree owns one overarching feature domain and one designated `feature/...` or `defect/...` branch. The active integration branch belongs only to the main worktree.
- Related temporary change branches may be created and checked out inside that same worktree; they do not receive separate worktrees.
- Keep all implementation, tests, generated evidence, and related fixes for the feature inside its assigned worktree.
- Before creating a worktree, run `git worktree list` and confirm that no existing worktree already covers the feature domain.

## Worktree Lifecycle

- Keep a feature worktree until its changes are:
  - committed;
  - proportionately verified;
  - merged into its designated base or integration branch when integration is authorized; and
  - no longer needed for generated artifacts or cached evidence.
- Before removing a worktree, verify:
  - `git status --porcelain` is empty;
  - its commits are preserved on a branch or contained in the designated base;
  - it contains no unique untracked or ignored artifacts that must be retained.
- Remove completed worktrees promptly after those checks pass.
- Removing a worktree must not automatically delete its branch.
- Never force-remove a dirty worktree unless the user explicitly authorizes discarding its remaining contents.

# System
- Trading process parameters that affect trades, go in the trading playbook config stored in 'trading_processes'. parameters that affect global systems go in environment variables.
- All plaintext env vars that are non sensitive, go in docker-compose files in the 'environment' section. do not put plantext env vars in .env file. 
- Refactoring should not render trading processes as no longer compatible. 
- Refactoring a system, introducing a new system, or removing a system and causing a lack of compatibility is an anti pattern in the trading bot.

## Backfills
- Reuse the existing shared backfill infrastructure, standards, and contracts for every backfill domain; do not create domain-specific backfill workers or Docker Compose files.
- Use the existing shared backfill job tables for every backfill domain; do not create domain-specific job tables or drift from the established contract.
- `ingester.backfill_jobs` is the sole job ledger, history, and scheduling source. `ingester-master` is the sole scheduler and API; `ingester-worker` is the sole worker runtime name.
- Every collection unit implements the documented `RealtimeWorkerStrategy`, `BackfillWorkerStrategy`, or both in `packages/market-data-ingester`. Do not create another strategy contract.
- Add a strategy to the existing registry and schedule it through `POST /backfills`. Never add a provider-specific worker binary, image, Compose service, planner command, queue table, job schema, or scheduling pathway.
- Sharding is deterministic strategy logic invoked by the master. Realtime strategies are not sharded. Worker and deployment selectors are execution controls, not new queues.
- Preserve the API and persistence contracts. Change their versions only for a necessary incompatibility documented in `packages/market-data-ingester/docs/strategy-and-backfill-contract.md`.

## Observability provisioning
- Provision every Grafana, Prometheus, Loki, and Alloy deployment, including all environment configuration changes; do not make manual, unprovisioned changes.

## Observability deployment provenance
- After provisioning Grafana, Prometheus, Loki, or Alloy configuration, create one annotated tag on the exact source commit using `provisioned/observability/<environment>/<YYYYMMDDTHHMMSSZ>`.
- Record the target environment, provisioning timestamp, originating branch, deployment result, configuration hash, and every component and configuration path provisioned.
- Create one tag per provisioning event, including when multiple observability components are deployed together. Do not create separate component tags for the same event.
- A provisioning tag records deployment history only; it does not imply image, integration, candidate, or golden-release status.

## Data artifacts
- Store all backtesting and model-training data only in Parquet format, including weather-model predictions and Kraken futures test data.
- Direct Chainlink Data Streams reference prices use only `market_data.chainlink_btcusd_reference_prices`; PMData reference prices use only `market_data.pmdata_chainlink_btcusd_reference_prices`. Strategies must call the corresponding shared persistence function and must not issue table-specific insert SQL.
- A one-off historical drain is copy-only: it may read source tables and write canonical Parquet, but it must not mutate its database sources. Source removal happens only through a separately guarded database migration after complete manifest validation.

# Database and migrations
- all non trading process database mutations or changes must be executed through database migrations.
- Do not create or apply database migrations without explicit user permission; first provide a narrow, unambiguous summary of the proposed migration.
- all diagnostic database queries to read the database must be optimized for performance to prevent database crashes.
- do not scan large tables without considering performance ramifications.
- use the db-migrate microservice job to apply migrations
- apply migrations by creating new migration files inside packages/db-migrate/src/migrations
- always confirm a migration was already applied before running migrations.
- never create trading processes through migrations. always use the api endpoint for trading  process creation or modification.
- apply migrations by recreating the container:

```bash
docker compose up -d --force-recreate --no-deps db-migrate
```

# Diagnostics
- Conservative query resource usage when performing diagnostics in the database.
- do not run large table scans or inefficient queries that starve resources and cause crashes.

# Implementation
- Execute the narrow implementation plan, only focusing on the instructed plan parameters.
- do not over correct and change code or systems outside the agreed upon plan.
- Always use process_id of the trading process to scope. do not depend on the experiment_id for scoping. 

# Known Issues
- experiment sub system was created with the incorrect assumptions. this system duplicates the id scoping and artifact ownership. experiment sub system will be removed in a later release. do not depend on the experiment system. trading_processes and it's process_id remains the canonical source of truth for record ownership and scoping.

## Unified Model Runtime contract

- Follow [the UMR architecture and adapter contracts](docs/unified-model-runtime/README.md) and its instrumentation/integration standards when changing model inference, feature bindings or monitoring.
- Preserve frozen model behavior, existing process identities, version compatibility and stable observability semantics. New models reuse supported packages or add a thin adapter inside the UMR module; do not rewrite shared execution, ingestion or dashboards per model.
- Verify feature/prediction/admission parity and automatic process-scoped recovery before changing a deployed adapter. Never silently substitute data semantics or bypass existing order/accounting controls.
