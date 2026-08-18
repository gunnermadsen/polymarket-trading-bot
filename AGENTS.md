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
- rebuild docker image after each code change, ensuring new rust change build as packages with the image.
- build the polymarket bot with provenance env var: POLYMARKET_GIT_REVISION=<GIT_COMMIT_HASH> docker compose build polymarket-bot
- do not rebuild polymarket-bot or rust-related packages on grafana dashboard changes or database configuration changes.
- put sensitive secrets in .env files
- put non-sensitive runtime configuration in docker-compose files
- .env example files are templates, do not put plaintext env vars in the example env files.

# Source Control and Worktrees

## Integration Cycle and Trading Release Roles

- `development` is the settlement branch for accepted trading-capable releases. Do not implement features, fixes, experiments, or integration corrections directly on `development`.
- A `development` tip is golden only when an annotated `golden/<image_name>/sha256-<docker-sha256-hash>` tag identifies the exact accepted image. Branch position or an `image/...` build tag alone is not golden evidence.
- Feature verification proves readiness for integration; it does not make a feature branch or its image golden.
- Use exactly one active integration branch for each golden-image build cycle. Name it `integration-<YYYY-MM-DD>`.
- Create the integration branch once from the current accepted `development` tip at the beginning of the cycle. Record a unique cycle-opening commit and an annotated `integration-cycle/<YYYY-MM-DD>` tag before creating feature or defect branches so branch eligibility can be verified by ancestry.
- The active integration branch is always checked out in the main repository worktree. Never create or keep it in a disposable worktree under `target/worktrees`.
- The integration branch is the single collection point for the cycle. Do not create feature-specific, defect-specific, candidate-specific, or secondary integration branches.
- Creating the integration branch is the only point where the cycle branches from `development`. After it exists, every new feature or defect intended for that cycle starts from the latest integration tip and merges back into that same integration branch.
- A narrowly scoped integration-policy or coordination correction may be committed directly on the integration branch when the user explicitly requests it. Feature and defect implementation still use branches rooted in the active integration lineage.
- Merge a selected verified feature or defect branch into the integration branch with `--no-ff` only after the user explicitly authorizes merging that exact branch. Never merge a feature or defect branch directly into `development`.
- Abandon a rejected candidate branch rather than repairing its integration history with merge reverts. Preserve the rejected branch until its result and any reusable commits are accounted for.

## Abandoned Lineages

- Before deleting or otherwise retiring an intentionally discarded branch, divergent commit, rejected candidate, or superseded release snapshot, create an annotated tag on its final retained commit using `abandoned/<domain>/git-<full-git-commit-id>`.
- The abandoned tag annotation records the original branch or ref when known, the reason for abandonment, the replacement or superseding commit when one exists, any image identity built from it, and whether it was ever deployed.
- An `abandoned/...` tag permanently excludes that lineage and its images from integration, candidate admission, golden promotion, and rollback selection unless the user explicitly restores it through a new reviewed lineage.
- Preserve abandoned tags when removing worktrees or branches. An image build tag may remain for provenance, but it does not override abandoned status.
- Release and integration tasks must inspect `abandoned/...` tags before selecting branches, commits, or images and must fail closed rather than merge or deploy an abandoned lineage implicitly.

## Golden Image Admission and Promotion

- The user may explicitly authorize minting a golden image from an exact integration commit and immutable image at any time. That instruction is sufficient promotion authorization and must not be delayed or refused because of an undefined waiting period, elapsed-time requirement, or missing operational evidence.
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
- Promote by fast-forwarding `development` with `--ff-only` to the exact selected candidate commit. Do not create a promotion merge commit, rebuild the image, rewrite `development`, or force-push.
- Create an annotated `golden/<image_name>/sha256-<docker-sha256-hash>` tag recording the Git revision, immutable image identity, migration/config/model identity, checks performed, and accepted limitations.
- Promote or alias the selected already-built image manifest; never mint a replacement build as golden. Verify the resulting branch, tag, image identity, and embedded provenance.
- If the candidate diverges from `development`, `development` advanced after candidate creation, multiple candidates claim eligibility, or promotion would require history rewriting, stop and report the exact condition.
- Roll back by deploying a previously annotated immutable golden image. Do not rebuild the old commit and do not reset `development` merely to change the deployed image.

## Branching

- Commit changes in coherent groups organized by feature domain.
- When building a new image, tag the commit for which the image was built:
-- using this pattern for container images: image/<image_name>/sha256-<docker-sha256-hash>
-- using this pattern for a container image using a new model: model/<model-name-with-metadata>
- Use only these branch names for new cycle work: `integration-<YYYY-MM-DD>`, `feature/<feature-name>`, and `defect/<defect-name>`.
- Every new, independent feature domain must use a dedicated `feature/<feature-name>` branch. Every defect must use a dedicated `defect/<defect-name>` branch.
- Create feature and defect branches from the latest tip of the active integration branch, never from `development` while an integration cycle is active.
- Follow-up work that must inherit an existing feature or training lineage starts from that lineage’s designated base or integration branch, not from `development`.
- Keep unrelated feature domains on separate branches.
- Do not merge a feature or defect branch directly into `development`.
- Creating a `feature/...` or `defect/...` branch from the active integration branch authorizes isolated work on that branch only; it does not authorize merging it back. Keep the branch unmerged until the user explicitly grants permission to merge that exact branch into integration. Completing implementation, committing, testing, reviewing, or declaring the branch ready does not imply merge permission. If permission is absent or ambiguous, stop before the merge and ask for authorization.
- Before merging a feature or defect branch into integration, verify that the exact selected branch:
  - was explicitly identified for integration;
  - descends from the active cycle marker `integration-cycle/<YYYY-MM-DD>`;
  - is not marked by an `abandoned/...` tag;
  - has clean, committed, proportionately tested work;
  - has incorporated the latest integration tip when other work has been collected since it branched; and
  - has a reviewed commit log, merge base, and diff against the active integration branch.
- Merge each admitted feature or defect branch into the active integration branch using `--no-ff`. Never infer merge eligibility from recency, branch-name similarity, worktree existence, dirty state, or whether Git reports the branch as unmerged.
- Do not merge an old, pre-cycle, cross-cycle, or otherwise unrelated branch unless the user explicitly authorizes that exact branch or commit. Prefer applying explicitly selected commits onto a new branch rooted in the active integration cycle when old work must be recovered.
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

# Database and migrations
- all non trading process database mutations or changes must be executed through database migrations.
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
