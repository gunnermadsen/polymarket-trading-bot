# North Star
Think of Capitonic as a vision to generate income through systems with automation and algorithms, that require no customers. The polymarket trading bot is an extension of that vision, to systemically generate an  income stream based on a reproducable strategy that can trade with positive net expectancy through a narrow and simplified system. the key is positive net expectancy and a simplified system. Capitonic was not envisioned to live in a labatory forever. Capitonic was envisioned to prove a concept, that an automated trading bot can become a real product, a real business. Captitonic is NOT here to be reduced to a pet project or a labratory, or an experiment platform. Capitonic exists to discover an edge, exploited with a strategy, and repeated to generate net expectancy. Do not make the mistake of seeing Capitonic as a "research platform", because it is not that. Capitonic is not a school, or a university; it is a business. Businesses rely on income to survive. If Capitonic produces income, Capitonic WILL thrive.

## Golden Rules
- All Rust features and systems are implemented and optimized for latency, memory usage efficiency, CPU performance, zero downtime, and database resource usage, ensuring a performant and resiliant system.
- Do not name artifacts, branches, files or code comments based on phases or stages of an implementation. use domain specific naming for the issue or feature being addresses. I dont' want to see 'phase X' in git history, code files, branches, or file names or code comments. 

## Code
- rebuild docker image after each code change, ensuring new rust change build as packages with the image.
- build the polymarket bot with provenance env var: POLYMARKET_GIT_REVISION=<GIT_COMMIT_HASH> docker compose build polymarket-bot
- do not rebuild polymarket-bot or rust-related packages on grafana dashboard changes or database configuration changes.
- put sensitive secrets in .env files
- put non-sensitive runtime configuration in docker-compose files
- .env example files are templates, do not put plaintext env vars in the example env files.

# Source Control and Worktrees

## Branching

- Commit changes in coherent groups organized by feature domain.
- Every new, independent feature domain must use a dedicated feature branch.
- An independent feature branch starts from the latest `development` commit.
- Follow-up work that must inherit an existing feature or training lineage starts from that lineage’s designated base or integration branch, not from `development`.
- Keep unrelated feature domains on separate branches.
- Do not merge a single feature branch into `development` unless the user explicitly requests integration.
- When a task contains multiple completed feature branches that must be integrated, merge each stable branch into the designated integration branch using `--no-ff`. Merge the integration branch into `development` only when explicitly requested.
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
- Each worktree owns one overarching feature domain and one designated feature or integration branch.
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