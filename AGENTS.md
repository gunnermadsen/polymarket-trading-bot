## Golden Rules
- All Rust features and systems are implemented and optimized for latency, memory usage efficiency, CPU performance, zero downtime, and database resource usage, ensuring a performant and resiliant system.
- Do not name artifacts, branches, files or code comments based on phases or stages of an implementation. use domain specific naming for the issue or feature being addresses. I dont' want to see 'phase X' in git history, code files, branches, or file names or code comments. 

## Code
- rebuild image and rebuild rust package after each code change.
- do not rebuild polymarket-bot or rust-related packages on grafana dashboard changes or database configuration changes.
- put sensitive secrets in .env files
- put non-sensitive runtime configuration in docker-compose files
- .env example files are templates, do not put plaintext env vars in the example env files.

# Source Control
- Commit changes, grouped by feature domain.
- New features must be committed to a new branch, based from the latest commit on development branch.

# Database
- all database mutations or changes must be executed through database migrations.
- all diagnostic database queries to read the database must be optimized for performance to prevent database crashes.
- use the db-migrate microservice job to apply migrations
- apply migrations by creating new migration files inside packages/db-migrate/src/migrations
- then recreate the container:

```bash
docker compose up -d --force-recreate --no-deps db-migrate
```

# Implementation
- Execute the narrow implementation plan, only focusing on the instructed plan parameters.
- do not over correct and change code or systems outside the agreed upon plan.
- Always use process_id of the trading process to scope. do not depend on the experiment_id for scoping. 

# Known Issues
- experiment sub system was created with the incorrect assumptions. this system duplicates the id scoping and artifact ownership. experiment sub system will be removed in a later release. do not depend on the experiment system. trading_processes and it's process_id remains the canonical source of truth for record ownership and scoping.