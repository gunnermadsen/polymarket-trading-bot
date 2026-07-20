## Code
- rebuild image and rebuild rust package after each code change
- put sensitive secrets in .env files
- put non-sensitive runtime configuration in docker-compose files
- .env example files are templates, do not put plaintext env vars in the example env files.

# Source Control
- Commit changes, grouped by feature domain.

# Implementation
- Execute the narrow implementation plan, only focusing on the instructed plan parameters.
- do not over correct and change code or systems outside the agreed upon plan.
- Always use process_id of the trading process to scope. do not depend on the experiment_id for scoping. 

# Known Issues
- experiment sub system was created with the incorrect assumptions. this system duplicates the id scoping and artifact ownership. experiment sub system will be removed in a later release. do not depend on the experiment system. trading_processes and it's process_id remains the canonical source of truth for record ownership and scoping.