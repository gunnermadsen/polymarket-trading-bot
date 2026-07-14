FROM postgres:14-alpine

RUN apk add --no-cache bash ca-certificates coreutils curl jq

WORKDIR /workspace

COPY scripts/btc-paper-experiment-audit.sh /workspace/scripts/btc-paper-experiment-audit.sh
COPY docs/sql/btc-5m-paper-experiment-readiness.sql /workspace/docs/sql/btc-5m-paper-experiment-readiness.sql
COPY docs/sql/btc-5m-paper-experiment-report.sql /workspace/docs/sql/btc-5m-paper-experiment-report.sql
COPY docs/experiments /workspace/docs/experiments
COPY packages/db-migrate/src/migrations/1777123000000-AddBtcPhase6PaperCapital.ts \
  /workspace/packages/db-migrate/src/migrations/1777123000000-AddBtcPhase6PaperCapital.ts
COPY packages/db-migrate/src/migrations/1777124000000-AllowUnstartedTradingProcesses.ts \
  /workspace/packages/db-migrate/src/migrations/1777124000000-AllowUnstartedTradingProcesses.ts

RUN chmod 0555 /workspace/scripts/btc-paper-experiment-audit.sh \
    && mkdir -p /workspace/docs/evidence

ENTRYPOINT ["/workspace/scripts/btc-paper-experiment-audit.sh"]
