FROM postgres:14-alpine

RUN apk add --no-cache bash ca-certificates coreutils curl jq

WORKDIR /workspace

COPY scripts/btc-phase6-audit.sh /workspace/scripts/btc-phase6-audit.sh
COPY docs/sql/btc-5m-paper-readiness.sql /workspace/docs/sql/btc-5m-paper-readiness.sql
COPY docs/sql/btc-5m-phase6-report.sql /workspace/docs/sql/btc-5m-phase6-report.sql
COPY docs/experiments /workspace/docs/experiments
COPY packages/db-migrate/src/migrations/1777123000000-AddBtcPhase6PaperCapital.ts \
  /workspace/packages/db-migrate/src/migrations/1777123000000-AddBtcPhase6PaperCapital.ts

RUN chmod 0555 /workspace/scripts/btc-phase6-audit.sh \
    && mkdir -p /workspace/docs/evidence

ENTRYPOINT ["/workspace/scripts/btc-phase6-audit.sh"]
