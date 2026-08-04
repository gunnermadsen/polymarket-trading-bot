FROM rust:1.91-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY packages/polymarket-bot/Cargo.toml packages/polymarket-bot/Cargo.lock ./
COPY packages/polymarket-bot/build.rs ./build.rs
COPY packages/polymarket-bot/src ./src
COPY packages/polymarket-bot/tests ./tests
COPY packages/btc-directional-model/runtime-models /opt/polymarket-models
COPY README.md /README.md

ENV POLYMARKET_BTC_MODEL_DIR=/opt/polymarket-models

RUN chmod -R u=rwX,go=rX /opt/polymarket-models \
    && cargo test --locked --all-targets -- --include-ignored \
    && cargo build --locked --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10002 --create-home --shell /usr/sbin/nologin polymarket \
    && mkdir -p /var/lib/polymarket/backfill-cache /var/lib/kraken-data \
    && chown -R polymarket:polymarket /var/lib/polymarket /var/lib/kraken-data

ARG POLYMARKET_GIT_REVISION
RUN case "$POLYMARKET_GIT_REVISION" in \
      ""|*[!0-9a-f]*) \
        echo "POLYMARKET_GIT_REVISION must be a lowercase hexadecimal Git commit ID" >&2; \
        exit 64 \
        ;; \
    esac \
    && case "${#POLYMARKET_GIT_REVISION}" in \
      40|64) ;; \
      *) \
        echo "POLYMARKET_GIT_REVISION must contain 40 or 64 characters" >&2; \
        exit 64 \
        ;; \
    esac

LABEL org.opencontainers.image.source="https://github.com/gunnermadsen/polymarket-trading-bot" \
      org.opencontainers.image.revision="$POLYMARKET_GIT_REVISION"

COPY --from=builder /app/target/release/polymarket-bot /usr/local/bin/polymarket-bot
COPY --from=builder /app/target/release/polymarket-backfill-worker /usr/local/bin/polymarket-backfill-worker
COPY --from=builder /app/target/release/binance-l2-backfill-plan /usr/local/bin/binance-l2-backfill-plan
COPY --from=builder /app/target/release/binance-spot-l2-backfill-plan /usr/local/bin/binance-spot-l2-backfill-plan
COPY --from=builder /app/target/release/binance-spot-l2-huggingface-backfill-plan /usr/local/bin/binance-spot-l2-huggingface-backfill-plan
COPY --from=builder /app/target/release/kraken-backfill-worker /usr/local/bin/kraken-backfill-worker
COPY --from=builder /app/target/release/kraken-backfill-plan /usr/local/bin/kraken-backfill-plan
COPY --from=builder /app/target/release/derive-clob-creds /usr/local/bin/derive-clob-creds
COPY --from=builder /opt/polymarket-models /usr/local/share/polymarket-bot/models

USER polymarket
ENV RUST_LOG=info
ENTRYPOINT ["/usr/local/bin/polymarket-bot"]
