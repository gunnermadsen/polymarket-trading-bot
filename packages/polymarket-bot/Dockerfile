FROM rust:1.91-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY packages/polymarket-bot/Cargo.toml packages/polymarket-bot/Cargo.lock ./
COPY vendor/polymarket_client_sdk_v2 /vendor/polymarket_client_sdk_v2
COPY packages/polymarket-bot/src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10002 --create-home --shell /usr/sbin/nologin polymarket

COPY --from=builder /app/target/release/polymarket-bot /usr/local/bin/polymarket-bot
COPY --from=builder /app/target/release/derive-clob-creds /usr/local/bin/derive-clob-creds

USER polymarket
ENV RUST_LOG=info
ENTRYPOINT ["/usr/local/bin/polymarket-bot"]
