FROM rust:1.91-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY packages/polymarket-bot/Cargo.toml packages/polymarket-bot/Cargo.lock ./
COPY packages/polymarket-bot/build.rs ./build.rs
COPY packages/polymarket-bot/src ./src

RUN cargo build --locked --release

FROM builder AS test

COPY packages/polymarket-bot/tests ./tests
COPY experiments/btc-updown/fixtures /experiments/btc-updown/fixtures

RUN cargo test --locked --all-targets --no-run

CMD ["cargo", "test", "--locked", "--all-targets", "--", "--include-ignored"]

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
