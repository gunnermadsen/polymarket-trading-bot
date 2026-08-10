# syntax=docker/dockerfile:1.7

ARG RUST_VERSION=1.92.0
FROM rust:${RUST_VERSION}-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
RUN mkdir src \
    && printf 'fn main() {}\n' > src/main.rs \
    && printf 'pub fn placeholder() {}\n' > src/lib.rs \
    && cargo build --locked --release \
    && rm -rf src

COPY src ./src
RUN find src -type f -exec touch {} + \
    && cargo build --locked --release \
    && cp target/release/market-data-ingester /usr/local/bin/market-data-ingester

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10003 --create-home --shell /usr/sbin/nologin ingester

ARG MARKET_DATA_INGESTER_GIT_REVISION
RUN case "$MARKET_DATA_INGESTER_GIT_REVISION" in \
      ""|*[!0-9a-f]*) \
        echo "MARKET_DATA_INGESTER_GIT_REVISION must be a lowercase hexadecimal Git commit ID" >&2; \
        exit 64 \
        ;; \
    esac \
    && case "${#MARKET_DATA_INGESTER_GIT_REVISION}" in \
      40|64) ;; \
      *) \
        echo "MARKET_DATA_INGESTER_GIT_REVISION must contain 40 or 64 characters" >&2; \
        exit 64 \
        ;; \
    esac

LABEL org.opencontainers.image.source="https://github.com/gunnermadsen/polymarket-trading-bot" \
      org.opencontainers.image.revision="$MARKET_DATA_INGESTER_GIT_REVISION"

COPY --from=builder /usr/local/bin/market-data-ingester /usr/local/bin/market-data-ingester

USER ingester
ENV RUST_LOG=info
ENTRYPOINT ["/usr/local/bin/market-data-ingester"]
