FROM python:3.12-slim

ARG POLYMARKET_GIT_REVISION
RUN case "$POLYMARKET_GIT_REVISION" in \
      ""|*[!0-9a-f]*) \
        echo "POLYMARKET_GIT_REVISION must be a 40-character lowercase Git commit ID" >&2; \
        exit 1 ;; \
    esac \
    && if [ "${#POLYMARKET_GIT_REVISION}" -ne 40 ]; then \
      echo "POLYMARKET_GIT_REVISION must be a 40-character lowercase Git commit ID" >&2; \
      exit 1; \
    fi

LABEL org.opencontainers.image.revision="$POLYMARKET_GIT_REVISION"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POLYMARKET_GIT_REVISION="$POLYMARKET_GIT_REVISION"

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libeccodes0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY packages/nyc-temperature-model/pyproject.toml ./
COPY packages/nyc-temperature-model/src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install .

ENTRYPOINT ["nyc-temperature-model"]
