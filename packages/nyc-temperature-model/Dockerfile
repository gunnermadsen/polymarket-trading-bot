FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libeccodes0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY packages/nyc-temperature-model/pyproject.toml ./
COPY packages/nyc-temperature-model/src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install .

ENTRYPOINT ["nyc-temperature-model"]
