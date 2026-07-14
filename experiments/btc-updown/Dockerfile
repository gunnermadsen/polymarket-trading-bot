FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/btc-updown/src \
    HOME=/tmp

WORKDIR /opt/btc-updown

RUN mkdir -p /datasets /artifacts \
    && chown -R 10003:10003 /opt/btc-updown /datasets /artifacts

COPY --chown=10003:10003 experiments/btc-updown/pyproject.toml ./pyproject.toml
COPY --chown=10003:10003 experiments/btc-updown/README.md ./README.md
COPY --chown=10003:10003 experiments/btc-updown/src ./src
COPY --chown=10003:10003 experiments/btc-updown/tests ./tests
COPY --chown=10003:10003 experiments/btc-updown/fixtures ./fixtures

USER 10003:10003

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
