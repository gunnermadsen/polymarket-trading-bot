from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_INGESTERS = (
    "polymarket_temperature_markets",
    "polymarket_temperature_price_history",
    "asos_station_observations",
    "asos_resolution_observations",
    "asos_one_minute_observations",
    "hrrr_point_forecasts",
    "pmxt_temperature_execution",
)


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _env_nonnegative_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _env_worker_ingesters() -> tuple[str, ...]:
    raw = os.environ.get("WEATHER_WORKER_INGESTERS")
    if raw is None:
        return SUPPORTED_INGESTERS
    ingesters = tuple(dict.fromkeys(value.strip() for value in raw.split(",") if value.strip()))
    if not ingesters:
        raise ValueError("WEATHER_WORKER_INGESTERS must contain at least one ingester")
    unsupported = sorted(set(ingesters) - set(SUPPORTED_INGESTERS))
    if unsupported:
        raise ValueError(
            "WEATHER_WORKER_INGESTERS contains unsupported ingesters: " + ", ".join(unsupported)
        )
    return ingesters


@dataclass(frozen=True)
class Settings:
    database_url: str
    cache_directory: Path
    model_directory: Path
    report_directory: Path
    worker_id: str
    worker_ingesters: tuple[str, ...]
    poll_seconds: int
    lease_seconds: int
    gamma_base_url: str
    clob_base_url: str
    pmxt_base_url: str
    iem_asos_metar_url: str
    iem_asos_one_minute_url: str
    hrrr_download_attempts: int
    hrrr_retry_base_ms: int
    hrrr_retry_max_ms: int
    hrrr_request_interval_ms: int
    hrrr_source_priority: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        database_url = os.environ.get("WEATHER_DATABASE_URL")
        if not database_url:
            host = os.environ.get("POSTGRES_HOST", "temperature-postgres")
            port = os.environ.get("POSTGRES_PORT", "5432")
            user = os.environ.get("POSTGRES_USER", "postgres")
            password = os.environ.get("POSTGRES_PASSWORD", "")
            auth = f"{user}:{password}" if password else user
            database = os.environ.get("POSTGRES_DB", "temperature_expectancy")
            database_url = f"postgresql://{auth}@{host}:{port}/{database}"
        cache = Path(os.environ.get("WEATHER_CACHE_DIR", "/var/lib/weather/cache"))
        models = Path(os.environ.get("WEATHER_MODEL_DIR", "/var/lib/weather/models"))
        reports = Path(os.environ.get("WEATHER_REPORT_DIR", "/var/lib/weather/reports"))
        hrrr_retry_base_ms = _env_int("HRRR_RETRY_BASE_MS", 1000)
        hrrr_retry_max_ms = _env_int("HRRR_RETRY_MAX_MS", 30000)
        if hrrr_retry_max_ms < hrrr_retry_base_ms:
            raise ValueError("HRRR_RETRY_MAX_MS must be at least HRRR_RETRY_BASE_MS")
        hrrr_source_priority = tuple(
            source.strip().lower()
            for source in os.environ.get("HRRR_SOURCE_PRIORITY", "google,aws,nomads").split(",")
            if source.strip()
        )
        if not hrrr_source_priority:
            raise ValueError("HRRR_SOURCE_PRIORITY must contain at least one source")
        return cls(
            database_url=database_url,
            cache_directory=cache,
            model_directory=models,
            report_directory=reports,
            worker_id=os.environ.get("WEATHER_WORKER_ID", "temperature-worker-1"),
            worker_ingesters=_env_worker_ingesters(),
            poll_seconds=_env_int("WEATHER_WORKER_POLL_SECONDS", 2),
            lease_seconds=_env_int("WEATHER_WORKER_LEASE_SECONDS", 1800),
            gamma_base_url=os.environ.get(
                "POLYMARKET_GAMMA_BASE_URL", "https://gamma-api.polymarket.com"
            ).rstrip("/"),
            clob_base_url=os.environ.get(
                "POLYMARKET_CLOB_BASE_URL", "https://clob.polymarket.com"
            ).rstrip("/"),
            pmxt_base_url=os.environ.get(
                "POLYMARKET_PMXT_ARCHIVE_BASE_URL", "https://r2v2.pmxt.dev"
            ).rstrip("/"),
            iem_asos_metar_url=os.environ.get(
                "IEM_ASOS_METAR_URL",
                "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
            ),
            iem_asos_one_minute_url=os.environ.get(
                "IEM_ASOS_ONE_MINUTE_URL",
                "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py",
            ),
            hrrr_download_attempts=_env_int("HRRR_DOWNLOAD_ATTEMPTS", 8),
            hrrr_retry_base_ms=hrrr_retry_base_ms,
            hrrr_retry_max_ms=hrrr_retry_max_ms,
            hrrr_request_interval_ms=_env_nonnegative_int("HRRR_REQUEST_INTERVAL_MS", 500),
            hrrr_source_priority=hrrr_source_priority,
        )

    def prepare_directories(self) -> None:
        for directory in (self.cache_directory, self.model_directory, self.report_directory):
            directory.mkdir(parents=True, exist_ok=True)
