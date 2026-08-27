from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nyc_temperature_model.goes_ingestion import (
    GOES_TRANSITION,
    PRODUCTS,
    _extract_patch,
    operational_satellite,
    select_causal_scan,
)
from nyc_temperature_model.hrrr_environment_ingestion import (
    FIELD_ALIASES,
    _extract_environment_patch,
)
from nyc_temperature_model.spatial_features import numeric_summary, spatial_mask


def _goes_key(satellite: int, start: str, end: str) -> str:
    return (
        "ABI-L2-CMIPC/2025/097/12/"
        f"OR_ABI-L2-CMIPC-M6C13_G{satellite}_s{start}_e{end}_c{end}.nc"
    )


def test_goes_operational_transition_uses_noaa_declaration_instant():
    assert operational_satellite(GOES_TRANSITION - timedelta(seconds=1)) == "G16"
    assert operational_satellite(GOES_TRANSITION) == "G19"


def test_goes_scan_selection_is_causal_and_nearest_to_requested_offset():
    decision = datetime(2025, 4, 7, 12, 25, tzinfo=UTC)
    keys = [
        _goes_key(16, "20250971150000", "20250971155000"),
        _goes_key(16, "20250971200000", "20250971205000"),
        _goes_key(16, "20250971210000", "20250971215000"),
    ]
    selected = select_causal_scan(keys, target=decision - timedelta(minutes=15), decision_time=decision)

    assert selected is not None
    assert selected[2] == datetime(2025, 4, 7, 12, 5, tzinfo=UTC)
    assert selected[2] <= decision - timedelta(minutes=15)


def test_goes_scan_selection_returns_missing_evidence_instead_of_future_data():
    decision = datetime(2025, 4, 7, 12, 25, tzinfo=UTC)
    keys = [_goes_key(16, "20250971210000", "20250971215000")]

    assert (
        select_causal_scan(keys, target=decision - timedelta(minutes=15), decision_time=decision)
        is None
    )


def test_spatial_sectors_are_fixed_and_do_not_treat_missing_pixels_as_zero():
    latitudes = np.array([[40.7, 40.7, 40.7], [40.77, 40.77, 40.77], [40.84, 40.84, 40.84]])
    longitudes = np.array([[-73.95, -73.87, -73.79]] * 3)
    values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    north = spatial_mask(latitudes, longitudes, 25, "north")
    summary = numeric_summary(values, north)

    assert summary["mean"] > 0
    assert 0 < summary["valid_pixel_fraction"] <= 1


def test_small_goes_netcdf_fixture_is_cropped_and_summarized(tmp_path, monkeypatch):
    source = tmp_path / "goes-source.nc"
    patch = tmp_path / "goes-patch.nc"
    values = np.arange(25, dtype=float).reshape(5, 5) + 250
    dataset = xr.Dataset(
        data_vars={
            "CMI": (("y", "x"), values),
            "DQF": (("y", "x"), np.zeros((5, 5), dtype=np.int16)),
            "goes_imager_projection": ((), 0),
        },
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    dataset["goes_imager_projection"].attrs.update(
        {
            "perspective_point_height": 35786023.0,
            "longitude_of_projection_origin": -75.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31414,
        }
    )
    dataset.to_netcdf(source, engine="h5netcdf")
    latitude = np.linspace(39.8, 41.8, 25).reshape(5, 5)
    longitude = np.linspace(-74.8, -72.8, 25).reshape(5, 5)
    latitude[2, 2] = 40.7769
    longitude[2, 2] = -73.8740
    monkeypatch.setattr(
        "nyc_temperature_model.goes_ingestion._goes_lat_lon",
        lambda _dataset: (latitude, longitude),
    )

    summaries, metadata = _extract_patch(source, PRODUCTS[0], patch)

    assert patch.is_file()
    assert summaries[(25, "all")]["mean"] == 261.5
    assert summaries[(25, "all")]["valid_pixel_fraction"] == 1.0
    assert metadata["variable"] == "CMI"


def test_small_hrrr_grib_fixture_fields_are_cropped_to_reproducible_netcdf(
    tmp_path, monkeypatch
):
    source = tmp_path / "hrrr-fixture.grib2"
    source.write_bytes(b"synthetic fixture routed through decoded field adapter")
    patch = tmp_path / "hrrr-patch.nc"
    latitude = np.linspace(40.0, 41.5, 25).reshape(5, 5)
    longitude = np.linspace(-74.6, -73.1, 25).reshape(5, 5)
    latitude[2, 2] = 40.7769
    longitude[2, 2] = -73.8740
    fields = {name: np.full((5, 5), float(index + 1)) for index, name in enumerate(FIELD_ALIASES)}
    fields["temperature_2m"][:] = 290.0
    fields["dew_point_2m"][:] = 280.0
    fields["total_cloud_cover"][:] = 0.4
    monkeypatch.setattr(
        "nyc_temperature_model.hrrr_environment_ingestion._load_grib_fields",
        lambda _path: (fields, latitude, longitude, {name: "fixture" for name in fields}),
    )

    summaries, present, units = _extract_environment_patch(source, patch)

    assert patch.is_file()
    assert set(present) == set(FIELD_ALIASES)
    assert summaries[(0, "all")]["temperature_2m"]["mean"] == 290.0
    assert summaries[(100, "all")]["total_cloud_cover"]["mean"] == pytest.approx(0.4)
    assert units["temperature_2m"] == "fixture"


def test_environment_migration_uses_typed_columns_and_versioned_keys():
    migration = Path(__file__).parents[2] / "db-migrate/src/migrations/weather/1787760000000-AddTemperatureEnvironmentalFeatures.ts"
    text = migration.read_text()

    assert "weather.goes_abi_features" in text
    assert "weather.goes_abi_window_coverage" in text
    assert "weather.hrrr_environment_features" in text
    assert "weather.hrrr_environment_window_coverage" in text
    assert "feature_schema_version" in text
    assert "infrared_brightness_temperature_mean_k double precision" in text
    assert "total_cloud_cover_mean_fraction double precision" in text


def test_temperature_compose_has_exactly_two_restricted_environment_workers():
    compose = (Path(__file__).parents[3] / "docker-compose.temperature.yml").read_text()

    assert compose.count("temperature-environment-worker-1:") == 1
    assert compose.count("temperature-environment-worker-2:") == 1
    assert compose.count(
        'WEATHER_WORKER_INGESTERS: "goes_abi_klga_features,hrrr_environment_features"'
    ) == 2
    assert "polymarket-bot:" not in compose
