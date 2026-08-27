import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddTemperatureEnvironmentalFeatures1787760000000 implements MigrationInterface {
  name = 'AddTemperatureEnvironmentalFeatures1787760000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE weather.goes_abi_features (
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        station_id text NOT NULL,
        decision_time timestamptz NOT NULL,
        requested_offset_minutes integer NOT NULL,
        spatial_radius_km integer NOT NULL,
        sector text NOT NULL,
        feature_schema_version text NOT NULL,
        satellite text NOT NULL,
        scan_end timestamptz NOT NULL,
        clear_pixel_fraction double precision,
        cloudy_pixel_fraction double precision,
        infrared_brightness_temperature_mean_k double precision,
        infrared_brightness_temperature_stddev_k double precision,
        infrared_brightness_temperature_p10_k double precision,
        infrared_brightness_temperature_p50_k double precision,
        infrared_brightness_temperature_p90_k double precision,
        cloud_top_temperature_mean_k double precision,
        cloud_top_temperature_p10_k double precision,
        cloud_top_temperature_p50_k double precision,
        cloud_top_temperature_p90_k double precision,
        cloud_top_height_mean_m double precision,
        cloud_top_height_p10_m double precision,
        cloud_top_height_p50_m double precision,
        cloud_top_height_p90_m double precision,
        visible_reflectance_mean double precision,
        visible_reflectance_stddev double precision,
        visible_reflectance_p10 double precision,
        visible_reflectance_p50 double precision,
        visible_reflectance_p90 double precision,
        cloud_optical_depth_mean double precision,
        cloud_optical_depth_p50 double precision,
        cloud_optical_depth_p90 double precision,
        water_vapor_brightness_temperature_mean_k double precision,
        infrared_change_45m_k double precision,
        infrared_change_165m_k double precision,
        valid_pixel_fraction double precision NOT NULL,
        quality_flags jsonb NOT NULL DEFAULT '{}'::jsonb,
        source_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (
          process_id, station_id, decision_time, requested_offset_minutes,
          spatial_radius_km, sector, feature_schema_version
        ),
        CONSTRAINT chk_goes_station CHECK (station_id = 'KLGA'),
        CONSTRAINT chk_goes_decision_hour CHECK (
          EXTRACT(HOUR FROM decision_time AT TIME ZONE 'America/New_York') IN (0, 12)
        ),
        CONSTRAINT chk_goes_offset CHECK (requested_offset_minutes IN (15, 60, 180)),
        CONSTRAINT chk_goes_radius CHECK (spatial_radius_km IN (25, 50, 100)),
        CONSTRAINT chk_goes_sector CHECK (sector IN ('all', 'north', 'south', 'east', 'west')),
        CONSTRAINT chk_goes_satellite CHECK (satellite IN ('G16', 'G19')),
        CONSTRAINT chk_goes_causal CHECK (scan_end <= decision_time - interval '15 minutes'),
        CONSTRAINT chk_goes_fractions CHECK (
          valid_pixel_fraction BETWEEN 0 AND 1
          AND (clear_pixel_fraction IS NULL OR clear_pixel_fraction BETWEEN 0 AND 1)
          AND (cloudy_pixel_fraction IS NULL OR cloudy_pixel_fraction BETWEEN 0 AND 1)
        ),
        CONSTRAINT chk_goes_temperatures CHECK (
          (infrared_brightness_temperature_mean_k IS NULL OR infrared_brightness_temperature_mean_k BETWEEN 100 AND 400)
          AND (cloud_top_temperature_mean_k IS NULL OR cloud_top_temperature_mean_k BETWEEN 100 AND 400)
          AND (water_vapor_brightness_temperature_mean_k IS NULL OR water_vapor_brightness_temperature_mean_k BETWEEN 100 AND 400)
        ),
        CONSTRAINT chk_goes_height CHECK (
          cloud_top_height_mean_m IS NULL OR cloud_top_height_mean_m BETWEEN -1000 AND 30000
        ),
        CONSTRAINT chk_goes_documents CHECK (
          jsonb_typeof(quality_flags) = 'object' AND jsonb_typeof(source_metadata) = 'object'
        )
      );

      CREATE INDEX idx_goes_abi_features_decision
        ON weather.goes_abi_features (process_id, station_id, decision_time, feature_schema_version);

      CREATE TABLE weather.goes_abi_window_coverage (
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        station_id text NOT NULL,
        decision_time timestamptz NOT NULL,
        requested_offset_minutes integer NOT NULL,
        product text NOT NULL,
        feature_schema_version text NOT NULL,
        status text NOT NULL,
        satellite text NOT NULL,
        scan_start timestamptz,
        scan_end timestamptz,
        valid_pixel_fraction double precision,
        source_artifact_id uuid REFERENCES weather.source_artifacts (artifact_id) ON DELETE RESTRICT,
        cropped_artifact_path text,
        cropped_artifact_sha256 text,
        quality_flags jsonb NOT NULL DEFAULT '{}'::jsonb,
        source_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        checked_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (
          process_id, station_id, decision_time, requested_offset_minutes,
          product, feature_schema_version
        ),
        CONSTRAINT chk_goes_coverage_station CHECK (station_id = 'KLGA'),
        CONSTRAINT chk_goes_coverage_offset CHECK (requested_offset_minutes IN (15, 60, 180)),
        CONSTRAINT chk_goes_coverage_status CHECK (status IN (
          'complete', 'valid_zero', 'missing_source', 'insufficient_valid_pixels',
          'satellite_transition_issue', 'download_failure', 'processing_failure'
        )),
        CONSTRAINT chk_goes_coverage_satellite CHECK (satellite IN ('G16', 'G19')),
        CONSTRAINT chk_goes_coverage_causal CHECK (
          scan_end IS NULL OR scan_end <= decision_time - interval '15 minutes'
        ),
        CONSTRAINT chk_goes_coverage_fraction CHECK (
          valid_pixel_fraction IS NULL OR valid_pixel_fraction BETWEEN 0 AND 1
        ),
        CONSTRAINT chk_goes_coverage_patch_sha CHECK (
          cropped_artifact_sha256 IS NULL OR cropped_artifact_sha256 ~ '^[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_goes_coverage_documents CHECK (
          jsonb_typeof(quality_flags) = 'object' AND jsonb_typeof(source_metadata) = 'object'
        )
      );

      CREATE INDEX idx_goes_abi_coverage_status
        ON weather.goes_abi_window_coverage (
          process_id, feature_schema_version, status, decision_time, product
        );

      CREATE TABLE weather.hrrr_environment_features (
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        station_id text NOT NULL,
        decision_time timestamptz NOT NULL,
        model_run timestamptz NOT NULL,
        valid_at timestamptz NOT NULL,
        lead_hours integer NOT NULL,
        spatial_radius_km integer NOT NULL,
        sector text NOT NULL,
        feature_schema_version text NOT NULL,
        temperature_2m_mean_k double precision,
        temperature_2m_stddev_k double precision,
        dew_point_2m_mean_k double precision,
        dew_point_2m_stddev_k double precision,
        total_cloud_cover_mean_fraction double precision,
        total_cloud_cover_stddev_fraction double precision,
        downward_shortwave_radiation_mean_w_m2 double precision,
        wind_u_10m_mean_m_s double precision,
        wind_v_10m_mean_m_s double precision,
        wind_speed_10m_mean_m_s double precision,
        boundary_layer_height_mean_m double precision,
        accumulated_precipitation_mean_mm double precision,
        composite_reflectivity_mean_dbz double precision,
        composite_reflectivity_max_dbz double precision,
        valid_pixel_fraction double precision NOT NULL,
        quality_flags jsonb NOT NULL DEFAULT '{}'::jsonb,
        source_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (
          process_id, station_id, decision_time, valid_at,
          spatial_radius_km, sector, feature_schema_version
        ),
        CONSTRAINT chk_hrrr_environment_station CHECK (station_id = 'KLGA'),
        CONSTRAINT chk_hrrr_environment_decision_hour CHECK (
          EXTRACT(HOUR FROM decision_time AT TIME ZONE 'America/New_York') IN (0, 12)
        ),
        CONSTRAINT chk_hrrr_environment_availability CHECK (model_run <= decision_time - interval '75 minutes'),
        CONSTRAINT chk_hrrr_environment_lead CHECK (lead_hours >= 0),
        CONSTRAINT chk_hrrr_environment_radius CHECK (spatial_radius_km IN (0, 25, 50, 100)),
        CONSTRAINT chk_hrrr_environment_sector CHECK (sector IN ('all', 'north', 'south', 'east', 'west')),
        CONSTRAINT chk_hrrr_environment_fraction CHECK (
          valid_pixel_fraction BETWEEN 0 AND 1
          AND (total_cloud_cover_mean_fraction IS NULL OR total_cloud_cover_mean_fraction BETWEEN 0 AND 1)
        ),
        CONSTRAINT chk_hrrr_environment_temperature CHECK (
          (temperature_2m_mean_k IS NULL OR temperature_2m_mean_k BETWEEN 180 AND 340)
          AND (dew_point_2m_mean_k IS NULL OR dew_point_2m_mean_k BETWEEN 150 AND 330)
        ),
        CONSTRAINT chk_hrrr_environment_documents CHECK (
          jsonb_typeof(quality_flags) = 'object' AND jsonb_typeof(source_metadata) = 'object'
        )
      );

      CREATE INDEX idx_hrrr_environment_features_decision
        ON weather.hrrr_environment_features (
          process_id, station_id, decision_time, valid_at, feature_schema_version
        );

      CREATE TABLE weather.hrrr_environment_window_coverage (
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        station_id text NOT NULL,
        decision_time timestamptz NOT NULL,
        model_run timestamptz NOT NULL,
        valid_at timestamptz NOT NULL,
        feature_schema_version text NOT NULL,
        status text NOT NULL,
        fields_present text[] NOT NULL DEFAULT '{}',
        fields_missing text[] NOT NULL DEFAULT '{}',
        source_artifact_id uuid REFERENCES weather.source_artifacts (artifact_id) ON DELETE RESTRICT,
        cropped_artifact_path text,
        cropped_artifact_sha256 text,
        valid_pixel_fraction double precision,
        quality_flags jsonb NOT NULL DEFAULT '{}'::jsonb,
        source_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        checked_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (
          process_id, station_id, decision_time, valid_at, feature_schema_version
        ),
        CONSTRAINT chk_hrrr_environment_coverage_station CHECK (station_id = 'KLGA'),
        CONSTRAINT chk_hrrr_environment_coverage_availability CHECK (
          model_run <= decision_time - interval '75 minutes'
        ),
        CONSTRAINT chk_hrrr_environment_coverage_status CHECK (status IN (
          'complete', 'valid_zero', 'missing_source', 'insufficient_valid_pixels',
          'download_failure', 'processing_failure'
        )),
        CONSTRAINT chk_hrrr_environment_coverage_fraction CHECK (
          valid_pixel_fraction IS NULL OR valid_pixel_fraction BETWEEN 0 AND 1
        ),
        CONSTRAINT chk_hrrr_environment_coverage_patch_sha CHECK (
          cropped_artifact_sha256 IS NULL OR cropped_artifact_sha256 ~ '^[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_hrrr_environment_coverage_documents CHECK (
          jsonb_typeof(quality_flags) = 'object' AND jsonb_typeof(source_metadata) = 'object'
        )
      );

      CREATE INDEX idx_hrrr_environment_coverage_status
        ON weather.hrrr_environment_window_coverage (
          process_id, feature_schema_version, status, decision_time, valid_at
        );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DROP TABLE IF EXISTS weather.hrrr_environment_window_coverage;
      DROP TABLE IF EXISTS weather.hrrr_environment_features;
      DROP TABLE IF EXISTS weather.goes_abi_window_coverage;
      DROP TABLE IF EXISTS weather.goes_abi_features;
    `);
  }
}
