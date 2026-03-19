-- WattCast v2 Feature Store + Forecast Archive
-- Executed by Postgres on first start via docker-compose volume mount.

CREATE TABLE IF NOT EXISTS raw_demand (
    id              SERIAL PRIMARY KEY,
    region          VARCHAR(10) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    demand_mw       DOUBLE PRECISION,
    forecast_mw     DOUBLE PRECISION,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (region, timestamp)
);

CREATE TABLE IF NOT EXISTS raw_weather (
    id                          SERIAL PRIMARY KEY,
    region                      VARCHAR(10) NOT NULL,
    timestamp                   TIMESTAMPTZ NOT NULL,
    temperature_2m              DOUBLE PRECISION,
    apparent_temperature        DOUBLE PRECISION,
    relative_humidity_2m        DOUBLE PRECISION,
    dew_point_2m                DOUBLE PRECISION,
    wind_speed_10m              DOUBLE PRECISION,
    wind_speed_80m              DOUBLE PRECISION,
    wind_speed_120m             DOUBLE PRECISION,
    wind_direction_10m          DOUBLE PRECISION,
    shortwave_radiation         DOUBLE PRECISION,
    direct_normal_irradiance    DOUBLE PRECISION,
    diffuse_radiation           DOUBLE PRECISION,
    cloud_cover                 DOUBLE PRECISION,
    precipitation               DOUBLE PRECISION,
    snowfall                    DOUBLE PRECISION,
    surface_pressure            DOUBLE PRECISION,
    soil_temperature_0cm        DOUBLE PRECISION,
    weather_code                INTEGER,
    ingested_at                 TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (region, timestamp)
);

CREATE TABLE IF NOT EXISTS forecasts (
    id                  SERIAL PRIMARY KEY,
    region              VARCHAR(10) NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    predicted_demand_mw DOUBLE PRECISION NOT NULL,
    scored_at           TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (region, timestamp, scored_at)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_raw_demand_region_ts ON raw_demand (region, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_weather_region_ts ON raw_weather (region, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_region_ts ON forecasts (region, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_scored_at ON forecasts (scored_at);

-- Audit trail — records model versions, data hashes, feature hashes per scoring run
CREATE TABLE IF NOT EXISTS audit_trail (
    id                  SERIAL PRIMARY KEY,
    region              VARCHAR(10) NOT NULL,
    scored_at           TEXT NOT NULL,
    demand_source       VARCHAR(20) NOT NULL,
    weather_source      VARCHAR(20) NOT NULL,
    demand_rows         INTEGER NOT NULL,
    weather_rows        INTEGER NOT NULL,
    model_versions      JSONB,
    ensemble_weights    JSONB,
    feature_count       INTEGER,
    feature_hash        TEXT,
    mape                JSONB,
    peak_forecast_mw    DOUBLE PRECISION,
    scoring_mode        VARCHAR(30),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_region ON audit_trail (region, created_at DESC);

-- Pipeline logs — step-by-step timing for each scoring run
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id              SERIAL PRIMARY KEY,
    pipeline_name   VARCHAR(50) NOT NULL,
    region          VARCHAR(10),
    scored_at       TEXT,
    steps           JSONB NOT NULL,
    total_ms        DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_logs_ts ON pipeline_logs (created_at DESC);

-- Data freshness tracking — prevents wasted work by recording last-seen timestamps
CREATE TABLE IF NOT EXISTS data_freshness (
    source          VARCHAR(30) PRIMARY KEY,
    last_timestamp  TIMESTAMPTZ,
    last_checked_at TIMESTAMPTZ DEFAULT NOW(),
    row_count       INTEGER
);

-- Model metadata — tracks persisted model artifacts
CREATE TABLE IF NOT EXISTS model_metadata (
    id              SERIAL PRIMARY KEY,
    region          VARCHAR(10) NOT NULL,
    model_name      VARCHAR(30) NOT NULL,
    artifact_path   TEXT NOT NULL,
    trained_at      TIMESTAMPTZ NOT NULL,
    feature_hash    TEXT,
    metrics         JSONB,
    weights         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_metadata_region ON model_metadata (region, created_at DESC);
