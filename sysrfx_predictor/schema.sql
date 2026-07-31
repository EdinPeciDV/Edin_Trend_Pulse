-- sysrfx_predictor/schema.sql
-- ===================================================================
-- Run this once against your Supabase project (SQL Editor, or via
-- `psql "$SUPABASE_DB_URL" -f schema.sql`) before the first run.
--
-- WHY THIS ISN'T AUTO-RUN VIA THE SUPABASE REST CLIENT: the standard
-- Supabase Python client (`supabase-py`) talks to PostgREST, which
-- exposes your tables as a REST API — it has no endpoint for arbitrary
-- DDL (CREATE TABLE), by design (that's a security boundary, not a
-- missing feature). sysrfx_predictor.py *will* auto-create these tables
-- for you on startup, but only if you set SUPABASE_DB_URL (a direct
-- Postgres connection string — Supabase project settings -> Database ->
-- Connection string) so it can run this same DDL over a real Postgres
-- connection (via psycopg2). Without SUPABASE_DB_URL, run this file by
-- hand once; after that, all normal reads/writes go through the REST
-- client as usual.
-- ===================================================================

-- Table: bars — OHLCV + the full engineered feature vector per bar.
CREATE TABLE IF NOT EXISTS bars (
    id BIGSERIAL PRIMARY KEY,
    pair TEXT NOT NULL,
    timeframe TEXT NOT NULL DEFAULT '1h',
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    features JSONB,
    UNIQUE (pair, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_bars_pair_ts ON bars (pair, timeframe, timestamp DESC);

-- Table: predictions — every forecast the script has ever emitted.
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    pair TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    direction INTEGER,        -- 1 = UP, 0 = DOWN
    confidence DOUBLE PRECISION,
    regime TEXT,               -- 'trending' | 'ranging'
    horizon_bars INTEGER,
    actual_result INTEGER,      -- NULL until graded; 1 = correct, 0 = incorrect
    resolved_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_predictions_pair_ts ON predictions (pair, timestamp DESC);

-- Table: model_metadata — one row per (re)training run.
CREATE TABLE IF NOT EXISTS model_metadata (
    id BIGSERIAL PRIMARY KEY,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pair TEXT NOT NULL,
    regime TEXT,
    n_train_rows INTEGER,
    walk_forward_accuracy DOUBLE PRECISION,
    walk_forward_accuracy_std DOUBLE PRECISION,
    base_rate DOUBLE PRECISION,
    confidence_filtered_accuracy DOUBLE PRECISION,
    hyperparams JSONB,
    feature_importances JSONB
);
CREATE INDEX IF NOT EXISTS idx_model_metadata_pair_ts ON model_metadata (pair, trained_at DESC);
