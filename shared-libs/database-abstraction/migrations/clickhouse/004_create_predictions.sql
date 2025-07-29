CREATE TABLE IF NOT EXISTS predictions (
    prediction_id String,
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    predicted_price Decimal64(8),
    confidence Float32,
    horizon_minutes UInt16,
    strategy_name LowCardinality(String),
    model_version String,
    explanation Nullable(String), -- JSON string
    ingestion_timestamp DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(ingestion_timestamp)
PARTITION BY (symbol, toYYYYMM(timestamp))
ORDER BY (symbol, timestamp, prediction_id)
SETTINGS index_granularity = 8192;

-- Create index for faster prediction ID lookups
ALTER TABLE predictions ADD INDEX idx_prediction_id prediction_id TYPE bloom_filter(0.01) GRANULARITY 1;