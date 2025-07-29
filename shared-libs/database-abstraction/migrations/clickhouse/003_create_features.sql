CREATE TABLE IF NOT EXISTS features (
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    features String, -- JSON string of feature values
    feature_metadata String, -- JSON string of metadata
    feature_version UInt16,
    ingestion_timestamp DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(ingestion_timestamp)
PARTITION BY (symbol, toYYYYMM(timestamp))
ORDER BY (symbol, timestamp, feature_version)
SETTINGS index_granularity = 8192;

-- Technical indicators table for structured storage
CREATE TABLE IF NOT EXISTS technical_indicators (
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    sma_20 Nullable(Float64),
    ema_12 Nullable(Float64),
    ema_26 Nullable(Float64),
    macd Nullable(Float64),
    macd_signal Nullable(Float64),
    rsi Nullable(Float64),
    bollinger_upper Nullable(Float64),
    bollinger_lower Nullable(Float64),
    volume_sma_20 Nullable(Float64),
    ingestion_timestamp DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(ingestion_timestamp)
PARTITION BY (symbol, toYYYYMM(timestamp))
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;