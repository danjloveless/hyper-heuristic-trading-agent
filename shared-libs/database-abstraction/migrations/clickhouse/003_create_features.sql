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