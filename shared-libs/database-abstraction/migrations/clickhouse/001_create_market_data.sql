CREATE TABLE IF NOT EXISTS market_data (
    symbol String,
    timestamp DateTime,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume UInt64,
    adjusted_close Float64,
    ingestion_timestamp DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(ingestion_timestamp)
PARTITION BY (symbol, toYYYYMM(timestamp))
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;