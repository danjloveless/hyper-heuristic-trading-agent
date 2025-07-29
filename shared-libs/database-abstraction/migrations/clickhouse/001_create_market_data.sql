CREATE TABLE IF NOT EXISTS market_data (
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    open Decimal64(8),
    high Decimal64(8),
    low Decimal64(8),
    close Decimal64(8),
    volume UInt64,
    adjusted_close Decimal64(8),
    ingestion_timestamp DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(ingestion_timestamp)
PARTITION BY (symbol, toYYYYMM(timestamp))
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;

-- Create materialized view for daily aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_daily
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, date)
AS SELECT
    symbol,
    toDate(timestamp) as date,
    argMinState(open, timestamp) as open,
    maxState(high) as high,
    minState(low) as low,
    argMaxState(close, timestamp) as close,
    sumState(volume) as volume
FROM market_data
GROUP BY symbol, toDate(timestamp);