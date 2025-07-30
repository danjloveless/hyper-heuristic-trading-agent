CREATE TABLE IF NOT EXISTS sentiment_data (
    article_id String,
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    title String,
    content String,
    source LowCardinality(String),
    sentiment_score Float32,
    confidence Float32,
    entities String, -- JSON string
    relevance_score Float32,
    market_impact Nullable(Float32),
    ingestion_timestamp DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(ingestion_timestamp)
PARTITION BY (symbol, toYYYYMM(timestamp))
ORDER BY (symbol, timestamp, article_id)
SETTINGS index_granularity = 8192;