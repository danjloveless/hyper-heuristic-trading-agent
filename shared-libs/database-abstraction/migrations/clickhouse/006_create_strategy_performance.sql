CREATE TABLE IF NOT EXISTS strategy_performance (
    strategy_name LowCardinality(String),
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    total_predictions UInt32,
    accuracy_rate Float32,
    avg_confidence Float32,
    profit_loss Decimal64(8),
    sharpe_ratio Float32,
    max_drawdown Float32,
    ingestion_timestamp DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(ingestion_timestamp)
PARTITION BY (strategy_name, toYYYYMM(timestamp))
ORDER BY (strategy_name, symbol, timestamp)
SETTINGS index_granularity = 8192;