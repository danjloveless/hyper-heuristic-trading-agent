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

-- Create materialized view for strategy rankings
CREATE MATERIALIZED VIEW IF NOT EXISTS strategy_rankings
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (strategy_name, timestamp)
AS SELECT
    strategy_name,
    toStartOfMonth(timestamp) as timestamp,
    avgState(accuracy_rate) as avg_accuracy,
    avgState(sharpe_ratio) as avg_sharpe,
    maxState(profit_loss) as max_profit,
    minState(max_drawdown) as min_drawdown
FROM strategy_performance
GROUP BY strategy_name, toStartOfMonth(timestamp);