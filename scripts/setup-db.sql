-- ClickHouse Database Setup for QuantumTrade AI
-- This script creates all necessary tables for the financial forecasting system

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS quantumtrade;

USE quantumtrade;

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    open Decimal64(4),
    high Decimal64(4),
    low Decimal64(4),
    close Decimal64(4),
    volume UInt64,
    adjusted_close Decimal64(4)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;

-- Sentiment data table
CREATE TABLE IF NOT EXISTS sentiment_data (
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    source LowCardinality(String),
    sentiment_score Float32,
    confidence Float32,
    mention_count UInt32,
    raw_data String,
    hash UInt64
) ENGINE = ReplacingMergeTree(hash)
PARTITION BY (source, toYYYYMM(timestamp))
ORDER BY (symbol, source, timestamp)
SETTINGS index_granularity = 8192;

-- Features table
CREATE TABLE IF NOT EXISTS features (
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    feature_name LowCardinality(String),
    feature_value Float64,
    feature_metadata String,
    feature_version UInt16
) ENGINE = MergeTree()
PARTITION BY (feature_name, toYYYYMM(timestamp))
ORDER BY (symbol, feature_name, timestamp)
SETTINGS index_granularity = 8192;

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    prediction_id String,
    model_version LowCardinality(String),
    strategy_name LowCardinality(String),
    prediction_horizon UInt16,
    predicted_price Decimal64(4),
    confidence Float32,
    regime LowCardinality(String),
    features_used Array(String),
    explanation_id UUID,
    model_latency_ms UInt16,
    strategy_parameters Map(String, Float64),
    heuristic_confidence Float32
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), model_version)
ORDER BY (symbol, timestamp, model_version)
SETTINGS index_granularity = 8192;

-- Prediction outcomes table
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    prediction_id String,
    symbol LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    actual_price Decimal64(4),
    predicted_price Decimal64(4),
    profit_loss Float64,
    accuracy_score Float32,
    strategy_name LowCardinality(String),
    regime LowCardinality(String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (prediction_id, timestamp)
SETTINGS index_granularity = 8192;

-- Strategy performance table
CREATE TABLE IF NOT EXISTS strategy_performance (
    id UUID DEFAULT generateUUIDv4(),
    strategy_name LowCardinality(String),
    regime LowCardinality(String),
    timestamp DateTime64(3, 'UTC'),
    prediction_accuracy Float32,
    profit_loss Float64,
    sharpe_ratio Float32,
    max_drawdown Float32,
    trade_count UInt32,
    win_rate Float32,
    parameters Map(String, Float64),
    market_conditions Map(String, Float64)
) ENGINE = MergeTree()
PARTITION BY (strategy_name, toYYYYMM(timestamp))
ORDER BY (strategy_name, regime, timestamp)
SETTINGS index_granularity = 8192;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_id ON predictions(prediction_id);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp ON features(symbol, timestamp);

-- Create materialized views for optimized queries
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_1min
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
AS SELECT
    symbol,
    toStartOfMinute(timestamp) as timestamp,
    argMin(open, timestamp) as open,
    max(high) as high,
    min(low) as low,
    argMax(close, timestamp) as close,
    sum(volume) as volume
FROM market_data
GROUP BY symbol, toStartOfMinute(timestamp);

-- Features wide view for ML training
CREATE MATERIALIZED VIEW IF NOT EXISTS features_wide
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
AS SELECT
    symbol,
    timestamp,
    argMax(feature_value, feature_version) as rsi_14,
    argMax(feature_value, feature_version) as macd_signal,
    argMax(feature_value, feature_version) as bollinger_upper,
    argMax(feature_value, feature_version) as bollinger_lower,
    argMax(feature_value, feature_version) as volume_sma_ratio,
    argMax(feature_value, feature_version) as reddit_sentiment,
    argMax(feature_value, feature_version) as news_sentiment
FROM features
WHERE feature_name IN ('rsi_14', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 'volume_sma_ratio', 'reddit_sentiment', 'news_sentiment')
GROUP BY symbol, timestamp; 