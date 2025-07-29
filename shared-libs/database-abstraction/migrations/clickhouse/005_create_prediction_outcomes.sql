CREATE TABLE IF NOT EXISTS prediction_outcomes (
    prediction_id String,
    actual_price Decimal64(8),
    accuracy_score Float32,
    directional_accuracy Bool,
    outcome_timestamp DateTime64(3, 'UTC'),
    ingestion_timestamp DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(ingestion_timestamp)
PARTITION BY toYYYYMM(outcome_timestamp)
ORDER BY (prediction_id, outcome_timestamp)
SETTINGS index_granularity = 8192;