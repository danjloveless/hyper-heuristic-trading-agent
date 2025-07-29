pub struct Migration {
    pub version: u32,
    pub name: String,
    pub sql: String,
}

pub fn get_migrations() -> Vec<Migration> {
    vec![
        Migration {
            version: 1,
            name: "create_market_data".to_string(),
            sql: include_str!("../../migrations/clickhouse/001_create_market_data.sql").to_string(),
        },
        Migration {
            version: 2,
            name: "create_sentiment_data".to_string(),
            sql: include_str!("../../migrations/clickhouse/002_create_sentiment_data.sql").to_string(),
        },
        Migration {
            version: 3,
            name: "create_features".to_string(),
            sql: include_str!("../../migrations/clickhouse/003_create_features.sql").to_string(),
        },
        Migration {
            version: 4,
            name: "create_predictions".to_string(),
            sql: include_str!("../../migrations/clickhouse/004_create_predictions.sql").to_string(),
        },
        Migration {
            version: 5,
            name: "create_prediction_outcomes".to_string(),
            sql: include_str!("../../migrations/clickhouse/005_create_prediction_outcomes.sql").to_string(),
        },
        Migration {
            version: 6,
            name: "create_strategy_performance".to_string(),
            sql: include_str!("../../migrations/clickhouse/006_create_strategy_performance.sql").to_string(),
        },
    ]
}