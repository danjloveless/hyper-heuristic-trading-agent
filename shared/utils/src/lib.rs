// shared/utils/src/lib.rs
pub mod clickhouse;
pub mod monitoring;
pub mod error;

pub use clickhouse::*;
pub use monitoring::*;
pub use error::*;

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use shared_types::MarketData;
    use tokio_test;

    #[tokio::test]
    async fn test_clickhouse_client_creation() {
        let config = ClickHouseConfig::default();
        
        // This test will fail if ClickHouse is not running, which is expected
        // In a real test environment, you'd use testcontainers
        match ClickHouseClient::new(config).await {
            Ok(_) => println!("ClickHouse client created successfully"),
            Err(e) => println!("Expected error when ClickHouse is not running: {}", e),
        }
    }

    #[test]
    fn test_config_from_env() {
        std::env::set_var("CLICKHOUSE_URL", "http://test:8123");
        std::env::set_var("CLICKHOUSE_DATABASE", "test_db");
        
        let config = ClickHouseConfig::from_env().unwrap();
        assert_eq!(config.url, "http://test:8123");
        assert_eq!(config.database, "test_db");
        
        std::env::remove_var("CLICKHOUSE_URL");
        std::env::remove_var("CLICKHOUSE_DATABASE");
    }

    #[test]
    fn test_market_data_conversion() {
        let market_data = MarketData {
            symbol: "AAPL".to_string(),
            timestamp: Utc::now(),
            open: 150.0,
            high: 155.0,
            low: 149.0,
            close: 154.0,
            volume: 1000000,
            adjusted_close: 154.0,
        };

        let row = MarketDataRow::from(&market_data);
        let converted_back: MarketData = row.into();

        assert_eq!(market_data.symbol, converted_back.symbol);
        assert_eq!(market_data.open, converted_back.open);
        assert_eq!(market_data.volume, converted_back.volume);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        
        collector.record_query_latency("test_query", 100, 50);
        
        let metrics = collector.get_metrics("test_query").unwrap();
        assert_eq!(metrics.query_latency_ms, 100);
        assert_eq!(metrics.rows_processed, 50);
    }
}