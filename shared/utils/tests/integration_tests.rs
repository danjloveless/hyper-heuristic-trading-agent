// shared/utils/tests/integration_tests.rs
#[cfg(test)]
mod integration_tests {
    use super::*;
    use chrono::Utc;
    use shared_types::{MarketData, SentimentData};
    use shared_utils::{ClickHouseClient, ClickHouseConfig};
    use std::collections::HashMap;
    use tokio_test;

    async fn setup_test_client() -> ClickHouseClient {
        dotenvy::dotenv().ok();
        
        let config = ClickHouseConfig::from_env()
            .expect("Failed to load ClickHouse config from environment");
            
        let client = ClickHouseClient::new(config)
            .await
            .expect("Failed to create ClickHouse client");
            
        // Initialize schema
        client.initialize_schema()
            .await
            .expect("Failed to initialize schema");
            
        client
    }

    #[tokio::test]
    async fn test_market_data_operations() {
        let client = setup_test_client().await;
        
        // Create test data
        let test_data = vec![
            MarketData {
                symbol: "TEST".to_string(),
                timestamp: Utc::now(),
                open: 100.0,
                high: 105.0,
                low: 99.0,
                close: 104.0,
                volume: 1000000,
                adjusted_close: 104.0,
            },
            MarketData {
                symbol: "TEST".to_string(),
                timestamp: Utc::now() - chrono::Duration::minutes(1),
                open: 104.0,
                high: 106.0,
                low: 103.0,
                close: 105.0,
                volume: 1200000,
                adjusted_close: 105.0,
            },
        ];
        
        // Insert data
        let rows_inserted = client.insert_market_data_batch(test_data.clone())
            .await
            .expect("Failed to insert market data");
            
        assert_eq!(rows_inserted, 2);
        
        // Query data back
        let retrieved_data = client.get_latest_market_data("TEST", 10)
            .await
            .expect("Failed to retrieve market data");
            
        assert_eq!(retrieved_data.len(), 2);
        assert_eq!(retrieved_data[0].symbol, "TEST");
        
        println!("✅ Market data operations test passed");
    }

    #[tokio::test]
    async fn test_sentiment_data_operations() {
        let client = setup_test_client().await;
        
        let test_sentiment = vec![
            SentimentData {
                symbol: "TEST".to_string(),
                timestamp: Utc::now(),
                source: "reddit".to_string(),
                sentiment_score: 0.8,
                confidence: 0.9,
                mention_count: 150,
                raw_data: r#"{"posts": 150, "avg_sentiment": 0.8}"#.to_string(),
            }
        ];
        
        let rows_inserted = client.insert_sentiment_data_batch(test_sentiment)
            .await
            .expect("Failed to insert sentiment data");
            
        assert_eq!(rows_inserted, 1);
        
        let retrieved_sentiment = client.get_latest_sentiment_data("TEST", Some("reddit"), 10)
            .await
            .expect("Failed to retrieve sentiment data");
            
        assert_eq!(retrieved_sentiment.len(), 1);
        assert_eq!(retrieved_sentiment[0].sentiment_score, 0.8);
        
        println!("✅ Sentiment data operations test passed");
    }

    #[tokio::test]
    async fn test_feature_operations() {
        let client = setup_test_client().await;
        
        let mut features = HashMap::new();
        features.insert("rsi_14".to_string(), 65.5);
        features.insert("macd_signal".to_string(), 0.8);
        
        let mut metadata = HashMap::new();
        metadata.insert("rsi_14".to_string(), r#"{"period": 14}"#.to_string());
        
        let feature_set = shared_types::FeatureSet {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            features,
            feature_metadata: metadata,
            feature_version: 1,
        };
        
        let rows_inserted = client.insert_features_batch(vec![feature_set])
            .await
            .expect("Failed to insert features");
            
        assert!(rows_inserted > 0);
        
        let retrieved_features = client.get_latest_features("TEST")
            .await
            .expect("Failed to retrieve features");
            
        assert!(retrieved_features.is_some());
        let features = retrieved_features.unwrap();
        assert_eq!(features.features.len(), 2);
        assert!(features.features.contains_key("rsi_14"));
        
        println!("✅ Feature operations test passed");
    }

    #[tokio::test]
    async fn test_connection_and_schema() {
        let client = setup_test_client().await;
        
        // Test connection
        client.test_connection()
            .await
            .expect("Connection test failed");
            
        println!("✅ Connection test passed");
    }
}