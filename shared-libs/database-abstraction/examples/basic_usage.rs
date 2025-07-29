use database_abstraction::{DatabaseManager, DatabaseConfig, DatabaseClient};
use shared_types::{MarketData, SentimentData, FeatureSet};
use chrono::Utc;
use rust_decimal::prelude::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create database configuration
    let config = DatabaseConfig::default();
    
    // Create database manager
    let db_manager = DatabaseManager::new(config).await?;
    
    println!("Database manager created successfully!");
    
    // Get individual clients
    let clickhouse = db_manager.clickhouse();
    let redis = db_manager.redis();
    
    // Run health checks
    let clickhouse_health = clickhouse.health_check().await?;
    let redis_health = redis.health_check().await?;
    
    println!("ClickHouse health: {:?}", clickhouse_health.status);
    println!("Redis health: {:?}", redis_health.status);
    
    // Example: Create some test data
    let market_data = MarketData {
        symbol: "AAPL".to_string(),
        timestamp: Utc::now(),
        open: Decimal::from_f64(150.0).unwrap(),
        high: Decimal::from_f64(155.0).unwrap(),
        low: Decimal::from_f64(149.0).unwrap(),
        close: Decimal::from_f64(152.0).unwrap(),
        volume: 1000000,
        adjusted_close: Decimal::from_f64(152.0).unwrap(),
    };
    
    let sentiment_data = SentimentData {
        article_id: "test_article_1".to_string(),
        symbol: "AAPL".to_string(),
        timestamp: Utc::now(),
        title: "Apple Reports Strong Q4 Earnings".to_string(),
        content: "Apple Inc. reported better-than-expected quarterly earnings...".to_string(),
        source: "Reuters".to_string(),
        sentiment_score: 0.8,
        confidence: 0.9,
        entities: HashMap::new(),
        relevance_score: 0.95,
        market_impact: Some(0.7),
    };
    
    let features = FeatureSet {
        symbol: "AAPL".to_string(),
        timestamp: Utc::now(),
        features: HashMap::from([
            ("price_momentum".to_string(), 0.65),
            ("volume_trend".to_string(), 0.72),
            ("volatility".to_string(), 0.45),
        ]),
        feature_metadata: HashMap::from([
            ("version".to_string(), "1.0".to_string()),
            ("model".to_string(), "technical_indicators".to_string()),
        ]),
        feature_version: 1,
    };
    
    println!("Test data created successfully!");
    println!("Market data: {:?}", market_data.symbol);
    println!("Sentiment data: {:?}", sentiment_data.article_id);
    println!("Features: {:?}", features.symbol);
    
    // Note: In a real application, you would insert this data into the databases
    // For this example, we'll just demonstrate the structure
    
    println!("Database abstraction layer is working correctly!");
    
    Ok(())
}