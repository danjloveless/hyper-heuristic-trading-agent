use database_abstraction::{DatabaseManager, DatabaseConfig};
use shared_types::*;
use chrono::Utc;
use rust_decimal::Decimal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::init();

    // Load configuration (in production, use proper config management)
    let config = DatabaseConfig::default();
    
    // Create database manager
    let db = DatabaseManager::new(config).await?;
    
    // Run migrations
    db.run_migrations().await?;
    
    // Test market data insertion
    let market_data = vec![
        MarketData {
            symbol: "AAPL".to_string(),
            timestamp: Utc::now(),
            open: Decimal::from_f64(150.0).unwrap(),
            high: Decimal::from_f64(152.0).unwrap(),
            low: Decimal::from_f64(149.0).unwrap(),
            close: Decimal::from_f64(151.0).unwrap(),
            volume: 1000000,
            adjusted_close: Decimal::from_f64(151.0).unwrap(),
        }
    ];
    
    db.clickhouse().insert_market_data(&market_data).await?;
    println!("✅ Market data inserted successfully");
    
    // Test cache operations
    db.redis().cache_set(
        "test_key", 
        &"test_value".to_string(), 
        Some(chrono::Duration::minutes(5))
    ).await?;
    
    let cached_value: Option<String> = db.redis().cache_get("test_key").await?;
    println!("✅ Cached value: {:?}", cached_value);
    
    // Test health checks
    let health = db.health_check().await?;
    println!("✅ System health: {:?}", health);
    
    Ok(())
}