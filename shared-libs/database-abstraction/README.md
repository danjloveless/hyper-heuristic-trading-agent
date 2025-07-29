# QuantumTrade AI Database Abstraction Layer

A high-performance database abstraction layer for the QuantumTrade AI system, providing unified access to ClickHouse (time-series data) and Redis (caching).

## 🚀 Quick Start

### 1. Start the Databases
```powershell
.\scripts\start-databases.ps1
```

### 2. Test the System
```powershell
cargo run --example basic_usage
```

## 📋 Prerequisites
- **Docker Desktop** - For running ClickHouse and Redis
- **Rust** - For building the database abstraction layer

## 🏗️ Architecture

### Database Layout
```
┌─────────────────┐    ┌─────────────────┐
│   ClickHouse    │    │     Redis       │
│  (Time-Series)  │    │    (Caching)    │
├─────────────────┤    ├─────────────────┤
│ • Market Data   │    │ • Cache Keys    │
│ • Sentiment     │    │ • Session Data  │
│ • Features      │    │ • Rate Limits   │
│ • Predictions   │    │ • Real-time     │
│ • Performance   │    │   Data          │
└─────────────────┘    └─────────────────┘
```

### Key Components
- **`DatabaseManager`** - Central orchestrator for both databases
- **`ClickHouseClient`** - Handles time-series data operations
- **`RedisClient`** - Manages caching and real-time data
- **`DatabaseClient`** - Unified trait for database operations
- **`CacheClient`** - Specialized trait for caching operations

## 📊 Database Schema

### ClickHouse Tables
1. **`market_data`** - OHLCV market data with daily aggregates
2. **`sentiment_data`** - News sentiment analysis with text search
3. **`features`** - ML feature sets with versioning
4. **`technical_indicators`** - Technical analysis indicators
5. **`predictions`** - Model predictions with explanations
6. **`prediction_outcomes`** - Prediction accuracy tracking
7. **`strategy_performance`** - Strategy performance metrics

### Redis Keys
- **`quantumtrade:market_data:{symbol}:{timestamp}`** - Cached market data
- **`quantumtrade:features:{symbol}:{timestamp}`** - Cached features
- **`quantumtrade:prediction:{prediction_id}`** - Cached predictions
- **`quantumtrade:realtime:price:{symbol}`** - Real-time prices

## 💻 Usage Example

```rust
use database_abstraction::{DatabaseManager, DatabaseConfig, DatabaseClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize database manager
    let config = DatabaseConfig::default();
    let db_manager = DatabaseManager::new(config).await?;
    
    // Get individual clients
    let clickhouse = db_manager.clickhouse();
    let redis = db_manager.redis();
    
    // Insert market data
    let market_data = vec![
        MarketData {
            symbol: "AAPL".to_string(),
            timestamp: Utc::now(),
            open: Decimal::from_f64(150.0).unwrap(),
            high: Decimal::from_f64(155.0).unwrap(),
            low: Decimal::from_f64(149.0).unwrap(),
            close: Decimal::from_f64(152.0).unwrap(),
            volume: 1000000,
            adjusted_close: Decimal::from_f64(152.0).unwrap(),
        }
    ];
    
    clickhouse.insert_market_data(&market_data).await?;
    
    // Cache data in Redis
    redis.cache_set("test_key", &market_data[0], Some(Duration::hours(1))).await?;
    
    Ok(())
}
```

## 🛠️ Management

### Database Management
```powershell
# Start databases (default)
.\scripts\start-databases.ps1

# Check status
.\scripts\start-databases.ps1 -Status

# View logs
.\scripts\start-databases.ps1 -Logs

# Restart databases
.\scripts\start-databases.ps1 -Restart

# Stop databases
.\scripts\start-databases.ps1 -Stop
```

### Testing
```powershell
# Test basic functionality
cargo run --example basic_usage

# Run tests
cargo test
```

## 🔍 Troubleshooting

### Common Issues
1. **Connection Refused**
   ```powershell
   # Check if Docker is running
   docker version
   
   # Start databases
   .\scripts\start-databases.ps1
   ```

2. **Migration Errors**
   ```powershell
   # Restart databases
   .\scripts\start-databases.ps1 -Restart
   ```

## 📈 Performance

- **ClickHouse**: Optimized for time-series queries with partitioning by symbol and month
- **Redis**: Configured with 256MB memory limit and LRU eviction policy
- **Connection Pooling**: Automatic connection management with configurable pool sizes

## 🔐 Security

- **Authentication**: Configurable username/password for ClickHouse
- **Network Security**: Services run on localhost by default
- **Data Encryption**: TLS support for production deployments

## 📄 License

This project is licensed under the MIT License.