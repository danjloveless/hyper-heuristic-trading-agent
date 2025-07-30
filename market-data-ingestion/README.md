# Market Data Ingestion Service

A high-performance, production-ready market data ingestion service that collects financial market data from Alpha Vantage API and processes it through a standardized data pipeline.

## 🏗️ Architecture Overview

This service follows the **core infrastructure compliance** pattern with proper dependency injection and standardized interfaces:

```
┌─────────────────────────────────────────────────────────────┐
│                    Market Data Ingestion                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Service   │  │  Collector  │  │  Processor  │        │
│  │             │  │             │  │             │        │
│  │ • Health    │  │ • API Calls │  │ • Validation│        │
│  │ • Error     │  │ • Rate      │  │ • Quality   │        │
│  │   Handling  │  │   Limiting  │  │   Checks    │        │
│  │ • Metrics   │  │ • Parsing   │  │ • Storage   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Core Infrastructure                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Config    │  │  Database   │  │   Error     │        │
│  │  Provider   │  │  Manager    │  │  Handler    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐                                          │
│  │ Monitoring  │                                          │
│  │  Provider   │                                          │
│  └─────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 🔧 **Core Infrastructure Integration**
- **Dependency Injection**: All core infrastructure components properly injected
- **Standardized Interfaces**: Implements `HealthCheckable`, `ErrorHandler`, etc.
- **Configuration Management**: Uses `ConfigurationProvider` for all config
- **Database Integration**: Uses `DatabaseManager` for ClickHouse and Redis
- **Monitoring Integration**: Records metrics through `MonitoringProvider`

### 📊 **Data Processing Pipeline**
- **Alpha Vantage Integration**: Collects real-time and historical market data
- **Rate Limiting**: Respects API rate limits with intelligent backoff
- **Data Validation**: Comprehensive quality checks and validation
- **Deduplication**: Removes duplicate data points automatically
- **Caching Strategy**: Intelligent Redis caching for performance

### 🛡️ **Error Handling & Resilience**
- **Retry Logic**: Automatic retry with exponential backoff
- **Fallback Mechanisms**: Cache fallback when database fails
- **Graceful Degradation**: Service continues operating during partial failures
- **Error Classification**: Intelligent error categorization and handling

### 📈 **Monitoring & Observability**
- **Health Checks**: Comprehensive component health monitoring
- **Metrics Collection**: Detailed performance and business metrics
- **Structured Logging**: Context-aware logging with tracing
- **Quality Scoring**: Data quality assessment and reporting

## 🚀 Quick Start

### Prerequisites

1. **Core Infrastructure**: Ensure core infrastructure components are available
2. **Database**: ClickHouse and Redis instances running
3. **API Key**: Alpha Vantage API key configured

### Basic Usage

```rust
use market_data_ingestion::*;
use core_traits::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create core infrastructure components
    let config_provider = Arc::new(create_config_provider());
    let database_manager = Arc::new(create_database_manager().await?);
    let error_handler = Arc::new(create_error_handler());
    let monitoring = Arc::new(create_monitoring_provider());
    
    // 2. Create service with dependency injection
    let service = market_data_ingestion::create_service(
        config_provider,
        database_manager,
        error_handler,
        monitoring,
    ).await?;
    
    // 3. Start the service
    service.start().await?;
    
    // 4. Collect market data
    let result = service.collect_symbol_data("AAPL", Interval::FiveMin).await?;
    
    println!("Collected {} data points with quality score {}", 
             result.processed_count, 
             result.quality_score.unwrap_or(0));
    
    Ok(())
}
```

### Configuration

The service uses the `ConfigurationProvider` for all configuration:

```rust
// Alpha Vantage configuration
let alpha_vantage_config = config_provider
    .get_config_section::<AlphaVantageConfig>("alpha_vantage")
    .await?;

// Rate limits configuration  
let rate_limits_config = config_provider
    .get_config_section::<RateLimitsConfig>("rate_limits")
    .await?;

// Collection configuration
let collection_config = config_provider
    .get_config_section::<CollectionConfig>("collection")
    .await?;
```

## 📋 API Reference

### Core Service

#### `MarketDataIngestionService`

The main service that orchestrates data collection and processing.

```rust
impl MarketDataIngestionService {
    /// Create a new service with dependency injection
    pub async fn new(
        config_provider: Arc<dyn ConfigurationProvider>,
        database_manager: Arc<DatabaseManager>,
        error_handler: Arc<dyn ErrorHandler>,
        monitoring: Arc<dyn MonitoringProvider>,
    ) -> ServiceResult<Self>
    
    /// Start the service and background tasks
    pub async fn start(&self) -> ServiceResult<()>
    
    /// Collect market data for a specific symbol
    pub async fn collect_symbol_data(
        &self, 
        symbol: &str, 
        interval: Interval
    ) -> ServiceResult<CollectionResult>
}
```

#### `Interval`

Supported data collection intervals:

```rust
pub enum Interval {
    OneMin,      // 1 minute
    FiveMin,     // 5 minutes  
    FifteenMin,  // 15 minutes
    ThirtyMin,   // 30 minutes
    SixtyMin,    // 1 hour
    Daily,       // Daily
}
```

#### `CollectionResult`

Result of a data collection operation:

```rust
pub struct CollectionResult {
    pub symbol: String,
    pub interval: Interval,
    pub collected_count: usize,    // Raw data points collected
    pub processed_count: usize,    // Data points after processing
    pub cached_count: usize,       // Data points from cache
    pub collection_time: DateTime<Utc>,
    pub processing_duration_ms: u64,
    pub batch_id: String,
    pub source: DataSource,
    pub quality_score: Option<u8>, // 0-100 quality score
}
```

### Health Checks

The service implements the `HealthCheckable` trait:

```rust
// Check overall service health
let health = service.health_check().await;
println!("Service status: {:?}", health.status);

// Check if service is ready to handle requests
let readiness = service.ready_check().await;
match readiness {
    ReadinessStatus::Ready => println!("Service is ready"),
    ReadinessStatus::NotReady { reason } => println!("Service not ready: {}", reason),
}
```

## 🔧 Configuration

### Alpha Vantage Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaVantageConfig {
    pub base_url: String,           // API base URL
    pub timeout_seconds: u64,       // Request timeout
    pub max_retries: u32,           // Maximum retry attempts
    pub default_output_size: String, // "compact" or "full"
}
```

### Rate Limits Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitsConfig {
    pub calls_per_minute: u32,  // API calls per minute
    pub calls_per_day: u32,     // API calls per day
    pub is_premium: bool,       // Premium API tier
    pub burst_allowance: u32,   // Burst allowance
}
```

### Data Quality Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityConfig {
    pub min_quality_score: u8,              // Minimum acceptable quality (0-100)
    pub enable_validation: bool,            // Enable data validation
    pub max_price_deviation_percent: f64,   // Max price change allowed
    pub min_volume_threshold: u64,          // Minimum volume threshold
    pub enable_deduplication: bool,         // Enable duplicate removal
}
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test integration_tests

# Run with logging
RUST_LOG=debug cargo test
```

### Test Examples

```rust
#[tokio::test]
async fn test_data_collection() {
    // Setup mock infrastructure
    let service = create_test_service().await;
    
    // Test data collection
    let result = service.collect_symbol_data("AAPL", Interval::FiveMin).await
        .expect("Collection should succeed");
    
    assert!(result.processed_count > 0);
    assert!(result.quality_score.unwrap_or(0) > 70);
}

#[tokio::test]
async fn test_error_handling() {
    // Test error scenarios
    let service = create_test_service_with_errors().await;
    
    let result = service.collect_symbol_data("INVALID", Interval::FiveMin).await;
    
    // Should handle errors gracefully
    assert!(result.is_err());
}
```

## 📊 Monitoring & Metrics

### Key Metrics

The service records comprehensive metrics:

- **`data_collection_attempts`**: Number of collection attempts
- **`data_collection_duration`**: Time taken for data collection
- **`data_points_collected`**: Number of data points collected
- **`data_points_stored`**: Number of data points stored in database
- **`data_quality_failures`**: Number of data quality failures
- **`alpha_vantage_api_errors`**: API error counts
- **`average_data_quality`**: Average quality score of collected data

### Health Checks

The service provides detailed health checks for:

- **ClickHouse Database**: Connection and query health
- **Redis Cache**: Connection and operation health  
- **Alpha Vantage API**: API connectivity and response health

### Logging

Structured logging with context:

```rust
// Info level for normal operations
info!("Successfully collected {} data points for {}", count, symbol);

// Warn level for recoverable issues
warn!("Data quality check failed for {} (score: {})", symbol, score);

// Error level for failures
error!("Failed to collect data for {}: {:?}", symbol, error);
```

## 🔒 Security Considerations

### API Key Management

- Store Alpha Vantage API key in secure configuration provider
- Never log API keys
- Rotate keys regularly
- Use environment variables for local development

### Data Validation

- Validate all incoming data before processing
- Sanitize data before storage
- Implement rate limiting to prevent abuse
- Monitor for suspicious activity patterns

## 🚀 Deployment

### Environment Variables

```bash
# Database configuration
CLICKHOUSE_URL=http://localhost:8123
CLICKHOUSE_DATABASE=quantumtrade
REDIS_URL=redis://localhost:6379

# API configuration
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Logging
RUST_LOG=info
```

### Docker Deployment

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
COPY --from=builder /app/target/release/market-data-ingestion /usr/local/bin/
CMD ["market-data-ingestion"]
```

### Production Considerations

1. **Resource Limits**: Set appropriate CPU and memory limits
2. **Health Checks**: Configure health check endpoints
3. **Monitoring**: Set up metrics collection and alerting
4. **Backup**: Ensure database backups are configured
5. **Scaling**: Consider horizontal scaling for high throughput

## 🤝 Contributing

### Development Setup

1. **Clone the repository**
2. **Install dependencies**: `cargo build`
3. **Set up databases**: Start ClickHouse and Redis
4. **Configure environment**: Set required environment variables
5. **Run tests**: `cargo test`

### Code Standards

- Follow Rust coding conventions
- Implement comprehensive tests
- Use proper error handling
- Document public APIs
- Follow the core infrastructure patterns

### Testing Guidelines

- Write unit tests for all components
- Include integration tests for end-to-end scenarios
- Test error conditions and edge cases
- Use mocks for external dependencies
- Maintain high test coverage

## 📚 Examples

See the `examples/` directory for complete usage examples:

- **`complete_integration.rs`**: Full integration example with mock infrastructure
- **Error handling examples**: Various error scenarios and recovery patterns
- **Configuration examples**: Different configuration setups

## 🔗 Related Documentation

- [Core Infrastructure Documentation](../docs/)
- [Database Abstraction Layer](../shared-libs/database-abstraction/)
- [Shared Types](../shared-libs/shared-types/)
- [Core Traits](../shared-libs/core-traits/)

## 📄 License

This project is part of the QuantumTrade AI system and follows the same licensing terms. 