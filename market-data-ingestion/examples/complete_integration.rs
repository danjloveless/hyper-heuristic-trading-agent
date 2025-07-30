//! Complete integration example showing how to properly wire up
//! the market data ingestion service with all core infrastructure components

use market_data_ingestion::*;
use market_data_ingestion::config_provider::CoreConfigurationProvider;
use core_traits::*;
use std::sync::Arc;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file in root directory
    dotenv::from_path("../.env").ok();
    
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("market_data_ingestion=debug".parse().unwrap())
            .add_directive("complete_integration=info".parse().unwrap()))
        .init();

    println!("🚀 Complete Market Data Ingestion Integration Example");
    println!("=====================================================");

    // Step 1: Create core infrastructure components
    let core_infrastructure = setup_core_infrastructure().await?;
    
    // Step 2: Create market data ingestion service with proper dependency injection
    let ingestion_service = market_data_ingestion::create_service(
        core_infrastructure.config_provider,
        core_infrastructure.database_manager,
        core_infrastructure.error_handler,
        core_infrastructure.monitoring,
    ).await?;
    
    // Step 3: Start the service
    ingestion_service.start().await?;
    
    // Step 4: Perform health check
    println!("\n🔍 Performing health check...");
    let health = ingestion_service.health_check().await;
    println!("Service health: {:?}", health.status);
    
    for (component, check) in &health.checks {
        println!("  {} - {:?}: {}", 
                 component, 
                 check.status, 
                 check.message.as_deref().unwrap_or("OK"));
    }
    
    // Step 5: Test data collection
    println!("\n📊 Testing data collection...");
    let test_symbols = vec!["AAPL", "GOOGL", "MSFT"];
    
    for symbol in &test_symbols {
        match ingestion_service.collect_symbol_data(symbol, Interval::FiveMin).await {
            Ok(result) => {
                println!("✅ {}: Collected {} points, Processed {} points (Quality: {})", 
                         symbol, 
                         result.collected_count,
                         result.processed_count,
                         result.quality_score.unwrap_or(0));
            }
            Err(e) => {
                println!("❌ {}: Failed - {:?}", symbol, e);
            }
        }
        
        // Small delay between requests to respect rate limits
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    
    // Step 6: Test error handling
    println!("\n🧪 Testing error handling...");
    match ingestion_service.collect_symbol_data("INVALID_SYMBOL", Interval::FiveMin).await {
        Ok(result) => {
            println!("🤔 Unexpected success with invalid symbol: {:?}", result);
        }
        Err(e) => {
            println!("✅ Error handling worked correctly: {:?}", e);
        }
    }
    
    // Step 7: Final health check
    println!("\n🏁 Final health check...");
    let final_health = ingestion_service.health_check().await;
    println!("Final service status: {:?}", final_health.status);
    println!("Service uptime: {} seconds", final_health.uptime_seconds);
    
    println!("\n✅ Integration example completed successfully!");
    
    Ok(())
}

struct CoreInfrastructure {
    config_provider: Arc<dyn ConfigurationProvider>,
    database_manager: Arc<database_abstraction::DatabaseManager>,
    error_handler: Arc<dyn ErrorHandler>,
    monitoring: Arc<dyn MonitoringProvider>,
}

async fn setup_core_infrastructure() -> Result<CoreInfrastructure, Box<dyn std::error::Error>> {
    println!("🔧 Setting up core infrastructure...");
    
    // 1. Configuration Provider (using core infrastructure)
    let config_provider = Arc::new(CoreConfigurationProvider::new().await?);
    
    // 2. Database Manager
    let db_config = database_abstraction::DatabaseConfig {
        clickhouse: database_abstraction::ClickHouseConfig {
            url: std::env::var("CLICKHOUSE_URL").unwrap_or_else(|_| "http://localhost:8123".to_string()),
            database: std::env::var("CLICKHOUSE_DATABASE").unwrap_or_else(|_| "quantumtrade".to_string()),
            username: Some("default".to_string()),
            password: None,
            connection_timeout: std::time::Duration::from_secs(30),
            query_timeout: std::time::Duration::from_secs(60),
            max_connections: 100,
            retry_attempts: 3,
        },
        redis: database_abstraction::RedisConfig {
            url: std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            pool_size: 10,
            connection_timeout: std::time::Duration::from_secs(30),
            default_ttl: std::time::Duration::from_secs(3600),
            max_connections: 100,
        },
    };
    
    let database_manager = Arc::new(database_abstraction::DatabaseManager::new(db_config).await?);
    
    // Run migrations
    println!("📦 Running database migrations...");
    database_manager.run_migrations().await?;
    
    // 3. Error Handler
    let error_handler = Arc::new(MockErrorHandler::new());
    
    // 4. Monitoring Provider
    let monitoring = Arc::new(MockMonitoringProvider::new());
    
    println!("✅ Core infrastructure setup complete");
    
    Ok(CoreInfrastructure {
        config_provider,
        database_manager,
        error_handler,
        monitoring,
    })
}

// ================================================================================================
// MOCK IMPLEMENTATIONS FOR ERROR HANDLER AND MONITORING
// ================================================================================================

struct MockErrorHandler;

impl MockErrorHandler {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ErrorHandler for MockErrorHandler {
    async fn handle_error(&self, error: &(dyn std::error::Error + Send + Sync), context: &ErrorContext) -> ErrorDecision {
        eprintln!("🚨 Error in {}: {}", context.operation, error);
        
        let error_str = error.to_string();
        if error_str.contains("rate limit") {
            ErrorDecision::Retry { delay: std::time::Duration::from_secs(60), max_attempts: 3 }
        } else if error_str.contains("network") {
            ErrorDecision::Retry { delay: std::time::Duration::from_secs(5), max_attempts: 3 }
        } else {
            ErrorDecision::Fail
        }
    }
    
    async fn classify_error(&self, error: &(dyn std::error::Error + Send + Sync)) -> ErrorClassification {
        let error_str = error.to_string();
        
        if error_str.contains("rate limit") {
            ErrorClassification {
                error_type: ErrorType::Transient,
                severity: ErrorSeverity::Medium,
                retryable: true,
                timeout_ms: Some(60000),
            }
        } else if error_str.contains("network") {
            ErrorClassification {
                error_type: ErrorType::Transient,
                severity: ErrorSeverity::Low,
                retryable: true,
                timeout_ms: Some(5000),
            }
        } else {
            ErrorClassification {
                error_type: ErrorType::System,
                severity: ErrorSeverity::High,
                retryable: false,
                timeout_ms: None,
            }
        }
    }
    
    async fn should_retry(&self, error: &(dyn std::error::Error + Send + Sync), attempt: u32) -> bool {
        let classification = self.classify_error(error).await;
        classification.retryable && attempt < 3
    }
    
    async fn report_error(&self, error: &(dyn std::error::Error + Send + Sync), context: &ErrorContext) {
        eprintln!("📊 Error reported - Service: {}, Operation: {}, Error: {}",
                 context.service_name, context.operation, error);
    }
}

struct MockMonitoringProvider;

impl MockMonitoringProvider {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl MonitoringProvider for MockMonitoringProvider {
    async fn record_metric(&self, name: &str, value: f64, tags: &[(&str, &str)]) {
        println!("📊 Metric: {} = {} {:?}", name, value, tags);
    }
    
    async fn record_counter(&self, name: &str, tags: &[(&str, &str)]) {
        println!("📊 Counter: {} {:?}", name, tags);
    }
    
    async fn record_timing(&self, name: &str, duration: std::time::Duration, tags: &[(&str, &str)]) {
        println!("📊 Timing: {} = {:?} {:?}", name, duration, tags);
    }
    
    async fn log_info(&self, message: &str, context: &HashMap<String, String>) {
        println!("ℹ️  {}: {:?}", message, context);
    }
    
    async fn log_warn(&self, message: &str, context: &HashMap<String, String>) {
        println!("⚠️  {}: {:?}", message, context);
    }
    
    async fn log_error(&self, message: &str, context: &HashMap<String, String>) {
        println!("❌ {}: {:?}", message, context);
    }
} 