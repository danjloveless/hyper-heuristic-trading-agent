//! QuantumTrade AI Market Data Ingestion Service
//! 
//! This service collects financial market data from Alpha Vantage API
//! and processes it through the standardized data pipeline.

pub mod service;
pub mod collectors;
pub mod processors;
pub mod models;
pub mod config;
pub mod errors;
pub mod config_provider;

// Re-export main types
pub use service::MarketDataIngestionService;
pub use config::*;
pub use models::*;
pub use errors::*;

use core_traits::ServiceResult;

/// Create a new market data ingestion service with dependency injection
pub async fn create_service(
    config_provider: std::sync::Arc<dyn core_traits::ConfigurationProvider>,
    database_manager: std::sync::Arc<database_abstraction::DatabaseManager>,
    error_handler: std::sync::Arc<dyn core_traits::ErrorHandler>,
    monitoring: std::sync::Arc<dyn core_traits::MonitoringProvider>,
) -> ServiceResult<MarketDataIngestionService> {
    MarketDataIngestionService::new(
        config_provider,
        database_manager,
        error_handler,
        monitoring,
    ).await
} 