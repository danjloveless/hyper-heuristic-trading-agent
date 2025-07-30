use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Result type for market data ingestion
pub type Result<T> = std::result::Result<T, IngestionError>;

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum IngestionError {
    #[error("Alpha Vantage API error: {message}")]
    ApiError { message: String, status_code: u16 },
    
    #[error("Rate limit exceeded: {limit_type}")]
    RateLimitExceeded { limit_type: String },
    
    #[error("Data quality below threshold: {score} < {threshold}")]
    QualityBelowThreshold { score: u8, threshold: u8 },
    
    #[error("Symbol not found: {symbol}")]
    SymbolNotFound { symbol: String },
    
    #[error("Data parsing error: {field} - {error}")]
    ParsingError { field: String, error: String },
    
    #[error("Storage error: {operation}")]
    StorageError { operation: String },
    
    #[error("Configuration error: {parameter}")]
    ConfigurationError { parameter: String },
    
    #[error("Deduplication error: {message}")]
    DeduplicationError { message: String },
    
    #[error("Batch processing error: {batch_id}")]
    BatchProcessingError { batch_id: String },
    
    #[error("Authentication failed: {reason}")]
    AuthenticationFailed { reason: String },
    
    #[error("Request timeout: {timeout_ms}ms")]
    RequestTimeout { timeout_ms: u64 },
} 