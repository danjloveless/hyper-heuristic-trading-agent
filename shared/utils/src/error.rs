// shared/utils/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QuantumTradeError {
    #[error("Database connection error: {0}")]
    DatabaseConnection(#[from] clickhouse::error::Error),
    
    #[error("Redis connection error: {0}")]
    RedisConnection(#[from] redis::RedisError),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Data validation error: {message}")]
    DataValidation { message: String },
    
    #[error("Query execution error: {message}")]
    QueryExecution { message: String },
    
    #[error("Data not found: {entity} with identifier {id}")]
    NotFound { entity: String, id: String },
    
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    #[error("Rate limit exceeded for operation: {operation}")]
    RateLimit { operation: String },
    
    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, QuantumTradeError>;