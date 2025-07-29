use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection error: {message}")]
    ConnectionError { message: String },
    
    #[error("Query error: {message}")]
    QueryError { message: String },
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("ClickHouse error: {0}")]
    ClickHouseError(#[from] clickhouse::error::Error),
    
    #[error("Redis error: {0}")]
    RedisError(#[from] redis::RedisError),
    
    #[error("Migration error: {message}")]
    MigrationError { message: String },
    
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },
    
    #[error("Pool error: {message}")]
    PoolError { message: String },
    
    #[error("Timeout error: operation timed out after {seconds}s")]
    TimeoutError { seconds: u64 },
    
    #[error("Data integrity error: {message}")]
    DataIntegrityError { message: String },
    
    #[error("Schema version mismatch: expected {expected}, found {found}")]
    SchemaVersionMismatch { expected: u32, found: u32 },
}

impl DatabaseError {
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            Self::ConnectionError { .. }
                | Self::TimeoutError { .. }
                | Self::PoolError { .. }
        )
    }
    
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::SchemaVersionMismatch { .. }
                | Self::ConfigurationError { .. }
        )
    }
}