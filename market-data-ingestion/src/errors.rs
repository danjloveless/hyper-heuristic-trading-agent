use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum IngestionError {
    #[error("Alpha Vantage API error: {message} (status: {status_code:?})")]
    AlphaVantageApi {
        message: String,
        status_code: Option<u16>,
    },
    
    #[error("Rate limit exceeded: {service}. Retry after: {retry_after:?}")]
    RateLimit {
        service: String,
        retry_after: Option<std::time::Duration>,
    },
    
    #[error("Data parsing error in field '{field}': {message}")]
    DataParsing {
        field: String,
        message: String,
    },
    
    #[error("Data quality check failed: {reason} (score: {quality_score})")]
    DataQuality {
        reason: String,
        quality_score: u8,
    },
    
    #[error("Configuration error: {message}")]
    Configuration {
        message: String,
    },
    
    #[error("Storage error: {message}")]
    Storage {
        message: String,
    },
}

impl From<IngestionError> for core_traits::ServiceError {
    fn from(err: IngestionError) -> Self {
        match err {
            IngestionError::AlphaVantageApi { message, status_code } => {
                core_traits::ServiceError::ExternalApi {
                    api: "alpha_vantage".to_string(),
                    message,
                    status_code,
                }
            }
            IngestionError::RateLimit { service, retry_after } => {
                core_traits::ServiceError::RateLimit { service, retry_after }
            }
            IngestionError::DataQuality { reason, quality_score } => {
                core_traits::ServiceError::DataQuality {
                    message: reason,
                    quality_score,
                }
            }
            IngestionError::Configuration { message } => {
                core_traits::ServiceError::Configuration { message }
            }
            IngestionError::Storage { message } => {
                core_traits::ServiceError::Database {
                    message,
                    retryable: true,
                }
            }
            IngestionError::DataParsing { field, message } => {
                core_traits::ServiceError::System {
                    message: format!("Data parsing error in {}: {}", field, message),
                }
            }
        }
    }
} 