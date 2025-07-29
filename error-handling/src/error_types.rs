use super::*;

/// Main error types for the QuantumTrade AI system
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum QuantumTradeError {
    // Configuration errors
    #[error("Configuration not found: {key}")]
    ConfigurationNotFound { key: String },
    
    #[error("Invalid configuration: {key} - {message}")]
    InvalidConfiguration { key: String, message: String },
    
    #[error("Secret access denied: {secret_name}")]
    SecretAccessDenied { secret_name: String },
    
    // Database errors
    #[error("Database connection failed: {host}")]
    DatabaseConnectionFailed { host: String },
    
    #[error("Query execution failed: {query} - {message}")]
    QueryExecutionFailed { query: String, message: String },
    
    #[error("Transaction failed: {operation}")]
    TransactionFailed { operation: String },
    
    #[error("Schema validation failed: {schema}")]
    SchemaValidationFailed { schema: String },
    
    // System errors
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    // Business errors
    #[error("Invalid trading parameters: {message}")]
    InvalidTradingParameters { message: String },
    
    #[error("Market data quality issue: {symbol} - {message}")]
    DataQualityIssue { symbol: String, message: String },
    
    #[error("Prediction confidence too low: {confidence}")]
    LowPredictionConfidence { confidence: f32 },
    
    #[error("Risk limit exceeded: {limit_type}")]
    RiskLimitExceeded { limit_type: String },
    
    // Network errors
    #[error("Request timeout: {operation}")]
    RequestTimeout { operation: String, timeout_ms: u64 },
    
    #[error("Rate limit exceeded: {api}")]
    RateLimitExceeded { api: String, reset_time: DateTime<Utc> },
    
    #[error("Network connection failed: {host}")]
    NetworkConnection { host: String },
    
    // Authentication errors
    #[error("Authentication failed: {reason}")]
    AuthenticationFailed { reason: String },
    
    #[error("Authorization denied: {resource}")]
    AuthorizationDenied { resource: String },
    
    #[error("Token expired")]
    TokenExpired,
    
    // Validation errors
    #[error("Input validation failed: {field} - {message}")]
    ValidationFailed { field: String, message: String },
    
    // Internal errors
    #[error("Internal server error: {message}")]
    Internal { message: String, correlation_id: String },
}

/// Error handling specific errors
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandlingError {
    #[error("Error classification failed: {message}")]
    ClassificationFailed { message: String },
    
    #[error("Retry limit exceeded: {max_attempts}")]
    RetryLimitExceeded { max_attempts: u32 },
    
    #[error("Circuit breaker is open: {service}")]
    CircuitBreakerOpen { service: String },
    
    #[error("Fallback strategy failed: {strategy}")]
    FallbackStrategyFailed { strategy: String },
    
    #[error("Error reporting failed: {destination}")]
    ReportingFailed { destination: String },
}

/// Error context for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub service_name: String,
    pub operation: String,
    pub request_id: Option<String>,
    pub user_id: Option<String>,
    pub trace_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub additional_data: HashMap<String, serde_json::Value>,
}

impl ErrorContext {
    pub fn new(service_name: String, operation: String) -> Self {
        Self {
            service_name,
            operation,
            request_id: None,
            user_id: None,
            trace_id: None,
            timestamp: Utc::now(),
            additional_data: HashMap::new(),
        }
    }
    
    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }
    
    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }
    
    pub fn with_trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = Some(trace_id);
        self
    }
    
    pub fn with_data(mut self, key: String, value: serde_json::Value) -> Self {
        self.additional_data.insert(key, value);
        self
    }
}

/// Error classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorClassification {
    pub error_type: ErrorType,
    pub severity: ErrorSeverity,
    pub retryable: bool,
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ErrorType {
    Transient,
    Permanent,
    System,
    Business,
    Network,
    Authentication,
    Validation,
    RateLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ErrorSeverity {
    Critical,  // System down, immediate attention required
    High,      // Significant impact, needs attention soon
    Medium,    // Some impact, should be addressed
    Low,       // Minor issue, can be handled later
    Info,      // Informational, no action needed
} 