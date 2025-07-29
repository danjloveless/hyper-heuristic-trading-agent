use super::*;

/// Configuration for the error handling system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    pub retry: RetryConfig,
    pub circuit_breaker: CircuitBreakerConfig,
    pub fallback: FallbackConfig,
    pub reporting: ReportingConfig,
    pub classification: ClassificationConfig,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            fallback: FallbackConfig::default(),
            reporting: ReportingConfig::default(),
            classification: ClassificationConfig::default(),
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
    pub jitter: bool,
    pub retryable_errors: Vec<ErrorType>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 100,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            jitter: true,
            retryable_errors: vec![
                ErrorType::Transient,
                ErrorType::Network,
                ErrorType::System,
            ],
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_ms: u64,
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_ms: 60000,
            half_open_max_calls: 3,
        }
    }
}

/// Fallback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackConfig {
    pub enabled: bool,
    pub strategies: HashMap<String, FallbackStrategy>,
    pub default_strategy: FallbackStrategy,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        let mut strategies = HashMap::new();
        strategies.insert("default".to_string(), FallbackStrategy::DefaultValue);
        strategies.insert("cache".to_string(), FallbackStrategy::CacheLookup);
        strategies.insert("previous".to_string(), FallbackStrategy::PreviousResult);
        
        Self {
            enabled: true,
            strategies,
            default_strategy: FallbackStrategy::DefaultValue,
        }
    }
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    pub enabled: bool,
    pub destinations: Vec<ReportingDestination>,
    pub batch_size: u32,
    pub flush_interval_ms: u64,
    pub severity_threshold: ErrorSeverity,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            destinations: vec![
                ReportingDestination::Logs,
                ReportingDestination::Metrics,
            ],
            batch_size: 100,
            flush_interval_ms: 5000,
            severity_threshold: ErrorSeverity::Medium,
        }
    }
}

/// Classification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationConfig {
    pub rules: Vec<ClassificationRule>,
    pub default_classification: ErrorClassification,
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self {
            rules: Self::default_rules(),
            default_classification: ErrorClassification {
                error_type: ErrorType::System,
                severity: ErrorSeverity::Medium,
                retryable: false,
                timeout_ms: None,
                max_retries: None,
            },
        }
    }
}

impl ClassificationConfig {
    fn default_rules() -> Vec<ClassificationRule> {
        vec![
            ClassificationRule {
                pattern: "RequestTimeout".to_string(),
                classification: ErrorClassification {
                    error_type: ErrorType::Transient,
                    severity: ErrorSeverity::Medium,
                    retryable: true,
                    timeout_ms: Some(30000),
                    max_retries: Some(3),
                },
            },
            ClassificationRule {
                pattern: "NetworkConnection".to_string(),
                classification: ErrorClassification {
                    error_type: ErrorType::Network,
                    severity: ErrorSeverity::High,
                    retryable: true,
                    timeout_ms: Some(10000),
                    max_retries: Some(5),
                },
            },
            ClassificationRule {
                pattern: "DatabaseConnection".to_string(),
                classification: ErrorClassification {
                    error_type: ErrorType::System,
                    severity: ErrorSeverity::Critical,
                    retryable: true,
                    timeout_ms: Some(5000),
                    max_retries: Some(3),
                },
            },
            ClassificationRule {
                pattern: "AuthenticationFailed".to_string(),
                classification: ErrorClassification {
                    error_type: ErrorType::Authentication,
                    severity: ErrorSeverity::High,
                    retryable: false,
                    timeout_ms: None,
                    max_retries: None,
                },
            },
            ClassificationRule {
                pattern: "ValidationFailed".to_string(),
                classification: ErrorClassification {
                    error_type: ErrorType::Validation,
                    severity: ErrorSeverity::Low,
                    retryable: false,
                    timeout_ms: None,
                    max_retries: None,
                },
            },
            ClassificationRule {
                pattern: "RateLimitExceeded".to_string(),
                classification: ErrorClassification {
                    error_type: ErrorType::RateLimit,
                    severity: ErrorSeverity::Medium,
                    retryable: true,
                    timeout_ms: Some(60000),
                    max_retries: Some(3),
                },
            },
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationRule {
    pub pattern: String,
    pub classification: ErrorClassification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStrategy {
    DefaultValue,
    CacheLookup,
    PreviousResult,
    AlternativeService,
    GracefulDegradation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportingDestination {
    Logs,
    Metrics,
    CloudWatch,
    SNS,
    Slack,
    Email,
} 