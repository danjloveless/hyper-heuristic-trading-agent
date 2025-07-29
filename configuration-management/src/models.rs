use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServiceConfig {
    pub service_name: String,
    pub environment: Environment,
    pub database: DatabaseConfig,
    pub apis: ApiConfig,
    pub logging: LoggingConfig,
    pub monitoring: MonitoringConfig,
    pub features: HashMap<String, FeatureConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DatabaseConfig {
    pub clickhouse: ClickHouseConfig,
    pub redis: RedisConfig,
    pub connection_pool: ConnectionPoolConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ClickHouseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub connection_pool: ConnectionPoolConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RedisConfig {
    pub host: String,
    pub port: u16,
    pub database: u8,
    pub username: Option<String>,
    pub password: Option<String>,
    pub connection_pool: ConnectionPoolConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ConnectionPoolConfig {
    pub max_connections: u32,
    pub min_connections: u32,
    pub timeout_seconds: u32,
    pub idle_timeout_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ApiConfig {
    pub alpha_vantage: AlphaVantageConfig,
    pub alpha_intelligence: AlphaIntelligenceConfig,
    pub rate_limits: RateLimitConfig,
    pub timeout_ms: u32,
    pub retry_config: RetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AlphaVantageConfig {
    pub base_url: String,
    pub api_key: String,
    pub rate_limit: RateLimitConfig,
    pub timeout_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AlphaIntelligenceConfig {
    pub base_url: String,
    pub api_key: String,
    pub rate_limit: RateLimitConfig,
    pub timeout_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub requests_per_day: u32,
    pub burst_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
    pub jitter: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LoggingConfig {
    pub level: String,
    pub format: LogFormat,
    pub output: LogOutput,
    pub structured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    Json,
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutput {
    Console,
    File { path: String },
    CloudWatch { log_group: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub tracing_enabled: bool,
    pub health_check_interval_seconds: u32,
    pub alerting: AlertingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AlertingConfig {
    pub enabled: bool,
    pub sns_topic_arn: Option<String>,
    pub slack_webhook: Option<String>,
    pub email_recipients: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct FeatureConfig {
    pub enabled: bool,
    pub rollout_percentage: f32,
    pub target_groups: Vec<String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Environment {
    Development,
    Staging,
    Production,
}

impl std::fmt::Display for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Environment::Development => write!(f, "development"),
            Environment::Staging => write!(f, "staging"),
            Environment::Production => write!(f, "production"),
        }
    }
}

impl std::str::FromStr for Environment {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "development" | "dev" => Ok(Environment::Development),
            "staging" | "stage" => Ok(Environment::Staging),
            "production" | "prod" => Ok(Environment::Production),
            _ => Err(format!("Unknown environment: {}", s)),
        }
    }
}

// Service-specific configurations
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MarketDataIngestionConfig {
    pub alpha_vantage: AlphaVantageConfig,
    pub symbols: Vec<String>,
    pub update_intervals: UpdateIntervalConfig,
    pub data_quality: DataQualityConfig,
    pub error_handling: ErrorHandlingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct UpdateIntervalConfig {
    pub real_time_seconds: u32,
    pub intraday_minutes: u32,
    pub daily_hours: u32,
    pub weekly_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DataQualityConfig {
    pub min_quality_score: u8,
    pub validation_rules: Vec<ValidationRule>,
    pub outlier_detection: OutlierDetectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub field: String,
    pub rule_type: ValidationRuleType,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    Range { min: f64, max: f64 },
    NotNull,
    NotEmpty,
    Pattern { regex: String },
    Custom { function: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct OutlierDetectionConfig {
    pub enabled: bool,
    pub method: OutlierDetectionMethod,
    pub threshold: f64,
    pub window_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierDetectionMethod {
    ZScore,
    IQR,
    IsolationForest,
    LocalOutlierFactor,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ErrorHandlingConfig {
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub circuit_breaker: CircuitBreakerConfig,
    pub fallback_strategies: Vec<FallbackStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_ms: u64,
    pub half_open_max_calls: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStrategy {
    Cache,
    DefaultValue { value: serde_json::Value },
    AlternativeSource { source: String },
    DegradedMode,
}

// Configuration change tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationChange {
    pub key: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: serde_json::Value,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub severity: ValidationSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

// Default implementations
impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 20,
            min_connections: 5,
            timeout_seconds: 30,
            idle_timeout_seconds: 300,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            requests_per_day: 1000,
            burst_size: 10,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: LogFormat::Json,
            output: LogOutput::Console,
            structured: true,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_enabled: true,
            tracing_enabled: true,
            health_check_interval_seconds: 30,
            alerting: AlertingConfig::default(),
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sns_topic_arn: None,
            slack_webhook: None,
            email_recipients: Vec::new(),
        }
    }
} 