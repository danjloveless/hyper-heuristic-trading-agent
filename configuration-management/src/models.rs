use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::Validate;
use tokio::sync::broadcast;

// Type alias for configuration change stream
pub type ConfigChangeStream = broadcast::Receiver<ConfigurationChange>;

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
pub struct TechnicalIndicatorsConfig {
    pub supported_indicators: Vec<IndicatorType>,
    pub calculation_modes: CalculationModeConfig,
    pub caching: CachingConfig,
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PredictionServiceConfig {
    pub models: ModelConfig,
    pub prediction_horizons: Vec<u32>,
    pub confidence_thresholds: ConfidenceConfig,
    pub performance_targets: PerformanceTargetConfig,
}

// Supporting types for TechnicalIndicatorsConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndicatorType {
    SMA,
    EMA,
    RSI,
    MACD,
    BollingerBands,
    Stochastic,
    WilliamsR,
    ATR,
    Custom { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CalculationModeConfig {
    pub real_time: bool,
    pub batch_processing: bool,
    pub parallel_calculation: bool,
    pub max_parallel_tasks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CachingConfig {
    pub enabled: bool,
    pub ttl_seconds: u32,
    pub max_cache_size: usize,
    pub cache_strategy: CacheStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStrategy {
    LRU,
    LFU,
    FIFO,
    TTL,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PerformanceConfig {
    pub max_calculation_time_ms: u64,
    pub memory_limit_mb: u32,
    pub cpu_limit_percent: u8,
    pub enable_profiling: bool,
}

// Supporting types for PredictionServiceConfig
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub version: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub training_config: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LSTM,
    GRU,
    Transformer,
    RandomForest,
    XGBoost,
    Custom { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub validation_split: f64,
    pub early_stopping: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ConfidenceConfig {
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub confidence_thresholds: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PerformanceTargetConfig {
    pub accuracy_target: f64,
    pub latency_target_ms: u64,
    pub throughput_target: u32,
    pub resource_limits: ResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ResourceLimits {
    pub max_memory_mb: u32,
    pub max_cpu_percent: u8,
    pub max_gpu_memory_mb: Option<u32>,
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

// Production-ready default configurations
impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            service_name: "quantumtrade-service".to_string(),
            environment: Environment::Development,
            database: DatabaseConfig::default(),
            apis: ApiConfig::default(),
            logging: LoggingConfig::default(),
            monitoring: MonitoringConfig::default(),
            features: HashMap::new(),
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            clickhouse: ClickHouseConfig::default(),
            redis: RedisConfig::default(),
            connection_pool: ConnectionPoolConfig::default(),
        }
    }
}

impl Default for ClickHouseConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8123,
            database: "quantumtrade".to_string(),
            username: None,
            password: None,
            connection_pool: ConnectionPoolConfig::default(),
        }
    }
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 6379,
            database: 0,
            username: None,
            password: None,
            connection_pool: ConnectionPoolConfig::default(),
        }
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            alpha_vantage: AlphaVantageConfig::default(),
            alpha_intelligence: AlphaIntelligenceConfig::default(),
            rate_limits: RateLimitConfig::default(),
            timeout_ms: 10000,
            retry_config: RetryConfig::default(),
        }
    }
}

impl Default for AlphaVantageConfig {
    fn default() -> Self {
        Self {
            base_url: "https://www.alphavantage.co".to_string(),
            api_key: "".to_string(), // Must be set via environment
            rate_limit: RateLimitConfig::default(),
            timeout_ms: 10000,
        }
    }
}

impl Default for AlphaIntelligenceConfig {
    fn default() -> Self {
        Self {
            base_url: "https://www.alphavantage.co/query".to_string(),
            api_key: "".to_string(), // Must be set via environment
            rate_limit: RateLimitConfig::default(),
            timeout_ms: 10000,
        }
    }
}

impl Default for MarketDataIngestionConfig {
    fn default() -> Self {
        Self {
            alpha_vantage: AlphaVantageConfig::default(),
            symbols: vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()],
            update_intervals: UpdateIntervalConfig::default(),
            data_quality: DataQualityConfig::default(),
            error_handling: ErrorHandlingConfig::default(),
        }
    }
}

impl Default for TechnicalIndicatorsConfig {
    fn default() -> Self {
        Self {
            supported_indicators: vec![
                IndicatorType::SMA,
                IndicatorType::EMA,
                IndicatorType::RSI,
                IndicatorType::MACD,
            ],
            calculation_modes: CalculationModeConfig::default(),
            caching: CachingConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for PredictionServiceConfig {
    fn default() -> Self {
        Self {
            models: ModelConfig::default(),
            prediction_horizons: vec![1, 5, 10, 30],
            confidence_thresholds: ConfidenceConfig::default(),
            performance_targets: PerformanceTargetConfig::default(),
        }
    }
}

impl Default for UpdateIntervalConfig {
    fn default() -> Self {
        Self {
            real_time_seconds: 1,
            intraday_minutes: 5,
            daily_hours: 1,
            weekly_hours: 24,
        }
    }
}

impl Default for DataQualityConfig {
    fn default() -> Self {
        Self {
            min_quality_score: 80,
            validation_rules: Vec::new(),
            outlier_detection: OutlierDetectionConfig::default(),
        }
    }
}

impl Default for OutlierDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: OutlierDetectionMethod::ZScore,
            threshold: 3.0,
            window_size: 100,
        }
    }
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay_ms: 1000,
            circuit_breaker: CircuitBreakerConfig::default(),
            fallback_strategies: vec![FallbackStrategy::Cache],
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout_ms: 30000,
            half_open_max_calls: 3,
        }
    }
}

impl Default for CalculationModeConfig {
    fn default() -> Self {
        Self {
            real_time: true,
            batch_processing: false,
            parallel_calculation: true,
            max_parallel_tasks: 4,
        }
    }
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl_seconds: 300,
            max_cache_size: 10000,
            cache_strategy: CacheStrategy::LRU,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_calculation_time_ms: 5000,
            memory_limit_mb: 512,
            cpu_limit_percent: 80,
            enable_profiling: false,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::LSTM,
            version: "1.0.0".to_string(),
            parameters: HashMap::new(),
            training_config: TrainingConfig::default(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            validation_split: 0.2,
            early_stopping: true,
        }
    }
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_confidence: 0.95,
            confidence_thresholds: HashMap::new(),
        }
    }
}

impl Default for PerformanceTargetConfig {
    fn default() -> Self {
        Self {
            accuracy_target: 0.85,
            latency_target_ms: 100,
            throughput_target: 1000,
            resource_limits: ResourceLimits::default(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048,
            max_cpu_percent: 90,
            max_gpu_memory_mb: None,
        }
    }
} 