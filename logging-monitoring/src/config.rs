//! Configuration for the logging and monitoring module

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Main configuration for the logging and monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingMonitoringConfig {
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Tracing configuration
    pub tracing: TracingConfig,
    
    /// Metrics configuration
    pub metrics: MetricsConfig,
    
    /// Health monitoring configuration
    pub health: HealthConfig,
    
    /// Event management configuration
    pub events: EventsConfig,
    
    /// General monitoring configuration
    pub monitoring: MonitoringConfig,
}

impl Default for LoggingMonitoringConfig {
    fn default() -> Self {
        Self {
            logging: LoggingConfig::default(),
            tracing: TracingConfig::default(),
            metrics: MetricsConfig::default(),
            health: HealthConfig::default(),
            events: EventsConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (DEBUG, INFO, WARN, ERROR)
    pub level: String,
    
    /// Environment filter for tracing subscriber
    pub env_filter: String,
    
    /// Whether to enable JSON formatting
    pub json_format: bool,
    
    /// Whether to include file and line information
    pub include_file_line: bool,
    
    /// Whether to include thread information
    pub include_thread_info: bool,
    
    /// CloudWatch configuration
    pub cloudwatch: CloudWatchLogConfig,
    
    /// Local file logging configuration
    pub local_file: LocalFileLogConfig,
    
    /// Buffer configuration
    pub buffer: LogBufferConfig,
    
    /// Performance settings
    pub performance: LogPerformanceConfig,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "INFO".to_string(),
            env_filter: "info,quantumtrade=debug".to_string(),
            json_format: true,
            include_file_line: true,
            include_thread_info: true,
            cloudwatch: CloudWatchLogConfig::default(),
            local_file: LocalFileLogConfig::default(),
            buffer: LogBufferConfig::default(),
            performance: LogPerformanceConfig::default(),
        }
    }
}

/// CloudWatch logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudWatchLogConfig {
    /// Whether to enable CloudWatch logging
    pub enabled: bool,
    
    /// CloudWatch log group name
    pub log_group: String,
    
    /// CloudWatch log stream name
    pub log_stream: String,
    
    /// AWS region
    pub region: String,
    
    /// Batch size for log delivery
    pub batch_size: usize,
    
    /// Flush interval
    pub flush_interval: Duration,
    
    /// Retry configuration
    pub retry: RetryConfig,
}

impl Default for CloudWatchLogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_group: "/quantumtrade/logs".to_string(),
            log_stream: "application".to_string(),
            region: "us-east-1".to_string(),
            batch_size: 100,
            flush_interval: Duration::from_secs(5),
            retry: RetryConfig::default(),
        }
    }
}

/// Local file logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalFileLogConfig {
    /// Whether to enable local file logging
    pub enabled: bool,
    
    /// Log file path
    pub file_path: String,
    
    /// Maximum file size in bytes
    pub max_file_size: usize,
    
    /// Maximum number of backup files
    pub max_backup_files: usize,
    
    /// Whether to compress old log files
    pub compress_backups: bool,
}

impl Default for LocalFileLogConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            file_path: "logs/quantumtrade.log".to_string(),
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_backup_files: 10,
            compress_backups: true,
        }
    }
}

/// Log buffer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogBufferConfig {
    /// Buffer size in number of log entries
    pub size: usize,
    
    /// Whether to drop logs when buffer is full
    pub drop_when_full: bool,
    
    /// Flush interval
    pub flush_interval: Duration,
}

impl Default for LogBufferConfig {
    fn default() -> Self {
        Self {
            size: 1000,
            drop_when_full: true,
            flush_interval: Duration::from_secs(1),
        }
    }
}

/// Log performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogPerformanceConfig {
    /// Whether to enable async logging
    pub async_logging: bool,
    
    /// Worker thread count
    pub worker_threads: usize,
    
    /// Channel capacity
    pub channel_capacity: usize,
    
    /// Whether to enable log sampling
    pub enable_sampling: bool,
    
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
}

impl Default for LogPerformanceConfig {
    fn default() -> Self {
        Self {
            async_logging: true,
            worker_threads: 2,
            channel_capacity: 10000, // Increased from default
            enable_sampling: false,
            sampling_rate: 1.0,
        }
    }
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Whether to enable distributed tracing
    pub enabled: bool,
    
    /// X-Ray configuration
    pub xray: XRayConfig,
    
    /// Sampling configuration
    pub sampling: TraceSamplingConfig,
    
    /// Span configuration
    pub span: SpanConfig,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            xray: XRayConfig::default(),
            sampling: TraceSamplingConfig::default(),
            span: SpanConfig::default(),
        }
    }
}

/// X-Ray configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XRayConfig {
    /// Whether to enable X-Ray integration
    pub enabled: bool,
    
    /// X-Ray daemon endpoint
    pub daemon_endpoint: String,
    
    /// Service name
    pub service_name: String,
    
    /// Environment name
    pub environment: String,
    
    /// Whether to enable subsegment creation
    pub enable_subsegments: bool,
}

impl Default for XRayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            daemon_endpoint: "127.0.0.1:2000".to_string(),
            service_name: "quantumtrade".to_string(),
            environment: "production".to_string(),
            enable_subsegments: true,
        }
    }
}

/// Trace sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSamplingConfig {
    /// Sampling rate (0.0 to 1.0)
    pub rate: f64,
    
    /// Whether to enable adaptive sampling
    pub adaptive: bool,
    
    /// Minimum sampling rate
    pub min_rate: f64,
    
    /// Maximum sampling rate
    pub max_rate: f64,
}

impl Default for TraceSamplingConfig {
    fn default() -> Self {
        Self {
            rate: 0.1,
            adaptive: true,
            min_rate: 0.01,
            max_rate: 0.5,
        }
    }
}

/// Span configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanConfig {
    /// Maximum span duration
    pub max_duration: Duration,
    
    /// Whether to include span metadata
    pub include_metadata: bool,
    
    /// Maximum annotation count per span
    pub max_annotations: usize,
    
    /// Whether to enable span compression
    pub enable_compression: bool,
}

impl Default for SpanConfig {
    fn default() -> Self {
        Self {
            max_duration: Duration::from_secs(300), // 5 minutes
            include_metadata: true,
            max_annotations: 100,
            enable_compression: true,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Whether to enable metrics collection
    pub enabled: bool,
    
    /// CloudWatch metrics configuration
    pub cloudwatch: CloudWatchMetricsConfig,
    
    /// Prometheus configuration
    pub prometheus: PrometheusConfig,
    
    /// Aggregation configuration
    pub aggregation: MetricsAggregationConfig,
    
    /// Performance configuration
    pub performance: MetricsPerformanceConfig,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cloudwatch: CloudWatchMetricsConfig::default(),
            prometheus: PrometheusConfig::default(),
            aggregation: MetricsAggregationConfig::default(),
            performance: MetricsPerformanceConfig::default(),
        }
    }
}

/// CloudWatch metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudWatchMetricsConfig {
    /// Whether to enable CloudWatch metrics
    pub enabled: bool,
    
    /// Namespace for metrics
    pub namespace: String,
    
    /// AWS region
    pub region: String,
    
    /// Batch size for metric delivery
    pub batch_size: usize,
    
    /// Flush interval
    pub flush_interval: Duration,
    
    /// Retry configuration
    pub retry: RetryConfig,
}

impl Default for CloudWatchMetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            namespace: "QuantumTrade".to_string(),
            region: "us-east-1".to_string(),
            batch_size: 20,
            flush_interval: Duration::from_secs(60),
            retry: RetryConfig::default(),
        }
    }
}

/// Prometheus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    /// Whether to enable Prometheus metrics
    pub enabled: bool,
    
    /// HTTP endpoint for metrics exposition
    pub endpoint: String,
    
    /// Port for metrics server
    pub port: u16,
    
    /// Metrics path
    pub path: String,
    
    /// Whether to enable default metrics
    pub enable_default_metrics: bool,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "0.0.0.0".to_string(),
            port: 9090,
            path: "/metrics".to_string(),
            enable_default_metrics: true,
        }
    }
}

/// Metrics aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAggregationConfig {
    /// Aggregation intervals
    pub intervals: Vec<Duration>,
    
    /// Whether to enable pre-aggregation
    pub pre_aggregation: bool,
    
    /// Retention periods by metric type
    pub retention: HashMap<String, Duration>,
}

impl Default for MetricsAggregationConfig {
    fn default() -> Self {
        let mut retention = HashMap::new();
        retention.insert("counter".to_string(), Duration::from_secs(86400 * 30)); // 30 days
        retention.insert("gauge".to_string(), Duration::from_secs(86400 * 7)); // 7 days
        retention.insert("histogram".to_string(), Duration::from_secs(86400 * 14)); // 14 days
        
        Self {
            intervals: vec![
                Duration::from_secs(60),   // 1 minute
                Duration::from_secs(300),  // 5 minutes
                Duration::from_secs(3600), // 1 hour
                Duration::from_secs(86400), // 1 day
            ],
            pre_aggregation: true,
            retention,
        }
    }
}

/// Metrics performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsPerformanceConfig {
    /// Whether to enable async metric recording
    pub async_recording: bool,
    
    /// Buffer size for metrics
    pub buffer_size: usize,
    
    /// Worker thread count
    pub worker_threads: usize,
    
    /// Whether to enable metric sampling
    pub enable_sampling: bool,
    
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
}

impl Default for MetricsPerformanceConfig {
    fn default() -> Self {
        Self {
            async_recording: true,
            buffer_size: 10000,
            worker_threads: 2,
            enable_sampling: false,
            sampling_rate: 1.0,
        }
    }
}

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Whether to enable health monitoring
    pub enabled: bool,
    
    /// Health check interval
    pub check_interval: Duration,
    
    /// Health check timeout
    pub check_timeout: Duration,
    
    /// Service health configuration
    pub services: HashMap<String, ServiceHealthConfig>,
    
    /// Alerting configuration
    pub alerting: HealthAlertingConfig,
}

impl Default for HealthConfig {
    fn default() -> Self {
        let mut services = HashMap::new();
        services.insert("default".to_string(), ServiceHealthConfig::default());
        
        Self {
            enabled: true,
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(10),
            services,
            alerting: HealthAlertingConfig::default(),
        }
    }
}

/// Service health configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealthConfig {
    /// Health check endpoint
    pub endpoint: String,
    
    /// Expected response time
    pub expected_response_time: Duration,
    
    /// Retry count for health checks
    pub retry_count: u32,
    
    /// Whether to enable circuit breaker
    pub circuit_breaker: bool,
    
    /// Circuit breaker failure threshold
    pub failure_threshold: u32,
}

impl Default for ServiceHealthConfig {
    fn default() -> Self {
        Self {
            endpoint: "/health".to_string(),
            expected_response_time: Duration::from_secs(5),
            retry_count: 3,
            circuit_breaker: true,
            failure_threshold: 5,
        }
    }
}

/// Health alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlertingConfig {
    /// Whether to enable health alerts
    pub enabled: bool,
    
    /// Alert thresholds
    pub thresholds: HealthThresholds,
    
    /// Notification channels
    pub notification_channels: Vec<String>,
}

impl Default for HealthAlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: HealthThresholds::default(),
            notification_channels: vec!["sns".to_string()],
        }
    }
}

/// Health thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthThresholds {
    /// Critical response time threshold
    pub critical_response_time: Duration,
    
    /// Warning response time threshold
    pub warning_response_time: Duration,
    
    /// Error rate threshold
    pub error_rate_threshold: f64,
    
    /// Availability threshold
    pub availability_threshold: f64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            critical_response_time: Duration::from_secs(10),
            warning_response_time: Duration::from_secs(5),
            error_rate_threshold: 0.05, // 5%
            availability_threshold: 0.99, // 99%
        }
    }
}

/// Event performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPerformanceConfig {
    /// Whether to enable async processing
    pub async_processing: bool,
    
    /// Channel capacity
    pub channel_capacity: usize,
    
    /// Worker thread count
    pub worker_threads: usize,
    
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for EventPerformanceConfig {
    fn default() -> Self {
        Self {
            async_processing: true,
            channel_capacity: 10000, // Increased capacity
            worker_threads: 1,
            batch_size: 100,
        }
    }
}

/// Events configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventsConfig {
    /// Whether to enable event management
    pub enabled: bool,
    
    /// SNS configuration
    pub sns: SnsConfig,
    
    /// Event routing configuration
    pub routing: EventRoutingConfig,
    
    /// Event filtering configuration
    pub filtering: EventFilteringConfig,
    
    /// Performance configuration
    pub performance: EventPerformanceConfig,
}

impl Default for EventsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sns: SnsConfig::default(),
            routing: EventRoutingConfig::default(),
            filtering: EventFilteringConfig::default(),
            performance: EventPerformanceConfig::default(),
        }
    }
}

/// SNS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnsConfig {
    /// Whether to enable SNS notifications
    pub enabled: bool,
    
    /// AWS region
    pub region: String,
    
    /// Topic ARN for alerts
    pub alert_topic_arn: String,
    
    /// Topic ARN for notifications
    pub notification_topic_arn: String,
    
    /// Retry configuration
    pub retry: RetryConfig,
}

impl Default for SnsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            region: "us-east-1".to_string(),
            alert_topic_arn: "arn:aws:sns:us-east-1:123456789012:quantumtrade-alerts".to_string(),
            notification_topic_arn: "arn:aws:sns:us-east-1:123456789012:quantumtrade-notifications".to_string(),
            retry: RetryConfig::default(),
        }
    }
}

/// Event routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRoutingConfig {
    /// Event type to topic mapping
    pub event_routes: HashMap<String, String>,
    
    /// Default topic for unhandled events
    pub default_topic: String,
    
    /// Whether to enable event persistence
    pub enable_persistence: bool,
}

impl Default for EventRoutingConfig {
    fn default() -> Self {
        let mut event_routes = HashMap::new();
        event_routes.insert("alert".to_string(), "alerts".to_string());
        event_routes.insert("notification".to_string(), "notifications".to_string());
        event_routes.insert("metric".to_string(), "metrics".to_string());
        
        Self {
            event_routes,
            default_topic: "events".to_string(),
            enable_persistence: false,
        }
    }
}

/// Event filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilteringConfig {
    /// Whether to enable event filtering
    pub enabled: bool,
    
    /// Event type filters
    pub event_type_filters: Vec<String>,
    
    /// Severity filters
    pub severity_filters: Vec<String>,
    
    /// Service filters
    pub service_filters: Vec<String>,
}

impl Default for EventFilteringConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            event_type_filters: vec![],
            severity_filters: vec!["critical".to_string(), "high".to_string()],
            service_filters: vec![],
        }
    }
}

/// General monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Whether to enable general monitoring
    pub enabled: bool,
    
    /// Performance monitoring configuration
    pub performance: PerformanceMonitoringConfig,
    
    /// Resource monitoring configuration
    pub resources: ResourceMonitoringConfig,
    
    /// Business metrics configuration
    pub business: BusinessMetricsConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            performance: PerformanceMonitoringConfig::default(),
            resources: ResourceMonitoringConfig::default(),
            business: BusinessMetricsConfig::default(),
        }
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    /// Whether to enable performance monitoring
    pub enabled: bool,
    
    /// Latency thresholds
    pub latency_thresholds: LatencyThresholds,
    
    /// Throughput monitoring
    pub throughput_monitoring: bool,
    
    /// Error rate monitoring
    pub error_rate_monitoring: bool,
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            latency_thresholds: LatencyThresholds::default(),
            throughput_monitoring: true,
            error_rate_monitoring: true,
        }
    }
}

/// Latency thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyThresholds {
    /// P50 latency threshold
    pub p50_threshold: Duration,
    
    /// P95 latency threshold
    pub p95_threshold: Duration,
    
    /// P99 latency threshold
    pub p99_threshold: Duration,
}

impl Default for LatencyThresholds {
    fn default() -> Self {
        Self {
            p50_threshold: Duration::from_millis(100),
            p95_threshold: Duration::from_millis(500),
            p99_threshold: Duration::from_millis(1000),
        }
    }
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    /// Whether to enable resource monitoring
    pub enabled: bool,
    
    /// CPU monitoring
    pub cpu_monitoring: bool,
    
    /// Memory monitoring
    pub memory_monitoring: bool,
    
    /// Network monitoring
    pub network_monitoring: bool,
    
    /// Disk monitoring
    pub disk_monitoring: bool,
}

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cpu_monitoring: true,
            memory_monitoring: true,
            network_monitoring: true,
            disk_monitoring: true,
        }
    }
}

/// Business metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessMetricsConfig {
    /// Whether to enable business metrics
    pub enabled: bool,
    
    /// Prediction accuracy monitoring
    pub prediction_accuracy: bool,
    
    /// Trading performance monitoring
    pub trading_performance: bool,
    
    /// User activity monitoring
    pub user_activity: bool,
}

impl Default for BusinessMetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prediction_accuracy: true,
            trading_performance: true,
            user_activity: true,
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    
    /// Initial delay
    pub initial_delay: Duration,
    
    /// Maximum delay
    pub max_delay: Duration,
    
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Whether to enable jitter
    pub enable_jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            enable_jitter: true,
        }
    }
} 