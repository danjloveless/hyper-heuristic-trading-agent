//! Error types for the logging and monitoring module

use thiserror::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result type for logging and monitoring operations
pub type Result<T> = std::result::Result<T, LoggingMonitoringError>;

/// Comprehensive error types for logging and monitoring operations
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum LoggingMonitoringError {
    // Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Invalid configuration value for {field}: {value}")]
    InvalidConfiguration { field: String, value: String },

    // Logging errors
    #[error("Failed to write log: {message}")]
    LogWriteFailed { message: String },

    #[error("Log buffer full: {buffer_size}")]
    LogBufferFull { buffer_size: usize },

    #[error("Log serialization failed: {message}")]
    LogSerializationFailed { message: String },

    #[error("CloudWatch log delivery failed: {message}")]
    CloudWatchLogFailed { message: String },

    // Tracing errors
    #[error("Failed to create trace: {message}")]
    TraceCreationFailed { message: String },

    #[error("Failed to create span: {message}")]
    SpanCreationFailed { message: String },

    #[error("Invalid trace ID: {trace_id}")]
    InvalidTraceId { trace_id: String },

    #[error("Invalid span ID: {span_id}")]
    InvalidSpanId { span_id: String },

    #[error("X-Ray trace delivery failed: {message}")]
    XRayTraceFailed { message: String },

    // Metrics errors
    #[error("Failed to record metric {name}: {message}")]
    MetricRecordingFailed { name: String, message: String },

    #[error("Invalid metric value for {name}: {value}")]
    InvalidMetricValue { name: String, value: String },

    #[error("Metric aggregation failed: {message}")]
    MetricAggregationFailed { message: String },

    #[error("CloudWatch metric delivery failed: {message}")]
    CloudWatchMetricFailed { message: String },

    #[error("Prometheus metric export failed: {message}")]
    PrometheusExportFailed { message: String },

    // Health monitoring errors
    #[error("Health check failed for service {service}: {message}")]
    HealthCheckFailed { service: String, message: String },

    #[error("Failed to update health status: {message}")]
    HealthUpdateFailed { message: String },

    #[error("Health status not found for service: {service}")]
    HealthStatusNotFound { service: String },

    // Event errors
    #[error("Failed to emit event: {message}")]
    EventEmissionFailed { message: String },

    #[error("Failed to send alert: {message}")]
    AlertDeliveryFailed { message: String },

    #[error("Failed to send notification: {message}")]
    NotificationDeliveryFailed { message: String },

    #[error("SNS delivery failed: {message}")]
    SnsDeliveryFailed { message: String },

    // AWS SDK errors
    #[error("AWS SDK error: {service} - {message}")]
    AwsSdkError { service: String, message: String },

    #[error("AWS credentials error: {message}")]
    AwsCredentialsError { message: String },

    #[error("AWS region configuration error: {message}")]
    AwsRegionError { message: String },

    // Network and I/O errors
    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Connection timeout: {operation}")]
    ConnectionTimeout { operation: String },

    #[error("Request timeout: {operation}")]
    RequestTimeout { operation: String },

    #[error("IO error: {message}")]
    IoError { message: String },

    // Serialization errors
    #[error("JSON serialization failed: {message}")]
    JsonSerializationError { message: String },

    #[error("JSON deserialization failed: {message}")]
    JsonDeserializationError { message: String },

    // Resource errors
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    #[error("Memory allocation failed: {size} bytes")]
    MemoryAllocationFailed { size: usize },

    #[error("File system error: {operation} - {message}")]
    FileSystemError { operation: String, message: String },

    // Validation errors
    #[error("Validation failed for {field}: {message}")]
    ValidationError { field: String, message: String },

    #[error("Required field missing: {field}")]
    MissingRequiredField { field: String },

    #[error("Invalid format for {field}: {value}")]
    InvalidFormat { field: String, value: String },

    // Internal errors
    #[error("Internal error: {message}")]
    Internal { message: String },

    #[error("Unexpected state: {state}")]
    UnexpectedState { state: String },

    #[error("Operation not supported: {operation}")]
    OperationNotSupported { operation: String },

    // Shutdown errors
    #[error("System shutdown in progress")]
    ShutdownInProgress,

    #[error("Component already shutdown: {component}")]
    AlreadyShutdown { component: String },

    // Rate limiting errors
    #[error("Rate limit exceeded: {limit} requests per {period}")]
    RateLimitExceeded { limit: u32, period: String },

    #[error("Throttling applied: {reason}")]
    ThrottlingApplied { reason: String },

    // Wrapped errors
    #[error("Wrapped error: {source}")]
    Wrapped {
        #[from]
        source: Box<LoggingMonitoringError>,
    },
}

impl LoggingMonitoringError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            // Retryable errors
            LoggingMonitoringError::LogWriteFailed { .. } => true,
            LoggingMonitoringError::LogBufferFull { .. } => true,
            LoggingMonitoringError::CloudWatchLogFailed { .. } => true,
            LoggingMonitoringError::TraceCreationFailed { .. } => true,
            LoggingMonitoringError::XRayTraceFailed { .. } => true,
            LoggingMonitoringError::MetricRecordingFailed { .. } => true,
            LoggingMonitoringError::CloudWatchMetricFailed { .. } => true,
            LoggingMonitoringError::PrometheusExportFailed { .. } => true,
            LoggingMonitoringError::HealthUpdateFailed { .. } => true,
            LoggingMonitoringError::EventEmissionFailed { .. } => true,
            LoggingMonitoringError::AlertDeliveryFailed { .. } => true,
            LoggingMonitoringError::NotificationDeliveryFailed { .. } => true,
            LoggingMonitoringError::SnsDeliveryFailed { .. } => true,
            LoggingMonitoringError::AwsSdkError { .. } => true,
            LoggingMonitoringError::NetworkError { .. } => true,
            LoggingMonitoringError::ConnectionTimeout { .. } => true,
            LoggingMonitoringError::RequestTimeout { .. } => true,
            LoggingMonitoringError::IoError { .. } => true,
            LoggingMonitoringError::ResourceExhausted { .. } => true,
            LoggingMonitoringError::RateLimitExceeded { .. } => true,
            LoggingMonitoringError::ThrottlingApplied { .. } => true,

            // Non-retryable errors
            LoggingMonitoringError::Configuration { .. } => false,
            LoggingMonitoringError::InvalidConfiguration { .. } => false,
            LoggingMonitoringError::LogSerializationFailed { .. } => false,
            LoggingMonitoringError::InvalidTraceId { .. } => false,
            LoggingMonitoringError::InvalidSpanId { .. } => false,
            LoggingMonitoringError::InvalidMetricValue { .. } => false,
            LoggingMonitoringError::HealthCheckFailed { .. } => false,
            LoggingMonitoringError::HealthStatusNotFound { .. } => false,
            LoggingMonitoringError::AwsCredentialsError { .. } => false,
            LoggingMonitoringError::AwsRegionError { .. } => false,
            LoggingMonitoringError::JsonSerializationError { .. } => false,
            LoggingMonitoringError::JsonDeserializationError { .. } => false,
            LoggingMonitoringError::MemoryAllocationFailed { .. } => false,
            LoggingMonitoringError::ValidationError { .. } => false,
            LoggingMonitoringError::MissingRequiredField { .. } => false,
            LoggingMonitoringError::InvalidFormat { .. } => false,
            LoggingMonitoringError::Internal { .. } => false,
            LoggingMonitoringError::UnexpectedState { .. } => false,
            LoggingMonitoringError::OperationNotSupported { .. } => false,
            LoggingMonitoringError::ShutdownInProgress => false,
            LoggingMonitoringError::AlreadyShutdown { .. } => false,
            LoggingMonitoringError::Wrapped { source } => source.is_retryable(),
            LoggingMonitoringError::SpanCreationFailed { .. } => true,
            LoggingMonitoringError::MetricAggregationFailed { .. } => true,
            LoggingMonitoringError::FileSystemError { .. } => true,
        }
    }

    /// Get the error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Critical errors
            LoggingMonitoringError::Configuration { .. } => ErrorSeverity::Critical,
            LoggingMonitoringError::AwsCredentialsError { .. } => ErrorSeverity::Critical,
            LoggingMonitoringError::AwsRegionError { .. } => ErrorSeverity::Critical,
            LoggingMonitoringError::MemoryAllocationFailed { .. } => ErrorSeverity::Critical,
            LoggingMonitoringError::Internal { .. } => ErrorSeverity::Critical,
            LoggingMonitoringError::UnexpectedState { .. } => ErrorSeverity::Critical,

            // High severity errors
            LoggingMonitoringError::CloudWatchLogFailed { .. } => ErrorSeverity::High,
            LoggingMonitoringError::XRayTraceFailed { .. } => ErrorSeverity::High,
            LoggingMonitoringError::CloudWatchMetricFailed { .. } => ErrorSeverity::High,
            LoggingMonitoringError::HealthCheckFailed { .. } => ErrorSeverity::High,
            LoggingMonitoringError::AlertDeliveryFailed { .. } => ErrorSeverity::High,
            LoggingMonitoringError::NetworkError { .. } => ErrorSeverity::High,
            LoggingMonitoringError::ResourceExhausted { .. } => ErrorSeverity::High,

            // Medium severity errors
            LoggingMonitoringError::LogWriteFailed { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::LogBufferFull { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::TraceCreationFailed { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::MetricRecordingFailed { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::PrometheusExportFailed { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::EventEmissionFailed { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::NotificationDeliveryFailed { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::SnsDeliveryFailed { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::AwsSdkError { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::ConnectionTimeout { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::RequestTimeout { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::RateLimitExceeded { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::ThrottlingApplied { .. } => ErrorSeverity::Medium,

            // Low severity errors
            LoggingMonitoringError::InvalidConfiguration { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::LogSerializationFailed { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::InvalidTraceId { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::InvalidSpanId { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::InvalidMetricValue { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::HealthUpdateFailed { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::HealthStatusNotFound { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::IoError { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::JsonSerializationError { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::JsonDeserializationError { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::FileSystemError { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::ValidationError { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::MissingRequiredField { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::InvalidFormat { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::OperationNotSupported { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::ShutdownInProgress => ErrorSeverity::Low,
            LoggingMonitoringError::AlreadyShutdown { .. } => ErrorSeverity::Low,
            LoggingMonitoringError::Wrapped { source } => source.severity(),
            LoggingMonitoringError::SpanCreationFailed { .. } => ErrorSeverity::Medium,
            LoggingMonitoringError::MetricAggregationFailed { .. } => ErrorSeverity::Medium,
        }
    }

    /// Get additional context for the error
    pub fn context(&self) -> HashMap<String, String> {
        let mut context = HashMap::new();
        
        match self {
            LoggingMonitoringError::Configuration { message } => {
                context.insert("error_type".to_string(), "configuration".to_string());
                context.insert("message".to_string(), message.clone());
            },
            LoggingMonitoringError::InvalidConfiguration { field, value } => {
                context.insert("error_type".to_string(), "invalid_configuration".to_string());
                context.insert("field".to_string(), field.clone());
                context.insert("value".to_string(), value.clone());
            },
            LoggingMonitoringError::LogWriteFailed { message } => {
                context.insert("error_type".to_string(), "log_write_failed".to_string());
                context.insert("message".to_string(), message.clone());
            },
            LoggingMonitoringError::LogBufferFull { buffer_size } => {
                context.insert("error_type".to_string(), "log_buffer_full".to_string());
                context.insert("buffer_size".to_string(), buffer_size.to_string());
            },
            LoggingMonitoringError::CloudWatchLogFailed { message } => {
                context.insert("error_type".to_string(), "cloudwatch_log_failed".to_string());
                context.insert("message".to_string(), message.clone());
            },
            LoggingMonitoringError::TraceCreationFailed { message } => {
                context.insert("error_type".to_string(), "trace_creation_failed".to_string());
                context.insert("message".to_string(), message.clone());
            },
            LoggingMonitoringError::XRayTraceFailed { message } => {
                context.insert("error_type".to_string(), "xray_trace_failed".to_string());
                context.insert("message".to_string(), message.clone());
            },
            LoggingMonitoringError::MetricRecordingFailed { name, message } => {
                context.insert("error_type".to_string(), "metric_recording_failed".to_string());
                context.insert("metric_name".to_string(), name.clone());
                context.insert("message".to_string(), message.clone());
            },
            LoggingMonitoringError::CloudWatchMetricFailed { message } => {
                context.insert("error_type".to_string(), "cloudwatch_metric_failed".to_string());
                context.insert("message".to_string(), message.clone());
            },
            LoggingMonitoringError::HealthCheckFailed { service, message } => {
                context.insert("error_type".to_string(), "health_check_failed".to_string());
                context.insert("service".to_string(), service.clone());
                context.insert("message".to_string(), message.clone());
            },
            LoggingMonitoringError::NetworkError { message } => {
                context.insert("error_type".to_string(), "network_error".to_string());
                context.insert("message".to_string(), message.clone());
            },
            LoggingMonitoringError::ConnectionTimeout { operation } => {
                context.insert("error_type".to_string(), "connection_timeout".to_string());
                context.insert("operation".to_string(), operation.clone());
            },
            LoggingMonitoringError::RequestTimeout { operation } => {
                context.insert("error_type".to_string(), "request_timeout".to_string());
                context.insert("operation".to_string(), operation.clone());
            },
            LoggingMonitoringError::ResourceExhausted { resource } => {
                context.insert("error_type".to_string(), "resource_exhausted".to_string());
                context.insert("resource".to_string(), resource.clone());
            },
            LoggingMonitoringError::RateLimitExceeded { limit, period } => {
                context.insert("error_type".to_string(), "rate_limit_exceeded".to_string());
                context.insert("limit".to_string(), limit.to_string());
                context.insert("period".to_string(), period.clone());
            },
            LoggingMonitoringError::AwsSdkError { service, message } => {
                context.insert("error_type".to_string(), "aws_sdk_error".to_string());
                context.insert("service".to_string(), service.clone());
                context.insert("message".to_string(), message.clone());
            },
            _ => {
                context.insert("error_type".to_string(), "unknown".to_string());
            }
        }
        
        context.insert("retryable".to_string(), self.is_retryable().to_string());
        context.insert("severity".to_string(), format!("{:?}", self.severity()));
        
        context
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Critical,  // System down, immediate attention required
    High,      // Significant impact, needs attention soon
    Medium,    // Some impact, should be addressed
    Low,       // Minor issue, can be handled later
    Info,      // Informational, no action needed
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Info => write!(f, "INFO"),
        }
    }
}

/// Error context for additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub service_name: String,
    pub operation: String,
    pub request_id: Option<String>,
    pub user_id: Option<String>,
    pub trace_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub additional_data: HashMap<String, serde_json::Value>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(service_name: String, operation: String) -> Self {
        Self {
            service_name,
            operation,
            request_id: None,
            user_id: None,
            trace_id: None,
            timestamp: chrono::Utc::now(),
            additional_data: HashMap::new(),
        }
    }

    /// Add a request ID to the context
    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }

    /// Add a user ID to the context
    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Add a trace ID to the context
    pub fn with_trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = Some(trace_id);
        self
    }

    /// Add additional data to the context
    pub fn with_additional_data(mut self, key: String, value: serde_json::Value) -> Self {
        self.additional_data.insert(key, value);
        self
    }
} 