//! Logging & Monitoring Module for QuantumTrade AI
//! 
//! This module provides comprehensive observability across all QuantumTrade AI services
//! through structured logging, distributed tracing, metrics collection, and real-time monitoring.
//! 
//! ## Features
//! - Structured JSON logging with correlation IDs and context
//! - Distributed tracing with AWS X-Ray integration
//! - Metrics collection and aggregation
//! - Real-time health monitoring
//! - CloudWatch integration for centralized logging
//! - Performance monitoring with minimal overhead
//! 
//! ## Quick Start
//! 
//! ```rust
//! use logging_monitoring::{LoggingMonitoringSystem, LoggingMonitoringConfig, LogContext, LogLevel, LoggingMonitoring, SpanResult, HealthStatus, SystemEvent, EventType, events::EventSeverity};
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the system
//!     let mut config = LoggingMonitoringConfig::default();
//!     config.tracing.sampling.rate = 1.0; // 100% sampling for example
//!     config.tracing.sampling.adaptive = false;
//!     LoggingMonitoringSystem::initialize(config.clone()).await?;
//!     
//!     // Create a system instance
//!     let system = LoggingMonitoringSystem::new(config).await?;
//!     
//!     // Log some events
//!     let context = LogContext::new("my_service".to_string())
//!         .with_request_id("req-123".to_string());
//!     
//!     system.log_info("Application started", context.clone()).await?;
//!     
//!     // Record metrics
//!     let mut tags = std::collections::HashMap::new();
//!     tags.insert("service".to_string(), "my_service".to_string());
//!     system.record_counter("requests_total", 1, tags).await?;
//!     
//!     // Start tracing
//!     let trace_info = system.start_trace("api_request").await?;
//!     let span_id = system.start_span(trace_info.trace_id, "database_query").await?;
//!     
//!     // ... perform work ...
//!     
//!     system.end_span(span_id, SpanResult {
//!         success: true,
//!         error_message: None,
//!         duration_ms: 100,
//!         metadata: std::collections::HashMap::new(),
//!     }).await?;
//!     
//!     // Report health
//!     system.report_health("my_service", HealthStatus::Healthy).await?;
//!     
//!     // Emit events
//!     let event = SystemEvent::new(
//!         EventType::System,
//!         EventSeverity::Info,
//!         "Service Status".to_string(),
//!         "Service is running normally".to_string(),
//!         "my_service".to_string(),
//!     );
//!     system.emit_event(event).await?;
//!     
//!     Ok(())
//! }
//! ```
//! 
//! ## Advanced Usage
//! 
//! ### Custom Configuration
//! 
//! ```rust
//! use logging_monitoring::{LoggingMonitoringConfig, config::{LoggingConfig, TracingConfig}};
//! 
//! let mut config = LoggingMonitoringConfig::default();
//! 
//! // Configure logging
//! config.logging.level = "DEBUG".to_string();
//! config.logging.cloudwatch.enabled = true;
//! config.logging.cloudwatch.log_group = "/myapp/logs".to_string();
//! 
//! // Configure tracing
//! config.tracing.enabled = true;
//! config.tracing.xray.enabled = true;
//! config.tracing.xray.service_name = "my-service".to_string();
//! 
//! // Configure metrics
//! config.metrics.enabled = true;
//! config.metrics.prometheus.enabled = true;
//! config.metrics.cloudwatch.enabled = true;
//! ```
//! 
//! ### Health Monitoring
//! 
//! ```rust
//! use logging_monitoring::{LoggingMonitoringSystem, LoggingMonitoringConfig, config::ServiceHealthConfig, LoggingMonitoring};
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = LoggingMonitoringConfig::default();
//!     let system = LoggingMonitoringSystem::new(config).await?;
//!     
//!     // Add custom health checks
//!     system.health_manager().add_service(
//!         "database".to_string(),
//!         ServiceHealthConfig {
//!             endpoint: "/health".to_string(),
//!             expected_response_time: std::time::Duration::from_secs(5),
//!             retry_count: 3,
//!             circuit_breaker: true,
//!             failure_threshold: 5,
//!         }
//!     ).await?;
//! 
//!     // Check system health
//!     let health = system.get_system_health().await?;
//!     println!("System health: {:?}", health.overall_status);
//!     
//!     Ok(())
//! }
//! ```
//! 
//! ### Event Management
//! 
//! ```rust
//! use logging_monitoring::{LoggingMonitoringSystem, LoggingMonitoringConfig, events::{EventSeverity, NotificationPriority, NotificationChannel}, LoggingMonitoring};
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = LoggingMonitoringConfig::default();
//!     let system = LoggingMonitoringSystem::new(config).await?;
//!     
//!     // Create alerts
//!     let alert = system.event_manager().alert()
//!         .title("High CPU Usage".to_string())
//!         .message("CPU usage is above 90%".to_string())
//!         .severity(EventSeverity::High)
//!         .source("monitoring".to_string())
//!         .category("performance".to_string())
//!         .build()?;
//! 
//!     system.emit_alert(alert).await?;
//! 
//!     // Create notifications
//!     let notification = system.event_manager().notification()
//!         .title("Deployment Complete".to_string())
//!         .message("New version deployed successfully".to_string())
//!         .priority(NotificationPriority::Normal)
//!         .channel(NotificationChannel::Slack)
//!         .build()?;
//! 
//!     system.emit_notification(notification).await?;
//!     
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod logging;
pub mod monitoring;
pub mod tracing;
pub mod metrics;
pub mod health;
pub mod events;
pub mod config;

pub use error::{LoggingMonitoringError, Result};
pub use logging::{LogManager, LogLevel, LogContext, StructuredEvent};
pub use monitoring::MonitoringManager;
pub use tracing::{TraceManager, TraceSpan, SpanStatus};
pub use metrics::{MetricsManager, Tags};
pub use health::{HealthManager, HealthStatus, SystemHealthStatus};
pub use events::{EventManager, SystemEvent, Alert, Notification, EventType};
pub use config::LoggingMonitoringConfig;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;


/// Main interface for logging and monitoring operations
#[async_trait]
pub trait LoggingMonitoring {
    // Logging operations
    async fn log_info(&self, message: &str, context: LogContext) -> Result<()>;
    async fn log_warn(&self, message: &str, context: LogContext) -> Result<()>;
    async fn log_error(&self, error: &LoggingMonitoringError, context: LogContext) -> Result<()>;
    async fn log_debug(&self, message: &str, context: LogContext) -> Result<()>;
    
    // Structured logging
    async fn log_structured(&self, level: LogLevel, event: StructuredEvent) -> Result<()>;
    async fn log_audit(&self, action: AuditAction, user: UserId, context: AuditContext) -> Result<()>;
    
    // Tracing operations
    async fn start_trace(&self, operation: &str) -> Result<TraceInfo>;
    async fn start_span(&self, parent: TraceId, operation: &str) -> Result<SpanId>;
    async fn end_span(&self, span: SpanId, result: SpanResult) -> Result<()>;
    async fn add_trace_annotation(&self, trace: TraceId, key: &str, value: &str) -> Result<()>;
    
    // Metrics operations
    async fn record_counter(&self, name: &str, value: i64, tags: Tags) -> Result<()>;
    async fn record_gauge(&self, name: &str, value: f64, tags: Tags) -> Result<()>;
    async fn record_histogram(&self, name: &str, value: f64, tags: Tags) -> Result<()>;
    async fn record_timer(&self, name: &str, duration: Duration, tags: Tags) -> Result<()>;
    
    // Event operations
    async fn emit_event(&self, event: SystemEvent) -> Result<()>;
    async fn emit_alert(&self, alert: Alert) -> Result<()>;
    async fn emit_notification(&self, notification: Notification) -> Result<()>;
    
    // Health monitoring
    async fn report_health(&self, service: &str, status: HealthStatus) -> Result<()>;
    async fn get_service_health(&self, service: &str) -> Result<HealthStatus>;
    async fn get_system_health(&self) -> Result<SystemHealthStatus>;
}

/// Main implementation of the logging and monitoring system
#[derive(Clone)]
pub struct LoggingMonitoringSystem {
    log_manager: LogManager,
    trace_manager: TraceManager,
    metrics_manager: MetricsManager,
    health_manager: HealthManager,
    event_manager: EventManager,
    monitoring_manager: MonitoringManager,
}

impl LoggingMonitoringSystem {
    /// Create a new logging and monitoring system with the given configuration
    pub async fn new(config: LoggingMonitoringConfig) -> Result<Self> {
        let log_manager = LogManager::new(config.logging.clone()).await?;
        let trace_manager = TraceManager::new(config.tracing.clone()).await?;
        let metrics_manager = MetricsManager::new(config.metrics.clone()).await?;
        let health_manager = HealthManager::new(config.health.clone()).await?;
        let event_manager = EventManager::new(config.events.clone()).await?;
        let monitoring_manager = MonitoringManager::new(config.monitoring.clone()).await?;

        Ok(Self {
            log_manager,
            trace_manager,
            metrics_manager,
            health_manager,
            event_manager,
            monitoring_manager,
        })
    }

    /// Initialize the global logging and monitoring system
    pub async fn initialize(config: LoggingMonitoringConfig) -> Result<()> {
        let _system = Self::new(config).await?;
        
        // Set up global tracing subscriber
        tracing_subscriber::fmt()
            .with_target(false)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_file(true)
            .with_line_number(true)
            .with_ansi(false)
            .with_timer(tracing_subscriber::fmt::time::UtcTime::rfc_3339())
            .init();

        Ok(())
    }

    /// Get a reference to the log manager
    pub fn log_manager(&self) -> &LogManager {
        &self.log_manager
    }

    /// Get a reference to the trace manager
    pub fn trace_manager(&self) -> &TraceManager {
        &self.trace_manager
    }

    /// Get a reference to the metrics manager
    pub fn metrics_manager(&self) -> &MetricsManager {
        &self.metrics_manager
    }

    /// Get a reference to the health manager
    pub fn health_manager(&self) -> &HealthManager {
        &self.health_manager
    }

    /// Get a reference to the event manager
    pub fn event_manager(&self) -> &EventManager {
        &self.event_manager
    }

    /// Get a reference to the monitoring manager
    pub fn monitoring_manager(&self) -> &MonitoringManager {
        &self.monitoring_manager
    }

    /// Shutdown the logging and monitoring system gracefully
    pub async fn shutdown(&self) -> Result<()> {
        self.log_manager.shutdown().await?;
        self.trace_manager.shutdown().await?;
        self.metrics_manager.shutdown().await?;
        self.health_manager.shutdown().await?;
        self.event_manager.shutdown().await?;
        self.monitoring_manager.shutdown().await?;
        Ok(())
    }
}

#[async_trait]
impl LoggingMonitoring for LoggingMonitoringSystem {
    async fn log_info(&self, message: &str, context: LogContext) -> Result<()> {
        self.log_manager.log_info(message, context).await
    }

    async fn log_warn(&self, message: &str, context: LogContext) -> Result<()> {
        self.log_manager.log_warn(message, context).await
    }

    async fn log_error(&self, error: &LoggingMonitoringError, context: LogContext) -> Result<()> {
        self.log_manager.log_error(error, context).await
    }

    async fn log_debug(&self, message: &str, context: LogContext) -> Result<()> {
        self.log_manager.log_debug(message, context).await
    }

    async fn log_structured(&self, level: LogLevel, event: StructuredEvent) -> Result<()> {
        self.log_manager.log_structured(level, event).await
    }

    async fn log_audit(&self, action: AuditAction, user: UserId, context: AuditContext) -> Result<()> {
        self.log_manager.log_audit(action, user, context).await
    }

    async fn start_trace(&self, operation: &str) -> Result<TraceInfo> {
        self.trace_manager.start_trace(operation).await
    }

    async fn start_span(&self, parent: TraceId, operation: &str) -> Result<SpanId> {
        self.trace_manager.start_span(parent, operation).await
    }

    async fn end_span(&self, span: SpanId, result: SpanResult) -> Result<()> {
        self.trace_manager.end_span(span, result).await
    }

    async fn add_trace_annotation(&self, trace: TraceId, key: &str, value: &str) -> Result<()> {
        self.trace_manager.add_annotation(trace, key, value).await
    }

    async fn record_counter(&self, name: &str, value: i64, tags: Tags) -> Result<()> {
        self.metrics_manager.record_counter(name, value, tags).await
    }

    async fn record_gauge(&self, name: &str, value: f64, tags: Tags) -> Result<()> {
        self.metrics_manager.record_gauge(name, value, tags).await
    }

    async fn record_histogram(&self, name: &str, value: f64, tags: Tags) -> Result<()> {
        self.metrics_manager.record_histogram(name, value, tags).await
    }

    async fn record_timer(&self, name: &str, duration: Duration, tags: Tags) -> Result<()> {
        self.metrics_manager.record_timer(name, duration, tags).await
    }

    async fn emit_event(&self, event: SystemEvent) -> Result<()> {
        self.event_manager.emit_event(event).await
    }

    async fn emit_alert(&self, alert: Alert) -> Result<()> {
        self.event_manager.emit_alert(alert).await
    }

    async fn emit_notification(&self, notification: Notification) -> Result<()> {
        self.event_manager.emit_notification(notification).await
    }

    async fn report_health(&self, service: &str, status: HealthStatus) -> Result<()> {
        self.health_manager.report_health(service, status).await
    }

    async fn get_service_health(&self, service: &str) -> Result<HealthStatus> {
        self.health_manager.get_service_health(service).await
    }

    async fn get_system_health(&self) -> Result<SystemHealthStatus> {
        self.health_manager.get_system_health().await
    }
}

// Additional types for the interface
pub type UserId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditAction {
    pub action: String,
    pub resource: String,
    pub details: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditContext {
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub session_id: Option<String>,
    pub additional_data: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanResult {
    pub success: bool,
    pub error_message: Option<String>,
    pub duration_ms: u64,
    pub metadata: HashMap<String, String>,
}

// Re-export commonly used types
pub type TraceId = uuid::Uuid;
pub type SpanId = uuid::Uuid;

/// Re-export TraceInfo from tracing module
pub use tracing::TraceInfo;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_creation() {
        let config = LoggingMonitoringConfig::default();
        let system = LoggingMonitoringSystem::new(config).await;
        assert!(system.is_ok());
    }

    #[tokio::test]
    async fn test_basic_logging() {
        let config = LoggingMonitoringConfig::default();
        let system = LoggingMonitoringSystem::new(config).await.unwrap();
        
        let context = LogContext::new("test_service".to_string());
        let result = system.log_info("Test message", context).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let config = LoggingMonitoringConfig::default();
        let system = LoggingMonitoringSystem::new(config).await.unwrap();
        
        let mut tags = HashMap::new();
        tags.insert("test".to_string(), "value".to_string());
        
        let result = system.record_counter("test_counter", 1, tags).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_tracing() {
        let mut config = LoggingMonitoringConfig::default();
        config.tracing.sampling.rate = 1.0; // 100% sampling for tests
        config.tracing.sampling.adaptive = false; // Disable adaptive sampling for tests
        let system = LoggingMonitoringSystem::new(config).await.unwrap();
        
        let trace_info = system.start_trace("test_operation").await;
        assert!(trace_info.is_ok());
        
        let trace_info = trace_info.unwrap();
        let span_id = system.start_span(trace_info.trace_id, "test_span").await;
        assert!(span_id.is_ok());
        
        let span_id = span_id.unwrap();
        let result = SpanResult {
            success: true,
            error_message: None,
            duration_ms: 100,
            metadata: HashMap::new(),
        };
        
        let end_result = system.end_span(span_id, result).await;
        assert!(end_result.is_ok());
    }

    #[tokio::test]
    async fn test_health_monitoring() {
        let config = LoggingMonitoringConfig::default();
        let system = LoggingMonitoringSystem::new(config).await.unwrap();
        
        let result = system.report_health("test_service", HealthStatus::Healthy).await;
        assert!(result.is_ok());
        
        let health = system.get_service_health("test_service").await;
        assert!(health.is_ok());
        assert_eq!(health.unwrap(), HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_event_emission() {
        let config = LoggingMonitoringConfig::default();
        let system = LoggingMonitoringSystem::new(config).await.unwrap();
        
        let event = SystemEvent::new(
            EventType::System,
            events::EventSeverity::Info,
            "Test Event".to_string(),
            "This is a test event".to_string(),
            "test_service".to_string(),
        );
        
        let result = system.emit_event(event).await;
        assert!(result.is_ok());
    }
} 