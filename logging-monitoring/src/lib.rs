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
pub use monitoring::{MonitoringManager, MetricType, MetricValue, MetricData};
pub use tracing::{TraceManager, TraceId, SpanId, TraceSpan, SpanStatus};
pub use metrics::{MetricsManager, Tags};
pub use health::{HealthManager, HealthStatus, SystemHealthStatus};
pub use events::{EventManager, SystemEvent, Alert, Notification, EventType};
pub use config::LoggingMonitoringConfig;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use chrono::{DateTime, Utc};
use uuid::Uuid;

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
    async fn start_trace(&self, operation: &str) -> Result<TraceId>;
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
        let system = Self::new(config).await?;
        
        // Set up global tracing subscriber
        tracing_subscriber::fmt()
            .with_env_filter(&system.log_manager.get_env_filter())
            .with_target(false)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_file(true)
            .with_line_number(true)
            .with_ansi(false)
            .with_timer(tracing_subscriber::fmt::time::UtcTime::rfc_3339())
            .json()
            .init();

        Ok(())
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

    async fn start_trace(&self, operation: &str) -> Result<TraceId> {
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
pub use uuid::Uuid as TraceId;
pub use uuid::Uuid as SpanId; 