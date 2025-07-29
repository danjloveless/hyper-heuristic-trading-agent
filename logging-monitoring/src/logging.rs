//! Logging functionality for the monitoring system

use crate::error::{LoggingMonitoringError, Result};
use crate::config::{LoggingConfig, CloudWatchLogConfig, LocalFileLogConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn, error, debug};
use aws_sdk_cloudwatchlogs::Client as CloudWatchLogsClient;
use aws_sdk_cloudwatchlogs::types::InputLogEvent;

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

impl From<LogLevel> for tracing::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

/// Log context with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogContext {
    pub service_name: String,
    pub request_id: Option<String>,
    pub trace_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub additional_fields: HashMap<String, Value>,
}

impl LogContext {
    /// Create a new log context
    pub fn new(service_name: String) -> Self {
        Self {
            service_name,
            request_id: None,
            trace_id: None,
            user_id: None,
            session_id: None,
            timestamp: Utc::now(),
            additional_fields: HashMap::new(),
        }
    }

    /// Add a request ID to the context
    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }

    /// Add a trace ID to the context
    pub fn with_trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = Some(trace_id);
        self
    }

    /// Add a user ID to the context
    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Add a session ID to the context
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Add additional fields to the context
    pub fn with_field(mut self, key: String, value: Value) -> Self {
        self.additional_fields.insert(key, value);
        self
    }
}

/// Structured log event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredEvent {
    pub event_type: String,
    pub level: LogLevel,
    pub message: String,
    pub context: LogContext,
    pub data: HashMap<String, Value>,
}

/// Log entry for internal processing
#[derive(Debug, Clone)]
struct LogEntry {
    level: LogLevel,
    message: String,
    context: LogContext,
    timestamp: DateTime<Utc>,
    structured_data: Option<StructuredEvent>,
}

/// Main log manager
#[derive(Clone)]
pub struct LogManager {
    config: LoggingConfig,
    cloudwatch_client: Option<CloudWatchLogsClient>,
    log_sender: mpsc::Sender<LogEntry>,
    buffer: Arc<Mutex<Vec<LogEntry>>>,
    shutdown: Arc<AtomicBool>,
}

impl LogManager {
    /// Create a new log manager
    pub async fn new(config: LoggingConfig) -> Result<Self> {
        let (log_sender, log_receiver) = mpsc::channel(config.performance.channel_capacity);
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        // Initialize CloudWatch client if enabled
        let cloudwatch_client = if config.cloudwatch.enabled {
            let aws_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
                .region(aws_config::Region::new(config.cloudwatch.region.clone()))
                .load()
                .await;
            Some(CloudWatchLogsClient::new(&aws_config))
        } else {
            None
        };

        let manager = Self {
            config: config.clone(),
            cloudwatch_client,
            log_sender,
            buffer: buffer.clone(),
            shutdown: shutdown.clone(),
        };

        // Start background processing with proper task handle management
        manager.start_background_processing(log_receiver, buffer, shutdown, config).await;

        Ok(manager)
    }

    /// Start background log processing with proper lifecycle management
    async fn start_background_processing(
        &self,
        mut receiver: mpsc::Receiver<LogEntry>,
        buffer: Arc<Mutex<Vec<LogEntry>>>,
        shutdown: Arc<AtomicBool>,
        config: LoggingConfig,
    ) {
        let cloudwatch_client = self.cloudwatch_client.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.buffer.flush_interval);
            let mut pending_entries = Vec::new();
            
            loop {
                tokio::select! {
                    // Handle incoming log entries
                    result = receiver.recv() => {
                        match result {
                            Some(entry) => {
                                pending_entries.push(entry);
                                
                                // Flush if buffer is full
                                if pending_entries.len() >= config.buffer.size {
                                    Self::flush_entries(
                                        &mut pending_entries,
                                        &buffer,
                                        &cloudwatch_client,
                                        &config
                                    ).await;
                                }
                            },
                            None => {
                                // Channel closed, flush remaining entries and exit
                                if !pending_entries.is_empty() {
                                    Self::flush_entries(
                                        &mut pending_entries,
                                        &buffer,
                                        &cloudwatch_client,
                                        &config
                                    ).await;
                                }
                                break;
                            }
                        }
                    },
                    
                    // Periodic flush
                    _ = interval.tick() => {
                        if !pending_entries.is_empty() {
                            Self::flush_entries(
                                &mut pending_entries,
                                &buffer,
                                &cloudwatch_client,
                                &config
                            ).await;
                        }
                    },
                    
                    // Check for shutdown signal
                    _ = async {
                        while !shutdown.load(Ordering::Relaxed) {
                            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                        }
                    } => {
                        // Shutdown requested, flush remaining entries and exit
                        if !pending_entries.is_empty() {
                            Self::flush_entries(
                                &mut pending_entries,
                                &buffer,
                                &cloudwatch_client,
                                &config
                            ).await;
                        }
                        break;
                    }
                }
            }
        });
    }

    /// Flush log entries to storage
    async fn flush_entries(
        pending_entries: &mut Vec<LogEntry>,
        buffer: &Arc<Mutex<Vec<LogEntry>>>,
        cloudwatch_client: &Option<CloudWatchLogsClient>,
        config: &LoggingConfig,
    ) {
        if pending_entries.is_empty() {
            return;
        }

        // Add to local buffer and process
        {
            let mut buffer_guard = buffer.lock().await;
            buffer_guard.extend(pending_entries.drain(..));
            
            // Process the buffer using existing method
            if let Err(e) = Self::process_log_buffer(config, cloudwatch_client, &mut buffer_guard).await {
                eprintln!("Failed to process log buffer: {:?}", e);
            }
        }
    }

    /// Process log buffer
    async fn process_log_buffer(
        config: &LoggingConfig,
        cloudwatch_client: &Option<CloudWatchLogsClient>,
        buffer: &mut Vec<LogEntry>,
    ) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Convert to structured format
        let log_events: Vec<StructuredLogEvent> = buffer
            .drain(..)
            .map(|entry| Self::convert_to_structured_event(entry))
            .collect();

        // Send to CloudWatch if enabled
        if let Some(client) = cloudwatch_client {
            if config.cloudwatch.enabled {
                Self::send_to_cloudwatch(client, &config.cloudwatch, &log_events).await?;
            }
        }

        // Write to local file if enabled
        if config.local_file.enabled {
            Self::write_to_local_file(&config.local_file, &log_events).await?;
        }

        // Output to console/stderr
        Self::output_to_console(&log_events);

        Ok(())
    }

    /// Convert log entry to structured event
    fn convert_to_structured_event(entry: LogEntry) -> StructuredLogEvent {
        StructuredLogEvent {
            timestamp: entry.timestamp,
            level: entry.level,
            message: entry.message,
            service_name: entry.context.service_name,
            request_id: entry.context.request_id,
            trace_id: entry.context.trace_id,
            user_id: entry.context.user_id,
            session_id: entry.context.session_id,
            additional_fields: entry.context.additional_fields,
            structured_data: entry.structured_data,
        }
    }

    /// Send logs to CloudWatch
    async fn send_to_cloudwatch(
        client: &CloudWatchLogsClient,
        config: &CloudWatchLogConfig,
        events: &[StructuredLogEvent],
    ) -> Result<()> {
        let log_events: Vec<InputLogEvent> = events
            .iter()
            .map(|event| {
                let json = serde_json::to_string(&event)
                    .map_err(|e| LoggingMonitoringError::JsonSerializationError { message: e.to_string() })?;
                
                InputLogEvent::builder()
                    .timestamp(event.timestamp.timestamp_millis())
                    .message(json)
                    .build()
                    .map_err(|e| LoggingMonitoringError::CloudWatchLogFailed { message: e.to_string() })
            })
            .collect::<Result<Vec<_>>>()?;

        // Send in batches
        for chunk in log_events.chunks(config.batch_size) {
            let mut request = client
                .put_log_events()
                .log_group_name(&config.log_group)
                .log_stream_name(&config.log_stream);
            
            for event in chunk {
                request = request.log_events(event.clone());
            }
            
            let result = request.send().await;

            if let Err(e) = result {
                return Err(LoggingMonitoringError::CloudWatchLogFailed {
                    message: e.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Write logs to local file
    async fn write_to_local_file(
        config: &LocalFileLogConfig,
        events: &[StructuredLogEvent],
    ) -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.file_path)
            .map_err(|e| LoggingMonitoringError::FileSystemError {
                operation: "open_log_file".to_string(),
                message: e.to_string(),
            })?;

        let mut writer = std::io::BufWriter::new(file);

        for event in events {
            let json = serde_json::to_string(&event)
                .map_err(|e| LoggingMonitoringError::JsonSerializationError { message: e.to_string() })?;
            
            writeln!(writer, "{}", json)
                .map_err(|e| LoggingMonitoringError::IoError { message: e.to_string() })?;
        }

        writer.flush()
            .map_err(|e| LoggingMonitoringError::IoError { message: e.to_string() })?;

        Ok(())
    }

    /// Output logs to console
    fn output_to_console(events: &[StructuredLogEvent]) {
        for event in events {
            let log_message = format!(
                "[{}] {} - {} - {}",
                event.timestamp.format("%Y-%m-%d %H:%M:%S%.3f"),
                event.level,
                event.service_name,
                event.message
            );

            match event.level {
                LogLevel::Debug => debug!("{}", log_message),
                LogLevel::Info => info!("{}", log_message),
                LogLevel::Warn => warn!("{}", log_message),
                LogLevel::Error => error!("{}", log_message),
            }
        }
    }

    /// Log an info message
    pub async fn log_info(&self, message: &str, context: LogContext) -> Result<()> {
        self.log(LogLevel::Info, message, context, None).await
    }

    /// Log a warning message
    pub async fn log_warn(&self, message: &str, context: LogContext) -> Result<()> {
        self.log(LogLevel::Warn, message, context, None).await
    }

    /// Log an error message
    pub async fn log_error(&self, error: &LoggingMonitoringError, context: LogContext) -> Result<()> {
        let error_message = format!("Error: {}", error);
        let mut error_context = context;
        
        // Add error context to additional fields
        for (key, value) in error.context() {
            error_context.additional_fields.insert(
                format!("error_{}", key),
                Value::String(value),
            );
        }

        self.log(LogLevel::Error, &error_message, error_context, None).await
    }

    /// Log a debug message
    pub async fn log_debug(&self, message: &str, context: LogContext) -> Result<()> {
        self.log(LogLevel::Debug, message, context, None).await
    }

    /// Log a structured event
    pub async fn log_structured(&self, level: LogLevel, event: StructuredEvent) -> Result<()> {
        let context = event.context.clone();
        let message = event.message.clone();
        self.log(level, &message, context, Some(event)).await
    }

    /// Log an audit event
    pub async fn log_audit(&self, action: crate::AuditAction, user: crate::UserId, context: crate::AuditContext) -> Result<()> {
        let mut audit_context = LogContext::new("audit".to_string())
            .with_user_id(user)
            .with_session_id(context.session_id.unwrap_or_default());

        // Add audit-specific fields
        let action_str = action.action.clone();
        let resource_str = action.resource.clone();
        
        audit_context.additional_fields.insert(
            "audit_action".to_string(),
            Value::String(action_str.clone()),
        );
        audit_context.additional_fields.insert(
            "audit_resource".to_string(),
            Value::String(resource_str.clone()),
        );
        audit_context.additional_fields.insert(
            "audit_details".to_string(),
            serde_json::to_value(action.details)
                .map_err(|e| LoggingMonitoringError::JsonSerializationError { message: e.to_string() })?,
        );

        if let Some(ip) = context.ip_address {
            audit_context.additional_fields.insert("ip_address".to_string(), Value::String(ip));
        }
        if let Some(user_agent) = context.user_agent {
            audit_context.additional_fields.insert("user_agent".to_string(), Value::String(user_agent));
        }

        let structured_event = StructuredEvent {
            event_type: "audit".to_string(),
            level: LogLevel::Info,
            message: format!("Audit: {} on {}", action_str, resource_str),
            context: audit_context.clone(),
            data: audit_context.additional_fields.clone(),
        };

        self.log_structured(LogLevel::Info, structured_event).await
    }

    /// Internal log method
    async fn log(
        &self,
        level: LogLevel,
        message: &str,
        context: LogContext,
        structured_data: Option<StructuredEvent>,
    ) -> Result<()> {
        // Check if we should sample this log
        if self.config.performance.enable_sampling {
            let sample_rate = self.config.performance.sampling_rate;
            if fastrand::f64() > sample_rate {
                return Ok(());
            }
        }

        let entry = LogEntry {
            level,
            message: message.to_string(),
            context,
            timestamp: Utc::now(),
            structured_data,
        };

        self.log_sender
            .send(entry)
            .await
            .map_err(|_| LoggingMonitoringError::LogWriteFailed {
                message: "Failed to send log entry to background processor".to_string(),
            })?;

        Ok(())
    }

    /// Get environment filter for tracing subscriber
    pub fn get_env_filter(&self) -> String {
        self.config.env_filter.clone()
    }

    /// Add proper shutdown mechanism
    pub async fn shutdown(&self) -> Result<()> {
        // Signal shutdown
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Give background task time to finish
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        
        Ok(())
    }
}

/// Structured log event for internal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StructuredLogEvent {
    timestamp: DateTime<Utc>,
    level: LogLevel,
    message: String,
    service_name: String,
    request_id: Option<String>,
    trace_id: Option<String>,
    user_id: Option<String>,
    session_id: Option<String>,
    additional_fields: HashMap<String, Value>,
    structured_data: Option<StructuredEvent>,
}

impl Drop for LogManager {
    fn drop(&mut self) {
        // Set shutdown flag without blocking operations
        // The background task will check this flag and exit gracefully
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_log_manager_creation() {
        let config = LoggingConfig::default();
        let manager = LogManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_log_info() {
        let config = LoggingConfig::default();
        let manager = LogManager::new(config).await.unwrap();
        
        let context = LogContext::new("test_service".to_string());
        let result = manager.log_info("Test message", context).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_log_error() {
        let config = LoggingConfig::default();
        let manager = LogManager::new(config).await.unwrap();
        
        let context = LogContext::new("test_service".to_string());
        let error = LoggingMonitoringError::Configuration {
            message: "Test error".to_string(),
        };
        
        let result = manager.log_error(&error, context).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_structured_logging() {
        let config = LoggingConfig::default();
        let manager = LogManager::new(config).await.unwrap();
        
        let context = LogContext::new("test_service".to_string());
        let event = StructuredEvent {
            event_type: "test_event".to_string(),
            level: LogLevel::Info,
            message: "Test structured event".to_string(),
            context: context.clone(),
            data: HashMap::new(),
        };
        
        let result = manager.log_structured(LogLevel::Info, event).await;
        assert!(result.is_ok());
    }
} 