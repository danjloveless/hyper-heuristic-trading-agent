//! Basic usage example for the logging-monitoring system
//! 
//! This example demonstrates the most common use cases for logging and monitoring.

use logging_monitoring::{
    LoggingMonitoringSystem, LoggingMonitoringConfig, LogContext, LogLevel,
    SpanResult, HealthStatus, SystemEvent, EventType, LoggingMonitoring,
};
use logging_monitoring::events::EventSeverity;
use std::collections::HashMap;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Basic Logging & Monitoring Example");
    
    // Create a simple configuration
    let config = LoggingMonitoringConfig::default();
    
    // Initialize the system
    LoggingMonitoringSystem::initialize(config.clone()).await?;
    
    // Create the system instance
    let system = LoggingMonitoringSystem::new(config).await?;
    
    // Basic logging
    let context = LogContext::new("my_app".to_string())
        .with_request_id("req-123".to_string());
    
    system.log_info("Application started", context.clone()).await?;
    system.log_debug("Debug information", context.clone()).await?;
    
    // Structured logging with custom data
    let mut data = HashMap::new();
    data.insert("user_id".to_string(), serde_json::Value::String("user-456".to_string()));
    data.insert("action".to_string(), serde_json::Value::String("login".to_string()));
    
    let structured_event = logging_monitoring::StructuredEvent {
        event_type: "user_action".to_string(),
        level: LogLevel::Info,
        message: "User logged in successfully".to_string(),
        context: context.clone(),
        data,
    };
    
    system.log_structured(LogLevel::Info, structured_event).await?;
    
    // Tracing
    let trace_id = system.start_trace("api_request").await?;
    system.add_trace_annotation(trace_id, "endpoint", "/api/users").await?;
    
    let span_id = system.start_span(trace_id, "database_query").await?;
    
    // Simulate some work
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    system.end_span(span_id, SpanResult {
        success: true,
        error_message: None,
        duration_ms: 100,
        metadata: HashMap::new(),
    }).await?;
    
    // End the trace
    system.end_span(trace_id, SpanResult {
        success: true,
        error_message: None,
        duration_ms: 100,
        metadata: HashMap::new(),
    }).await?;
    
    // Metrics
    let mut tags = HashMap::new();
    tags.insert("endpoint".to_string(), "/api/users".to_string());
    tags.insert("method".to_string(), "GET".to_string());
    
    system.record_counter("http_requests_total", 1, tags.clone()).await?;
    system.record_gauge("active_connections", 25.0, tags.clone()).await?;
    system.record_histogram("response_time_ms", 150.0, tags).await?;
    
    // Health monitoring
    system.report_health("my_app", HealthStatus::Healthy).await?;
    
    let health = system.get_service_health("my_app").await?;
    println!("📊 Service health: {:?}", health);
    
    // Events
    let event = SystemEvent::new(
        EventType::System,
        EventSeverity::Info,
        "Application Status".to_string(),
        "Application is running normally".to_string(),
        "my_app".to_string(),
    );
    
    system.emit_event(event).await?;
    
    // Error logging
    let error = logging_monitoring::LoggingMonitoringError::NetworkError {
        message: "Connection timeout".to_string(),
    };
    
    system.log_error(&error, context).await?;
    
    println!("✅ Basic example completed successfully!");
    
    // Shutdown
    system.shutdown().await?;
    
    Ok(())
} 