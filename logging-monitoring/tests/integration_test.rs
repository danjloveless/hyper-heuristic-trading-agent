//! Integration tests for the logging-monitoring system

use logging_monitoring::{
    LoggingMonitoringSystem, LoggingMonitoringConfig, LogContext, LogLevel,
    StructuredEvent, SpanResult, HealthStatus, SystemEvent, EventType,
    AuditAction, AuditContext, LoggingMonitoring,
};
use logging_monitoring::events::EventSeverity;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_full_system_integration() {
    // Create a test configuration
    let mut config = LoggingMonitoringConfig::default();
    config.logging.cloudwatch.enabled = false; // Disable for testing
    config.logging.local_file.enabled = false; // Disable for testing
    config.tracing.xray.enabled = false; // Disable for testing
    config.tracing.sampling.rate = 1.0; // 100% sampling for tests
    config.tracing.sampling.adaptive = false; // Disable adaptive sampling for tests
    config.metrics.cloudwatch.enabled = false; // Disable for testing
    config.events.sns.enabled = false; // Disable for testing
    
    // Create the system without initializing global subscriber
    let system = LoggingMonitoringSystem::new(config).await.unwrap();
    
    // Test 1: Basic logging
    let context = LogContext::new("integration_test".to_string())
        .with_request_id("test-req-123".to_string());
    
    system.log_info("Integration test started", context.clone()).await.unwrap();
    system.log_debug("Debug message", context.clone()).await.unwrap();
    
    // Test 2: Structured logging
    let mut data = HashMap::new();
    data.insert("test_key".to_string(), serde_json::Value::String("test_value".to_string()));
    
    let structured_event = StructuredEvent {
        event_type: "test_event".to_string(),
        level: LogLevel::Info,
        message: "Test structured event".to_string(),
        context: context.clone(),
        data,
    };
    
    system.log_structured(LogLevel::Info, structured_event).await.unwrap();
    
    // Test 3: Tracing
    let trace_id = system.start_trace("integration_test_trace").await.unwrap();
    system.add_trace_annotation(trace_id, "test_annotation", "test_value").await.unwrap();
    
    let span_id = system.start_span(trace_id, "test_span").await.unwrap();
    sleep(Duration::from_millis(10)).await; // Simulate work
    
    system.end_span(span_id, SpanResult {
        success: true,
        error_message: None,
        duration_ms: 10,
        metadata: HashMap::new(),
    }).await.unwrap();
    
    system.end_span(trace_id, SpanResult {
        success: true,
        error_message: None,
        duration_ms: 10,
        metadata: HashMap::new(),
    }).await.unwrap();
    
    // Test 4: Metrics
    let mut tags = HashMap::new();
    tags.insert("test_tag".to_string(), "test_value".to_string());
    
    system.record_counter("test_counter", 1, tags.clone()).await.unwrap();
    system.record_gauge("test_gauge", 42.0, tags.clone()).await.unwrap();
    system.record_histogram("test_histogram", 100.0, tags.clone()).await.unwrap();
    system.record_timer("test_timer", Duration::from_millis(50), tags).await.unwrap();
    
    // Test 5: Health monitoring
    system.report_health("test_service", HealthStatus::Healthy).await.unwrap();
    
    let health = system.get_service_health("test_service").await.unwrap();
    assert_eq!(health, HealthStatus::Healthy);
    
    let system_health = system.get_system_health().await.unwrap();
    assert_eq!(system_health.overall_status, HealthStatus::Healthy);
    
    // Test 6: Events
    let event = SystemEvent::new(
        EventType::System,
        EventSeverity::Info,
        "Integration Test Event".to_string(),
        "This is a test event".to_string(),
        "integration_test".to_string(),
    );
    
    system.emit_event(event).await.unwrap();
    
    // Test 7: Audit logging
    let audit_action = AuditAction {
        action: "test_action".to_string(),
        resource: "test_resource".to_string(),
        details: HashMap::new(),
    };
    
    let audit_context = AuditContext {
        ip_address: Some("127.0.0.1".to_string()),
        user_agent: Some("IntegrationTest/1.0".to_string()),
        session_id: Some("test-session".to_string()),
        additional_data: HashMap::new(),
    };
    
    system.log_audit(audit_action, "test_user".to_string(), audit_context).await.unwrap();
    
    // Test 8: Error logging
    let error = logging_monitoring::LoggingMonitoringError::NetworkError {
        message: "Test network error".to_string(),
    };
    
    system.log_error(&error, context.clone()).await.unwrap();
    
    // Test 9: Get aggregated metrics
    let aggregated_metrics = system.metrics_manager().get_aggregated_metrics().await;
    assert!(!aggregated_metrics.is_empty());
    
    // Test 10: Get monitoring data
    let monitoring_data = system.monitoring_manager().get_monitoring_data().await.unwrap();
    assert!(monitoring_data.timestamp > chrono::Utc::now() - chrono::Duration::minutes(1));
    
    // Test 11: Get monitoring summary
    let summary = system.monitoring_manager().get_monitoring_summary().await.unwrap();
    assert!(summary.alert_count() >= 0);
    
    // Test 12: Get health statistics
    let health_stats = system.health_manager().get_health_statistics().await;
    assert!(health_stats.total_checks >= 0);
    
    // Test 13: Shutdown
    system.shutdown().await.unwrap();
    
    system.log_info("Integration test completed successfully", context).await.unwrap();
}

#[tokio::test]
async fn test_error_handling() {
    let mut config = LoggingMonitoringConfig::default();
    config.logging.cloudwatch.enabled = false;
    config.tracing.xray.enabled = false;
    config.tracing.sampling.rate = 1.0; // 100% sampling for tests
    config.tracing.sampling.adaptive = false; // Disable adaptive sampling for tests
    config.metrics.cloudwatch.enabled = false;
    config.events.sns.enabled = false;
    
    let system = LoggingMonitoringSystem::new(config).await.unwrap();
    
    let context = LogContext::new("error_test".to_string());
    
    // Test various error scenarios
    let errors = vec![
        logging_monitoring::LoggingMonitoringError::Configuration {
            message: "Test configuration error".to_string(),
        },
        logging_monitoring::LoggingMonitoringError::NetworkError {
            message: "Test network error".to_string(),
        },
        logging_monitoring::LoggingMonitoringError::ValidationError {
            field: "test_field".to_string(),
            message: "Test validation error".to_string(),
        },
    ];
    
    for error in errors {
        system.log_error(&error, context.clone()).await.unwrap();
    }
    
    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_concurrent_operations() {
    let mut config = LoggingMonitoringConfig::default();
    config.logging.cloudwatch.enabled = false;
    config.tracing.xray.enabled = false;
    config.tracing.sampling.rate = 1.0; // 100% sampling for tests
    config.tracing.sampling.adaptive = false; // Disable adaptive sampling for tests
    config.metrics.cloudwatch.enabled = false;
    config.events.sns.enabled = false;
    
    let system = LoggingMonitoringSystem::new(config).await.unwrap();
    
    // Test concurrent logging operations
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let system_clone = system.clone();
        let handle = tokio::spawn(async move {
            let context = LogContext::new(format!("concurrent_test_{}", i));
            system_clone.log_info(&format!("Concurrent message {}", i), context).await.unwrap();
        });
        handles.push(handle);
    }
    
    // Test concurrent metric recording
    for i in 0..10 {
        let system_clone = system.clone();
        let handle = tokio::spawn(async move {
            let mut tags = HashMap::new();
            tags.insert("thread".to_string(), i.to_string());
            system_clone.record_counter("concurrent_counter", 1, tags).await.unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_health_monitoring_scenarios() {
    let mut config = LoggingMonitoringConfig::default();
    config.logging.cloudwatch.enabled = false;
    config.tracing.xray.enabled = false;
    config.tracing.sampling.rate = 1.0; // 100% sampling for tests
    config.tracing.sampling.adaptive = false; // Disable adaptive sampling for tests
    config.metrics.cloudwatch.enabled = false;
    config.events.sns.enabled = false;
    
    let system = LoggingMonitoringSystem::new(config).await.unwrap();
    
    // Test different health statuses
    let health_statuses = vec![
        HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Unhealthy,
    ];
    
    for (i, status) in health_statuses.into_iter().enumerate() {
        let service_name = format!("test_service_{}", i);
        system.report_health(&service_name, status).await.unwrap();
        
        let reported_status = system.get_service_health(&service_name).await.unwrap();
        assert_eq!(reported_status, status);
    }
    
    // Test system health aggregation
    let system_health = system.get_system_health().await.unwrap();
    assert_eq!(system_health.total_services, 3);
    assert_eq!(system_health.healthy_services, 1);
    assert_eq!(system_health.degraded_services, 1);
    assert_eq!(system_health.unhealthy_services, 1);
    assert_eq!(system_health.overall_status, HealthStatus::Unhealthy);
    
    system.shutdown().await.unwrap();
} 