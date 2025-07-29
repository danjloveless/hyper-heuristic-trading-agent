//! Comprehensive example demonstrating the logging-monitoring system
//! 
//! This example shows how to use all the features of the logging-monitoring system
//! in a realistic trading application scenario.

use logging_monitoring::{
    LoggingMonitoringSystem, LoggingMonitoringConfig, LogContext,
    SpanResult, HealthStatus, SystemEvent, Alert, Notification,
    EventType, AuditAction, AuditContext, LoggingMonitoring,
};
use logging_monitoring::events::{EventSeverity, NotificationPriority, NotificationChannel};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting QuantumTrade AI Logging & Monitoring Example");
    
    // Initialize the logging and monitoring system
    let config = create_configuration();
    LoggingMonitoringSystem::initialize(config.clone()).await?;
    
    let system = LoggingMonitoringSystem::new(config).await?;
    
    // Simulate a trading application
    simulate_trading_application(&system).await?;
    
    // Simulate system monitoring
    simulate_system_monitoring(&system).await?;
    
    // Simulate error scenarios
    simulate_error_scenarios(&system).await?;
    
    // Simulate audit logging
    simulate_audit_logging(&system).await?;
    
    // Generate a comprehensive report
    generate_system_report(&system).await?;
    
    println!("✅ Example completed successfully!");
    
    // Graceful shutdown
    system.shutdown().await?;
    
    Ok(())
}

/// Create a comprehensive configuration for the example
fn create_configuration() -> LoggingMonitoringConfig {
    let mut config = LoggingMonitoringConfig::default();
    
    // Configure logging
    config.logging.level = "DEBUG".to_string();
    config.logging.json_format = true;
    config.logging.cloudwatch.enabled = false; // Disable for local testing
    config.logging.local_file.enabled = true;
    config.logging.local_file.file_path = "logs/quantumtrade_example.log".to_string();
    
    // Configure tracing
    config.tracing.enabled = true;
    config.tracing.xray.enabled = false; // Disable for local testing
    config.tracing.sampling.rate = 1.0; // Sample all traces for demo
    
    // Configure metrics
    config.metrics.enabled = true;
    config.metrics.prometheus.enabled = true;
    config.metrics.cloudwatch.enabled = false; // Disable for local testing
    
    // Configure health monitoring
    config.health.enabled = true;
    config.health.check_interval = Duration::from_secs(10);
    
    // Configure events
    config.events.enabled = true;
    config.events.sns.enabled = false; // Disable for local testing
    
    config
}

/// Simulate a trading application with various operations
async fn simulate_trading_application(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Simulating trading application...");
    
    let context = LogContext::new("trading_engine".to_string())
        .with_request_id("trade-12345".to_string())
        .with_user_id("user-67890".to_string());
    
    // Log application startup
    system.log_info("Trading engine started", context.clone()).await?;
    
    // Start a trace for the trading operation
    let trace_info = system.start_trace("trading_operation").await?;
    system.add_trace_annotation(trace_info.trace_id, "user_id", "user-67890").await?;
    system.add_trace_annotation(trace_info.trace_id, "operation_type", "buy_order").await?;
    
    // Simulate market data analysis
    let analysis_span = system.start_span(trace_info.trace_id, "market_analysis").await?;
    sleep(Duration::from_millis(50)).await;
    
    // Record analysis metrics
    let mut analysis_tags = HashMap::new();
    analysis_tags.insert("analysis_type".to_string(), "technical".to_string());
    analysis_tags.insert("symbol".to_string(), "AAPL".to_string());
    system.record_histogram("analysis_duration_ms", 50.0, analysis_tags).await?;
    
    system.end_span(analysis_span, SpanResult {
        success: true,
        error_message: None,
        duration_ms: 50,
        metadata: HashMap::new(),
    }).await?;
    
    // Simulate order execution
    let execution_span = system.start_span(trace_info.trace_id, "order_execution").await?;
    sleep(Duration::from_millis(100)).await;
    
    // Record execution metrics
    let mut execution_tags = HashMap::new();
    execution_tags.insert("order_type".to_string(), "market".to_string());
    execution_tags.insert("symbol".to_string(), "AAPL".to_string());
    system.record_counter("orders_executed", 1, execution_tags.clone()).await?;
    system.record_timer("execution_duration", Duration::from_millis(100), execution_tags).await?;
    
    system.end_span(execution_span, SpanResult {
        success: true,
        error_message: None,
        duration_ms: 100,
        metadata: HashMap::new(),
    }).await?;
    
    // End the main trace
    system.end_span(trace_info.root_span_id, SpanResult {
        success: true,
        error_message: None,
        duration_ms: 150,
        metadata: HashMap::new(),
    }).await?;
    
    // Log successful trade
    system.log_info("Trade executed successfully: BUY 100 AAPL @ $150.25", context.clone()).await?;
    
    // Record business metrics
    let mut trade_tags = HashMap::new();
    trade_tags.insert("symbol".to_string(), "AAPL".to_string());
    trade_tags.insert("side".to_string(), "buy".to_string());
    system.record_gauge("position_size", 100.0, trade_tags).await?;
    
    Ok(())
}

/// Simulate system monitoring and health checks
async fn simulate_system_monitoring(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Simulating system monitoring...");
    
    // Add custom health checks
    system.health_manager().add_service(
        "database".to_string(),
        logging_monitoring::config::ServiceHealthConfig {
            endpoint: "/health".to_string(),
            expected_response_time: Duration::from_secs(5),
            retry_count: 3,
            circuit_breaker: true,
            failure_threshold: 5,
        }
    ).await?;
    
    system.health_manager().add_service(
        "market_data_feed".to_string(),
        logging_monitoring::config::ServiceHealthConfig {
            endpoint: "/health".to_string(),
            expected_response_time: Duration::from_secs(2),
            retry_count: 2,
            circuit_breaker: true,
            failure_threshold: 3,
        }
    ).await?;
    
    // Report health status for various services
    system.report_health("trading_engine", HealthStatus::Healthy).await?;
    system.report_health("database", HealthStatus::Healthy).await?;
    system.report_health("market_data_feed", HealthStatus::Degraded).await?;
    
    // Record system metrics
    let mut system_tags = HashMap::new();
    system_tags.insert("service".to_string(), "trading_engine".to_string());
    system.record_gauge("cpu_usage_percent", 45.2, system_tags.clone()).await?;
    system.record_gauge("memory_usage_mb", 1024.0, system_tags.clone()).await?;
    system.record_gauge("active_connections", 150.0, system_tags).await?;
    
    // Emit system events
    let system_event = SystemEvent::new(
        EventType::System,
        EventSeverity::Info,
        "System Status Update".to_string(),
        "All core services are operational".to_string(),
        "monitoring".to_string(),
    ).with_tag("environment".to_string(), "production".to_string());
    
    system.emit_event(system_event).await?;
    
    // Check system health
    let health = system.get_system_health().await?;
    println!("📊 System Health: {:?}", health.overall_status);
    println!("   - Total services: {}", health.total_services);
    println!("   - Healthy: {}", health.healthy_services);
    println!("   - Degraded: {}", health.degraded_services);
    println!("   - Unhealthy: {}", health.unhealthy_services);
    
    Ok(())
}

/// Simulate error scenarios and alerting
async fn simulate_error_scenarios(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚠️  Simulating error scenarios...");
    
    let context = LogContext::new("trading_engine".to_string())
        .with_request_id("error-12345".to_string());
    
    // Simulate a database connection error
    let db_error = logging_monitoring::LoggingMonitoringError::ConnectionTimeout {
        operation: "database_query".to_string(),
    };
    
    system.log_error(&db_error, context.clone()).await?;
    
    // Create an alert for the error
    let alert = Alert::new(
        "Database Connection Timeout".to_string(),
        "Failed to connect to primary database after 3 attempts".to_string(),
        EventSeverity::High,
        "trading_engine".to_string(),
        "infrastructure".to_string(),
    ).with_tag("environment".to_string(), "production".to_string())
     .with_tag("retry_count".to_string(), "3".to_string());
    
    system.emit_alert(alert).await?;
    
    // Simulate a performance degradation
    let mut perf_tags = HashMap::new();
    perf_tags.insert("endpoint".to_string(), "/api/trades".to_string());
    system.record_histogram("response_time_ms", 2500.0, perf_tags).await?;
    
    // Create a performance alert
    let perf_alert = Alert::new(
        "High Response Time Detected".to_string(),
        "API response time exceeded 2 seconds threshold".to_string(),
        EventSeverity::Medium,
        "api_gateway".to_string(),
        "performance".to_string(),
    ).with_tag("threshold_ms".to_string(), "2000".to_string())
     .with_tag("actual_ms".to_string(), "2500".to_string());
    
    system.emit_alert(perf_alert).await?;
    
    // Send a notification to the team
    let notification = Notification::new(
        "System Alert".to_string(),
        "Database connection issues detected. Team has been notified.".to_string(),
        NotificationPriority::High,
        NotificationChannel::Slack,
    ).with_recipient("trading-team".to_string())
     .with_tag("channel".to_string(), "#alerts".to_string());
    
    system.emit_notification(notification).await?;
    
    Ok(())
}

/// Simulate audit logging for compliance
async fn simulate_audit_logging(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Simulating audit logging...");
    
    // Simulate user login
    let login_action = AuditAction {
        action: "user_login".to_string(),
        resource: "authentication".to_string(),
        details: {
            let mut details = HashMap::new();
            details.insert("ip_address".to_string(), serde_json::Value::String("192.168.1.100".to_string()));
            details.insert("user_agent".to_string(), serde_json::Value::String("Mozilla/5.0...".to_string()));
            details.insert("success".to_string(), serde_json::Value::Bool(true));
            details
        },
    };
    
    let login_context = AuditContext {
        ip_address: Some("192.168.1.100".to_string()),
        user_agent: Some("Mozilla/5.0...".to_string()),
        session_id: Some("sess-12345".to_string()),
        additional_data: HashMap::new(),
    };
    
    system.log_audit(login_action, "user-67890".to_string(), login_context).await?;
    
    // Simulate trade execution audit
    let trade_action = AuditAction {
        action: "trade_execution".to_string(),
        resource: "trading".to_string(),
        details: {
            let mut details = HashMap::new();
            details.insert("symbol".to_string(), serde_json::Value::String("AAPL".to_string()));
            details.insert("quantity".to_string(), serde_json::Value::Number(100.into()));
            details.insert("price".to_string(), serde_json::Value::Number(15025.into())); // $150.25
            details.insert("side".to_string(), serde_json::Value::String("buy".to_string()));
            details.insert("order_id".to_string(), serde_json::Value::String("ord-12345".to_string()));
            details
        },
    };
    
    let trade_context = AuditContext {
        ip_address: Some("192.168.1.100".to_string()),
        user_agent: Some("TradingApp/1.0".to_string()),
        session_id: Some("sess-12345".to_string()),
        additional_data: HashMap::new(),
    };
    
    system.log_audit(trade_action, "user-67890".to_string(), trade_context).await?;
    
    // Simulate configuration change audit
    let config_action = AuditAction {
        action: "configuration_change".to_string(),
        resource: "system_config".to_string(),
        details: {
            let mut details = HashMap::new();
            details.insert("parameter".to_string(), serde_json::Value::String("max_order_size".to_string()));
            details.insert("old_value".to_string(), serde_json::Value::Number(1000.into()));
            details.insert("new_value".to_string(), serde_json::Value::Number(2000.into()));
            details.insert("reason".to_string(), serde_json::Value::String("risk_management_update".to_string()));
            details
        },
    };
    
    let config_context = AuditContext {
        ip_address: Some("10.0.0.50".to_string()),
        user_agent: Some("AdminPanel/2.0".to_string()),
        session_id: Some("admin-sess-67890".to_string()),
        additional_data: HashMap::new(),
    };
    
    system.log_audit(config_action, "admin-12345".to_string(), config_context).await?;
    
    Ok(())
}

/// Generate a comprehensive system report
async fn generate_system_report(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Generating system report...");
    
    // Get monitoring data
    let monitoring_data = system.monitoring_manager().get_monitoring_data().await?;
    println!("📈 Monitoring Data:");
    if let Some(perf) = monitoring_data.performance {
        println!("   - P50 Latency: {:?}", perf.latency_p50);
        println!("   - P95 Latency: {:?}", perf.latency_p95);
        println!("   - P99 Latency: {:?}", perf.latency_p99);
        println!("   - Throughput: {:.2} RPS", perf.throughput_rps);
        println!("   - Error Rate: {:.2}%", perf.error_rate * 100.0);
    }
    
    if let Some(resources) = monitoring_data.resources {
        println!("   - CPU Usage: {:.1}%", resources.cpu_usage_percent);
        println!("   - Memory Usage: {:.1}%", resources.memory_usage_percent);
        println!("   - Disk Usage: {:.1}%", resources.disk_usage_percent);
    }
    
    if let Some(business) = monitoring_data.business {
        println!("   - Prediction Accuracy: {:.1}%", business.prediction_accuracy * 100.0);
        println!("   - Trading Performance: {:.2}%", business.trading_performance * 100.0);
        println!("   - User Activity: {}", business.user_activity_count);
    }
    
    // Get monitoring summary
    let summary = system.monitoring_manager().get_monitoring_summary().await?;
    println!("📋 Monitoring Summary:");
    println!("   - Overall Status: {:?}", summary.overall_status);
    println!("   - Performance Status: {:?}", summary.performance_status);
    println!("   - Resource Utilization: {:?}", summary.resource_utilization);
    println!("   - Business Health: {:?}", summary.business_health);
    println!("   - Active Alerts: {}", summary.alert_count());
    
    // Get health statistics
    let health_stats = system.health_manager().get_health_statistics().await;
    println!("🏥 Health Statistics:");
    println!("   - Total Checks: {}", health_stats.total_checks);
    println!("   - Success Rate: {:.1}%", health_stats.success_rate());
    println!("   - Failure Rate: {:.1}%", health_stats.failure_rate());
    println!("   - Avg Response Time: {:.1}ms", health_stats.average_response_time_ms);
    
    // Get aggregated metrics
    let aggregated_metrics = system.metrics_manager().get_aggregated_metrics().await;
    println!("📊 Aggregated Metrics:");
    for (name, metric) in aggregated_metrics.iter().take(5) {
        println!("   - {}: count={}, avg={:.2}, min={:.2}, max={:.2}", 
                 name, metric.count, metric.average(), metric.min, metric.max);
    }
    
    // Get Prometheus metrics
    if let Ok(prometheus_metrics) = system.metrics_manager().get_prometheus_metrics().await {
        println!("📈 Prometheus Metrics (first 500 chars):");
        println!("{}", &prometheus_metrics[..prometheus_metrics.len().min(500)]);
    }
    
    Ok(())
} 