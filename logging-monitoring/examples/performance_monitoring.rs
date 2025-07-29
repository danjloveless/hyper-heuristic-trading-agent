//! Performance monitoring example
//! 
//! This example demonstrates advanced performance monitoring capabilities
//! including latency tracking, throughput monitoring, and performance alerts.

use logging_monitoring::{
    LoggingMonitoringSystem, LoggingMonitoringConfig, LogContext, LogLevel,
    SpanResult, SystemEvent, EventType, Alert, Notification, LoggingMonitoring,
};
use logging_monitoring::events::{EventSeverity, NotificationPriority, NotificationChannel};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use rand::Rng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Performance Monitoring Example");
    
    // Create configuration optimized for performance monitoring
    let mut config = LoggingMonitoringConfig::default();
    config.logging.level = "INFO".to_string();
    config.metrics.enabled = true;
    config.metrics.prometheus.enabled = true;
    config.events.enabled = true;
    
    // Initialize the system
    LoggingMonitoringSystem::initialize(config.clone()).await?;
    let system = LoggingMonitoringSystem::new(config).await?;
    
    // Simulate a high-traffic API with performance monitoring
    simulate_high_traffic_api(&system).await?;
    
    // Simulate performance degradation scenarios
    simulate_performance_degradation(&system).await?;
    
    // Simulate performance recovery
    simulate_performance_recovery(&system).await?;
    
    // Generate performance report
    generate_performance_report(&system).await?;
    
    println!("✅ Performance monitoring example completed!");
    
    system.shutdown().await?;
    Ok(())
}

/// Simulate a high-traffic API with comprehensive performance monitoring
async fn simulate_high_traffic_api(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Simulating high-traffic API...");
    
    let context = LogContext::new("api_gateway".to_string());
    system.log_info("Starting high-traffic simulation", context.clone()).await?;
    
    // Simulate multiple concurrent requests
    let mut handles = Vec::new();
    
    for i in 0..100 {
        let system_clone = system.clone();
        let context_clone = context.clone();
        
        let handle = tokio::spawn(async move {
            let request_id = format!("req-{}", i);
            let endpoint = if i % 3 == 0 { "/api/users" } else if i % 3 == 1 { "/api/orders" } else { "/api/products" };
            
            let trace_id = system_clone.start_trace("api_request").await.unwrap();
            system_clone.add_trace_annotation(trace_id, "request_id", &request_id).await.unwrap();
            system_clone.add_trace_annotation(trace_id, "endpoint", endpoint).await.unwrap();
            
            let start_time = Instant::now();
            
            // Simulate different types of processing
            let processing_span = system_clone.start_span(trace_id, "request_processing").await.unwrap();
            
            // Simulate variable processing time
            let processing_time = rand::thread_rng().gen_range(10..200);
            sleep(Duration::from_millis(processing_time)).await;
            
            system_clone.end_span(processing_span, SpanResult {
                success: true,
                error_message: None,
                duration_ms: processing_time,
                metadata: HashMap::new(),
            }).await.unwrap();
            
            // Simulate database query
            let db_span = system_clone.start_span(trace_id, "database_query").await.unwrap();
            let db_time = rand::thread_rng().gen_range(5..50);
            sleep(Duration::from_millis(db_time)).await;
            
            system_clone.end_span(db_span, SpanResult {
                success: true,
                error_message: None,
                duration_ms: db_time,
                metadata: HashMap::new(),
            }).await.unwrap();
            
            let total_time = start_time.elapsed();
            
            // Record metrics
            let mut tags = HashMap::new();
            tags.insert("endpoint".to_string(), endpoint.to_string());
            tags.insert("method".to_string(), "GET".to_string());
            tags.insert("status".to_string(), "200".to_string());
            
            system_clone.record_counter("http_requests_total", 1, tags.clone()).await.unwrap();
            system_clone.record_histogram("http_request_duration_ms", total_time.as_millis() as f64, tags.clone()).await.unwrap();
            system_clone.record_timer("http_request_duration", total_time, tags).await.unwrap();
            
            // End the trace
            system_clone.end_span(trace_id, SpanResult {
                success: true,
                error_message: None,
                duration_ms: total_time.as_millis() as u64,
                metadata: HashMap::new(),
            }).await.unwrap();
        });
        
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    system.log_info("High-traffic simulation completed", context).await?;
    Ok(())
}

/// Simulate performance degradation scenarios
async fn simulate_performance_degradation(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚠️  Simulating performance degradation...");
    
    let context = LogContext::new("api_gateway".to_string());
    
    // Simulate slow database queries
    for i in 0..10 {
        let trace_id = system.start_trace("slow_query").await?;
        system.add_trace_annotation(trace_id, "query_type", "complex_join").await?;
        
        let start_time = Instant::now();
        
        // Simulate slow query
        sleep(Duration::from_millis(rand::thread_rng().gen_range(2000..5000))).await;
        
        let duration = start_time.elapsed();
        
        // Record slow query metrics
        let mut tags = HashMap::new();
        tags.insert("query_type".to_string(), "complex_join".to_string());
        tags.insert("table".to_string(), "orders".to_string());
        
        system.record_histogram("database_query_duration_ms", duration.as_millis() as f64, tags.clone()).await?;
        system.record_counter("slow_queries_total", 1, tags).await?;
        
        system.end_span(trace_id, SpanResult {
            success: true,
            error_message: None,
            duration_ms: duration.as_millis() as u64,
            metadata: HashMap::new(),
        }).await?;
        
        // Create performance alert if query is very slow
        if duration.as_millis() > 3000 {
            let alert = Alert::new(
                "Slow Database Query Detected".to_string(),
                format!("Query took {}ms, exceeding 3 second threshold", duration.as_millis()),
                EventSeverity::Medium,
                "database".to_string(),
                "performance".to_string(),
            ).with_tag("duration_ms".to_string(), duration.as_millis().to_string())
             .with_tag("threshold_ms".to_string(), "3000".to_string());
            
            system.emit_alert(alert).await?;
        }
    }
    
    // Simulate high error rate
    for i in 0..20 {
        let trace_id = system.start_trace("error_request").await?;
        
        // Simulate random errors
        let is_error = rand::thread_rng().gen_bool(0.3); // 30% error rate
        
        if is_error {
            let error_span = system.start_span(trace_id, "error_processing").await?;
            
            sleep(Duration::from_millis(100)).await;
            
            system.end_span(error_span, SpanResult {
                success: false,
                error_message: Some("Internal server error".to_string()),
                duration_ms: 100,
                metadata: HashMap::new(),
            }).await?;
            
            // Record error metrics
            let mut tags = HashMap::new();
            tags.insert("endpoint".to_string(), "/api/users".to_string());
            tags.insert("error_type".to_string(), "internal_error".to_string());
            
            system.record_counter("http_errors_total", 1, tags).await?;
        }
        
        system.end_span(trace_id, SpanResult {
            success: !is_error,
            error_message: if is_error { Some("Request failed".to_string()) } else { None },
            duration_ms: 100,
            metadata: HashMap::new(),
        }).await?;
    }
    
    // Create high error rate alert
    let alert = Alert::new(
        "High Error Rate Detected".to_string(),
        "Error rate has exceeded 25% threshold".to_string(),
        EventSeverity::High,
        "api_gateway".to_string(),
        "reliability".to_string(),
    ).with_tag("error_rate".to_string(), "30%".to_string())
     .with_tag("threshold".to_string(), "25%".to_string());
    
    system.emit_alert(alert).await?;
    
    // Send notification to team
    let notification = Notification::new(
        "Performance Alert".to_string(),
        "High error rate and slow queries detected. Immediate attention required.".to_string(),
        NotificationPriority::High,
        NotificationChannel::Slack,
    ).with_recipient("devops-team".to_string())
     .with_tag("channel".to_string(), "#alerts".to_string());
    
    system.emit_notification(notification).await?;
    
    system.log_warn("Performance degradation simulation completed", context).await?;
    Ok(())
}

/// Simulate performance recovery
async fn simulate_performance_recovery(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Simulating performance recovery...");
    
    let context = LogContext::new("api_gateway".to_string());
    
    // Simulate normal performance after recovery
    for i in 0..50 {
        let trace_id = system.start_trace("recovered_request").await?;
        
        let start_time = Instant::now();
        
        // Simulate normal processing time
        sleep(Duration::from_millis(rand::thread_rng().gen_range(10..100))).await;
        
        let duration = start_time.elapsed();
        
        // Record recovered metrics
        let mut tags = HashMap::new();
        tags.insert("endpoint".to_string(), "/api/users".to_string());
        tags.insert("status".to_string(), "200".to_string());
        
        system.record_counter("http_requests_total", 1, tags.clone()).await?;
        system.record_histogram("http_request_duration_ms", duration.as_millis() as f64, tags).await?;
        
        system.end_span(trace_id, SpanResult {
            success: true,
            error_message: None,
            duration_ms: duration.as_millis() as u64,
            metadata: HashMap::new(),
        }).await?;
    }
    
    // Create recovery event
    let recovery_event = SystemEvent::new(
        EventType::System,
        EventSeverity::Info,
        "Performance Recovery".to_string(),
        "System performance has returned to normal levels".to_string(),
        "monitoring".to_string(),
    ).with_tag("recovery_time".to_string(), "5 minutes".to_string())
     .with_tag("action_taken".to_string(), "database_optimization".to_string());
    
    system.emit_event(recovery_event).await?;
    
    system.log_info("Performance recovery simulation completed", context).await?;
    Ok(())
}

/// Generate comprehensive performance report
async fn generate_performance_report(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Generating performance report...");
    
    // Get monitoring data
    let monitoring_data = system.monitoring_manager().get_monitoring_data().await?;
    
    println!("📈 Performance Metrics:");
    if let Some(ref perf) = monitoring_data.performance {
        println!("   - P50 Latency: {:?}", perf.latency_p50);
        println!("   - P95 Latency: {:?}", perf.latency_p95);
        println!("   - P99 Latency: {:?}", perf.latency_p99);
        println!("   - Throughput: {:.2} RPS", perf.throughput_rps);
        println!("   - Error Rate: {:.2}%", perf.error_rate * 100.0);
        println!("   - Success Rate: {:.2}%", perf.success_rate * 100.0);
    }
    
    // Get aggregated metrics
    let aggregated_metrics = system.metrics_manager().get_aggregated_metrics().await;
    
    println!("📊 Key Metrics Summary:");
    for (name, metric) in aggregated_metrics.iter() {
        if name.contains("http_request") || name.contains("database") || name.contains("error") {
            println!("   - {}: count={}, avg={:.2}ms, min={:.2}ms, max={:.2}ms", 
                     name, metric.count, metric.average(), metric.min, metric.max);
        }
    }
    
    // Get monitoring summary
    let summary = system.monitoring_manager().get_monitoring_summary().await?;
    println!("📋 Overall Performance Status:");
    println!("   - Status: {:?}", summary.overall_status);
    println!("   - Performance: {:?}", summary.performance_status);
    println!("   - Resource Utilization: {:?}", summary.resource_utilization);
    println!("   - Active Alerts: {}", summary.alert_count());
    
    // Performance recommendations
    println!("💡 Performance Recommendations:");
    if let Some(ref perf) = monitoring_data.performance {
        if perf.latency_p99 > Duration::from_millis(1000) {
            println!("   - Consider optimizing database queries (P99 latency > 1s)");
        }
        if perf.error_rate > 0.05 {
            println!("   - Investigate error sources (error rate > 5%)");
        }
        if perf.throughput_rps < 100.0 {
            println!("   - Consider scaling up resources (throughput < 100 RPS)");
        }
    }
    
    Ok(())
} 