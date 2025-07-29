//! Performance monitoring example
//! 
//! This example demonstrates advanced performance monitoring capabilities
//! including latency tracking, throughput monitoring, and performance alerts.

use logging_monitoring::{
    LoggingMonitoringSystem, LoggingMonitoringConfig, LogContext,
    SpanResult, SystemEvent, EventType, LoggingMonitoring,
};
use logging_monitoring::events::EventSeverity;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use rand::Rng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Performance Monitoring Example");
    
    // Create configuration with 100% sampling for the example
    let mut config = LoggingMonitoringConfig::default();
    config.tracing.sampling.rate = 1.0; // 100% sampling for example
    config.tracing.sampling.adaptive = false;
    
    // Initialize the system
    LoggingMonitoringSystem::initialize(config.clone()).await?;
    
    // Create the system instance
    let system = LoggingMonitoringSystem::new(config).await?;
    
    // Simulate high-traffic API
    simulate_high_traffic_api(&system).await?;
    
    // Simulate performance degradation
    simulate_performance_degradation(&system).await?;
    
    // Simulate performance recovery
    simulate_performance_recovery(&system).await?;
    
    // Generate performance report
    generate_performance_report(&system).await?;
    
    println!("✅ Performance monitoring example completed successfully!");
    
    // Shutdown
    system.shutdown().await?;
    
    Ok(())
}

/// Simulate high-traffic API with performance monitoring
async fn simulate_high_traffic_api(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Simulating high-traffic API...");
    
    let context = LogContext::new("api_gateway".to_string());
    
    // Simulate 20 sequential API requests instead of 100 concurrent ones
    for i in 0..20 {
        let request_id = format!("req-{}", i);
        let endpoint = if i % 3 == 0 { "/api/users" } else if i % 3 == 1 { "/api/orders" } else { "/api/products" };
        
        // Handle trace creation errors gracefully
        let trace_info = match system.start_trace("api_request").await {
            Ok(info) => info,
            Err(e) => {
                eprintln!("Failed to start trace for request {}: {}", request_id, e);
                continue;
            }
        };
        
        // Add trace annotations
        if let Err(e) = system.add_trace_annotation(trace_info.trace_id, "request_id", &request_id).await {
            eprintln!("Failed to add request_id annotation: {}", e);
        }
        if let Err(e) = system.add_trace_annotation(trace_info.trace_id, "endpoint", endpoint).await {
            eprintln!("Failed to add endpoint annotation: {}", e);
        }
        
        let start_time = Instant::now();
        
        // Simulate different types of processing
        let processing_span = match system.start_span(trace_info.trace_id, "request_processing").await {
            Ok(span) => span,
            Err(e) => {
                eprintln!("Failed to start processing span: {}", e);
                continue;
            }
        };
        
        // Simulate variable processing time
        let processing_time = rand::thread_rng().gen_range(10..200);
        sleep(Duration::from_millis(processing_time)).await;
        
        if let Err(e) = system.end_span(processing_span, SpanResult {
            success: true,
            error_message: None,
            duration_ms: processing_time,
            metadata: HashMap::new(),
        }).await {
            eprintln!("Failed to end processing span: {}", e);
        }
        
        // Simulate database query
        let db_span = match system.start_span(trace_info.trace_id, "database_query").await {
            Ok(span) => span,
            Err(e) => {
                eprintln!("Failed to start database span: {}", e);
                continue;
            }
        };
        let db_time = rand::thread_rng().gen_range(5..50);
        sleep(Duration::from_millis(db_time)).await;
        
        if let Err(e) = system.end_span(db_span, SpanResult {
            success: true,
            error_message: None,
            duration_ms: db_time,
            metadata: HashMap::new(),
        }).await {
            eprintln!("Failed to end database span: {}", e);
        }
        
        let total_time = start_time.elapsed();
        
        // Record metrics
        let mut tags = HashMap::new();
        tags.insert("endpoint".to_string(), endpoint.to_string());
        tags.insert("method".to_string(), "GET".to_string());
        tags.insert("status".to_string(), "200".to_string());
        
        if let Err(e) = system.record_counter("http_requests_total", 1, tags.clone()).await {
            eprintln!("Failed to record counter: {}", e);
        }
        if let Err(e) = system.record_histogram("http_request_duration_ms", total_time.as_millis() as f64, tags.clone()).await {
            eprintln!("Failed to record histogram: {}", e);
        }
        if let Err(e) = system.record_timer("http_request_duration", total_time, tags).await {
            eprintln!("Failed to record timer: {}", e);
        }
        
        // End the trace
        if let Err(e) = system.end_span(trace_info.root_span_id, SpanResult {
            success: true,
            error_message: None,
            duration_ms: total_time.as_millis() as u64,
            metadata: HashMap::new(),
        }).await {
            eprintln!("Failed to end trace: {}", e);
        }
        
        // Small delay between requests
        sleep(Duration::from_millis(10)).await;
    }
    
    system.log_info("High-traffic simulation completed", context).await?;
    Ok(())
}

/// Simulate performance degradation scenarios
async fn simulate_performance_degradation(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚠️  Simulating performance degradation...");
    
    let context = LogContext::new("api_gateway".to_string());
    
    // Simulate slow database queries
    for _i in 0..10 {
        let trace_info = system.start_trace("slow_query").await?;
        system.add_trace_annotation(trace_info.trace_id, "query_type", "complex_join").await?;
        
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
        
        system.end_span(trace_info.root_span_id, SpanResult {
            success: true,
            error_message: None,
            duration_ms: duration.as_millis() as u64,
            metadata: HashMap::new(),
        }).await?;
    }
    
    // Simulate error scenarios
    for _i in 0..5 {
        let trace_info = system.start_trace("error_scenario").await?;
        
        let error_span = system.start_span(trace_info.trace_id, "error_processing").await?;
        
        // Simulate error
        sleep(Duration::from_millis(rand::thread_rng().gen_range(100..500))).await;
        
        system.end_span(error_span, SpanResult {
            success: false,
            error_message: Some("Database connection timeout".to_string()),
            duration_ms: 300,
            metadata: HashMap::new(),
        }).await?;
        
        system.end_span(trace_info.root_span_id, SpanResult {
            success: false,
            error_message: Some("Request failed due to database error".to_string()),
            duration_ms: 300,
            metadata: HashMap::new(),
        }).await?;
    }
    
    system.log_warn("Performance degradation simulation completed", context).await?;
    Ok(())
}

/// Simulate performance recovery
async fn simulate_performance_recovery(system: &LoggingMonitoringSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Simulating performance recovery...");
    
    let context = LogContext::new("api_gateway".to_string());
    
    // Simulate normal performance after recovery
    for _i in 0..50 {
        let trace_info = system.start_trace("recovered_request").await?;
        
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
        
        system.end_span(trace_info.root_span_id, SpanResult {
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