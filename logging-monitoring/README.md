# Logging & Monitoring Module

A comprehensive observability solution for QuantumTrade AI services, providing structured logging, distributed tracing, metrics collection, and real-time monitoring capabilities.

## Features

### 🔍 **Structured Logging**
- JSON-formatted logs with correlation IDs and context
- CloudWatch integration for centralized logging
- Local file logging with rotation and compression
- Configurable log levels and sampling
- Audit logging for compliance

### 🕵️ **Distributed Tracing**
- AWS X-Ray integration for distributed tracing
- Span-based tracing with annotations and tags
- Adaptive sampling for performance optimization
- Trace correlation across services

### 📊 **Metrics Collection**
- Prometheus metrics exposition
- CloudWatch metrics integration
- Custom metric types (counters, gauges, histograms, timers)
- Metric aggregation and retention policies
- Real-time metric processing

### 🏥 **Health Monitoring**
- Service health checks with circuit breakers
- Configurable health check intervals and timeouts
- Health status aggregation and reporting
- Health statistics and trend analysis

### 🚨 **Event Management**
- System events, alerts, and notifications
- SNS integration for event delivery
- Event filtering and routing
- Multiple notification channels (Slack, Email, SMS)

### 📈 **Performance Monitoring**
- Latency tracking (P50, P95, P99)
- Throughput monitoring
- Error rate tracking
- Resource utilization monitoring
- Business metrics tracking

## Quick Start

### Basic Usage

```rust
use logging_monitoring::{LoggingMonitoringSystem, LoggingMonitoringConfig, LogContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the system
    let config = LoggingMonitoringConfig::default();
    LoggingMonitoringSystem::initialize(config.clone()).await?;
    
    let system = LoggingMonitoringSystem::new(config).await?;
    
    // Log some events
    let context = LogContext::new("my_service".to_string())
        .with_request_id("req-123".to_string());
    
    system.log_info("Application started", context.clone()).await?;
    
    // Record metrics
    let mut tags = std::collections::HashMap::new();
    tags.insert("service".to_string(), "my_service".to_string());
    system.record_counter("requests_total", 1, tags).await?;
    
    // Start tracing
    let trace_id = system.start_trace("api_request").await?;
    let span_id = system.start_span(trace_id, "database_query").await?;
    
    // ... perform work ...
    
    system.end_span(span_id, crate::SpanResult {
        success: true,
        error_message: None,
        duration_ms: 100,
        metadata: std::collections::HashMap::new(),
    }).await?;
    
    // Report health
    system.report_health("my_service", crate::health::HealthStatus::Healthy).await?;
    
    Ok(())
}
```

### Advanced Configuration

```rust
use logging_monitoring::{LoggingMonitoringConfig, LoggingConfig, TracingConfig};

let mut config = LoggingMonitoringConfig::default();

// Configure logging
config.logging.level = "DEBUG".to_string();
config.logging.cloudwatch.enabled = true;
config.logging.cloudwatch.log_group = "/myapp/logs".to_string();

// Configure tracing
config.tracing.enabled = true;
config.tracing.xray.enabled = true;
config.tracing.xray.service_name = "my-service".to_string();

// Configure metrics
config.metrics.enabled = true;
config.metrics.prometheus.enabled = true;
config.metrics.cloudwatch.enabled = true;
```

## Examples

### Basic Usage Example
```bash
cargo run --example basic_usage
```

### Comprehensive Example
```bash
cargo run --example comprehensive_example
```

### Performance Monitoring Example
```bash
cargo run --example performance_monitoring
```

## Configuration

### Logging Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `level` | Log level (DEBUG, INFO, WARN, ERROR) | "INFO" |
| `json_format` | Enable JSON formatting | true |
| `cloudwatch.enabled` | Enable CloudWatch logging | true |
| `cloudwatch.log_group` | CloudWatch log group name | "/quantumtrade/logs" |
| `local_file.enabled` | Enable local file logging | false |
| `local_file.file_path` | Local log file path | "logs/quantumtrade.log" |

### Tracing Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `enabled` | Enable distributed tracing | true |
| `xray.enabled` | Enable X-Ray integration | true |
| `xray.service_name` | Service name for X-Ray | "quantumtrade" |
| `sampling.rate` | Sampling rate (0.0 to 1.0) | 0.1 |
| `sampling.adaptive` | Enable adaptive sampling | true |

### Metrics Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `enabled` | Enable metrics collection | true |
| `prometheus.enabled` | Enable Prometheus metrics | true |
| `prometheus.port` | Prometheus metrics port | 9090 |
| `cloudwatch.enabled` | Enable CloudWatch metrics | true |
| `cloudwatch.namespace` | CloudWatch namespace | "QuantumTrade" |

### Health Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `enabled` | Enable health monitoring | true |
| `check_interval` | Health check interval | 30s |
| `check_timeout` | Health check timeout | 10s |
| `alerting.enabled` | Enable health alerts | true |

## API Reference

### Logging

```rust
// Basic logging
system.log_info(message, context).await?;
system.log_warn(message, context).await?;
system.log_error(error, context).await?;
system.log_debug(message, context).await?;

// Structured logging
system.log_structured(level, structured_event).await?;

// Audit logging
system.log_audit(action, user, context).await?;
```

### Tracing

```rust
// Start a trace
let trace_id = system.start_trace("operation_name").await?;

// Add annotations
system.add_trace_annotation(trace_id, "key", "value").await?;

// Start a span
let span_id = system.start_span(trace_id, "span_name").await?;

// End a span
system.end_span(span_id, result).await?;
```

### Metrics

```rust
// Record different metric types
system.record_counter("metric_name", value, tags).await?;
system.record_gauge("metric_name", value, tags).await?;
system.record_histogram("metric_name", value, tags).await?;
system.record_timer("metric_name", duration, tags).await?;
```

### Health Monitoring

```rust
// Report health status
system.report_health("service_name", HealthStatus::Healthy).await?;

// Get service health
let health = system.get_service_health("service_name").await?;

// Get system health
let system_health = system.get_system_health().await?;
```

### Events

```rust
// Emit system events
system.emit_event(event).await?;

// Emit alerts
system.emit_alert(alert).await?;

// Emit notifications
system.emit_notification(notification).await?;
```

## Integration with AWS Services

### CloudWatch Logs
- Automatic log delivery to CloudWatch
- Configurable log groups and streams
- Batch processing for efficiency
- Retry logic with exponential backoff

### CloudWatch Metrics
- Custom metric publishing
- Metric aggregation and retention
- Alarm integration
- Cost optimization through batching

### X-Ray
- Distributed tracing across services
- Trace visualization in X-Ray console
- Performance bottleneck identification
- Service dependency mapping

### SNS
- Event delivery to SNS topics
- Alert and notification routing
- Multiple notification channels
- Event filtering and transformation

## Performance Considerations

### Logging Performance
- Asynchronous logging with buffering
- Configurable buffer sizes and flush intervals
- Sampling for high-volume scenarios
- Background processing to minimize latency

### Metrics Performance
- In-memory metric aggregation
- Batch processing for external systems
- Configurable sampling rates
- Efficient storage and retrieval

### Tracing Performance
- Adaptive sampling based on traffic
- Span compression and optimization
- Background trace processing
- Minimal overhead in production

## Monitoring and Alerting

### Built-in Alerts
- High error rates
- Slow response times
- Service health degradation
- Resource utilization thresholds

### Custom Alerts
- Business metric thresholds
- Custom performance indicators
- Compliance violations
- Security events

### Notification Channels
- Slack integration
- Email notifications
- SMS alerts
- Webhook callbacks

## Best Practices

### Logging
1. Use structured logging with consistent field names
2. Include correlation IDs for request tracing
3. Log at appropriate levels (DEBUG, INFO, WARN, ERROR)
4. Avoid logging sensitive information
5. Use sampling for high-volume logs

### Tracing
1. Create meaningful span names
2. Add relevant annotations and tags
3. Keep spans focused and granular
4. Use adaptive sampling in production
5. Monitor trace overhead

### Metrics
1. Use descriptive metric names
2. Include relevant tags for filtering
3. Choose appropriate metric types
4. Set up meaningful thresholds
5. Monitor metric cardinality

### Health Checks
1. Implement comprehensive health checks
2. Use appropriate timeouts and retries
3. Monitor circuit breaker states
4. Set up escalation procedures
5. Document health check endpoints

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce buffer sizes
   - Increase flush intervals
   - Enable sampling

2. **Slow Performance**
   - Check sampling rates
   - Optimize metric cardinality
   - Review health check intervals

3. **Missing Logs**
   - Verify log levels
   - Check CloudWatch permissions
   - Review buffer configuration

4. **Tracing Issues**
   - Verify X-Ray daemon connectivity
   - Check sampling configuration
   - Review span lifecycle management

### Debug Mode

Enable debug logging for troubleshooting:

```rust
config.logging.level = "DEBUG".to_string();
config.logging.include_file_line = true;
config.logging.include_thread_info = true;
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 