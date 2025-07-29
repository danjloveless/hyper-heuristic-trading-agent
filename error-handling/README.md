# QuantumTrade AI - Error Handling Module

A comprehensive error handling system for the QuantumTrade AI platform, providing robust error classification, retry mechanisms, circuit breakers, fallback strategies, and error reporting.

## Features

- **Error Classification**: Automatic classification of errors by type and severity
- **Retry Mechanisms**: Configurable exponential backoff with jitter
- **Circuit Breakers**: Prevent cascading failures with configurable thresholds
- **Fallback Strategies**: Graceful degradation with multiple fallback options
- **Error Reporting**: Multi-destination error reporting (logs, metrics, alerts)
- **Async Support**: Full async/await support for high-performance applications

## Quick Start

```rust
use error_handling::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a default error handler
    let error_handler = create_default_error_handler();
    
    // Create error context
    let context = create_error_context("my_service", "my_operation")
        .with_request_id("req-123".to_string())
        .with_user_id("user-456".to_string());
    
    // Handle an error
    let error = QuantumTradeError::NetworkConnection {
        host: "api.example.com".to_string(),
    };
    
    match error_handler.handle_error(&error, &context).await {
        Ok(result) => {
            match result.action {
                ErrorAction::Retry => {
                    println!("Will retry after {:?}", result.retry_after);
                }
                ErrorAction::UseFallback => {
                    if let Some(fallback_value) = result.fallback_value {
                        println!("Using fallback: {}", fallback_value);
                    }
                }
                ErrorAction::FailFast => {
                    println!("Failing fast - no retry");
                }
                ErrorAction::Escalate => {
                    println!("Escalating to human intervention");
                }
            }
        }
        Err(e) => println!("Error handling failed: {}", e),
    }
    
    Ok(())
}
```

## Error Types

The system defines comprehensive error types for different scenarios:

### Configuration Errors
- `ConfigurationNotFound`
- `InvalidConfiguration`
- `SecretAccessDenied`

### Database Errors
- `DatabaseConnectionFailed`
- `QueryExecutionFailed`
- `TransactionFailed`
- `SchemaValidationFailed`

### Network Errors
- `RequestTimeout`
- `RateLimitExceeded`
- `NetworkConnection`

### Authentication Errors
- `AuthenticationFailed`
- `AuthorizationDenied`
- `TokenExpired`

### Business Errors
- `InvalidTradingParameters`
- `DataQualityIssue`
- `LowPredictionConfidence`
- `RiskLimitExceeded`

### Validation Errors
- `ValidationFailed`

## Configuration

The error handling system is highly configurable:

```rust
let config = ErrorHandlingConfig {
    retry: RetryConfig {
        max_attempts: 3,
        initial_delay_ms: 100,
        max_delay_ms: 30000,
        backoff_multiplier: 2.0,
        jitter: true,
        retryable_errors: vec![
            ErrorType::Transient,
            ErrorType::Network,
            ErrorType::System,
        ],
    },
    circuit_breaker: CircuitBreakerConfig {
        failure_threshold: 5,
        success_threshold: 3,
        timeout_ms: 60000,
        half_open_max_calls: 3,
    },
    fallback: FallbackConfig {
        enabled: true,
        strategies: HashMap::new(),
        default_strategy: FallbackStrategy::DefaultValue,
    },
    reporting: ReportingConfig {
        enabled: true,
        destinations: vec![
            ReportingDestination::Logs,
            ReportingDestination::Metrics,
        ],
        batch_size: 100,
        flush_interval_ms: 5000,
        severity_threshold: ErrorSeverity::Medium,
    },
    classification: ClassificationConfig::default(),
};

let error_handler = ErrorHandlerImpl::new(config);
```

## Circuit Breaker

The circuit breaker pattern prevents cascading failures:

```rust
let mut circuit_breaker = CircuitBreaker::new(
    "service_name".to_string(),
    CircuitBreakerConfig::default(),
);

if circuit_breaker.can_execute() {
    match perform_operation().await {
        Ok(result) => {
            circuit_breaker.record_success();
            Ok(result)
        }
        Err(e) => {
            circuit_breaker.record_failure();
            Err(e)
        }
    }
} else {
    Err(ErrorHandlingError::CircuitBreakerOpen {
        service: "service_name".to_string(),
    })
}
```

## Retry Strategy

Configurable retry with exponential backoff:

```rust
let retry_strategy = RetryStrategy::new(RetryConfig::default());
let retry_executor = RetryExecutor::new(retry_strategy);

let result = retry_executor.execute(|| {
    Box::pin(async {
        // Your async operation here
        perform_operation().await
    })
}).await;
```

## Fallback Strategies

Multiple fallback strategies for graceful degradation:

- `DefaultValue`: Return a predefined default value
- `CacheLookup`: Try to get value from cache
- `PreviousResult`: Use the last known good result
- `AlternativeService`: Try an alternative service
- `GracefulDegradation`: Provide degraded functionality

## Error Reporting

Multi-destination error reporting:

```rust
let config = ReportingConfig {
    enabled: true,
    destinations: vec![
        ReportingDestination::Logs,
        ReportingDestination::Metrics,
        ReportingDestination::CloudWatch,
        ReportingDestination::SNS,
        ReportingDestination::Slack,
        ReportingDestination::Email,
    ],
    severity_threshold: ErrorSeverity::Medium,
    ..Default::default()
};
```

## Error Classification

Automatic error classification based on configurable rules:

```rust
let classification = error_handler.classify_error(&error).await?;
println!("Error type: {:?}", classification.error_type);
println!("Severity: {:?}", classification.severity);
println!("Retryable: {}", classification.retryable);
```

## Error Context

Rich error context for better debugging and monitoring:

```rust
let context = ErrorContext::new("service_name".to_string(), "operation_name".to_string())
    .with_request_id("req-123".to_string())
    .with_user_id("user-456".to_string())
    .with_trace_id("trace-789".to_string())
    .with_data("key".to_string(), serde_json::json!("value"));
```

## Running Examples

```bash
# Run the basic usage example
cargo run --example basic_usage

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run --example basic_usage
```

## Integration

To integrate with your existing error handling:

```rust
use error_handling::*;

// Convert your errors to QuantumTradeError
fn convert_error(your_error: YourError) -> QuantumTradeError {
    match your_error {
        YourError::NetworkError(msg) => QuantumTradeError::NetworkConnection {
            host: msg,
        },
        YourError::TimeoutError => QuantumTradeError::RequestTimeout {
            operation: "your_operation".to_string(),
            timeout_ms: 5000,
        },
        // ... other conversions
    }
}

// Use in your error handling
async fn handle_my_error(error: YourError) -> Result<()> {
    let quantum_error = convert_error(error);
    let context = create_error_context("my_service", "my_operation");
    let error_handler = create_default_error_handler();
    
    let result = error_handler.handle_error(&quantum_error, &context).await?;
    // Handle the result...
    Ok(())
}
```

## Dependencies

- `async-trait`: For async trait support
- `chrono`: For timestamp handling
- `serde`: For serialization/deserialization
- `thiserror`: For error derivation
- `tokio`: For async runtime
- `tracing`: For structured logging
- `rand`: For jitter in retry delays

## License

This module is part of the QuantumTrade AI platform and follows the same licensing terms. 