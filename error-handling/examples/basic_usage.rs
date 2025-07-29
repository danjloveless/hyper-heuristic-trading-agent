use error_handling::*;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();
    
    println!("=== QuantumTrade AI Error Handling System Demo ===\n");
    
    // Create a default error handler
    let error_handler = create_default_error_handler();
    
    // Example 1: Handle a network connection error
    println!("1. Handling Network Connection Error:");
    let network_error = QuantumTradeError::NetworkConnection {
        host: "api.example.com".to_string(),
    };
    let context = create_error_context("market_data_service", "fetch_price_data")
        .with_request_id("req-123".to_string())
        .with_user_id("user-456".to_string());
    
    match error_handler.handle_error(&network_error, &context).await {
        Ok(result) => {
            println!("   Action: {:?}", result.action);
            println!("   Retry after: {:?}", result.retry_after);
            println!("   Should alert: {}", result.should_alert);
            if let Some(fallback) = result.fallback_value {
                println!("   Fallback value: {}", fallback);
            }
        }
        Err(e) => println!("   Error handling failed: {}", e),
    }
    
    println!();
    
    // Example 2: Handle a validation error
    println!("2. Handling Validation Error:");
    let validation_error = QuantumTradeError::ValidationFailed {
        field: "price".to_string(),
        message: "Price must be positive".to_string(),
    };
    let context = create_error_context("trading_service", "place_order");
    
    match error_handler.handle_error(&validation_error, &context).await {
        Ok(result) => {
            println!("   Action: {:?}", result.action);
            println!("   Should alert: {}", result.should_alert);
        }
        Err(e) => println!("   Error handling failed: {}", e),
    }
    
    println!();
    
    // Example 3: Test circuit breaker
    println!("3. Testing Circuit Breaker:");
    let mut circuit_breaker = CircuitBreaker::new(
        "database_service".to_string(),
        CircuitBreakerConfig::default(),
    );
    
    println!("   Initial state: {:?}", circuit_breaker.get_state());
    
    // Record some failures to trip the circuit breaker
    for i in 1..=6 {
        circuit_breaker.record_failure();
        println!("   After {} failures: {:?}", i, circuit_breaker.get_state());
    }
    
    // Test if we can execute
    println!("   Can execute: {}", circuit_breaker.can_execute());
    
    // Reset the circuit breaker
    circuit_breaker.reset();
    println!("   After reset: {:?}", circuit_breaker.get_state());
    println!("   Can execute: {}", circuit_breaker.can_execute());
    
    println!();
    
    // Example 4: Test retry strategy
    println!("4. Testing Retry Strategy:");
    let retry_config = RetryConfig::default();
    let retry_strategy = RetryStrategy::new(retry_config);
    
    let timeout_error = QuantumTradeError::RequestTimeout {
        operation: "api_call".to_string(),
        timeout_ms: 5000,
    };
    
    for attempt in 1..=4 {
        let should_retry = retry_strategy.should_retry(&timeout_error, attempt);
        let delay = retry_strategy.calculate_delay(attempt);
        println!("   Attempt {}: should_retry={}, delay={:?}", attempt, should_retry, delay);
    }
    
    println!();
    
    // Example 5: Test error classification
    println!("5. Testing Error Classification:");
    let errors = vec![
        QuantumTradeError::RequestTimeout {
            operation: "fetch_data".to_string(),
            timeout_ms: 3000,
        },
        QuantumTradeError::AuthenticationFailed {
            reason: "Invalid token".to_string(),
        },
        QuantumTradeError::DatabaseConnectionFailed {
            host: "db.example.com".to_string(),
        },
    ];
    
    for error in errors {
        let classification = error_handler.classify_error(&error).await?;
        println!("   {:?}:", error);
        println!("     Type: {:?}", classification.error_type);
        println!("     Severity: {:?}", classification.severity);
        println!("     Retryable: {}", classification.retryable);
    }
    
    println!("\n=== Demo Complete ===");
    
    Ok(())
} 