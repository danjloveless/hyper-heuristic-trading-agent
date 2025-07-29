// QuantumTrade AI - Error Handling Module
// Complete implementation following the technical specification

pub mod config;
pub mod error_types;
pub mod error_handler;
pub mod retry;
pub mod circuit_breaker;
pub mod fallback;
pub mod reporter;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;
use thiserror::Error;


// Re-export main types
pub use config::*;
pub use error_types::*;
pub use error_handler::*;
pub use retry::*;
pub use circuit_breaker::*;
pub use fallback::*;
pub use reporter::*;

/// Result type for the error handling module
pub type Result<T> = std::result::Result<T, ErrorHandlingError>;

// ================================================================================================
// CONVENIENCE MACROS AND HELPERS
// ================================================================================================

// ================================================================================================
// CONVENIENCE MACROS AND HELPERS
// ================================================================================================

/// Macro for easy error handling with automatic context
#[macro_export]
macro_rules! handle_error {
    ($handler:expr, $error:expr, $service:expr, $operation:expr) => {{
        let context = ErrorContext::new($service.to_string(), $operation.to_string());
        $handler.handle_error(&$error, &context).await
    }};
    
    ($handler:expr, $error:expr, $context:expr) => {{
        $handler.handle_error(&$error, &$context).await
    }};
}

/// Helper function to create error handler with default configuration
pub fn create_default_error_handler() -> ErrorHandlerImpl {
    ErrorHandlerImpl::new(ErrorHandlingConfig::default())
}

/// Helper function to create error context
pub fn create_error_context(service: &str, operation: &str) -> ErrorContext {
    ErrorContext::new(service.to_string(), operation.to_string())
}

// ================================================================================================
// TESTS MODULE
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_classification() {
        let config = ClassificationConfig::default();
        assert!(!config.rules.is_empty());
        
        // Test that we have classifications for common error types
        let rule_patterns: Vec<&String> = config.rules.iter().map(|r| &r.pattern).collect();
        assert!(rule_patterns.contains(&&"RequestTimeout".to_string()));
        assert!(rule_patterns.contains(&&"NetworkConnection".to_string()));
        assert!(rule_patterns.contains(&&"AuthenticationFailed".to_string()));
    }
    
    #[test]
    fn test_retry_strategy() {
        let config = RetryConfig::default();
        let strategy = RetryStrategy::new(config);
        
        let error = QuantumTradeError::RequestTimeout {
            operation: "test".to_string(),
            timeout_ms: 5000,
        };
        
        // Should retry transient errors
        assert!(strategy.should_retry(&error, 1));
        assert!(strategy.should_retry(&error, 2));
        assert!(!strategy.should_retry(&error, 3)); // Max attempts reached
        
        // Test delay calculation
        let delay1 = strategy.calculate_delay(1);
        let delay2 = strategy.calculate_delay(2);
        assert!(delay2 > delay1); // Exponential backoff
    }
    
    #[test]
    fn test_circuit_breaker() {
        let config = CircuitBreakerConfig::default();
        let mut circuit_breaker = CircuitBreaker::new("test_service".to_string(), config);
        
        // Initially closed
        assert_eq!(circuit_breaker.get_state(), CircuitState::Closed);
        assert!(circuit_breaker.can_execute());
        
        // Record failures to trip the circuit breaker
        for _ in 0..5 {
            circuit_breaker.record_failure();
        }
        
        // Should be open now
        assert_eq!(circuit_breaker.get_state(), CircuitState::Open);
        assert!(!circuit_breaker.can_execute());
        
        // Reset and test success
        circuit_breaker.reset();
        assert_eq!(circuit_breaker.get_state(), CircuitState::Closed);
        circuit_breaker.record_success();
        assert_eq!(circuit_breaker.get_stats().success_count, 1);
    }
    
    #[tokio::test]
    async fn test_error_handler() {
        let handler = create_default_error_handler();
        let error = QuantumTradeError::NetworkConnection {
            host: "test.example.com".to_string(),
        };
        let context = create_error_context("test_service", "test_operation");
        
        let result = handler.handle_error(&error, &context).await.unwrap();
        
        // Network errors should typically be retried
        assert!(matches!(result.action, ErrorAction::Retry | ErrorAction::UseFallback));
    }
    
    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_service".to_string(), "test_operation".to_string())
            .with_request_id("req-123".to_string())
            .with_user_id("user-456".to_string())
            .with_trace_id("trace-789".to_string())
            .with_data("key".to_string(), serde_json::json!("value"));
        
        assert_eq!(context.service_name, "test_service");
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.request_id, Some("req-123".to_string()));
        assert_eq!(context.user_id, Some("user-456".to_string()));
        assert_eq!(context.trace_id, Some("trace-789".to_string()));
        assert_eq!(context.additional_data.get("key"), Some(&serde_json::json!("value")));
    }
} 