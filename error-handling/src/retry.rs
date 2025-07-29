use super::*;

/// Retry strategy implementation
#[derive(Debug, Clone)]
pub struct RetryStrategy {
    config: RetryConfig,
}

impl RetryStrategy {
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }
    
    pub fn should_retry(&self, error: &QuantumTradeError, attempt: u32) -> bool {
        if attempt >= self.config.max_attempts {
            return false;
        }
        
        let error_type = self.classify_error(error);
        self.config.retryable_errors.contains(&error_type)
    }
    
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::from_millis(0);
        }
        
        let base_delay = self.config.initial_delay_ms as f64;
        let multiplier = self.config.backoff_multiplier;
        let max_delay = self.config.max_delay_ms;
        
        let delay = base_delay * multiplier.powi((attempt - 1) as i32);
        let mut delay_ms = delay.min(max_delay as f64) as u64;
        
        if self.config.jitter {
            use rand::Rng;
            let jitter = rand::thread_rng().gen_range(0.8..1.2);
            delay_ms = (delay_ms as f64 * jitter) as u64;
        }
        
        Duration::from_millis(delay_ms)
    }
    
    fn classify_error(&self, error: &QuantumTradeError) -> ErrorType {
        match error {
            QuantumTradeError::RequestTimeout { .. } => ErrorType::Transient,
            QuantumTradeError::NetworkConnection { .. } => ErrorType::Network,
            QuantumTradeError::DatabaseConnectionFailed { .. } => ErrorType::System,
            QuantumTradeError::RateLimitExceeded { .. } => ErrorType::RateLimit,
            QuantumTradeError::AuthenticationFailed { .. } => ErrorType::Authentication,
            QuantumTradeError::ValidationFailed { .. } => ErrorType::Validation,
            QuantumTradeError::InvalidTradingParameters { .. } => ErrorType::Business,
            _ => ErrorType::System,
        }
    }
}

/// Retry executor with exponential backoff
pub struct RetryExecutor {
    strategy: RetryStrategy,
}

impl RetryExecutor {
    pub fn new(strategy: RetryStrategy) -> Self {
        Self { strategy }
    }
    
    pub async fn execute<F, T, E>(&self, mut operation: F) -> std::result::Result<T, E>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = std::result::Result<T, E>> + Send>>,
        E: From<QuantumTradeError> + fmt::Debug,
    {
        let mut attempt = 0;
        
        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    attempt += 1;
                    
                    // For now, we'll assume all errors are retryable
                    // In practice, you'd need to convert E to QuantumTradeError for classification
                    if attempt >= self.strategy.config.max_attempts {
                        return Err(error);
                    }
                    
                    let delay = self.strategy.calculate_delay(attempt);
                    if delay.as_millis() > 0 {
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
    }
} 