use super::*;

/// Main error handler trait
#[async_trait]
pub trait ErrorHandler: Send + Sync {
    async fn handle_error(
        &self,
        error: &QuantumTradeError,
        context: &ErrorContext,
    ) -> Result<ErrorHandlingResult>;
    
    async fn classify_error(&self, error: &QuantumTradeError) -> Result<ErrorClassification>;
    async fn should_retry(&self, error: &QuantumTradeError, attempt: u32) -> Result<bool>;
    async fn execute_fallback(&self, error: &QuantumTradeError, context: &ErrorContext) -> Result<Option<serde_json::Value>>;
    async fn report_error(&self, error: &QuantumTradeError, context: &ErrorContext) -> Result<()>;
}

/// Error handling result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingResult {
    pub action: ErrorAction,
    pub retry_after: Option<Duration>,
    pub fallback_value: Option<serde_json::Value>,
    pub should_alert: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorAction {
    Retry,
    UseFallback,
    FailFast,
    Escalate,
}

/// Main error handler implementation
pub struct ErrorHandlerImpl {
    config: ErrorHandlingConfig,
    retry_strategy: RetryStrategy,
    circuit_breakers: std::collections::HashMap<String, CircuitBreaker>,
    fallback_handler: FallbackHandler,
    error_reporter: ErrorReporter,
}

impl ErrorHandlerImpl {
    pub fn new(config: ErrorHandlingConfig) -> Self {
        let retry_strategy = RetryStrategy::new(config.retry.clone());
        let fallback_handler = FallbackHandler::new(config.fallback.clone());
        let error_reporter = ErrorReporter::new(config.reporting.clone());
        
        Self {
            config,
            retry_strategy,
            circuit_breakers: std::collections::HashMap::new(),
            fallback_handler,
            error_reporter,
        }
    }
    
    pub fn get_or_create_circuit_breaker(&mut self, service: &str) -> &mut CircuitBreaker {
        self.circuit_breakers
            .entry(service.to_string())
            .or_insert_with(|| CircuitBreaker::new(service.to_string(), self.config.circuit_breaker.clone()))
    }
}

#[async_trait]
impl ErrorHandler for ErrorHandlerImpl {
    async fn handle_error(
        &self,
        error: &QuantumTradeError,
        context: &ErrorContext,
    ) -> Result<ErrorHandlingResult> {
        // Classify the error
        let classification = self.classify_error(error).await?;
        
        // Determine the action based on error classification
        let action = match classification.error_type {
            ErrorType::Transient | ErrorType::Network => {
                if classification.retryable {
                    ErrorAction::Retry
                } else {
                    ErrorAction::UseFallback
                }
            },
            ErrorType::Authentication | ErrorType::Validation => ErrorAction::FailFast,
            ErrorType::System => {
                if classification.severity == ErrorSeverity::Critical {
                    ErrorAction::Escalate
                } else {
                    ErrorAction::UseFallback
                }
            },
            _ => ErrorAction::UseFallback,
        };
        
        // Get fallback value if needed
        let fallback_value = match action {
            ErrorAction::UseFallback => self.execute_fallback(error, context).await?,
            _ => None,
        };
        
        // Calculate retry delay
        let retry_after = match action {
            ErrorAction::Retry => classification.timeout_ms.map(Duration::from_millis),
            _ => None,
        };
        
        // Determine if we should alert
        let should_alert = classification.severity == ErrorSeverity::Critical 
            || classification.severity == ErrorSeverity::High;
        
        // Report the error
        if should_alert || classification.severity >= self.config.reporting.severity_threshold {
            self.report_error(error, context).await?;
        }
        
        Ok(ErrorHandlingResult {
            action,
            retry_after,
            fallback_value,
            should_alert,
        })
    }
    
    async fn classify_error(&self, error: &QuantumTradeError) -> Result<ErrorClassification> {
        let error_name = format!("{:?}", error).split('(').next().unwrap_or("Unknown").to_string();
        
        // Find matching classification rule
        for rule in &self.config.classification.rules {
            if error_name.contains(&rule.pattern) {
                return Ok(rule.classification.clone());
            }
        }
        
        // Return default classification
        Ok(self.config.classification.default_classification.clone())
    }
    
    async fn should_retry(&self, error: &QuantumTradeError, attempt: u32) -> Result<bool> {
        Ok(self.retry_strategy.should_retry(error, attempt))
    }
    
    async fn execute_fallback(&self, error: &QuantumTradeError, context: &ErrorContext) -> Result<Option<serde_json::Value>> {
        self.fallback_handler.execute_fallback(error, context).await
    }
    
    async fn report_error(&self, error: &QuantumTradeError, context: &ErrorContext) -> Result<()> {
        self.error_reporter.report_error(error, context).await
    }
} 