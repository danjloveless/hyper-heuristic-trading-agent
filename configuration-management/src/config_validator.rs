use crate::{errors::ConfigurationError, models::*};
use serde_json::Value;
use std::collections::HashMap;
use tracing::debug;

pub type Result<T> = std::result::Result<T, ConfigurationError>;

pub struct ConfigValidator {
    validation_rules: HashMap<String, Vec<ValidationRule>>,
}

impl ConfigValidator {
    pub async fn new() -> Result<Self> {
        debug!("Initializing configuration validator");
        Ok(Self {
            validation_rules: HashMap::new(),
        })
    }
    
    pub async fn validate_config(&self, key: &str, value: &Value) -> Result<ValidationResult> {
        debug!("Validating configuration for key: {}", key);
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Get validation rules for this key
        if let Some(rules) = self.validation_rules.get(key) {
            for rule in rules {
                match self.validate_rule(rule, value).await {
                    Ok(validation_result) => {
                        errors.extend(validation_result.errors);
                        warnings.extend(validation_result.warnings);
                    }
                    Err(e) => {
                        errors.push(ValidationError {
                            field: key.to_string(),
                            message: format!("Rule validation failed: {}", e),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
            }
        }
        
        // Perform basic validation based on key patterns
        self.validate_by_key_pattern(key, value, &mut errors, &mut warnings).await;
        
        let is_valid = errors.is_empty();
        
        Ok(ValidationResult {
            is_valid,
            errors,
            warnings,
        })
    }
    
    async fn validate_rule(&self, rule: &ValidationRule, value: &Value) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        match &rule.rule_type {
            ValidationRuleType::Range { min, max } => {
                if let Some(num) = value.as_f64() {
                    if num < *min || num > *max {
                        errors.push(ValidationError {
                            field: rule.field.clone(),
                            message: format!("Value {} is outside range [{}, {}]", num, min, max),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
            }
            ValidationRuleType::NotNull => {
                if value.is_null() {
                    errors.push(ValidationError {
                        field: rule.field.clone(),
                        message: "Value cannot be null".to_string(),
                        severity: ValidationSeverity::Error,
                    });
                }
            }
            ValidationRuleType::NotEmpty => {
                if let Some(s) = value.as_str() {
                    if s.trim().is_empty() {
                        errors.push(ValidationError {
                            field: rule.field.clone(),
                            message: "Value cannot be empty".to_string(),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
            }
            ValidationRuleType::Pattern { regex } => {
                if let Some(s) = value.as_str() {
                    // Simple regex validation (in production, use proper regex crate)
                    if !s.contains(regex) {
                        errors.push(ValidationError {
                            field: rule.field.clone(),
                            message: format!("Value does not match pattern: {}", regex),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
            }
            ValidationRuleType::Custom { function } => {
                // Implement custom validation logic
                match function.as_str() {
                    "email" => {
                        if let Some(email) = value.as_str() {
                            if !email.contains('@') || !email.contains('.') {
                                errors.push(ValidationError {
                                    field: rule.field.clone(),
                                    message: format!("Invalid email format: {}", email),
                                    severity: ValidationSeverity::Error,
                                });
                            }
                        }
                    }
                    "url" => {
                        if let Some(url) = value.as_str() {
                            if !url.starts_with("http://") && !url.starts_with("https://") {
                                errors.push(ValidationError {
                                    field: rule.field.clone(),
                                    message: format!("Invalid URL format: {}", url),
                                    severity: ValidationSeverity::Error,
                                });
                            }
                        }
                    }
                    "port_range" => {
                        if let Some(port) = value.as_u64() {
                            if port < 1 || port > 65535 {
                                errors.push(ValidationError {
                                    field: rule.field.clone(),
                                    message: format!("Port {} is outside valid range (1-65535)", port),
                                    severity: ValidationSeverity::Error,
                                });
                            }
                        }
                    }
                    "positive_number" => {
                        if let Some(num) = value.as_f64() {
                            if num <= 0.0 {
                                errors.push(ValidationError {
                                    field: rule.field.clone(),
                                    message: format!("Value {} must be positive", num),
                                    severity: ValidationSeverity::Error,
                                });
                            }
                        }
                    }
                    _ => {
                        warnings.push(ValidationWarning {
                            field: rule.field.clone(),
                            message: format!("Unknown custom validation function: {}", function),
                            suggestion: Some("Add implementation for this validation function".to_string()),
                        });
                    }
                }
            }
        }
        
        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }
    
    async fn validate_by_key_pattern(
        &self,
        key: &str,
        value: &Value,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        // Validate common configuration patterns
        if key.contains("port") {
            if let Some(port) = value.as_u64() {
                if port == 0 || port > 65535 {
                    errors.push(ValidationError {
                        field: key.to_string(),
                        message: format!("Port {} is invalid (must be 1-65535)", port),
                        severity: ValidationSeverity::Error,
                    });
                }
            }
        }
        
        if key.contains("timeout") || key.contains("delay") {
            if let Some(timeout) = value.as_u64() {
                if timeout == 0 {
                    warnings.push(ValidationWarning {
                        field: key.to_string(),
                        message: "Timeout value is 0, which may cause issues".to_string(),
                        suggestion: Some("Consider setting a reasonable timeout value".to_string()),
                    });
                }
            }
        }
        
        if key.contains("url") || key.contains("endpoint") {
            if let Some(url) = value.as_str() {
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    warnings.push(ValidationWarning {
                        field: key.to_string(),
                        message: format!("URL '{}' may not be valid", url),
                        suggestion: Some("Ensure URL starts with http:// or https://".to_string()),
                    });
                }
            }
        }
        
        if key.contains("percentage") || key.contains("ratio") {
            if let Some(percentage) = value.as_f64() {
                if percentage < 0.0 || percentage > 100.0 {
                    errors.push(ValidationError {
                        field: key.to_string(),
                        message: format!("Percentage {} is invalid (must be 0-100)", percentage),
                        severity: ValidationSeverity::Error,
                    });
                }
            }
        }
    }
    
    pub async fn add_validation_rule(&mut self, key: &str, rule: ValidationRule) -> Result<()> {
        debug!("Adding validation rule for key: {}", key);
        
        self.validation_rules
            .entry(key.to_string())
            .or_insert_with(Vec::new)
            .push(rule);
        
        Ok(())
    }
    
    pub async fn remove_validation_rules(&mut self, key: &str) -> Result<()> {
        debug!("Removing validation rules for key: {}", key);
        
        self.validation_rules.remove(key);
        Ok(())
    }
    
    pub async fn validate_service_config(&self, config: &ServiceConfig) -> Result<ValidationResult> {
        debug!("Validating service configuration");
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Validate service name
        if config.service_name.trim().is_empty() {
            errors.push(ValidationError {
                field: "service_name".to_string(),
                message: "Service name cannot be empty".to_string(),
                severity: ValidationSeverity::Error,
            });
        }
        
        // Validate database configuration
        match self.validate_database_config(&config.database).await {
            Ok(db_validation) => {
                errors.extend(db_validation.errors);
                warnings.extend(db_validation.warnings);
            }
            Err(_) => {
                errors.push(ValidationError {
                    field: "database".to_string(),
                    message: "Database configuration validation failed".to_string(),
                    severity: ValidationSeverity::Error,
                });
            }
        }
        
        // Validate API configuration
        match self.validate_api_config(&config.apis).await {
            Ok(api_validation) => {
                errors.extend(api_validation.errors);
                warnings.extend(api_validation.warnings);
            }
            Err(_) => {
                errors.push(ValidationError {
                    field: "apis".to_string(),
                    message: "API configuration validation failed".to_string(),
                    severity: ValidationSeverity::Error,
                });
            }
        }
        
        let is_valid = errors.is_empty();
        
        Ok(ValidationResult {
            is_valid,
            errors,
            warnings,
        })
    }
    
    async fn validate_database_config(&self, config: &DatabaseConfig) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let warnings = Vec::new();
        
        // Validate ClickHouse configuration
        if config.clickhouse.host.trim().is_empty() {
            errors.push(ValidationError {
                field: "database.clickhouse.host".to_string(),
                message: "ClickHouse host cannot be empty".to_string(),
                severity: ValidationSeverity::Error,
            });
        }
        
        if config.clickhouse.port == 0 {
            errors.push(ValidationError {
                field: "database.clickhouse.port".to_string(),
                message: "ClickHouse port cannot be 0".to_string(),
                severity: ValidationSeverity::Error,
            });
        }
        
        // Validate Redis configuration
        if config.redis.host.trim().is_empty() {
            errors.push(ValidationError {
                field: "database.redis.host".to_string(),
                message: "Redis host cannot be empty".to_string(),
                severity: ValidationSeverity::Error,
            });
        }
        
        if config.redis.port == 0 {
            errors.push(ValidationError {
                field: "database.redis.port".to_string(),
                message: "Redis port cannot be 0".to_string(),
                severity: ValidationSeverity::Error,
            });
        }
        
        let is_valid = errors.is_empty();
        
        Ok(ValidationResult {
            is_valid,
            errors,
            warnings,
        })
    }
    
    async fn validate_api_config(&self, config: &ApiConfig) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Validate Alpha Vantage configuration
        if config.alpha_vantage.api_key.trim().is_empty() {
            errors.push(ValidationError {
                field: "apis.alpha_vantage.api_key".to_string(),
                message: "Alpha Vantage API key cannot be empty".to_string(),
                severity: ValidationSeverity::Error,
            });
        }
        
        if config.alpha_vantage.base_url.trim().is_empty() {
            errors.push(ValidationError {
                field: "apis.alpha_vantage.base_url".to_string(),
                message: "Alpha Vantage base URL cannot be empty".to_string(),
                severity: ValidationSeverity::Error,
            });
        }
        
        // Validate timeout values
        if config.timeout_ms == 0 {
            warnings.push(ValidationWarning {
                field: "apis.timeout_ms".to_string(),
                message: "API timeout is 0, which may cause issues".to_string(),
                suggestion: Some("Consider setting a reasonable timeout value".to_string()),
            });
        }
        
        let is_valid = errors.is_empty();
        
        Ok(ValidationResult {
            is_valid,
            errors,
            warnings,
        })
    }
} 