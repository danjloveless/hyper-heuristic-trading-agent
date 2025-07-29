use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigurationError {
    #[error("Configuration not found: {key}")]
    ConfigNotFound { key: String },

    #[error("Invalid configuration value: {key} - {message}")]
    InvalidValue { key: String, message: String },

    #[error("Configuration validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("Secret not found: {secret_name}")]
    SecretNotFound { secret_name: String },

    #[error("Secret access denied: {secret_name}")]
    SecretAccessDenied { secret_name: String },

    #[error("AWS Parameter Store error: {message}")]
    ParameterStoreError { message: String },

    #[error("AWS Secrets Manager error: {message}")]
    SecretsManagerError { message: String },

    #[error("S3 configuration error: {message}")]
    S3Error { message: String },

    #[error("Configuration parsing error: {message}")]
    ParsingError { message: String },

    #[error("Configuration reload failed: {message}")]
    ReloadError { message: String },

    #[error("Feature flag error: {message}")]
    FeatureFlagError { message: String },

    #[error("Configuration watcher error: {message}")]
    WatcherError { message: String },

    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl From<aws_sdk_ssm::Error> for ConfigurationError {
    fn from(err: aws_sdk_ssm::Error) -> Self {
        ConfigurationError::ParameterStoreError {
            message: err.to_string(),
        }
    }
}

impl From<aws_sdk_secretsmanager::Error> for ConfigurationError {
    fn from(err: aws_sdk_secretsmanager::Error) -> Self {
        ConfigurationError::SecretsManagerError {
            message: err.to_string(),
        }
    }
}

impl From<aws_sdk_s3::Error> for ConfigurationError {
    fn from(err: aws_sdk_s3::Error) -> Self {
        ConfigurationError::S3Error {
            message: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for ConfigurationError {
    fn from(err: serde_json::Error) -> Self {
        ConfigurationError::ParsingError {
            message: err.to_string(),
        }
    }
}

impl From<config::ConfigError> for ConfigurationError {
    fn from(err: config::ConfigError) -> Self {
        ConfigurationError::ParsingError {
            message: err.to_string(),
        }
    }
} 