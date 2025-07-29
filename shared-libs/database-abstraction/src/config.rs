use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatabaseConfig {
    pub clickhouse: ClickHouseConfig,
    pub redis: RedisConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClickHouseConfig {
    pub url: String,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
    #[serde(default = "default_connection_timeout")]
    pub connection_timeout: Duration,
    #[serde(default = "default_query_timeout")]
    pub query_timeout: Duration,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    #[serde(default = "default_retry_attempts")]
    pub retry_attempts: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RedisConfig {
    pub url: String,
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
    #[serde(default = "default_connection_timeout")]
    pub connection_timeout: Duration,
    #[serde(default = "default_default_ttl")]
    pub default_ttl: Duration,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

fn default_connection_timeout() -> Duration {
    Duration::from_secs(30)
}

fn default_query_timeout() -> Duration {
    Duration::from_secs(60)
}

fn default_pool_size() -> usize {
    10
}

fn default_default_ttl() -> Duration {
    Duration::from_secs(3600) // 1 hour
}

fn default_max_connections() -> u32 {
    100
}

fn default_retry_attempts() -> u32 {
    3
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            clickhouse: ClickHouseConfig {
                url: "http://localhost:8123".to_string(),
                database: "quantumtrade".to_string(),
                username: Some("default".to_string()),
                password: None,
                connection_timeout: default_connection_timeout(),
                query_timeout: default_query_timeout(),
                max_connections: default_max_connections(),
                retry_attempts: default_retry_attempts(),
            },
            redis: RedisConfig {
                url: "redis://localhost:6379".to_string(),
                pool_size: default_pool_size(),
                connection_timeout: default_connection_timeout(),
                default_ttl: default_default_ttl(),
                max_connections: default_max_connections(),
            },
        }
    }
}