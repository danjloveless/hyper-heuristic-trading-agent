pub mod clickhouse;
pub mod redis;
pub mod traits;
pub mod errors;
pub mod config;
pub mod health;

pub use traits::*;
pub use errors::*;
pub use config::*;
pub use health::*;

use async_trait::async_trait;
use std::sync::Arc;

/// Database manager that provides unified access to both ClickHouse and Redis
pub struct DatabaseManager {
    clickhouse: Arc<clickhouse::ClickHouseClient>,
    redis: Arc<redis::RedisClient>,
    config: DatabaseConfig,
}

impl DatabaseManager {
    pub async fn new(config: DatabaseConfig) -> Result<Self, DatabaseError> {
        let clickhouse = Arc::new(
            clickhouse::ClickHouseClient::new(config.clickhouse.clone()).await?
        );
        
        let redis = Arc::new(
            redis::RedisClient::new(config.redis.clone()).await?
        );

        Ok(Self {
            clickhouse,
            redis, 
            config,
        })
    }

    pub fn clickhouse(&self) -> Arc<clickhouse::ClickHouseClient> {
        self.clickhouse.clone()
    }

    pub fn redis(&self) -> Arc<redis::RedisClient> {
        self.redis.clone()
    }

    pub async fn run_migrations(&self) -> Result<(), DatabaseError> {
        self.clickhouse.run_migrations().await
    }

    pub async fn health_check(&self) -> Result<DatabaseHealthStatus, DatabaseError> {
        let clickhouse_health = self.clickhouse.health_check().await?;
        let redis_health = self.redis.health_check().await?;

        Ok(DatabaseHealthStatus {
            clickhouse_healthy: matches!(clickhouse_health.status, shared_types::ServiceStatus::Healthy),
            redis_healthy: matches!(redis_health.status, shared_types::ServiceStatus::Healthy),
            schema_version: 1, // TODO: Implement schema versioning
            connection_count: 0, // TODO: Implement connection counting
            last_successful_query: chrono::Utc::now(),
        })
    }
}