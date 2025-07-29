use async_trait::async_trait;
use bb8_redis::{bb8, RedisConnectionManager};
use chrono::Duration;
use redis::AsyncCommands;
use serde::{de::DeserializeOwned, Serialize};
use shared_types::ServiceStatus;
use tracing::{error, info, warn};

use crate::{
    config::RedisConfig,
    errors::DatabaseError,
    health::{HealthCheck, HealthStatus},
    traits::CacheClient,
};

pub struct RedisClient {
    pool: bb8::Pool<RedisConnectionManager>,
    config: RedisConfig,
}

impl RedisClient {
    pub async fn new(config: RedisConfig) -> Result<Self, DatabaseError> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| DatabaseError::ConnectionError {
                message: format!("Failed to create Redis client: {}", e),
            })?;

        let manager = RedisConnectionManager::new(client);
        let pool = bb8::Pool::builder()
            .max_size(config.max_connections)
            .build(manager)
            .await
            .map_err(|e| DatabaseError::PoolError {
                message: format!("Failed to create Redis connection pool: {}", e),
            })?;

        // Test connection
        let mut conn = pool
            .get()
            .await
            .map_err(|e| DatabaseError::ConnectionError {
                message: format!("Failed to get Redis connection: {}", e),
            })?;

        let _: String = conn
            .ping()
            .await
            .map_err(|e| DatabaseError::ConnectionError {
                message: format!("Redis ping failed: {}", e),
            })?;

        info!("Successfully connected to Redis: {}", config.url);

        Ok(Self { pool, config })
    }

    fn cache_key(&self, key: &str) -> String {
        format!("quantumtrade:{}", key)
    }

    async fn get_connection(&self) -> Result<bb8::PooledConnection<RedisConnectionManager>, DatabaseError> {
        self.pool.get().await.map_err(|e| DatabaseError::PoolError {
            message: format!("Failed to get Redis connection from pool: {}", e),
        })
    }
}

#[async_trait]
impl CacheClient for RedisClient {
    async fn cache_set<T: Serialize + Send>(
        &self,
        key: &str,
        value: &T,
        ttl: Option<Duration>,
    ) -> Result<(), DatabaseError> {
        let mut conn = self.get_connection().await?;
        let cache_key = self.cache_key(key);
        let serialized = serde_json::to_string(value)
            .map_err(|e| DatabaseError::SerializationError(e))?;

        if let Some(ttl) = ttl {
            let ttl_seconds = ttl.num_seconds();
            if ttl_seconds > 0 {
                conn.set_ex(cache_key, serialized, ttl_seconds as u64)
                    .await
                    .map_err(DatabaseError::RedisError)?;
            } else {
                conn.set(cache_key, serialized)
                    .await
                    .map_err(DatabaseError::RedisError)?;
            }
        } else {
            let default_ttl = self.config.default_ttl.num_seconds() as u64;
            conn.set_ex(cache_key, serialized, default_ttl)
                .await
                .map_err(DatabaseError::RedisError)?;
        }

        Ok(())
    }

    async fn cache_get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, DatabaseError> {
        let mut conn = self.get_connection().await?;
        let cache_key = self.cache_key(key);

        let result: Option<String> = conn
            .get(cache_key)
            .await
            .map_err(DatabaseError::RedisError)?;

        match result {
            Some(data) => {
                let deserialized = serde_json::from_str(&data)
                    .map_err(|e| DatabaseError::SerializationError(e))?;
                Ok(Some(deserialized))
            }
            None => Ok(None),
        }
    }

    async fn cache_delete(&self, key: &str) -> Result<(), DatabaseError> {
        let mut conn = self.get_connection().await?;
        let cache_key = self.cache_key(key);

        conn.del(cache_key)
            .await
            .map_err(DatabaseError::RedisError)?;

        Ok(())
    }

    async fn cache_exists(&self, key: &str) -> Result<bool, DatabaseError> {
        let mut conn = self.get_connection().await?;
        let cache_key = self.cache_key(key);

        let exists: bool = conn
            .exists(cache_key)
            .await
            .map_err(DatabaseError::RedisError)?;

        Ok(exists)
    }

    async fn cache_expire(&self, key: &str, ttl: Duration) -> Result<(), DatabaseError> {
        let mut conn = self.get_connection().await?;
        let cache_key = self.cache_key(key);
        let ttl_seconds = ttl.num_seconds();

        if ttl_seconds > 0 {
            conn.expire(cache_key, ttl_seconds as u64)
                .await
                .map_err(DatabaseError::RedisError)?;
        }

        Ok(())
    }

    async fn cache_flush(&self) -> Result<(), DatabaseError> {
        let mut conn = self.get_connection().await?;
        
        conn.flushdb()
            .await
            .map_err(DatabaseError::RedisError)?;

        Ok(())
    }

    async fn cache_info(&self) -> Result<serde_json::Value, DatabaseError> {
        let mut conn = self.get_connection().await?;
        
        let info: String = conn
            .info("memory")
            .await
            .map_err(DatabaseError::RedisError)?;

        // Parse Redis INFO response into JSON
        let mut info_map = serde_json::Map::new();
        for line in info.lines() {
            if let Some((key, value)) = line.split_once(':') {
                info_map.insert(key.to_string(), serde_json::Value::String(value.to_string()));
            }
        }

        Ok(serde_json::Value::Object(info_map))
    }
}

#[async_trait]
impl crate::traits::DatabaseClient for RedisClient {
    // For Redis, we implement cache operations and delegate time-series operations to ClickHouse
    async fn insert_market_data(&self, _data: &[shared_types::MarketData]) -> Result<(), DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Market data insertion should be handled by ClickHouse client".to_string(),
        })
    }

    async fn get_market_data(
        &self,
        _symbol: &str,
        _start: chrono::DateTime<chrono::Utc>,
        _end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<shared_types::MarketData>, DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Market data retrieval should be handled by ClickHouse client".to_string(),
        })
    }

    async fn get_latest_market_data(&self, _symbol: &str) -> Result<Option<shared_types::MarketData>, DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Market data retrieval should be handled by ClickHouse client".to_string(),
        })
    }

    async fn insert_sentiment_data(&self, _data: &[shared_types::SentimentData]) -> Result<(), DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Sentiment data insertion should be handled by ClickHouse client".to_string(),
        })
    }

    async fn get_sentiment_data(
        &self,
        _symbol: &str,
        _start: chrono::DateTime<chrono::Utc>,
        _end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<shared_types::SentimentData>, DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Sentiment data retrieval should be handled by ClickHouse client".to_string(),
        })
    }

    async fn get_aggregated_sentiment(
        &self,
        _symbol: &str,
        _timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<shared_types::AggregatedSentiment>, DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Sentiment data aggregation should be handled by ClickHouse client".to_string(),
        })
    }

    async fn insert_features(&self, _features: &[shared_types::FeatureSet]) -> Result<(), DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Feature insertion should be handled by ClickHouse client".to_string(),
        })
    }

    async fn get_features(
        &self,
        _symbol: &str,
        _timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<shared_types::FeatureSet>, DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Feature retrieval should be handled by ClickHouse client".to_string(),
        })
    }

    async fn get_latest_features(&self, _symbol: &str) -> Result<Option<shared_types::FeatureSet>, DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Feature retrieval should be handled by ClickHouse client".to_string(),
        })
    }

    async fn insert_technical_indicators(&self, _indicators: &[shared_types::TechnicalIndicators]) -> Result<(), DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Technical indicators insertion should be handled by ClickHouse client".to_string(),
        })
    }

    async fn insert_predictions(&self, _predictions: &[shared_types::PredictionResult]) -> Result<(), DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Prediction insertion should be handled by ClickHouse client".to_string(),
        })
    }

    async fn get_predictions(
        &self,
        _symbol: &str,
        _start: chrono::DateTime<chrono::Utc>,
        _end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<shared_types::PredictionResult>, DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Prediction retrieval should be handled by ClickHouse client".to_string(),
        })
    }

    async fn insert_prediction_outcomes(&self, _outcomes: &[shared_types::PredictionOutcome]) -> Result<(), DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Prediction outcome insertion should be handled by ClickHouse client".to_string(),
        })
    }

    async fn insert_strategy_performance(&self, _performance: &[shared_types::StrategyPerformance]) -> Result<(), DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Strategy performance insertion should be handled by ClickHouse client".to_string(),
        })
    }

    async fn get_strategy_performance(
        &self,
        _strategy_name: &str,
        _symbol: Option<&str>,
        _start: chrono::DateTime<chrono::Utc>,
        _end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<shared_types::StrategyPerformance>, DatabaseError> {
        Err(DatabaseError::QueryError {
            message: "Strategy performance retrieval should be handled by ClickHouse client".to_string(),
        })
    }

    async fn health_check(&self) -> Result<HealthStatus, DatabaseError> {
        let start = std::time::Instant::now();
        let mut health = HealthStatus::new("Redis".to_string());

        // Connection check
        let connection_result = async {
            let mut conn = self.get_connection().await?;
            let _: String = conn.ping().await?;
            Ok::<(), DatabaseError>(())
        }.await;
        
        let connection_latency = start.elapsed().as_millis() as u64;

        let connection_check = match connection_result {
            Ok(_) => HealthCheck {
                name: "connection".to_string(),
                status: ServiceStatus::Healthy,
                latency_ms: connection_latency,
                error: None,
            },
            Err(e) => HealthCheck {
                name: "connection".to_string(),
                status: ServiceStatus::Unhealthy,
                latency_ms: connection_latency,
                error: Some(e.to_string()),
            },
        };

        health.add_check(connection_check);

        // Memory check
        let memory_start = std::time::Instant::now();
        let memory_result = self.cache_info().await;
        let memory_latency = memory_start.elapsed().as_millis() as u64;

        let memory_check = match memory_result {
            Ok(_) => HealthCheck {
                name: "memory_info".to_string(),
                status: ServiceStatus::Healthy,
                latency_ms: memory_latency,
                error: None,
            },
            Err(e) => HealthCheck {
                name: "memory_info".to_string(),
                status: ServiceStatus::Degraded,
                latency_ms: memory_latency,
                error: Some(e.to_string()),
            },
        };

        health.add_check(memory_check);

        Ok(health)
    }
}