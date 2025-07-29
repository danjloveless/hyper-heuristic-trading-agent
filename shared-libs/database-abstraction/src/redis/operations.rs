use chrono::Duration;
use serde::Serialize;
use shared_types::*;

use crate::{errors::DatabaseError, redis::RedisClient, traits::CacheClient};

pub struct RedisOperations<'a> {
    client: &'a RedisClient,
}

impl<'a> RedisOperations<'a> {
    pub fn new(client: &'a RedisClient) -> Self {
        Self { client }
    }

    // Market Data Cache Operations
    pub async fn cache_market_data(
        &self,
        symbol: &str,
        data: &MarketData,
        ttl: Option<Duration>,
    ) -> Result<(), DatabaseError> {
        let key = format!("market_data:{}:{}", symbol, data.timestamp.timestamp());
        self.client.cache_set(&key, data, ttl).await
    }

    pub async fn get_cached_market_data(
        &self,
        symbol: &str,
        timestamp: i64,
    ) -> Result<Option<MarketData>, DatabaseError> {
        let key = format!("market_data:{}:{}", symbol, timestamp);
        self.client.cache_get(&key).await
    }

    // Feature Cache Operations
    pub async fn cache_features(
        &self,
        symbol: &str,
        features: &FeatureSet,
        ttl: Option<Duration>,
    ) -> Result<(), DatabaseError> {
        let key = format!("features:{}:{}", symbol, features.timestamp.timestamp());
        self.client.cache_set(&key, features, ttl).await
    }

    pub async fn get_cached_features(
        &self,
        symbol: &str,
        timestamp: i64,
    ) -> Result<Option<FeatureSet>, DatabaseError> {
        let key = format!("features:{}:{}", symbol, timestamp);
        self.client.cache_get(&key).await
    }

    // Prediction Cache Operations
    pub async fn cache_prediction(
        &self,
        prediction: &PredictionResult,
        ttl: Option<Duration>,
    ) -> Result<(), DatabaseError> {
        let key = format!("prediction:{}", prediction.prediction_id);
        self.client.cache_set(&key, prediction, ttl).await
    }

    pub async fn get_cached_prediction(
        &self,
        prediction_id: &str,
    ) -> Result<Option<PredictionResult>, DatabaseError> {
        let key = format!("prediction:{}", prediction_id);
        self.client.cache_get(&key).await
    }

    // Real-time Data Operations
    pub async fn set_realtime_price(
        &self,
        symbol: &str,
        price: f64,
        ttl: Option<Duration>,
    ) -> Result<(), DatabaseError> {
        let key = format!("realtime:price:{}", symbol);
        self.client.cache_set(&key, &price, ttl).await
    }

    pub async fn get_realtime_price(&self, symbol: &str) -> Result<Option<f64>, DatabaseError> {
        let key = format!("realtime:price:{}", symbol);
        self.client.cache_get(&key).await
    }

    // Session and Rate Limiting
    pub async fn increment_rate_limit(
        &self,
        user_id: &str,
        window: Duration,
    ) -> Result<u32, DatabaseError> {
        let key = format!("rate_limit:{}:{}", user_id, chrono::Utc::now().timestamp() / window.num_seconds());
        
        // This is a simplified implementation - in production, use Redis INCR with expiration
        let current: Option<u32> = self.client.cache_get(&key).await?;
        let new_count = current.unwrap_or(0) + 1;
        self.client.cache_set(&key, &new_count, Some(window)).await?;
        
        Ok(new_count)
    }

    // Cache Statistics
    pub async fn get_cache_stats(&self) -> Result<CacheStats, DatabaseError> {
        let info = self.client.cache_info().await?;
        
        Ok(CacheStats {
            total_keys: 0, // Would need to implement key counting
            memory_usage: info.get("used_memory")
                .and_then(serde_json::Value::as_str)
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0),
            hit_rate: 0.0, // Would need to implement hit/miss tracking
            last_updated: chrono::Utc::now(),
        })
    }
}

#[derive(Debug, Serialize)]
pub struct CacheStats {
    pub total_keys: u64,
    pub memory_usage: u64,
    pub hit_rate: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}