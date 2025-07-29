use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use serde::{Serialize, de::DeserializeOwned};
use shared_types::*;
use crate::errors::DatabaseError;
use crate::health::HealthStatus;

#[async_trait]
pub trait DatabaseClient: Send + Sync {
    // Market Data Operations
    async fn insert_market_data(&self, data: &[MarketData]) -> Result<(), DatabaseError>;
    async fn get_market_data(
        &self, 
        symbol: &str, 
        start: DateTime<Utc>, 
        end: DateTime<Utc>
    ) -> Result<Vec<MarketData>, DatabaseError>;
    async fn get_latest_market_data(&self, symbol: &str) -> Result<Option<MarketData>, DatabaseError>;
    
    // Sentiment Data Operations
    async fn insert_sentiment_data(&self, data: &[SentimentData]) -> Result<(), DatabaseError>;
    async fn get_sentiment_data(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>
    ) -> Result<Vec<SentimentData>, DatabaseError>;
    async fn get_aggregated_sentiment(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>
    ) -> Result<Option<AggregatedSentiment>, DatabaseError>;
    
    // Feature Operations
    async fn insert_features(&self, features: &[FeatureSet]) -> Result<(), DatabaseError>;
    async fn get_features(
        &self, 
        symbol: &str, 
        timestamp: DateTime<Utc>
    ) -> Result<Option<FeatureSet>, DatabaseError>;
    async fn get_latest_features(&self, symbol: &str) -> Result<Option<FeatureSet>, DatabaseError>;
    async fn insert_technical_indicators(&self, indicators: &[TechnicalIndicators]) -> Result<(), DatabaseError>;
    
    // Prediction Operations
    async fn insert_predictions(&self, predictions: &[PredictionResult]) -> Result<(), DatabaseError>;
    async fn get_predictions(
        &self, 
        symbol: &str, 
        start: DateTime<Utc>, 
        end: DateTime<Utc>
    ) -> Result<Vec<PredictionResult>, DatabaseError>;
    async fn insert_prediction_outcomes(&self, outcomes: &[PredictionOutcome]) -> Result<(), DatabaseError>;
    
    // Strategy Performance Operations
    async fn insert_strategy_performance(&self, performance: &[StrategyPerformance]) -> Result<(), DatabaseError>;
    async fn get_strategy_performance(
        &self,
        strategy_name: &str,
        symbol: Option<&str>,
        start: DateTime<Utc>,
        end: DateTime<Utc>
    ) -> Result<Vec<StrategyPerformance>, DatabaseError>;
    
    // Health and Management
    async fn health_check(&self) -> Result<HealthStatus, DatabaseError>;
}

#[async_trait]
pub trait CacheClient: Send + Sync {
    // Cache Operations
    async fn cache_set<T: Serialize + Send>(
        &self, 
        key: &str, 
        value: &T, 
        ttl: Option<Duration>
    ) -> Result<(), DatabaseError>;
    
    async fn cache_get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, DatabaseError>;
    async fn cache_delete(&self, key: &str) -> Result<(), DatabaseError>;
    async fn cache_exists(&self, key: &str) -> Result<bool, DatabaseError>;
    async fn cache_expire(&self, key: &str, ttl: Duration) -> Result<(), DatabaseError>;
    
    // Cache Management
    async fn cache_flush(&self) -> Result<(), DatabaseError>;
    async fn cache_info(&self) -> Result<serde_json::Value, DatabaseError>;
}