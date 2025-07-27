use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;

// Basic market data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub adjusted_close: f64,
}

// Sentiment data from social media/news
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub source: String, // 'reddit', 'news', etc.
    pub sentiment_score: f32,
    pub confidence: f32,
    pub mention_count: u32,
    pub raw_data: String, // JSON as string for flexibility
}

// Feature set for ML (flexible structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub features: HashMap<String, f64>,
    pub feature_metadata: HashMap<String, String>, // JSON metadata
    pub feature_version: u16,
}

// Prediction outcome tracking (for performance evaluation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOutcome {
    pub prediction_id: String,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub actual_price: f64,
    pub predicted_price: f64,
    pub profit_loss: f64,
    pub accuracy_score: f32,
    pub strategy_name: String,
    pub regime: String,
}

// Strategy performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPerformance {
    pub id: Uuid,
    pub strategy_name: String,
    pub regime: String,
    pub timestamp: DateTime<Utc>,
    pub prediction_accuracy: f32,
    pub profit_loss: f64,
    pub sharpe_ratio: f32,
    pub max_drawdown: f32,
    pub trade_count: u32,
    pub win_rate: f32,
    pub parameters: HashMap<String, f64>,
    pub market_conditions: HashMap<String, f64>,
}
