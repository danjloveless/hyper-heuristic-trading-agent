use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub features: HashMap<String, f64>,
    pub feature_metadata: HashMap<String, String>, // JSON metadata
    pub feature_version: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicators {
    pub rsi_14: f64,
    pub macd_signal: f64,
    pub macd_histogram: f64,
    pub bollinger_upper: f64,
    pub bollinger_lower: f64,
    pub bollinger_middle: f64,
    pub volume_sma_ratio: f64,
    pub price_momentum_5d: f64,
    pub price_momentum_20d: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentFeatures {
    pub reddit_sentiment: f64,
    pub news_sentiment: f64,
    pub combined_sentiment: f64,
    pub sentiment_momentum: f64,
    pub mention_frequency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegime {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub regime: String, // 'low_volatility_uptrend', 'high_volatility_downtrend', etc.
    pub confidence: f32,
    pub volatility_level: f64,
    pub trend_direction: String, // 'up', 'down', 'sideways'
    pub regime_metadata: HashMap<String, f64>,
} 