// ClickHouse-specific data models and row structures
// This module contains models optimized for ClickHouse operations

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataRow {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub adjusted_close: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentDataRow {
    pub article_id: String,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub title: String,
    pub content: String,
    pub source: String,
    pub sentiment_score: f32,
    pub confidence: f32,
    pub entities: String, // JSON string
    pub relevance_score: f32,
    pub market_impact: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRow {
    pub prediction_id: String,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub predicted_price: f64,
    pub confidence: f32,
    pub horizon_minutes: u16,
    pub strategy_name: String,
    pub model_version: String,
    pub explanation: Option<String>, // JSON string
}