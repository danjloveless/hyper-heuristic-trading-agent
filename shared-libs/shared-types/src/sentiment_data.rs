use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentData {
    pub article_id: String,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub title: String,
    pub content: String,
    pub source: String,
    pub sentiment_score: f32, // -1.0 to 1.0
    pub confidence: f32,      // 0.0 to 1.0
    pub entities: HashMap<String, f32>,
    pub relevance_score: f32,
    pub market_impact: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedSentiment {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub overall_sentiment: f32,
    pub confidence_weighted_sentiment: f32,
    pub article_count: u32,
    pub bullish_count: u32,
    pub bearish_count: u32,
    pub neutral_count: u32,
}