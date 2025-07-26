use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub title: String,
    pub content: String,
    pub sentiment_score: f32,
    pub relevance_score: f32,
    pub source: String,
    pub url: String,
} 