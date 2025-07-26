use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub symbol: String,
    pub horizon_minutes: u16,
    pub include_explanation: bool,
    pub features: FeatureRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRequest {
    pub technical_indicators: bool,
    pub sentiment_data: bool,
    pub news_sentiment: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResponse {
    pub prediction_id: String,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub prediction: Prediction,
    pub strategy: StrategyInfo,
    pub explanation: Option<Explanation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub target_price: f64,
    pub confidence: f32,
    pub direction: String, // 'bullish', 'bearish', 'neutral'
    pub horizon_minutes: u16,
    pub regime: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyInfo {
    pub selected_strategy: String,
    pub parameter_weights: HashMap<String, f64>,
    pub heuristic_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    pub top_features: Vec<FeatureImportance>,
    pub reasoning: String,
    pub confidence_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    pub name: String,
    pub importance: f64,
    pub value: f64,
}

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