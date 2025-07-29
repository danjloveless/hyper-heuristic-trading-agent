use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub prediction_id: String,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub predicted_price: Decimal,
    pub confidence: f32,
    pub horizon_minutes: u16,
    pub strategy_name: String,
    pub model_version: String,
    pub explanation: Option<PredictionExplanation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionExplanation {
    pub feature_importance: HashMap<String, f32>,
    pub regime_context: String,
    pub key_factors: Vec<String>,
    pub risk_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOutcome {
    pub prediction_id: String,
    pub actual_price: Decimal,
    pub accuracy_score: f32,
    pub directional_accuracy: bool,
    pub outcome_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPerformance {
    pub strategy_name: String,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub total_predictions: u32,
    pub accuracy_rate: f32,
    pub avg_confidence: f32,
    pub profit_loss: Decimal,
    pub sharpe_ratio: f32,
    pub max_drawdown: f32,
}