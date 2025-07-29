use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub features: HashMap<String, f64>,
    pub feature_metadata: HashMap<String, String>,
    pub feature_version: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicators {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub sma_20: Option<f64>,
    pub ema_12: Option<f64>,
    pub ema_26: Option<f64>,
    pub macd: Option<f64>,
    pub macd_signal: Option<f64>,
    pub rsi: Option<f64>,
    pub bollinger_upper: Option<f64>,
    pub bollinger_lower: Option<f64>,
    pub volume_sma_20: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegimeFeatures {
    pub timestamp: DateTime<Utc>,
    pub volatility_regime: String,
    pub trend_regime: String,
    pub market_stress_index: f64,
    pub correlation_breakdown: bool,
}