use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Market data point
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MarketData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: u64,
    pub adjusted_close: Decimal,
    pub source: String,
    pub quality_score: u8,
}

impl MarketData {
    /// Validate market data integrity
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.high < self.low {
            return Err("High price cannot be less than low price".to_string());
        }
        
        if self.open <= Decimal::ZERO || self.high <= Decimal::ZERO 
           || self.low <= Decimal::ZERO || self.close <= Decimal::ZERO {
            return Err("Prices must be positive".to_string());
        }
        
        if self.open > self.high || self.open < self.low {
            return Err("Open price must be between high and low".to_string());
        }
        
        if self.close > self.high || self.close < self.low {
            return Err("Close price must be between high and low".to_string());
        }
        
        if self.symbol.is_empty() {
            return Err("Symbol cannot be empty".to_string());
        }
        
        Ok(())
    }
    
    /// Calculate OHLC percentage change
    pub fn percentage_change(&self) -> Decimal {
        if self.open != Decimal::ZERO {
            ((self.close - self.open) / self.open) * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }
    
    /// Calculate true range (for volatility analysis)
    pub fn true_range(&self, previous_close: Option<Decimal>) -> Decimal {
        let high_low = self.high - self.low;
        
        if let Some(prev_close) = previous_close {
            let high_prev_close = (self.high - prev_close).abs();
            let low_prev_close = (self.low - prev_close).abs();
            
            high_low.max(high_prev_close).max(low_prev_close)
        } else {
            high_low
        }
    }
}

/// Batch of market data for efficient processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataBatch {
    pub batch_id: String,
    pub symbol: String,
    pub data_points: Vec<MarketData>,
    pub collection_timestamp: DateTime<Utc>,
    pub source: String,
    pub metadata: HashMap<String, String>,
}

impl MarketDataBatch {
    pub fn new(symbol: String, source: String) -> Self {
        Self {
            batch_id: Uuid::new_v4().to_string(),
            symbol,
            data_points: Vec::new(),
            collection_timestamp: Utc::now(),
            source,
            metadata: HashMap::new(),
        }
    }
    
    pub fn add_data_point(&mut self, data_point: MarketData) {
        self.data_points.push(data_point);
    }
    
    pub fn size(&self) -> usize {
        self.data_points.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data_points.is_empty()
    }
    
    pub fn sort_by_timestamp(&mut self) {
        self.data_points.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }
    
    pub fn get_time_range(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        if self.data_points.is_empty() {
            return None;
        }
        
        let mut min_time = self.data_points[0].timestamp;
        let mut max_time = self.data_points[0].timestamp;
        
        for data_point in &self.data_points {
            if data_point.timestamp < min_time {
                min_time = data_point.timestamp;
            }
            if data_point.timestamp > max_time {
                max_time = data_point.timestamp;
            }
        }
        
        Some((min_time, max_time))
    }
} 