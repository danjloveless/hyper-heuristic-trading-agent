use crate::config::DataQualityConfig;
use crate::models::{MarketData, MarketDataBatch};
use rust_decimal::Decimal;

/// Data quality controller
#[derive(Debug, Clone)]
pub struct DataQualityController {
    config: DataQualityConfig,
}

impl DataQualityController {
    pub fn new(config: DataQualityConfig) -> Self {
        Self { config }
    }
    
    pub fn calculate_quality_score(&self, data: &MarketData) -> u8 {
        let mut score = 100u8;
        
        // Check for zero volume
        if data.volume == 0 {
            score = score.saturating_sub(20);
        }
        
        // Check for suspicious price movements
        let daily_change = data.percentage_change().abs();
        if daily_change > Decimal::from_f64_retain(20.0).unwrap_or_default() {
            score = score.saturating_sub(15);
        }
        
        // Check volume threshold
        if data.volume < self.config.volume_threshold {
            score = score.saturating_sub(10);
        }
        
        // Check price consistency
        if data.high == data.low && data.volume > 0 {
            score = score.saturating_sub(25); // Suspicious flat pricing
        }
        
        score
    }
    
    pub fn deduplicate_batch(&self, batch: &mut MarketDataBatch) {
        batch.data_points.dedup_by(|a, b| {
            a.symbol == b.symbol && a.timestamp == b.timestamp
        });
    }
} 