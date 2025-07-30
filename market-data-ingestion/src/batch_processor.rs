use crate::config::StorageConfig;
use crate::errors::Result;
use crate::models::MarketDataBatch;
use std::sync::Arc;
use tracing::{debug, info, error};
use shared_types::MarketData;

/// Batch processor for efficient storage
pub struct BatchProcessor {
    config: StorageConfig,
    pub database: Arc<dyn database_abstraction::traits::DatabaseClient>,
}

impl BatchProcessor {
    pub fn new(config: StorageConfig, database: Arc<dyn database_abstraction::traits::DatabaseClient>) -> Self {
        Self { config, database }
    }
    
    pub async fn process_batch(&self, batch: MarketDataBatch) -> Result<()> {
        debug!("Processing batch {} with {} data points for symbol {}", 
               batch.batch_id, batch.size(), batch.symbol);
        
        if batch.data_points.is_empty() {
            info!("Batch {} is empty, skipping storage", batch.batch_id);
            return Ok(());
        }
        
        // Convert MarketDataBatch to Vec<MarketData> for database storage
        let market_data: Vec<MarketData> = batch.data_points.iter().map(|data_point| {
            MarketData {
                symbol: batch.symbol.clone(),
                timestamp: data_point.timestamp,
                open: data_point.open,
                high: data_point.high,
                low: data_point.low,
                close: data_point.close,
                volume: data_point.volume,
                adjusted_close: data_point.adjusted_close,
            }
        }).collect();
        
        // Store data to database
        match self.database.insert_market_data(&market_data).await {
            Ok(_) => {
                info!("Successfully stored {} market data points for symbol {} to database", 
                      market_data.len(), batch.symbol);
                Ok(())
            },
            Err(e) => {
                error!("Failed to store batch {} to database: {}", batch.batch_id, e);
                Err(crate::errors::IngestionError::StorageError {
                    operation: format!("Database storage failed: {}", e),
                })
            }
        }
    }
} 