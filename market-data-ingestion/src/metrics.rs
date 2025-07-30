use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    collections_completed: Arc<std::sync::atomic::AtomicU64>,
    collections_failed: Arc<std::sync::atomic::AtomicU64>,
    api_calls_made: Arc<std::sync::atomic::AtomicU64>,
    api_calls_failed: Arc<std::sync::atomic::AtomicU64>,
    rate_limit_hits: Arc<std::sync::atomic::AtomicU64>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            collections_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            collections_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            api_calls_made: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            api_calls_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            rate_limit_hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
    
    pub fn record_collection_attempt(&self) {
        self.api_calls_made.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn record_collection_success(&self, _data_points: usize, _duration: std::time::Duration) {
        self.collections_completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn record_batch_processing_success(&self, _duration: std::time::Duration) {
        // Record batch processing metrics
    }
    
    pub async fn get_metrics(&self) -> IngestionMetrics {
        IngestionMetrics {
            collections_completed: self.collections_completed.load(std::sync::atomic::Ordering::Relaxed),
            collections_failed: self.collections_failed.load(std::sync::atomic::Ordering::Relaxed),
            api_calls_made: self.api_calls_made.load(std::sync::atomic::Ordering::Relaxed),
            api_calls_failed: self.api_calls_failed.load(std::sync::atomic::Ordering::Relaxed),
            rate_limit_hits: self.rate_limit_hits.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionMetrics {
    pub collections_completed: u64,
    pub collections_failed: u64,
    pub api_calls_made: u64,
    pub api_calls_failed: u64,
    pub rate_limit_hits: u64,
} 