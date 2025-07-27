// shared/utils/src/monitoring.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub query_latency_ms: u64,
    pub rows_processed: u64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct MetricsCollector {
    metrics: HashMap<String, PerformanceMetrics>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    pub fn record_query_latency(&mut self, operation: &str, latency_ms: u64, rows: u64) {
        debug!("Recording query latency for {}: {}ms, {} rows", operation, latency_ms, rows);
        
        let metrics = self.metrics.entry(operation.to_string()).or_insert_with(|| {
            PerformanceMetrics {
                query_latency_ms: 0,
                rows_processed: 0,
                cache_hit_rate: 0.0,
                error_rate: 0.0,
                timestamp: Utc::now(),
            }
        });

        metrics.query_latency_ms = latency_ms;
        metrics.rows_processed = rows;
        metrics.timestamp = Utc::now();
    }

    pub fn get_metrics(&self, operation: &str) -> Option<&PerformanceMetrics> {
        self.metrics.get(operation)
    }

    pub fn get_all_metrics(&self) -> &HashMap<String, PerformanceMetrics> {
        &self.metrics
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}