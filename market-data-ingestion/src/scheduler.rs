use crate::config::CollectionConfig;

/// Ingestion scheduler
#[derive(Debug)]
pub struct IngestionScheduler {
    config: CollectionConfig,
}

impl IngestionScheduler {
    pub fn new(config: CollectionConfig) -> Self {
        Self { config }
    }
} 