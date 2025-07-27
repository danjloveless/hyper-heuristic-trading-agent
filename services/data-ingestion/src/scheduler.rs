// services/data-ingestion/src/scheduler.rs
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tokio_cron_scheduler::{Job, JobScheduler, JobSchedulerError};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::config::DataIngestionConfig;
use crate::{yahoo_client::YahooFinanceClient, reddit_client::RedditClient, news_client::NewsClient};
use shared_utils::{ClickHouseClient, QuantumTradeError, Result};

#[derive(Debug, Clone)]
pub struct DataIngestionScheduler {
    scheduler: Arc<Mutex<Option<JobScheduler>>>,
    config: DataIngestionConfig,
    clickhouse: ClickHouseClient,
    yahoo_client: YahooFinanceClient,
    reddit_client: RedditClient,
    news_client: NewsClient,
    status: Arc<RwLock<SchedulerStatus>>,
}

#[derive(Debug, Clone)]
pub struct SchedulerStatus {
    pub is_running: bool,
    pub last_market_data_run: Option<DateTime<Utc>>,
    pub last_reddit_run: Option<DateTime<Utc>>,
    pub last_news_run: Option<DateTime<Utc>>,
    pub last_cleanup_run: Option<DateTime<Utc>>,
    pub market_data_job_id: Option<Uuid>,
    pub reddit_job_id: Option<Uuid>,
    pub news_job_id: Option<Uuid>,
    pub cleanup_job_id: Option<Uuid>,
    pub total_market_data_collected: u64,
    pub total_reddit_data_collected: u64,
    pub total_news_data_collected: u64,
    pub last_error: Option<String>,
}

impl Default for SchedulerStatus {
    fn default() -> Self {
        Self {
            is_running: false,
            last_market_data_run: None,
            last_reddit_run: None,
            last_news_run: None,
            last_cleanup_run: None,
            market_data_job_id: None,
            reddit_job_id: None,
            news_job_id: None,
            cleanup_job_id: None,
            total_market_data_collected: 0,
            total_reddit_data_collected: 0,
            total_news_data_collected: 0,
            last_error: None,
        }
    }
}

impl DataIngestionScheduler {
    /// Create a new data ingestion scheduler
    pub fn new(
        clickhouse: ClickHouseClient,
        yahoo_client: YahooFinanceClient,
        reddit_client: RedditClient,
        news_client: NewsClient,
        config: DataIngestionConfig,
    ) -> Self {
        info!("Creating data ingestion scheduler");
        
        Self {
            scheduler: Arc::new(Mutex::new(None)),
            config,
            clickhouse,
            yahoo_client,
            reddit_client,
            news_client,
            status: Arc::new(RwLock::new(SchedulerStatus::default())),
        }
    }
    
    /// Start the background scheduler
    pub async fn start(&self) -> Result<()> {
        info!("Starting data ingestion scheduler");
        
        let mut scheduler_guard = self.scheduler.lock().await;
        
        if scheduler_guard.is_some() {
            warn!("Scheduler is already running");
            return Ok(());
        }
        
        let sched = JobScheduler::new().await
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to create job scheduler: {}", e)
            })?;
        
        // Schedule market data collection job
        let market_data_job = self.create_market_data_job().await?;
        let market_data_job_id = sched.add(market_data_job).await
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to add market data job: {}", e)
            })?;
        
        // Schedule Reddit sentiment collection job
        let reddit_job = self.create_reddit_job().await?;
        let reddit_job_id = sched.add(reddit_job).await
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to add Reddit job: {}", e)
            })?;
        
        // Schedule news sentiment collection job
        let news_job = self.create_news_job().await?;
        let news_job_id = sched.add(news_job).await
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to add news job: {}", e)
            })?;
        
        // Schedule cleanup job
        let cleanup_job = self.create_cleanup_job().await?;
        let cleanup_job_id = sched.add(cleanup_job).await
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to add cleanup job: {}", e)
            })?;
        
        // Start the scheduler
        sched.start().await
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to start scheduler: {}", e)
            })?;
        
        *scheduler_guard = Some(sched);
        drop(scheduler_guard);
        
        // Update status
        {
            let mut status = self.status.write().await;
            status.is_running = true;
            status.market_data_job_id = Some(market_data_job_id);
            status.reddit_job_id = Some(reddit_job_id);
            status.news_job_id = Some(news_job_id);
            status.cleanup_job_id = Some(cleanup_job_id);
        }
        
        info!("Data ingestion scheduler started successfully");
        info!("Market data job scheduled: {}", self.config.scheduler.market_data_cron);
        info!("Reddit sentiment job scheduled: {}", self.config.scheduler.reddit_sentiment_cron);
        info!("News sentiment job scheduled: {}", self.config.scheduler.news_sentiment_cron);
        info!("Cleanup job scheduled: {}", self.config.scheduler.cleanup_cron);
        
        Ok(())
    }
    
    /// Stop the background scheduler
    pub async fn stop(&self) {
        info!("Stopping data ingestion scheduler");
        
        let mut scheduler_guard = self.scheduler.lock().await;
        
        if let Some(sched) = scheduler_guard.take() {
            if let Err(e) = sched.shutdown().await {
                error!("Error shutting down scheduler: {}", e);
            }
        }
        
        // Update status
        {
            let mut status = self.status.write().await;
            status.is_running = false;
            status.market_data_job_id = None;
            status.reddit_job_id = None;
            status.news_job_id = None;
            status.cleanup_job_id = None;
        }
        
        info!("Data ingestion scheduler stopped");
    }
    
    /// Get current scheduler status
    pub async fn get_status(&self) -> SchedulerStatus {
        self.status.read().await.clone()
    }
    
    /// Create market data collection job
    async fn create_market_data_job(&self) -> Result<Job> {
        let clickhouse = self.clickhouse.clone();
        let yahoo_client = self.yahoo_client.clone();
        let config = self.config.clone();
        let status = self.status.clone();
        
        let job = Job::new_async(&config.scheduler.market_data_cron, move |_uuid, _l| {
            let clickhouse = clickhouse.clone();
            let yahoo_client = yahoo_client.clone();
            let config = config.clone();
            let status = status.clone();
            
            Box::pin(async move {
                info!("Starting scheduled market data collection");
                
                // Check if we should collect data during market hours
                if !config.should_collect_now() {
                    debug!("Skipping market data collection outside market hours");
                    return;
                }
                
                let start_time = std::time::Instant::now();
                let mut total_records = 0u64;
                let mut errors = Vec::new();
                
                // Get all symbols to process
                let symbols = config.get_all_symbols();
                let batch_size = config.symbols.max_symbols_per_batch as usize;
                
                // Process symbols in batches
                for batch in symbols.chunks(batch_size) {
                    match yahoo_client.get_multiple_quotes(batch).await {
                        Ok(mut data) => {
                            if !data.is_empty() {
                                match clickhouse.insert_market_data_batch(data.clone()).await {
                                    Ok(count) => {
                                        total_records += count;
                                        debug!("Inserted {} market data records for batch", count);
                                    }
                                    Err(e) => {
                                        let error_msg = format!("Failed to insert market data batch: {}", e);
                                        error!("{}", error_msg);
                                        errors.push(error_msg);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("Failed to fetch market data for batch: {}", e);
                            error!("{}", error_msg);
                            errors.push(error_msg);
                        }
                    }
                    
                    // Rate limiting between batches
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                }
                
                let duration = start_time.elapsed();
                
                // Update status
                {
                    let mut status_guard = status.write().await;
                    status_guard.last_market_data_run = Some(Utc::now());
                    status_guard.total_market_data_collected += total_records;
                    if !errors.is_empty() {
                        status_guard.last_error = Some(format!("Market data collection errors: {}", errors.join("; ")));
                    }
                }
                
                if errors.is_empty() {
                    info!("Completed scheduled market data collection: {} records in {:?}", 
                          total_records, duration);
                } else {
                    warn!("Completed scheduled market data collection with {} errors: {} records in {:?}", 
                          errors.len(), total_records, duration);
                }
            })
        })
        .map_err(|e| QuantumTradeError::Configuration {
            message: format!("Failed to create market data job: {}", e)
        })?;
        
        Ok(job)
    }
    
    /// Create Reddit sentiment collection job
    async fn create_reddit_job(&self) -> Result<Job> {
        let clickhouse = self.clickhouse.clone();
        let reddit_client = self.reddit_client.clone();
        let config = self.config.clone();
        let status = self.status.clone();
        
        let job = Job::new_async(&config.scheduler.reddit_sentiment_cron, move |_uuid, _l| {
            let clickhouse = clickhouse.clone();
            let reddit_client = reddit_client.clone();
            let config = config.clone();
            let status = status.clone();
            
            Box::pin(async move {
                info!("Starting scheduled Reddit sentiment collection");
                
                let start_time = std::time::Instant::now();
                let mut total_records = 0u64;
                let mut errors = Vec::new();
                
                // Collect sentiment for primary symbols
                for symbol in &config.symbols.primary_symbols {
                    match reddit_client.get_sentiment_data(symbol, 50).await {
                        Ok(sentiment_data) => {
                            if !sentiment_data.is_empty() {
                                match clickhouse.insert_sentiment_data_batch(sentiment_data.clone()).await {
                                    Ok(count) => {
                                        total_records += count;
                                        debug!("Inserted {} Reddit sentiment records for {}", count, symbol);
                                    }
                                    Err(e) => {
                                        let error_msg = format!("Failed to insert Reddit data for {}: {}", symbol, e);
                                        error!("{}", error_msg);
                                        errors.push(error_msg);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("Failed to fetch Reddit data for {}: {}", symbol, e);
                            error!("{}", error_msg);
                            errors.push(error_msg);
                        }
                    }
                    
                    // Rate limiting between symbols
                    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                }
                
                // Also collect general market sentiment
                match reddit_client.get_general_market_sentiment(100).await {
                    Ok(sentiment_data) => {
                        if !sentiment_data.is_empty() {
                            match clickhouse.insert_sentiment_data_batch(sentiment_data).await {
                                Ok(count) => {
                                    total_records += count;
                                    debug!("Inserted {} general market sentiment records from Reddit", count);
                                }
                                Err(e) => {
                                    let error_msg = format!("Failed to insert general Reddit sentiment: {}", e);
                                    error!("{}", error_msg);
                                    errors.push(error_msg);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("Failed to fetch general Reddit sentiment: {}", e);
                        error!("{}", error_msg);
                        errors.push(error_msg);
                    }
                }
                
                let duration = start_time.elapsed();
                
                // Update status
                {
                    let mut status_guard = status.write().await;
                    status_guard.last_reddit_run = Some(Utc::now());
                    status_guard.total_reddit_data_collected += total_records;
                    if !errors.is_empty() {
                        status_guard.last_error = Some(format!("Reddit collection errors: {}", errors.join("; ")));
                    }
                }
                
                if errors.is_empty() {
                    info!("Completed scheduled Reddit sentiment collection: {} records in {:?}", 
                          total_records, duration);
                } else {
                    warn!("Completed scheduled Reddit sentiment collection with {} errors: {} records in {:?}", 
                          errors.len(), total_records, duration);
                }
            })
        })
        .map_err(|e| QuantumTradeError::Configuration {
            message: format!("Failed to create Reddit job: {}", e)
        })?;
        
        Ok(job)
    }
    
    /// Create news sentiment collection job
    async fn create_news_job(&self) -> Result<Job> {
        let clickhouse = self.clickhouse.clone();
        let news_client = self.news_client.clone();
        let config = self.config.clone();
        let status = self.status.clone();
        
        let job = Job::new_async(&config.scheduler.news_sentiment_cron, move |_uuid, _l| {
            let clickhouse = clickhouse.clone();
            let news_client = news_client.clone();
            let config = config.clone();
            let status = status.clone();
            
            Box::pin(async move {
                info!("Starting scheduled news sentiment collection");
                
                let start_time = std::time::Instant::now();
                let mut total_records = 0u64;
                let mut errors = Vec::new();
                
                // Collect sentiment for primary symbols
                for symbol in &config.symbols.primary_symbols {
                    match news_client.get_sentiment_data(symbol, 20).await {
                        Ok(sentiment_data) => {
                            if !sentiment_data.is_empty() {
                                match clickhouse.insert_sentiment_data_batch(sentiment_data.clone()).await {
                                    Ok(count) => {
                                        total_records += count;
                                        debug!("Inserted {} news sentiment records for {}", count, symbol);
                                    }
                                    Err(e) => {
                                        let error_msg = format!("Failed to insert news data for {}: {}", symbol, e);
                                        error!("{}", error_msg);
                                        errors.push(error_msg);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("Failed to fetch news data for {}: {}", symbol, e);
                            error!("{}", error_msg);
                            errors.push(error_msg);
                        }
                    }
                    
                    // Rate limiting between symbols (news APIs are more restrictive)
                    tokio::time::sleep(tokio::time::Duration::from_millis(2000)).await;
                }
                
                // Also collect general market sentiment
                match news_client.get_general_market_sentiment(50).await {
                    Ok(sentiment_data) => {
                        if !sentiment_data.is_empty() {
                            match clickhouse.insert_sentiment_data_batch(sentiment_data).await {
                                Ok(count) => {
                                    total_records += count;
                                    debug!("Inserted {} general market sentiment records from news", count);
                                }
                                Err(e) => {
                                    let error_msg = format!("Failed to insert general news sentiment: {}", e);
                                    error!("{}", error_msg);
                                    errors.push(error_msg);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("Failed to fetch general news sentiment: {}", e);
                        error!("{}", error_msg);
                        errors.push(error_msg);
                    }
                }
                
                let duration = start_time.elapsed();
                
                // Update status
                {
                    let mut status_guard = status.write().await;
                    status_guard.last_news_run = Some(Utc::now());
                    status_guard.total_news_data_collected += total_records;
                    if !errors.is_empty() {
                        status_guard.last_error = Some(format!("News collection errors: {}", errors.join("; ")));
                    }
                }
                
                if errors.is_empty() {
                    info!("Completed scheduled news sentiment collection: {} records in {:?}", 
                          total_records, duration);
                } else {
                    warn!("Completed scheduled news sentiment collection with {} errors: {} records in {:?}", 
                          errors.len(), total_records, duration);
                }
            })
        })
        .map_err(|e| QuantumTradeError::Configuration {
            message: format!("Failed to create news job: {}", e)
        })?;
        
        Ok(job)
    }
    
    /// Create cleanup job for old data
    async fn create_cleanup_job(&self) -> Result<Job> {
        let clickhouse = self.clickhouse.clone();
        let status = self.status.clone();
        
        let job = Job::new_async(&self.config.scheduler.cleanup_cron, move |_uuid, _l| {
            let clickhouse = clickhouse.clone();
            let status = status.clone();
            
            Box::pin(async move {
                info!("Starting scheduled data cleanup");
                
                let start_time = std::time::Instant::now();
                
                // Clean up data older than 90 days (configurable)
                let retention_days = 90;
                
                match clickhouse.cleanup_old_data(retention_days).await {
                    Ok(result) => {
                        info!("Completed scheduled data cleanup in {:?}: {:?}", 
                              start_time.elapsed(), result);
                        
                        // Update status
                        {
                            let mut status_guard = status.write().await;
                            status_guard.last_cleanup_run = Some(Utc::now());
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("Data cleanup failed: {}", e);
                        error!("{}", error_msg);
                        
                        // Update status with error
                        {
                            let mut status_guard = status.write().await;
                            status_guard.last_cleanup_run = Some(Utc::now());
                            status_guard.last_error = Some(error_msg);
                        }
                    }
                }
            })
        })
        .map_err(|e| QuantumTradeError::Configuration {
            message: format!("Failed to create cleanup job: {}", e)
        })?;
        
        Ok(job)
    }
    
    /// Manually trigger market data collection
    pub async fn trigger_market_data_collection(&self) -> Result<u64> {
        info!("Manually triggering market data collection");
        
        let symbols = self.config.get_all_symbols();
        let data = self.yahoo_client.get_multiple_quotes(&symbols).await?;
        
        if data.is_empty() {
            return Ok(0);
        }
        
        let count = self.clickhouse.insert_market_data_batch(data).await?;
        
        // Update status
        {
            let mut status = self.status.write().await;
            status.last_market_data_run = Some(Utc::now());
            status.total_market_data_collected += count;
        }
        
        info!("Manual market data collection completed: {} records", count);
        Ok(count)
    }
    
    /// Manually trigger Reddit sentiment collection
    pub async fn trigger_reddit_collection(&self) -> Result<u64> {
        info!("Manually triggering Reddit sentiment collection");
        
        let mut total_count = 0u64;
        
        for symbol in &self.config.symbols.primary_symbols {
            let data = self.reddit_client.get_sentiment_data(symbol, 25).await?;
            if !data.is_empty() {
                let count = self.clickhouse.insert_sentiment_data_batch(data).await?;
                total_count += count;
            }
        }
        
        // Update status
        {
            let mut status = self.status.write().await;
            status.last_reddit_run = Some(Utc::now());
            status.total_reddit_data_collected += total_count;
        }
        
        info!("Manual Reddit sentiment collection completed: {} records", total_count);
        Ok(total_count)
    }
    
    /// Manually trigger news sentiment collection
    pub async fn trigger_news_collection(&self) -> Result<u64> {
        info!("Manually triggering news sentiment collection");
        
        let mut total_count = 0u64;
        
        for symbol in &self.config.symbols.primary_symbols {
            let data = self.news_client.get_sentiment_data(symbol, 10).await?;
            if !data.is_empty() {
                let count = self.clickhouse.insert_sentiment_data_batch(data).await?;
                total_count += count;
            }
        }
        
        // Update status
        {
            let mut status = self.status.write().await;
            status.last_news_run = Some(Utc::now());
            status.total_news_data_collected += total_count;
        }
        
        info!("Manual news sentiment collection completed: {} records", total_count);
        Ok(total_count)
    }
}

impl Drop for DataIngestionScheduler {
    fn drop(&mut self) {
        // Note: We can't call async functions in Drop, but the scheduler will be cleaned up
        // when the process exits. In a real application, you'd want to call stop() explicitly.
        debug!("DataIngestionScheduler dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scheduler_status_default() {
        let status = SchedulerStatus::default();
        assert!(!status.is_running);
        assert!(status.last_market_data_run.is_none());
        assert_eq!(status.total_market_data_collected, 0);
    }
    
    // Note: Testing the actual scheduler would require complex async setup and mocking
    // In a production environment, you'd want integration tests with testcontainers
}