// shared/utils/src/clickhouse.rs
use clickhouse::{Client, Row};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use shared_types::{MarketData, SentimentData, FeatureSet, PredictionOutcome, StrategyPerformance};
use crate::error::{QuantumTradeError, Result};

/// Configuration for ClickHouse client
#[derive(Debug, Clone)]
pub struct ClickHouseConfig {
    pub url: String,
    pub database: String,
    pub username: String,
    pub password: String,
    pub connection_timeout: u64,
    pub request_timeout: u64,
    pub max_connections: u32,
    pub enable_compression: bool,
}

impl Default for ClickHouseConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8123".to_string(),
            database: "quantumtrade".to_string(),
            username: "default".to_string(),
            password: "".to_string(),
            connection_timeout: 30,
            request_timeout: 60,
            max_connections: 10,
            enable_compression: true,
        }
    }
}

impl ClickHouseConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            url: std::env::var("CLICKHOUSE_URL")
                .unwrap_or_else(|_| "http://localhost:8123".to_string()),
            database: std::env::var("CLICKHOUSE_DATABASE")
                .unwrap_or_else(|_| "quantumtrade".to_string()),
            username: std::env::var("CLICKHOUSE_USERNAME")
                .unwrap_or_else(|_| "default".to_string()),
            password: std::env::var("CLICKHOUSE_PASSWORD")
                .unwrap_or_else(|_| "".to_string()),
            connection_timeout: std::env::var("CLICKHOUSE_CONNECTION_TIMEOUT")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .map_err(|e| QuantumTradeError::Configuration { 
                    message: format!("Invalid connection timeout: {}", e) 
                })?,
            request_timeout: std::env::var("CLICKHOUSE_REQUEST_TIMEOUT")
                .unwrap_or_else(|_| "60".to_string())
                .parse()
                .map_err(|e| QuantumTradeError::Configuration { 
                    message: format!("Invalid request timeout: {}", e) 
                })?,
            max_connections: std::env::var("CLICKHOUSE_MAX_CONNECTIONS")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .map_err(|e| QuantumTradeError::Configuration { 
                    message: format!("Invalid max connections: {}", e) 
                })?,
            enable_compression: std::env::var("CLICKHOUSE_ENABLE_COMPRESSION")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
        })
    }
}

/// High-performance ClickHouse client for financial time series data
#[derive(Clone)]
pub struct ClickHouseClient {
    client: Client,
    config: ClickHouseConfig,
}

impl ClickHouseClient {
    /// Create a new ClickHouse client with configuration
    pub async fn new(config: ClickHouseConfig) -> Result<Self> {
        info!("Initializing ClickHouse client with URL: {}", config.url);
        
        let client = Client::default()
            .with_url(&config.url)
            .with_database(&config.database)
            .with_user(&config.username)
            .with_password(&config.password);

        let instance = Self { client, config };
        
        // Test connection
        instance.test_connection().await?;
        info!("ClickHouse client initialized successfully");
        
        Ok(instance)
    }

    /// Create client from environment variables
    pub async fn from_env() -> Result<Self> {
        let config = ClickHouseConfig::from_env()?;
        Self::new(config).await
    }

    /// Test database connectivity
    pub async fn test_connection(&self) -> Result<()> {
        debug!("Testing ClickHouse connection");
        
        let result: u64 = self.client
            .query("SELECT 1")
            .fetch_one()
            .await
            .map_err(|e| {
                error!("Failed to connect to ClickHouse: {}", e);
                QuantumTradeError::DatabaseConnection(e)
            })?;
            
        if result != 1 {
            return Err(QuantumTradeError::DatabaseConnection(
                clickhouse::error::Error::Custom("Connection test failed".to_string())
            ));
        }
        
        debug!("ClickHouse connection test successful");
        Ok(())
    }

    /// Initialize database schema
    pub async fn initialize_schema(&self) -> Result<()> {
        info!("Initializing ClickHouse database schema");
        
        let schema_queries = vec![
            include_str!("../../../scripts/setup-db.sql"),
        ];

        for query in schema_queries {
            if !query.trim().is_empty() {
                self.client
                    .query(query)
                    .execute()
                    .await
                    .map_err(QuantumTradeError::DatabaseConnection)?;
            }
        }
        
        info!("Database schema initialized successfully");
        Ok(())
    }

    // Market Data Operations
    
    /// Insert market data in batch for high performance
    pub async fn insert_market_data_batch(&self, data: Vec<MarketData>) -> Result<u64> {
        if data.is_empty() {
            return Ok(0);
        }

        debug!("Inserting {} market data records", data.len());
        
        let mut inserter = self.client
            .insert("market_data")?;
            
        for record in &data {
            inserter.write(&MarketDataRow::from(record)).await?;
        }
        
        let rows_inserted = inserter.end().await?;
        info!("Successfully inserted {} market data records", rows_inserted);
        
        Ok(rows_inserted)
    }

    /// Get latest market data for a symbol
    pub async fn get_latest_market_data(&self, symbol: &str, limit: u32) -> Result<Vec<MarketData>> {
        debug!("Fetching latest {} market data records for {}", limit, symbol);
        
        let query = "
            SELECT symbol, timestamp, open, high, low, close, volume, adjusted_close
            FROM market_data 
            WHERE symbol = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ";
        
        let rows = self.client
            .query(query)
            .bind(symbol)
            .bind(limit)
            .fetch_all::<MarketDataRow>()
            .await?;
            
        let data: Vec<MarketData> = rows.into_iter().map(Into::into).collect();
        debug!("Retrieved {} market data records for {}", data.len(), symbol);
        
        Ok(data)
    }

    /// Get market data for a time range
    pub async fn get_market_data_range(
        &self, 
        symbol: &str, 
        start_time: DateTime<Utc>, 
        end_time: DateTime<Utc>
    ) -> Result<Vec<MarketData>> {
        debug!("Fetching market data for {} from {} to {}", symbol, start_time, end_time);
        
        let query = "
            SELECT symbol, timestamp, open, high, low, close, volume, adjusted_close
            FROM market_data 
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        ";
        
        let rows = self.client
            .query(query)
            .bind(symbol)
            .bind(start_time)
            .bind(end_time)
            .fetch_all::<MarketDataRow>()
            .await?;
            
        let data: Vec<MarketData> = rows.into_iter().map(Into::into).collect();
        debug!("Retrieved {} market data records for {} in time range", data.len(), symbol);
        
        Ok(data)
    }

    // Sentiment Data Operations
    
    /// Insert sentiment data in batch
    pub async fn insert_sentiment_data_batch(&self, data: Vec<SentimentData>) -> Result<u64> {
        if data.is_empty() {
            return Ok(0);
        }

        debug!("Inserting {} sentiment data records", data.len());
        
        let mut inserter = self.client
            .insert("sentiment_data")?;
            
        for record in &data {
            inserter.write(&SentimentDataRow::from(record)).await?;
        }
        
        let rows_inserted = inserter.end().await?;
        info!("Successfully inserted {} sentiment data records", rows_inserted);
        
        Ok(rows_inserted)
    }

    /// Get latest sentiment data for a symbol and source
    pub async fn get_latest_sentiment_data(
        &self, 
        symbol: &str, 
        source: Option<&str>, 
        limit: u32
    ) -> Result<Vec<SentimentData>> {
        debug!("Fetching latest {} sentiment records for {}", limit, symbol);
        
        let (query, has_source) = if let Some(src) = source {
            (
                "SELECT symbol, timestamp, source, sentiment_score, confidence, mention_count, raw_data
                 FROM sentiment_data 
                 WHERE symbol = ? AND source = ?
                 ORDER BY timestamp DESC 
                 LIMIT ?",
                true
            )
        } else {
            (
                "SELECT symbol, timestamp, source, sentiment_score, confidence, mention_count, raw_data
                 FROM sentiment_data 
                 WHERE symbol = ?
                 ORDER BY timestamp DESC 
                 LIMIT ?",
                false
            )
        };
        
        let mut query_builder = self.client.query(query).bind(symbol);
        
        if has_source {
            query_builder = query_builder.bind(source.unwrap());
        }
        
        let rows = query_builder
            .bind(limit)
            .fetch_all::<SentimentDataRow>()
            .await?;
            
        let data: Vec<SentimentData> = rows.into_iter().map(Into::into).collect();
        debug!("Retrieved {} sentiment records for {}", data.len(), symbol);
        
        Ok(data)
    }

    // Feature Operations
    
    /// Insert feature data in batch
    pub async fn insert_features_batch(&self, features: Vec<FeatureSet>) -> Result<u64> {
        if features.is_empty() {
            return Ok(0);
        }

        debug!("Inserting {} feature sets", features.len());
        
        let mut inserter = self.client
            .insert("features")?;
            
        for feature_set in &features {
            for (feature_name, feature_value) in &feature_set.features {
                let row = FeatureRow {
                    symbol: feature_set.symbol.clone(),
                    timestamp: feature_set.timestamp,
                    feature_name: feature_name.clone(),
                    feature_value: *feature_value,
                    feature_metadata: feature_set.feature_metadata
                        .get(feature_name)
                        .cloned()
                        .unwrap_or_default(),
                    feature_version: feature_set.feature_version,
                };
                inserter.write(&row).await?;
            }
        }
        
        let rows_inserted = inserter.end().await?;
        info!("Successfully inserted {} feature records", rows_inserted);
        
        Ok(rows_inserted)
    }

    /// Get latest features for a symbol
    pub async fn get_latest_features(&self, symbol: &str) -> Result<Option<FeatureSet>> {
        debug!("Fetching latest features for {}", symbol);
        
        let query = "
            SELECT symbol, timestamp, feature_name, feature_value, feature_metadata, feature_version
            FROM features 
            WHERE symbol = ?
            AND timestamp = (
                SELECT max(timestamp) 
                FROM features 
                WHERE symbol = ?
            )
            ORDER BY feature_name
        ";
        
        let rows = self.client
            .query(query)
            .bind(symbol)
            .bind(symbol)
            .fetch_all::<FeatureRow>()
            .await?;
            
        if rows.is_empty() {
            return Ok(None);
        }
        
        let mut features = HashMap::new();
        let mut feature_metadata = HashMap::new();
        let mut timestamp = rows[0].timestamp;
        let mut feature_version = rows[0].feature_version;
        
        for row in rows {
            features.insert(row.feature_name.clone(), row.feature_value);
            if !row.feature_metadata.is_empty() {
                feature_metadata.insert(row.feature_name, row.feature_metadata);
            }
            timestamp = row.timestamp; // Should be the same for all
            feature_version = std::cmp::max(feature_version, row.feature_version);
        }
        
        let feature_set = FeatureSet {
            symbol: symbol.to_string(),
            timestamp,
            features,
            feature_metadata,
            feature_version,
        };
        
        debug!("Retrieved {} features for {}", feature_set.features.len(), symbol);
        Ok(Some(feature_set))
    }

    // Prediction Operations
    
    /// Insert a prediction
    pub async fn insert_prediction(&self, prediction: &PredictionRecord) -> Result<()> {
        debug!("Inserting prediction with ID: {}", prediction.prediction_id);
        
        let mut inserter = self.client
            .insert("predictions")?;
            
        inserter.write(&PredictionRow::from(prediction)).await?;
        inserter.end().await?;
        
        info!("Successfully inserted prediction: {}", prediction.prediction_id);
        Ok(())
    }

    /// Get prediction by ID
    pub async fn get_prediction(&self, prediction_id: &str) -> Result<Option<PredictionRecord>> {
        debug!("Fetching prediction with ID: {}", prediction_id);
        
        let query = "
            SELECT symbol, timestamp, prediction_id, model_version, strategy_name,
                   prediction_horizon, predicted_price, confidence, regime,
                   features_used, explanation_id, model_latency_ms,
                   strategy_parameters, heuristic_confidence
            FROM predictions 
            WHERE prediction_id = ?
            LIMIT 1
        ";
        
        let row = self.client
            .query(query)
            .bind(prediction_id)
            .fetch_optional::<PredictionRow>()
            .await?;
            
        Ok(row.map(Into::into))
    }

    /// Get predictions ready for evaluation (where actual outcome is available)
    pub async fn get_predictions_for_evaluation(&self, horizon_minutes: u16) -> Result<Vec<PredictionRecord>> {
        debug!("Fetching predictions ready for evaluation (horizon: {} minutes)", horizon_minutes);
        
        let cutoff_time = Utc::now() - chrono::Duration::minutes(horizon_minutes as i64);
        
        let query = "
            SELECT p.symbol, p.timestamp, p.prediction_id, p.model_version, p.strategy_name,
                   p.prediction_horizon, p.predicted_price, p.confidence, p.regime,
                   p.features_used, p.explanation_id, p.model_latency_ms,
                   p.strategy_parameters, p.heuristic_confidence
            FROM predictions p
            LEFT JOIN prediction_outcomes o ON p.prediction_id = o.prediction_id
            WHERE p.timestamp <= ? 
            AND p.prediction_horizon = ?
            AND o.prediction_id IS NULL
        ";
        
        let rows = self.client
            .query(query)
            .bind(cutoff_time)
            .bind(horizon_minutes)
            .fetch_all::<PredictionRow>()
            .await?;
            
        let predictions: Vec<PredictionRecord> = rows.into_iter().map(Into::into).collect();
        debug!("Found {} predictions ready for evaluation", predictions.len());
        
        Ok(predictions)
    }

    // Prediction Outcome Operations
    
    /// Insert prediction outcomes in batch
    pub async fn insert_prediction_outcomes_batch(&self, outcomes: Vec<PredictionOutcome>) -> Result<u64> {
        if outcomes.is_empty() {
            return Ok(0);
        }

        debug!("Inserting {} prediction outcomes", outcomes.len());
        
        let mut inserter = self.client
            .insert("prediction_outcomes")?;
            
        for outcome in &outcomes {
            inserter.write(&PredictionOutcomeRow::from(outcome)).await?;
        }
        
        let rows_inserted = inserter.end().await?;
        info!("Successfully inserted {} prediction outcomes", rows_inserted);
        
        Ok(rows_inserted)
    }

    // Strategy Performance Operations
    
    /// Insert strategy performance data
    pub async fn insert_strategy_performance(&self, performance: &StrategyPerformance) -> Result<()> {
        debug!("Inserting strategy performance for {}", performance.strategy_name);
        
        let mut inserter = self.client
            .insert("strategy_performance")?;
            
        inserter.write(&StrategyPerformanceRow::from(performance)).await?;
        inserter.end().await?;
        
        info!("Successfully inserted strategy performance for {}", performance.strategy_name);
        Ok(())
    }

    /// Get strategy performance for a specific strategy and regime
    pub async fn get_strategy_performance(
        &self, 
        strategy_name: &str, 
        regime: Option<&str>,
        limit: u32
    ) -> Result<Vec<StrategyPerformance>> {
        debug!("Fetching strategy performance for {} (regime: {:?})", strategy_name, regime);
        
        let (query, has_regime) = if let Some(r) = regime {
            (
                "SELECT id, strategy_name, regime, timestamp, prediction_accuracy,
                        profit_loss, sharpe_ratio, max_drawdown, trade_count, win_rate,
                        parameters, market_conditions
                 FROM strategy_performance 
                 WHERE strategy_name = ? AND regime = ?
                 ORDER BY timestamp DESC 
                 LIMIT ?",
                true
            )
        } else {
            (
                "SELECT id, strategy_name, regime, timestamp, prediction_accuracy,
                        profit_loss, sharpe_ratio, max_drawdown, trade_count, win_rate,
                        parameters, market_conditions
                 FROM strategy_performance 
                 WHERE strategy_name = ?
                 ORDER BY timestamp DESC 
                 LIMIT ?",
                false
            )
        };
        
        let mut query_builder = self.client.query(query).bind(strategy_name);
        
        if has_regime {
            query_builder = query_builder.bind(regime.unwrap());
        }
        
        let rows = query_builder
            .bind(limit)
            .fetch_all::<StrategyPerformanceRow>()
            .await?;
            
        let performance: Vec<StrategyPerformance> = rows.into_iter().map(Into::into).collect();
        debug!("Retrieved {} strategy performance records", performance.len());
        
        Ok(performance)
    }

    // Analytics and Time Series Operations
    
    /// Get aggregated market data by time period
    pub async fn get_market_data_aggregated(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        interval_minutes: u32,
    ) -> Result<Vec<AggregatedMarketData>> {
        debug!("Fetching aggregated market data for {} ({}min intervals)", symbol, interval_minutes);
        
        let query = "
            SELECT 
                symbol,
                toStartOfInterval(timestamp, INTERVAL ? MINUTE) as interval_start,
                argMin(open, timestamp) as open,
                max(high) as high,
                min(low) as low,
                argMax(close, timestamp) as close,
                sum(volume) as volume,
                avg(adjusted_close) as avg_adjusted_close
            FROM market_data
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            GROUP BY symbol, interval_start
            ORDER BY interval_start
        ";
        
        let rows = self.client
            .query(query)
            .bind(interval_minutes)
            .bind(symbol)
            .bind(start_time)
            .bind(end_time)
            .fetch_all::<AggregatedMarketDataRow>()
            .await?;
            
        let data: Vec<AggregatedMarketData> = rows.into_iter().map(Into::into).collect();
        debug!("Retrieved {} aggregated market data points", data.len());
        
        Ok(data)
    }

    /// Get performance metrics for all strategies
    pub async fn get_all_strategy_metrics(&self) -> Result<Vec<StrategyMetrics>> {
        debug!("Fetching performance metrics for all strategies");
        
        let query = "
            SELECT 
                strategy_name,
                regime,
                avg(prediction_accuracy) as avg_accuracy,
                avg(profit_loss) as avg_profit_loss,
                avg(sharpe_ratio) as avg_sharpe_ratio,
                avg(max_drawdown) as avg_max_drawdown,
                sum(trade_count) as total_trades,
                avg(win_rate) as avg_win_rate,
                count(*) as total_records
            FROM strategy_performance
            WHERE timestamp >= now() - INTERVAL 30 DAY
            GROUP BY strategy_name, regime
            ORDER BY avg_accuracy DESC
        ";
        
        let rows = self.client
            .query(query)
            .fetch_all::<StrategyMetricsRow>()
            .await?;
            
        let metrics: Vec<StrategyMetrics> = rows.into_iter().map(Into::into).collect();
        debug!("Retrieved metrics for {} strategy-regime combinations", metrics.len());
        
        Ok(metrics)
    }

    /// Clean up old data based on retention policies
    pub async fn cleanup_old_data(&self, retention_days: u32) -> Result<CleanupResult> {
        info!("Starting data cleanup for data older than {} days", retention_days);
        
        let cutoff_date = Utc::now() - chrono::Duration::days(retention_days as i64);
        
        let mut result = CleanupResult::default();
        
        // Clean up old market data
        let market_data_deleted = self.client
            .query("DELETE FROM market_data WHERE timestamp < ?")
            .bind(cutoff_date)
            .execute()
            .await?;
        result.market_data_deleted = market_data_deleted;
        
        // Clean up old sentiment data
        let sentiment_data_deleted = self.client
            .query("DELETE FROM sentiment_data WHERE timestamp < ?")
            .bind(cutoff_date)
            .execute()
            .await?;
        result.sentiment_data_deleted = sentiment_data_deleted;
        
        // Clean up old features
        let features_deleted = self.client
            .query("DELETE FROM features WHERE timestamp < ?")
            .bind(cutoff_date)
            .execute()
            .await?;
        result.features_deleted = features_deleted;
        
        // Clean up old predictions
        let predictions_deleted = self.client
            .query("DELETE FROM predictions WHERE timestamp < ?")
            .bind(cutoff_date)
            .execute()
            .await?;
        result.predictions_deleted = predictions_deleted;
        
        info!("Data cleanup completed: {:?}", result);
        Ok(result)
    }
}

// Row structures for ClickHouse operations

#[derive(Row, Serialize, Deserialize, Debug)]
struct MarketDataRow {
    symbol: String,
    timestamp: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
    adjusted_close: f64,
}

impl From<&MarketData> for MarketDataRow {
    fn from(data: &MarketData) -> Self {
        Self {
            symbol: data.symbol.clone(),
            timestamp: data.timestamp,
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            volume: data.volume,
            adjusted_close: data.adjusted_close,
        }
    }
}

impl Into<MarketData> for MarketDataRow {
    fn into(self) -> MarketData {
        MarketData {
            symbol: self.symbol,
            timestamp: self.timestamp,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            adjusted_close: self.adjusted_close,
        }
    }
}

#[derive(Row, Serialize, Deserialize, Debug)]
struct SentimentDataRow {
    symbol: String,
    timestamp: DateTime<Utc>,
    source: String,
    sentiment_score: f32,
    confidence: f32,
    mention_count: u32,
    raw_data: String,
}

impl From<&SentimentData> for SentimentDataRow {
    fn from(data: &SentimentData) -> Self {
        Self {
            symbol: data.symbol.clone(),
            timestamp: data.timestamp,
            source: data.source.clone(),
            sentiment_score: data.sentiment_score,
            confidence: data.confidence,
            mention_count: data.mention_count,
            raw_data: data.raw_data.clone(),
        }
    }
}

impl Into<SentimentData> for SentimentDataRow {
    fn into(self) -> SentimentData {
        SentimentData {
            symbol: self.symbol,
            timestamp: self.timestamp,
            source: self.source,
            sentiment_score: self.sentiment_score,
            confidence: self.confidence,
            mention_count: self.mention_count,
            raw_data: self.raw_data,
        }
    }
}

#[derive(Row, Serialize, Deserialize, Debug)]
struct FeatureRow {
    symbol: String,
    timestamp: DateTime<Utc>,
    feature_name: String,
    feature_value: f64,
    feature_metadata: String,
    feature_version: u16,
}

#[derive(Row, Serialize, Deserialize, Debug)]
struct PredictionRow {
    symbol: String,
    timestamp: DateTime<Utc>,
    prediction_id: String,
    model_version: String,
    strategy_name: String,
    prediction_horizon: u16,
    predicted_price: f64,
    confidence: f32,
    regime: String,
    features_used: Vec<String>,
    explanation_id: Uuid,
    model_latency_ms: u16,
    strategy_parameters: HashMap<String, f64>,
    heuristic_confidence: f32,
}

#[derive(Row, Serialize, Deserialize, Debug)]
struct PredictionOutcomeRow {
    prediction_id: String,
    symbol: String,
    timestamp: DateTime<Utc>,
    actual_price: f64,
    predicted_price: f64,
    profit_loss: f64,
    accuracy_score: f32,
    strategy_name: String,
    regime: String,
}

impl From<&PredictionOutcome> for PredictionOutcomeRow {
    fn from(outcome: &PredictionOutcome) -> Self {
        Self {
            prediction_id: outcome.prediction_id.clone(),
            symbol: outcome.symbol.clone(),
            timestamp: outcome.timestamp,
            actual_price: outcome.actual_price,
            predicted_price: outcome.predicted_price,
            profit_loss: outcome.profit_loss,
            accuracy_score: outcome.accuracy_score,
            strategy_name: outcome.strategy_name.clone(),
            regime: outcome.regime.clone(),
        }
    }
}

#[derive(Row, Serialize, Deserialize, Debug)]
struct StrategyPerformanceRow {
    id: Uuid,
    strategy_name: String,
    regime: String,
    timestamp: DateTime<Utc>,
    prediction_accuracy: f32,
    profit_loss: f64,
    sharpe_ratio: f32,
    max_drawdown: f32,
    trade_count: u32,
    win_rate: f32,
    parameters: HashMap<String, f64>,
    market_conditions: HashMap<String, f64>,
}

impl From<&StrategyPerformance> for StrategyPerformanceRow {
    fn from(perf: &StrategyPerformance) -> Self {
        Self {
            id: perf.id,
            strategy_name: perf.strategy_name.clone(),
            regime: perf.regime.clone(),
            timestamp: perf.timestamp,
            prediction_accuracy: perf.prediction_accuracy,
            profit_loss: perf.profit_loss,
            sharpe_ratio: perf.sharpe_ratio,
            max_drawdown: perf.max_drawdown,
            trade_count: perf.trade_count,
            win_rate: perf.win_rate,
            parameters: perf.parameters.clone(),
            market_conditions: perf.market_conditions.clone(),
        }
    }
}

impl Into<StrategyPerformance> for StrategyPerformanceRow {
    fn into(self) -> StrategyPerformance {
        StrategyPerformance {
            id: self.id,
            strategy_name: self.strategy_name,
            regime: self.regime,
            timestamp: self.timestamp,
            prediction_accuracy: self.prediction_accuracy,
            profit_loss: self.profit_loss,
            sharpe_ratio: self.sharpe_ratio,
            max_drawdown: self.max_drawdown,
            trade_count: self.trade_count,
            win_rate: self.win_rate,
            parameters: self.parameters,
            market_conditions: self.market_conditions,
        }
    }
}

// Additional data structures for analytics

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub prediction_id: String,
    pub model_version: String,
    pub strategy_name: String,
    pub prediction_horizon: u16,
    pub predicted_price: f64,
    pub confidence: f32,
    pub regime: String,
    pub features_used: Vec<String>,
    pub explanation_id: Uuid,
    pub model_latency_ms: u16,
    pub strategy_parameters: HashMap<String, f64>,
    pub heuristic_confidence: f32,
}

impl From<&PredictionRecord> for PredictionRow {
    fn from(pred: &PredictionRecord) -> Self {
        Self {
            symbol: pred.symbol.clone(),
            timestamp: pred.timestamp,
            prediction_id: pred.prediction_id.clone(),
            model_version: pred.model_version.clone(),
            strategy_name: pred.strategy_name.clone(),
            prediction_horizon: pred.prediction_horizon,
            predicted_price: pred.predicted_price,
            confidence: pred.confidence,
            regime: pred.regime.clone(),
            features_used: pred.features_used.clone(),
            explanation_id: pred.explanation_id,
            model_latency_ms: pred.model_latency_ms,
            strategy_parameters: pred.strategy_parameters.clone(),
            heuristic_confidence: pred.heuristic_confidence,
        }
    }
}

impl Into<PredictionRecord> for PredictionRow {
    fn into(self) -> PredictionRecord {
        PredictionRecord {
            symbol: self.symbol,
            timestamp: self.timestamp,
            prediction_id: self.prediction_id,
            model_version: self.model_version,
            strategy_name: self.strategy_name,
            prediction_horizon: self.prediction_horizon,
            predicted_price: self.predicted_price,
            confidence: self.confidence,
            regime: self.regime,
            features_used: self.features_used,
            explanation_id: self.explanation_id,
            model_latency_ms: self.model_latency_ms,
            strategy_parameters: self.strategy_parameters,
            heuristic_confidence: self.heuristic_confidence,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMarketData {
    pub symbol: String,
    pub interval_start: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub avg_adjusted_close: f64,
}

#[derive(Row, Serialize, Deserialize, Debug)]
struct AggregatedMarketDataRow {
    symbol: String,
    interval_start: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
    avg_adjusted_close: f64,
}

impl Into<AggregatedMarketData> for AggregatedMarketDataRow {
    fn into(self) -> AggregatedMarketData {
        AggregatedMarketData {
            symbol: self.symbol,
            interval_start: self.interval_start,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            avg_adjusted_close: self.avg_adjusted_close,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub strategy_name: String,
    pub regime: String,
    pub avg_accuracy: f64,
    pub avg_profit_loss: f64,
    pub avg_sharpe_ratio: f64,
    pub avg_max_drawdown: f64,
    pub total_trades: u64,
    pub avg_win_rate: f64,
    pub total_records: u64,
}

#[derive(Row, Serialize, Deserialize, Debug)]
struct StrategyMetricsRow {
    strategy_name: String,
    regime: String,
    avg_accuracy: f64,
    avg_profit_loss: f64,
    avg_sharpe_ratio: f64,
    avg_max_drawdown: f64,
    total_trades: u64,
    avg_win_rate: f64,
    total_records: u64,
}

impl Into<StrategyMetrics> for StrategyMetricsRow {
    fn into(self) -> StrategyMetrics {
        StrategyMetrics {
            strategy_name: self.strategy_name,
            regime: self.regime,
            avg_accuracy: self.avg_accuracy,
            avg_profit_loss: self.avg_profit_loss,
            avg_sharpe_ratio: self.avg_sharpe_ratio,
            avg_max_drawdown: self.avg_max_drawdown,
            total_trades: self.total_trades,
            avg_win_rate: self.avg_win_rate,
            total_records: self.total_records,
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CleanupResult {
    pub market_data_deleted: u64,
    pub sentiment_data_deleted: u64,
    pub features_deleted: u64,
    pub predictions_deleted: u64,
}