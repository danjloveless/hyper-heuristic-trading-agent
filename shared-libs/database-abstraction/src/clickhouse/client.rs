use async_trait::async_trait;
use chrono::{DateTime, Utc};
use clickhouse::{Client, Row};
use shared_types::*;
use tracing::info;
use num_traits::cast::ToPrimitive;
use num_traits::FromPrimitive;

use crate::{
    config::ClickHouseConfig,
    errors::DatabaseError,
    health::{HealthCheck, HealthStatus},
    traits::DatabaseClient,
};

pub struct ClickHouseClient {
    client: Client,
    config: ClickHouseConfig,
}

impl ClickHouseClient {
    pub async fn new(config: ClickHouseConfig) -> Result<Self, DatabaseError> {
        let mut client_builder = Client::default()
            .with_url(&config.url)
            .with_database(&config.database);

        if let Some(username) = &config.username {
            client_builder = client_builder.with_user(username);
        }

        if let Some(password) = &config.password {
            client_builder = client_builder.with_password(password);
        }

        let client = client_builder;

        // Test connection
        let test_query = "SELECT 1";
        client.query(test_query)
            .fetch_all::<u8>()
            .await
            .map_err(|e| DatabaseError::ConnectionError {
                message: format!("Failed to connect to ClickHouse: {}", e),
            })?;

        info!("Successfully connected to ClickHouse database: {}", config.database);

        Ok(Self { client, config })
    }

    pub async fn run_migrations(&self) -> Result<(), DatabaseError> {
        use crate::clickhouse::migrations::*;
        
        info!("Running ClickHouse migrations");
        
        // Create migrations table if it doesn't exist
        let create_migrations_table = r#"
            CREATE TABLE IF NOT EXISTS __migrations (
                version UInt32,
                name String,
                executed_at DateTime64(3, 'UTC') DEFAULT now64(3)
            ) ENGINE = MergeTree()
            ORDER BY version
        "#;
        
        self.client.query(create_migrations_table)
            .execute()
            .await
            .map_err(|e| DatabaseError::MigrationError {
                message: format!("Failed to create migrations table: {}", e),
            })?;

        // Get executed migrations
        let executed_migrations: Vec<u32> = self.client
            .query("SELECT version FROM __migrations ORDER BY version")
            .fetch_all()
            .await
            .map_err(|e| DatabaseError::MigrationError {
                message: format!("Failed to fetch executed migrations: {}", e),
            })?;

        // Execute pending migrations
        let migrations = get_migrations();
        for migration in migrations {
            if !executed_migrations.contains(&migration.version) {
                info!("Executing migration {}: {}", migration.version, migration.name);
                
                self.client.query(&migration.sql)
                    .execute()
                    .await
                    .map_err(|e| DatabaseError::MigrationError {
                        message: format!("Failed to execute migration {} ({}): {}", 
                            migration.version, migration.name, e),
                    })?;
                
                // Record migration as executed
                self.client.query(
                    "INSERT INTO __migrations (version, name) VALUES (?, ?)"
                )
                .bind(migration.version)
                .bind(&migration.name)
                .execute()
                .await
                .map_err(|e| DatabaseError::MigrationError {
                    message: format!("Failed to record migration {}: {}", migration.version, e),
                })?;
            }
        }
        
        info!("ClickHouse migrations completed successfully");
        Ok(())
    }
}

// Row structs for ClickHouse tables
#[derive(Row, serde::Deserialize, serde::Serialize)]
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

#[derive(Row, serde::Deserialize, serde::Serialize)]
struct SentimentDataRow {
    article_id: String,
    symbol: String,
    timestamp: DateTime<Utc>,
    title: String,
    content: String,
    source: String,
    sentiment_score: f32,
    confidence: f32,
    entities: String,
    relevance_score: f32,
    market_impact: Option<f32>,
}

#[derive(Row, serde::Deserialize, serde::Serialize)]
struct AggregatedSentimentRow {
    symbol: String,
    timestamp: DateTime<Utc>,
    overall_sentiment: f32,
    confidence_weighted_sentiment: f32,
    article_count: u32,
    bullish_count: u32,
    bearish_count: u32,
    neutral_count: u32,
}

#[derive(Row, serde::Deserialize, serde::Serialize)]
struct FeatureSetRow {
    symbol: String,
    timestamp: DateTime<Utc>,
    features: String,
    feature_metadata: String,
    feature_version: u16,
}

#[derive(Row, serde::Deserialize, serde::Serialize)]
struct TechnicalIndicatorsRow {
    symbol: String,
    timestamp: DateTime<Utc>,
    sma_20: f64,
    ema_12: f64,
    ema_26: f64,
    macd: f64,
    macd_signal: f64,
    rsi: f64,
    bollinger_upper: f64,
    bollinger_lower: f64,
    volume_sma_20: f64,
}

#[derive(Row, serde::Deserialize, serde::Serialize)]
struct PredictionResultRow {
    prediction_id: String,
    symbol: String,
    timestamp: DateTime<Utc>,
    predicted_price: f64,
    confidence: f32,
    horizon_minutes: u16,
    strategy_name: String,
    model_version: String,
    explanation: Option<String>,
}

#[derive(Row, serde::Deserialize, serde::Serialize)]
struct PredictionOutcomeRow {
    prediction_id: String,
    actual_price: f64,
    accuracy_score: f32,
    directional_accuracy: bool,
    outcome_timestamp: DateTime<Utc>,
}

#[derive(Row, serde::Deserialize, serde::Serialize)]
struct StrategyPerformanceRow {
    strategy_name: String,
    symbol: String,
    timestamp: DateTime<Utc>,
    total_predictions: u32,
    accuracy_rate: f32,
    avg_confidence: f32,
    profit_loss: f64,
    sharpe_ratio: f32,
    max_drawdown: f32,
}

impl From<MarketDataRow> for MarketData {
    fn from(row: MarketDataRow) -> Self {
        Self {
            symbol: row.symbol,
            timestamp: row.timestamp,
            open: rust_decimal::Decimal::from_f64(row.open).unwrap_or_default(),
            high: rust_decimal::Decimal::from_f64(row.high).unwrap_or_default(),
            low: rust_decimal::Decimal::from_f64(row.low).unwrap_or_default(),
            close: rust_decimal::Decimal::from_f64(row.close).unwrap_or_default(),
            volume: row.volume,
            adjusted_close: rust_decimal::Decimal::from_f64(row.adjusted_close).unwrap_or_default(),
        }
    }
}

impl From<SentimentDataRow> for SentimentData {
    fn from(row: SentimentDataRow) -> Self {
        Self {
            article_id: row.article_id,
            symbol: row.symbol,
            timestamp: row.timestamp,
            title: row.title,
            content: row.content,
            source: row.source,
            sentiment_score: row.sentiment_score,
            confidence: row.confidence,
            entities: serde_json::from_str(&row.entities).unwrap_or_default(),
            relevance_score: row.relevance_score,
            market_impact: row.market_impact,
        }
    }
}

impl From<AggregatedSentimentRow> for AggregatedSentiment {
    fn from(row: AggregatedSentimentRow) -> Self {
        Self {
            symbol: row.symbol,
            timestamp: row.timestamp,
            overall_sentiment: row.overall_sentiment,
            confidence_weighted_sentiment: row.confidence_weighted_sentiment,
            article_count: row.article_count,
            bullish_count: row.bullish_count,
            bearish_count: row.bearish_count,
            neutral_count: row.neutral_count,
        }
    }
}

impl From<FeatureSetRow> for FeatureSet {
    fn from(row: FeatureSetRow) -> Self {
        Self {
            symbol: row.symbol,
            timestamp: row.timestamp,
            features: serde_json::from_str(&row.features).unwrap_or_default(),
            feature_metadata: serde_json::from_str(&row.feature_metadata).unwrap_or_default(),
            feature_version: row.feature_version,
        }
    }
}

impl From<PredictionResultRow> for PredictionResult {
    fn from(row: PredictionResultRow) -> Self {
        Self {
            prediction_id: row.prediction_id,
            symbol: row.symbol,
            timestamp: row.timestamp,
            predicted_price: rust_decimal::Decimal::from_f64(row.predicted_price).unwrap_or_default(),
            confidence: row.confidence,
            horizon_minutes: row.horizon_minutes,
            strategy_name: row.strategy_name,
            model_version: row.model_version,
            explanation: row.explanation.and_then(|s| serde_json::from_str(&s).ok()),
        }
    }
}

impl From<PredictionOutcomeRow> for PredictionOutcome {
    fn from(row: PredictionOutcomeRow) -> Self {
        Self {
            prediction_id: row.prediction_id,
            actual_price: rust_decimal::Decimal::from_f64(row.actual_price).unwrap_or_default(),
            accuracy_score: row.accuracy_score,
            directional_accuracy: row.directional_accuracy,
            outcome_timestamp: row.outcome_timestamp,
        }
    }
}

impl From<StrategyPerformanceRow> for StrategyPerformance {
    fn from(row: StrategyPerformanceRow) -> Self {
        Self {
            strategy_name: row.strategy_name,
            symbol: row.symbol,
            timestamp: row.timestamp,
            total_predictions: row.total_predictions,
            accuracy_rate: row.accuracy_rate,
            avg_confidence: row.avg_confidence,
            profit_loss: rust_decimal::Decimal::from_f64(row.profit_loss).unwrap_or_default(),
            sharpe_ratio: row.sharpe_ratio,
            max_drawdown: row.max_drawdown,
        }
    }
}

#[async_trait]
impl DatabaseClient for ClickHouseClient {
    async fn insert_market_data(&self, data: &[MarketData]) -> Result<(), DatabaseError> {
        if data.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("market_data")?;
        
        for item in data {
            insert.write(&MarketDataRow {
                symbol: item.symbol.clone(),
                timestamp: item.timestamp,
                open: item.open.to_f64().unwrap_or(0.0),
                high: item.high.to_f64().unwrap_or(0.0),
                low: item.low.to_f64().unwrap_or(0.0),
                close: item.close.to_f64().unwrap_or(0.0),
                volume: item.volume,
                adjusted_close: item.adjusted_close.to_f64().unwrap_or(0.0),
            }).await?;
        }
        
        insert.end().await?;
        Ok(())
    }

    async fn get_market_data(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MarketData>, DatabaseError> {
        let query = r#"
            SELECT symbol, timestamp, open, high, low, close, volume, adjusted_close
            FROM market_data
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        "#;

        let rows: Vec<MarketDataRow> = self.client
            .query(query)
            .bind(symbol)
            .bind(start)
            .bind(end)
            .fetch_all()
            .await?;

        Ok(rows.into_iter().map(MarketData::from).collect())
    }

    async fn get_latest_market_data(&self, symbol: &str) -> Result<Option<MarketData>, DatabaseError> {
        let query = r#"
            SELECT symbol, timestamp, open, high, low, close, volume, adjusted_close
            FROM market_data
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        "#;

        let rows: Vec<MarketDataRow> = self.client
            .query(query)
            .bind(symbol)
            .fetch_all()
            .await?;

        Ok(rows.into_iter().next().map(MarketData::from))
    }

    async fn insert_sentiment_data(&self, data: &[SentimentData]) -> Result<(), DatabaseError> {
        if data.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("sentiment_data")?;
        
        for item in data {
            let entities_json = serde_json::to_string(&item.entities)?;
            insert.write(&SentimentDataRow {
                article_id: item.article_id.clone(),
                symbol: item.symbol.clone(),
                timestamp: item.timestamp,
                title: item.title.clone(),
                content: item.content.clone(),
                source: item.source.clone(),
                sentiment_score: item.sentiment_score,
                confidence: item.confidence,
                entities: entities_json,
                relevance_score: item.relevance_score,
                market_impact: item.market_impact,
            }).await?;
        }
        
        insert.end().await?;
        Ok(())
    }

    async fn get_sentiment_data(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<SentimentData>, DatabaseError> {
        let query = r#"
            SELECT article_id, symbol, timestamp, title, content, source, 
                   sentiment_score, confidence, entities, relevance_score, market_impact
            FROM sentiment_data
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        "#;

        let rows: Vec<SentimentDataRow> = self.client
            .query(query)
            .bind(symbol)
            .bind(start)
            .bind(end)
            .fetch_all()
            .await?;

        Ok(rows.into_iter().map(SentimentData::from).collect())
    }

    async fn get_aggregated_sentiment(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<AggregatedSentiment>, DatabaseError> {
        let query = r#"
            SELECT 
                symbol,
                ? as timestamp,
                avg(sentiment_score) as overall_sentiment,
                sum(sentiment_score * confidence) / sum(confidence) as confidence_weighted_sentiment,
                count(*) as article_count,
                countIf(sentiment_score > 0.1) as bullish_count,
                countIf(sentiment_score < -0.1) as bearish_count,
                countIf(sentiment_score >= -0.1 AND sentiment_score <= 0.1) as neutral_count
            FROM sentiment_data
            WHERE symbol = ? AND timestamp >= ? - INTERVAL 1 HOUR AND timestamp <= ?
            GROUP BY symbol
        "#;

        let rows: Vec<AggregatedSentimentRow> = self.client
            .query(query)
            .bind(timestamp)
            .bind(symbol)
            .bind(timestamp)
            .bind(timestamp)
            .fetch_all()
            .await?;

        Ok(rows.into_iter().next().map(AggregatedSentiment::from))
    }

    async fn insert_features(&self, features: &[FeatureSet]) -> Result<(), DatabaseError> {
        if features.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("features")?;
        
        for item in features {
            let features_json = serde_json::to_string(&item.features)?;
            let metadata_json = serde_json::to_string(&item.feature_metadata)?;
            
            insert.write(&FeatureSetRow {
                symbol: item.symbol.clone(),
                timestamp: item.timestamp,
                features: features_json,
                feature_metadata: metadata_json,
                feature_version: item.feature_version,
            }).await?;
        }
        
        insert.end().await?;
        Ok(())
    }

    async fn get_features(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<FeatureSet>, DatabaseError> {
        let query = r#"
            SELECT symbol, timestamp, features, feature_metadata, feature_version
            FROM features
            WHERE symbol = ? AND timestamp = ?
            LIMIT 1
        "#;

        let rows: Vec<FeatureSetRow> = self.client
            .query(query)
            .bind(symbol)
            .bind(timestamp)
            .fetch_all()
            .await?;

        Ok(rows.into_iter().next().map(FeatureSet::from))
    }

    async fn get_latest_features(&self, symbol: &str) -> Result<Option<FeatureSet>, DatabaseError> {
        let query = r#"
            SELECT symbol, timestamp, features, feature_metadata, feature_version
            FROM features
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        "#;

        let rows: Vec<FeatureSetRow> = self.client
            .query(query)
            .bind(symbol)
            .fetch_all()
            .await?;

        Ok(rows.into_iter().next().map(FeatureSet::from))
    }

    async fn insert_technical_indicators(&self, indicators: &[TechnicalIndicators]) -> Result<(), DatabaseError> {
        if indicators.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("technical_indicators")?;
        
        for item in indicators {
            insert.write(&TechnicalIndicatorsRow {
                symbol: item.symbol.clone(),
                timestamp: item.timestamp,
                sma_20: item.sma_20.unwrap_or(0.0),
                ema_12: item.ema_12.unwrap_or(0.0),
                ema_26: item.ema_26.unwrap_or(0.0),
                macd: item.macd.unwrap_or(0.0),
                macd_signal: item.macd_signal.unwrap_or(0.0),
                rsi: item.rsi.unwrap_or(0.0),
                bollinger_upper: item.bollinger_upper.unwrap_or(0.0),
                bollinger_lower: item.bollinger_lower.unwrap_or(0.0),
                volume_sma_20: item.volume_sma_20.unwrap_or(0.0),
            }).await?;
        }
        
        insert.end().await?;
        Ok(())
    }

    async fn insert_predictions(&self, predictions: &[PredictionResult]) -> Result<(), DatabaseError> {
        if predictions.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("predictions")?;
        
        for item in predictions {
            let explanation_json = if let Some(ref explanation) = item.explanation {
                Some(serde_json::to_string(explanation)?)
            } else {
                None
            };
            
            insert.write(&PredictionResultRow {
                prediction_id: item.prediction_id.clone(),
                symbol: item.symbol.clone(),
                timestamp: item.timestamp,
                predicted_price: item.predicted_price.to_f64().unwrap_or(0.0),
                confidence: item.confidence,
                horizon_minutes: item.horizon_minutes,
                strategy_name: item.strategy_name.clone(),
                model_version: item.model_version.clone(),
                explanation: explanation_json,
            }).await?;
        }
        
        insert.end().await?;
        Ok(())
    }

    async fn get_predictions(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<PredictionResult>, DatabaseError> {
        let query = r#"
            SELECT prediction_id, symbol, timestamp, predicted_price, confidence, 
                   horizon_minutes, strategy_name, model_version, explanation
            FROM predictions
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        "#;

        let rows: Vec<PredictionResultRow> = self.client
            .query(query)
            .bind(symbol)
            .bind(start)
            .bind(end)
            .fetch_all()
            .await?;

        Ok(rows.into_iter().map(PredictionResult::from).collect())
    }

    async fn insert_prediction_outcomes(&self, outcomes: &[PredictionOutcome]) -> Result<(), DatabaseError> {
        if outcomes.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("prediction_outcomes")?;
        
        for item in outcomes {
            insert.write(&PredictionOutcomeRow {
                prediction_id: item.prediction_id.clone(),
                actual_price: item.actual_price.to_f64().unwrap_or(0.0),
                accuracy_score: item.accuracy_score,
                directional_accuracy: item.directional_accuracy,
                outcome_timestamp: item.outcome_timestamp,
            }).await?;
        }
        
        insert.end().await?;
        Ok(())
    }

    async fn insert_strategy_performance(&self, performance: &[StrategyPerformance]) -> Result<(), DatabaseError> {
        if performance.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("strategy_performance")?;
        
        for item in performance {
            insert.write(&StrategyPerformanceRow {
                strategy_name: item.strategy_name.clone(),
                symbol: item.symbol.clone(),
                timestamp: item.timestamp,
                total_predictions: item.total_predictions,
                accuracy_rate: item.accuracy_rate,
                avg_confidence: item.avg_confidence,
                profit_loss: item.profit_loss.to_f64().unwrap_or(0.0),
                sharpe_ratio: item.sharpe_ratio,
                max_drawdown: item.max_drawdown,
            }).await?;
        }
        
        insert.end().await?;
        Ok(())
    }

    async fn get_strategy_performance(
        &self,
        strategy_name: &str,
        symbol: Option<&str>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<StrategyPerformance>, DatabaseError> {
        let query = if let Some(_symbol) = symbol {
            r#"
            SELECT strategy_name, symbol, timestamp, total_predictions, accuracy_rate,
                   avg_confidence, profit_loss, sharpe_ratio, max_drawdown
            FROM strategy_performance
            WHERE strategy_name = ? AND symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            "#.to_string()
        } else {
            r#"
            SELECT strategy_name, symbol, timestamp, total_predictions, accuracy_rate,
                   avg_confidence, profit_loss, sharpe_ratio, max_drawdown
            FROM strategy_performance
            WHERE strategy_name = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            "#.to_string()
        };

        let mut query_builder = self.client.query(&query);
        query_builder = query_builder.bind(strategy_name);
        if let Some(symbol) = symbol {
            query_builder = query_builder.bind(symbol);
        }
        query_builder = query_builder.bind(start).bind(end);

        let rows: Vec<StrategyPerformanceRow> = query_builder
            .fetch_all()
            .await?;

        Ok(rows.into_iter().map(StrategyPerformance::from).collect())
    }

    async fn health_check(&self) -> Result<HealthStatus, DatabaseError> {
        let start = std::time::Instant::now();
        let mut health = HealthStatus::new("ClickHouse".to_string());

        // Connection check
        let connection_result = self.client.query("SELECT 1").fetch_all::<u8>().await;
        let connection_latency = start.elapsed().as_millis() as u64;

        let connection_check = match connection_result {
            Ok(_) => HealthCheck {
                name: "connection".to_string(),
                status: ServiceStatus::Healthy,
                latency_ms: connection_latency,
                error: None,
            },
            Err(e) => HealthCheck {
                name: "connection".to_string(),
                status: ServiceStatus::Unhealthy,
                latency_ms: connection_latency,
                error: Some(e.to_string()),
            },
        };

        health.add_check(connection_check);

        // Schema check
        let schema_start = std::time::Instant::now();
        let schema_result = self.client
            .query("SELECT count(*) FROM information_schema.tables WHERE table_schema = ?")
            .bind(&self.config.database)
            .fetch_all::<u64>()
            .await;
        let schema_latency = schema_start.elapsed().as_millis() as u64;

        let schema_check = match schema_result {
            Ok(tables) if !tables.is_empty() && tables[0] > 0 => HealthCheck {
                name: "schema".to_string(),
                status: ServiceStatus::Healthy,
                latency_ms: schema_latency,
                error: None,
            },
            Ok(_) => HealthCheck {
                name: "schema".to_string(),
                status: ServiceStatus::Degraded,
                latency_ms: schema_latency,
                error: Some("No tables found in database".to_string()),
            },
            Err(e) => HealthCheck {
                name: "schema".to_string(),
                status: ServiceStatus::Unhealthy,
                latency_ms: schema_latency,
                error: Some(e.to_string()),
            },
        };

        health.add_check(schema_check);

        Ok(health)
    }
}