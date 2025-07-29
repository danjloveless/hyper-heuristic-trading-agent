use async_trait::async_trait;
use chrono::{DateTime, Utc};
use clickhouse::{Client, Row};
use shared_types::*;
use tracing::{error, info, warn};

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
                        message: format!(
                            "Failed to execute migration {} ({}): {}", 
                            migration.version, migration.name, e
                        ),
                    })?;

                // Record migration execution
                let record_sql = format!(
                    "INSERT INTO __migrations (version, name) VALUES ({}, '{}')",
                    migration.version, migration.name
                );
                
                self.client.query(&record_sql)
                    .execute()
                    .await
                    .map_err(|e| DatabaseError::MigrationError {
                        message: format!("Failed to record migration execution: {}", e),
                    })?;
            }
        }

        info!("ClickHouse migrations completed successfully");
        Ok(())
    }

    async fn execute_with_retry<F, T>(&self, operation: F) -> Result<T, DatabaseError>
    where
        F: Fn() -> futures::future::BoxFuture<'_, Result<T, clickhouse::error::Error>>,
    {
        let mut last_error = None;
        
        for attempt in 0..self.config.retry_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.retry_attempts - 1 {
                        warn!("ClickHouse operation failed, retrying (attempt {})", attempt + 1);
                        tokio::time::sleep(std::time::Duration::from_millis(100 * (attempt + 1) as u64)).await;
                    }
                }
            }
        }
        
        Err(DatabaseError::ClickHouseError(last_error.unwrap()))
    }
}

#[derive(Row, serde::Deserialize)]
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

#[async_trait]
impl DatabaseClient for ClickHouseClient {
    async fn insert_market_data(&self, data: &[MarketData]) -> Result<(), DatabaseError> {
        if data.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("market_data")?;
        
        for item in data {
            insert.write(&(
                &item.symbol,
                item.timestamp,
                item.open.to_f64().unwrap_or(0.0),
                item.high.to_f64().unwrap_or(0.0),
                item.low.to_f64().unwrap_or(0.0),
                item.close.to_f64().unwrap_or(0.0),
                item.volume,
                item.adjusted_close.to_f64().unwrap_or(0.0),
            )).await?;
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
            insert.write(&(
                &item.article_id,
                &item.symbol,
                item.timestamp,
                &item.title,
                &item.content,
                &item.source,
                item.sentiment_score,
                item.confidence,
                entities_json,
                item.relevance_score,
                item.market_impact,
            )).await?;
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

        let rows: Vec<(String, String, DateTime<Utc>, String, String, String, f32, f32, String, f32, Option<f32>)> = 
            self.client
                .query(query)
                .bind(symbol)
                .bind(start)
                .bind(end)
                .fetch_all()
                .await?;

        let mut results = Vec::new();
        for row in rows {
            let entities = serde_json::from_str(&row.8).unwrap_or_default();
            results.push(SentimentData {
                article_id: row.0,
                symbol: row.1,
                timestamp: row.2,
                title: row.3,
                content: row.4,
                source: row.5,
                sentiment_score: row.6,
                confidence: row.7,
                entities,
                relevance_score: row.9,
                market_impact: row.10,
            });
        }

        Ok(results)
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

        let rows: Vec<(String, DateTime<Utc>, f32, f32, u32, u32, u32, u32)> = self.client
            .query(query)
            .bind(timestamp)
            .bind(symbol)
            .bind(timestamp)
            .bind(timestamp)
            .fetch_all()
            .await?;

        Ok(rows.into_iter().next().map(|row| AggregatedSentiment {
            symbol: row.0,
            timestamp: row.1,
            overall_sentiment: row.2,
            confidence_weighted_sentiment: row.3,
            article_count: row.4,
            bullish_count: row.5,
            bearish_count: row.6,
            neutral_count: row.7,
        }))
    }

    async fn insert_features(&self, features: &[FeatureSet]) -> Result<(), DatabaseError> {
        if features.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("features")?;
        
        for item in features {
            let features_json = serde_json::to_string(&item.features)?;
            let metadata_json = serde_json::to_string(&item.feature_metadata)?;
            
            insert.write(&(
                &item.symbol,
                item.timestamp,
                features_json,
                metadata_json,
                item.feature_version,
            )).await?;
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

        let rows: Vec<(String, DateTime<Utc>, String, String, u16)> = self.client
            .query(query)
            .bind(symbol)
            .bind(timestamp)
            .fetch_all()
            .await?;

        if let Some(row) = rows.into_iter().next() {
            Ok(Some(FeatureSet {
                symbol: row.0,
                timestamp: row.1,
                features: serde_json::from_str(&row.2)?,
                feature_metadata: serde_json::from_str(&row.3)?,
                feature_version: row.4,
            }))
        } else {
            Ok(None)
        }
    }

    async fn get_latest_features(&self, symbol: &str) -> Result<Option<FeatureSet>, DatabaseError> {
        let query = r#"
            SELECT symbol, timestamp, features, feature_metadata, feature_version
            FROM features
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        "#;

        let rows: Vec<(String, DateTime<Utc>, String, String, u16)> = self.client
            .query(query)
            .bind(symbol)
            .fetch_all()
            .await?;

        if let Some(row) = rows.into_iter().next() {
            Ok(Some(FeatureSet {
                symbol: row.0,
                timestamp: row.1,
                features: serde_json::from_str(&row.2)?,
                feature_metadata: serde_json::from_str(&row.3)?,
                feature_version: row.4,
            }))
        } else {
            Ok(None)
        }
    }

    async fn insert_technical_indicators(&self, indicators: &[TechnicalIndicators]) -> Result<(), DatabaseError> {
        if indicators.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("technical_indicators")?;
        
        for item in indicators {
            insert.write(&(
                &item.symbol,
                item.timestamp,
                item.sma_20,
                item.ema_12,
                item.ema_26,
                item.macd,
                item.macd_signal,
                item.rsi,
                item.bollinger_upper,
                item.bollinger_lower,
                item.volume_sma_20,
            )).await?;
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
            
            insert.write(&(
                &item.prediction_id,
                &item.symbol,
                item.timestamp,
                item.predicted_price.to_f64().unwrap_or(0.0),
                item.confidence,
                item.horizon_minutes,
                &item.strategy_name,
                &item.model_version,
                explanation_json,
            )).await?;
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

        let rows: Vec<(String, String, DateTime<Utc>, f64, f32, u16, String, String, Option<String>)> = 
            self.client
                .query(query)
                .bind(symbol)
                .bind(start)
                .bind(end)
                .fetch_all()
                .await?;

        let mut results = Vec::new();
        for row in rows {
            let explanation = if let Some(explanation_str) = row.8 {
                serde_json::from_str(&explanation_str).ok()
            } else {
                None
            };
            
            results.push(PredictionResult {
                prediction_id: row.0,
                symbol: row.1,
                timestamp: row.2,
                predicted_price: rust_decimal::Decimal::from_f64(row.3).unwrap_or_default(),
                confidence: row.4,
                horizon_minutes: row.5,
                strategy_name: row.6,
                model_version: row.7,
                explanation,
            });
        }

        Ok(results)
    }

    async fn insert_prediction_outcomes(&self, outcomes: &[PredictionOutcome]) -> Result<(), DatabaseError> {
        if outcomes.is_empty() {
            return Ok(());
        }

        let mut insert = self.client.insert("prediction_outcomes")?;
        
        for item in outcomes {
            insert.write(&(
                &item.prediction_id,
                item.actual_price.to_f64().unwrap_or(0.0),
                item.accuracy_score,
                item.directional_accuracy,
                item.outcome_timestamp,
            )).await?;
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
            insert.write(&(
                &item.strategy_name,
                &item.symbol,
                item.timestamp,
                item.total_predictions,
                item.accuracy_rate,
                item.avg_confidence,
                item.profit_loss.to_f64().unwrap_or(0.0),
                item.sharpe_ratio,
                item.max_drawdown,
            )).await?;
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
        let (query, params): (String, Vec<&dyn clickhouse::bind::Bind>) = if let Some(symbol) = symbol {
            (
                r#"
                SELECT strategy_name, symbol, timestamp, total_predictions, accuracy_rate,
                       avg_confidence, profit_loss, sharpe_ratio, max_drawdown
                FROM strategy_performance
                WHERE strategy_name = ? AND symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                "#.to_string(),
                vec![&strategy_name, &symbol, &start, &end]
            )
        } else {
            (
                r#"
                SELECT strategy_name, symbol, timestamp, total_predictions, accuracy_rate,
                       avg_confidence, profit_loss, sharpe_ratio, max_drawdown
                FROM strategy_performance
                WHERE strategy_name = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                "#.to_string(),
                vec![&strategy_name, &start, &end]
            )
        };

        let mut query_builder = self.client.query(&query);
        for param in params {
            query_builder = query_builder.bind(param);
        }

        let rows: Vec<(String, String, DateTime<Utc>, u32, f32, f32, f64, f32, f32)> = query_builder
            .fetch_all()
            .await?;

        let results = rows.into_iter().map(|row| StrategyPerformance {
            strategy_name: row.0,
            symbol: row.1,
            timestamp: row.2,
            total_predictions: row.3,
            accuracy_rate: row.4,
            avg_confidence: row.5,
            profit_loss: rust_decimal::Decimal::from_f64(row.6).unwrap_or_default(),
            sharpe_ratio: row.7,
            max_drawdown: row.8,
        }).collect();

        Ok(results)
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