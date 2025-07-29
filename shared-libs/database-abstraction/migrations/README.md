# QuantumTrade AI Database Migrations

This directory contains all database migrations for the QuantumTrade AI system.

## Migration Files

### ClickHouse Migrations

1. **001_create_market_data.sql** - Creates the market data table with optimized schema for time-series data
2. **002_create_sentiment_data.sql** - Creates sentiment data table with text search capabilities
3. **003_create_features.sql** - Creates features and technical indicators tables
4. **004_create_predictions.sql** - Creates predictions table with explanation support
5. **005_create_prediction_outcomes.sql** - Creates prediction outcomes table for tracking accuracy
6. **006_create_strategy_performance.sql** - Creates strategy performance table with rankings

## Schema Design

### Market Data Table
- **Engine**: ReplacingMergeTree for efficient time-series operations
- **Partitioning**: By symbol and month for optimal query performance
- **Indexing**: Primary key on (symbol, timestamp) for fast lookups
- **Materialized Views**: Daily aggregates for common analytics

### Sentiment Data Table
- **Engine**: ReplacingMergeTree with text search capabilities
- **Features**: Token-based full-text search index on title and content
- **JSON Storage**: Entities stored as JSON strings for flexibility

### Features Table
- **Flexible Schema**: Features stored as JSON for easy extension
- **Versioning**: Feature version tracking for model compatibility
- **Metadata**: Separate metadata storage for feature descriptions

### Predictions Table
- **Explanation Support**: Optional JSON explanations for model interpretability
- **Bloom Filter**: Fast prediction ID lookups
- **Strategy Tracking**: Links predictions to specific strategies

## Running Migrations

Migrations are automatically run when the database manager is initialized:

```rust
let db = DatabaseManager::new(config).await?;
db.run_migrations().await?;
```

## Migration Tracking

Migrations are tracked in the `__migrations` table with:
- Version number
- Migration name
- Execution timestamp

## Best Practices

1. **Never modify existing migrations** - Create new ones instead
2. **Test migrations** in development before production
3. **Backup data** before running migrations in production
4. **Monitor performance** after schema changes
5. **Use appropriate data types** for ClickHouse optimization

## Performance Considerations

- **Partitioning**: Monthly partitions for optimal query performance
- **Indexing**: Strategic indexes for common query patterns
- **Materialized Views**: Pre-computed aggregates for analytics
- **Data Types**: Optimized types (LowCardinality, Decimal64) for storage efficiency