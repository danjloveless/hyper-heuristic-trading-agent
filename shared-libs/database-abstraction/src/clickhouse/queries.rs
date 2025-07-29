// Complex ClickHouse queries for advanced analytics
// This module contains optimized queries for common analytical operations

pub const MARKET_DATA_ANALYTICS_QUERY: &str = r#"
    SELECT 
        symbol,
        toStartOfHour(timestamp) as hour,
        avg(close) as avg_price,
        max(high) as max_price,
        min(low) as min_price,
        sum(volume) as total_volume,
        count(*) as data_points
    FROM market_data
    WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
    GROUP BY symbol, hour
    ORDER BY hour
"#;

pub const SENTIMENT_ANALYSIS_QUERY: &str = r#"
    SELECT 
        symbol,
        toStartOfHour(timestamp) as hour,
        avg(sentiment_score) as avg_sentiment,
        avg(confidence) as avg_confidence,
        count(*) as article_count,
        countIf(sentiment_score > 0.1) as bullish_articles,
        countIf(sentiment_score < -0.1) as bearish_articles
    FROM sentiment_data
    WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
    GROUP BY symbol, hour
    ORDER BY hour
"#;

pub const STRATEGY_PERFORMANCE_ANALYTICS_QUERY: &str = r#"
    SELECT 
        strategy_name,
        symbol,
        toStartOfDay(timestamp) as day,
        avg(accuracy_rate) as avg_accuracy,
        avg(sharpe_ratio) as avg_sharpe,
        sum(profit_loss) as total_pnl,
        max(max_drawdown) as max_drawdown,
        count(*) as performance_records
    FROM strategy_performance
    WHERE strategy_name = ? AND timestamp >= ? AND timestamp <= ?
    GROUP BY strategy_name, symbol, day
    ORDER BY day DESC
"#;