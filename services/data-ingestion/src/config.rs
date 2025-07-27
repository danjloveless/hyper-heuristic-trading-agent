// services/data-ingestion/src/config.rs
use serde::{Deserialize, Serialize};
use std::time::Duration;
use shared_utils::{QuantumTradeError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataIngestionConfig {
    pub yahoo_finance: YahooFinanceConfig,
    pub reddit: RedditConfig,
    pub news: NewsConfig,
    pub scheduler: SchedulerConfig,
    pub rate_limits: RateLimitConfig,
    pub symbols: SymbolConfig,
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YahooFinanceConfig {
    pub base_url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub retry_delay_seconds: u64,
    pub user_agent: String,
    pub rate_limit_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedditConfig {
    pub client_id: String,
    pub client_secret: String,
    pub user_agent: String,
    pub username: String,
    pub password: String,
    pub base_url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub rate_limit_per_minute: u32,
    pub subreddits: Vec<String>,
    pub min_score: i32,
    pub max_posts_per_request: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub rate_limit_per_minute: u32,
    pub sources: Vec<String>,
    pub languages: Vec<String>,
    pub max_articles_per_request: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub market_data_cron: String,        // "0 */5 * * * *" - every 5 minutes
    pub reddit_sentiment_cron: String,   // "0 */15 * * * *" - every 15 minutes
    pub news_sentiment_cron: String,     // "0 */30 * * * *" - every 30 minutes
    pub cleanup_cron: String,           // "0 0 2 * * *" - daily at 2 AM
    pub enable_market_hours_only: bool,
    pub timezone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub burst_capacity: u32,
    pub backoff_multiplier: f64,
    pub max_backoff_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolConfig {
    pub primary_symbols: Vec<String>,
    pub watchlist_symbols: Vec<String>,
    pub crypto_symbols: Vec<String>,
    pub max_symbols_per_batch: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub batch_size: u32,
    pub buffer_timeout_seconds: u64,
    pub max_memory_buffer_mb: u32,
    pub enable_compression: bool,
}

impl Default for DataIngestionConfig {
    fn default() -> Self {
        Self {
            yahoo_finance: YahooFinanceConfig::default(),
            reddit: RedditConfig::default(),
            news: NewsConfig::default(),
            scheduler: SchedulerConfig::default(),
            rate_limits: RateLimitConfig::default(),
            symbols: SymbolConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

impl Default for YahooFinanceConfig {
    fn default() -> Self {
        Self {
            base_url: "https://query1.finance.yahoo.com".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            retry_delay_seconds: 2,
            user_agent: "QuantumTrade-AI/1.0".to_string(),
            rate_limit_per_minute: 60,
        }
    }
}

impl Default for RedditConfig {
    fn default() -> Self {
        Self {
            client_id: std::env::var("REDDIT_CLIENT_ID").unwrap_or_default(),
            client_secret: std::env::var("REDDIT_CLIENT_SECRET").unwrap_or_default(),
            user_agent: "QuantumTrade-AI/1.0".to_string(),
            username: std::env::var("REDDIT_USERNAME").unwrap_or_default(),
            password: std::env::var("REDDIT_PASSWORD").unwrap_or_default(),
            base_url: "https://oauth.reddit.com".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            rate_limit_per_minute: 60,
            subreddits: vec![
                "wallstreetbets".to_string(),
                "stocks".to_string(),
                "investing".to_string(),
                "SecurityAnalysis".to_string(),
                "StockMarket".to_string(),
            ],
            min_score: 10,
            max_posts_per_request: 100,
        }
    }
}

impl Default for NewsConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("NEWS_API_KEY").unwrap_or_default(),
            base_url: "https://newsapi.org/v2".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            rate_limit_per_minute: 100,
            sources: vec![
                "bloomberg".to_string(),
                "reuters".to_string(),
                "financial-post".to_string(),
                "the-wall-street-journal".to_string(),
            ],
            languages: vec!["en".to_string()],
            max_articles_per_request: 100,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            market_data_cron: "0 */5 * * * *".to_string(),      // Every 5 minutes
            reddit_sentiment_cron: "0 */15 * * * *".to_string(), // Every 15 minutes
            news_sentiment_cron: "0 */30 * * * *".to_string(),   // Every 30 minutes
            cleanup_cron: "0 0 2 * * *".to_string(),            // Daily at 2 AM
            enable_market_hours_only: true,
            timezone: "America/New_York".to_string(),
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 100,
            burst_capacity: 10,
            backoff_multiplier: 2.0,
            max_backoff_seconds: 60,
        }
    }
}

impl Default for SymbolConfig {
    fn default() -> Self {
        Self {
            primary_symbols: vec![
                "AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string(),
                "AMZN".to_string(), "TSLA".to_string(), "META".to_string(),
                "NVDA".to_string(), "NFLX".to_string(), "DIS".to_string(),
                "SPY".to_string(), "QQQ".to_string(), "IWM".to_string(),
            ],
            watchlist_symbols: vec![
                "AMD".to_string(), "INTC".to_string(), "CRM".to_string(),
                "ORCL".to_string(), "ADBE".to_string(), "PYPL".to_string(),
            ],
            crypto_symbols: vec![
                "BTC-USD".to_string(), "ETH-USD".to_string(), "ADA-USD".to_string(),
            ],
            max_symbols_per_batch: 50,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            buffer_timeout_seconds: 60,
            max_memory_buffer_mb: 100,
            enable_compression: true,
        }
    }
}

impl DataIngestionConfig {
    /// Load configuration from a TOML file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to read config file {}: {}", path, e)
            })?;
        
        let config: Self = toml::parse(&content)
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to parse config file {}: {}", path, e)
            })?;
        
        config.validate()?;
        Ok(config)
    }
    
    /// Load configuration from environment variables and defaults
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        
        // Override with environment variables if present
        if let Ok(val) = std::env::var("YAHOO_FINANCE_RATE_LIMIT") {
            config.yahoo_finance.rate_limit_per_minute = val.parse()
                .map_err(|e| QuantumTradeError::Configuration {
                    message: format!("Invalid YAHOO_FINANCE_RATE_LIMIT: {}", e)
                })?;
        }
        
        if let Ok(val) = std::env::var("REDDIT_RATE_LIMIT") {
            config.reddit.rate_limit_per_minute = val.parse()
                .map_err(|e| QuantumTradeError::Configuration {
                    message: format!("Invalid REDDIT_RATE_LIMIT: {}", e)
                })?;
        }
        
        if let Ok(val) = std::env::var("NEWS_RATE_LIMIT") {
            config.news.rate_limit_per_minute = val.parse()
                .map_err(|e| QuantumTradeError::Configuration {
                    message: format!("Invalid NEWS_RATE_LIMIT: {}", e)
                })?;
        }
        
        if let Ok(symbols) = std::env::var("PRIMARY_SYMBOLS") {
            config.symbols.primary_symbols = symbols
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .collect();
        }
        
        config.validate()?;
        Ok(config)
    }
    
    /// Validate configuration values
    pub fn validate(&self) -> Result<()> {
        // Validate Reddit credentials
        if self.reddit.client_id.is_empty() || self.reddit.client_secret.is_empty() {
            return Err(QuantumTradeError::Configuration {
                message: "Reddit client_id and client_secret are required".to_string()
            });
        }
        
        // Validate News API key
        if self.news.api_key.is_empty() {
            return Err(QuantumTradeError::Configuration {
                message: "News API key is required".to_string()
            });
        }
        
        // Validate rate limits
        if self.rate_limits.requests_per_minute == 0 {
            return Err(QuantumTradeError::Configuration {
                message: "Rate limit requests_per_minute must be greater than 0".to_string()
            });
        }
        
        // Validate symbols
        if self.symbols.primary_symbols.is_empty() {
            return Err(QuantumTradeError::Configuration {
                message: "At least one primary symbol must be configured".to_string()
            });
        }
        
        // Validate cron expressions (basic validation)
        for (name, cron) in [
            ("market_data_cron", &self.scheduler.market_data_cron),
            ("reddit_sentiment_cron", &self.scheduler.reddit_sentiment_cron),
            ("news_sentiment_cron", &self.scheduler.news_sentiment_cron),
            ("cleanup_cron", &self.scheduler.cleanup_cron),
        ] {
            if cron.split_whitespace().count() != 6 {
                return Err(QuantumTradeError::Configuration {
                    message: format!("Invalid cron expression for {}: {}", name, cron)
                });
            }
        }
        
        Ok(())
    }
    
    /// Get all symbols (primary + watchlist + crypto)
    pub fn get_all_symbols(&self) -> Vec<String> {
        let mut symbols = Vec::new();
        symbols.extend(self.symbols.primary_symbols.clone());
        symbols.extend(self.symbols.watchlist_symbols.clone());
        symbols.extend(self.symbols.crypto_symbols.clone());
        symbols.sort();
        symbols.dedup();
        symbols
    }
    
    /// Get timeout duration for external API calls
    pub fn get_timeout(&self, service: &str) -> Duration {
        let seconds = match service {
            "yahoo" => self.yahoo_finance.timeout_seconds,
            "reddit" => self.reddit.timeout_seconds,
            "news" => self.news.timeout_seconds,
            _ => 30,
        };
        Duration::from_secs(seconds)
    }
    
    /// Check if we should collect data during current market hours
    pub fn should_collect_now(&self) -> bool {
        if !self.scheduler.enable_market_hours_only {
            return true;
        }
        
        // TODO: Implement proper market hours checking based on timezone
        // For now, always return true
        true
    }
}

// Helper function for parsing TOML (since we can't use the `config` crate directly)
fn parse_toml(content: &str) -> Result<DataIngestionConfig> {
    toml::from_str(content)
        .map_err(|e| QuantumTradeError::Configuration {
            message: format!("Failed to parse TOML: {}", e)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = DataIngestionConfig::default();
        assert!(!config.symbols.primary_symbols.is_empty());
        assert!(config.yahoo_finance.timeout_seconds > 0);
        assert!(config.rate_limits.requests_per_minute > 0);
    }
    
    #[test]
    fn test_get_all_symbols() {
        let config = DataIngestionConfig::default();
        let symbols = config.get_all_symbols();
        
        // Should contain primary symbols
        assert!(symbols.contains(&"AAPL".to_string()));
        assert!(symbols.contains(&"SPY".to_string()));
        
        // Should be sorted and deduplicated
        let mut sorted_symbols = symbols.clone();
        sorted_symbols.sort();
        assert_eq!(symbols, sorted_symbols);
    }
    
    #[test]
    fn test_timeout_calculation() {
        let config = DataIngestionConfig::default();
        
        assert_eq!(config.get_timeout("yahoo"), Duration::from_secs(30));
        assert_eq!(config.get_timeout("reddit"), Duration::from_secs(30));
        assert_eq!(config.get_timeout("unknown"), Duration::from_secs(30));
    }
}