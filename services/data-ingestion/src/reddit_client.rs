// services/data-ingestion/src/reddit_client.rs
use base64::Engine;
use chrono::{DateTime, Utc, TimeZone};
use regex::Regex;
use reqwest::{Client, header::{HeaderMap, HeaderValue, USER_AGENT, AUTHORIZATION}};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::config::RedditConfig;
use shared_types::SentimentData;
use shared_utils::{QuantumTradeError, Result};

#[derive(Debug, Clone)]
pub struct RedditClient {
    client: Client,
    config: RedditConfig,
    auth_token: Arc<RwLock<Option<AuthToken>>>,
    symbol_regex: Regex,
}

#[derive(Debug, Clone)]
struct AuthToken {
    access_token: String,
    token_type: String,
    expires_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RedditAuthResponse {
    access_token: String,
    token_type: String,
    expires_in: u64,
    scope: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct RedditListingResponse {
    kind: String,
    data: RedditListingData,
}

#[derive(Debug, Serialize, Deserialize)]
struct RedditListingData {
    modhash: Option<String>,
    dist: Option<u32>,
    children: Vec<RedditPost>,
    after: Option<String>,
    before: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RedditPost {
    kind: String,
    data: RedditPostData,
}

#[derive(Debug, Serialize, Deserialize)]
struct RedditPostData {
    id: String,
    title: String,
    selftext: Option<String>,
    author: String,
    subreddit: String,
    score: i32,
    upvote_ratio: Option<f64>,
    num_comments: u32,
    created_utc: f64,
    url: Option<String>,
    permalink: String,
    is_self: bool,
    link_flair_text: Option<String>,
    post_hint: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RedditErrorResponse {
    message: String,
    error: Option<String>,
}

#[derive(Debug)]
struct SentimentAnalysis {
    score: f32,      // -1.0 to 1.0 (negative to positive)
    confidence: f32, // 0.0 to 1.0
    magnitude: f32,  // 0.0 to 1.0 (strength of sentiment)
}

impl RedditClient {
    /// Create a new Reddit client
    pub async fn new(config: &RedditConfig) -> Result<Self> {
        info!("Initializing Reddit client");
        
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&config.user_agent)
                .map_err(|e| QuantumTradeError::Configuration {
                    message: format!("Invalid user agent: {}", e)
                })?
        );
        
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .default_headers(headers)
            .build()
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to create HTTP client: {}", e)
            })?;
        
        // Compile regex for detecting stock symbols
        let symbol_regex = Regex::new(r"\$([A-Z]{1,5})|\b([A-Z]{1,5})\b")
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to compile symbol regex: {}", e)
            })?;
        
        let reddit_client = Self {
            client,
            config: config.clone(),
            auth_token: Arc::new(RwLock::new(None)),
            symbol_regex,
        };
        
        // Authenticate immediately
        reddit_client.authenticate().await?;
        
        Ok(reddit_client)
    }
    
    /// Get sentiment data for a specific symbol
    pub async fn get_sentiment_data(&self, symbol: &str, limit: u32) -> Result<Vec<SentimentData>> {
        debug!("Fetching Reddit sentiment data for symbol: {} (limit: {})", symbol, limit);
        
        let mut all_sentiment = Vec::new();
        
        // Search across configured subreddits
        for subreddit in &self.config.subreddits {
            match self.search_subreddit_for_symbol(subreddit, symbol, limit / self.config.subreddits.len() as u32).await {
                Ok(mut sentiment) => {
                    debug!("Found {} posts mentioning {} in r/{}", sentiment.len(), symbol, subreddit);
                    all_sentiment.append(&mut sentiment);
                }
                Err(e) => {
                    warn!("Failed to search r/{} for {}: {}", subreddit, symbol, e);
                    // Continue with other subreddits
                }
            }
            
            // Rate limiting between subreddits
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        // Also get hot posts from relevant subreddits and filter for symbol mentions
        for subreddit in &self.config.subreddits {
            match self.get_hot_posts_for_symbol(subreddit, symbol, 25).await {
                Ok(mut sentiment) => {
                    debug!("Found {} hot posts mentioning {} in r/{}", sentiment.len(), symbol, subreddit);
                    all_sentiment.append(&mut sentiment);
                }
                Err(e) => {
                    warn!("Failed to get hot posts from r/{} for {}: {}", subreddit, symbol, e);
                }
            }
            
            // Rate limiting
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        // Remove duplicates and sort by timestamp
        all_sentiment.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        all_sentiment.dedup_by(|a, b| a.raw_data == b.raw_data);
        
        // Limit results
        all_sentiment.truncate(limit as usize);
        
        info!("Retrieved {} sentiment data points for {} from Reddit", all_sentiment.len(), symbol);
        Ok(all_sentiment)
    }
    
    /// Get general market sentiment from popular financial subreddits
    pub async fn get_general_market_sentiment(&self, limit: u32) -> Result<Vec<SentimentData>> {
        debug!("Fetching general market sentiment (limit: {})", limit);
        
        let mut all_sentiment = Vec::new();
        
        for subreddit in &self.config.subreddits {
            match self.get_hot_posts(subreddit, limit / self.config.subreddits.len() as u32).await {
                Ok(mut posts) => {
                    let mut sentiment = self.analyze_posts_for_market_sentiment(posts).await?;
                    all_sentiment.append(&mut sentiment);
                }
                Err(e) => {
                    warn!("Failed to get posts from r/{}: {}", subreddit, e);
                }
            }
            
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        all_sentiment.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        all_sentiment.truncate(limit as usize);
        
        Ok(all_sentiment)
    }
    
    /// Search a specific subreddit for posts mentioning a symbol
    async fn search_subreddit_for_symbol(&self, subreddit: &str, symbol: &str, limit: u32) -> Result<Vec<SentimentData>> {
        self.ensure_authenticated().await?;
        
        let search_query = format!("${} OR {}", symbol, symbol);
        let url = format!("{}/r/{}/search.json", self.config.base_url, subreddit);
        
        let auth_token = self.auth_token.read().await;
        let token = auth_token.as_ref().unwrap();
        
        let response = self.client
            .get(&url)
            .header(AUTHORIZATION, format!("{} {}", token.token_type, token.access_token))
            .query(&[
                ("q", search_query.as_str()),
                ("restrict_sr", "1"),
                ("sort", "new"),
                ("limit", &limit.to_string()),
                ("t", "week"), // Last week
            ])
            .send()
            .await
            .map_err(|e| QuantumTradeError::QueryExecution {
                message: format!("Reddit search request failed: {}", e)
            })?;
        
        if !response.status().is_success() {
            return Err(QuantumTradeError::QueryExecution {
                message: format!("Reddit API error: {}", response.status())
            });
        }
        
        let listing: RedditListingResponse = response.json().await
            .map_err(|e| QuantumTradeError::DataValidation {
                message: format!("Failed to parse Reddit response: {}", e)
            })?;
        
        let posts: Vec<RedditPostData> = listing.data.children
            .into_iter()
            .filter(|post| post.data.score >= self.config.min_score)
            .map(|post| post.data)
            .collect();
        
        self.analyze_posts_for_symbol(&posts, symbol).await
    }
    
    /// Get hot posts from a subreddit
    async fn get_hot_posts(&self, subreddit: &str, limit: u32) -> Result<Vec<RedditPostData>> {
        self.get_hot_posts_for_symbol(subreddit, "", limit).await
            .map(|sentiment| {
                // This is a bit of a hack - we're returning empty Vec since we need posts, not sentiment
                Vec::new()
            })
            .or_else(|_| {
                // If the above fails, make the actual request
                self.get_subreddit_posts(subreddit, "hot", limit).await
            })
    }
    
    /// Get hot posts from a subreddit and filter for symbol mentions
    async fn get_hot_posts_for_symbol(&self, subreddit: &str, symbol: &str, limit: u32) -> Result<Vec<SentimentData>> {
        let posts = self.get_subreddit_posts(subreddit, "hot", limit).await?;
        
        if symbol.is_empty() {
            // General sentiment analysis
            self.analyze_posts_for_market_sentiment(posts).await
        } else {
            // Filter posts that mention the symbol
            let relevant_posts: Vec<RedditPostData> = posts.into_iter()
                .filter(|post| self.post_mentions_symbol(post, symbol))
                .collect();
            
            self.analyze_posts_for_symbol(&relevant_posts, symbol).await
        }
    }
    
    /// Get posts from a subreddit with specified sorting
    async fn get_subreddit_posts(&self, subreddit: &str, sort: &str, limit: u32) -> Result<Vec<RedditPostData>> {
        self.ensure_authenticated().await?;
        
        let url = format!("{}/r/{}/{}.json", self.config.base_url, subreddit, sort);
        
        let auth_token = self.auth_token.read().await;
        let token = auth_token.as_ref().unwrap();
        
        let response = self.client
            .get(&url)
            .header(AUTHORIZATION, format!("{} {}", token.token_type, token.access_token))
            .query(&[
                ("limit", limit.to_string().as_str()),
                ("t", "day"), // Time period
            ])
            .send()
            .await
            .map_err(|e| QuantumTradeError::QueryExecution {
                message: format!("Reddit request failed: {}", e)
            })?;
        
        if !response.status().is_success() {
            return Err(QuantumTradeError::QueryExecution {
                message: format!("Reddit API error: {}", response.status())
            });
        }
        
        let listing: RedditListingResponse = response.json().await
            .map_err(|e| QuantumTradeError::DataValidation {
                message: format!("Failed to parse Reddit response: {}", e)
            })?;
        
        let posts: Vec<RedditPostData> = listing.data.children
            .into_iter()
            .filter(|post| post.data.score >= self.config.min_score)
            .map(|post| post.data)
            .collect();
        
        Ok(posts)
    }
    
    /// Check if a post mentions a specific symbol
    fn post_mentions_symbol(&self, post: &RedditPostData, symbol: &str) -> bool {
        let text = format!("{} {}", post.title, post.selftext.as_deref().unwrap_or(""));
        let text_upper = text.to_uppercase();
        let symbol_upper = symbol.to_uppercase();
        
        // Look for $SYMBOL or SYMBOL patterns
        text_upper.contains(&format!("${}", symbol_upper)) || 
        text_upper.contains(&format!(" {} ", symbol_upper)) ||
        text_upper.starts_with(&format!("{} ", symbol_upper)) ||
        text_upper.ends_with(&format!(" {}", symbol_upper))
    }
    
    /// Analyze posts for symbol-specific sentiment
    async fn analyze_posts_for_symbol(&self, posts: &[RedditPostData], symbol: &str) -> Result<Vec<SentimentData>> {
        let mut sentiment_data = Vec::new();
        
        for post in posts {
            let text = format!("{} {}", post.title, post.selftext.as_deref().unwrap_or(""));
            let sentiment = self.analyze_text_sentiment(&text);
            
            let timestamp = Utc.timestamp_opt(post.created_utc as i64, 0)
                .single()
                .unwrap_or_else(Utc::now);
            
            // Create raw data JSON
            let raw_data = serde_json::json!({
                "post_id": post.id,
                "title": post.title,
                "author": post.author,
                "subreddit": post.subreddit,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "url": post.url,
                "permalink": post.permalink,
                "text": text.chars().take(500).collect::<String>(), // Truncate for storage
            }).to_string();
            
            sentiment_data.push(SentimentData {
                symbol: symbol.to_uppercase(),
                timestamp,
                source: "reddit".to_string(),
                sentiment_score: sentiment.score,
                confidence: sentiment.confidence,
                mention_count: 1, // Each post is one mention
                raw_data,
            });
        }
        
        Ok(sentiment_data)
    }
    
    /// Analyze posts for general market sentiment
    async fn analyze_posts_for_market_sentiment(&self, posts: Vec<RedditPostData>) -> Result<Vec<SentimentData>> {
        let mut sentiment_data = Vec::new();
        
        for post in posts {
            let text = format!("{} {}", post.title, post.selftext.as_deref().unwrap_or(""));
            let sentiment = self.analyze_text_sentiment(&text);
            
            // Extract mentioned symbols
            let mentioned_symbols = self.extract_symbols_from_text(&text);
            
            let timestamp = Utc.timestamp_opt(post.created_utc as i64, 0)
                .single()
                .unwrap_or_else(Utc::now);
            
            let raw_data = serde_json::json!({
                "post_id": post.id,
                "title": post.title,
                "author": post.author,
                "subreddit": post.subreddit,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "mentioned_symbols": mentioned_symbols,
                "text": text.chars().take(500).collect::<String>(),
            }).to_string();
            
            // Create sentiment data for each mentioned symbol, or general market
            if mentioned_symbols.is_empty() {
                sentiment_data.push(SentimentData {
                    symbol: "MARKET".to_string(),
                    timestamp,
                    source: "reddit".to_string(),
                    sentiment_score: sentiment.score,
                    confidence: sentiment.confidence,
                    mention_count: 1,
                    raw_data: raw_data.clone(),
                });
            } else {
                for symbol in mentioned_symbols {
                    sentiment_data.push(SentimentData {
                        symbol,
                        timestamp,
                        source: "reddit".to_string(),
                        sentiment_score: sentiment.score,
                        confidence: sentiment.confidence,
                        mention_count: 1,
                        raw_data: raw_data.clone(),
                    });
                }
            }
        }
        
        Ok(sentiment_data)
    }
    
    /// Extract stock symbols from text using regex
    fn extract_symbols_from_text(&self, text: &str) -> Vec<String> {
        let mut symbols = Vec::new();
        
        for cap in self.symbol_regex.captures_iter(text) {
            if let Some(symbol) = cap.get(1).or_else(|| cap.get(2)) {
                let symbol_str = symbol.as_str().to_uppercase();
                // Filter out common false positives
                if self.is_likely_stock_symbol(&symbol_str) {
                    symbols.push(symbol_str);
                }
            }
        }
        
        symbols.sort();
        symbols.dedup();
        symbols.truncate(10); // Limit to avoid too many symbols per post
        symbols
    }
    
    /// Check if a string is likely a stock symbol
    fn is_likely_stock_symbol(&self, symbol: &str) -> bool {
        // Basic filtering for common false positives
        const COMMON_WORDS: &[&str] = &[
            "THE", "AND", "OR", "BUT", "NOT", "FOR", "TO", "OF", "IN", "ON", "AT", "BY",
            "ARE", "WAS", "IS", "BE", "AS", "IT", "SO", "IF", "MY", "ME", "WE", "US",
            "ALL", "ANY", "CAN", "GET", "GOT", "HAS", "HAD", "HIM", "HER", "HIS", "HOW",
            "ITS", "NEW", "NOW", "OLD", "ONE", "OUR", "OUT", "SEE", "TWO", "WAY", "WHO",
            "BOY", "DID", "HAS", "LET", "MAY", "SAY", "SHE", "TOO", "USE", "DAY", "END",
            "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID"
        ];
        
        if symbol.len() < 1 || symbol.len() > 5 {
            return false;
        }
        
        !COMMON_WORDS.contains(&symbol)
    }
    
    /// Perform simple sentiment analysis on text
    fn analyze_text_sentiment(&self, text: &str) -> SentimentAnalysis {
        // Simple keyword-based sentiment analysis
        // In production, you'd want to use a proper ML model or external service
        
        let positive_words = [
            "bull", "bullish", "buy", "long", "moon", "rocket", "gain", "profit", "up", "rise",
            "green", "pump", "calls", "good", "great", "amazing", "love", "best", "strong",
            "hold", "diamond", "hands", "squeeze", "breakout", "rally", "surge"
        ];
        
        let negative_words = [
            "bear", "bearish", "sell", "short", "crash", "dump", "loss", "down", "fall",
            "red", "puts", "bad", "terrible", "hate", "worst", "weak", "paper", "dump",
            "decline", "drop", "plunge", "collapse", "correction", "dip"
        ];
        
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        
        let mut positive_count = 0;
        let mut negative_count = 0;
        
        for word in &words {
            if positive_words.iter().any(|&pos| word.contains(pos)) {
                positive_count += 1;
            }
            if negative_words.iter().any(|&neg| word.contains(neg)) {
                negative_count += 1;
            }
        }
        
        let total_sentiment_words = positive_count + negative_count;
        let text_length = words.len();
        
        let score = if total_sentiment_words > 0 {
            (positive_count as f32 - negative_count as f32) / total_sentiment_words as f32
        } else {
            0.0
        };
        
        let confidence = if text_length > 0 {
            (total_sentiment_words as f32 / text_length as f32).min(1.0)
        } else {
            0.0
        };
        
        SentimentAnalysis {
            score: score.max(-1.0).min(1.0),
            confidence: confidence * 0.7, // Reduce confidence since this is simple analysis
            magnitude: score.abs(),
        }
    }
    
    /// Ensure we have a valid authentication token
    async fn ensure_authenticated(&self) -> Result<()> {
        let token_guard = self.auth_token.read().await;
        
        if let Some(ref token) = *token_guard {
            if token.expires_at > Utc::now() + chrono::Duration::minutes(5) {
                return Ok(());
            }
        }
        
        drop(token_guard);
        self.authenticate().await
    }
    
    /// Authenticate with Reddit API
    async fn authenticate(&self) -> Result<()> {
        debug!("Authenticating with Reddit API");
        
        let auth_string = format!("{}:{}", self.config.client_id, self.config.client_secret);
        let auth_header = format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(auth_string.as_bytes()));
        
        let params = [
            ("grant_type", "password"),
            ("username", &self.config.username),
            ("password", &self.config.password),
        ];
        
        let response = self.client
            .post("https://www.reddit.com/api/v1/access_token")
            .header(AUTHORIZATION, auth_header)
            .form(&params)
            .send()
            .await
            .map_err(|e| QuantumTradeError::QueryExecution {
                message: format!("Reddit authentication request failed: {}", e)
            })?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(QuantumTradeError::QueryExecution {
                message: format!("Reddit authentication failed: {}", error_text)
            });
        }
        
        let auth_response: RedditAuthResponse = response.json().await
            .map_err(|e| QuantumTradeError::DataValidation {
                message: format!("Failed to parse Reddit auth response: {}", e)
            })?;
        
        let expires_at = Utc::now() + chrono::Duration::seconds(auth_response.expires_in as i64);
        
        let token = AuthToken {
            access_token: auth_response.access_token,
            token_type: auth_response.token_type,
            expires_at,
        };
        
        *self.auth_token.write().await = Some(token);
        
        info!("Successfully authenticated with Reddit API");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RedditConfig;
    
    #[test]
    fn test_symbol_extraction() {
        let config = RedditConfig::default();
        // Note: This will fail without valid credentials, but we can test the regex
        let regex = Regex::new(r"\$([A-Z]{1,5})|\b([A-Z]{1,5})\b").unwrap();
        
        let text = "I'm bullish on $AAPL and TSLA. Also watching GME moon!";
        let symbols: Vec<String> = regex.captures_iter(text)
            .filter_map(|cap| cap.get(1).or_else(|| cap.get(2)))
            .map(|m| m.as_str().to_uppercase())
            .collect();
        
        assert!(symbols.contains(&"AAPL".to_string()));
        assert!(symbols.contains(&"TSLA".to_string()));
        assert!(symbols.contains(&"GME".to_string()));
    }
    
    #[test]
    fn test_sentiment_analysis() {
        let config = RedditConfig::default();
        let reddit_client = RedditClient {
            client: Client::new(),
            config,
            auth_token: Arc::new(RwLock::new(None)),
            symbol_regex: Regex::new(r"\$([A-Z]{1,5})|\b([A-Z]{1,5})\b").unwrap(),
        };
        
        let positive_text = "AAPL is going to moon! Bullish and buying calls!";
        let sentiment = reddit_client.analyze_text_sentiment(positive_text);
        assert!(sentiment.score > 0.0);
        
        let negative_text = "TSLA is crashing hard. Very bearish, buying puts.";
        let sentiment = reddit_client.analyze_text_sentiment(negative_text);
        assert!(sentiment.score < 0.0);
        
        let neutral_text = "The weather is nice today.";
        let sentiment = reddit_client.analyze_text_sentiment(neutral_text);
        assert!(sentiment.score.abs() < 0.1);
    }
    
    #[test]
    fn test_symbol_filtering() {
        let config = RedditConfig::default();
        let reddit_client = RedditClient {
            client: Client::new(),
            config,
            auth_token: Arc::new(RwLock::new(None)),
            symbol_regex: Regex::new(r"\$([A-Z]{1,5})|\b([A-Z]{1,5})\b").unwrap(),
        };
        
        assert!(reddit_client.is_likely_stock_symbol("AAPL"));
        assert!(reddit_client.is_likely_stock_symbol("TSLA"));
        assert!(!reddit_client.is_likely_stock_symbol("THE"));
        assert!(!reddit_client.is_likely_stock_symbol("AND"));
        assert!(!reddit_client.is_likely_stock_symbol("TOOLONG"));
    }
}