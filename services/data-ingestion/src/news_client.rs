// services/data-ingestion/src/news_client.rs
use chrono::{DateTime, Utc, TimeZone};
use regex::Regex;
use reqwest::{Client, header::{HeaderMap, HeaderValue, USER_AGENT}};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, error, info, warn};
use htmlescape::decode_html;

use crate::config::NewsConfig;
use shared_types::SentimentData;
use shared_utils::{QuantumTradeError, Result};

#[derive(Debug, Clone)]
pub struct NewsClient {
    client: Client,
    config: NewsConfig,
    symbol_regex: Regex,
    company_names: HashMap<String, String>, // symbol -> company name mapping
}

#[derive(Debug, Serialize, Deserialize)]
struct NewsApiResponse {
    status: String,
    #[serde(rename = "totalResults")]
    total_results: u32,
    articles: Vec<NewsArticle>,
}

#[derive(Debug, Serialize, Deserialize)]
struct NewsArticle {
    source: NewsSource,
    author: Option<String>,
    title: String,
    description: Option<String>,
    url: String,
    #[serde(rename = "urlToImage")]
    url_to_image: Option<String>,
    #[serde(rename = "publishedAt")]
    published_at: String,
    content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct NewsSource {
    id: Option<String>,
    name: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct NewsApiError {
    status: String,
    code: String,
    message: String,
}

#[derive(Debug)]
struct NewsSentimentAnalysis {
    score: f32,      // -1.0 to 1.0 (negative to positive)
    confidence: f32, // 0.0 to 1.0
    relevance: f32,  // 0.0 to 1.0 (how relevant to the symbol)
}

impl NewsClient {
    /// Create a new News client
    pub fn new(config: &NewsConfig) -> Result<Self> {
        info!("Initializing News client");
        
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str("QuantumTrade-AI/1.0")
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
        
        // Compile regex for detecting stock symbols and financial terms
        let symbol_regex = Regex::new(r"\b([A-Z]{1,5})\b|\$([A-Z]{1,5})")
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Failed to compile symbol regex: {}", e)
            })?;
        
        // Initialize company name mappings for better symbol detection
        let company_names = Self::init_company_names();
        
        Ok(Self {
            client,
            config: config.clone(),
            symbol_regex,
            company_names,
        })
    }
    
    /// Get sentiment data for a specific symbol from news articles
    pub async fn get_sentiment_data(&self, symbol: &str, limit: u32) -> Result<Vec<SentimentData>> {
        debug!("Fetching news sentiment data for symbol: {} (limit: {})", symbol, limit);
        
        let mut all_sentiment = Vec::new();
        
        // Search by symbol
        match self.search_news_by_symbol(symbol, limit / 2).await {
            Ok(mut sentiment) => {
                debug!("Found {} articles mentioning symbol {}", sentiment.len(), symbol);
                all_sentiment.append(&mut sentiment);
            }
            Err(e) => {
                warn!("Failed to search news for symbol {}: {}", symbol, e);
            }
        }
        
        // Search by company name if we have it
        if let Some(company_name) = self.company_names.get(symbol) {
            match self.search_news_by_company(company_name, symbol, limit / 2).await {
                Ok(mut sentiment) => {
                    debug!("Found {} articles mentioning company {}", sentiment.len(), company_name);
                    all_sentiment.append(&mut sentiment);
                }
                Err(e) => {
                    warn!("Failed to search news for company {}: {}", company_name, e);
                }
            }
        }
        
        // Remove duplicates based on URL
        all_sentiment.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        all_sentiment.dedup_by(|a, b| {
            // Extract URL from raw_data for deduplication
            let a_url = extract_url_from_raw_data(&a.raw_data);
            let b_url = extract_url_from_raw_data(&b.raw_data);
            a_url == b_url
        });
        
        // Limit results
        all_sentiment.truncate(limit as usize);
        
        info!("Retrieved {} sentiment data points for {} from news", all_sentiment.len(), symbol);
        Ok(all_sentiment)
    }
    
    /// Get general market sentiment from financial news
    pub async fn get_general_market_sentiment(&self, limit: u32) -> Result<Vec<SentimentData>> {
        debug!("Fetching general market sentiment from news (limit: {})", limit);
        
        let market_keywords = vec![
            "stock market",
            "financial markets", 
            "Wall Street",
            "trading",
            "market outlook",
            "economic indicators",
            "Federal Reserve",
            "interest rates",
            "inflation",
            "GDP"
        ];
        
        let mut all_sentiment = Vec::new();
        
        for keyword in &market_keywords {
            match self.search_news_general(keyword, limit / market_keywords.len() as u32).await {
                Ok(mut sentiment) => {
                    all_sentiment.append(&mut sentiment);
                }
                Err(e) => {
                    warn!("Failed to search news for keyword {}: {}", keyword, e);
                }
            }
            
            // Rate limiting between searches
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
        
        all_sentiment.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        all_sentiment.truncate(limit as usize);
        
        Ok(all_sentiment)
    }
    
    /// Search news specifically for a symbol
    async fn search_news_by_symbol(&self, symbol: &str, limit: u32) -> Result<Vec<SentimentData>> {
        let query = format!("{} stock OR ${}", symbol, symbol);
        let articles = self.search_news(&query, limit).await?;
        self.analyze_articles_for_symbol(&articles, symbol).await
    }
    
    /// Search news by company name
    async fn search_news_by_company(&self, company_name: &str, symbol: &str, limit: u32) -> Result<Vec<SentimentData>> {
        let query = format!("{} stock OR {} earnings OR {} financial", company_name, company_name, company_name);
        let articles = self.search_news(&query, limit).await?;
        self.analyze_articles_for_symbol(&articles, symbol).await
    }
    
    /// Search general financial news
    async fn search_news_general(&self, keyword: &str, limit: u32) -> Result<Vec<SentimentData>> {
        let articles = self.search_news(keyword, limit).await?;
        self.analyze_articles_for_market_sentiment(&articles).await
    }
    
    /// Search news using NewsAPI
    async fn search_news(&self, query: &str, limit: u32) -> Result<Vec<NewsArticle>> {
        let url = format!("{}/everything", self.config.base_url);
        
        // Calculate date range (last 7 days)
        let to_date = Utc::now();
        let from_date = to_date - chrono::Duration::days(7);
        
        let mut query_params = vec![
            ("q", query),
            ("apiKey", &self.config.api_key),
            ("language", "en"),
            ("sortBy", "publishedAt"),
            ("pageSize", &limit.min(100).to_string()), // NewsAPI max is 100
            ("from", &from_date.format("%Y-%m-%d").to_string()),
            ("to", &to_date.format("%Y-%m-%d").to_string()),
        ];
        
        // Add sources if configured
        if !self.config.sources.is_empty() {
            let sources_str = self.config.sources.join(",");
            query_params.push(("sources", &sources_str));
        }
        
        debug!("Searching news with query: {}", query);
        
        let response = self.client
            .get(&url)
            .query(&query_params)
            .send()
            .await
            .map_err(|e| QuantumTradeError::QueryExecution {
                message: format!("News API request failed: {}", e)
            })?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(QuantumTradeError::QueryExecution {
                message: format!("News API error {}: {}", response.status(), error_text)
            });
        }
        
        let news_response: NewsApiResponse = response.json().await
            .map_err(|e| QuantumTradeError::DataValidation {
                message: format!("Failed to parse News API response: {}", e)
            })?;
        
        if news_response.status != "ok" {
            return Err(QuantumTradeError::QueryExecution {
                message: format!("News API returned status: {}", news_response.status)
            });
        }
        
        debug!("Retrieved {} articles from News API", news_response.articles.len());
        Ok(news_response.articles)
    }
    
    /// Analyze articles for symbol-specific sentiment
    async fn analyze_articles_for_symbol(&self, articles: &[NewsArticle], symbol: &str) -> Result<Vec<SentimentData>> {
        let mut sentiment_data = Vec::new();
        
        for article in articles {
            let full_text = self.extract_article_text(article);
            
            // Check if article is relevant to the symbol
            if !self.is_article_relevant_to_symbol(&full_text, symbol) {
                continue;
            }
            
            let sentiment = self.analyze_text_sentiment(&full_text, Some(symbol));
            
            // Parse publication date
            let timestamp = self.parse_published_date(&article.published_at)
                .unwrap_or_else(Utc::now);
            
            // Create raw data JSON
            let raw_data = serde_json::json!({
                "title": article.title,
                "description": article.description,
                "author": article.author,
                "source": article.source.name,
                "url": article.url,
                "published_at": article.published_at,
                "content_preview": full_text.chars().take(500).collect::<String>(),
                "relevance_score": sentiment.relevance
            }).to_string();
            
            sentiment_data.push(SentimentData {
                symbol: symbol.to_uppercase(),
                timestamp,
                source: "news".to_string(),
                sentiment_score: sentiment.score,
                confidence: sentiment.confidence * sentiment.relevance, // Weight by relevance
                mention_count: self.count_symbol_mentions(&full_text, symbol),
                raw_data,
            });
        }
        
        Ok(sentiment_data)
    }
    
    /// Analyze articles for general market sentiment
    async fn analyze_articles_for_market_sentiment(&self, articles: &[NewsArticle]) -> Result<Vec<SentimentData>> {
        let mut sentiment_data = Vec::new();
        
        for article in articles {
            let full_text = self.extract_article_text(article);
            let sentiment = self.analyze_text_sentiment(&full_text, None);
            
            // Extract mentioned symbols
            let mentioned_symbols = self.extract_symbols_from_text(&full_text);
            
            let timestamp = self.parse_published_date(&article.published_at)
                .unwrap_or_else(Utc::now);
            
            let raw_data = serde_json::json!({
                "title": article.title,
                "description": article.description,
                "author": article.author,
                "source": article.source.name,
                "url": article.url,
                "published_at": article.published_at,
                "mentioned_symbols": mentioned_symbols,
                "content_preview": full_text.chars().take(500).collect::<String>(),
            }).to_string();
            
            // Create sentiment data for each mentioned symbol, or general market
            if mentioned_symbols.is_empty() {
                sentiment_data.push(SentimentData {
                    symbol: "MARKET".to_string(),
                    timestamp,
                    source: "news".to_string(),
                    sentiment_score: sentiment.score,
                    confidence: sentiment.confidence,
                    mention_count: 1,
                    raw_data: raw_data.clone(),
                });
            } else {
                for symbol in mentioned_symbols {
                    let relevance = self.calculate_symbol_relevance(&full_text, &symbol);
                    sentiment_data.push(SentimentData {
                        symbol,
                        timestamp,
                        source: "news".to_string(),
                        sentiment_score: sentiment.score,
                        confidence: sentiment.confidence * relevance,
                        mention_count: 1,
                        raw_data: raw_data.clone(),
                    });
                }
            }
        }
        
        Ok(sentiment_data)
    }
    
    /// Extract full text from article (title + description + content)
    fn extract_article_text(&self, article: &NewsArticle) -> String {
        let mut text = article.title.clone();
        
        if let Some(ref description) = article.description {
            text.push(' ');
            text.push_str(description);
        }
        
        if let Some(ref content) = article.content {
            text.push(' ');
            text.push_str(content);
        }
        
        // Decode HTML entities
        decode_html(&text).unwrap_or(text)
    }
    
    /// Check if article is relevant to a specific symbol
    fn is_article_relevant_to_symbol(&self, text: &str, symbol: &str) -> bool {
        let text_upper = text.to_uppercase();
        let symbol_upper = symbol.to_uppercase();
        
        // Check for direct symbol mentions
        if text_upper.contains(&format!("${}", symbol_upper)) ||
           text_upper.contains(&format!(" {} ", symbol_upper)) {
            return true;
        }
        
        // Check for company name mentions
        if let Some(company_name) = self.company_names.get(symbol) {
            if text_upper.contains(&company_name.to_uppercase()) {
                return true;
            }
        }
        
        false
    }
    
    /// Count how many times a symbol is mentioned in text
    fn count_symbol_mentions(&self, text: &str, symbol: &str) -> u32 {
        let text_upper = text.to_uppercase();
        let symbol_upper = symbol.to_uppercase();
        
        let mut count = 0;
        
        // Count $SYMBOL mentions
        count += text_upper.matches(&format!("${}", symbol_upper)).count() as u32;
        
        // Count standalone symbol mentions (with word boundaries)
        let words: Vec<&str> = text_upper.split_whitespace().collect();
        count += words.iter().filter(|&&word| word == symbol_upper).count() as u32;
        
        // Count company name mentions
        if let Some(company_name) = self.company_names.get(symbol) {
            count += text_upper.matches(&company_name.to_uppercase()).count() as u32;
        }
        
        count
    }
    
    /// Calculate how relevant an article is to a specific symbol
    fn calculate_symbol_relevance(&self, text: &str, symbol: &str) -> f32 {
        let mention_count = self.count_symbol_mentions(text, symbol);
        let text_length = text.split_whitespace().count();
        
        if text_length == 0 {
            return 0.0;
        }
        
        // Base relevance on mention frequency
        let mention_ratio = mention_count as f32 / text_length as f32;
        
        // Boost relevance if symbol appears in title
        let title_boost = if text.lines().next()
            .map(|line| line.to_uppercase().contains(&symbol.to_uppercase()))
            .unwrap_or(false) {
            0.5
        } else {
            0.0
        };
        
        (mention_ratio * 100.0 + title_boost).min(1.0)
    }
    
    /// Extract stock symbols from text
    fn extract_symbols_from_text(&self, text: &str) -> Vec<String> {
        let mut symbols = Vec::new();
        
        for cap in self.symbol_regex.captures_iter(text) {
            if let Some(symbol) = cap.get(1).or_else(|| cap.get(2)) {
                let symbol_str = symbol.as_str().to_uppercase();
                if self.is_likely_stock_symbol(&symbol_str) {
                    symbols.push(symbol_str);
                }
            }
        }
        
        symbols.sort();
        symbols.dedup();
        symbols.truncate(10); // Limit to avoid too many symbols per article
        symbols
    }
    
    /// Check if a string is likely a stock symbol
    fn is_likely_stock_symbol(&self, symbol: &str) -> bool {
        if symbol.len() < 1 || symbol.len() > 5 {
            return false;
        }
        
        // Check against known company names
        if self.company_names.contains_key(symbol) {
            return true;
        }
        
        // Filter out common false positives
        const COMMON_WORDS: &[&str] = &[
            "THE", "AND", "OR", "BUT", "NOT", "FOR", "TO", "OF", "IN", "ON", "AT", "BY",
            "ARE", "WAS", "IS", "BE", "AS", "IT", "SO", "IF", "MY", "ME", "WE", "US",
            "ALL", "ANY", "CAN", "GET", "GOT", "HAS", "HAD", "HIM", "HER", "HIS", "HOW",
            "ITS", "NEW", "NOW", "OLD", "ONE", "OUR", "OUT", "SEE", "TWO", "WAY", "WHO",
            "CEO", "CFO", "CTO", "ETF", "IPO", "SEC", "FDA", "API", "AI", "VR", "AR"
        ];
        
        !COMMON_WORDS.contains(&symbol)
    }
    
    /// Perform sentiment analysis on text
    fn analyze_text_sentiment(&self, text: &str, symbol: Option<&str>) -> NewsSentimentAnalysis {
        // Enhanced sentiment analysis with financial context
        let positive_words = [
            "growth", "profit", "gain", "rise", "increase", "strong", "bullish", "buy",
            "upgrade", "outperform", "beat", "exceed", "positive", "good", "excellent",
            "success", "expansion", "revenue", "earnings", "dividend", "rally", "surge",
            "breakthrough", "innovation", "partnership", "acquisition", "merger", "deal"
        ];
        
        let negative_words = [
            "loss", "decline", "fall", "decrease", "weak", "bearish", "sell", "downgrade",
            "underperform", "miss", "below", "negative", "bad", "poor", "failure",
            "contraction", "layoffs", "bankruptcy", "lawsuit", "investigation", "fraud",
            "scandal", "crash", "plunge", "warning", "concern", "risk", "threat"
        ];
        
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut total_weight = 0.0;
        
        for (i, word) in words.iter().enumerate() {
            // Give more weight to words near the symbol mention
            let weight = if let Some(sym) = symbol {
                let distance_to_symbol = words.iter()
                    .enumerate()
                    .filter(|(_, &w)| w.to_uppercase().contains(&sym.to_uppercase()))
                    .map(|(j, _)| (i as i32 - j as i32).abs())
                    .min()
                    .unwrap_or(100);
                
                if distance_to_symbol <= 5 {
                    2.0 // Double weight for words close to symbol
                } else if distance_to_symbol <= 20 {
                    1.5 // 1.5x weight for words somewhat close
                } else {
                    1.0 // Normal weight
                }
            } else {
                1.0
            };
            
            if positive_words.iter().any(|&pos| word.contains(pos)) {
                positive_count += weight;
            }
            if negative_words.iter().any(|&neg| word.contains(neg)) {
                negative_count += weight;
            }
            total_weight += weight;
        }
        
        let total_sentiment_weight = positive_count + negative_count;
        
        let score = if total_sentiment_weight > 0.0 {
            (positive_count - negative_count) / total_sentiment_weight
        } else {
            0.0
        };
        
        let confidence = if total_weight > 0.0 {
            (total_sentiment_weight / total_weight).min(1.0) * 0.8 // News is generally more reliable than social media
        } else {
            0.0
        };
        
        let relevance = if let Some(sym) = symbol {
            self.calculate_symbol_relevance(text, sym)
        } else {
            1.0
        };
        
        NewsSentimentAnalysis {
            score: score.max(-1.0).min(1.0),
            confidence,
            relevance,
        }
    }
    
    /// Parse published date from NewsAPI format
    fn parse_published_date(&self, date_str: &str) -> Option<DateTime<Utc>> {
        // NewsAPI returns ISO 8601 format: "2023-01-27T10:30:00Z"
        DateTime::parse_from_rfc3339(date_str)
            .map(|dt| dt.with_timezone(&Utc))
            .ok()
    }
    
    /// Initialize mapping of stock symbols to company names
    fn init_company_names() -> HashMap<String, String> {
        let mut map = HashMap::new();
        
        // Major tech companies
        map.insert("AAPL".to_string(), "Apple".to_string());
        map.insert("GOOGL".to_string(), "Google".to_string());
        map.insert("GOOG".to_string(), "Alphabet".to_string());
        map.insert("MSFT".to_string(), "Microsoft".to_string());
        map.insert("AMZN".to_string(), "Amazon".to_string());
        map.insert("META".to_string(), "Meta".to_string());
        map.insert("TSLA".to_string(), "Tesla".to_string());
        map.insert("NVDA".to_string(), "Nvidia".to_string());
        map.insert("NFLX".to_string(), "Netflix".to_string());
        map.insert("ORCL".to_string(), "Oracle".to_string());
        map.insert("CRM".to_string(), "Salesforce".to_string());
        map.insert("ADBE".to_string(), "Adobe".to_string());
        
        // Financial institutions
        map.insert("JPM".to_string(), "JPMorgan".to_string());
        map.insert("BAC".to_string(), "Bank of America".to_string());
        map.insert("WFC".to_string(), "Wells Fargo".to_string());
        map.insert("GS".to_string(), "Goldman Sachs".to_string());
        map.insert("MS".to_string(), "Morgan Stanley".to_string());
        
        // Other major companies
        map.insert("JNJ".to_string(), "Johnson & Johnson".to_string());
        map.insert("PG".to_string(), "Procter & Gamble".to_string());
        map.insert("KO".to_string(), "Coca-Cola".to_string());
        map.insert("DIS".to_string(), "Disney".to_string());
        map.insert("NKE".to_string(), "Nike".to_string());
        map.insert("MCD".to_string(), "McDonald's".to_string());
        
        map
    }
}

/// Helper function to extract URL from raw data JSON
fn extract_url_from_raw_data(raw_data: &str) -> Option<String> {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(raw_data) {
        json.get("url").and_then(|v| v.as_str()).map(|s| s.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NewsConfig;
    
    #[test]
    fn test_symbol_extraction() {
        let config = NewsConfig::default();
        let client = NewsClient::new(&config).unwrap();
        
        let text = "Apple (AAPL) reported strong earnings, while Tesla stock (TSLA) declined.";
        let symbols = client.extract_symbols_from_text(text);
        
        assert!(symbols.contains(&"AAPL".to_string()));
        assert!(symbols.contains(&"TSLA".to_string()));
    }
    
    #[test]
    fn test_sentiment_analysis() {
        let config = NewsConfig::default();
        let client = NewsClient::new(&config).unwrap();
        
        let positive_text = "Apple reported excellent earnings growth and strong revenue increase.";
        let sentiment = client.analyze_text_sentiment(positive_text, Some("AAPL"));
        assert!(sentiment.score > 0.0);
        
        let negative_text = "Tesla faces lawsuit and investigation over safety concerns.";
        let sentiment = client.analyze_text_sentiment(negative_text, Some("TSLA"));
        assert!(sentiment.score < 0.0);
    }
    
    #[test]
    fn test_symbol_relevance() {
        let config = NewsConfig::default();
        let client = NewsClient::new(&config).unwrap();
        
        let relevant_text = "Apple CEO announces new iPhone. AAPL stock surges on the news.";
        let relevance = client.calculate_symbol_relevance(relevant_text, "AAPL");
        assert!(relevance > 0.5);
        
        let irrelevant_text = "The weather is nice today. Many people are happy.";
        let relevance = client.calculate_symbol_relevance(irrelevant_text, "AAPL");
        assert!(relevance < 0.1);
    }
    
    #[test]
    fn test_mention_counting() {
        let config = NewsConfig::default();
        let client = NewsClient::new(&config).unwrap();
        
        let text = "Apple (AAPL) and $AAPL stock both mentioned. Apple again.";
        let count = client.count_symbol_mentions(text, "AAPL");
        assert!(count >= 3); // Should find multiple mentions
    }
    
    #[test]
    fn test_date_parsing() {
        let config = NewsConfig::default();
        let client = NewsClient::new(&config).unwrap();
        
        let date_str = "2023-01-27T10:30:00Z";
        let parsed = client.parse_published_date(date_str);
        assert!(parsed.is_some());
        
        let invalid_date = "invalid-date";
        let parsed = client.parse_published_date(invalid_date);
        assert!(parsed.is_none());
    }
}