// services/data-ingestion/src/yahoo_client.rs
use chrono::{DateTime, Utc, TimeZone};
use reqwest::{Client, Response, header::{HeaderMap, HeaderValue, USER_AGENT}};
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};
use url::Url;

use crate::config::YahooFinanceConfig;
use shared_types::MarketData;
use shared_utils::{QuantumTradeError, Result};

#[derive(Debug, Clone)]
pub struct YahooFinanceClient {
    client: Client,
    config: YahooFinanceConfig,
    base_url: Url,
}

#[derive(Debug, Serialize, Deserialize)]
struct YahooQuoteResponse {
    chart: YahooChart,
}

#[derive(Debug, Serialize, Deserialize)]
struct YahooChart {
    result: Option<Vec<YahooResult>>,
    error: Option<YahooError>,
}

#[derive(Debug, Serialize, Deserialize)]
struct YahooResult {
    meta: YahooMeta,
    timestamp: Vec<i64>,
    indicators: YahooIndicators,
}

#[derive(Debug, Serialize, Deserialize)]
struct YahooMeta {
    symbol: String,
    #[serde(rename = "exchangeName")]
    exchange_name: Option<String>,
    #[serde(rename = "instrumentType")]
    instrument_type: Option<String>,
    #[serde(rename = "firstTradeDate")]
    first_trade_date: Option<i64>,
    #[serde(rename = "regularMarketTime")]
    regular_market_time: Option<i64>,
    gmtoffset: Option<i32>,
    timezone: Option<String>,
    #[serde(rename = "exchangeTimezoneName")]
    exchange_timezone_name: Option<String>,
    #[serde(rename = "regularMarketPrice")]
    regular_market_price: Option<f64>,
    #[serde(rename = "chartPreviousClose")]
    chart_previous_close: Option<f64>,
    #[serde(rename = "previousClose")]
    previous_close: Option<f64>,
    scale: Option<i32>,
    #[serde(rename = "priceHint")]
    price_hint: Option<i32>,
    #[serde(rename = "currentTradingPeriod")]
    current_trading_period: Option<serde_json::Value>,
    #[serde(rename = "tradingPeriods")]
    trading_periods: Option<Vec<Vec<serde_json::Value>>>,
    #[serde(rename = "dataGranularity")]
    data_granularity: Option<String>,
    range: Option<String>,
    #[serde(rename = "validRanges")]
    valid_ranges: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct YahooIndicators {
    quote: Vec<YahooQuote>,
    #[serde(rename = "adjclose")]
    adj_close: Option<Vec<YahooAdjClose>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct YahooQuote {
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<u64>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct YahooAdjClose {
    #[serde(rename = "adjclose")]
    adj_close: Vec<Option<f64>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct YahooError {
    code: String,
    description: String,
}

impl YahooFinanceClient {
    /// Create a new Yahoo Finance client
    pub fn new(config: &YahooFinanceConfig) -> Result<Self> {
        info!("Initializing Yahoo Finance client");
        
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
        
        let base_url = Url::parse(&config.base_url)
            .map_err(|e| QuantumTradeError::Configuration {
                message: format!("Invalid base URL: {}", e)
            })?;
        
        Ok(Self {
            client,
            config: config.clone(),
            base_url,
        })
    }
    
    /// Get historical market data for a symbol
    pub async fn get_historical_data(&self, symbol: &str, limit: u32) -> Result<Vec<MarketData>> {
        debug!("Fetching historical data for symbol: {} (limit: {})", symbol, limit);
        
        let url = self.build_quote_url(symbol, "1d", limit)?;
        
        let response = self.make_request(&url).await?;
        let yahoo_response: YahooQuoteResponse = response.json().await
            .map_err(|e| QuantumTradeError::DataValidation {
                message: format!("Failed to parse Yahoo Finance response: {}", e)
            })?;
        
        self.parse_quote_response(symbol, yahoo_response)
    }
    
    /// Get real-time quote for a symbol
    pub async fn get_realtime_quote(&self, symbol: &str) -> Result<Option<MarketData>> {
        debug!("Fetching real-time quote for symbol: {}", symbol);
        
        let url = self.build_quote_url(symbol, "1m", 1)?;
        
        let response = self.make_request(&url).await?;
        let yahoo_response: YahooQuoteResponse = response.json().await
            .map_err(|e| QuantumTradeError::DataValidation {
                message: format!("Failed to parse Yahoo Finance response: {}", e)
            })?;
        
        let mut data = self.parse_quote_response(symbol, yahoo_response)?;
        Ok(data.pop()) // Return the most recent data point
    }
    
    /// Get intraday data with specified interval
    pub async fn get_intraday_data(
        &self, 
        symbol: &str, 
        interval: &str, 
        limit: u32
    ) -> Result<Vec<MarketData>> {
        debug!("Fetching intraday data for symbol: {} (interval: {}, limit: {})", 
               symbol, interval, limit);
        
        let url = self.build_quote_url(symbol, interval, limit)?;
        
        let response = self.make_request(&url).await?;
        let yahoo_response: YahooQuoteResponse = response.json().await
            .map_err(|e| QuantumTradeError::DataValidation {
                message: format!("Failed to parse Yahoo Finance response: {}", e)
            })?;
        
        self.parse_quote_response(symbol, yahoo_response)
    }
    
    /// Get market data for multiple symbols (batch request)
    pub async fn get_multiple_quotes(&self, symbols: &[String]) -> Result<Vec<MarketData>> {
        debug!("Fetching quotes for {} symbols", symbols.len());
        
        let mut all_data = Vec::new();
        
        // Process symbols in smaller batches to avoid overwhelming the API
        const BATCH_SIZE: usize = 10;
        
        for batch in symbols.chunks(BATCH_SIZE) {
            for symbol in batch {
                match self.get_realtime_quote(symbol).await {
                    Ok(Some(data)) => all_data.push(data),
                    Ok(None) => warn!("No data available for symbol: {}", symbol),
                    Err(e) => {
                        warn!("Failed to get data for symbol {}: {}", symbol, e);
                        // Continue with other symbols rather than failing the entire batch
                    }
                }
                
                // Small delay between requests to respect rate limits
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
        
        info!("Successfully retrieved data for {} out of {} symbols", 
              all_data.len(), symbols.len());
        
        Ok(all_data)
    }
    
    /// Build the URL for quote requests
    fn build_quote_url(&self, symbol: &str, interval: &str, limit: u32) -> Result<Url> {
        let mut url = self.base_url.clone();
        url.set_path("/v8/finance/chart/");
        url.path_segments_mut()
            .map_err(|_| QuantumTradeError::Configuration {
                message: "Cannot modify URL path".to_string()
            })?
            .push(symbol);
        
        // Calculate time range based on interval and limit
        let (period1, period2) = self.calculate_time_range(interval, limit);
        
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("interval", interval);
            query.append_pair("period1", &period1.to_string());
            query.append_pair("period2", &period2.to_string());
            query.append_pair("includePrePost", "false");
            query.append_pair("events", "div,splits");
        }
        
        debug!("Built Yahoo Finance URL: {}", url.as_str());
        Ok(url)
    }
    
    /// Calculate appropriate time range for the request
    fn calculate_time_range(&self, interval: &str, limit: u32) -> (i64, i64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        // Calculate how far back to go based on interval and limit
        let seconds_per_interval = match interval {
            "1m" => 60,
            "2m" => 120,
            "5m" => 300,
            "15m" => 900,
            "30m" => 1800,
            "60m" | "1h" => 3600,
            "90m" => 5400,
            "1d" => 86400,
            "5d" => 432000,
            "1wk" => 604800,
            "1mo" => 2592000,
            "3mo" => 7776000,
            _ => 86400, // Default to 1 day
        };
        
        let period1 = now - (seconds_per_interval * limit as i64);
        (period1, now)
    }
    
    /// Make HTTP request with retry logic
    async fn make_request(&self, url: &Url) -> Result<Response> {
        let mut last_error = None;
        
        for attempt in 1..=self.config.max_retries {
            debug!("Making request to Yahoo Finance (attempt {}): {}", attempt, url.as_str());
            
            match self.client.get(url.clone()).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        debug!("Successfully received response from Yahoo Finance");
                        return Ok(response);
                    } else if response.status().as_u16() == 429 {
                        // Rate limited - wait longer before retry
                        warn!("Rate limited by Yahoo Finance, waiting before retry");
                        tokio::time::sleep(Duration::from_secs(
                            self.config.retry_delay_seconds * attempt as u64 * 2
                        )).await;
                        last_error = Some(QuantumTradeError::RateLimit {
                            operation: "yahoo_finance_request".to_string()
                        });
                    } else {
                        let status = response.status();
                        let error_text = response.text().await.unwrap_or_default();
                        last_error = Some(QuantumTradeError::QueryExecution {
                            message: format!("Yahoo Finance API error {}: {}", status, error_text)
                        });
                        
                        if status.is_client_error() {
                            // Don't retry client errors (4xx)
                            break;
                        }
                    }
                }
                Err(e) => {
                    warn!("Request failed (attempt {}): {}", attempt, e);
                    last_error = Some(QuantumTradeError::QueryExecution {
                        message: format!("HTTP request failed: {}", e)
                    });
                }
            }
            
            if attempt < self.config.max_retries {
                let delay = Duration::from_secs(self.config.retry_delay_seconds * attempt as u64);
                debug!("Waiting {:?} before retry", delay);
                tokio::time::sleep(delay).await;
            }
        }
        
        Err(last_error.unwrap_or_else(|| QuantumTradeError::QueryExecution {
            message: "All retry attempts failed".to_string()
        }))
    }
    
    /// Parse Yahoo Finance quote response into MarketData structures
    fn parse_quote_response(&self, symbol: &str, response: YahooQuoteResponse) -> Result<Vec<MarketData>> {
        if let Some(error) = response.chart.error {
            return Err(QuantumTradeError::QueryExecution {
                message: format!("Yahoo Finance API error: {} - {}", error.code, error.description)
            });
        }
        
        let result = response.chart.result
            .and_then(|mut results| results.pop())
            .ok_or_else(|| QuantumTradeError::DataValidation {
                message: "No data in Yahoo Finance response".to_string()
            })?;
        
        if result.timestamp.is_empty() {
            return Ok(Vec::new());
        }
        
        let quote = result.indicators.quote
            .into_iter()
            .next()
            .ok_or_else(|| QuantumTradeError::DataValidation {
                message: "No quote data in response".to_string()
            })?;
        
        let adj_close = result.indicators.adj_close
            .and_then(|mut adj| adj.pop())
            .map(|adj| adj.adj_close)
            .unwrap_or_else(|| quote.close.clone());
        
        let mut market_data = Vec::new();
        
        for (i, &timestamp) in result.timestamp.iter().enumerate() {
            // Skip if essential data is missing
            let open = quote.open.get(i).and_then(|&x| x);
            let high = quote.high.get(i).and_then(|&x| x);
            let low = quote.low.get(i).and_then(|&x| x);
            let close = quote.close.get(i).and_then(|&x| x);
            let volume = quote.volume.get(i).and_then(|&x| x).unwrap_or(0);
            let adjusted_close = adj_close.get(i).and_then(|&x| x);
            
            if let (Some(open), Some(high), Some(low), Some(close)) = (open, high, low, close) {
                let dt = Utc.timestamp_opt(timestamp, 0)
                    .single()
                    .ok_or_else(|| QuantumTradeError::DataValidation {
                        message: format!("Invalid timestamp: {}", timestamp)
                    })?;
                
                market_data.push(MarketData {
                    symbol: symbol.to_uppercase(),
                    timestamp: dt,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    adjusted_close: adjusted_close.unwrap_or(close),
                });
            } else {
                debug!("Skipping incomplete data point for {} at timestamp {}", symbol, timestamp);
            }
        }
        
        debug!("Parsed {} market data points for symbol {}", market_data.len(), symbol);
        Ok(market_data)
    }
    
    /// Validate symbol format
    pub fn is_valid_symbol(&self, symbol: &str) -> bool {
        // Basic validation - symbols should be alphanumeric with possible dots and hyphens
        if symbol.is_empty() || symbol.len() > 10 {
            return false;
        }
        
        symbol.chars().all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '^')
    }
    
    /// Get available intervals
    pub fn get_available_intervals() -> Vec<&'static str> {
        vec!["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    }
    
    /// Check if interval is valid
    pub fn is_valid_interval(&self, interval: &str) -> bool {
        Self::get_available_intervals().contains(&interval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::YahooFinanceConfig;
    
    #[test]
    fn test_symbol_validation() {
        let config = YahooFinanceConfig::default();
        let client = YahooFinanceClient::new(&config).unwrap();
        
        assert!(client.is_valid_symbol("AAPL"));
        assert!(client.is_valid_symbol("BRK.A"));
        assert!(client.is_valid_symbol("BTC-USD"));
        assert!(client.is_valid_symbol("^GSPC"));
        
        assert!(!client.is_valid_symbol(""));
        assert!(!client.is_valid_symbol("TOOLONGSYMBOL"));
        assert!(!client.is_valid_symbol("INVALID@"));
    }
    
    #[test]
    fn test_interval_validation() {
        let config = YahooFinanceConfig::default();
        let client = YahooFinanceClient::new(&config).unwrap();
        
        assert!(client.is_valid_interval("1m"));
        assert!(client.is_valid_interval("1d"));
        assert!(client.is_valid_interval("1wk"));
        
        assert!(!client.is_valid_interval("invalid"));
        assert!(!client.is_valid_interval(""));
    }
    
    #[test]
    fn test_time_range_calculation() {
        let config = YahooFinanceConfig::default();
        let client = YahooFinanceClient::new(&config).unwrap();
        
        let (period1, period2) = client.calculate_time_range("1d", 5);
        assert!(period2 > period1);
        assert!(period2 - period1 >= 5 * 86400); // At least 5 days
    }
    
    #[test]
    fn test_url_building() {
        let config = YahooFinanceConfig::default();
        let client = YahooFinanceClient::new(&config).unwrap();
        
        let url = client.build_quote_url("AAPL", "1d", 10).unwrap();
        assert!(url.as_str().contains("AAPL"));
        assert!(url.as_str().contains("interval=1d"));
    }
}