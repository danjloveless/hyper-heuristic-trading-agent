use crate::errors::{Result, IngestionError};
use crate::models::MarketData;
use crate::config::AlphaVantageConfig;
use crate::rate_limiter::RateLimiter;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::debug;

/// Alpha Vantage API intervals
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Interval {
    OneMin,
    FiveMin,
    FifteenMin,
    ThirtyMin,
    SixtyMin,
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Interval::OneMin => write!(f, "1min"),
            Interval::FiveMin => write!(f, "5min"),
            Interval::FifteenMin => write!(f, "15min"),
            Interval::ThirtyMin => write!(f, "30min"),
            Interval::SixtyMin => write!(f, "60min"),
        }
    }
}

/// Alpha Vantage output size
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OutputSize {
    Compact,
    Full,
}

impl std::fmt::Display for OutputSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputSize::Compact => write!(f, "compact"),
            OutputSize::Full => write!(f, "full"),
        }
    }
}

/// Alpha Vantage API endpoints
#[derive(Debug, Clone)]
pub enum AlphaVantageEndpoint {
    IntradayData { symbol: String, interval: Interval, outputsize: OutputSize },
    DailyData { symbol: String, outputsize: OutputSize },
    DailyAdjusted { symbol: String, outputsize: OutputSize },
    WeeklyData { symbol: String },
    MonthlyData { symbol: String },
    GlobalQuote { symbol: String },
    SearchEndpoint { keywords: String },
}

impl AlphaVantageEndpoint {
    pub fn to_query_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        
        match self {
            AlphaVantageEndpoint::IntradayData { symbol, interval, outputsize } => {
                params.insert("function".to_string(), "TIME_SERIES_INTRADAY".to_string());
                params.insert("symbol".to_string(), symbol.clone());
                params.insert("interval".to_string(), interval.to_string());
                params.insert("outputsize".to_string(), outputsize.to_string());
            },
            AlphaVantageEndpoint::DailyData { symbol, outputsize } => {
                params.insert("function".to_string(), "TIME_SERIES_DAILY".to_string());
                params.insert("symbol".to_string(), symbol.clone());
                params.insert("outputsize".to_string(), outputsize.to_string());
            },
            AlphaVantageEndpoint::DailyAdjusted { symbol, outputsize } => {
                params.insert("function".to_string(), "TIME_SERIES_DAILY_ADJUSTED".to_string());
                params.insert("symbol".to_string(), symbol.clone());
                params.insert("outputsize".to_string(), outputsize.to_string());
            },
            AlphaVantageEndpoint::WeeklyData { symbol } => {
                params.insert("function".to_string(), "TIME_SERIES_WEEKLY".to_string());
                params.insert("symbol".to_string(), symbol.clone());
            },
            AlphaVantageEndpoint::MonthlyData { symbol } => {
                params.insert("function".to_string(), "TIME_SERIES_MONTHLY".to_string());
                params.insert("symbol".to_string(), symbol.clone());
            },
            AlphaVantageEndpoint::GlobalQuote { symbol } => {
                params.insert("function".to_string(), "GLOBAL_QUOTE".to_string());
                params.insert("symbol".to_string(), symbol.clone());
            },
            AlphaVantageEndpoint::SearchEndpoint { keywords } => {
                params.insert("function".to_string(), "SYMBOL_SEARCH".to_string());
                params.insert("keywords".to_string(), keywords.clone());
            },
        }
        
        params
    }
}

/// Alpha Vantage API response structures
#[derive(Debug, Deserialize)]
pub struct IntradayResponse {
    #[serde(rename = "Meta Data")]
    pub meta_data: Option<MetaData>,
    
    #[serde(flatten)]
    pub time_series: HashMap<String, HashMap<String, TimeSeriesEntry>>,
    
    #[serde(rename = "Error Message")]
    pub error_message: Option<String>,
    
    #[serde(rename = "Note")]
    pub note: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MetaData {
    #[serde(rename = "1. Information")]
    pub information: String,
    
    #[serde(rename = "2. Symbol")]
    pub symbol: String,
    
    #[serde(rename = "3. Last Refreshed")]
    pub last_refreshed: String,
    
    #[serde(rename = "4. Interval")]
    pub interval: Option<String>,
    
    #[serde(rename = "5. Output Size")]
    pub output_size: Option<String>,
    
    #[serde(rename = "6. Time Zone")]
    pub time_zone: String,
}

#[derive(Debug, Deserialize)]
pub struct TimeSeriesEntry {
    #[serde(rename = "1. open")]
    pub open: String,
    
    #[serde(rename = "2. high")]
    pub high: String,
    
    #[serde(rename = "3. low")]
    pub low: String,
    
    #[serde(rename = "4. close")]
    pub close: String,
    
    #[serde(rename = "5. volume")]
    pub volume: String,
    
    #[serde(rename = "6. adjusted close")]
    pub adjusted_close: Option<String>,
}

impl TimeSeriesEntry {
    pub fn to_market_data(&self, symbol: &str, timestamp: DateTime<Utc>) -> Result<MarketData> {
        let open = self.open.parse::<f64>()
            .map_err(|e| IngestionError::ParsingError { 
                field: "open".to_string(), 
                error: e.to_string() 
            })?;
        
        let high = self.high.parse::<f64>()
            .map_err(|e| IngestionError::ParsingError { 
                field: "high".to_string(), 
                error: e.to_string() 
            })?;
        
        let low = self.low.parse::<f64>()
            .map_err(|e| IngestionError::ParsingError { 
                field: "low".to_string(), 
                error: e.to_string() 
            })?;
        
        let close = self.close.parse::<f64>()
            .map_err(|e| IngestionError::ParsingError { 
                field: "close".to_string(), 
                error: e.to_string() 
            })?;
        
        let volume = self.volume.parse::<u64>()
            .map_err(|e| IngestionError::ParsingError { 
                field: "volume".to_string(), 
                error: e.to_string() 
            })?;
        
        let adjusted_close = if let Some(adj_close_str) = &self.adjusted_close {
            adj_close_str.parse::<f64>()
                .map_err(|e| IngestionError::ParsingError { 
                    field: "adjusted_close".to_string(), 
                    error: e.to_string() 
                })?
        } else {
            close
        };
        
        Ok(MarketData {
            symbol: symbol.to_string(),
            timestamp,
            open: Decimal::from_f64_retain(open).unwrap_or_default(),
            high: Decimal::from_f64_retain(high).unwrap_or_default(),
            low: Decimal::from_f64_retain(low).unwrap_or_default(),
            close: Decimal::from_f64_retain(close).unwrap_or_default(),
            volume,
            adjusted_close: Decimal::from_f64_retain(adjusted_close).unwrap_or_default(),
            source: "Alpha Vantage".to_string(),
            quality_score: 100, // Will be calculated by data quality controller
        })
    }
}

/// Alpha Vantage API client
#[derive(Debug, Clone)]
pub struct AlphaVantageClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    timeout: Duration,
    max_retries: u32,
}

impl AlphaVantageClient {
    pub fn new(config: AlphaVantageConfig, rate_limiter: Arc<RwLock<RateLimiter>>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            client,
            api_key: config.api_key,
            base_url: config.base_url,
            rate_limiter,
            timeout: Duration::from_secs(config.timeout_seconds),
            max_retries: config.max_retries,
        }
    }
    
    pub async fn get_intraday_data(
        &self,
        symbol: &str,
        interval: Interval,
        outputsize: OutputSize,
    ) -> Result<IntradayResponse> {
        let endpoint = AlphaVantageEndpoint::IntradayData {
            symbol: symbol.to_string(),
            interval,
            outputsize,
        };
        
        self.execute_request(endpoint).await
    }
    
    pub async fn get_daily_data(
        &self,
        symbol: &str,
        outputsize: OutputSize,
    ) -> Result<IntradayResponse> {
        let endpoint = AlphaVantageEndpoint::DailyData {
            symbol: symbol.to_string(),
            outputsize,
        };
        
        self.execute_request(endpoint).await
    }
    
    pub async fn get_daily_adjusted_data(
        &self,
        symbol: &str,
        outputsize: OutputSize,
    ) -> Result<IntradayResponse> {
        let endpoint = AlphaVantageEndpoint::DailyAdjusted {
            symbol: symbol.to_string(),
            outputsize,
        };
        
        self.execute_request(endpoint).await
    }
    
    async fn execute_request(&self, endpoint: AlphaVantageEndpoint) -> Result<IntradayResponse> {
        // Check rate limits
        {
            let mut rate_limiter = self.rate_limiter.write().await;
            if !rate_limiter.can_make_request().await {
                return Err(IngestionError::RateLimitExceeded {
                    limit_type: "API calls per minute".to_string(),
                });
            }
        }
        
        let mut params = endpoint.to_query_params();
        params.insert("apikey".to_string(), self.api_key.clone());
        
        let url = reqwest::Url::parse_with_params(&self.base_url, &params)
            .map_err(|e| IngestionError::ConfigurationError {
                parameter: format!("URL construction: {}", e),
            })?;
        
        debug!("Making request to: {}", url);
        
        let mut attempts = 0;
        let mut last_error = None;
        
        while attempts < self.max_retries {
            attempts += 1;
            
            match self.client.get(url.clone()).send().await {
                Ok(response) => {
                    let status = response.status();
                    
                    if status.is_success() {
                        // Record successful request
                        {
                            let mut rate_limiter = self.rate_limiter.write().await;
                            rate_limiter.record_request().await;
                        }
                        
                        let response_text = response.text().await
                            .map_err(|e| IngestionError::ApiError {
                                message: format!("Failed to read response: {}", e),
                                status_code: status.as_u16(),
                            })?;
                        
                        debug!("API response: {}", response_text);
                        
                        let parsed_response: IntradayResponse = serde_json::from_str(&response_text)
                            .map_err(|e| IngestionError::ParsingError {
                                field: "API response".to_string(),
                                error: format!("JSON parsing failed: {}", e),
                            })?;
                        
                        // Check for API errors in response
                        if let Some(error_msg) = &parsed_response.error_message {
                            return Err(IngestionError::ApiError {
                                message: error_msg.clone(),
                                status_code: status.as_u16(),
                            });
                        }
                        
                        // Check for rate limit notes
                        if let Some(note) = &parsed_response.note {
                            if note.contains("rate limit") || note.contains("frequency") {
                                return Err(IngestionError::RateLimitExceeded {
                                    limit_type: "API rate limit".to_string(),
                                });
                            }
                        }
                        
                        return Ok(parsed_response);
                    } else if status == 429 {
                        // Rate limited, update rate limiter
                        {
                            let mut rate_limiter = self.rate_limiter.write().await;
                            rate_limiter.record_rate_limit().await;
                        }
                        
                        last_error = Some(IngestionError::RateLimitExceeded {
                            limit_type: format!("HTTP 429: {}", status),
                        });
                    } else {
                        last_error = Some(IngestionError::ApiError {
                            message: format!("HTTP error: {}", status),
                            status_code: status.as_u16(),
                        });
                    }
                },
                Err(e) => {
                    if e.is_timeout() {
                        last_error = Some(IngestionError::RequestTimeout {
                            timeout_ms: self.timeout.as_millis() as u64,
                        });
                    } else {
                        last_error = Some(IngestionError::ApiError {
                            message: format!("Request failed: {}", e),
                            status_code: 0,
                        });
                    }
                }
            }
            
            // Exponential backoff
            if attempts < self.max_retries {
                let delay = Duration::from_millis(1000 * 2_u64.pow(attempts - 1));
                tokio::time::sleep(delay).await;
            }
        }
        
        Err(last_error.unwrap_or(IngestionError::ApiError {
            message: "Max retries exceeded".to_string(),
            status_code: 0,
        }))
    }
} 