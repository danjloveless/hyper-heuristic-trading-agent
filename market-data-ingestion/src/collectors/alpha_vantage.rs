use crate::models::*;
use configuration_management::models::{AlphaVantageConfig, RateLimitConfig};
use core_traits::*;
use reqwest::Client;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

pub struct AlphaVantageCollector {
    config: AlphaVantageConfig,
    rate_limits: RateLimitConfig,
    client: Client,
    error_handler: Arc<dyn ErrorHandler>,
    monitoring: Arc<dyn MonitoringProvider>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    api_key: String,
}

impl AlphaVantageCollector {
    pub async fn new(
        config: AlphaVantageConfig,
        rate_limits: RateLimitConfig,
        error_handler: Arc<dyn ErrorHandler>,
        monitoring: Arc<dyn MonitoringProvider>,
    ) -> ServiceResult<Self> {
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms.into()))
            .build()
            .map_err(|e| ServiceError::System {
                message: format!("Failed to create HTTP client: {}", e),
            })?;
        
        let rate_limiter = Arc::new(RwLock::new(RateLimiter::new(
            rate_limits.requests_per_minute,
            rate_limits.requests_per_day,
        )));
        
        Ok(Self {
            config: config.clone(),
            rate_limits,
            client,
            error_handler,
            monitoring,
            rate_limiter,
            api_key: config.api_key.clone(),
        })
    }
    
    pub async fn collect_symbol_data(&self, symbol: &str, interval: Interval) -> ServiceResult<Vec<RawMarketData>> {
        let _context = ErrorContext::new(
            "alpha_vantage_collector".to_string(),
            format!("collect_{}_{}", symbol, interval.as_str()),
        );
        
        // Check rate limits
        {
            let mut limiter = self.rate_limiter.write().await;
            if !limiter.can_make_request() {
                let retry_after = limiter.time_until_reset();
                return Err(ServiceError::RateLimit {
                    service: "alpha_vantage".to_string(),
                    retry_after: Some(retry_after),
                });
            }
            limiter.record_request();
        }
        
        // Make API request
        let start_time = Instant::now();
        let url = self.build_url(symbol, &interval);
        
        debug!("Making Alpha Vantage API request: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| ServiceError::ExternalApi {
                api: "alpha_vantage".to_string(),
                message: format!("HTTP request failed: {}", e),
                status_code: None,
            })?;
        
        let status = response.status();
        if !status.is_success() {
            self.monitoring.record_counter("alpha_vantage_api_errors", &[
                ("status_code", &status.as_u16().to_string()),
                ("symbol", symbol),
            ]).await;
            
            return Err(ServiceError::ExternalApi {
                api: "alpha_vantage".to_string(),
                message: format!("API returned error status: {}", status),
                status_code: Some(status.as_u16()),
            });
        }
        
        // Parse response
        let response_text = response.text().await
            .map_err(|e| ServiceError::System {
                message: format!("Failed to read response body: {}", e),
            })?;
        
        let api_duration = start_time.elapsed();
        self.monitoring.record_timing("alpha_vantage_api_duration", api_duration, &[
            ("symbol", symbol),
            ("interval", interval.as_str()),
        ]).await;
        
        // Log response for debugging (only in debug builds)
        if cfg!(debug_assertions) {
            debug!("Alpha Vantage API response for {}: {}", symbol, response_text);
        }
        
        // Check for error responses first
        if response_text.contains("\"Information\"") || response_text.contains("\"Error Message\"") || response_text.contains("\"Note\"") {
            // This is an error response from Alpha Vantage
            let error_response: serde_json::Value = serde_json::from_str(&response_text)
                .map_err(|e| ServiceError::System {
                    message: format!("Failed to parse error response: {}", e),
                })?;
            
            let error_message = if let Some(info) = error_response.get("Information") {
                info.as_str().unwrap_or("Unknown error")
            } else if let Some(error_msg) = error_response.get("Error Message") {
                error_msg.as_str().unwrap_or("Unknown error")
            } else if let Some(note) = error_response.get("Note") {
                note.as_str().unwrap_or("Unknown error")
            } else {
                "Unknown Alpha Vantage error"
            };
            
            return Err(ServiceError::ExternalApi {
                api: "alpha_vantage".to_string(),
                message: format!("Alpha Vantage API error: {}", error_message),
                status_code: Some(200), // Alpha Vantage returns 200 even for errors
            });
        }
        
        // Parse JSON response
        let parsed_response: AlphaVantageResponse = serde_json::from_str(&response_text)
            .map_err(|e| ServiceError::System {
                message: format!("Failed to parse API response: {}", e),
            })?;
        
        // Convert to internal format
        let mut raw_data = Vec::new();
        for (timestamp_str, entry) in parsed_response.time_series {
            match RawMarketData::try_from((timestamp_str.as_str(), entry)) {
                Ok(data_point) => raw_data.push(data_point),
                Err(e) => {
                    warn!("Failed to parse data point for {}: {:?}", symbol, e);
                    self.monitoring.record_counter("data_parsing_errors", &[
                        ("symbol", symbol),
                        ("error_type", "timestamp_parse"),
                    ]).await;
                }
            }
        }
        
        debug!("Collected {} raw data points for {}", raw_data.len(), symbol);
        
        self.monitoring.record_metric("raw_data_points_collected", raw_data.len() as f64, &[
            ("symbol", symbol),
            ("interval", interval.as_str()),
        ]).await;
        
        Ok(raw_data)
    }
    
    pub async fn health_check(&self) -> bool {
        // Simple health check - try to make a test request
        let test_url = format!("{}?function=GLOBAL_QUOTE&symbol=AAPL", self.config.base_url);
        
        match self.client.get(&test_url).send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }
    
    pub async fn start_background_tasks(&self) -> ServiceResult<()> {
        // Could start rate limit reset tasks, health monitoring, etc.
        Ok(())
    }
    
    fn build_url(&self, symbol: &str, interval: &Interval) -> String {
        let function = match interval {
            Interval::Daily => "TIME_SERIES_DAILY",
            _ => "TIME_SERIES_INTRADAY",
        };
        
        let mut url = format!(
            "{}?function={}&symbol={}&outputsize=compact",
            self.config.base_url,
            function,
            symbol
        );
        
        if *interval != Interval::Daily {
            url.push_str(&format!("&interval={}", interval.as_alpha_vantage_param()));
        }
        
        // API key should come from configuration provider via secrets
        url.push_str(&format!("&apikey={}", self.api_key));
        
        url
    }
}

// Simple rate limiter implementation
struct RateLimiter {
    calls_per_minute: u32,
    calls_per_day: u32,
    minute_calls: Vec<Instant>,
    day_calls: Vec<Instant>,
}

impl RateLimiter {
    fn new(calls_per_minute: u32, calls_per_day: u32) -> Self {
        Self {
            calls_per_minute,
            calls_per_day,
            minute_calls: Vec::new(),
            day_calls: Vec::new(),
        }
    }
    
    fn can_make_request(&mut self) -> bool {
        self.cleanup_old_calls();
        
        self.minute_calls.len() < self.calls_per_minute as usize &&
        self.day_calls.len() < self.calls_per_day as usize
    }
    
    fn record_request(&mut self) {
        let now = Instant::now();
        self.minute_calls.push(now);
        self.day_calls.push(now);
    }
    
    fn time_until_reset(&self) -> Duration {
        if let Some(&oldest_call) = self.minute_calls.first() {
            let minute_ago = Instant::now() - Duration::from_secs(60);
            if oldest_call > minute_ago {
                oldest_call - minute_ago + Duration::from_secs(60)
            } else {
                Duration::from_secs(1)
            }
        } else {
            Duration::from_secs(1)
        }
    }
    
    fn cleanup_old_calls(&mut self) {
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);
        let day_ago = now - Duration::from_secs(24 * 60 * 60);
        
        self.minute_calls.retain(|&call_time| call_time > minute_ago);
        self.day_calls.retain(|&call_time| call_time > day_ago);
    }
} 