use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::warn;

#[derive(Debug)]
pub struct RateLimiter {
    calls_per_minute: u32,
    calls_per_day: u32,
    minute_counter: u32,
    day_counter: u32,
    last_minute_reset: DateTime<Utc>,
    last_day_reset: DateTime<Utc>,
    consecutive_rate_limits: u32,
    backoff_until: Option<DateTime<Utc>>,
}

impl RateLimiter {
    pub fn new(calls_per_minute: u32, calls_per_day: u32) -> Self {
        let now = Utc::now();
        
        Self {
            calls_per_minute,
            calls_per_day,
            minute_counter: 0,
            day_counter: 0,
            last_minute_reset: now,
            last_day_reset: now,
            consecutive_rate_limits: 0,
            backoff_until: None,
        }
    }
    
    pub async fn can_make_request(&mut self) -> bool {
        let now = Utc::now();
        
        // Check if in backoff period
        if let Some(backoff_until) = self.backoff_until {
            if now < backoff_until {
                return false;
            } else {
                self.backoff_until = None;
                self.consecutive_rate_limits = 0;
            }
        }
        
        // Reset counters if needed
        self.reset_counters_if_needed(now);
        
        // Check limits
        self.minute_counter < self.calls_per_minute && self.day_counter < self.calls_per_day
    }
    
    pub async fn record_request(&mut self) {
        self.minute_counter += 1;
        self.day_counter += 1;
        self.consecutive_rate_limits = 0;
    }
    
    pub async fn record_rate_limit(&mut self) {
        self.consecutive_rate_limits += 1;
        
        // Implement exponential backoff
        let backoff_minutes = 2_u32.pow(self.consecutive_rate_limits.min(6));
        self.backoff_until = Some(Utc::now() + chrono::Duration::minutes(backoff_minutes as i64));
        
        warn!("Rate limit hit. Backing off for {} minutes", backoff_minutes);
    }
    
    fn reset_counters_if_needed(&mut self, now: DateTime<Utc>) {
        // Reset minute counter
        if now.signed_duration_since(self.last_minute_reset).num_seconds() >= 60 {
            self.minute_counter = 0;
            self.last_minute_reset = now;
        }
        
        // Reset day counter
        if now.signed_duration_since(self.last_day_reset).num_hours() >= 24 {
            self.day_counter = 0;
            self.last_day_reset = now;
        }
    }
    
    pub fn get_status(&self) -> RateLimitStatus {
        RateLimitStatus {
            calls_remaining_minute: self.calls_per_minute.saturating_sub(self.minute_counter),
            calls_remaining_day: self.calls_per_day.saturating_sub(self.day_counter),
            next_minute_reset: self.last_minute_reset + chrono::Duration::minutes(1),
            next_day_reset: self.last_day_reset + chrono::Duration::hours(24),
            in_backoff: self.backoff_until.is_some(),
            backoff_until: self.backoff_until,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub calls_remaining_minute: u32,
    pub calls_remaining_day: u32,
    pub next_minute_reset: DateTime<Utc>,
    pub next_day_reset: DateTime<Utc>,
    pub in_backoff: bool,
    pub backoff_until: Option<DateTime<Utc>>,
} 