use super::*;

/// Circuit breaker state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CircuitState {
    Closed,    // Normal operation
    Open,      // Failing fast, not executing calls
    HalfOpen,  // Testing if service has recovered
}

/// Circuit breaker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitStats {
    pub failure_count: u32,
    pub success_count: u32,
    pub total_calls: u64,
    pub last_failure_time: Option<DateTime<Utc>>,
    pub last_success_time: Option<DateTime<Utc>>,
}

/// Circuit breaker implementation
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    name: String,
    config: CircuitBreakerConfig,
    state: CircuitState,
    stats: CircuitStats,
    state_change_time: DateTime<Utc>,
    half_open_calls: u32,
}

impl CircuitBreaker {
    pub fn new(name: String, config: CircuitBreakerConfig) -> Self {
        Self {
            name,
            config,
            state: CircuitState::Closed,
            stats: CircuitStats {
                failure_count: 0,
                success_count: 0,
                total_calls: 0,
                last_failure_time: None,
                last_success_time: None,
            },
            state_change_time: Utc::now(),
            half_open_calls: 0,
        }
    }
    
    pub fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                let now = Utc::now();
                let elapsed = now
                    .signed_duration_since(self.state_change_time)
                    .num_milliseconds() as u64;
                
                if elapsed >= self.config.timeout_ms {
                    self.state = CircuitState::HalfOpen;
                    self.state_change_time = now;
                    self.half_open_calls = 0;
                    true
                } else {
                    false
                }
            },
            CircuitState::HalfOpen => {
                if self.half_open_calls < self.config.half_open_max_calls {
                    self.half_open_calls += 1;
                    true
                } else {
                    false
                }
            }
        }
    }
    
    pub fn record_success(&mut self) {
        self.stats.success_count += 1;
        self.stats.total_calls += 1;
        self.stats.last_success_time = Some(Utc::now());
        
        match self.state {
            CircuitState::HalfOpen => {
                if self.stats.success_count >= self.config.success_threshold {
                    self.state = CircuitState::Closed;
                    self.state_change_time = Utc::now();
                    self.stats.failure_count = 0;
                }
            },
            _ => {
                self.stats.failure_count = 0;
            }
        }
    }
    
    pub fn record_failure(&mut self) {
        self.stats.failure_count += 1;
        self.stats.total_calls += 1;
        self.stats.last_failure_time = Some(Utc::now());
        
        if self.stats.failure_count >= self.config.failure_threshold {
            self.state = CircuitState::Open;
            self.state_change_time = Utc::now();
        }
    }
    
    pub fn get_state(&self) -> CircuitState {
        self.state.clone()
    }
    
    pub fn get_stats(&self) -> &CircuitStats {
        &self.stats
    }
    
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.state_change_time = Utc::now();
        self.stats = CircuitStats {
            failure_count: 0,
            success_count: 0,
            total_calls: 0,
            last_failure_time: None,
            last_success_time: None,
        };
        self.half_open_calls = 0;
    }
} 