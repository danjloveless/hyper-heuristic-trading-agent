//! Health monitoring functionality

use crate::error::{LoggingMonitoringError, Result};
use crate::config::{HealthConfig, ServiceHealthConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use tokio::sync::Mutex;
use dashmap::DashMap;
use reqwest::Client as HttpClient;

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "HEALTHY"),
            HealthStatus::Degraded => write!(f, "DEGRADED"),
            HealthStatus::Unhealthy => write!(f, "UNHEALTHY"),
            HealthStatus::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub service_name: String,
    pub status: HealthStatus,
    pub response_time_ms: u64,
    pub last_check: DateTime<Utc>,
    pub error_message: Option<String>,
    pub details: HashMap<String, serde_json::Value>,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    pub overall_status: HealthStatus,
    pub services: HashMap<String, HealthCheckResult>,
    pub last_updated: DateTime<Utc>,
    pub total_services: usize,
    pub healthy_services: usize,
    pub degraded_services: usize,
    pub unhealthy_services: usize,
}

/// Health check configuration for a service
#[derive(Debug, Clone)]
struct ServiceHealthCheck {
    name: String,
    config: ServiceHealthConfig,
    last_check: Option<DateTime<Utc>>,
    consecutive_failures: u32,
    circuit_breaker_state: CircuitBreakerState,
}

/// Circuit breaker state
#[derive(Debug, Clone)]
enum CircuitBreakerState {
    Closed,    // Normal operation
    Open,      // Failing fast
    HalfOpen,  // Testing recovery
}

/// Main health manager
#[derive(Clone)]
pub struct HealthManager {
    config: HealthConfig,
    services: DashMap<String, ServiceHealthCheck>,
    health_results: DashMap<String, HealthCheckResult>,
    http_client: HttpClient,
    shutdown: Arc<Mutex<bool>>,
}

impl HealthManager {
    /// Create a new health manager
    pub async fn new(config: HealthConfig) -> Result<Self> {
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| LoggingMonitoringError::NetworkError {
                message: format!("Failed to create HTTP client: {}", e),
            })?;

        let manager = Self {
            config,
            services: DashMap::new(),
            health_results: DashMap::new(),
            http_client,
            shutdown: Arc::new(Mutex::new(false)),
        };

        // Initialize services from config
        manager.initialize_services().await;

        // Start background health checking
        manager.start_background_health_checks().await;

        Ok(manager)
    }

    /// Initialize services from configuration
    async fn initialize_services(&self) {
        for (service_name, service_config) in &self.config.services {
            let health_check = ServiceHealthCheck {
                name: service_name.clone(),
                config: service_config.clone(),
                last_check: None,
                consecutive_failures: 0,
                circuit_breaker_state: CircuitBreakerState::Closed,
            };
            
            self.services.insert(service_name.clone(), health_check);
        }
    }

    /// Start background health checking
    async fn start_background_health_checks(&self) {
        let config = self.config.clone();
        let services = self.services.clone();
        let health_results = self.health_results.clone();
        let http_client = self.http_client.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.check_interval);
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Perform health checks for all services
                        for entry in services.iter() {
                            let service_name = entry.key().clone();
                            let mut service_check = entry.value().clone();
                            
                            let result = Self::perform_health_check(
                                &http_client,
                                &service_check,
                                &config,
                            ).await;
                            
                            // Update service state
                            match &result.status {
                                HealthStatus::Healthy => {
                                    service_check.consecutive_failures = 0;
                                    service_check.circuit_breaker_state = CircuitBreakerState::Closed;
                                },
                                HealthStatus::Degraded | HealthStatus::Unhealthy => {
                                    service_check.consecutive_failures += 1;
                                    
                                    // Check circuit breaker
                                    if service_check.consecutive_failures >= service_check.config.failure_threshold {
                                        service_check.circuit_breaker_state = CircuitBreakerState::Open;
                                    }
                                },
                                HealthStatus::Unknown => {
                                    // Don't update failure count for unknown status
                                }
                            }
                            
                            service_check.last_check = Some(Utc::now());
                            services.insert(service_name.clone(), service_check);
                            health_results.insert(service_name, result);
                        }
                    }
                    
                    _ = async {
                        let shutdown_guard = shutdown.lock().await;
                        *shutdown_guard
                    } => {
                        break;
                    }
                }
            }
        });
    }

    /// Perform a health check for a service
    async fn perform_health_check(
        http_client: &HttpClient,
        service_check: &ServiceHealthCheck,
        config: &HealthConfig,
    ) -> HealthCheckResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        // Check circuit breaker state
        match service_check.circuit_breaker_state {
            CircuitBreakerState::Open => {
                return HealthCheckResult {
                    service_name: service_check.name.clone(),
                    status: HealthStatus::Unhealthy,
                    response_time_ms: 0,
                    last_check: Utc::now(),
                    error_message: Some("Circuit breaker is open".to_string()),
                    details,
                };
            },
            CircuitBreakerState::HalfOpen => {
                // Allow limited checks in half-open state
                details.insert("circuit_breaker_state".to_string(), serde_json::Value::String("half_open".to_string()));
            },
            CircuitBreakerState::Closed => {
                // Normal operation
            }
        }

        // Perform HTTP health check
        let health_url = format!("http://localhost:8080{}", service_check.config.endpoint);
        
        let result = http_client
            .get(&health_url)
            .timeout(service_check.config.expected_response_time)
            .send()
            .await;

        let response_time = start_time.elapsed();
        let response_time_ms = response_time.as_millis() as u64;

        match result {
            Ok(response) => {
                let status_code = response.status();
                
                if status_code.is_success() {
                    // Check response time against thresholds
                    let status = if response_time_ms > config.alerting.thresholds.critical_response_time.as_millis() as u64 {
                        HealthStatus::Unhealthy
                    } else if response_time_ms > config.alerting.thresholds.warning_response_time.as_millis() as u64 {
                        HealthStatus::Degraded
                    } else {
                        HealthStatus::Healthy
                    };

                    details.insert("status_code".to_string(), serde_json::Value::Number(status_code.as_u16().into()));
                    details.insert("response_time_ms".to_string(), serde_json::Value::Number(response_time_ms.into()));

                    HealthCheckResult {
                        service_name: service_check.name.clone(),
                        status,
                        response_time_ms,
                        last_check: Utc::now(),
                        error_message: None,
                        details,
                    }
                } else {
                    details.insert("status_code".to_string(), serde_json::Value::Number(status_code.as_u16().into()));
                    details.insert("response_time_ms".to_string(), serde_json::Value::Number(response_time_ms.into()));

                    HealthCheckResult {
                        service_name: service_check.name.clone(),
                        status: HealthStatus::Unhealthy,
                        response_time_ms,
                        last_check: Utc::now(),
                        error_message: Some(format!("HTTP {} returned", status_code)),
                        details,
                    }
                }
            },
            Err(e) => {
                details.insert("error_type".to_string(), serde_json::Value::String("request_failed".to_string()));
                details.insert("response_time_ms".to_string(), serde_json::Value::Number(response_time_ms.into()));

                HealthCheckResult {
                    service_name: service_check.name.clone(),
                    status: HealthStatus::Unhealthy,
                    response_time_ms,
                    last_check: Utc::now(),
                    error_message: Some(format!("Request failed: {}", e)),
                    details,
                }
            }
        }
    }

    /// Report health status for a service
    pub async fn report_health(&self, service: &str, status: HealthStatus) -> Result<()> {
        let result = HealthCheckResult {
            service_name: service.to_string(),
            status,
            response_time_ms: 0,
            last_check: Utc::now(),
            error_message: None,
            details: HashMap::new(),
        };

        self.health_results.insert(service.to_string(), result);
        Ok(())
    }

    /// Get health status for a specific service
    pub async fn get_service_health(&self, service: &str) -> Result<HealthStatus> {
        if let Some(result) = self.health_results.get(service) {
            Ok(result.status)
        } else {
            Err(LoggingMonitoringError::HealthStatusNotFound {
                service: service.to_string(),
            })
        }
    }

    /// Get system health status
    pub async fn get_system_health(&self) -> Result<SystemHealthStatus> {
        let mut services = HashMap::new();
        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut unhealthy_count = 0;

        for entry in self.health_results.iter() {
            let service_name = entry.key().clone();
            let result = entry.value().clone();
            
            services.insert(service_name, result.clone());
            
            match result.status {
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Degraded => degraded_count += 1,
                HealthStatus::Unhealthy => unhealthy_count += 1,
                HealthStatus::Unknown => {
                    // Don't count unknown status
                }
            }
        }

        let total_services = services.len();
        let overall_status = if unhealthy_count > 0 {
            HealthStatus::Unhealthy
        } else if degraded_count > 0 {
            HealthStatus::Degraded
        } else if healthy_count > 0 {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        };

        Ok(SystemHealthStatus {
            overall_status,
            services,
            last_updated: Utc::now(),
            total_services,
            healthy_services: healthy_count,
            degraded_services: degraded_count,
            unhealthy_services: unhealthy_count,
        })
    }

    /// Add a new service for health monitoring
    pub async fn add_service(&self, name: String, config: ServiceHealthConfig) -> Result<()> {
        let health_check = ServiceHealthCheck {
            name: name.clone(),
            config,
            last_check: None,
            consecutive_failures: 0,
            circuit_breaker_state: CircuitBreakerState::Closed,
        };
        
        self.services.insert(name, health_check);
        Ok(())
    }

    /// Remove a service from health monitoring
    pub async fn remove_service(&self, name: &str) -> Result<()> {
        self.services.remove(name);
        self.health_results.remove(name);
        Ok(())
    }

    /// Get all health check results
    pub async fn get_all_health_results(&self) -> HashMap<String, HealthCheckResult> {
        self.health_results
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Check if system is healthy
    pub async fn is_system_healthy(&self) -> Result<bool> {
        let system_health = self.get_system_health().await?;
        Ok(system_health.overall_status == HealthStatus::Healthy)
    }

    /// Get services with specific health status
    pub async fn get_services_by_status(&self, status: HealthStatus) -> Vec<String> {
        self.health_results
            .iter()
            .filter(|entry| entry.value().status == status)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Reset circuit breaker for a service
    pub async fn reset_circuit_breaker(&self, service: &str) -> Result<()> {
        if let Some(mut service_check) = self.services.get_mut(service) {
            service_check.consecutive_failures = 0;
            service_check.circuit_breaker_state = CircuitBreakerState::Closed;
        }
        Ok(())
    }

    /// Get health check statistics
    pub async fn get_health_statistics(&self) -> HealthStatistics {
        let mut stats = HealthStatistics {
            total_checks: 0,
            successful_checks: 0,
            failed_checks: 0,
            average_response_time_ms: 0.0,
            total_response_time_ms: 0,
        };

        for result in self.health_results.iter() {
            stats.total_checks += 1;
            stats.total_response_time_ms += result.value().response_time_ms;
            
            match result.value().status {
                HealthStatus::Healthy => stats.successful_checks += 1,
                HealthStatus::Unhealthy => stats.failed_checks += 1,
                _ => {
                    // Count degraded as partial success
                    stats.successful_checks += 1;
                }
            }
        }

        if stats.total_checks > 0 {
            stats.average_response_time_ms = stats.total_response_time_ms as f64 / stats.total_checks as f64;
        }

        stats
    }

    /// Shutdown the health manager
    pub async fn shutdown(&self) -> Result<()> {
        let mut shutdown_guard = self.shutdown.lock().await;
        *shutdown_guard = true;
        Ok(())
    }
}

/// Health statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatistics {
    pub total_checks: u64,
    pub successful_checks: u64,
    pub failed_checks: u64,
    pub average_response_time_ms: f64,
    pub total_response_time_ms: u64,
}

impl HealthStatistics {
    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_checks == 0 {
            0.0
        } else {
            (self.successful_checks as f64 / self.total_checks as f64) * 100.0
        }
    }

    /// Get failure rate as percentage
    pub fn failure_rate(&self) -> f64 {
        if self.total_checks == 0 {
            0.0
        } else {
            (self.failed_checks as f64 / self.total_checks as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_health_manager_creation() {
        let config = HealthConfig::default();
        let manager = HealthManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_health_status_reporting() {
        let config = HealthConfig::default();
        let manager = HealthManager::new(config).await.unwrap();
        
        let result = manager.report_health("test_service", HealthStatus::Healthy).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_service_health_retrieval() {
        let config = HealthConfig::default();
        let manager = HealthManager::new(config).await.unwrap();
        
        manager.report_health("test_service", HealthStatus::Healthy).await.unwrap();
        
        let status = manager.get_service_health("test_service").await;
        assert!(status.is_ok());
        assert_eq!(status.unwrap(), HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_system_health_status() {
        let config = HealthConfig::default();
        let manager = HealthManager::new(config).await.unwrap();
        
        manager.report_health("service1", HealthStatus::Healthy).await.unwrap();
        manager.report_health("service2", HealthStatus::Degraded).await.unwrap();
        
        let system_health = manager.get_system_health().await.unwrap();
        assert_eq!(system_health.overall_status, HealthStatus::Degraded);
        assert_eq!(system_health.total_services, 2);
        assert_eq!(system_health.healthy_services, 1);
        assert_eq!(system_health.degraded_services, 1);
    }

    #[tokio::test]
    async fn test_health_statistics() {
        let config = HealthConfig::default();
        let manager = HealthManager::new(config).await.unwrap();
        
        // Add some test results
        let result1 = HealthCheckResult {
            service_name: "test1".to_string(),
            status: HealthStatus::Healthy,
            response_time_ms: 100,
            last_check: Utc::now(),
            error_message: None,
            details: HashMap::new(),
        };
        
        let result2 = HealthCheckResult {
            service_name: "test2".to_string(),
            status: HealthStatus::Unhealthy,
            response_time_ms: 200,
            last_check: Utc::now(),
            error_message: Some("Test error".to_string()),
            details: HashMap::new(),
        };
        
        manager.health_results.insert("test1".to_string(), result1);
        manager.health_results.insert("test2".to_string(), result2);
        
        let stats = manager.get_health_statistics().await;
        assert_eq!(stats.total_checks, 2);
        assert_eq!(stats.successful_checks, 1);
        assert_eq!(stats.failed_checks, 1);
        assert_eq!(stats.average_response_time_ms, 150.0);
        assert_eq!(stats.success_rate(), 50.0);
        assert_eq!(stats.failure_rate(), 50.0);
    }
} 