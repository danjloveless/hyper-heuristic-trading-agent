//! Unified monitoring interface

use crate::error::{LoggingMonitoringError, Result};
use crate::config::{MonitoringConfig, PerformanceMonitoringConfig, ResourceMonitoringConfig, BusinessMetricsConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use tokio::sync::Mutex;

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub latency_p50: Duration,
    pub latency_p95: Duration,
    pub latency_p99: Duration,
    pub throughput_rps: f64,
    pub error_rate: f64,
    pub success_rate: f64,
    pub timestamp: DateTime<Utc>,
}

/// Resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_bytes_in: u64,
    pub network_bytes_out: u64,
    pub timestamp: DateTime<Utc>,
}

/// Business metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessMetrics {
    pub prediction_accuracy: f64,
    pub trading_performance: f64,
    pub user_activity_count: u64,
    pub revenue_generated: f64,
    pub timestamp: DateTime<Utc>,
}

/// Monitoring data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringData {
    pub performance: Option<PerformanceMetrics>,
    pub resources: Option<ResourceMetrics>,
    pub business: Option<BusinessMetrics>,
    pub timestamp: DateTime<Utc>,
}

/// Main monitoring manager
pub struct MonitoringManager {
    config: MonitoringConfig,
    performance_metrics: Arc<Mutex<Vec<PerformanceMetrics>>>,
    resource_metrics: Arc<Mutex<Vec<ResourceMetrics>>>,
    business_metrics: Arc<Mutex<Vec<BusinessMetrics>>>,
    shutdown: Arc<Mutex<bool>>,
}

impl MonitoringManager {
    /// Create a new monitoring manager
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        let manager = Self {
            config,
            performance_metrics: Arc::new(Mutex::new(Vec::new())),
            resource_metrics: Arc::new(Mutex::new(Vec::new())),
            business_metrics: Arc::new(Mutex::new(Vec::new())),
            shutdown: Arc::new(Mutex::new(false)),
        };

        // Start background monitoring
        manager.start_background_monitoring().await;

        Ok(manager)
    }

    /// Start background monitoring tasks
    async fn start_background_monitoring(&self) {
        let config = self.config.clone();
        let performance_metrics = self.performance_metrics.clone();
        let resource_metrics = self.resource_metrics.clone();
        let business_metrics = self.business_metrics.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Collect every minute
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Collect performance metrics
                        if config.performance.enabled {
                            if let Ok(metrics) = Self::collect_performance_metrics(&config.performance).await {
                                let mut guard = performance_metrics.lock().await;
                                guard.push(metrics);
                                
                                // Keep only last 1000 metrics
                                if guard.len() > 1000 {
                                    guard.drain(0..guard.len() - 1000);
                                }
                            }
                        }

                        // Collect resource metrics
                        if config.resources.enabled {
                            if let Ok(metrics) = Self::collect_resource_metrics(&config.resources).await {
                                let mut guard = resource_metrics.lock().await;
                                guard.push(metrics);
                                
                                // Keep only last 1000 metrics
                                if guard.len() > 1000 {
                                    guard.drain(0..guard.len() - 1000);
                                }
                            }
                        }

                        // Collect business metrics
                        if config.business.enabled {
                            if let Ok(metrics) = Self::collect_business_metrics(&config.business).await {
                                let mut guard = business_metrics.lock().await;
                                guard.push(metrics);
                                
                                // Keep only last 1000 metrics
                                if guard.len() > 1000 {
                                    guard.drain(0..guard.len() - 1000);
                                }
                            }
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

    /// Collect performance metrics
    async fn collect_performance_metrics(config: &PerformanceMonitoringConfig) -> Result<PerformanceMetrics> {
        // In a real implementation, this would collect actual performance data
        // For now, we'll return mock data
        
        let latency_p50 = Duration::from_millis(50);
        let latency_p95 = Duration::from_millis(200);
        let latency_p99 = Duration::from_millis(500);
        let throughput_rps = 1000.0;
        let error_rate = 0.01; // 1%
        let success_rate = 0.99; // 99%

        Ok(PerformanceMetrics {
            latency_p50,
            latency_p95,
            latency_p99,
            throughput_rps,
            error_rate,
            success_rate,
            timestamp: Utc::now(),
        })
    }

    /// Collect resource metrics
    async fn collect_resource_metrics(config: &ResourceMonitoringConfig) -> Result<ResourceMetrics> {
        // In a real implementation, this would collect actual resource data
        // For now, we'll return mock data
        
        let cpu_usage_percent = 25.0;
        let memory_usage_bytes = 1024 * 1024 * 1024; // 1GB
        let memory_usage_percent = 50.0;
        let disk_usage_percent = 30.0;
        let network_bytes_in = 1024 * 1024; // 1MB
        let network_bytes_out = 512 * 1024; // 512KB

        Ok(ResourceMetrics {
            cpu_usage_percent,
            memory_usage_bytes,
            memory_usage_percent,
            disk_usage_percent,
            network_bytes_in,
            network_bytes_out,
            timestamp: Utc::now(),
        })
    }

    /// Collect business metrics
    async fn collect_business_metrics(config: &BusinessMetricsConfig) -> Result<BusinessMetrics> {
        // In a real implementation, this would collect actual business data
        // For now, we'll return mock data
        
        let prediction_accuracy = 0.85; // 85%
        let trading_performance = 0.12; // 12% return
        let user_activity_count = 1000;
        let revenue_generated = 50000.0; // $50,000

        Ok(BusinessMetrics {
            prediction_accuracy,
            trading_performance,
            user_activity_count,
            revenue_generated,
            timestamp: Utc::now(),
        })
    }

    /// Get current monitoring data
    pub async fn get_monitoring_data(&self) -> Result<MonitoringData> {
        let performance = if self.config.performance.enabled {
            let guard = self.performance_metrics.lock().await;
            guard.last().cloned()
        } else {
            None
        };

        let resources = if self.config.resources.enabled {
            let guard = self.resource_metrics.lock().await;
            guard.last().cloned()
        } else {
            None
        };

        let business = if self.config.business.enabled {
            let guard = self.business_metrics.lock().await;
            guard.last().cloned()
        } else {
            None
        };

        Ok(MonitoringData {
            performance,
            resources,
            business,
            timestamp: Utc::now(),
        })
    }

    /// Get performance metrics history
    pub async fn get_performance_history(&self, hours: u32) -> Result<Vec<PerformanceMetrics>> {
        if !self.config.performance.enabled {
            return Ok(Vec::new());
        }

        let cutoff_time = Utc::now() - chrono::Duration::hours(hours as i64);
        let guard = self.performance_metrics.lock().await;
        
        let history: Vec<PerformanceMetrics> = guard
            .iter()
            .filter(|metrics| metrics.timestamp >= cutoff_time)
            .cloned()
            .collect();

        Ok(history)
    }

    /// Get resource metrics history
    pub async fn get_resource_history(&self, hours: u32) -> Result<Vec<ResourceMetrics>> {
        if !self.config.resources.enabled {
            return Ok(Vec::new());
        }

        let cutoff_time = Utc::now() - chrono::Duration::hours(hours as i64);
        let guard = self.resource_metrics.lock().await;
        
        let history: Vec<ResourceMetrics> = guard
            .iter()
            .filter(|metrics| metrics.timestamp >= cutoff_time)
            .cloned()
            .collect();

        Ok(history)
    }

    /// Get business metrics history
    pub async fn get_business_history(&self, hours: u32) -> Result<Vec<BusinessMetrics>> {
        if !self.config.business.enabled {
            return Ok(Vec::new());
        }

        let cutoff_time = Utc::now() - chrono::Duration::hours(hours as i64);
        let guard = self.business_metrics.lock().await;
        
        let history: Vec<BusinessMetrics> = guard
            .iter()
            .filter(|metrics| metrics.timestamp >= cutoff_time)
            .cloned()
            .collect();

        Ok(history)
    }

    /// Check if performance is within acceptable thresholds
    pub async fn check_performance_thresholds(&self) -> Result<PerformanceThresholdStatus> {
        if !self.config.performance.enabled {
            return Ok(PerformanceThresholdStatus::Disabled);
        }

        let current_data = self.get_monitoring_data().await?;
        
        if let Some(performance) = current_data.performance {
            let thresholds = &self.config.performance.latency_thresholds;
            
            if performance.latency_p99 > thresholds.p99_threshold {
                Ok(PerformanceThresholdStatus::Critical)
            } else if performance.latency_p95 > thresholds.p95_threshold {
                Ok(PerformanceThresholdStatus::Warning)
            } else if performance.latency_p50 > thresholds.p50_threshold {
                Ok(PerformanceThresholdStatus::Degraded)
            } else {
                Ok(PerformanceThresholdStatus::Healthy)
            }
        } else {
            Ok(PerformanceThresholdStatus::Unknown)
        }
    }

    /// Get monitoring summary
    pub async fn get_monitoring_summary(&self) -> Result<MonitoringSummary> {
        let current_data = self.get_monitoring_data().await?;
        let performance_status = self.check_performance_thresholds().await?;

        let mut summary = MonitoringSummary {
            overall_status: MonitoringStatus::Healthy,
            performance_status,
            resource_utilization: ResourceUtilization::Normal,
            business_health: BusinessHealth::Good,
            last_updated: Utc::now(),
            alerts: Vec::new(),
        };

        // Check resource utilization
        if let Some(resources) = &current_data.resources {
            if resources.cpu_usage_percent > 90.0 || resources.memory_usage_percent > 90.0 {
                summary.resource_utilization = ResourceUtilization::High;
                summary.alerts.push("High resource utilization detected".to_string());
            }
        }

        // Check business health
        if let Some(business) = &current_data.business {
            if business.prediction_accuracy < 0.8 {
                summary.business_health = BusinessHealth::Poor;
                summary.alerts.push("Low prediction accuracy detected".to_string());
            } else if business.prediction_accuracy < 0.85 {
                summary.business_health = BusinessHealth::Fair;
            }
        }

        // Determine overall status
        summary.overall_status = match (performance_status, summary.resource_utilization, summary.business_health) {
            (PerformanceThresholdStatus::Critical, _, _) |
            (_, ResourceUtilization::High, _) |
            (_, _, BusinessHealth::Poor) => MonitoringStatus::Critical,
            
            (PerformanceThresholdStatus::Warning, _, _) |
            (_, _, BusinessHealth::Fair) => MonitoringStatus::Warning,
            
            (PerformanceThresholdStatus::Degraded, _, _) => MonitoringStatus::Degraded,
            
            _ => MonitoringStatus::Healthy,
        };

        Ok(summary)
    }

    /// Shutdown the monitoring manager
    pub async fn shutdown(&self) -> Result<()> {
        let mut shutdown_guard = self.shutdown.lock().await;
        *shutdown_guard = true;
        Ok(())
    }
}

/// Performance threshold status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceThresholdStatus {
    Healthy,
    Degraded,
    Warning,
    Critical,
    Unknown,
    Disabled,
}

impl std::fmt::Display for PerformanceThresholdStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerformanceThresholdStatus::Healthy => write!(f, "HEALTHY"),
            PerformanceThresholdStatus::Degraded => write!(f, "DEGRADED"),
            PerformanceThresholdStatus::Warning => write!(f, "WARNING"),
            PerformanceThresholdStatus::Critical => write!(f, "CRITICAL"),
            PerformanceThresholdStatus::Unknown => write!(f, "UNKNOWN"),
            PerformanceThresholdStatus::Disabled => write!(f, "DISABLED"),
        }
    }
}

/// Monitoring status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonitoringStatus {
    Healthy,
    Degraded,
    Warning,
    Critical,
}

impl std::fmt::Display for MonitoringStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitoringStatus::Healthy => write!(f, "HEALTHY"),
            MonitoringStatus::Degraded => write!(f, "DEGRADED"),
            MonitoringStatus::Warning => write!(f, "WARNING"),
            MonitoringStatus::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Resource utilization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceUtilization {
    Low,
    Normal,
    High,
    Critical,
}

impl std::fmt::Display for ResourceUtilization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceUtilization::Low => write!(f, "LOW"),
            ResourceUtilization::Normal => write!(f, "NORMAL"),
            ResourceUtilization::High => write!(f, "HIGH"),
            ResourceUtilization::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Business health level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusinessHealth {
    Poor,
    Fair,
    Good,
    Excellent,
}

impl std::fmt::Display for BusinessHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BusinessHealth::Poor => write!(f, "POOR"),
            BusinessHealth::Fair => write!(f, "FAIR"),
            BusinessHealth::Good => write!(f, "GOOD"),
            BusinessHealth::Excellent => write!(f, "EXCELLENT"),
        }
    }
}

/// Monitoring summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSummary {
    pub overall_status: MonitoringStatus,
    pub performance_status: PerformanceThresholdStatus,
    pub resource_utilization: ResourceUtilization,
    pub business_health: BusinessHealth,
    pub last_updated: DateTime<Utc>,
    pub alerts: Vec<String>,
}

impl MonitoringSummary {
    /// Check if the system is healthy
    pub fn is_healthy(&self) -> bool {
        self.overall_status == MonitoringStatus::Healthy
    }

    /// Check if there are any critical issues
    pub fn has_critical_issues(&self) -> bool {
        self.overall_status == MonitoringStatus::Critical
    }

    /// Get the number of active alerts
    pub fn alert_count(&self) -> usize {
        self.alerts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_manager_creation() {
        let config = MonitoringConfig::default();
        let manager = MonitoringManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_get_monitoring_data() {
        let config = MonitoringConfig::default();
        let manager = MonitoringManager::new(config).await.unwrap();
        
        let data = manager.get_monitoring_data().await;
        assert!(data.is_ok());
    }

    #[tokio::test]
    async fn test_get_performance_history() {
        let config = MonitoringConfig::default();
        let manager = MonitoringManager::new(config).await.unwrap();
        
        let history = manager.get_performance_history(1).await;
        assert!(history.is_ok());
    }

    #[tokio::test]
    async fn test_get_resource_history() {
        let config = MonitoringConfig::default();
        let manager = MonitoringManager::new(config).await.unwrap();
        
        let history = manager.get_resource_history(1).await;
        assert!(history.is_ok());
    }

    #[tokio::test]
    async fn test_get_business_history() {
        let config = MonitoringConfig::default();
        let manager = MonitoringManager::new(config).await.unwrap();
        
        let history = manager.get_business_history(1).await;
        assert!(history.is_ok());
    }

    #[tokio::test]
    async fn test_check_performance_thresholds() {
        let config = MonitoringConfig::default();
        let manager = MonitoringManager::new(config).await.unwrap();
        
        let status = manager.check_performance_thresholds().await;
        assert!(status.is_ok());
    }

    #[tokio::test]
    async fn test_get_monitoring_summary() {
        let config = MonitoringConfig::default();
        let manager = MonitoringManager::new(config).await.unwrap();
        
        let summary = manager.get_monitoring_summary().await;
        assert!(summary.is_ok());
        
        let summary = summary.unwrap();
        assert!(summary.alert_count() >= 0);
    }

    #[tokio::test]
    async fn test_monitoring_summary_methods() {
        let summary = MonitoringSummary {
            overall_status: MonitoringStatus::Healthy,
            performance_status: PerformanceThresholdStatus::Healthy,
            resource_utilization: ResourceUtilization::Normal,
            business_health: BusinessHealth::Good,
            last_updated: Utc::now(),
            alerts: vec!["Test alert".to_string()],
        };

        assert!(summary.is_healthy());
        assert!(!summary.has_critical_issues());
        assert_eq!(summary.alert_count(), 1);
    }
} 