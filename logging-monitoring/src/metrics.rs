//! Metrics collection and aggregation

use crate::error::{LoggingMonitoringError, Result};
use crate::config::{MetricsConfig, CloudWatchMetricsConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, Mutex};
use dashmap::DashMap;
use metrics::{counter, gauge, histogram};
use prometheus_client::{
    metrics::{counter::Counter as PrometheusCounter, gauge::Gauge as PrometheusGauge, histogram::Histogram as PrometheusHistogram},
    registry::Registry as PrometheusRegistry,
};

/// Metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Timer,
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricType::Counter => write!(f, "counter"),
            MetricType::Gauge => write!(f, "gauge"),
            MetricType::Histogram => write!(f, "histogram"),
            MetricType::Timer => write!(f, "timer"),
        }
    }
}

/// Metric values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Integer(i64),
    Float(f64),
    Duration(Duration),
}

/// Tags for metrics
pub type Tags = HashMap<String, String>;

/// Metric data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricData {
    pub name: String,
    pub metric_type: MetricType,
    pub value: MetricValue,
    pub timestamp: DateTime<Utc>,
    pub tags: Tags,
    pub dimensions: HashMap<String, String>,
}

/// Metric entry for internal processing
#[derive(Debug, Clone)]
pub struct MetricEntry {
    pub name: String,
    pub metric_type: MetricType,
    pub value: MetricValue,
    pub tags: Tags,
    pub timestamp: DateTime<Utc>,
}

/// Main metrics manager
#[derive(Clone)]
pub struct MetricsManager {
    config: MetricsConfig,
    metric_sender: mpsc::Sender<MetricEntry>,
    aggregated_metrics: Arc<DashMap<String, AggregatedMetric>>,
    shutdown: Arc<AtomicBool>,
    
    // Prometheus registry and metrics
    prometheus_registry: Option<Arc<Mutex<PrometheusRegistry>>>,
    prometheus_counters: DashMap<String, PrometheusCounter>,
    prometheus_gauges: DashMap<String, PrometheusGauge>,
    prometheus_histograms: DashMap<String, PrometheusHistogram>,
}

impl MetricsManager {
    /// Create a new metrics manager
    pub async fn new(config: MetricsConfig) -> Result<Self> {
        let (metric_sender, metric_receiver) = mpsc::channel(config.performance.buffer_size);
        let aggregated_metrics = Arc::new(DashMap::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        // Initialize Prometheus registry if enabled
        let prometheus_registry = if config.prometheus.enabled {
            Some(Arc::new(Mutex::new(PrometheusRegistry::default())))
        } else {
            None
        };

        let manager = Self {
            config: config.clone(),
            metric_sender,
            aggregated_metrics: aggregated_metrics.clone(),
            shutdown: shutdown.clone(),
            prometheus_registry,
            prometheus_counters: DashMap::new(),
            prometheus_gauges: DashMap::new(),
            prometheus_histograms: DashMap::new(),
        };

        // Start background processing
        manager.start_background_processing(metric_receiver, aggregated_metrics, shutdown, config).await;

        Ok(manager)
    }

    /// Start background metric processing
    async fn start_background_processing(
        &self,
        mut receiver: mpsc::Receiver<MetricEntry>,
        aggregated_metrics: Arc<DashMap<String, AggregatedMetric>>,
        shutdown: Arc<AtomicBool>,
        config: MetricsConfig,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.aggregation.intervals[0]);
            
            loop {
                tokio::select! {
                    result = receiver.recv() => {
                        match result {
                            Some(entry) => {
                                Self::process_metric_message(entry, &aggregated_metrics, &config).await;
                            },
                            None => break,
                        }
                    },
                    
                    _ = interval.tick() => {
                        Self::aggregate_metrics(&aggregated_metrics, &config).await;
                    },
                    
                    _ = async {
                        while !shutdown.load(Ordering::Relaxed) {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                    } => {
                        break;
                    }
                }
            }
        })
    }

    /// Process metric message
    async fn process_metric_message(
        entry: MetricEntry,
        aggregated_metrics: &Arc<DashMap<String, AggregatedMetric>>,
        _config: &MetricsConfig,
    ) {
        // Update aggregated metrics
        let metric_key = Self::create_metric_key(&entry.name, &entry.tags);
        
        aggregated_metrics
            .entry(metric_key.clone())
            .and_modify(|metric| {
                metric.update(&entry);
            })
            .or_insert_with(|| AggregatedMetric::new(entry.clone()));

        // Send to CloudWatch if enabled
        if _config.cloudwatch.enabled {
            if let Err(e) = Self::send_to_cloudwatch(&_config.cloudwatch, &entry).await {
                tracing::error!("Failed to send metric to CloudWatch: {}", e);
            }
        }

        // Update metrics crate
        if let Err(e) = Self::update_metrics_crate(&entry) {
            tracing::error!("Failed to update metrics crate: {}", e);
        }
    }

    /// Aggregate metrics
    async fn aggregate_metrics(
        aggregated_metrics: &Arc<DashMap<String, AggregatedMetric>>,
        _config: &MetricsConfig,
    ) {
        // This would perform additional aggregation logic
        // For now, we'll just log that aggregation is happening
        tracing::debug!("Aggregating {} metrics", aggregated_metrics.len());
    }

    /// Process a single metric
    async fn process_single_metric(
        config: &MetricsConfig,
        aggregated_metrics: &DashMap<String, AggregatedMetric>,
        entry: MetricEntry,
    ) -> Result<()> {
        // Update aggregated metrics
        let metric_key = Self::create_metric_key(&entry.name, &entry.tags);
        
        aggregated_metrics
            .entry(metric_key.clone())
            .and_modify(|metric| {
                metric.update(&entry);
            })
            .or_insert_with(|| AggregatedMetric::new(entry.clone()));

        // Send to CloudWatch if enabled
        if config.cloudwatch.enabled {
            Self::send_to_cloudwatch(&config.cloudwatch, &entry).await?;
        }

        // Update Prometheus metrics if enabled
        if config.prometheus.enabled {
            // Note: This would need to be called on the manager instance
            // For now, we'll skip this in the static context
        }

        // Update metrics crate
        Self::update_metrics_crate(&entry)?;

        Ok(())
    }

    /// Create a metric key for aggregation
    fn create_metric_key(name: &str, tags: &Tags) -> String {
        let mut key = name.to_string();
        if !tags.is_empty() {
            let mut sorted_tags: Vec<_> = tags.iter().collect();
            sorted_tags.sort_by_key(|(k, _)| *k);
            
            let tag_string = sorted_tags
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(",");
            
            key.push_str(&format!("{{{}}}", tag_string));
        }
        key
    }

    /// Send metric to CloudWatch
    async fn send_to_cloudwatch(config: &CloudWatchMetricsConfig, entry: &MetricEntry) -> Result<()> {
        use aws_sdk_cloudwatch::Client as CloudWatchClient;
        use aws_sdk_cloudwatch::types::{MetricDatum, StandardUnit};

        let aws_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_config::Region::new(config.region.clone()))
            .load()
            .await;
        
        let client = CloudWatchClient::new(&aws_config);

        let (value, unit) = match &entry.value {
            MetricValue::Integer(v) => (*v as f64, StandardUnit::Count),
            MetricValue::Float(v) => (*v, StandardUnit::None),
            MetricValue::Duration(d) => (d.as_millis() as f64, StandardUnit::Milliseconds),
        };

        let mut dimensions = Vec::new();
        for (key, value) in &entry.tags {
            dimensions.push(aws_sdk_cloudwatch::types::Dimension::builder()
                .name(key)
                .value(value)
                .build());
        }

        let metric_datum = MetricDatum::builder()
            .metric_name(&entry.name)
            .value(value)
            .unit(unit)
            .set_dimensions(Some(dimensions))
            .build();

        let result = client
            .put_metric_data()
            .namespace(&config.namespace)
            .metric_data(metric_datum)
            .send()
            .await;

        if let Err(e) = result {
            return Err(LoggingMonitoringError::CloudWatchMetricFailed {
                message: e.to_string(),
            });
        }

        Ok(())
    }

    /// Update Prometheus metric
    async fn update_prometheus_metric(&self, entry: &MetricEntry) -> Result<()> {
        let metric_key = Self::create_metric_key(&entry.name, &entry.tags);

        match entry.metric_type {
            MetricType::Counter => {
                let counter = self.prometheus_counters
                    .entry(metric_key.clone())
                    .or_insert_with(|| PrometheusCounter::default());

                // Register the metric if it's new
                if !self.prometheus_counters.contains_key(&metric_key) {
                    if let Some(_registry) = &self.prometheus_registry {
                        // Note: This is a limitation - we can't register in async context here
                        // In a real implementation, you'd want to handle this differently
                    }
                }

                if let MetricValue::Integer(value) = entry.value {
                    counter.inc_by(value as u64);
                }
            },
            MetricType::Gauge => {
                let gauge = self.prometheus_gauges
                    .entry(metric_key.clone())
                    .or_insert_with(|| PrometheusGauge::default());

                // Register the metric if it's new
                if !self.prometheus_gauges.contains_key(&metric_key) {
                    if let Some(_registry) = &self.prometheus_registry {
                        // Note: This is a limitation - we can't register in async context here
                        // In a real implementation, you'd want to handle this differently
                    }
                }

                match entry.value {
                    MetricValue::Integer(value) => { gauge.set(value); },
                    MetricValue::Float(value) => { gauge.set(value as i64); },
                    MetricValue::Duration(duration) => { gauge.set(duration.as_millis() as i64); },
                }
            },
            MetricType::Histogram | MetricType::Timer => {
                let histogram = self.prometheus_histograms
                    .entry(metric_key.clone())
                    .or_insert_with(|| {
                        PrometheusHistogram::new(
                            vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0].into_iter()
                        )
                    });

                // Register the metric if it's new
                if !self.prometheus_histograms.contains_key(&metric_key) {
                    if let Some(_registry) = &self.prometheus_registry {
                        // Note: This is a limitation - we can't register in async context here
                        // In a real implementation, you'd want to handle this differently
                    }
                }

                let value = match entry.value {
                    MetricValue::Integer(v) => v as f64,
                    MetricValue::Float(v) => v,
                    MetricValue::Duration(d) => d.as_millis() as f64,
                };

                histogram.observe(value);
            },
        }

        Ok(())
    }

    /// Update metrics crate
    fn update_metrics_crate(entry: &MetricEntry) -> Result<()> {
        let tags: Vec<_> = entry.tags.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        match entry.metric_type {
            MetricType::Counter => {
                if let MetricValue::Integer(value) = entry.value {
                    counter!(entry.name.clone(), value.try_into().unwrap_or(0), &tags);
                }
            },
            MetricType::Gauge => {
                let value = match entry.value {
                    MetricValue::Integer(v) => v as f64,
                    MetricValue::Float(v) => v,
                    MetricValue::Duration(d) => d.as_millis() as f64,
                };
                gauge!(entry.name.clone(), value, &tags);
            },
            MetricType::Histogram => {
                let value = match entry.value {
                    MetricValue::Integer(v) => v as f64,
                    MetricValue::Float(v) => v,
                    MetricValue::Duration(d) => d.as_millis() as f64,
                };
                histogram!(entry.name.clone(), value, &tags);
            },
            MetricType::Timer => {
                if let MetricValue::Duration(duration) = entry.value {
                    histogram!(entry.name.clone(), duration.as_millis() as f64, &tags);
                }
            },
        }

        Ok(())
    }

    /// Record a counter metric
    pub async fn record_counter(&self, name: &str, value: i64, tags: Tags) -> Result<()> {
        self.record_metric(name, MetricType::Counter, MetricValue::Integer(value), tags).await
    }

    /// Record a gauge metric
    pub async fn record_gauge(&self, name: &str, value: f64, tags: Tags) -> Result<()> {
        self.record_metric(name, MetricType::Gauge, MetricValue::Float(value), tags).await
    }

    /// Record a histogram metric
    pub async fn record_histogram(&self, name: &str, value: f64, tags: Tags) -> Result<()> {
        self.record_metric(name, MetricType::Histogram, MetricValue::Float(value), tags).await
    }

    /// Record a timer metric
    pub async fn record_timer(&self, name: &str, duration: Duration, tags: Tags) -> Result<()> {
        self.record_metric(name, MetricType::Timer, MetricValue::Duration(duration), tags).await
    }

    /// Internal method to record any metric
    async fn record_metric(
        &self,
        name: &str,
        metric_type: MetricType,
        value: MetricValue,
        tags: Tags,
    ) -> Result<()> {
        // Check if we should sample this metric
        if self.config.performance.enable_sampling {
            let sample_rate = self.config.performance.sampling_rate;
            if fastrand::f64() > sample_rate {
                return Ok(());
            }
        }

        let entry = MetricEntry {
            name: name.to_string(),
            metric_type,
            value,
            tags,
            timestamp: Utc::now(),
        };

        self.metric_sender
            .send(entry)
            .await
            .map_err(|_| LoggingMonitoringError::MetricRecordingFailed {
                name: name.to_string(),
                message: "Failed to send metric to background processor".to_string(),
            })?;

        Ok(())
    }

    /// Get aggregated metrics
    pub async fn get_aggregated_metrics(&self) -> HashMap<String, AggregatedMetric> {
        self.aggregated_metrics
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get Prometheus metrics as string
    pub async fn get_prometheus_metrics(&self) -> Result<String> {
        if let Some(registry) = &self.prometheus_registry {
            let registry = registry.lock().await;
            let mut buffer = String::new();
            prometheus_client::encoding::text::encode(&mut buffer, &*registry)
                .map_err(|e| LoggingMonitoringError::PrometheusExportFailed {
                    message: e.to_string(),
                })?;
            
            Ok(buffer)
        } else {
            Err(LoggingMonitoringError::PrometheusExportFailed {
                message: "Prometheus not enabled".to_string(),
            })
        }
    }

    /// Shutdown the metrics manager
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(200)).await;
        Ok(())
    }
}

/// Aggregated metric data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    pub name: String,
    pub metric_type: MetricType,
    pub count: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub last_value: f64,
    pub last_timestamp: DateTime<Utc>,
    pub tags: Tags,
}

impl AggregatedMetric {
    /// Create a new aggregated metric
    pub fn new(entry: MetricEntry) -> Self {
        let value = match entry.value {
            MetricValue::Integer(v) => v as f64,
            MetricValue::Float(v) => v,
            MetricValue::Duration(d) => d.as_millis() as f64,
        };

        Self {
            name: entry.name,
            metric_type: entry.metric_type,
            count: 1,
            sum: value,
            min: value,
            max: value,
            last_value: value,
            last_timestamp: entry.timestamp,
            tags: entry.tags,
        }
    }

    /// Update the aggregated metric with a new value
    pub fn update(&mut self, entry: &MetricEntry) {
        let value = match &entry.value {
            MetricValue::Integer(v) => *v as f64,
            MetricValue::Float(v) => *v,
            MetricValue::Duration(d) => d.as_millis() as f64,
        };

        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.last_value = value;
        self.last_timestamp = entry.timestamp;
    }

    /// Get the average value
    pub fn average(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_metrics_manager_creation() {
        let config = MetricsConfig::default();
        let manager = MetricsManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_counter_recording() {
        let config = MetricsConfig::default();
        let manager = MetricsManager::new(config).await.unwrap();
        
        let tags = HashMap::new();
        let result = manager.record_counter("test_counter", 42, tags).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_gauge_recording() {
        let config = MetricsConfig::default();
        let manager = MetricsManager::new(config).await.unwrap();
        
        let tags = HashMap::new();
        let result = manager.record_gauge("test_gauge", 3.14, tags).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_histogram_recording() {
        let config = MetricsConfig::default();
        let manager = MetricsManager::new(config).await.unwrap();
        
        let tags = HashMap::new();
        let result = manager.record_histogram("test_histogram", 100.0, tags).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_timer_recording() {
        let config = MetricsConfig::default();
        let manager = MetricsManager::new(config).await.unwrap();
        
        let tags = HashMap::new();
        let duration = Duration::from_millis(100);
        let result = manager.record_timer("test_timer", duration, tags).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_aggregated_metric() {
        let entry = MetricEntry {
            name: "test".to_string(),
            metric_type: MetricType::Counter,
            value: MetricValue::Integer(10),
            tags: HashMap::new(),
            timestamp: Utc::now(),
        };

        let mut aggregated = AggregatedMetric::new(entry);
        assert_eq!(aggregated.count, 1);
        assert_eq!(aggregated.sum, 10.0);

        let update_entry = MetricEntry {
            name: "test".to_string(),
            metric_type: MetricType::Counter,
            value: MetricValue::Integer(20),
            tags: HashMap::new(),
            timestamp: Utc::now(),
        };

        aggregated.update(&update_entry);
        assert_eq!(aggregated.count, 2);
        assert_eq!(aggregated.sum, 30.0);
        assert_eq!(aggregated.average(), 15.0);
    }
} 