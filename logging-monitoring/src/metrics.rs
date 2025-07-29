//! Metrics collection and aggregation

use crate::error::{LoggingMonitoringError, Result};
use crate::config::{MetricsConfig, CloudWatchMetricsConfig, PrometheusConfig, MetricsAggregationConfig, MetricsPerformanceConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, Mutex};
use dashmap::DashMap;
use metrics::{counter, gauge, histogram, timing};
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, Opts, Registry};
use prometheus_client::{
    encoding::EncodedMetric,
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
struct MetricEntry {
    name: String,
    metric_type: MetricType,
    value: MetricValue,
    tags: Tags,
    timestamp: DateTime<Utc>,
}

/// Main metrics manager
pub struct MetricsManager {
    config: MetricsConfig,
    metric_sender: mpsc::Sender<MetricEntry>,
    buffer: Arc<Mutex<Vec<MetricEntry>>>,
    shutdown: Arc<Mutex<bool>>,
    
    // Prometheus registry and metrics
    prometheus_registry: Option<PrometheusRegistry>,
    prometheus_counters: DashMap<String, PrometheusCounter>,
    prometheus_gauges: DashMap<String, PrometheusGauge>,
    prometheus_histograms: DashMap<String, PrometheusHistogram>,
    
    // Aggregated metrics storage
    aggregated_metrics: DashMap<String, AggregatedMetric>,
}

impl MetricsManager {
    /// Create a new metrics manager
    pub async fn new(config: MetricsConfig) -> Result<Self> {
        let (metric_sender, metric_receiver) = mpsc::channel(config.performance.buffer_size);
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(Mutex::new(false));

        // Initialize Prometheus registry if enabled
        let prometheus_registry = if config.prometheus.enabled {
            Some(PrometheusRegistry::default())
        } else {
            None
        };

        let manager = Self {
            config,
            metric_sender,
            buffer: buffer.clone(),
            shutdown: shutdown.clone(),
            prometheus_registry,
            prometheus_counters: DashMap::new(),
            prometheus_gauges: DashMap::new(),
            prometheus_histograms: DashMap::new(),
            aggregated_metrics: DashMap::new(),
        };

        // Start background processing
        manager.start_background_processing(metric_receiver, buffer, shutdown).await;

        Ok(manager)
    }

    /// Start background metric processing
    async fn start_background_processing(
        &self,
        mut receiver: mpsc::Receiver<MetricEntry>,
        buffer: Arc<Mutex<Vec<MetricEntry>>>,
        shutdown: Arc<Mutex<bool>>,
    ) {
        let config = self.config.clone();
        let aggregated_metrics = self.aggregated_metrics.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                tokio::select! {
                    // Process incoming metrics
                    Some(entry) = receiver.recv() => {
                        let mut buffer_guard = buffer.lock().await;
                        buffer_guard.push(entry);
                        
                        // Check if buffer is full
                        if buffer_guard.len() >= config.performance.buffer_size {
                            if let Err(e) = Self::process_metric_buffer(&config, &aggregated_metrics, &mut buffer_guard).await {
                                tracing::error!("Failed to process metric buffer: {}", e);
                            }
                        }
                    }
                    
                    // Periodic flush
                    _ = interval.tick() => {
                        let mut buffer_guard = buffer.lock().await;
                        if !buffer_guard.is_empty() {
                            if let Err(e) = Self::process_metric_buffer(&config, &aggregated_metrics, &mut buffer_guard).await {
                                tracing::error!("Failed to process metric buffer: {}", e);
                            }
                        }
                    }
                    
                    // Check shutdown
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

    /// Process metric buffer
    async fn process_metric_buffer(
        config: &MetricsConfig,
        aggregated_metrics: &DashMap<String, AggregatedMetric>,
        buffer: &mut Vec<MetricEntry>,
    ) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Process each metric
        for entry in buffer.drain(..) {
            Self::process_single_metric(config, aggregated_metrics, entry).await?;
        }

        Ok(())
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

        let aws_config = aws_config::from_env()
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
            .timestamp(entry.timestamp)
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
                    .or_insert_with(|| {
                        let counter = PrometheusCounter::default();
                        if let Some(registry) = &self.prometheus_registry {
                            registry.register(&entry.name, "Metric counter", counter.clone());
                        }
                        counter
                    });

                if let MetricValue::Integer(value) = entry.value {
                    counter.inc_by(value as u64);
                }
            },
            MetricType::Gauge => {
                let gauge = self.prometheus_gauges
                    .entry(metric_key.clone())
                    .or_insert_with(|| {
                        let gauge = PrometheusGauge::default();
                        if let Some(registry) = &self.prometheus_registry {
                            registry.register(&entry.name, "Metric gauge", gauge.clone());
                        }
                        gauge
                    });

                match entry.value {
                    MetricValue::Integer(value) => gauge.set(value as f64),
                    MetricValue::Float(value) => gauge.set(value),
                    MetricValue::Duration(duration) => gauge.set(duration.as_millis() as f64),
                }
            },
            MetricType::Histogram | MetricType::Timer => {
                let histogram = self.prometheus_histograms
                    .entry(metric_key.clone())
                    .or_insert_with(|| {
                        let histogram = PrometheusHistogram::new(
                            vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
                        );
                        if let Some(registry) = &self.prometheus_registry {
                            registry.register(&entry.name, "Metric histogram", histogram.clone());
                        }
                        histogram
                    });

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
        let tags = entry.tags.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");

        match entry.metric_type {
            MetricType::Counter => {
                if let MetricValue::Integer(value) = entry.value {
                    counter!(entry.name, value, &tags);
                }
            },
            MetricType::Gauge => {
                let value = match entry.value {
                    MetricValue::Integer(v) => v as f64,
                    MetricValue::Float(v) => v,
                    MetricValue::Duration(d) => d.as_millis() as f64,
                };
                gauge!(entry.name, value, &tags);
            },
            MetricType::Histogram => {
                let value = match entry.value {
                    MetricValue::Integer(v) => v as f64,
                    MetricValue::Float(v) => v,
                    MetricValue::Duration(d) => d.as_millis() as f64,
                };
                histogram!(entry.name, value, &tags);
            },
            MetricType::Timer => {
                if let MetricValue::Duration(duration) = entry.value {
                    timing!(entry.name, duration, &tags);
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
            let mut buffer = Vec::new();
            prometheus_client::encoding::text::encode(&mut buffer, registry)
                .map_err(|e| LoggingMonitoringError::PrometheusExportFailed {
                    message: e.to_string(),
                })?;
            
            String::from_utf8(buffer)
                .map_err(|e| LoggingMonitoringError::PrometheusExportFailed {
                    message: e.to_string(),
                })
        } else {
            Err(LoggingMonitoringError::PrometheusExportFailed {
                message: "Prometheus not enabled".to_string(),
            })
        }
    }

    /// Shutdown the metrics manager
    pub async fn shutdown(&self) -> Result<()> {
        let mut shutdown_guard = self.shutdown.lock().await;
        *shutdown_guard = true;
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