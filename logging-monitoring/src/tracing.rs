//! Distributed tracing functionality

use crate::error::{LoggingMonitoringError, Result};
use crate::config::TracingConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tokio::sync::Mutex;
use dashmap::DashMap;
use std::time::Instant;

/// Trace ID type
pub type TraceId = Uuid;

/// Span ID type
pub type SpanId = Uuid;

/// Span status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Error,
    Timeout,
    Cancelled,
}

impl std::fmt::Display for SpanStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpanStatus::Ok => write!(f, "OK"),
            SpanStatus::Error => write!(f, "ERROR"),
            SpanStatus::Timeout => write!(f, "TIMEOUT"),
            SpanStatus::Cancelled => write!(f, "CANCELLED"),
        }
    }
}

/// Trace span with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub operation_name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
    pub status: SpanStatus,
    pub annotations: HashMap<String, String>,
    pub tags: HashMap<String, String>,
}

impl TraceSpan {
    /// Create a new trace span
    pub fn new(trace_id: TraceId, span_id: SpanId, operation_name: String) -> Self {
        Self {
            trace_id,
            span_id,
            parent_span_id: None,
            operation_name,
            start_time: Utc::now(),
            end_time: None,
            duration_ms: None,
            status: SpanStatus::Ok,
            annotations: HashMap::new(),
            tags: HashMap::new(),
        }
    }

    /// Set parent span ID
    pub fn with_parent(mut self, parent_span_id: SpanId) -> Self {
        self.parent_span_id = Some(parent_span_id);
        self
    }

    /// Add an annotation
    pub fn add_annotation(&mut self, key: String, value: String) {
        self.annotations.insert(key, value);
    }

    /// Add a tag
    pub fn add_tag(&mut self, key: String, value: String) {
        self.tags.insert(key, value);
    }

    /// End the span
    pub fn end(&mut self, status: SpanStatus) {
        self.end_time = Some(Utc::now());
        self.duration_ms = self.end_time
            .map(|end| end.signed_duration_since(self.start_time).num_milliseconds() as u64);
        self.status = status;
    }
}

/// Active span for tracking current operation
#[derive(Debug, Clone)]
pub struct ActiveSpan {
    span: TraceSpan,
    start_instant: Instant,
}

impl ActiveSpan {
    /// Create a new active span
    pub fn new(span: TraceSpan) -> Self {
        Self {
            start_instant: Instant::now(),
            span,
        }
    }

    /// Add an annotation
    pub fn add_annotation(&mut self, key: String, value: String) {
        self.span.add_annotation(key, value);
    }

    /// Add a tag
    pub fn add_tag(&mut self, key: String, value: String) {
        self.span.add_tag(key, value);
    }

    /// End the span
    pub fn end(mut self, status: SpanStatus) -> TraceSpan {
        self.span.end(status);
        self.span
    }

    /// Get the span ID
    pub fn span_id(&self) -> SpanId {
        self.span.span_id
    }

    /// Get the trace ID
    pub fn trace_id(&self) -> TraceId {
        self.span.trace_id
    }
}

/// Main trace manager
#[derive(Clone)]
pub struct TraceManager {
    config: TracingConfig,
    active_spans: DashMap<TraceId, Vec<ActiveSpan>>,
    completed_spans: DashMap<TraceId, Vec<TraceSpan>>,
    sampling_rate: f64,
    shutdown: Arc<AtomicBool>,
}

impl TraceManager {
    /// Create a new trace manager
    pub async fn new(config: TracingConfig) -> Result<Self> {
        let sampling_rate = config.sampling.rate; // Extract before moving config
        let manager = Self {
            config,
            active_spans: DashMap::new(),
            completed_spans: DashMap::new(),
            sampling_rate, // Use extracted sampling rate
            shutdown: Arc::new(AtomicBool::new(false)),
        };

        // Initialize X-Ray if enabled
        if manager.config.xray.enabled {
            manager.initialize_xray().await?;
        }

        Ok(manager)
    }

    /// Initialize X-Ray integration
    async fn initialize_xray(&self) -> Result<()> {
        // In a real implementation, this would set up the X-Ray daemon connection
        // For now, we'll just log that X-Ray is enabled
        tracing::info!("X-Ray tracing enabled for service: {}", self.config.xray.service_name);
        Ok(())
    }

    /// Start a new trace
    pub async fn start_trace(&self, operation: &str) -> Result<TraceId> {
        // Check sampling
        if !self.should_sample() {
            return Err(LoggingMonitoringError::TraceCreationFailed {
                message: "Trace not sampled".to_string(),
            });
        }

        let trace_id = Uuid::new_v4();
        let span_id = Uuid::new_v4();
        
        let mut span = TraceSpan::new(trace_id, span_id, operation.to_string());
        span.add_tag("service".to_string(), self.config.xray.service_name.clone());
        span.add_tag("environment".to_string(), self.config.xray.environment.clone());

        let active_span = ActiveSpan::new(span);
        
        self.active_spans.entry(trace_id).or_insert_with(Vec::new).push(active_span);

        Ok(trace_id)
    }

    /// Start a new span within an existing trace
    pub async fn start_span(&self, parent: TraceId, operation: &str) -> Result<SpanId> {
        let span_id = Uuid::new_v4();
        
        if let Some(mut spans) = self.active_spans.get_mut(&parent) {
            let mut span = TraceSpan::new(parent, span_id, operation.to_string())
                .with_parent(spans.last().map(|s| s.span_id()).unwrap_or_default());
            span.add_tag("service".to_string(), self.config.xray.service_name.clone());
            span.add_tag("environment".to_string(), self.config.xray.environment.clone());

            let active_span = ActiveSpan::new(span);
            spans.push(active_span);
            
            Ok(span_id)
        } else {
            Err(LoggingMonitoringError::TraceCreationFailed {
                message: format!("Parent trace {} not found", parent),
            })
        }
    }

    /// End a span
    pub async fn end_span(&self, span: SpanId, result: crate::SpanResult) -> Result<()> {
        let status = if result.success {
            SpanStatus::Ok
        } else {
            SpanStatus::Error
        };

        // Find and end the span
        for mut entry in self.active_spans.iter_mut() {
            let trace_id = *entry.key();
            let spans = entry.value_mut();
            
            if let Some(index) = spans.iter().position(|s| s.span_id() == span) {
                let mut active_span = spans.remove(index);
                
                // Add result metadata
                active_span.add_tag("success".to_string(), result.success.to_string());
                if let Some(error_msg) = &result.error_message {
                    active_span.add_annotation("error_message".to_string(), error_msg.clone());
                }
                
                for (key, value) in &result.metadata {
                    active_span.add_tag(key.clone(), value.clone());
                }

                let completed_span = active_span.end(status);
                
                // Store completed span
                self.completed_spans.entry(trace_id).or_insert_with(Vec::new).push(completed_span);
                
                return Ok(());
            }
        }

        Err(LoggingMonitoringError::InvalidSpanId { span_id: span.to_string() })
    }

    /// Add an annotation to a trace
    pub async fn add_annotation(&self, trace: TraceId, key: &str, value: &str) -> Result<()> {
        if let Some(mut spans) = self.active_spans.get_mut(&trace) {
            if let Some(active_span) = spans.last_mut() {
                active_span.add_annotation(key.to_string(), value.to_string());
                return Ok(());
            }
        }

        Err(LoggingMonitoringError::InvalidTraceId { trace_id: trace.to_string() })
    }

    /// Get all spans for a trace
    pub async fn get_trace_spans(&self, trace_id: TraceId) -> Result<Vec<TraceSpan>> {
        let mut spans = Vec::new();
        
        // Add active spans
        if let Some(active_spans) = self.active_spans.get(&trace_id) {
            for active_span in active_spans.iter() {
                spans.push(active_span.span.clone());
            }
        }
        
        // Add completed spans
        if let Some(completed_spans) = self.completed_spans.get(&trace_id) {
            spans.extend(completed_spans.clone());
        }
        
        Ok(spans)
    }

    /// Check if we should sample this trace
    fn should_sample(&self) -> bool {
        if !self.config.sampling.adaptive {
            return fastrand::f64() <= self.sampling_rate;
        }

        // Adaptive sampling logic
        let current_rate = self.get_current_sampling_rate();
        fastrand::f64() <= current_rate
    }

    /// Get current adaptive sampling rate
    fn get_current_sampling_rate(&self) -> f64 {
        let total_traces = self.active_spans.len() + self.completed_spans.len();
        
        if total_traces == 0 {
            return self.config.sampling.max_rate;
        }

        // Simple adaptive logic: increase rate if we have few traces
        let target_traces = 1000; // Target number of traces
        let current_rate = (target_traces as f64 / total_traces as f64) * self.sampling_rate;
        
        current_rate.clamp(self.config.sampling.min_rate, self.config.sampling.max_rate)
    }

    /// Export traces to X-Ray
    async fn export_to_xray(&self, trace_id: TraceId) -> Result<()> {
        if !self.config.xray.enabled {
            return Ok(());
        }

        let spans = self.get_trace_spans(trace_id).await?;
        
        // Convert to X-Ray format
        let xray_document = self.convert_to_xray_format(&spans)?;
        
        // Send to X-Ray daemon
        self.send_to_xray_daemon(&xray_document).await?;
        
        Ok(())
    }

    /// Convert spans to X-Ray format
    fn convert_to_xray_format(&self, spans: &[TraceSpan]) -> Result<String> {
        #[derive(Serialize)]
        struct XRaySegment {
            id: String,
            trace_id: String,
            parent_id: Option<String>,
            name: String,
            start_time: f64,
            end_time: Option<f64>,
            annotations: HashMap<String, String>,
            metadata: HashMap<String, String>,
        }

        #[derive(Serialize)]
        struct XRayDocument {
            format: String,
            version: u32,
            header: XRayHeader,
            segments: Vec<XRaySegment>,
        }

        #[derive(Serialize)]
        struct XRayHeader {
            trace_id: String,
        }

        let segments: Vec<XRaySegment> = spans
            .iter()
            .map(|span| XRaySegment {
                id: span.span_id.to_string(),
                trace_id: span.trace_id.to_string(),
                parent_id: span.parent_span_id.map(|id| id.to_string()),
                name: span.operation_name.clone(),
                start_time: span.start_time.timestamp_millis() as f64 / 1000.0,
                end_time: span.end_time.map(|end| end.timestamp_millis() as f64 / 1000.0),
                annotations: span.annotations.clone(),
                metadata: span.tags.clone(),
            })
            .collect();

        let document = XRayDocument {
            format: "json".to_string(),
            version: 1,
            header: XRayHeader {
                trace_id: spans.first().map(|s| s.trace_id.to_string()).unwrap_or_default(),
            },
            segments,
        };

        serde_json::to_string(&document)
            .map_err(|e| LoggingMonitoringError::JsonSerializationError { message: e.to_string() })
    }

    /// Send document to X-Ray daemon
    async fn send_to_xray_daemon(&self, document: &str) -> Result<()> {
        // In a real implementation, this would send to the X-Ray daemon
        // For now, we'll just log the document
        tracing::debug!("X-Ray document: {}", document);
        Ok(())
    }

    /// Clean up old traces
    async fn cleanup_old_traces(&self) {
        let max_age = self.config.span.max_duration;
        let cutoff_time = Utc::now() - chrono::Duration::from_std(max_age).unwrap_or_default();

        // Clean up active spans
        self.active_spans.retain(|_, spans| {
            spans.retain(|span| span.span.start_time > cutoff_time);
            !spans.is_empty()
        });

        // Clean up completed spans
        self.completed_spans.retain(|_, spans| {
            spans.retain(|span| span.start_time > cutoff_time);
            !spans.is_empty()
        });
    }

    /// Shutdown the trace manager
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        
        // Export remaining traces
        for entry in self.active_spans.iter() {
            let trace_id = *entry.key();
            if let Err(e) = self.export_to_xray(trace_id).await {
                tracing::error!("Failed to export trace {}: {}", trace_id, e);
            }
        }
        
        Ok(())
    }
}

impl Drop for TraceManager {
    fn drop(&mut self) {
        // Set shutdown flag without blocking operations
        // The background cleanup will check this flag and exit gracefully
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_trace_creation() {
        let mut config = TracingConfig::default();
        config.sampling.rate = 1.0; // 100% sampling for tests
        config.sampling.adaptive = false; // Disable adaptive sampling for tests
        let manager = TraceManager::new(config).await.unwrap();
        
        let trace_id = manager.start_trace("test_operation").await;
        assert!(trace_id.is_ok());
    }

    #[tokio::test]
    async fn test_span_creation() {
        let mut config = TracingConfig::default();
        config.sampling.rate = 1.0; // 100% sampling for tests
        config.sampling.adaptive = false; // Disable adaptive sampling for tests
        let manager = TraceManager::new(config).await.unwrap();
        
        let trace_id = manager.start_trace("test_operation").await.unwrap();
        let span_id = manager.start_span(trace_id, "test_span").await;
        assert!(span_id.is_ok());
    }

    #[tokio::test]
    async fn test_span_ending() {
        let mut config = TracingConfig::default();
        config.sampling.rate = 1.0; // 100% sampling for tests
        config.sampling.adaptive = false; // Disable adaptive sampling for tests
        let manager = TraceManager::new(config).await.unwrap();
        
        let trace_id = manager.start_trace("test_operation").await.unwrap();
        let span_id = manager.start_span(trace_id, "test_span").await.unwrap();
        
        let result = crate::SpanResult {
            success: true,
            error_message: None,
            duration_ms: 100,
            metadata: HashMap::new(),
        };
        
        let end_result = manager.end_span(span_id, result).await;
        assert!(end_result.is_ok());
    }

    #[tokio::test]
    async fn test_trace_spans() {
        let mut config = TracingConfig::default();
        config.sampling.rate = 1.0; // 100% sampling for tests
        config.sampling.adaptive = false; // Disable adaptive sampling for tests
        let manager = TraceManager::new(config).await.unwrap();
        
        let trace_id = manager.start_trace("test_operation").await.unwrap();
        let span_id = manager.start_span(trace_id, "test_span").await.unwrap();
        
        let result = crate::SpanResult {
            success: true,
            error_message: None,
            duration_ms: 100,
            metadata: HashMap::new(),
        };
        
        manager.end_span(span_id, result).await.unwrap();
        
        let spans = manager.get_trace_spans(trace_id).await.unwrap();
        assert_eq!(spans.len(), 2); // Root span + child span
    }
} 