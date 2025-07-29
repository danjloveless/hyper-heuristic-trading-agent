//! Event management functionality

use crate::error::{LoggingMonitoringError, Result};
use crate::config::{EventsConfig, SnsConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

/// Event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    System,
    Business,
    Security,
    Performance,
    Error,
    Alert,
    Notification,
    Metric,
    Audit,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::System => write!(f, "system"),
            EventType::Business => write!(f, "business"),
            EventType::Security => write!(f, "security"),
            EventType::Performance => write!(f, "performance"),
            EventType::Error => write!(f, "error"),
            EventType::Alert => write!(f, "alert"),
            EventType::Notification => write!(f, "notification"),
            EventType::Metric => write!(f, "metric"),
            EventType::Audit => write!(f, "audit"),
        }
    }
}

/// Event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl std::fmt::Display for EventSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventSeverity::Critical => write!(f, "CRITICAL"),
            EventSeverity::High => write!(f, "HIGH"),
            EventSeverity::Medium => write!(f, "MEDIUM"),
            EventSeverity::Low => write!(f, "LOW"),
            EventSeverity::Info => write!(f, "INFO"),
        }
    }
}

/// System event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEvent {
    pub event_id: String,
    pub event_type: EventType,
    pub severity: EventSeverity,
    pub title: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub tags: HashMap<String, String>,
    pub data: HashMap<String, serde_json::Value>,
}

impl SystemEvent {
    /// Create a new system event
    pub fn new(
        event_type: EventType,
        severity: EventSeverity,
        title: String,
        message: String,
        source: String,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4().to_string(),
            event_type,
            severity,
            title,
            message,
            timestamp: Utc::now(),
            source,
            tags: HashMap::new(),
            data: HashMap::new(),
        }
    }

    /// Add a tag to the event
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Add data to the event
    pub fn with_data(mut self, key: String, value: serde_json::Value) -> Self {
        self.data.insert(key, value);
        self
    }
}

/// Alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub alert_id: String,
    pub title: String,
    pub message: String,
    pub severity: EventSeverity,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub category: String,
    pub tags: HashMap<String, String>,
    pub data: HashMap<String, serde_json::Value>,
}

impl Alert {
    /// Create a new alert
    pub fn new(
        title: String,
        message: String,
        severity: EventSeverity,
        source: String,
        category: String,
    ) -> Self {
        Self {
            alert_id: Uuid::new_v4().to_string(),
            title,
            message,
            severity,
            timestamp: Utc::now(),
            source,
            category,
            tags: HashMap::new(),
            data: HashMap::new(),
        }
    }

    /// Add a tag to the alert
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Add data to the alert
    pub fn with_data(mut self, key: String, value: serde_json::Value) -> Self {
        self.data.insert(key, value);
        self
    }
}

/// Notification event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub notification_id: String,
    pub title: String,
    pub message: String,
    pub priority: NotificationPriority,
    pub timestamp: DateTime<Utc>,
    pub recipient: Option<String>,
    pub channel: NotificationChannel,
    pub tags: HashMap<String, String>,
    pub data: HashMap<String, serde_json::Value>,
}

/// Notification priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationPriority {
    Urgent,
    High,
    Normal,
    Low,
}

impl std::fmt::Display for NotificationPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotificationPriority::Urgent => write!(f, "URGENT"),
            NotificationPriority::High => write!(f, "HIGH"),
            NotificationPriority::Normal => write!(f, "NORMAL"),
            NotificationPriority::Low => write!(f, "LOW"),
        }
    }
}

/// Notification channel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email,
    Sms,
    Slack,
    Webhook,
    Sns,
}

impl std::fmt::Display for NotificationChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotificationChannel::Email => write!(f, "email"),
            NotificationChannel::Sms => write!(f, "sms"),
            NotificationChannel::Slack => write!(f, "slack"),
            NotificationChannel::Webhook => write!(f, "webhook"),
            NotificationChannel::Sns => write!(f, "sns"),
        }
    }
}

impl Notification {
    /// Create a new notification
    pub fn new(
        title: String,
        message: String,
        priority: NotificationPriority,
        channel: NotificationChannel,
    ) -> Self {
        Self {
            notification_id: Uuid::new_v4().to_string(),
            title,
            message,
            priority,
            timestamp: Utc::now(),
            recipient: None,
            channel,
            tags: HashMap::new(),
            data: HashMap::new(),
        }
    }

    /// Set the recipient
    pub fn with_recipient(mut self, recipient: String) -> Self {
        self.recipient = Some(recipient);
        self
    }

    /// Add a tag to the notification
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Add data to the notification
    pub fn with_data(mut self, key: String, value: serde_json::Value) -> Self {
        self.data.insert(key, value);
        self
    }
}

/// Event entry for internal processing
#[derive(Debug, Clone)]
enum EventEntry {
    System(SystemEvent),
    Alert(Alert),
    Notification(Notification),
}

/// Main event manager
#[derive(Clone)]
pub struct EventManager {
    config: EventsConfig,
    event_sender: mpsc::Sender<EventEntry>,
    buffer: Arc<Mutex<Vec<EventEntry>>>,
    shutdown: Arc<Mutex<bool>>,
    
    // SNS client for notifications
    sns_client: Option<aws_sdk_sns::Client>,
}

impl EventManager {
    /// Create a new event manager
    pub async fn new(config: EventsConfig) -> Result<Self> {
        let (event_sender, event_receiver) = mpsc::channel(1000);
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(Mutex::new(false));

        // Initialize SNS client if enabled
        let sns_client = if config.sns.enabled {
            let aws_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
                .region(aws_config::Region::new(config.sns.region.clone()))
                .load()
                .await;
            Some(aws_sdk_sns::Client::new(&aws_config))
        } else {
            None
        };

        let manager = Self {
            config,
            event_sender,
            buffer: buffer.clone(),
            shutdown: shutdown.clone(),
            sns_client,
        };

        // Start background processing
        manager.start_background_processing(event_receiver, buffer, shutdown).await;

        Ok(manager)
    }

    /// Start background event processing
    async fn start_background_processing(
        &self,
        mut receiver: mpsc::Receiver<EventEntry>,
        buffer: Arc<Mutex<Vec<EventEntry>>>,
        shutdown: Arc<Mutex<bool>>,
    ) {
        let config = self.config.clone();
        let sns_client = self.sns_client.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                tokio::select! {
                    // Process incoming events
                    Some(entry) = receiver.recv() => {
                        let mut buffer_guard = buffer.lock().await;
                        buffer_guard.push(entry);
                        
                        // Process buffer if it gets large
                        if buffer_guard.len() >= 100 {
                            if let Err(e) = Self::process_event_buffer(&config, &sns_client, &mut buffer_guard).await {
                                tracing::error!("Failed to process event buffer: {}", e);
                            }
                        }
                    }
                    
                    // Periodic flush
                    _ = interval.tick() => {
                        let mut buffer_guard = buffer.lock().await;
                        if !buffer_guard.is_empty() {
                            if let Err(e) = Self::process_event_buffer(&config, &sns_client, &mut buffer_guard).await {
                                tracing::error!("Failed to process event buffer: {}", e);
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

    /// Process event buffer
    async fn process_event_buffer(
        config: &EventsConfig,
        sns_client: &Option<aws_sdk_sns::Client>,
        buffer: &mut Vec<EventEntry>,
    ) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Process each event
        for entry in buffer.drain(..) {
            Self::process_single_event(config, sns_client, entry).await?;
        }

        Ok(())
    }

    /// Process a single event
    async fn process_single_event(
        config: &EventsConfig,
        sns_client: &Option<aws_sdk_sns::Client>,
        entry: EventEntry,
    ) -> Result<()> {
        // Apply filtering
        if !Self::should_process_event(config, &entry) {
            return Ok(());
        }

        // Route event
        let topic = Self::route_event(config, &entry);

        // Send to SNS if enabled
        if let Some(client) = sns_client {
            if config.sns.enabled {
                Self::send_to_sns(client, &config.sns, &topic, &entry).await?;
            }
        }

        // Log the event
        Self::log_event(&entry).await?;

        Ok(())
    }

    /// Check if event should be processed based on filtering rules
    fn should_process_event(config: &EventsConfig, entry: &EventEntry) -> bool {
        if !config.filtering.enabled {
            return true;
        }

        let (event_type, severity) = match entry {
            EventEntry::System(event) => (event.event_type, event.severity),
            EventEntry::Alert(alert) => (EventType::Alert, alert.severity),
            EventEntry::Notification(_) => (EventType::Notification, EventSeverity::Info),
        };

        // Check event type filters
        if !config.filtering.event_type_filters.is_empty() {
            let event_type_str = event_type.to_string();
            if !config.filtering.event_type_filters.contains(&event_type_str) {
                return false;
            }
        }

        // Check severity filters
        if !config.filtering.severity_filters.is_empty() {
            let severity_str = severity.to_string();
            if !config.filtering.severity_filters.contains(&severity_str) {
                return false;
            }
        }

        // Check service filters
        if !config.filtering.service_filters.is_empty() {
            let source = match entry {
                EventEntry::System(event) => &event.source,
                EventEntry::Alert(alert) => &alert.source,
                EventEntry::Notification(_) => "notification",
            };
            
            if !config.filtering.service_filters.contains(&source.to_string()) {
                return false;
            }
        }

        true
    }

    /// Route event to appropriate topic
    fn route_event(config: &EventsConfig, entry: &EventEntry) -> String {
        let event_type = match entry {
            EventEntry::System(event) => event.event_type,
            EventEntry::Alert(_) => EventType::Alert,
            EventEntry::Notification(_) => EventType::Notification,
        };

        config.routing.event_routes
            .get(&event_type.to_string())
            .cloned()
            .unwrap_or_else(|| config.routing.default_topic.clone())
    }

    /// Send event to SNS
    async fn send_to_sns(
        client: &aws_sdk_sns::Client,
        sns_config: &SnsConfig,
        topic: &str,
        entry: &EventEntry,
    ) -> Result<()> {
        let message = Self::serialize_event(entry)?;
        
        let topic_arn = match entry {
            EventEntry::Alert(_) => &sns_config.alert_topic_arn,
            EventEntry::Notification(_) => &sns_config.notification_topic_arn,
            _ => &sns_config.alert_topic_arn, // Default to alert topic
        };

        let result = client
            .publish()
            .topic_arn(topic_arn)
            .message(&message)
            .subject(topic)
            .send()
            .await;

        if let Err(e) = result {
            return Err(LoggingMonitoringError::SnsDeliveryFailed {
                message: e.to_string(),
            });
        }

        Ok(())
    }

    /// Serialize event to JSON
    fn serialize_event(entry: &EventEntry) -> Result<String> {
        match entry {
            EventEntry::System(event) => {
                serde_json::to_string(event)
                    .map_err(|e| LoggingMonitoringError::JsonSerializationError { message: e.to_string() })
            },
            EventEntry::Alert(alert) => {
                serde_json::to_string(alert)
                    .map_err(|e| LoggingMonitoringError::JsonSerializationError { message: e.to_string() })
            },
            EventEntry::Notification(notification) => {
                serde_json::to_string(notification)
                    .map_err(|e| LoggingMonitoringError::JsonSerializationError { message: e.to_string() })
            },
        }
    }

    /// Log event
    async fn log_event(entry: &EventEntry) -> Result<()> {
        let log_message = match entry {
            EventEntry::System(event) => {
                format!("[{}] {}: {}", event.severity, event.title, event.message)
            },
            EventEntry::Alert(alert) => {
                format!("[{}] ALERT: {} - {}", alert.severity, alert.title, alert.message)
            },
            EventEntry::Notification(notification) => {
                format!("[{}] NOTIFICATION: {} - {}", notification.priority, notification.title, notification.message)
            },
        };

        tracing::info!("{}", log_message);
        Ok(())
    }

    /// Emit a system event
    pub async fn emit_event(&self, event: SystemEvent) -> Result<()> {
        self.send_event(EventEntry::System(event)).await
    }

    /// Emit an alert
    pub async fn emit_alert(&self, alert: Alert) -> Result<()> {
        self.send_event(EventEntry::Alert(alert)).await
    }

    /// Emit a notification
    pub async fn emit_notification(&self, notification: Notification) -> Result<()> {
        self.send_event(EventEntry::Notification(notification)).await
    }

    /// Internal method to send any event
    async fn send_event(&self, entry: EventEntry) -> Result<()> {
        self.event_sender
            .send(entry)
            .await
            .map_err(|_| LoggingMonitoringError::EventEmissionFailed {
                message: "Failed to send event to background processor".to_string(),
            })?;

        Ok(())
    }

    /// Create a system event builder
    pub fn system_event(&self) -> SystemEventBuilder {
        SystemEventBuilder::new()
    }

    /// Create an alert builder
    pub fn alert(&self) -> AlertBuilder {
        AlertBuilder::new()
    }

    /// Create a notification builder
    pub fn notification(&self) -> NotificationBuilder {
        NotificationBuilder::new()
    }

    /// Shutdown the event manager
    pub async fn shutdown(&self) -> Result<()> {
        let mut shutdown_guard = self.shutdown.lock().await;
        *shutdown_guard = true;
        Ok(())
    }
}

/// System event builder
pub struct SystemEventBuilder {
    event_type: Option<EventType>,
    severity: Option<EventSeverity>,
    title: Option<String>,
    message: Option<String>,
    source: Option<String>,
    tags: HashMap<String, String>,
    data: HashMap<String, serde_json::Value>,
}

impl SystemEventBuilder {
    /// Create a new system event builder
    pub fn new() -> Self {
        Self {
            event_type: None,
            severity: None,
            title: None,
            message: None,
            source: None,
            tags: HashMap::new(),
            data: HashMap::new(),
        }
    }

    /// Set the event type
    pub fn event_type(mut self, event_type: EventType) -> Self {
        self.event_type = Some(event_type);
        self
    }

    /// Set the severity
    pub fn severity(mut self, severity: EventSeverity) -> Self {
        self.severity = Some(severity);
        self
    }

    /// Set the title
    pub fn title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    /// Set the message
    pub fn message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }

    /// Set the source
    pub fn source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Add a tag
    pub fn tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Add data
    pub fn data(mut self, key: String, value: serde_json::Value) -> Self {
        self.data.insert(key, value);
        self
    }

    /// Build the system event
    pub fn build(self) -> Result<SystemEvent> {
        let event_type = self.event_type.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "event_type".to_string() }
        })?;
        
        let severity = self.severity.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "severity".to_string() }
        })?;
        
        let title = self.title.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "title".to_string() }
        })?;
        
        let message = self.message.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "message".to_string() }
        })?;
        
        let source = self.source.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "source".to_string() }
        })?;

        let mut event = SystemEvent::new(event_type, severity, title, message, source);
        event.tags = self.tags;
        event.data = self.data;
        
        Ok(event)
    }
}

/// Alert builder
pub struct AlertBuilder {
    title: Option<String>,
    message: Option<String>,
    severity: Option<EventSeverity>,
    source: Option<String>,
    category: Option<String>,
    tags: HashMap<String, String>,
    data: HashMap<String, serde_json::Value>,
}

impl AlertBuilder {
    /// Create a new alert builder
    pub fn new() -> Self {
        Self {
            title: None,
            message: None,
            severity: None,
            source: None,
            category: None,
            tags: HashMap::new(),
            data: HashMap::new(),
        }
    }

    /// Set the title
    pub fn title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    /// Set the message
    pub fn message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }

    /// Set the severity
    pub fn severity(mut self, severity: EventSeverity) -> Self {
        self.severity = Some(severity);
        self
    }

    /// Set the source
    pub fn source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the category
    pub fn category(mut self, category: String) -> Self {
        self.category = Some(category);
        self
    }

    /// Add a tag
    pub fn tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Add data
    pub fn data(mut self, key: String, value: serde_json::Value) -> Self {
        self.data.insert(key, value);
        self
    }

    /// Build the alert
    pub fn build(self) -> Result<Alert> {
        let title = self.title.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "title".to_string() }
        })?;
        
        let message = self.message.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "message".to_string() }
        })?;
        
        let severity = self.severity.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "severity".to_string() }
        })?;
        
        let source = self.source.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "source".to_string() }
        })?;
        
        let category = self.category.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "category".to_string() }
        })?;

        let mut alert = Alert::new(title, message, severity, source, category);
        alert.tags = self.tags;
        alert.data = self.data;
        
        Ok(alert)
    }
}

/// Notification builder
pub struct NotificationBuilder {
    title: Option<String>,
    message: Option<String>,
    priority: Option<NotificationPriority>,
    channel: Option<NotificationChannel>,
    recipient: Option<String>,
    tags: HashMap<String, String>,
    data: HashMap<String, serde_json::Value>,
}

impl NotificationBuilder {
    /// Create a new notification builder
    pub fn new() -> Self {
        Self {
            title: None,
            message: None,
            priority: None,
            channel: None,
            recipient: None,
            tags: HashMap::new(),
            data: HashMap::new(),
        }
    }

    /// Set the title
    pub fn title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    /// Set the message
    pub fn message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }

    /// Set the priority
    pub fn priority(mut self, priority: NotificationPriority) -> Self {
        self.priority = Some(priority);
        self
    }

    /// Set the channel
    pub fn channel(mut self, channel: NotificationChannel) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Set the recipient
    pub fn recipient(mut self, recipient: String) -> Self {
        self.recipient = Some(recipient);
        self
    }

    /// Add a tag
    pub fn tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Add data
    pub fn data(mut self, key: String, value: serde_json::Value) -> Self {
        self.data.insert(key, value);
        self
    }

    /// Build the notification
    pub fn build(self) -> Result<Notification> {
        let title = self.title.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "title".to_string() }
        })?;
        
        let message = self.message.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "message".to_string() }
        })?;
        
        let priority = self.priority.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "priority".to_string() }
        })?;
        
        let channel = self.channel.ok_or_else(|| {
            LoggingMonitoringError::MissingRequiredField { field: "channel".to_string() }
        })?;

        let mut notification = Notification::new(title, message, priority, channel);
        if let Some(recipient) = self.recipient {
            notification.recipient = Some(recipient);
        }
        notification.tags = self.tags;
        notification.data = self.data;
        
        Ok(notification)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_manager_creation() {
        let config = EventsConfig::default();
        let manager = EventManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_system_event_emission() {
        let config = EventsConfig::default();
        let manager = EventManager::new(config).await.unwrap();
        
        let event = SystemEvent::new(
            EventType::System,
            EventSeverity::Info,
            "Test Event".to_string(),
            "This is a test event".to_string(),
            "test_service".to_string(),
        );
        
        let result = manager.emit_event(event).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_alert_emission() {
        let config = EventsConfig::default();
        let manager = EventManager::new(config).await.unwrap();
        
        let alert = Alert::new(
            "Test Alert".to_string(),
            "This is a test alert".to_string(),
            EventSeverity::High,
            "test_service".to_string(),
            "test_category".to_string(),
        );
        
        let result = manager.emit_alert(alert).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_notification_emission() {
        let config = EventsConfig::default();
        let manager = EventManager::new(config).await.unwrap();
        
        let notification = Notification::new(
            "Test Notification".to_string(),
            "This is a test notification".to_string(),
            NotificationPriority::Normal,
            NotificationChannel::Email,
        );
        
        let result = manager.emit_notification(notification).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_system_event_builder() {
        let event = SystemEventBuilder::new()
            .event_type(EventType::System)
            .severity(EventSeverity::Info)
            .title("Test Event".to_string())
            .message("Test message".to_string())
            .source("test_service".to_string())
            .tag("test_key".to_string(), "test_value".to_string())
            .build();
        
        assert!(event.is_ok());
    }

    #[tokio::test]
    async fn test_alert_builder() {
        let alert = AlertBuilder::new()
            .title("Test Alert".to_string())
            .message("Test message".to_string())
            .severity(EventSeverity::High)
            .source("test_service".to_string())
            .category("test_category".to_string())
            .tag("test_key".to_string(), "test_value".to_string())
            .build();
        
        assert!(alert.is_ok());
    }

    #[tokio::test]
    async fn test_notification_builder() {
        let notification = NotificationBuilder::new()
            .title("Test Notification".to_string())
            .message("Test message".to_string())
            .priority(NotificationPriority::Normal)
            .channel(NotificationChannel::Email)
            .recipient("test@example.com".to_string())
            .tag("test_key".to_string(), "test_value".to_string())
            .build();
        
        assert!(notification.is_ok());
    }
} 