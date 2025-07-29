use crate::{errors::ConfigurationError, models::ConfigurationChange};
use futures::stream::{self, Stream};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use notify::{Watcher, RecursiveMode, recommended_watcher};
use std::sync::mpsc;

pub type Result<T> = std::result::Result<T, ConfigurationError>;

pub struct ConfigWatcher {
    change_sender: broadcast::Sender<ConfigurationChange>,
    watched_keys: HashMap<String, Value>,
    polling_interval: Duration,
    is_running: bool,
}

impl ConfigWatcher {
    pub async fn new(change_sender: broadcast::Sender<ConfigurationChange>) -> Result<Self> {
        info!("Initializing configuration watcher");
        Ok(Self {
            change_sender,
            watched_keys: HashMap::new(),
            polling_interval: Duration::from_secs(30), // 30 second polling interval
            is_running: false,
        })
    }
    
    pub async fn watch_config(&self, key: &str) -> Result<ConfigWatcher> {
        debug!("Setting up config watcher for key: {}", key);
        
        // Create a new watcher instance for this specific key
        let watcher = ConfigWatcher {
            change_sender: self.change_sender.clone(),
            watched_keys: HashMap::new(),
            polling_interval: self.polling_interval,
            is_running: false,
        };
        
        Ok(watcher)
    }
    
    pub async fn watch_changes(&self) -> Result<impl Stream<Item = Result<ConfigurationChange>>> {
        debug!("Setting up configuration change stream");
        
        // Create a stream that combines file system events and polling
        let file_stream = self.create_file_watcher_stream().await?;
        let polling_stream = self.create_polling_stream().await?;
        
        // Combine both streams
        let combined_stream = stream::select(file_stream, polling_stream);
        
        Ok(combined_stream)
    }
    
    async fn create_file_watcher_stream(&self) -> Result<impl Stream<Item = Result<ConfigurationChange>>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        // Spawn file watcher in background
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            let (notify_tx, notify_rx) = mpsc::channel();
            
            let mut watcher = match recommended_watcher(notify_tx) {
                Ok(w) => w,
                Err(e) => {
                    error!("Failed to create file watcher: {}", e);
                    return;
                }
            };
            
            // Watch configuration directories
            let config_dirs = vec!["./config", "./configs", "/etc/quantumtrade"];
            for dir in config_dirs {
                let path = Path::new(dir);
                if path.exists() {
                    if let Err(e) = watcher.watch(path, RecursiveMode::Recursive) {
                        warn!("Failed to watch directory {}: {}", dir, e);
                    }
                }
            }
            
            // Process file system events
            for event in notify_rx {
                match event {
                    Ok(notify::Event { kind: notify::EventKind::Modify(_), paths, .. }) => {
                        for path in paths {
                            if let Some(file_name) = path.file_name() {
                                let key = file_name.to_string_lossy().to_string();
                                let change = ConfigurationChange {
                                    key: key.clone(),
                                    old_value: None,
                                    new_value: Value::String(format!("File modified: {}", path.display())),
                                    timestamp: chrono::Utc::now(),
                                    source: "file_system".to_string(),
                                    user: None,
                                };
                                
                                if let Err(e) = tx_clone.send(Ok(change)).await {
                                    error!("Failed to send file change notification: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("File watcher error: {}", e);
                    }
                    _ => {}
                }
            }
        });
        
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
    
    async fn create_polling_stream(&self) -> Result<impl Stream<Item = Result<ConfigurationChange>>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let mut interval = interval(self.polling_interval);
        
        // Spawn polling task
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            loop {
                interval.tick().await;
                
                // Poll for configuration changes
                // This would typically check AWS Parameter Store, S3, etc.
                debug!("Polling for configuration changes");
                
                // For now, we'll just send a heartbeat
                let change = ConfigurationChange {
                    key: "polling_heartbeat".to_string(),
                    old_value: None,
                    new_value: Value::String("polling_active".to_string()),
                    timestamp: chrono::Utc::now(),
                    source: "polling".to_string(),
                    user: None,
                };
                
                if let Err(e) = tx_clone.send(Ok(change)).await {
                    error!("Failed to send polling notification: {}", e);
                    break;
                }
            }
        });
        
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
    
    pub async fn notify_change(&self, change: ConfigurationChange) -> Result<()> {
        debug!("Notifying configuration change: {}", change.key);
        
        if let Err(e) = self.change_sender.send(change) {
            error!("Failed to send configuration change: {}", e);
            return Err(ConfigurationError::WatcherError {
                message: format!("Failed to send change notification: {}", e),
            });
        }
        
        Ok(())
    }
    
    pub async fn add_watched_key(&mut self, key: &str, current_value: Value) {
        self.watched_keys.insert(key.to_string(), current_value);
        debug!("Added watched key: {}", key);
    }
    
    pub async fn remove_watched_key(&mut self, key: &str) {
        self.watched_keys.remove(key);
        debug!("Removed watched key: {}", key);
    }
    
    pub async fn get_watched_keys(&self) -> Vec<String> {
        self.watched_keys.keys().cloned().collect()
    }
    
    pub async fn set_polling_interval(&mut self, interval: Duration) {
        self.polling_interval = interval;
        debug!("Set polling interval to {:?}", interval);
    }
    
    pub async fn start(&mut self) -> Result<()> {
        if self.is_running {
            return Ok(());
        }
        
        self.is_running = true;
        info!("Configuration watcher started");
        Ok(())
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        if !self.is_running {
            return Ok(());
        }
        
        self.is_running = false;
        info!("Configuration watcher stopped");
        Ok(())
    }
} 