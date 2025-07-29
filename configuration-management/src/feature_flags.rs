use crate::{errors::ConfigurationError, models::FeatureConfig};
use async_trait::async_trait;
use chrono::Utc;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use tracing::{debug, info};

pub type Result<T> = std::result::Result<T, ConfigurationError>;

pub struct FeatureFlags {
    features: HashMap<String, FeatureConfig>,
}

impl FeatureFlags {
    pub async fn new() -> Result<Self> {
        info!("Initializing feature flags");
        Ok(Self {
            features: HashMap::new(),
        })
    }
    
    pub async fn is_enabled(&self, feature_name: &str) -> Result<bool> {
        debug!("Checking if feature is enabled: {}", feature_name);
        
        if let Some(feature) = self.features.get(feature_name) {
            Ok(feature.enabled)
        } else {
            debug!("Feature not found: {}, defaulting to disabled", feature_name);
            Ok(false)
        }
    }
    
    pub async fn get_config<T: DeserializeOwned>(&self, feature_name: &str) -> Result<Option<T>> {
        debug!("Getting feature config: {}", feature_name);
        
        if let Some(feature) = self.features.get(feature_name) {
            if !feature.enabled {
                return Ok(None);
            }
            
            // Convert parameters to the requested type
            let config_value = serde_json::to_value(&feature.parameters)?;
            let config: T = serde_json::from_value(config_value)?;
            Ok(Some(config))
        } else {
            Ok(None)
        }
    }
    
    pub async fn enable_feature(&mut self, feature_name: &str, config: Option<serde_json::Value>) -> Result<()> {
        debug!("Enabling feature: {}", feature_name);
        
        let feature_config = FeatureConfig {
            enabled: true,
            rollout_percentage: 100.0,
            target_groups: Vec::new(),
            parameters: config.unwrap_or_default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        self.features.insert(feature_name.to_string(), feature_config);
        info!("Feature enabled: {}", feature_name);
        Ok(())
    }
    
    pub async fn disable_feature(&mut self, feature_name: &str) -> Result<()> {
        debug!("Disabling feature: {}", feature_name);
        
        if let Some(feature) = self.features.get_mut(feature_name) {
            feature.enabled = false;
            feature.updated_at = Utc::now();
            info!("Feature disabled: {}", feature_name);
        } else {
            return Err(ConfigurationError::FeatureFlagError {
                message: format!("Feature not found: {}", feature_name),
            });
        }
        
        Ok(())
    }
    
    pub async fn set_feature_config(&mut self, feature_name: &str, config: FeatureConfig) -> Result<()> {
        debug!("Setting feature config: {}", feature_name);
        
        self.features.insert(feature_name.to_string(), config);
        info!("Feature config updated: {}", feature_name);
        Ok(())
    }
    
    pub async fn list_features(&self) -> Vec<String> {
        self.features.keys().cloned().collect()
    }
    
    pub async fn get_feature_details(&self, feature_name: &str) -> Option<&FeatureConfig> {
        self.features.get(feature_name)
    }
} 