pub mod config_manager;
pub mod config_store;
pub mod config_validator;
pub mod config_watcher;
pub mod errors;
pub mod feature_flags;
pub mod health;
pub mod models;
pub mod secret_manager;
pub mod sources;

pub use config_manager::ConfigurationManager;
pub use errors::ConfigurationError;
pub use models::*;

#[cfg(test)]
mod tests; 