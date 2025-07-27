// services/data-ingestion/src/main.rs
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    timeout::TimeoutLayer,
};
use tracing::{error, info, warn};
use std::time::Duration;

mod config;
mod handlers;
mod scheduler;
mod yahoo_client;
mod reddit_client;
mod news_client;

use config::DataIngestionConfig;
use handlers::*;
use scheduler::DataIngestionScheduler;
use shared_utils::{ClickHouseClient, QuantumTradeError, Result};

#[derive(Parser)]
#[command(name = "data-ingestion")]
#[command(about = "QuantumTrade AI Data Ingestion Service")]
struct Cli {
    #[arg(long, env = "CONFIG_PATH", default_value = "config/data-ingestion.toml")]
    config: String,
    
    #[arg(long, env = "PORT", default_value = "3001")]
    port: u16,
    
    #[arg(long, env = "ENABLE_SCHEDULER", default_value = "true")]
    enable_scheduler: bool,
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub config: DataIngestionConfig,
    pub clickhouse: ClickHouseClient,
    pub yahoo_client: yahoo_client::YahooFinanceClient,
    pub reddit_client: reddit_client::RedditClient,
    pub news_client: news_client::NewsClient,
    pub scheduler: Option<Arc<DataIngestionScheduler>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub services: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestionResponse {
    pub success: bool,
    pub records_processed: u64,
    pub errors: Vec<String>,
    pub processing_time_ms: u64,
}

/// Health check endpoint
async fn health_check(State(state): State<AppState>) -> Result<Json<HealthResponse>, StatusCode> {
    let mut services = HashMap::new();
    
    // Test ClickHouse connection
    match state.clickhouse.test_connection().await {
        Ok(_) => services.insert("clickhouse".to_string(), "healthy".to_string()),
        Err(_) => services.insert("clickhouse".to_string(), "unhealthy".to_string()),
    };
    
    // Test external API clients
    services.insert("yahoo_finance".to_string(), "configured".to_string());
    services.insert("reddit".to_string(), "configured".to_string());
    services.insert("news".to_string(), "configured".to_string());
    
    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // TODO: Implement actual uptime tracking
        services,
    }))
}

/// Root route
async fn root() -> &'static str {
    "QuantumTrade AI Data Ingestion Service v0.1.0"
}

fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health_check))
        
        // Internal API routes for service-to-service communication
        .route("/api/v1/internal/ingest/yahoo", post(trigger_yahoo_ingestion))
        .route("/api/v1/internal/ingest/yahoo/:symbol", post(trigger_yahoo_symbol_ingestion))
        .route("/api/v1/internal/ingest/reddit", post(trigger_reddit_ingestion))
        .route("/api/v1/internal/ingest/reddit/:symbol", post(trigger_reddit_symbol_ingestion))
        .route("/api/v1/internal/ingest/news", post(trigger_news_ingestion))
        .route("/api/v1/internal/ingest/news/:symbol", post(trigger_news_symbol_ingestion))
        .route("/api/v1/internal/ingest/status", get(get_ingestion_status))
        
        // Admin routes for management
        .route("/api/v1/admin/scheduler/start", post(start_scheduler))
        .route("/api/v1/admin/scheduler/stop", post(stop_scheduler))
        .route("/api/v1/admin/scheduler/status", get(get_scheduler_status))
        .route("/api/v1/admin/config", get(get_config))
        
        .with_state(state)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
                .layer(TimeoutLayer::new(Duration::from_secs(60)))
        )
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Shutdown signal received, starting graceful shutdown");
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables
    dotenvy::dotenv().ok();
    
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "data_ingestion=debug,shared_utils=debug,info".into())
        )
        .json()
        .init();

    let cli = Cli::parse();
    
    info!("Starting QuantumTrade AI Data Ingestion Service v{}", env!("CARGO_PKG_VERSION"));
    info!("Loading configuration from: {}", cli.config);
    
    // Load configuration
    let config = DataIngestionConfig::from_file(&cli.config)
        .map_err(|e| {
            error!("Failed to load configuration: {}", e);
            std::process::exit(1);
        })?;
    
    info!("Configuration loaded successfully");
    
    // Initialize ClickHouse client
    let clickhouse = ClickHouseClient::from_env().await
        .map_err(|e| {
            error!("Failed to initialize ClickHouse client: {}", e);
            std::process::exit(1);
        })?;
    
    info!("ClickHouse client initialized");
    
    // Initialize external API clients
    let yahoo_client = yahoo_client::YahooFinanceClient::new(&config.yahoo_finance)
        .map_err(|e| {
            error!("Failed to initialize Yahoo Finance client: {}", e);
            std::process::exit(1);
        })?;
    
    let reddit_client = reddit_client::RedditClient::new(&config.reddit)
        .await
        .map_err(|e| {
            error!("Failed to initialize Reddit client: {}", e);
            std::process::exit(1);
        })?;
    
    let news_client = news_client::NewsClient::new(&config.news)
        .map_err(|e| {
            error!("Failed to initialize News client: {}", e);
            std::process::exit(1);
        })?;
    
    info!("External API clients initialized");
    
    // Initialize scheduler if enabled
    let scheduler = if cli.enable_scheduler {
        let scheduler = DataIngestionScheduler::new(
            clickhouse.clone(),
            yahoo_client.clone(),
            reddit_client.clone(),
            news_client.clone(),
            config.clone(),
        );
        
        scheduler.start().await?;
        info!("Background scheduler started");
        Some(Arc::new(scheduler))
    } else {
        warn!("Background scheduler disabled");
        None
    };
    
    // Create application state
    let app_state = AppState {
        config,
        clickhouse,
        yahoo_client,
        reddit_client,
        news_client,
        scheduler,
    };
    
    // Create router
    let app = create_router(app_state);
    
    // Start server
    let address = format!("0.0.0.0:{}", cli.port);
    let listener = TcpListener::bind(&address).await
        .map_err(|e| {
            error!("Failed to bind to address {}: {}", address, e);
            std::process::exit(1);
        })?;
    
    info!("🚀 Data Ingestion Service listening on {}", address);
    info!("Health check: http://{}/health", address);
    
    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| {
            error!("Server error: {}", e);
            QuantumTradeError::Internal(e.into())
        })?;
    
    info!("Data Ingestion Service shut down gracefully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    
    #[tokio::test]
    async fn test_health_endpoint() {
        // This is a basic test - in a real environment you'd mock the dependencies
        let config = DataIngestionConfig::default();
        
        // For testing, we'll create a minimal app state
        // In practice, you'd use test containers or mocks
        println!("Health endpoint test would go here");
        // TODO: Implement proper integration tests with testcontainers
    }
}