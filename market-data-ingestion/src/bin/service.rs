// market-data-ingestion/src/bin/service.rs
// Standalone service binary for Market Data Ingestion

use market_data_ingestion::{
    MarketDataIngestionService, MarketDataIngestionConfig,
    Interval, HealthStatus, IngestionMetrics,
};
use database_abstraction::{DatabaseClient, DatabaseManager, DatabaseConfig};
use shared_types;
use dotenv;
use async_trait;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::{Arg, ArgMatches, Command};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    timeout::TimeoutLayer,
};
use tracing::{info, warn, error, debug};
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    ingestion_service: Arc<MarketDataIngestionService>,
    config: MarketDataIngestionConfig,
    start_time: std::time::Instant,
}

// API request/response types
#[derive(Debug, Deserialize)]
struct CollectionRequest {
    symbol: String,
    interval: String,
    #[serde(default)]
    force: bool,
}

#[derive(Debug, Serialize)]
struct CollectionResponse {
    success: bool,
    message: String,
    data_points_collected: usize,
    processing_time_ms: u64,
    batch_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    skip_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latest_timestamp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    force_collection: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct CollectionQuery {
    #[serde(default)]
    symbols: Option<String>, // Comma-separated symbols
    #[serde(default)]
    interval: Option<String>,
}

#[derive(Debug, Serialize)]
struct ServiceInfo {
    service_name: String,
    version: String,
    uptime_seconds: u64,
    config_summary: ConfigSummary,
}

#[derive(Debug, Serialize)]
struct ConfigSummary {
    max_concurrent_collections: usize,
    rate_limit_per_minute: u32,
    quality_threshold: u8,
    default_symbols: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ApiError {
    error: String,
    code: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Parse command line arguments
    let matches = create_cli().get_matches();
    
    // Initialize logging
    setup_logging(&matches)?;
    
    info!("🚀 Starting QuantumTrade AI Market Data Ingestion Service");
    
    // Load configuration
    let config = load_configuration(&matches)?;
    info!("Configuration loaded successfully");
    
    // Validate environment
    validate_environment(&config)?;
    
    // Initialize database connection
    let database = initialize_database(&config).await?;
    info!("Database connection established");
    
    // Create ingestion service
    let ingestion_service = Arc::new(
        MarketDataIngestionService::new(config.clone(), database).await?
    );
    info!("Market Data Ingestion Service initialized");
    
    // Create application state
    let app_state = AppState {
        ingestion_service: ingestion_service.clone(),
        config: config.clone(),
        start_time: std::time::Instant::now(),
    };
    
    // Start background services
    start_background_services(ingestion_service.clone()).await?;
    
    // Create and start HTTP server
    let app = create_router(app_state);
    let port = config.service.port;
    
    info!("Starting HTTP server on port {}", port);
    
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    
    // Start server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    
    info!("Service shutdown complete");
    Ok(())
}

fn create_cli() -> Command {
    Command::new("market-data-service")
        .version("1.0.0")
        .about("QuantumTrade AI Market Data Ingestion Service")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .default_value("config.toml")
        )
        .arg(
            Arg::new("log-level")
                .short('l')
                .long("log-level")
                .value_name("LEVEL")
                .help("Log level (trace, debug, info, warn, error)")
                .default_value("info")
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("HTTP server port")
                .value_parser(clap::value_parser!(u16))
        )
        .arg(
            Arg::new("dry-run")
                .long("dry-run")
                .help("Run in dry-run mode (no actual API calls)")
                .action(clap::ArgAction::SetTrue)
        )
}

fn setup_logging(matches: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let log_level = matches.get_one::<String>("log-level").unwrap();
    
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("market_data_ingestion={},service={}", log_level, log_level)));
    
    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .init();
    
    Ok(())
}

fn load_configuration(matches: &ArgMatches) -> Result<MarketDataIngestionConfig, Box<dyn std::error::Error>> {
    let config_file = matches.get_one::<String>("config").unwrap();
    
    let mut config = if std::path::Path::new(config_file).exists() {
        let config_content = std::fs::read_to_string(config_file)?;
        toml::from_str(&config_content)?
    } else {
        warn!("Config file {} not found, using defaults", config_file);
        MarketDataIngestionConfig::default()
    };
    
    // Override with command line arguments
    if let Some(&port) = matches.get_one::<u16>("port") {
        config.service.port = port;
    }
    
    Ok(config)
}

fn validate_environment(config: &MarketDataIngestionConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Check for required environment variables
    if config.alpha_vantage.api_key.is_empty() || config.alpha_vantage.api_key == "your_api_key_here" {
        return Err("ALPHA_VANTAGE_API_KEY environment variable not set or invalid".into());
    }
    
    // Validate configuration values
    if config.service.port == 0 {
        return Err("Invalid port configuration".into());
    }
    
    if config.rate_limits.calls_per_minute == 0 || config.rate_limits.calls_per_day == 0 {
        return Err("Invalid rate limit configuration".into());
    }
    
    info!("Environment validation passed");
    Ok(())
}

async fn initialize_database(config: &MarketDataIngestionConfig) -> Result<Arc<dyn DatabaseClient>, Box<dyn std::error::Error>> {
    info!("Initializing database connection...");
    
    // Create database configuration from config file
    let db_config = DatabaseConfig {
        clickhouse: database_abstraction::ClickHouseConfig {
            url: std::env::var("CLICKHOUSE_URL").unwrap_or_else(|_| "http://localhost:8123".to_string()),
            database: std::env::var("CLICKHOUSE_DATABASE").unwrap_or_else(|_| "quantumtrade".to_string()),
            username: std::env::var("CLICKHOUSE_USERNAME").ok(),
            password: std::env::var("CLICKHOUSE_PASSWORD").ok(),
            connection_timeout: std::time::Duration::from_secs(30),
            query_timeout: std::time::Duration::from_secs(60),
            max_connections: 100,
            retry_attempts: 3,
        },
        redis: database_abstraction::RedisConfig {
            url: std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            pool_size: 10,
            connection_timeout: std::time::Duration::from_secs(30),
            default_ttl: std::time::Duration::from_secs(3600),
            max_connections: 100,
        },
    };
    
    // Initialize database manager
    let db_manager = DatabaseManager::new(db_config).await?;
    
    // Run migrations to ensure schema is up to date
    db_manager.run_migrations().await?;
    
    info!("Database connection established successfully");
    
    // Return the ClickHouse client as the primary database client
    Ok(db_manager.clickhouse())
}

async fn start_background_services(
    ingestion_service: Arc<MarketDataIngestionService>
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting background services...");
    
    // Start health monitoring
    let health_service = ingestion_service.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            let health = health_service.get_health().await;
            debug!("Health check: {:?}", health);
            
            if !health.alpha_vantage_reachable {
                warn!("Alpha Vantage API is unreachable");
            }
            if !health.database_connected {
                error!("Database connection lost");
            }
        }
    });
    
    // Start metrics collection
    let metrics_service = ingestion_service.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            let metrics = metrics_service.get_metrics().await;
            info!("Metrics - Collections: {}, Failures: {}, API Calls: {}", 
                  metrics.collections_completed, 
                  metrics.collections_failed, 
                  metrics.api_calls_made);
        }
    });
    
    info!("Background services started");
    Ok(())
}

fn create_router(app_state: AppState) -> Router {
    Router::new()
        // Health and status endpoints
        .route("/health", get(health_handler))
        .route("/health/detailed", get(detailed_health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/info", get(info_handler))
        
        // Collection endpoints
        .route("/collect/:symbol", post(collect_symbol_handler))
        .route("/collect", post(collect_multiple_handler))
        .route("/collections", get(list_collections_handler))
        
        // Configuration endpoints
        .route("/config", get(config_handler))
        .route("/config/symbols", get(symbols_handler))
        
        // Administrative endpoints
        .route("/admin/trigger-collection", post(trigger_collection_handler))
        .route("/admin/force-collection/:symbol", post(force_collection_handler))
        .route("/freshness/:symbol", get(check_freshness_handler))
        
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                .layer(CorsLayer::permissive())
        )
        .with_state(app_state)
}

// Handler implementations

async fn health_handler(State(state): State<AppState>) -> Result<Json<HealthStatus>, StatusCode> {
    let health = state.ingestion_service.get_health().await;
    Ok(Json(health))
}

async fn detailed_health_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    let health = state.ingestion_service.get_health().await;
    let metrics = state.ingestion_service.get_metrics().await;
    
    let detailed_health = serde_json::json!({
        "health": health,
        "metrics": metrics,
        "uptime_seconds": state.start_time.elapsed().as_secs(),
        "service_info": {
            "name": state.config.service.service_name,
            "version": "1.0.0",
            "worker_threads": state.config.service.worker_threads,
            "max_concurrent_collections": state.config.service.max_concurrent_collections
        }
    });
    
    Ok(Json(detailed_health))
}

async fn metrics_handler(State(state): State<AppState>) -> Result<Json<IngestionMetrics>, StatusCode> {
    let metrics = state.ingestion_service.get_metrics().await;
    Ok(Json(metrics))
}

async fn info_handler(State(state): State<AppState>) -> Result<Json<ServiceInfo>, StatusCode> {
    let info = ServiceInfo {
        service_name: state.config.service.service_name.clone(),
        version: "1.0.0".to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        config_summary: ConfigSummary {
            max_concurrent_collections: state.config.service.max_concurrent_collections,
            rate_limit_per_minute: state.config.rate_limits.calls_per_minute,
            quality_threshold: state.config.data_quality.quality_threshold,
            default_symbols: state.config.collection.default_symbols.clone(),
        },
    };
    
    Ok(Json(info))
}

async fn collect_symbol_handler(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<CollectionResponse>, (StatusCode, Json<ApiError>)> {
    let interval_str = params.get("interval").unwrap_or(&"5min".to_string()).clone();
    let force = params.get("force").unwrap_or(&"false".to_string()) == "true";
    
    let interval = parse_interval(&interval_str)
        .map_err(|e| create_api_error(StatusCode::BAD_REQUEST, "INVALID_INTERVAL", &e))?;
    
    info!("Collecting data for symbol: {} at interval: {:?} (force: {})", symbol, interval, force);
    
    let start_time = std::time::Instant::now();
    
    // Use force collection if requested, otherwise use intelligent collection
    let batch_result = if force {
        state.ingestion_service.force_collect_symbol_data(&symbol, interval).await
    } else {
        state.ingestion_service.collect_symbol_data(&symbol, interval).await
    };
    
    match batch_result {
        Ok(batch) => {
            let processing_time = start_time.elapsed();
            let data_points = batch.size();
            let batch_id = batch.batch_id.clone();
            
            // Check if this was a skip due to fresh data
            let skip_reason = batch.metadata.get("skip_reason").cloned();
            let latest_timestamp = batch.metadata.get("latest_timestamp").cloned();
            let message = if let Some(ref reason) = skip_reason {
                if reason == "data_fresh" {
                    format!("Data for {} is fresh, no new data collected", symbol)
                } else {
                    format!("Collection skipped for {}: {}", symbol, reason)
                }
            } else {
                format!("Successfully collected and processed data for {}", symbol)
            };
            
            // Only process the batch if it contains data
            if data_points > 0 {
                if let Err(e) = state.ingestion_service.process_batch(batch).await {
                    warn!("Failed to process batch: {}", e);
                    return Err(create_api_error(
                        StatusCode::INTERNAL_SERVER_ERROR, 
                        "PROCESSING_FAILED", 
                        &format!("Failed to process collected data: {}", e)
                    ));
                }
            }
            
            Ok(Json(CollectionResponse {
                success: true,
                message,
                data_points_collected: data_points,
                processing_time_ms: processing_time.as_millis() as u64,
                batch_id,
                skip_reason,
                latest_timestamp,
                force_collection: if force { Some(true) } else { None },
            }))
        },
        Err(e) => {
            error!("Collection failed for {}: {}", symbol, e);
            Err(create_api_error(
                StatusCode::BAD_REQUEST, 
                "COLLECTION_FAILED", 
                &format!("Failed to collect data: {}", e)
            ))
        }
    }
}

async fn collect_multiple_handler(
    State(state): State<AppState>,
    Json(request): Json<CollectionRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let symbols: Vec<String> = request.symbol.split(',').map(|s| s.trim().to_string()).collect();
    let interval = parse_interval(&request.interval)
        .map_err(|e| create_api_error(StatusCode::BAD_REQUEST, "INVALID_INTERVAL", &e))?;
    
    info!("Collecting data for {} symbols at interval: {:?}", symbols.len(), interval);
    
    let mut results = Vec::new();
    let start_time = std::time::Instant::now();
    
    for symbol in symbols {
        let symbol_start = std::time::Instant::now();
        
        match state.ingestion_service.force_collect_symbol_data(&symbol, interval).await {
            Ok(batch) => {
                let data_points = batch.size();
                let batch_id = batch.batch_id.clone();
                
                match state.ingestion_service.process_batch(batch).await {
                    Ok(_) => {
                        results.push(serde_json::json!({
                            "symbol": symbol,
                            "success": true,
                            "data_points": data_points,
                            "batch_id": batch_id,
                            "processing_time_ms": symbol_start.elapsed().as_millis()
                        }));
                    },
                    Err(e) => {
                        results.push(serde_json::json!({
                            "symbol": symbol,
                            "success": false,
                            "error": format!("Processing failed: {}", e),
                            "processing_time_ms": symbol_start.elapsed().as_millis()
                        }));
                    }
                }
            },
            Err(e) => {
                results.push(serde_json::json!({
                    "symbol": symbol,
                    "success": false,
                    "error": format!("Collection failed: {}", e),
                    "processing_time_ms": symbol_start.elapsed().as_millis()
                }));
            }
        }
        
        // Small delay between collections to respect rate limits
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    
    let total_time = start_time.elapsed();
    let successful_collections = results.iter().filter(|r| r["success"].as_bool().unwrap_or(false)).count();
    
    Ok(Json(serde_json::json!({
        "summary": {
            "total_symbols": results.len(),
            "successful_collections": successful_collections,
            "failed_collections": results.len() - successful_collections,
            "total_processing_time_ms": total_time.as_millis()
        },
        "results": results
    })))
}

async fn list_collections_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let symbols = &state.config.collection.default_symbols;
    let intervals = &state.config.collection.collection_intervals;
    
    Ok(Json(serde_json::json!({
        "default_symbols": symbols,
        "available_intervals": intervals,
        "priority_symbols": state.config.collection.priority_symbols,
        "max_batch_size": state.config.collection.max_batch_size,
        "parallel_collections": state.config.collection.parallel_collections
    })))
}

async fn config_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    // Return sanitized configuration (without sensitive data)
    let sanitized_config = serde_json::json!({
        "service": {
            "service_name": state.config.service.service_name,
            "port": state.config.service.port,
            "worker_threads": state.config.service.worker_threads,
            "max_concurrent_collections": state.config.service.max_concurrent_collections
        },
        "rate_limits": {
            "calls_per_minute": state.config.rate_limits.calls_per_minute,
            "calls_per_day": state.config.rate_limits.calls_per_day,
            "is_premium": state.config.rate_limits.is_premium
        },
        "collection": state.config.collection,
        "data_quality": state.config.data_quality,
        "storage": state.config.storage
    });
    
    Ok(Json(sanitized_config))
}

async fn symbols_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    Ok(Json(serde_json::json!({
        "default_symbols": state.config.collection.default_symbols,
        "priority_symbols": state.config.collection.priority_symbols,
        "total_symbols": state.config.collection.default_symbols.len() + state.config.collection.priority_symbols.len()
    })))
}

async fn trigger_collection_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    info!("Manual collection trigger requested");
    
    let symbols = state.config.collection.default_symbols.clone();
    let interval = Interval::FiveMin; // Default interval for manual triggers
    
    let mut results = Vec::new();
    
    for symbol in symbols {
        match state.ingestion_service.collect_symbol_data(&symbol, interval).await {
            Ok(batch) => {
                let data_points = batch.size();
                let batch_id = batch.batch_id.clone();
                
                if let Err(e) = state.ingestion_service.process_batch(batch).await {
                    warn!("Failed to process batch for {}: {}", symbol, e);
                    results.push(serde_json::json!({
                        "symbol": symbol,
                        "success": false,
                        "error": format!("Processing failed: {}", e)
                    }));
                } else {
                    results.push(serde_json::json!({
                        "symbol": symbol,
                        "success": true,
                        "data_points": data_points,
                        "batch_id": batch_id
                    }));
                }
            },
            Err(e) => {
                results.push(serde_json::json!({
                    "symbol": symbol,
                    "success": false,
                    "error": format!("Collection failed: {}", e)
                }));
            }
        }
        
        // Delay between collections
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    
    Ok(Json(serde_json::json!({
        "message": "Manual collection completed",
        "results": results
    })))
}

async fn force_collection_handler(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
) -> Result<Json<CollectionResponse>, (StatusCode, Json<ApiError>)> {
    info!("Force collection requested for symbol: {}", symbol);
    
    // Force collection bypasses some rate limiting (in a real implementation)
    let interval = Interval::FiveMin;
    let start_time = std::time::Instant::now();
    
    match state.ingestion_service.force_collect_symbol_data(&symbol, interval).await {
        Ok(batch) => {
            let processing_time = start_time.elapsed();
            let data_points = batch.size();
            let batch_id = batch.batch_id.clone();
            
            if let Err(e) = state.ingestion_service.process_batch(batch).await {
                return Err(create_api_error(
                    StatusCode::INTERNAL_SERVER_ERROR, 
                    "PROCESSING_FAILED", 
                    &format!("Failed to process collected data: {}", e)
                ));
            }
            
            Ok(Json(CollectionResponse {
                success: true,
                message: format!("Force collection completed for {}", symbol),
                data_points_collected: data_points,
                processing_time_ms: processing_time.as_millis() as u64,
                batch_id,
                skip_reason: None,
                latest_timestamp: None,
                force_collection: Some(true),
            }))
        },
        Err(e) => {
            Err(create_api_error(
                StatusCode::BAD_REQUEST, 
                "FORCE_COLLECTION_FAILED", 
                &format!("Force collection failed: {}", e)
            ))
        }
    }
}

async fn check_freshness_handler(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let interval_str = params.get("interval").unwrap_or(&"5min".to_string()).clone();
    let interval = parse_interval(&interval_str)
        .map_err(|e| create_api_error(StatusCode::BAD_REQUEST, "INVALID_INTERVAL", &e))?;
    
    // Get the latest timestamp from the database
    let latest_timestamp = match state.ingestion_service.get_latest_timestamp(&symbol, interval).await {
        Ok(timestamp) => timestamp,
        Err(e) => {
            return Err(create_api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "DATABASE_ERROR",
                &format!("Failed to get latest timestamp: {}", e)
            ));
        }
    };
    
    let now = chrono::Utc::now();
    let is_fresh = if let Some(timestamp) = latest_timestamp {
        state.ingestion_service.is_data_fresh(timestamp, interval)
    } else {
        false // No data means not fresh
    };
    
    let age_minutes = if let Some(timestamp) = latest_timestamp {
        let age = now - timestamp;
        age.num_seconds() / 60
    } else {
        -1 // No data
    };
    
    let response = serde_json::json!({
        "symbol": symbol,
        "interval": interval_str,
        "has_data": latest_timestamp.is_some(),
        "is_fresh": is_fresh,
        "latest_timestamp": latest_timestamp.map(|t| t.to_rfc3339()),
        "age_minutes": age_minutes,
        "checked_at": now.to_rfc3339(),
        "freshness_threshold_minutes": match interval {
            Interval::OneMin => 2,
            Interval::FiveMin => 7,
            Interval::FifteenMin => 20,
            Interval::ThirtyMin => 35,
            Interval::SixtyMin => 65,
        }
    });
    
    Ok(Json(response))
}

// Helper functions

fn parse_interval(interval_str: &str) -> Result<Interval, String> {
    match interval_str.to_lowercase().as_str() {
        "1min" => Ok(Interval::OneMin),
        "5min" => Ok(Interval::FiveMin),
        "15min" => Ok(Interval::FifteenMin),
        "30min" => Ok(Interval::ThirtyMin),
        "60min" | "1hour" => Ok(Interval::SixtyMin),
        _ => Err(format!("Invalid interval: {}. Supported: 1min, 5min, 15min, 30min, 60min", interval_str)),
    }
}

fn create_api_error(status: StatusCode, code: &str, message: &str) -> (StatusCode, Json<ApiError>) {
    (status, Json(ApiError {
        error: message.to_string(),
        code: code.to_string(),
        timestamp: chrono::Utc::now(),
    }))
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, initiating graceful shutdown");
        },
        _ = terminate => {
            info!("Received SIGTERM, initiating graceful shutdown");
        },
    }
}

