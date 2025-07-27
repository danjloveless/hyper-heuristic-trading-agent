// services/data-ingestion/src/handlers.rs
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, error, info, warn};

use crate::{AppState, IngestionResponse};
use shared_utils::{QuantumTradeError, Result};

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestionParams {
    pub symbols: Option<String>,     // Comma-separated list
    pub limit: Option<u32>,          // Max records to fetch
    pub force: Option<bool>,         // Skip rate limiting
    pub async_mode: Option<bool>,    // Run in background
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestionStatus {
    pub service: String,
    pub last_run: Option<String>,
    pub records_processed: u64,
    pub errors: Vec<String>,
    pub is_running: bool,
    pub average_latency_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchedulerStatus {
    pub enabled: bool,
    pub is_running: bool,
    pub next_market_data_run: Option<String>,
    pub next_reddit_run: Option<String>,
    pub next_news_run: Option<String>,
    pub last_cleanup: Option<String>,
}

/// Trigger Yahoo Finance data ingestion for all symbols
pub async fn trigger_yahoo_ingestion(
    State(state): State<AppState>,
    Query(params): Query<IngestionParams>,
) -> Result<Json<IngestionResponse>, StatusCode> {
    let start_time = Instant::now();
    info!("Starting Yahoo Finance ingestion with params: {:?}", params);
    
    let symbols = if let Some(symbol_list) = params.symbols {
        symbol_list
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .collect()
    } else {
        state.config.get_all_symbols()
    };
    
    let limit = params.limit.unwrap_or(100);
    
    // Process in batches to avoid overwhelming the API
    let batch_size = state.config.symbols.max_symbols_per_batch as usize;
    let mut total_records = 0u64;
    let mut errors = Vec::new();
    
    for batch in symbols.chunks(batch_size) {
        match process_yahoo_batch(&state, batch.to_vec(), limit).await {
            Ok(count) => {
                total_records += count;
                info!("Processed {} records for batch of {} symbols", count, batch.len());
            }
            Err(e) => {
                let error_msg = format!("Failed to process batch {:?}: {}", batch, e);
                error!("{}", error_msg);
                errors.push(error_msg);
            }
        }
        
        // Small delay between batches to respect rate limits
        if !params.force.unwrap_or(false) {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }
    }
    
    let processing_time = start_time.elapsed().as_millis() as u64;
    
    Ok(Json(IngestionResponse {
        success: errors.is_empty(),
        records_processed: total_records,
        errors,
        processing_time_ms: processing_time,
    }))
}

/// Trigger Yahoo Finance data ingestion for a specific symbol
pub async fn trigger_yahoo_symbol_ingestion(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
    Query(params): Query<IngestionParams>,
) -> Result<Json<IngestionResponse>, StatusCode> {
    let start_time = Instant::now();
    let symbol = symbol.to_uppercase();
    
    info!("Starting Yahoo Finance ingestion for symbol: {}", symbol);
    
    let limit = params.limit.unwrap_or(100);
    
    match process_yahoo_batch(&state, vec![symbol.clone()], limit).await {
        Ok(count) => {
            let processing_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(IngestionResponse {
                success: true,
                records_processed: count,
                errors: vec![],
                processing_time_ms: processing_time,
            }))
        }
        Err(e) => {
            error!("Failed to process symbol {}: {}", symbol, e);
            Ok(Json(IngestionResponse {
                success: false,
                records_processed: 0,
                errors: vec![format!("Failed to process {}: {}", symbol, e)],
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            }))
        }
    }
}

/// Trigger Reddit sentiment data ingestion
pub async fn trigger_reddit_ingestion(
    State(state): State<AppState>,
    Query(params): Query<IngestionParams>,
) -> Result<Json<IngestionResponse>, StatusCode> {
    let start_time = Instant::now();
    info!("Starting Reddit sentiment ingestion with params: {:?}", params);
    
    let symbols = if let Some(symbol_list) = params.symbols {
        symbol_list
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .collect()
    } else {
        state.config.symbols.primary_symbols.clone()
    };
    
    let limit = params.limit.unwrap_or(100);
    
    match process_reddit_batch(&state, symbols, limit).await {
        Ok(count) => {
            let processing_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(IngestionResponse {
                success: true,
                records_processed: count,
                errors: vec![],
                processing_time_ms: processing_time,
            }))
        }
        Err(e) => {
            error!("Failed to process Reddit data: {}", e);
            Ok(Json(IngestionResponse {
                success: false,
                records_processed: 0,
                errors: vec![format!("Failed to process Reddit data: {}", e)],
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            }))
        }
    }
}

/// Trigger Reddit sentiment ingestion for a specific symbol
pub async fn trigger_reddit_symbol_ingestion(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
    Query(params): Query<IngestionParams>,
) -> Result<Json<IngestionResponse>, StatusCode> {
    let start_time = Instant::now();
    let symbol = symbol.to_uppercase();
    
    info!("Starting Reddit sentiment ingestion for symbol: {}", symbol);
    
    let limit = params.limit.unwrap_or(50);
    
    match process_reddit_batch(&state, vec![symbol.clone()], limit).await {
        Ok(count) => {
            let processing_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(IngestionResponse {
                success: true,
                records_processed: count,
                errors: vec![],
                processing_time_ms: processing_time,
            }))
        }
        Err(e) => {
            error!("Failed to process Reddit data for {}: {}", symbol, e);
            Ok(Json(IngestionResponse {
                success: false,
                records_processed: 0,
                errors: vec![format!("Failed to process Reddit data for {}: {}", symbol, e)],
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            }))
        }
    }
}

/// Trigger news sentiment data ingestion
pub async fn trigger_news_ingestion(
    State(state): State<AppState>,
    Query(params): Query<IngestionParams>,
) -> Result<Json<IngestionResponse>, StatusCode> {
    let start_time = Instant::now();
    info!("Starting news sentiment ingestion with params: {:?}", params);
    
    let symbols = if let Some(symbol_list) = params.symbols {
        symbol_list
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .collect()
    } else {
        state.config.symbols.primary_symbols.clone()
    };
    
    let limit = params.limit.unwrap_or(100);
    
    match process_news_batch(&state, symbols, limit).await {
        Ok(count) => {
            let processing_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(IngestionResponse {
                success: true,
                records_processed: count,
                errors: vec![],
                processing_time_ms: processing_time,
            }))
        }
        Err(e) => {
            error!("Failed to process news data: {}", e);
            Ok(Json(IngestionResponse {
                success: false,
                records_processed: 0,
                errors: vec![format!("Failed to process news data: {}", e)],
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            }))
        }
    }
}

/// Trigger news sentiment ingestion for a specific symbol
pub async fn trigger_news_symbol_ingestion(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
    Query(params): Query<IngestionParams>,
) -> Result<Json<IngestionResponse>, StatusCode> {
    let start_time = Instant::now();
    let symbol = symbol.to_uppercase();
    
    info!("Starting news sentiment ingestion for symbol: {}", symbol);
    
    let limit = params.limit.unwrap_or(50);
    
    match process_news_batch(&state, vec![symbol.clone()], limit).await {
        Ok(count) => {
            let processing_time = start_time.elapsed().as_millis() as u64;
            Ok(Json(IngestionResponse {
                success: true,
                records_processed: count,
                errors: vec![],
                processing_time_ms: processing_time,
            }))
        }
        Err(e) => {
            error!("Failed to process news data for {}: {}", symbol, e);
            Ok(Json(IngestionResponse {
                success: false,
                records_processed: 0,
                errors: vec![format!("Failed to process news data for {}: {}", symbol, e)],
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            }))
        }
    }
}

/// Get ingestion status for all services
pub async fn get_ingestion_status(
    State(_state): State<AppState>,
) -> Result<Json<HashMap<String, IngestionStatus>>, StatusCode> {
    let mut status = HashMap::new();
    
    // TODO: Implement actual status tracking
    // For now, return mock data
    status.insert("yahoo_finance".to_string(), IngestionStatus {
        service: "yahoo_finance".to_string(),
        last_run: Some("2025-01-27T10:30:00Z".to_string()),
        records_processed: 1250,
        errors: vec![],
        is_running: false,
        average_latency_ms: 150,
    });
    
    status.insert("reddit".to_string(), IngestionStatus {
        service: "reddit".to_string(),
        last_run: Some("2025-01-27T10:25:00Z".to_string()),
        records_processed: 85,
        errors: vec![],
        is_running: false,
        average_latency_ms: 300,
    });
    
    status.insert("news".to_string(), IngestionStatus {
        service: "news".to_string(),
        last_run: Some("2025-01-27T10:20:00Z".to_string()),
        records_processed: 42,
        errors: vec![],
        is_running: false,
        average_latency_ms: 200,
    });
    
    Ok(Json(status))
}

/// Start the background scheduler
pub async fn start_scheduler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(scheduler) = &state.scheduler {
        match scheduler.start().await {
            Ok(_) => {
                info!("Background scheduler started successfully");
                Ok(Json(serde_json::json!({
                    "success": true,
                    "message": "Scheduler started successfully"
                })))
            }
            Err(e) => {
                error!("Failed to start scheduler: {}", e);
                Ok(Json(serde_json::json!({
                    "success": false,
                    "message": format!("Failed to start scheduler: {}", e)
                })))
            }
        }
    } else {
        warn!("Scheduler is not enabled");
        Ok(Json(serde_json::json!({
            "success": false,
            "message": "Scheduler is not enabled"
        })))
    }
}

/// Stop the background scheduler
pub async fn stop_scheduler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(scheduler) = &state.scheduler {
        scheduler.stop().await;
        info!("Background scheduler stopped");
        Ok(Json(serde_json::json!({
            "success": true,
            "message": "Scheduler stopped successfully"
        })))
    } else {
        Ok(Json(serde_json::json!({
            "success": false,
            "message": "Scheduler is not enabled"
        })))
    }
}

/// Get scheduler status
pub async fn get_scheduler_status(
    State(state): State<AppState>,
) -> Result<Json<SchedulerStatus>, StatusCode> {
    if let Some(_scheduler) = &state.scheduler {
        // TODO: Implement actual scheduler status tracking
        Ok(Json(SchedulerStatus {
            enabled: true,
            is_running: true,
            next_market_data_run: Some("2025-01-27T10:35:00Z".to_string()),
            next_reddit_run: Some("2025-01-27T10:40:00Z".to_string()),
            next_news_run: Some("2025-01-27T11:00:00Z".to_string()),
            last_cleanup: Some("2025-01-27T02:00:00Z".to_string()),
        }))
    } else {
        Ok(Json(SchedulerStatus {
            enabled: false,
            is_running: false,
            next_market_data_run: None,
            next_reddit_run: None,
            next_news_run: None,
            last_cleanup: None,
        }))
    }
}

/// Get current configuration (sanitized)
pub async fn get_config(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Return a sanitized version of the config (no secrets)
    Ok(Json(serde_json::json!({
        "symbols": {
            "primary_symbols": state.config.symbols.primary_symbols,
            "watchlist_symbols": state.config.symbols.watchlist_symbols,
            "crypto_symbols": state.config.symbols.crypto_symbols,
            "max_symbols_per_batch": state.config.symbols.max_symbols_per_batch
        },
        "rate_limits": state.config.rate_limits,
        "scheduler": {
            "market_data_cron": state.config.scheduler.market_data_cron,
            "reddit_sentiment_cron": state.config.scheduler.reddit_sentiment_cron,
            "news_sentiment_cron": state.config.scheduler.news_sentiment_cron,
            "cleanup_cron": state.config.scheduler.cleanup_cron,
            "enable_market_hours_only": state.config.scheduler.enable_market_hours_only,
            "timezone": state.config.scheduler.timezone
        },
        "storage": state.config.storage
    })))
}

// Helper functions for processing batches

async fn process_yahoo_batch(
    state: &AppState,
    symbols: Vec<String>,
    limit: u32,
) -> Result<u64> {
    debug!("Processing Yahoo Finance batch for {} symbols", symbols.len());
    
    let mut all_data = Vec::new();
    
    for symbol in &symbols {
        match state.yahoo_client.get_historical_data(symbol, limit).await {
            Ok(mut data) => {
                debug!("Retrieved {} records for symbol {}", data.len(), symbol);
                all_data.append(&mut data);
            }
            Err(e) => {
                warn!("Failed to get data for symbol {}: {}", symbol, e);
                // Continue with other symbols rather than failing the entire batch
            }
        }
        
        // Small delay between symbols to respect rate limits
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    if !all_data.is_empty() {
        let count = state.clickhouse.insert_market_data_batch(all_data).await?;
        Ok(count)
    } else {
        Ok(0)
    }
}

async fn process_reddit_batch(
    state: &AppState,
    symbols: Vec<String>,
    limit: u32,
) -> Result<u64> {
    debug!("Processing Reddit batch for {} symbols", symbols.len());
    
    let mut all_data = Vec::new();
    
    for symbol in &symbols {
        match state.reddit_client.get_sentiment_data(symbol, limit).await {
            Ok(mut data) => {
                debug!("Retrieved {} Reddit records for symbol {}", data.len(), symbol);
                all_data.append(&mut data);
            }
            Err(e) => {
                warn!("Failed to get Reddit data for symbol {}: {}", symbol, e);
                // Continue with other symbols
            }
        }
        
        // Delay between symbols for rate limiting
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }
    
    if !all_data.is_empty() {
        let count = state.clickhouse.insert_sentiment_data_batch(all_data).await?;
        Ok(count)
    } else {
        Ok(0)
    }
}

async fn process_news_batch(
    state: &AppState,
    symbols: Vec<String>,
    limit: u32,
) -> Result<u64> {
    debug!("Processing news batch for {} symbols", symbols.len());
    
    let mut all_data = Vec::new();
    
    for symbol in &symbols {
        match state.news_client.get_sentiment_data(symbol, limit).await {
            Ok(mut data) => {
                debug!("Retrieved {} news records for symbol {}", data.len(), symbol);
                all_data.append(&mut data);
            }
            Err(e) => {
                warn!("Failed to get news data for symbol {}: {}", symbol, e);
                // Continue with other symbols
            }
        }
        
        // Delay between symbols for rate limiting
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    }
    
    if !all_data.is_empty() {
        let count = state.clickhouse.insert_sentiment_data_batch(all_data).await?;
        Ok(count)
    } else {
        Ok(0)
    }
}