use axum::{
    routing::{get, post},
    Router,
    Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tracing::{info, error};

mod yahoo_client;
mod reddit_client;
mod news_client;

#[derive(Debug, Serialize, Deserialize)]
struct IngestResponse {
    status: String,
    message: String,
    records_processed: u32,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting data ingestion service...");
    
    // Build our application with a route
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/internal/ingest/yahoo", post(ingest_yahoo))
        .route("/api/v1/internal/ingest/reddit", post(ingest_reddit))
        .route("/api/v1/internal/ingest/news", post(ingest_news))
        .route("/api/v1/internal/ingest/status", get(ingest_status));
    
    // Run it
    let addr = SocketAddr::from(([127, 0, 0, 1], 3001));
    info!("Data ingestion service listening on {}", addr);
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn health_check() -> StatusCode {
    StatusCode::OK
}

async fn ingest_yahoo() -> Json<IngestResponse> {
    info!("Starting Yahoo Finance data ingestion");
    
    // TODO: Implement Yahoo Finance data collection
    // This would use the yahoo_client module
    
    Json(IngestResponse {
        status: "success".to_string(),
        message: "Yahoo Finance data ingestion completed".to_string(),
        records_processed: 0,
    })
}

async fn ingest_reddit() -> Json<IngestResponse> {
    info!("Starting Reddit sentiment data ingestion");
    
    // TODO: Implement Reddit sentiment data collection
    // This would use the reddit_client module
    
    Json(IngestResponse {
        status: "success".to_string(),
        message: "Reddit sentiment data ingestion completed".to_string(),
        records_processed: 0,
    })
}

async fn ingest_news() -> Json<IngestResponse> {
    info!("Starting news data ingestion");
    
    // TODO: Implement news data collection
    // This would use the news_client module
    
    Json(IngestResponse {
        status: "success".to_string(),
        message: "News data ingestion completed".to_string(),
        records_processed: 0,
    })
}

async fn ingest_status() -> Json<IngestResponse> {
    Json(IngestResponse {
        status: "running".to_string(),
        message: "Data ingestion service is running".to_string(),
        records_processed: 0,
    })
} 