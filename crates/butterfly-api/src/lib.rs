//! # Butterfly API
//!
//! HTTP API server for the Butterfly distributed inference system.
//! Provides REST endpoints for submitting inference requests and monitoring system status.

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use butterfly_schedule::InferenceTask;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Request body for submitting an inference task
#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub input: Vec<f32>,
    pub priority: Option<u32>,
}

/// Response containing the task ID for tracking
#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub task_id: u64,
    pub status: String,
}

/// System health status
#[derive(Debug, Serialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub active_nodes: usize,
    pub pending_tasks: usize,
}

/// Shared application state
pub struct ApiState {
    task_counter: Arc<Mutex<u64>>,
}

impl ApiState {
    pub fn new() -> Self {
        Self {
            task_counter: Arc::new(Mutex::new(0)),
        }
    }
}

impl Default for ApiState {
    fn default() -> Self {
        Self::new()
    }
}

/// Create the API router with all endpoints
pub fn create_router(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/inference", post(submit_inference))
        .with_state(Arc::new(state))
}

/// Health check endpoint
async fn health_check() -> Json<HealthStatus> {
    Json(HealthStatus {
        healthy: true,
        active_nodes: 0,
        pending_tasks: 0,
    })
}

/// Submit inference request endpoint
async fn submit_inference(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, StatusCode> {
    let mut counter = state.task_counter.lock().await;
    *counter += 1;
    let task_id = *counter;

    Ok(Json(InferenceResponse {
        task_id,
        status: "submitted".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_state_creation() {
        let state = ApiState::new();
        assert!(Arc::strong_count(&state.task_counter) >= 1);
    }
}
