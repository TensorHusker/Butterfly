//! # Butterfly Communication
//!
//! Network communication layer for the Butterfly distributed inference system.
//! Handles peer-to-peer messaging, tensor transfers, and node coordination.

use butterfly_core::NodeId;
use serde::{Deserialize, Serialize};

/// Message types exchanged between nodes in the distributed system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// Tensor data being passed between nodes
    TensorData {
        from: NodeId,
        to: NodeId,
        data: Vec<f32>,
    },
    /// Heartbeat to check node liveness
    Heartbeat { node_id: NodeId },
    /// Acknowledgment of message receipt
    Ack { message_id: u64 },
}

/// Trait for communication backends that transport messages between nodes
#[async_trait::async_trait]
pub trait CommunicationBackend: Send + Sync {
    /// Send a message to a specific node
    async fn send(&self, target: NodeId, message: Message) -> Result<(), CommunicationError>;

    /// Receive the next available message
    async fn receive(&self) -> Result<Message, CommunicationError>;

    /// Broadcast a message to all known nodes
    async fn broadcast(&self, message: Message) -> Result<(), CommunicationError>;
}

/// Errors that can occur during network communication
#[derive(Debug, thiserror::Error)]
pub enum CommunicationError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("Serialization failed: {0}")]
    Serialization(String),

    #[error("Node unreachable: {0:?}")]
    Unreachable(NodeId),
}

/// Placeholder implementation for testing
pub struct LocalBackend;

#[async_trait::async_trait]
impl CommunicationBackend for LocalBackend {
    async fn send(&self, _target: NodeId, _message: Message) -> Result<(), CommunicationError> {
        Ok(())
    }

    async fn receive(&self) -> Result<Message, CommunicationError> {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        Ok(Message::Heartbeat { node_id: NodeId(0) })
    }

    async fn broadcast(&self, _message: Message) -> Result<(), CommunicationError> {
        Ok(())
    }
}
