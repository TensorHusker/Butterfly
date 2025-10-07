use crate::node::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::mpsc;

/// Types of messages exchanged between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Forward pass data
    ForwardData {
        layer_id: usize,
        sequence_length: usize,
    },
    /// Backward pass gradients (for training)
    BackwardGradient {
        layer_id: usize,
    },
    /// Health check ping
    HealthCheck,
    /// Acknowledgment
    Ack,
    /// Node failure notification
    NodeFailure {
        failed_node_id: NodeId,
    },
    /// Request for layer computation
    ComputeRequest {
        layer_id: usize,
    },
    /// Response with computed results
    ComputeResponse {
        layer_id: usize,
    },
}

/// Message structure for inter-node communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub sender: NodeId,
    pub receiver: NodeId,
    pub msg_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

impl Message {
    pub fn new(sender: NodeId, receiver: NodeId, msg_type: MessageType, payload: Vec<u8>) -> Self {
        Self {
            sender,
            receiver,
            msg_type,
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

/// Communication layer for managing message passing between nodes
pub struct CommunicationLayer {
    node_id: NodeId,
    /// Channel for sending messages
    message_tx: mpsc::UnboundedSender<Message>,
    /// Channel for receiving messages
    message_rx: mpsc::UnboundedReceiver<Message>,
    /// Message handlers for different message types
    handlers: HashMap<String, Box<dyn Fn(&Message) + Send + Sync>>,
}

impl CommunicationLayer {
    pub fn new(node_id: NodeId) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        
        Self {
            node_id,
            message_tx: tx,
            message_rx: rx,
            handlers: HashMap::new(),
        }
    }

    /// Get the node ID for this communication layer
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Send a message to another node
    pub async fn send_message(&self, message: Message) -> Result<(), CommunicationError> {
        self.message_tx
            .send(message)
            .map_err(|_| CommunicationError::SendFailed)?;
        Ok(())
    }

    /// Receive a message (non-blocking)
    pub async fn receive_message(&mut self) -> Option<Message> {
        self.message_rx.recv().await
    }

    /// Broadcast a message to multiple nodes
    pub async fn broadcast(
        &self,
        receivers: &[NodeId],
        msg_type: MessageType,
        payload: Vec<u8>,
    ) -> Result<(), CommunicationError> {
        for receiver in receivers {
            let message = Message::new(self.node_id, *receiver, msg_type.clone(), payload.clone());
            self.send_message(message).await?;
        }
        Ok(())
    }

    /// Get a sender for this communication layer
    pub fn get_sender(&self) -> mpsc::UnboundedSender<Message> {
        self.message_tx.clone()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CommunicationError {
    #[error("Failed to send message")]
    SendFailed,
    #[error("Failed to receive message")]
    ReceiveFailed,
    #[error("Message serialization failed")]
    SerializationFailed,
    #[error("Message deserialization failed")]
    DeserializationFailed,
    #[error("Network error: {0}")]
    NetworkError(String),
}

/// Manages communication between multiple nodes
pub struct NetworkManager {
    /// Communication layers for each node
    comm_layers: HashMap<NodeId, CommunicationLayer>,
}

impl NetworkManager {
    pub fn new() -> Self {
        Self {
            comm_layers: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node_id: NodeId) -> mpsc::UnboundedSender<Message> {
        let comm_layer = CommunicationLayer::new(node_id);
        let sender = comm_layer.get_sender();
        self.comm_layers.insert(node_id, comm_layer);
        sender
    }

    pub fn get_comm_layer(&self, node_id: &NodeId) -> Option<&CommunicationLayer> {
        self.comm_layers.get(node_id)
    }

    pub fn get_comm_layer_mut(&mut self, node_id: &NodeId) -> Option<&mut CommunicationLayer> {
        self.comm_layers.get_mut(node_id)
    }

    /// Route a message from one node to another
    pub async fn route_message(&self, message: Message) -> Result<(), CommunicationError> {
        if let Some(comm_layer) = self.comm_layers.get(&message.receiver) {
            comm_layer.send_message(message).await?;
        }
        Ok(())
    }
}

impl Default for NetworkManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_message_creation() {
        let sender = NodeId::new();
        let receiver = NodeId::new();
        let msg_type = MessageType::HealthCheck;
        let payload = vec![1, 2, 3];

        let message = Message::new(sender, receiver, msg_type, payload.clone());
        
        assert_eq!(message.sender, sender);
        assert_eq!(message.receiver, receiver);
        assert_eq!(message.payload, payload);
    }

    #[tokio::test]
    async fn test_communication_layer() {
        let node_id = NodeId::new();
        let mut comm_layer = CommunicationLayer::new(node_id);

        let receiver = NodeId::new();
        let message = Message::new(
            node_id,
            receiver,
            MessageType::HealthCheck,
            vec![],
        );

        comm_layer.send_message(message).await.unwrap();
        let received = comm_layer.receive_message().await;
        assert!(received.is_some());
    }

    #[tokio::test]
    async fn test_network_manager() {
        let mut manager = NetworkManager::new();
        
        let node1 = NodeId::new();
        let node2 = NodeId::new();
        
        manager.add_node(node1);
        manager.add_node(node2);
        
        assert!(manager.get_comm_layer(&node1).is_some());
        assert!(manager.get_comm_layer(&node2).is_some());
    }
}
