//! # Butterfly Core
//!
//! Core types, traits, and abstractions for the Butterfly distributed inference system.
//! This crate provides the fundamental building blocks used across all other components.

pub mod tensor;
pub mod state;

use serde::{Deserialize, Serialize};

pub use tensor::{TensorRef, TensorPool};
pub use state::{
    SystemState, Phase, ComponentState, StateMachine, PersistentStateMachine,
    StateError, StateTransitionRecord, TransitionTrigger,
};

/// Unique identifier for a node in the distributed system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

/// Type of neural network layer
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Embedding layer (vocabulary -> hidden dimension)
    Embedding {
        vocab_size: usize,
        hidden_dim: usize,
    },
    /// Transformer block with attention and feed-forward
    TransformerBlock {
        hidden_dim: usize,
        num_heads: usize,
        ff_dim: usize,
    },
    /// Output projection head
    OutputHead {
        hidden_dim: usize,
        vocab_size: usize,
    },
    /// Generic feed-forward layer
    Linear {
        input_dim: usize,
        output_dim: usize,
    },
    /// Activation function
    Activation,
    /// Normalization layer
    Normalization,
}

/// Information about a single layer in the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub id: usize,
    pub layer_type: LayerType,
    /// Estimated computational cost in FLOPs
    pub compute_cost: f64,
    /// Memory requirement in bytes
    pub memory_bytes: u64,
    /// Tensor size output by this layer (in elements)
    pub output_size: usize,
}

/// Partitioning strategy type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Simple uniform distribution
    Uniform,
    /// Load-balanced based on computational cost
    LoadBalanced,
    /// Topology-aware to minimize communication
    TopologyAware,
    /// Tensor-parallel within layers
    TensorParallel { num_splits: usize },
    /// Pipeline-parallel with micro-batching
    PipelineParallel { num_stages: usize },
}

/// Represents a partition of a neural network model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPartition {
    pub id: u64,
    pub layer_range: (usize, usize),
    pub assigned_node: Option<NodeId>,
    /// Strategy used for this partition
    pub strategy: PartitionStrategy,
    /// Estimated execution time in milliseconds
    pub estimated_time_ms: f64,
}

/// Core trait for nodes that can participate in distributed inference
pub trait InferenceNode {
    /// Execute inference on this node's assigned partition
    fn execute(&self, input: &[f32]) -> Result<Vec<f32>, InferenceError>;

    /// Get the node's unique identifier
    fn node_id(&self) -> NodeId;

    /// Check if the node is ready to process requests
    fn is_ready(&self) -> bool;
}

/// Errors that can occur during distributed inference
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Node not ready: {0}")]
    NodeNotReady(String),

    #[error("Computation failed: {0}")]
    ComputationFailed(String),

    #[error("Invalid input shape")]
    InvalidInput,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_equality() {
        let id1 = NodeId(42);
        let id2 = NodeId(42);
        assert_eq!(id1, id2);
    }
}
