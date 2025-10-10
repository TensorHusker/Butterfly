use crate::node::{NodeId, NodeCapability};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy for partitioning transformer layers across nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Sequential partitioning: consecutive layers on the same node
    Sequential,
    /// Balanced: distribute layers evenly across nodes based on capability
    Balanced,
    /// Custom: user-defined partitioning
    Custom,
}

/// Configuration for partitioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    pub strategy: PartitionStrategy,
    pub num_layers: usize,
    pub num_nodes: usize,
}

/// Represents how a single layer is partitioned
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPartition {
    pub layer_id: usize,
    pub node_id: NodeId,
    /// Start index in the layer (for tensor parallelism)
    pub start_idx: usize,
    /// End index in the layer (for tensor parallelism)
    pub end_idx: usize,
}

/// Manages the partitioning of transformer layers across nodes
pub struct PartitionManager {
    config: PartitionConfig,
    /// Maps layer_id to list of partitions
    partitions: HashMap<usize, Vec<LayerPartition>>,
}

impl PartitionManager {
    pub fn new(config: PartitionConfig) -> Result<Self, PartitionError> {
        if config.num_layers == 0 {
            return Err(PartitionError::InvalidConfig("Number of layers must be positive".to_string()));
        }
        if config.num_nodes == 0 {
            return Err(PartitionError::InvalidConfig("Number of nodes must be positive".to_string()));
        }

        Ok(Self {
            config,
            partitions: HashMap::new(),
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &PartitionConfig {
        &self.config
    }

    /// Get total number of partitioned layers
    pub fn total_partitioned_layers(&self) -> usize {
        self.partitions.len()
    }

    /// Check if all layers are partitioned
    pub fn is_fully_partitioned(&self) -> bool {
        self.partitions.len() == self.config.num_layers
    }

    /// Partition layers across nodes based on their capabilities
    pub fn partition_layers(
        &mut self,
        node_capabilities: &[(NodeId, NodeCapability)],
    ) -> Result<(), PartitionError> {
        if node_capabilities.is_empty() {
            return Err(PartitionError::NoNodes);
        }

        match self.config.strategy {
            PartitionStrategy::Sequential => {
                self.partition_sequential(node_capabilities)
            }
            PartitionStrategy::Balanced => {
                self.partition_balanced(node_capabilities)
            }
            PartitionStrategy::Custom => {
                Ok(()) // Custom partitioning handled externally
            }
        }
    }

    /// Sequential partitioning: distribute consecutive layers to nodes
    fn partition_sequential(
        &mut self,
        node_capabilities: &[(NodeId, NodeCapability)],
    ) -> Result<(), PartitionError> {
        let num_nodes = node_capabilities.len();
        let layers_per_node = (self.config.num_layers + num_nodes - 1) / num_nodes;

        for (node_idx, (node_id, _)) in node_capabilities.iter().enumerate() {
            let start_layer = node_idx * layers_per_node;
            let end_layer = ((node_idx + 1) * layers_per_node).min(self.config.num_layers);

            for layer_id in start_layer..end_layer {
                let partition = LayerPartition {
                    layer_id,
                    node_id: *node_id,
                    start_idx: 0,
                    end_idx: usize::MAX, // Full layer
                };
                
                self.partitions
                    .entry(layer_id)
                    .or_insert_with(Vec::new)
                    .push(partition);
            }
        }

        Ok(())
    }

    /// Balanced partitioning: distribute based on node capabilities
    fn partition_balanced(
        &mut self,
        node_capabilities: &[(NodeId, NodeCapability)],
    ) -> Result<(), PartitionError> {
        // Calculate total performance score
        let total_score: f64 = node_capabilities
            .iter()
            .map(|(_, cap)| cap.performance_score())
            .sum();

        if total_score == 0.0 {
            return Err(PartitionError::InvalidCapabilities);
        }

        // Assign layers based on performance ratio
        let mut current_layer = 0;
        for (node_id, capability) in node_capabilities {
            let ratio = capability.performance_score() / total_score;
            let num_layers = (self.config.num_layers as f64 * ratio).ceil() as usize;
            let end_layer = (current_layer + num_layers).min(self.config.num_layers);

            for layer_id in current_layer..end_layer {
                let partition = LayerPartition {
                    layer_id,
                    node_id: *node_id,
                    start_idx: 0,
                    end_idx: usize::MAX,
                };
                
                self.partitions
                    .entry(layer_id)
                    .or_insert_with(Vec::new)
                    .push(partition);
            }

            current_layer = end_layer;
            if current_layer >= self.config.num_layers {
                break;
            }
        }

        // Handle any remaining layers
        if current_layer < self.config.num_layers {
            let (node_id, _) = &node_capabilities[0];
            for layer_id in current_layer..self.config.num_layers {
                let partition = LayerPartition {
                    layer_id,
                    node_id: *node_id,
                    start_idx: 0,
                    end_idx: usize::MAX,
                };
                
                self.partitions
                    .entry(layer_id)
                    .or_insert_with(Vec::new)
                    .push(partition);
            }
        }

        Ok(())
    }

    /// Get the partition assignment for a specific layer
    pub fn get_layer_partitions(&self, layer_id: usize) -> Option<&Vec<LayerPartition>> {
        self.partitions.get(&layer_id)
    }

    /// Get the node responsible for a specific layer
    pub fn get_layer_node(&self, layer_id: usize) -> Option<NodeId> {
        self.partitions
            .get(&layer_id)
            .and_then(|partitions| partitions.first())
            .map(|p| p.node_id)
    }

    /// Get all partitions
    pub fn all_partitions(&self) -> &HashMap<usize, Vec<LayerPartition>> {
        &self.partitions
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PartitionError {
    #[error("No nodes available for partitioning")]
    NoNodes,
    #[error("Invalid node capabilities")]
    InvalidCapabilities,
    #[error("Layer {0} not found")]
    LayerNotFound(usize),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Partitioning conflict: {0}")]
    PartitionConflict(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeId;

    #[test]
    fn test_sequential_partitioning() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::Sequential,
            num_layers: 12,
            num_nodes: 3,
        };

        let mut manager = PartitionManager::new(config).unwrap();
        
        let nodes = vec![
            (NodeId::new(), NodeCapability {
                memory_gb: 16.0,
                compute_flops: 10e12,
                network_bandwidth_gbps: 10.0,
                num_devices: 1,
                device_type: "GPU".to_string(),
            }),
            (NodeId::new(), NodeCapability {
                memory_gb: 16.0,
                compute_flops: 10e12,
                network_bandwidth_gbps: 10.0,
                num_devices: 1,
                device_type: "GPU".to_string(),
            }),
            (NodeId::new(), NodeCapability {
                memory_gb: 16.0,
                compute_flops: 10e12,
                network_bandwidth_gbps: 10.0,
                num_devices: 1,
                device_type: "GPU".to_string(),
            }),
        ];

        manager.partition_layers(&nodes).unwrap();
        
        // Each node should get 4 layers
        assert!(manager.get_layer_partitions(0).is_some());
        assert!(manager.get_layer_partitions(11).is_some());
    }

    #[test]
    fn test_balanced_partitioning() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::Balanced,
            num_layers: 12,
            num_nodes: 2,
        };

        let mut manager = PartitionManager::new(config).unwrap();
        
        let nodes = vec![
            (NodeId::new(), NodeCapability {
                memory_gb: 32.0, // Higher capability
                compute_flops: 20e12,
                network_bandwidth_gbps: 20.0,
                num_devices: 2,
                device_type: "GPU".to_string(),
            }),
            (NodeId::new(), NodeCapability {
                memory_gb: 16.0,
                compute_flops: 10e12,
                network_bandwidth_gbps: 10.0,
                num_devices: 1,
                device_type: "GPU".to_string(),
            }),
        ];

        manager.partition_layers(&nodes).unwrap();
        
        // All layers should be assigned
        for i in 0..12 {
            assert!(manager.get_layer_node(i).is_some());
        }
    }
}
