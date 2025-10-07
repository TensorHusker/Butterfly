use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for a node in the distributed system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// Hardware capabilities of a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapability {
    /// Available memory in GB
    pub memory_gb: f64,
    /// Compute capability (FLOPS)
    pub compute_flops: f64,
    /// Network bandwidth in Gbps
    pub network_bandwidth_gbps: f64,
    /// Number of compute devices (GPUs/accelerators)
    pub num_devices: usize,
    /// Device type (e.g., "GPU", "TPU", "CPU")
    pub device_type: String,
}

impl NodeCapability {
    /// Calculate a relative performance score for load balancing
    pub fn performance_score(&self) -> f64 {
        // Weighted score based on memory, compute, and bandwidth
        let memory_weight = 0.3;
        let compute_weight = 0.5;
        let bandwidth_weight = 0.2;
        
        (self.memory_gb * memory_weight) + 
        (self.compute_flops / 1e12 * compute_weight) + 
        (self.network_bandwidth_gbps * bandwidth_weight)
    }
}

/// Information about a node in the distributed system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: NodeId,
    pub address: String,
    pub capability: NodeCapability,
    pub is_active: bool,
}

/// Represents a compute node in the distributed system
#[derive(Debug)]
pub struct Node {
    pub info: NodeInfo,
    /// Assigned layer partitions
    pub assigned_layers: Vec<usize>,
}

impl Node {
    pub fn new(address: String, capability: NodeCapability) -> Self {
        Self {
            info: NodeInfo {
                id: NodeId::new(),
                address,
                capability,
                is_active: true,
            },
            assigned_layers: Vec::new(),
        }
    }

    pub fn assign_layer(&mut self, layer_id: usize) {
        if !self.assigned_layers.contains(&layer_id) {
            self.assigned_layers.push(layer_id);
        }
    }

    pub fn remove_layer(&mut self, layer_id: usize) {
        self.assigned_layers.retain(|&id| id != layer_id);
    }
}

/// Manages all nodes in the distributed system
#[derive(Debug)]
pub struct NodeRegistry {
    nodes: HashMap<NodeId, Node>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn register_node(&mut self, node: Node) -> NodeId {
        let id = node.info.id;
        self.nodes.insert(id, node);
        id
    }

    pub fn get_node(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn get_node_mut(&mut self, id: &NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    pub fn remove_node(&mut self, id: &NodeId) -> Option<Node> {
        self.nodes.remove(id)
    }

    pub fn active_nodes(&self) -> Vec<&Node> {
        self.nodes
            .values()
            .filter(|n| n.info.is_active)
            .collect()
    }

    pub fn all_nodes(&self) -> Vec<&Node> {
        self.nodes.values().collect()
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let capability = NodeCapability {
            memory_gb: 16.0,
            compute_flops: 10e12,
            network_bandwidth_gbps: 10.0,
            num_devices: 1,
            device_type: "GPU".to_string(),
        };
        
        let node = Node::new("127.0.0.1:8080".to_string(), capability);
        assert_eq!(node.info.address, "127.0.0.1:8080");
        assert!(node.info.is_active);
        assert!(node.assigned_layers.is_empty());
    }

    #[test]
    fn test_node_layer_assignment() {
        let capability = NodeCapability {
            memory_gb: 16.0,
            compute_flops: 10e12,
            network_bandwidth_gbps: 10.0,
            num_devices: 1,
            device_type: "GPU".to_string(),
        };
        
        let mut node = Node::new("127.0.0.1:8080".to_string(), capability);
        node.assign_layer(0);
        node.assign_layer(1);
        node.assign_layer(0); // Duplicate should be ignored
        
        assert_eq!(node.assigned_layers.len(), 2);
        assert!(node.assigned_layers.contains(&0));
        assert!(node.assigned_layers.contains(&1));
    }

    #[test]
    fn test_node_registry() {
        let mut registry = NodeRegistry::new();
        
        let capability = NodeCapability {
            memory_gb: 16.0,
            compute_flops: 10e12,
            network_bandwidth_gbps: 10.0,
            num_devices: 1,
            device_type: "GPU".to_string(),
        };
        
        let node = Node::new("127.0.0.1:8080".to_string(), capability);
        let id = registry.register_node(node);
        
        assert!(registry.get_node(&id).is_some());
        assert_eq!(registry.active_nodes().len(), 1);
    }
}
