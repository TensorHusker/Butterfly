use crate::node::{NodeId, NodeCapability, NodeRegistry};
use crate::partition::{PartitionManager, PartitionConfig};
use std::collections::HashMap;

/// Load balancer for distributing work across heterogeneous hardware
pub struct LoadBalancer {
    /// Current load on each node (0.0 to 1.0)
    node_loads: HashMap<NodeId, f64>,
    /// Node capabilities
    node_capabilities: HashMap<NodeId, NodeCapability>,
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            node_loads: HashMap::new(),
            node_capabilities: HashMap::new(),
        }
    }

    /// Register a node with its capabilities
    pub fn register_node(&mut self, node_id: NodeId, capability: NodeCapability) {
        self.node_capabilities.insert(node_id, capability);
        self.node_loads.insert(node_id, 0.0);
    }

    /// Update the current load on a node
    pub fn update_node_load(&mut self, node_id: &NodeId, load: f64) {
        if let Some(current_load) = self.node_loads.get_mut(node_id) {
            *current_load = load.clamp(0.0, 1.0);
        }
    }

    /// Get the current load on a node
    pub fn get_node_load(&self, node_id: &NodeId) -> Option<f64> {
        self.node_loads.get(node_id).copied()
    }

    /// Calculate effective capacity considering current load
    pub fn effective_capacity(&self, node_id: &NodeId) -> f64 {
        if let (Some(capability), Some(load)) = (
            self.node_capabilities.get(node_id),
            self.node_loads.get(node_id),
        ) {
            let base_score = capability.performance_score();
            // Reduce capacity based on current load
            base_score * (1.0 - load)
        } else {
            0.0
        }
    }

    /// Find the node with the most available capacity
    pub fn find_least_loaded_node(&self) -> Option<NodeId> {
        self.node_loads
            .iter()
            .max_by(|(id_a, load_a), (id_b, load_b)| {
                let capacity_a = self.effective_capacity(id_a);
                let capacity_b = self.effective_capacity(id_b);
                capacity_a.partial_cmp(&capacity_b).unwrap()
            })
            .map(|(id, _)| *id)
    }

    /// Balance the load across all nodes
    pub fn balance_load(
        &mut self,
        partition_manager: &mut PartitionManager,
        node_registry: &mut NodeRegistry,
    ) -> Result<(), LoadBalancerError> {
        // Get all nodes sorted by effective capacity
        let mut node_capacities: Vec<_> = self.node_loads
            .keys()
            .map(|id| (*id, self.effective_capacity(id)))
            .collect();
        
        node_capacities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if node_capacities.is_empty() {
            return Err(LoadBalancerError::NoNodes);
        }

        // Calculate average load
        let total_load: f64 = self.node_loads.values().sum();
        let avg_load = total_load / self.node_loads.len() as f64;

        // Identify overloaded and underloaded nodes
        let overloaded: Vec<_> = node_capacities
            .iter()
            .filter(|(id, _)| self.node_loads.get(id).unwrap_or(&0.0) > &(avg_load * 1.2))
            .map(|(id, _)| *id)
            .collect();

        let underloaded: Vec<_> = node_capacities
            .iter()
            .filter(|(id, _)| self.node_loads.get(id).unwrap_or(&0.0) < &(avg_load * 0.8))
            .map(|(id, _)| *id)
            .collect();

        // Migrate work from overloaded to underloaded nodes
        for overloaded_node in overloaded {
            if underloaded.is_empty() {
                break;
            }

            if let Some(node) = node_registry.get_node(&overloaded_node) {
                let layers = node.assigned_layers.clone();
                
                // Move some layers to underloaded nodes
                for (idx, layer_id) in layers.iter().enumerate() {
                    if idx % 2 == 0 && idx / 2 < underloaded.len() {
                        let target_node = underloaded[idx / 2];
                        
                        // Remove from overloaded node
                        if let Some(src_node) = node_registry.get_node_mut(&overloaded_node) {
                            src_node.remove_layer(*layer_id);
                        }
                        
                        // Add to underloaded node
                        if let Some(dst_node) = node_registry.get_node_mut(&target_node) {
                            dst_node.assign_layer(*layer_id);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute optimal work distribution based on node capabilities
    pub fn compute_optimal_distribution(
        &self,
        total_work: usize,
    ) -> HashMap<NodeId, usize> {
        let mut distribution = HashMap::new();

        // Calculate total capacity
        let total_capacity: f64 = self.node_capabilities
            .iter()
            .map(|(_, cap)| cap.performance_score())
            .sum();

        if total_capacity == 0.0 {
            return distribution;
        }

        // Distribute work proportionally to capacity
        let mut remaining_work = total_work;
        for (node_id, capability) in &self.node_capabilities {
            let ratio = capability.performance_score() / total_capacity;
            let work_amount = (total_work as f64 * ratio).round() as usize;
            let allocated = work_amount.min(remaining_work);
            
            distribution.insert(*node_id, allocated);
            remaining_work -= allocated;
        }

        // Distribute any remaining work to the most capable node
        if remaining_work > 0 {
            if let Some(best_node) = self.find_least_loaded_node() {
                *distribution.entry(best_node).or_insert(0) += remaining_work;
            }
        }

        distribution
    }

    /// Get statistics about current load distribution
    pub fn get_load_statistics(&self) -> LoadStatistics {
        let loads: Vec<f64> = self.node_loads.values().copied().collect();
        
        let min_load = loads.iter().copied().fold(f64::INFINITY, f64::min);
        let max_load = loads.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let avg_load = if loads.is_empty() {
            0.0
        } else {
            loads.iter().sum::<f64>() / loads.len() as f64
        };

        LoadStatistics {
            min_load,
            max_load,
            avg_load,
            num_nodes: self.node_loads.len(),
        }
    }
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about load distribution
#[derive(Debug, Clone)]
pub struct LoadStatistics {
    pub min_load: f64,
    pub max_load: f64,
    pub avg_load: f64,
    pub num_nodes: usize,
}

impl LoadStatistics {
    /// Calculate the load imbalance factor (closer to 0 is better)
    pub fn imbalance_factor(&self) -> f64 {
        if self.avg_load == 0.0 {
            0.0
        } else {
            (self.max_load - self.min_load) / self.avg_load
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoadBalancerError {
    #[error("No nodes available")]
    NoNodes,
    #[error("Invalid load value")]
    InvalidLoad,
    #[error("Rebalancing failed: {0}")]
    RebalancingFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_balancer_registration() {
        let mut lb = LoadBalancer::new();
        
        let node_id = NodeId::new();
        let capability = NodeCapability {
            memory_gb: 16.0,
            compute_flops: 10e12,
            network_bandwidth_gbps: 10.0,
            num_devices: 1,
            device_type: "GPU".to_string(),
        };
        
        lb.register_node(node_id, capability);
        
        assert_eq!(lb.get_node_load(&node_id), Some(0.0));
    }

    #[test]
    fn test_load_update() {
        let mut lb = LoadBalancer::new();
        
        let node_id = NodeId::new();
        let capability = NodeCapability {
            memory_gb: 16.0,
            compute_flops: 10e12,
            network_bandwidth_gbps: 10.0,
            num_devices: 1,
            device_type: "GPU".to_string(),
        };
        
        lb.register_node(node_id, capability);
        lb.update_node_load(&node_id, 0.5);
        
        assert_eq!(lb.get_node_load(&node_id), Some(0.5));
    }

    #[test]
    fn test_optimal_distribution() {
        let mut lb = LoadBalancer::new();
        
        let node1 = NodeId::new();
        let node2 = NodeId::new();
        
        let cap1 = NodeCapability {
            memory_gb: 32.0,
            compute_flops: 20e12,
            network_bandwidth_gbps: 20.0,
            num_devices: 2,
            device_type: "GPU".to_string(),
        };
        
        let cap2 = NodeCapability {
            memory_gb: 16.0,
            compute_flops: 10e12,
            network_bandwidth_gbps: 10.0,
            num_devices: 1,
            device_type: "GPU".to_string(),
        };
        
        lb.register_node(node1, cap1);
        lb.register_node(node2, cap2);
        
        let distribution = lb.compute_optimal_distribution(100);
        
        assert_eq!(distribution.len(), 2);
        // Node1 should get more work due to higher capability
        assert!(distribution.get(&node1).unwrap_or(&0) > distribution.get(&node2).unwrap_or(&0));
    }

    #[test]
    fn test_load_statistics() {
        let mut lb = LoadBalancer::new();
        
        let node1 = NodeId::new();
        let node2 = NodeId::new();
        
        let capability = NodeCapability {
            memory_gb: 16.0,
            compute_flops: 10e12,
            network_bandwidth_gbps: 10.0,
            num_devices: 1,
            device_type: "GPU".to_string(),
        };
        
        lb.register_node(node1, capability.clone());
        lb.register_node(node2, capability);
        
        lb.update_node_load(&node1, 0.3);
        lb.update_node_load(&node2, 0.7);
        
        let stats = lb.get_load_statistics();
        
        assert_eq!(stats.min_load, 0.3);
        assert_eq!(stats.max_load, 0.7);
        assert_eq!(stats.avg_load, 0.5);
        assert_eq!(stats.num_nodes, 2);
    }
}
