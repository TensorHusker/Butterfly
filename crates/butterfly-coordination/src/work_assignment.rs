//! Work assignment algorithm for distributing layers across nodes
//!
//! Implements a topology-aware assignment strategy that:
//! 1. Balances computational load across nodes
//! 2. Minimizes inter-node communication
//! 3. Respects data locality and affinity
//! 4. Creates efficient dependency graphs

use crate::types::{DependencyGraph, LayerAssignment, WorkAssignment};
use butterfly_core::{LayerInfo, NodeId};
use std::collections::HashMap;
use tracing::{debug, info};

/// Strategy for assigning work
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentStrategy {
    /// Simple round-robin distribution
    RoundRobin,
    /// Balance based on computational cost
    LoadBalanced,
    /// Minimize communication between nodes
    TopologyAware,
    /// Hybrid: balance load with topology awareness
    Hybrid,
}

/// Work assigner for distributing transformer layers
pub struct WorkAssigner {
    strategy: AssignmentStrategy,
    /// Affinity scores: how much a node prefers certain layers
    affinity_cache: HashMap<(NodeId, usize), f64>,
}

impl WorkAssigner {
    /// Create new work assigner
    pub fn new() -> Self {
        Self::with_strategy(AssignmentStrategy::Hybrid)
    }

    /// Create with specific strategy
    pub fn with_strategy(strategy: AssignmentStrategy) -> Self {
        Self {
            strategy,
            affinity_cache: HashMap::new(),
        }
    }

    /// Assign layers to nodes
    pub async fn assign(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeId],
    ) -> Result<WorkAssignment, String> {
        if layers.is_empty() {
            return Err("No layers to assign".to_string());
        }

        if nodes.is_empty() {
            return Err("No nodes available".to_string());
        }

        info!(
            num_layers = layers.len(),
            num_nodes = nodes.len(),
            strategy = ?self.strategy,
            "Assigning work to nodes"
        );

        let assignments = match self.strategy {
            AssignmentStrategy::RoundRobin => self.assign_round_robin(layers, nodes),
            AssignmentStrategy::LoadBalanced => self.assign_load_balanced(layers, nodes),
            AssignmentStrategy::TopologyAware => self.assign_topology_aware(layers, nodes),
            AssignmentStrategy::Hybrid => self.assign_hybrid(layers, nodes),
        };

        // Build dependency graph
        let dependencies = self.build_dependency_graph(&assignments, nodes)?;

        Ok(WorkAssignment {
            epoch: 0, // Will be set by coordinator
            assignments,
            dependencies,
        })
    }

    /// Round-robin assignment
    fn assign_round_robin(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeId],
    ) -> HashMap<NodeId, LayerAssignment> {
        let mut assignments: HashMap<NodeId, LayerAssignment> = HashMap::new();

        for (i, layer) in layers.iter().enumerate() {
            let node = nodes[i % nodes.len()];

            let assignment = assignments.entry(node).or_insert_with(|| LayerAssignment {
                node_id: node,
                layer_ids: Vec::new(),
                input_dependencies: Vec::new(),
                output_consumers: Vec::new(),
                estimated_compute_time_ms: 0.0,
            });

            assignment.layer_ids.push(layer.id);
            assignment.estimated_compute_time_ms += self.estimate_layer_time(layer);
        }

        debug!("Round-robin assignment complete");
        assignments
    }

    /// Load-balanced assignment using greedy algorithm
    fn assign_load_balanced(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeId],
    ) -> HashMap<NodeId, LayerAssignment> {
        // Track current load per node
        let mut node_loads: HashMap<NodeId, f64> = nodes.iter().map(|&n| (n, 0.0)).collect();

        let mut assignments: HashMap<NodeId, LayerAssignment> = HashMap::new();

        // Assign each layer to node with minimum load
        for layer in layers {
            // Find node with minimum current load
            let (&min_node, _) = node_loads
                .iter()
                .min_by(|(_, load1), (_, load2)| {
                    load1.partial_cmp(load2).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            let assignment = assignments
                .entry(min_node)
                .or_insert_with(|| LayerAssignment {
                    node_id: min_node,
                    layer_ids: Vec::new(),
                    input_dependencies: Vec::new(),
                    output_consumers: Vec::new(),
                    estimated_compute_time_ms: 0.0,
                });

            let layer_time = self.estimate_layer_time(layer);

            assignment.layer_ids.push(layer.id);
            assignment.estimated_compute_time_ms += layer_time;

            // Update node load
            *node_loads.get_mut(&min_node).unwrap() += layer_time;
        }

        debug!(
            max_load = node_loads.values().cloned().fold(f64::NAN, f64::max),
            min_load = node_loads.values().cloned().fold(f64::NAN, f64::min),
            "Load-balanced assignment complete"
        );

        assignments
    }

    /// Topology-aware assignment minimizing communication
    fn assign_topology_aware(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeId],
    ) -> HashMap<NodeId, LayerAssignment> {
        let mut assignments: HashMap<NodeId, LayerAssignment> = HashMap::new();

        // Group consecutive layers together to minimize communication
        let layers_per_node = (layers.len() + nodes.len() - 1) / nodes.len();

        for (node_idx, node) in nodes.iter().enumerate() {
            let start = node_idx * layers_per_node;
            let end = (start + layers_per_node).min(layers.len());

            if start >= layers.len() {
                break;
            }

            let node_layers = &layers[start..end];

            let assignment = LayerAssignment {
                node_id: *node,
                layer_ids: node_layers.iter().map(|l| l.id).collect(),
                input_dependencies: if node_idx > 0 {
                    vec![nodes[node_idx - 1]]
                } else {
                    vec![]
                },
                output_consumers: if node_idx < nodes.len() - 1 {
                    vec![nodes[node_idx + 1]]
                } else {
                    vec![]
                },
                estimated_compute_time_ms: node_layers
                    .iter()
                    .map(|l| self.estimate_layer_time(l))
                    .sum(),
            };

            assignments.insert(*node, assignment);
        }

        debug!("Topology-aware assignment complete");
        assignments
    }

    /// Hybrid assignment: balance load with topology awareness
    fn assign_hybrid(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeId],
    ) -> HashMap<NodeId, LayerAssignment> {
        // Start with topology-aware (consecutive layers)
        let mut assignments = self.assign_topology_aware(layers, nodes);

        // Compute load imbalance
        let loads: Vec<f64> = assignments
            .values()
            .map(|a| a.estimated_compute_time_ms)
            .collect();

        let avg_load = loads.iter().sum::<f64>() / loads.len() as f64;
        let max_load = loads.iter().cloned().fold(f64::NAN, f64::max);

        let imbalance = (max_load - avg_load) / avg_load;

        debug!(
            imbalance,
            max_load,
            avg_load,
            "Hybrid assignment imbalance"
        );

        // If imbalance > 20%, rebalance
        if imbalance > 0.2 {
            debug!("Rebalancing due to high imbalance");
            // Fall back to load-balanced for better distribution
            assignments = self.assign_load_balanced(layers, nodes);
        }

        assignments
    }

    /// Estimate execution time for a layer (milliseconds)
    fn estimate_layer_time(&self, layer: &LayerInfo) -> f64 {
        // Simplified cost model: time proportional to FLOPs
        // Real implementation would consider:
        // - Node capacity (GPU, CPU specs)
        // - Batch size
        // - Kernel efficiency
        // - Memory bandwidth

        let base_time_ms = layer.compute_cost / 1e9; // Assume 1 GFLOPS baseline

        // Scale by layer type
        let type_factor = match &layer.layer_type {
            butterfly_core::LayerType::TransformerBlock { .. } => 1.5, // Attention is expensive
            butterfly_core::LayerType::Linear { .. } => 1.0,
            butterfly_core::LayerType::Activation => 0.1,
            butterfly_core::LayerType::Normalization => 0.2,
            _ => 1.0,
        };

        base_time_ms * type_factor
    }

    /// Build dependency graph from assignments
    fn build_dependency_graph(
        &self,
        assignments: &HashMap<NodeId, LayerAssignment>,
        nodes: &[NodeId],
    ) -> Result<DependencyGraph, String> {
        let mut graph = DependencyGraph::new();

        // Create edges based on layer ordering
        // Assumption: layers are processed in order, so earlier nodes feed later ones

        for (node_id, assignment) in assignments {
            // Add edges to successor nodes
            for successor in &assignment.output_consumers {
                graph.add_edge(*node_id, *successor);
            }
        }

        // Compute execution order
        graph.compute_execution_order(nodes)?;

        debug!(
            execution_order = ?graph.execution_order,
            "Dependency graph computed"
        );

        Ok(graph)
    }

    /// Update affinity for future assignments
    pub fn update_affinity(&mut self, node_id: NodeId, layer_id: usize, score: f64) {
        self.affinity_cache.insert((node_id, layer_id), score);
    }

    /// Get affinity score
    pub fn get_affinity(&self, node_id: NodeId, layer_id: usize) -> f64 {
        self.affinity_cache
            .get(&(node_id, layer_id))
            .copied()
            .unwrap_or(1.0)
    }
}

impl Default for WorkAssigner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use butterfly_core::LayerType;

    fn create_test_layers(count: usize) -> Vec<LayerInfo> {
        (0..count)
            .map(|i| LayerInfo {
                id: i,
                layer_type: LayerType::Linear {
                    input_dim: 768,
                    output_dim: 768,
                },
                compute_cost: 1e9, // 1 GFLOP
                memory_bytes: 1024 * 1024, // 1 MB
                output_size: 768,
            })
            .collect()
    }

    #[tokio::test]
    async fn test_round_robin_assignment() {
        let assigner = WorkAssigner::with_strategy(AssignmentStrategy::RoundRobin);
        let layers = create_test_layers(6);
        let nodes = vec![NodeId(0), NodeId(1), NodeId(2)];

        let assignment = assigner.assign(&layers, &nodes).await.unwrap();

        // Each node should get 2 layers
        for node in &nodes {
            let node_assignment = assignment.assignments.get(node).unwrap();
            assert_eq!(node_assignment.layer_ids.len(), 2);
        }
    }

    #[tokio::test]
    async fn test_load_balanced_assignment() {
        let assigner = WorkAssigner::with_strategy(AssignmentStrategy::LoadBalanced);

        // Create layers with varying costs
        let mut layers = create_test_layers(6);
        layers[0].compute_cost = 5e9; // Heavy layer
        layers[1].compute_cost = 1e9; // Light layer

        let nodes = vec![NodeId(0), NodeId(1)];

        let assignment = assigner.assign(&layers, &nodes).await.unwrap();

        // Loads should be relatively balanced
        let load0 = assignment
            .assignments
            .get(&NodeId(0))
            .unwrap()
            .estimated_compute_time_ms;
        let load1 = assignment
            .assignments
            .get(&NodeId(1))
            .unwrap()
            .estimated_compute_time_ms;

        let imbalance = (load0 - load1).abs() / load0.max(load1);
        assert!(imbalance < 0.5); // Within 50% of each other
    }

    #[tokio::test]
    async fn test_topology_aware_assignment() {
        let assigner = WorkAssigner::with_strategy(AssignmentStrategy::TopologyAware);
        let layers = create_test_layers(9);
        let nodes = vec![NodeId(0), NodeId(1), NodeId(2)];

        let assignment = assigner.assign(&layers, &nodes).await.unwrap();

        // Layers should be grouped consecutively
        let node0_layers = &assignment.assignments.get(&NodeId(0)).unwrap().layer_ids;
        let node1_layers = &assignment.assignments.get(&NodeId(1)).unwrap().layer_ids;

        // Node 1's first layer should follow node 0's last layer
        assert_eq!(node1_layers[0], node0_layers.last().unwrap() + 1);
    }

    #[tokio::test]
    async fn test_dependency_graph() {
        let assigner = WorkAssigner::new();
        let layers = create_test_layers(6);
        let nodes = vec![NodeId(0), NodeId(1), NodeId(2)];

        let assignment = assigner.assign(&layers, &nodes).await.unwrap();

        // Should have valid execution order
        assert_eq!(assignment.dependencies.execution_order.len(), 3);
    }

    #[test]
    fn test_layer_time_estimation() {
        let assigner = WorkAssigner::new();

        let linear_layer = LayerInfo {
            id: 0,
            layer_type: LayerType::Linear {
                input_dim: 768,
                output_dim: 768,
            },
            compute_cost: 1e9,
            memory_bytes: 1024 * 1024,
            output_size: 768,
        };

        let transformer_layer = LayerInfo {
            id: 1,
            layer_type: LayerType::TransformerBlock {
                hidden_dim: 768,
                num_heads: 12,
                ff_dim: 3072,
            },
            compute_cost: 1e9,
            memory_bytes: 1024 * 1024,
            output_size: 768,
        };

        let linear_time = assigner.estimate_layer_time(&linear_layer);
        let transformer_time = assigner.estimate_layer_time(&transformer_layer);

        // Transformer should take longer
        assert!(transformer_time > linear_time);
    }
}
