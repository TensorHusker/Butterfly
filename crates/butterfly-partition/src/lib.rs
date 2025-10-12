//! # Butterfly Partition
//!
//! Advanced model partitioning strategies for distributing neural network layers across nodes.
//! Implements high-performance algorithms with computational cost modeling and topology awareness.

use butterfly_core::{LayerInfo, LayerType, ModelPartition, NodeId, PartitionStrategy};
use std::collections::HashMap;

/// Represents the computational capabilities of a node
#[derive(Debug, Clone)]
pub struct NodeCapability {
    pub node_id: NodeId,
    pub compute_power: f64,  // Relative compute capacity (1.0 = baseline)
    pub memory_gb: f64,
    pub bandwidth_mbps: f64,
}

/// Network topology information
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Bandwidth matrix: bandwidth[i][j] = bandwidth from node i to j (Mbps)
    pub bandwidth: Vec<Vec<f64>>,
    /// Latency matrix: latency[i][j] = latency from node i to j (ms)
    pub latency: Vec<Vec<f64>>,
}

impl NetworkTopology {
    /// Create uniform topology where all nodes have same connectivity
    pub fn uniform(num_nodes: usize, bandwidth_mbps: f64, latency_ms: f64) -> Self {
        let bandwidth = vec![vec![bandwidth_mbps; num_nodes]; num_nodes];
        let latency = vec![vec![latency_ms; num_nodes]; num_nodes];
        Self { bandwidth, latency }
    }
}

/// Quality metrics for a partition
#[derive(Debug, Clone)]
pub struct PartitionQuality {
    /// Load balance: ratio of min load to max load (0.0 = poor, 1.0 = perfect)
    pub load_balance: f64,
    /// Total communication volume in MB
    pub communication_volume_mb: f64,
    /// Peak memory usage across all nodes in GB
    pub peak_memory_gb: f64,
    /// Estimated end-to-end latency in milliseconds
    pub estimated_latency_ms: f64,
}

/// Strategy for partitioning a model across available nodes
pub trait PartitionStrategyTrait: Send + Sync {
    /// Partition a model given layer information and node capabilities
    fn partition(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Result<Vec<ModelPartition>, PartitionError>;

    /// Estimate the quality of a partition
    fn estimate_quality(
        &self,
        partitions: &[ModelPartition],
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> PartitionQuality;
}

/// Simple uniform partitioning strategy (baseline)
pub struct UniformPartitioner;

impl PartitionStrategyTrait for UniformPartitioner {
    fn partition(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Result<Vec<ModelPartition>, PartitionError> {
        if nodes.is_empty() {
            return Err(PartitionError::NoNodesAvailable);
        }
        if layers.is_empty() {
            return Err(PartitionError::InvalidConfiguration(
                "No layers to partition".to_string(),
            ));
        }

        let layers_per_node = (layers.len() + nodes.len() - 1) / nodes.len();
        let mut partitions = Vec::new();

        for (idx, node) in nodes.iter().enumerate() {
            let start = idx * layers_per_node;
            if start >= layers.len() {
                break;
            }
            let end = ((idx + 1) * layers_per_node).min(layers.len());

            let estimated_time = layers[start..end]
                .iter()
                .map(|l| estimate_layer_time(l, node.compute_power))
                .sum();

            partitions.push(ModelPartition {
                id: idx as u64,
                layer_range: (start, end),
                assigned_node: Some(node.node_id),
                strategy: PartitionStrategy::Uniform,
                estimated_time_ms: estimated_time,
            });
        }

        Ok(partitions)
    }

    fn estimate_quality(
        &self,
        partitions: &[ModelPartition],
        _layers: &[LayerInfo],
        _nodes: &[NodeCapability],
    ) -> PartitionQuality {
        if partitions.is_empty() {
            return PartitionQuality {
                load_balance: 0.0,
                communication_volume_mb: 0.0,
                peak_memory_gb: 0.0,
                estimated_latency_ms: 0.0,
            };
        }

        let times: Vec<f64> = partitions.iter().map(|p| p.estimated_time_ms).collect();
        let max_time = times.iter().fold(0.0_f64, |a, &b| a.max(b));
        let min_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        PartitionQuality {
            load_balance: if max_time > 0.0 { min_time / max_time } else { 1.0 },
            communication_volume_mb: 0.0,  // Will be computed by derived metrics
            peak_memory_gb: 0.0,
            estimated_latency_ms: max_time,
        }
    }
}

/// Load-balanced partitioning using greedy bin packing
pub struct LoadBalancedPartitioner;

impl LoadBalancedPartitioner {
    /// Create partitions by greedily assigning layers to least-loaded nodes
    fn greedy_partition(
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Vec<ModelPartition> {
        // Track load per node
        let mut node_loads = vec![0.0; nodes.len()];
        let mut node_layers: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];

        // Sort layers by cost (descending) for better packing
        let mut indexed_layers: Vec<(usize, f64)> = layers
            .iter()
            .enumerate()
            .map(|(i, l)| (i, l.compute_cost))
            .collect();
        indexed_layers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Greedy assignment
        for (layer_idx, cost) in indexed_layers {
            // Find least loaded node with sufficient memory
            let best_node = (0..nodes.len())
                .filter(|&i| {
                    // Check memory constraint
                    let current_memory: u64 = node_layers[i]
                        .iter()
                        .map(|&idx| layers[idx].memory_bytes)
                        .sum();
                    current_memory + layers[layer_idx].memory_bytes
                        <= (nodes[i].memory_gb * 1e9) as u64
                })
                .min_by(|&i, &j| {
                    node_loads[i]
                        .partial_cmp(&node_loads[j])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0); // Fallback to first node if constraints can't be met

            node_layers[best_node].push(layer_idx);
            node_loads[best_node] += cost / nodes[best_node].compute_power;
        }

        // Convert to partitions (group consecutive layers)
        let mut partitions = Vec::new();
        for (node_idx, layer_indices) in node_layers.iter().enumerate() {
            if layer_indices.is_empty() {
                continue;
            }

            // Sort layer indices to find consecutive ranges
            let mut sorted_indices = layer_indices.clone();
            sorted_indices.sort_unstable();

            // Group into consecutive ranges
            let mut ranges = Vec::new();
            let mut start = sorted_indices[0];
            let mut end = start;

            for &idx in &sorted_indices[1..] {
                if idx == end + 1 {
                    end = idx;
                } else {
                    ranges.push((start, end + 1));
                    start = idx;
                    end = idx;
                }
            }
            ranges.push((start, end + 1));

            // Create partition for each range
            for (i, (start, end)) in ranges.into_iter().enumerate() {
                let estimated_time: f64 = (start..end)
                    .map(|idx| estimate_layer_time(&layers[idx], nodes[node_idx].compute_power))
                    .sum();

                partitions.push(ModelPartition {
                    id: (node_idx * 1000 + i) as u64,
                    layer_range: (start, end),
                    assigned_node: Some(nodes[node_idx].node_id),
                    strategy: PartitionStrategy::LoadBalanced,
                    estimated_time_ms: estimated_time,
                });
            }
        }

        partitions
    }
}

impl PartitionStrategyTrait for LoadBalancedPartitioner {
    fn partition(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Result<Vec<ModelPartition>, PartitionError> {
        if nodes.is_empty() {
            return Err(PartitionError::NoNodesAvailable);
        }
        if layers.is_empty() {
            return Err(PartitionError::InvalidConfiguration(
                "No layers to partition".to_string(),
            ));
        }

        Ok(Self::greedy_partition(layers, nodes))
    }

    fn estimate_quality(
        &self,
        partitions: &[ModelPartition],
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> PartitionQuality {
        if partitions.is_empty() {
            return PartitionQuality {
                load_balance: 0.0,
                communication_volume_mb: 0.0,
                peak_memory_gb: 0.0,
                estimated_latency_ms: 0.0,
            };
        }

        // Group partitions by node
        let mut node_times: HashMap<NodeId, f64> = HashMap::new();
        let mut node_memory: HashMap<NodeId, u64> = HashMap::new();

        for partition in partitions {
            if let Some(node_id) = partition.assigned_node {
                *node_times.entry(node_id).or_insert(0.0) += partition.estimated_time_ms;

                let memory: u64 = (partition.layer_range.0..partition.layer_range.1)
                    .map(|idx| layers[idx].memory_bytes)
                    .sum();
                *node_memory.entry(node_id).or_insert(0) += memory;
            }
        }

        let times: Vec<f64> = node_times.values().copied().collect();
        let max_time = times.iter().fold(0.0_f64, |a, &b| a.max(b));
        let min_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let peak_memory = node_memory
            .values()
            .max()
            .copied()
            .unwrap_or(0) as f64
            / 1e9;

        // Estimate communication volume (simplified)
        let comm_volume = (partitions.len().saturating_sub(1) as f64)
            * estimate_avg_tensor_size(layers)
            / 1e6; // Convert to MB

        PartitionQuality {
            load_balance: if max_time > 0.0 { min_time / max_time } else { 1.0 },
            communication_volume_mb: comm_volume,
            peak_memory_gb: peak_memory,
            estimated_latency_ms: max_time,
        }
    }
}

/// Topology-aware partitioning to minimize communication cost
pub struct TopologyAwarePartitioner {
    topology: NetworkTopology,
    max_iterations: usize,
}

impl TopologyAwarePartitioner {
    pub fn new(topology: NetworkTopology) -> Self {
        Self {
            topology,
            max_iterations: 1000,
        }
    }

    /// Estimate communication cost for a partition assignment
    fn estimate_comm_cost(
        &self,
        assignment: &[usize],
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> f64 {
        let mut total_cost = 0.0;

        // Calculate inter-node communication
        for i in 0..layers.len().saturating_sub(1) {
            let src_node = assignment[i];
            let dst_node = assignment[i + 1];

            if src_node != dst_node {
                let tensor_size_mb = layers[i].output_size as f64 * 4.0 / 1e6; // f32 = 4 bytes
                let bandwidth = self.topology.bandwidth[src_node][dst_node];
                let latency = self.topology.latency[src_node][dst_node];

                // Cost = latency + transfer_time
                total_cost += latency + (tensor_size_mb * 8.0 / bandwidth); // Convert MB to Mb
            }
        }

        total_cost
    }

    /// Simulated annealing optimization
    fn optimize_partition(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Vec<usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize with random assignment
        let mut current: Vec<usize> = (0..layers.len())
            .map(|_| rng.gen_range(0..nodes.len()))
            .collect();

        let mut best = current.clone();
        let mut best_cost = self.estimate_comm_cost(&best, layers, nodes);

        let mut temperature = 100.0;
        let cooling_rate = 0.995;
        let min_temperature = 1.0;

        for _ in 0..self.max_iterations {
            if temperature < min_temperature {
                break;
            }

            // Perturb: randomly reassign one layer
            let mut neighbor = current.clone();
            let layer_idx = rng.gen_range(0..layers.len());
            neighbor[layer_idx] = rng.gen_range(0..nodes.len());

            let neighbor_cost = self.estimate_comm_cost(&neighbor, layers, nodes);
            let delta = neighbor_cost - best_cost;

            // Accept if better, or with probability based on temperature
            if delta < 0.0 || rng.gen::<f64>() < (-delta / temperature).exp() {
                current = neighbor.clone();
                if neighbor_cost < best_cost {
                    best = neighbor;
                    best_cost = neighbor_cost;
                }
            }

            temperature *= cooling_rate;
        }

        best
    }

    /// Convert node assignment to partitions
    fn assignment_to_partitions(
        &self,
        assignment: &[usize],
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Vec<ModelPartition> {
        let mut partitions = Vec::new();

        if assignment.is_empty() {
            return partitions;
        }

        let mut start = 0;
        let mut current_node = assignment[0];

        for i in 1..=assignment.len() {
            let is_end = i == assignment.len();
            let node_changed = !is_end && assignment[i] != current_node;

            if node_changed || is_end {
                let end = i;
                let estimated_time: f64 = (start..end)
                    .map(|idx| estimate_layer_time(&layers[idx], nodes[current_node].compute_power))
                    .sum();

                partitions.push(ModelPartition {
                    id: partitions.len() as u64,
                    layer_range: (start, end),
                    assigned_node: Some(nodes[current_node].node_id),
                    strategy: PartitionStrategy::TopologyAware,
                    estimated_time_ms: estimated_time,
                });

                if node_changed {
                    start = i;
                    current_node = assignment[i];
                }
            }
        }

        partitions
    }
}

impl PartitionStrategyTrait for TopologyAwarePartitioner {
    fn partition(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Result<Vec<ModelPartition>, PartitionError> {
        if nodes.is_empty() {
            return Err(PartitionError::NoNodesAvailable);
        }
        if layers.is_empty() {
            return Err(PartitionError::InvalidConfiguration(
                "No layers to partition".to_string(),
            ));
        }

        let assignment = self.optimize_partition(layers, nodes);
        Ok(self.assignment_to_partitions(&assignment, layers, nodes))
    }

    fn estimate_quality(
        &self,
        partitions: &[ModelPartition],
        layers: &[LayerInfo],
        _nodes: &[NodeCapability],
    ) -> PartitionQuality {
        if partitions.is_empty() {
            return PartitionQuality {
                load_balance: 0.0,
                communication_volume_mb: 0.0,
                peak_memory_gb: 0.0,
                estimated_latency_ms: 0.0,
            };
        }

        let times: Vec<f64> = partitions.iter().map(|p| p.estimated_time_ms).collect();
        let max_time = times.iter().fold(0.0_f64, |a, &b| a.max(b));
        let min_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Calculate actual communication volume
        let mut comm_volume = 0.0;
        for i in 0..partitions.len().saturating_sub(1) {
            if partitions[i].assigned_node != partitions[i + 1].assigned_node {
                let layer_idx = partitions[i].layer_range.1.saturating_sub(1);
                if layer_idx < layers.len() {
                    comm_volume += layers[layer_idx].output_size as f64 * 4.0 / 1e6;
                }
            }
        }

        PartitionQuality {
            load_balance: if max_time > 0.0 { min_time / max_time } else { 1.0 },
            communication_volume_mb: comm_volume,
            peak_memory_gb: 0.0,
            estimated_latency_ms: max_time + comm_volume * 0.1, // Add communication overhead
        }
    }
}

/// Errors that can occur during model partitioning
#[derive(Debug, thiserror::Error)]
pub enum PartitionError {
    #[error("No nodes available for partitioning")]
    NoNodesAvailable,

    #[error("Invalid partition configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),
}

// Helper functions

/// Estimate execution time for a layer given compute power
fn estimate_layer_time(layer: &LayerInfo, compute_power: f64) -> f64 {
    // Base time in milliseconds, adjusted by compute power
    layer.compute_cost / (compute_power * 1e9) * 1000.0
}

/// Estimate average tensor size across layers
fn estimate_avg_tensor_size(layers: &[LayerInfo]) -> f64 {
    if layers.is_empty() {
        return 0.0;
    }
    let total: usize = layers.iter().map(|l| l.output_size).sum();
    (total as f64 / layers.len() as f64) * 4.0 // f32 = 4 bytes
}

/// Cost estimation utilities
pub mod cost_estimation {
    use super::*;

    /// Standard sequence length for transformers
    pub const DEFAULT_SEQ_LEN: usize = 512;

    /// Estimate computational cost (FLOPs) for a layer
    pub fn estimate_layer_cost(layer_type: &LayerType, seq_len: usize) -> f64 {
        match layer_type {
            LayerType::Embedding { vocab_size, hidden_dim } => {
                // Lookup + position encoding
                (vocab_size * hidden_dim + seq_len * hidden_dim) as f64
            }
            LayerType::TransformerBlock {
                hidden_dim,
                num_heads,
                ff_dim,
            } => {
                let head_dim = hidden_dim / num_heads;
                // Multi-head attention: Q, K, V projections + attention + output projection
                let attn_qkv = 3 * seq_len * hidden_dim * hidden_dim;
                let attn_scores = num_heads * seq_len * seq_len * head_dim;
                let attn_output = seq_len * hidden_dim * hidden_dim;

                // Feed-forward network: two linear layers
                let ff = 2 * seq_len * hidden_dim * ff_dim;

                // Layer norms (negligible)
                let norms = 2 * seq_len * hidden_dim;

                (attn_qkv + attn_scores + attn_output + ff + norms) as f64
            }
            LayerType::OutputHead {
                hidden_dim,
                vocab_size,
            } => {
                // Final projection + softmax
                (hidden_dim * vocab_size + vocab_size) as f64
            }
            LayerType::Linear {
                input_dim,
                output_dim,
            } => {
                // Matrix multiplication
                (seq_len * input_dim * output_dim) as f64
            }
            LayerType::Activation => {
                // Typically negligible, but account for element-wise ops
                (seq_len * 1000) as f64 // Placeholder
            }
            LayerType::Normalization => {
                // Mean, variance, normalize
                (seq_len * 1000 * 3) as f64
            }
        }
    }

    /// Estimate memory requirement for a layer
    pub fn estimate_layer_memory(layer_type: &LayerType) -> u64 {
        match layer_type {
            LayerType::Embedding { vocab_size, hidden_dim } => {
                // Embedding table size
                (vocab_size * hidden_dim * 4) as u64 // f32 = 4 bytes
            }
            LayerType::TransformerBlock {
                hidden_dim,
                ff_dim,
                ..
            } => {
                // Weights: QKV projections + output + FFN
                let attention_weights = 4 * hidden_dim * hidden_dim;
                let ffn_weights = 2 * hidden_dim * ff_dim;
                ((attention_weights + ffn_weights) * 4) as u64
            }
            LayerType::OutputHead {
                hidden_dim,
                vocab_size,
            } => {
                // Output projection matrix
                (hidden_dim * vocab_size * 4) as u64
            }
            LayerType::Linear {
                input_dim,
                output_dim,
            } => {
                // Weight matrix
                (input_dim * output_dim * 4) as u64
            }
            LayerType::Activation | LayerType::Normalization => {
                // Minimal memory for parameters
                1024
            }
        }
    }

    /// Create LayerInfo from LayerType with estimated costs
    pub fn create_layer_info(
        id: usize,
        layer_type: LayerType,
        seq_len: usize,
    ) -> LayerInfo {
        LayerInfo {
            id,
            compute_cost: estimate_layer_cost(&layer_type, seq_len),
            memory_bytes: estimate_layer_memory(&layer_type),
            output_size: match &layer_type {
                LayerType::Embedding { hidden_dim, .. }
                | LayerType::TransformerBlock { hidden_dim, .. }
                | LayerType::OutputHead { hidden_dim, .. } => seq_len * hidden_dim,
                LayerType::Linear { output_dim, .. } => seq_len * output_dim,
                LayerType::Activation | LayerType::Normalization => seq_len * 1000,
            },
            layer_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cost_estimation::*;

    fn create_test_nodes(count: usize) -> Vec<NodeCapability> {
        (0..count)
            .map(|i| NodeCapability {
                node_id: NodeId(i as u64),
                compute_power: 1.0,
                memory_gb: 16.0,
                bandwidth_mbps: 1000.0,
            })
            .collect()
    }

    fn create_test_layers(count: usize) -> Vec<LayerInfo> {
        (0..count)
            .map(|i| {
                create_layer_info(
                    i,
                    LayerType::TransformerBlock {
                        hidden_dim: 768,
                        num_heads: 12,
                        ff_dim: 3072,
                    },
                    DEFAULT_SEQ_LEN,
                )
            })
            .collect()
    }

    #[test]
    fn test_uniform_partition() {
        let nodes = create_test_nodes(2);
        let layers = create_test_layers(10);

        let partitioner = UniformPartitioner;
        let partitions = partitioner.partition(&layers, &nodes).unwrap();

        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].layer_range, (0, 5));
        assert_eq!(partitions[1].layer_range, (5, 10));
    }

    #[test]
    fn test_load_balanced_partition() {
        let nodes = create_test_nodes(3);
        let mut layers = create_test_layers(6);

        // Make one layer much more expensive
        layers[2].compute_cost *= 5.0;

        let partitioner = LoadBalancedPartitioner;
        let partitions = partitioner.partition(&layers, &nodes).unwrap();

        // Should distribute load more evenly than uniform
        assert!(!partitions.is_empty());

        let quality = partitioner.estimate_quality(&partitions, &layers, &nodes);

        // Compare with uniform partitioning
        let uniform = UniformPartitioner;
        let uniform_partitions = uniform.partition(&layers, &nodes).unwrap();
        let uniform_quality = uniform.estimate_quality(&uniform_partitions, &layers, &nodes);

        // Load-balanced should be better than or equal to uniform
        assert!(quality.load_balance >= uniform_quality.load_balance);
    }

    #[test]
    fn test_topology_aware_partition() {
        let nodes = create_test_nodes(2);
        let layers = create_test_layers(8);
        let topology = NetworkTopology::uniform(2, 1000.0, 1.0);

        let partitioner = TopologyAwarePartitioner::new(topology);
        let partitions = partitioner.partition(&layers, &nodes).unwrap();

        assert!(!partitions.is_empty());
        let quality = partitioner.estimate_quality(&partitions, &layers, &nodes);
        assert!(quality.communication_volume_mb >= 0.0);
    }

    #[test]
    fn test_cost_estimation() {
        let layer = LayerType::TransformerBlock {
            hidden_dim: 768,
            num_heads: 12,
            ff_dim: 3072,
        };

        let cost = estimate_layer_cost(&layer, DEFAULT_SEQ_LEN);
        assert!(cost > 0.0);

        let memory = estimate_layer_memory(&layer);
        assert!(memory > 0);
    }

    #[test]
    fn test_partition_quality() {
        let nodes = create_test_nodes(2);
        let layers = create_test_layers(10);

        let partitioner = UniformPartitioner;
        let partitions = partitioner.partition(&layers, &nodes).unwrap();
        let quality = partitioner.estimate_quality(&partitions, &layers, &nodes);

        assert!(quality.load_balance > 0.0);
        assert!(quality.load_balance <= 1.0);
        assert!(quality.estimated_latency_ms > 0.0);
    }
}
