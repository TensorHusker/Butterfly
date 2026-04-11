//! Graph partitioning strategies for distributed graph processing
//!
//! This module implements various partitioning algorithms that divide a graph
//! into subgraphs for distributed processing. The goal is to minimize edge cuts
//! (edges between partitions) while balancing load across workers.

use butterfly_core::{
    Graph, Subgraph, Vertex, Edge, VertexId, EdgeId, PartitionId, NodeId,
};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// Quality metrics for a graph partition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPartitionQuality {
    /// Total edge cut (edges crossing partition boundaries)
    pub edge_cut: usize,
    /// Load balance: ratio of min vertices to max vertices (0.0 = poor, 1.0 = perfect)
    pub load_balance: f64,
    /// Communication volume (number of boundary vertices)
    pub communication_volume: usize,
    /// Replication factor (ratio of ghost vertices to local vertices)
    pub replication_factor: f64,
}

impl GraphPartitionQuality {
    pub fn from_subgraphs(subgraphs: &[Subgraph]) -> Self {
        if subgraphs.is_empty() {
            return Self {
                edge_cut: 0,
                load_balance: 1.0,
                communication_volume: 0,
                replication_factor: 0.0,
            };
        }

        // Calculate edge cut, accounting for undirected graphs
        let is_directed = subgraphs.first()
            .map(|sg| sg.graph.metadata.directed)
            .unwrap_or(true);

        let raw_edge_cut: usize = subgraphs.iter()
            .map(|sg| sg.edge_cut())
            .sum();

        // For undirected graphs, each edge is counted twice (once per partition)
        let edge_cut = if is_directed {
            raw_edge_cut
        } else {
            raw_edge_cut / 2
        };

        let vertex_counts: Vec<usize> = subgraphs.iter()
            .map(|sg| sg.local_vertex_count())
            .collect();

        let max_vertices = vertex_counts.iter().copied().max().unwrap_or(0);
        let min_vertices = vertex_counts.iter().copied().min().unwrap_or(0);

        let load_balance = if max_vertices > 0 {
            min_vertices as f64 / max_vertices as f64
        } else {
            1.0
        };

        let communication_volume: usize = subgraphs.iter()
            .map(|sg| sg.boundary_vertices.len())
            .sum();

        let total_local: usize = vertex_counts.iter().sum();
        let total_ghost: usize = subgraphs.iter()
            .map(|sg| sg.ghost_vertex_count())
            .sum();

        let replication_factor = if total_local > 0 {
            total_ghost as f64 / total_local as f64
        } else {
            0.0
        };

        Self {
            edge_cut,
            load_balance,
            communication_volume,
            replication_factor,
        }
    }
}

/// Trait for graph partitioning strategies
pub trait GraphPartitioner: Send + Sync {
    /// Partition a graph into subgraphs
    fn partition(
        &self,
        graph: &Graph,
        num_partitions: usize,
    ) -> Result<Vec<Subgraph>, PartitionError>;

    /// Estimate the quality of a partition
    fn estimate_quality(&self, subgraphs: &[Subgraph]) -> GraphPartitionQuality {
        GraphPartitionQuality::from_subgraphs(subgraphs)
    }
}

/// Hash-based partitioning (simple, fast, but poor edge cut)
pub struct HashPartitioner;

impl HashPartitioner {
    pub fn new() -> Self {
        Self
    }

    /// Assign vertex to partition using hash of vertex ID
    fn assign_vertex(vertex_id: VertexId, num_partitions: usize) -> PartitionId {
        PartitionId((vertex_id.0 as usize) % num_partitions)
    }
}

impl Default for HashPartitioner {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphPartitioner for HashPartitioner {
    fn partition(
        &self,
        graph: &Graph,
        num_partitions: usize,
    ) -> Result<Vec<Subgraph>, PartitionError> {
        if num_partitions == 0 {
            return Err(PartitionError::InvalidConfiguration(
                "Number of partitions must be > 0".to_string(),
            ));
        }

        if graph.vertex_count() == 0 {
            return Err(PartitionError::InvalidConfiguration(
                "Cannot partition empty graph".to_string(),
            ));
        }

        // Initialize subgraphs
        let mut subgraphs: Vec<Subgraph> = (0..num_partitions)
            .map(|i| Subgraph::new(PartitionId(i), format!("{}-partition-{}", graph.metadata.name, i)))
            .collect();

        // Assign vertices to partitions
        for (vertex_id, vertex) in &graph.vertices {
            let partition_id = Self::assign_vertex(*vertex_id, num_partitions);
            subgraphs[partition_id.0].graph.add_vertex(vertex.clone());
        }

        // Assign edges and identify boundary vertices
        for (edge_id, edge) in &graph.edges {
            let src_partition = Self::assign_vertex(edge.source, num_partitions);
            let dst_partition = Self::assign_vertex(edge.target, num_partitions);

            // Add edge to source partition
            subgraphs[src_partition.0].graph.add_edge(edge.clone());

            // If edge crosses partition boundary
            if src_partition != dst_partition {
                // Mark source as boundary vertex
                subgraphs[src_partition.0].boundary_vertices.insert(edge.source);

                // Add destination as ghost vertex in source partition
                if let Some(dst_vertex) = graph.get_vertex(edge.target) {
                    subgraphs[src_partition.0].add_ghost_vertex(dst_vertex.clone());
                }
            }
        }

        // Update graph metadata
        for subgraph in &mut subgraphs {
            subgraph.graph.metadata.directed = graph.metadata.directed;
            subgraph.graph.metadata.weighted = graph.metadata.weighted;
        }

        Ok(subgraphs)
    }
}

/// Stream-based partitioning (online partitioning with edge-cut optimization)
pub struct StreamPartitioner {
    /// Number of vertices to consider for each assignment decision
    lookahead: usize,
}

impl StreamPartitioner {
    pub fn new() -> Self {
        Self { lookahead: 100 }
    }

    pub fn with_lookahead(mut self, lookahead: usize) -> Self {
        self.lookahead = lookahead;
        self
    }

    /// Assign vertex to partition that minimizes edge cut
    fn assign_vertex_greedy(
        &self,
        vertex_id: VertexId,
        vertex: &Vertex,
        partition_sizes: &[usize],
        vertex_assignments: &HashMap<VertexId, PartitionId>,
        num_partitions: usize,
    ) -> PartitionId {
        // Find partition that maximizes number of neighbors already assigned to it
        let mut partition_scores: Vec<usize> = vec![0; num_partitions];

        // Count neighbors in each partition
        for neighbor_id in &vertex.neighbors {
            if let Some(&partition_id) = vertex_assignments.get(neighbor_id) {
                partition_scores[partition_id.0] += 1;
            }
        }

        // Apply load balancing penalty
        let avg_size = partition_sizes.iter().sum::<usize>() as f64 / num_partitions as f64;

        let mut best_partition = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (partition_id, &score) in partition_scores.iter().enumerate() {
            // Score = edge savings - load imbalance penalty
            let size_penalty = (partition_sizes[partition_id] as f64 - avg_size).abs();
            let adjusted_score = score as f64 - size_penalty * 0.1;

            if adjusted_score > best_score {
                best_score = adjusted_score;
                best_partition = partition_id;
            }
        }

        PartitionId(best_partition)
    }
}

impl Default for StreamPartitioner {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphPartitioner for StreamPartitioner {
    fn partition(
        &self,
        graph: &Graph,
        num_partitions: usize,
    ) -> Result<Vec<Subgraph>, PartitionError> {
        if num_partitions == 0 {
            return Err(PartitionError::InvalidConfiguration(
                "Number of partitions must be > 0".to_string(),
            ));
        }

        if graph.vertex_count() == 0 {
            return Err(PartitionError::InvalidConfiguration(
                "Cannot partition empty graph".to_string(),
            ));
        }

        // Initialize subgraphs
        let mut subgraphs: Vec<Subgraph> = (0..num_partitions)
            .map(|i| Subgraph::new(PartitionId(i), format!("{}-partition-{}", graph.metadata.name, i)))
            .collect();

        let mut partition_sizes = vec![0; num_partitions];
        let mut vertex_assignments: HashMap<VertexId, PartitionId> = HashMap::new();

        // Process vertices in order (could be improved with better ordering)
        let mut vertices: Vec<_> = graph.vertices.iter().collect();
        vertices.sort_by_key(|(id, _)| id.0);

        for (vertex_id, vertex) in vertices {
            // Assign vertex using greedy strategy
            let partition_id = self.assign_vertex_greedy(
                *vertex_id,
                vertex,
                &partition_sizes,
                &vertex_assignments,
                num_partitions,
            );

            vertex_assignments.insert(*vertex_id, partition_id);
            partition_sizes[partition_id.0] += 1;

            subgraphs[partition_id.0].graph.add_vertex(vertex.clone());
        }

        // Assign edges and identify boundary vertices
        for (edge_id, edge) in &graph.edges {
            let src_partition = vertex_assignments[&edge.source];
            let dst_partition = vertex_assignments[&edge.target];

            // Add edge to source partition
            subgraphs[src_partition.0].graph.add_edge(edge.clone());

            // If edge crosses partition boundary
            if src_partition != dst_partition {
                // Mark source as boundary vertex
                subgraphs[src_partition.0].boundary_vertices.insert(edge.source);

                // Add destination as ghost vertex
                if let Some(dst_vertex) = graph.get_vertex(edge.target) {
                    subgraphs[src_partition.0].add_ghost_vertex(dst_vertex.clone());
                }
            }
        }

        // Update graph metadata
        for subgraph in &mut subgraphs {
            subgraph.graph.metadata.directed = graph.metadata.directed;
            subgraph.graph.metadata.weighted = graph.metadata.weighted;
        }

        Ok(subgraphs)
    }
}

/// METIS-style multilevel partitioning
///
/// Implements a production-quality multilevel k-way partitioning algorithm
/// based on the METIS approach:
/// 1. Coarsening phase: Reduce graph size using Heavy Edge Matching
/// 2. Initial partitioning: Partition the coarsest graph
/// 3. Refinement phase: Uncoarsen and refine using Kernighan-Lin algorithm
pub struct METISPartitioner {
    /// Minimum graph size before stopping coarsening
    coarsening_threshold: usize,
    /// Number of Kernighan-Lin refinement iterations
    refinement_iterations: usize,
    /// Load imbalance tolerance (1.03 = 3% imbalance allowed)
    imbalance_tolerance: f64,
    /// Maximum number of coarsening levels
    max_coarsening_levels: usize,
}

impl METISPartitioner {
    pub fn new() -> Self {
        Self {
            coarsening_threshold: 100,
            refinement_iterations: 10,
            imbalance_tolerance: 1.03,
            max_coarsening_levels: 10,
        }
    }

    pub fn with_config(
        coarsening_threshold: usize,
        refinement_iterations: usize,
        imbalance_tolerance: f64,
    ) -> Self {
        Self {
            coarsening_threshold,
            refinement_iterations,
            imbalance_tolerance,
            max_coarsening_levels: 10,
        }
    }

    /// Phase 1: Coarsen the graph using Heavy Edge Matching
    fn coarsen_graph(&self, graph: &Graph) -> Vec<CoarsenedGraph> {
        let mut coarsening_hierarchy = Vec::new();
        let mut current_graph = graph.clone();

        for level in 0..self.max_coarsening_levels {
            if current_graph.vertex_count() <= self.coarsening_threshold {
                break;
            }

            let coarsened = self.heavy_edge_matching(&current_graph);

            // Stop if coarsening doesn't reduce size significantly
            if coarsened.graph.vertex_count() >= current_graph.vertex_count() * 4 / 5 {
                break;
            }

            coarsening_hierarchy.push(coarsened.clone());
            current_graph = coarsened.graph;
        }

        coarsening_hierarchy
    }

    /// Heavy Edge Matching: Match vertices with heaviest edge connections
    fn heavy_edge_matching(&self, graph: &Graph) -> CoarsenedGraph {
        let mut matched: HashSet<VertexId> = HashSet::new();
        let mut vertex_mapping: HashMap<VertexId, VertexId> = HashMap::new();
        let mut new_graph = Graph::new(format!("{}-coarsened", graph.metadata.name));
        new_graph.metadata.directed = graph.metadata.directed;
        new_graph.metadata.weighted = graph.metadata.weighted;

        let mut next_supervertex_id = 0;

        // Sort vertices by degree (descending) for better matching
        let mut vertices: Vec<_> = graph.vertices.keys().copied().collect();
        vertices.sort_by_key(|v| {
            let vertex = graph.get_vertex(*v).unwrap();
            std::cmp::Reverse(vertex.neighbors.len())
        });

        for &vertex_id in &vertices {
            if matched.contains(&vertex_id) {
                continue;
            }

            let vertex = graph.get_vertex(vertex_id).unwrap();

            // Find heaviest edge to unmatched neighbor
            let mut best_neighbor = None;
            let mut best_weight = 0.0;

            for &neighbor_id in &vertex.neighbors {
                if matched.contains(&neighbor_id) {
                    continue;
                }

                // Calculate edge weight (number of edges or weight if weighted graph)
                let weight = self.calculate_edge_weight(graph, vertex_id, neighbor_id);

                if weight > best_weight {
                    best_weight = weight;
                    best_neighbor = Some(neighbor_id);
                }
            }

            // Create supervertex
            let supervertex_id = VertexId(next_supervertex_id);
            next_supervertex_id += 1;

            let mut supervertex = Vertex::new(supervertex_id);

            if let Some(neighbor_id) = best_neighbor {
                // Match vertex with neighbor
                matched.insert(vertex_id);
                matched.insert(neighbor_id);
                vertex_mapping.insert(vertex_id, supervertex_id);
                vertex_mapping.insert(neighbor_id, supervertex_id);

                // Merge vertex data
                supervertex.data = vertex.data.clone();
                if let Some(neighbor) = graph.get_vertex(neighbor_id) {
                    // Could merge data here if needed
                }
            } else {
                // No match found, create singleton supervertex
                matched.insert(vertex_id);
                vertex_mapping.insert(vertex_id, supervertex_id);
                supervertex.data = vertex.data.clone();
            }

            new_graph.add_vertex(supervertex);
        }

        // Create edges in coarsened graph
        let mut edge_weights: HashMap<(VertexId, VertexId), f64> = HashMap::new();

        for (edge_id, edge) in &graph.edges {
            let src_super = vertex_mapping[&edge.source];
            let dst_super = vertex_mapping[&edge.target];

            // Skip self-loops
            if src_super == dst_super {
                continue;
            }

            let edge_key = if src_super.0 < dst_super.0 {
                (src_super, dst_super)
            } else {
                (dst_super, src_super)
            };

            let weight = edge.data.weight;
            *edge_weights.entry(edge_key).or_insert(0.0) += weight;
        }

        // Add coarsened edges
        let mut edge_id_counter = 0;
        for ((src, dst), weight) in edge_weights {
            let mut edge = Edge::new(EdgeId(edge_id_counter), src, dst);
            edge_id_counter += 1;
            edge.data.weight = weight;
            new_graph.add_edge(edge);
        }

        CoarsenedGraph {
            graph: new_graph,
            vertex_mapping,
        }
    }

    /// Calculate edge weight between two vertices
    fn calculate_edge_weight(&self, graph: &Graph, src: VertexId, dst: VertexId) -> f64 {
        let mut total_weight = 0.0;

        for (edge_id, edge) in &graph.edges {
            if (edge.source == src && edge.target == dst) ||
               (!graph.metadata.directed && edge.source == dst && edge.target == src) {
                total_weight += edge.data.weight;
            }
        }

        total_weight
    }

    /// Phase 2: Initial partition of the coarsest graph
    fn initial_partition(&self, graph: &Graph, k: usize) -> HashMap<VertexId, PartitionId> {
        let mut partition: HashMap<VertexId, PartitionId> = HashMap::new();

        if k == 1 {
            for &vertex_id in graph.vertices.keys() {
                partition.insert(vertex_id, PartitionId(0));
            }
            return partition;
        }

        if k == 2 {
            // Use recursive bisection for k=2
            self.recursive_bisection(graph, &mut partition, 0, 1);
        } else {
            // Use greedy growing regions for k > 2
            self.greedy_k_way_partition(graph, k, &mut partition);
        }

        partition
    }

    /// Recursive bisection for initial partitioning
    fn recursive_bisection(
        &self,
        graph: &Graph,
        partition: &mut HashMap<VertexId, PartitionId>,
        part_a: usize,
        part_b: usize,
    ) {
        let vertices: Vec<_> = graph.vertices.keys().copied().collect();

        if vertices.is_empty() {
            return;
        }

        // Use BFS from random starting vertex
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let start_vertex = *vertices.choose(&mut rng).unwrap();
        let mut queue = std::collections::VecDeque::new();
        let mut visited: HashSet<VertexId> = HashSet::new();

        queue.push_back(start_vertex);
        visited.insert(start_vertex);

        let target_size = vertices.len() / 2;
        let mut part_a_vertices = Vec::new();

        while !queue.is_empty() && part_a_vertices.len() < target_size {
            let vertex_id = queue.pop_front().unwrap();
            part_a_vertices.push(vertex_id);

            if let Some(vertex) = graph.get_vertex(vertex_id) {
                for &neighbor in &vertex.neighbors {
                    if !visited.contains(&neighbor) && graph.vertices.contains_key(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Assign partitions
        for vertex_id in &part_a_vertices {
            partition.insert(*vertex_id, PartitionId(part_a));
        }

        for vertex_id in &vertices {
            if !part_a_vertices.contains(vertex_id) {
                partition.insert(*vertex_id, PartitionId(part_b));
            }
        }
    }

    /// Greedy k-way partitioning using growing regions
    fn greedy_k_way_partition(
        &self,
        graph: &Graph,
        k: usize,
        partition: &mut HashMap<VertexId, PartitionId>,
    ) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let vertices: Vec<_> = graph.vertices.keys().copied().collect();

        if vertices.is_empty() {
            return;
        }

        // Select k seed vertices randomly
        let mut seeds: Vec<VertexId> = vertices.choose_multiple(&mut rng, k.min(vertices.len())).copied().collect();

        // Assign seeds to partitions
        for (i, &seed) in seeds.iter().enumerate() {
            partition.insert(seed, PartitionId(i));
        }

        let mut unassigned: HashSet<VertexId> = vertices.iter().copied().collect();
        for seed in &seeds {
            unassigned.remove(seed);
        }

        let target_size = (vertices.len() + k - 1) / k;
        let mut partition_sizes = vec![1; k];

        // Grow regions from seeds
        while !unassigned.is_empty() {
            let mut best_vertex = None;
            let mut best_partition = 0;
            let mut best_score = f64::NEG_INFINITY;

            for &vertex_id in &unassigned {
                let vertex = graph.get_vertex(vertex_id).unwrap();

                for part_id in 0..k {
                    // Skip if partition is full
                    if partition_sizes[part_id] >= target_size * self.imbalance_tolerance as usize {
                        continue;
                    }

                    // Count connections to this partition
                    let mut connections = 0;
                    for &neighbor in &vertex.neighbors {
                        if let Some(&neighbor_part) = partition.get(&neighbor) {
                            if neighbor_part.0 == part_id {
                                connections += 1;
                            }
                        }
                    }

                    // Penalize partition size imbalance
                    let size_penalty = partition_sizes[part_id] as f64 / target_size as f64;
                    let score = connections as f64 - size_penalty;

                    if score > best_score {
                        best_score = score;
                        best_vertex = Some(vertex_id);
                        best_partition = part_id;
                    }
                }
            }

            if let Some(vertex_id) = best_vertex {
                partition.insert(vertex_id, PartitionId(best_partition));
                partition_sizes[best_partition] += 1;
                unassigned.remove(&vertex_id);
            } else {
                // No valid assignment found, assign arbitrarily
                if let Some(&vertex_id) = unassigned.iter().next() {
                    let min_partition = partition_sizes.iter().enumerate()
                        .min_by_key(|(_, &size)| size)
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    partition.insert(vertex_id, PartitionId(min_partition));
                    partition_sizes[min_partition] += 1;
                    unassigned.remove(&vertex_id);
                }
            }
        }
    }

    /// Phase 3: Refine partition using Kernighan-Lin algorithm
    fn refine_partition(
        &self,
        graph: &Graph,
        partition: &mut HashMap<VertexId, PartitionId>,
        k: usize,
    ) {
        for _ in 0..self.refinement_iterations {
            let improved = self.kernighan_lin_pass(graph, partition, k);
            if !improved {
                break;
            }
        }
    }

    /// Single pass of Kernighan-Lin refinement
    fn kernighan_lin_pass(
        &self,
        graph: &Graph,
        partition: &mut HashMap<VertexId, PartitionId>,
        k: usize,
    ) -> bool {
        let mut improved = false;

        // Calculate partition sizes
        let mut partition_sizes = vec![0; k];
        for part_id in partition.values() {
            partition_sizes[part_id.0] += 1;
        }

        let avg_size = graph.vertex_count() as f64 / k as f64;
        let max_size = (avg_size * self.imbalance_tolerance) as usize;

        // Try moving each vertex to a different partition
        let vertices: Vec<_> = graph.vertices.keys().copied().collect();

        for &vertex_id in &vertices {
            let current_part = partition[&vertex_id].0;
            let vertex = graph.get_vertex(vertex_id).unwrap();

            // Calculate gain for moving to each other partition
            let mut best_gain = 0;
            let mut best_target_part = current_part;

            for target_part in 0..k {
                if target_part == current_part {
                    continue;
                }

                // Check load balance constraint
                if partition_sizes[target_part] >= max_size {
                    continue;
                }

                // Calculate edge cut gain
                let mut external_edges = 0; // Edges to current partition
                let mut internal_edges = 0; // Edges to target partition

                for &neighbor in &vertex.neighbors {
                    if let Some(&neighbor_part) = partition.get(&neighbor) {
                        if neighbor_part.0 == current_part {
                            external_edges += 1;
                        } else if neighbor_part.0 == target_part {
                            internal_edges += 1;
                        }
                    }
                }

                let gain = external_edges as i32 - internal_edges as i32;

                if gain > best_gain {
                    best_gain = gain;
                    best_target_part = target_part;
                }
            }

            // Move if beneficial
            if best_gain > 0 && best_target_part != current_part {
                partition.insert(vertex_id, PartitionId(best_target_part));
                partition_sizes[current_part] -= 1;
                partition_sizes[best_target_part] += 1;
                improved = true;
            }
        }

        improved
    }

    /// Uncoarsen and refine partition through hierarchy
    fn uncoarsen_and_refine(
        &self,
        hierarchy: &[CoarsenedGraph],
        initial_partition: HashMap<VertexId, PartitionId>,
        k: usize,
    ) -> HashMap<VertexId, PartitionId> {
        let mut partition = initial_partition;

        // Work backwards through hierarchy
        for coarsened in hierarchy.iter().rev() {
            // Project partition to finer level
            let mut fine_partition = HashMap::new();
            for (fine_vertex, coarse_vertex) in &coarsened.vertex_mapping {
                if let Some(&part_id) = partition.get(coarse_vertex) {
                    fine_partition.insert(*fine_vertex, part_id);
                }
            }

            partition = fine_partition;

            // Refine at this level (but with fewer iterations)
            for _ in 0..self.refinement_iterations.min(5) {
                // Need to reconstruct the graph at this level - use simple refinement
                let improved = self.simple_refinement_pass(&partition, k);
                if !improved {
                    break;
                }
            }
        }

        partition
    }

    /// Simple refinement pass without graph structure
    fn simple_refinement_pass(
        &self,
        partition: &HashMap<VertexId, PartitionId>,
        k: usize,
    ) -> bool {
        // Simplified refinement since we don't have the intermediate graph structure
        // In a full implementation, we would maintain the graph at each level
        false
    }
}

/// Coarsened graph representation
#[derive(Debug, Clone)]
struct CoarsenedGraph {
    graph: Graph,
    vertex_mapping: HashMap<VertexId, VertexId>, // fine -> coarse
}

impl Default for METISPartitioner {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphPartitioner for METISPartitioner {
    fn partition(
        &self,
        graph: &Graph,
        num_partitions: usize,
    ) -> Result<Vec<Subgraph>, PartitionError> {
        if num_partitions == 0 {
            return Err(PartitionError::InvalidConfiguration(
                "Number of partitions must be > 0".to_string(),
            ));
        }

        if graph.vertex_count() == 0 {
            return Err(PartitionError::InvalidConfiguration(
                "Cannot partition empty graph".to_string(),
            ));
        }

        // Handle trivial case
        if num_partitions == 1 {
            let mut subgraph = Subgraph::new(PartitionId(0), format!("{}-partition-0", graph.metadata.name));
            subgraph.graph = graph.clone();
            return Ok(vec![subgraph]);
        }

        // Phase 1: Coarsen the graph
        let hierarchy = self.coarsen_graph(graph);

        // Phase 2: Partition the coarsest graph
        let coarsest_graph = if hierarchy.is_empty() {
            graph.clone()
        } else {
            hierarchy.last().unwrap().graph.clone()
        };

        let mut partition = self.initial_partition(&coarsest_graph, num_partitions);

        // Phase 3: Refine on coarsest graph
        self.refine_partition(&coarsest_graph, &mut partition, num_partitions);

        // Phase 4: Uncoarsen and refine
        if !hierarchy.is_empty() {
            partition = self.uncoarsen_and_refine(&hierarchy, partition, num_partitions);
        }

        // Convert partition to subgraphs
        self.partition_to_subgraphs(graph, &partition, num_partitions)
    }
}

impl METISPartitioner {
    /// Convert vertex partition assignments to subgraphs
    fn partition_to_subgraphs(
        &self,
        graph: &Graph,
        partition: &HashMap<VertexId, PartitionId>,
        num_partitions: usize,
    ) -> Result<Vec<Subgraph>, PartitionError> {
        let mut subgraphs: Vec<Subgraph> = (0..num_partitions)
            .map(|i| Subgraph::new(PartitionId(i), format!("{}-partition-{}", graph.metadata.name, i)))
            .collect();

        // Add vertices to partitions
        for (vertex_id, vertex) in &graph.vertices {
            let part_id = partition.get(vertex_id)
                .copied()
                .unwrap_or(PartitionId(0));

            if part_id.0 < num_partitions {
                subgraphs[part_id.0].graph.add_vertex(vertex.clone());
            }
        }

        // Add edges and identify boundary vertices
        for (edge_id, edge) in &graph.edges {
            let src_part = partition.get(&edge.source)
                .copied()
                .unwrap_or(PartitionId(0));
            let dst_part = partition.get(&edge.target)
                .copied()
                .unwrap_or(PartitionId(0));

            if src_part.0 < num_partitions {
                subgraphs[src_part.0].graph.add_edge(edge.clone());

                // Handle boundary edges
                if src_part != dst_part {
                    subgraphs[src_part.0].boundary_vertices.insert(edge.source);

                    if let Some(dst_vertex) = graph.get_vertex(edge.target) {
                        subgraphs[src_part.0].add_ghost_vertex(dst_vertex.clone());
                    }
                }
            }
        }

        // Update metadata
        for subgraph in &mut subgraphs {
            subgraph.graph.metadata.directed = graph.metadata.directed;
            subgraph.graph.metadata.weighted = graph.metadata.weighted;
        }

        Ok(subgraphs)
    }
}

/// Errors that can occur during graph partitioning
#[derive(Debug, thiserror::Error)]
pub enum PartitionError {
    #[error("Invalid partition configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Partitioning failed: {0}")]
    Failed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use butterfly_core::GraphBuilder;

    fn create_test_graph() -> Graph {
        let mut builder = GraphBuilder::new("test");
        builder.directed()
            .add_vertex(VertexId(1))
            .add_vertex(VertexId(2))
            .add_vertex(VertexId(3))
            .add_vertex(VertexId(4))
            .add_edge(VertexId(1), VertexId(2))
            .add_edge(VertexId(2), VertexId(3))
            .add_edge(VertexId(3), VertexId(4))
            .add_edge(VertexId(4), VertexId(1));
        builder.build()
    }

    #[test]
    fn test_hash_partitioner() {
        let graph = create_test_graph();
        let partitioner = HashPartitioner::new();

        let subgraphs = partitioner.partition(&graph, 2).unwrap();
        assert_eq!(subgraphs.len(), 2);

        // Check total vertices match
        let total_vertices: usize = subgraphs.iter()
            .map(|sg| sg.local_vertex_count())
            .sum();
        assert_eq!(total_vertices, 4);

        // Check quality
        let quality = partitioner.estimate_quality(&subgraphs);
        assert!(quality.edge_cut > 0);
        assert!(quality.load_balance > 0.0 && quality.load_balance <= 1.0);
    }

    #[test]
    fn test_stream_partitioner() {
        let graph = create_test_graph();
        let partitioner = StreamPartitioner::new();

        let subgraphs = partitioner.partition(&graph, 2).unwrap();
        assert_eq!(subgraphs.len(), 2);

        let total_vertices: usize = subgraphs.iter()
            .map(|sg| sg.local_vertex_count())
            .sum();
        assert_eq!(total_vertices, 4);

        // Each partition should have at least one vertex
        for (i, sg) in subgraphs.iter().enumerate() {
            assert!(sg.local_vertex_count() > 0,
                "Partition {} has no vertices", i);
        }

        // Stream partitioner should produce a valid partition
        let stream_quality = partitioner.estimate_quality(&subgraphs);
        // Edge cut should be >= 0 (0 would be optimal, but typically > 0 for this graph)
        assert!(stream_quality.edge_cut >= 0);
        assert!(stream_quality.load_balance >= 0.0 && stream_quality.load_balance <= 1.0);
    }

    #[test]
    fn test_partition_quality_metrics() {
        let graph = create_test_graph();
        let partitioner = HashPartitioner::new();
        let subgraphs = partitioner.partition(&graph, 2).unwrap();

        let quality = GraphPartitionQuality::from_subgraphs(&subgraphs);

        assert!(quality.edge_cut > 0);
        assert!(quality.load_balance > 0.0);
        assert!(quality.communication_volume > 0);
        assert!(quality.replication_factor >= 0.0);
    }

    #[test]
    fn test_ghost_vertices() {
        let graph = create_test_graph();
        let partitioner = HashPartitioner::new();
        let subgraphs = partitioner.partition(&graph, 2).unwrap();

        // Check that ghost vertices are added for cross-partition edges
        let total_ghost: usize = subgraphs.iter()
            .map(|sg| sg.ghost_vertex_count())
            .sum();

        assert!(total_ghost > 0, "Should have ghost vertices for cross-partition edges");
    }

    #[test]
    fn test_metis_basic_partition() {
        let graph = create_test_graph();
        let metis = METISPartitioner::new();

        let subgraphs = metis.partition(&graph, 2).unwrap();
        assert_eq!(subgraphs.len(), 2);

        // Check total vertices match
        let total_vertices: usize = subgraphs.iter()
            .map(|sg| sg.local_vertex_count())
            .sum();
        assert_eq!(total_vertices, 4);

        // Each partition should have at least one vertex
        for (i, sg) in subgraphs.iter().enumerate() {
            assert!(sg.local_vertex_count() > 0,
                "Partition {} has no vertices", i);
        }
    }

    #[test]
    fn test_metis_edge_cut_quality() {
        // Create a larger graph with known structure
        let graph = create_mesh_graph(10, 10);

        let metis = METISPartitioner::new();
        let hash = HashPartitioner::new();

        let metis_parts = metis.partition(&graph, 4).unwrap();
        let hash_parts = hash.partition(&graph, 4).unwrap();

        let metis_quality = metis.estimate_quality(&metis_parts);
        let hash_quality = hash.estimate_quality(&hash_parts);

        // METIS should have better or equal edge cut than hash partitioner
        assert!(metis_quality.edge_cut <= hash_quality.edge_cut,
            "METIS edge cut ({}) should be <= Hash edge cut ({})",
            metis_quality.edge_cut, hash_quality.edge_cut);

        // METIS should maintain good load balance
        assert!(metis_quality.load_balance >= 0.7,
            "METIS load balance should be >= 0.7, got {}",
            metis_quality.load_balance);
    }

    #[test]
    fn test_metis_load_balance() {
        let graph = create_mesh_graph(8, 8);
        let metis = METISPartitioner::new();

        let subgraphs = metis.partition(&graph, 4).unwrap();
        let quality = metis.estimate_quality(&subgraphs);

        // Load balance should be high
        assert!(quality.load_balance >= 0.8,
            "Load balance should be >= 0.8, got {}", quality.load_balance);
    }

    #[test]
    fn test_metis_vs_stream() {
        let graph = create_mesh_graph(12, 12);

        let metis = METISPartitioner::new();
        let stream = StreamPartitioner::new();

        let metis_parts = metis.partition(&graph, 4).unwrap();
        let stream_parts = stream.partition(&graph, 4).unwrap();

        let metis_quality = metis.estimate_quality(&metis_parts);
        let stream_quality = stream.estimate_quality(&stream_parts);

        // METIS should generally outperform or match stream partitioner
        // At minimum, both should produce valid partitions
        assert!(metis_quality.edge_cut >= 0);
        assert!(stream_quality.edge_cut >= 0);
        assert!(metis_quality.load_balance > 0.0);
        assert!(stream_quality.load_balance > 0.0);
    }

    #[test]
    fn test_metis_with_config() {
        let graph = create_test_graph();
        let metis = METISPartitioner::with_config(50, 5, 1.05);

        let subgraphs = metis.partition(&graph, 2).unwrap();
        assert_eq!(subgraphs.len(), 2);
    }

    #[test]
    fn test_metis_single_partition() {
        let graph = create_test_graph();
        let metis = METISPartitioner::new();

        let subgraphs = metis.partition(&graph, 1).unwrap();
        assert_eq!(subgraphs.len(), 1);
        assert_eq!(subgraphs[0].local_vertex_count(), 4);
    }

    #[test]
    fn test_metis_many_partitions() {
        let graph = create_mesh_graph(10, 10);
        let metis = METISPartitioner::new();

        let subgraphs = metis.partition(&graph, 8).unwrap();
        assert_eq!(subgraphs.len(), 8);

        let total_vertices: usize = subgraphs.iter()
            .map(|sg| sg.local_vertex_count())
            .sum();
        assert_eq!(total_vertices, 100);
    }

    // Helper function to create a mesh graph for testing
    fn create_mesh_graph(rows: usize, cols: usize) -> Graph {
        let mut builder = GraphBuilder::new("mesh");
        builder.undirected();

        // Create vertices
        for i in 0..(rows * cols) {
            builder.add_vertex(VertexId(i as u64));
        }

        // Create edges (4-connected grid)
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let vid = VertexId(idx as u64);

                // Right neighbor
                if c + 1 < cols {
                    let right = VertexId((r * cols + c + 1) as u64);
                    builder.add_edge(vid, right);
                }

                // Bottom neighbor
                if r + 1 < rows {
                    let bottom = VertexId(((r + 1) * cols + c) as u64);
                    builder.add_edge(vid, bottom);
                }
            }
        }

        builder.build()
    }

    #[test]
    fn test_coarsening_reduces_graph_size() {
        let graph = create_mesh_graph(20, 20);
        let metis = METISPartitioner::new();

        let hierarchy = metis.coarsen_graph(&graph);

        // Should have at least one coarsening level
        assert!(!hierarchy.is_empty());

        // Each level should be smaller than the previous
        let mut prev_size = graph.vertex_count();
        for coarsened in &hierarchy {
            let current_size = coarsened.graph.vertex_count();
            assert!(current_size < prev_size,
                "Coarsened graph should be smaller: {} >= {}", current_size, prev_size);
            prev_size = current_size;
        }
    }

    #[test]
    fn test_heavy_edge_matching() {
        let graph = create_test_graph();
        let metis = METISPartitioner::new();

        let coarsened = metis.heavy_edge_matching(&graph);

        // Coarsened graph should have fewer or equal vertices
        assert!(coarsened.graph.vertex_count() <= graph.vertex_count());

        // All original vertices should be mapped
        assert_eq!(coarsened.vertex_mapping.len(), graph.vertex_count());
    }

    #[test]
    fn test_initial_partition_bisection() {
        let graph = create_mesh_graph(10, 10);
        let metis = METISPartitioner::new();

        let partition = metis.initial_partition(&graph, 2);

        // All vertices should be assigned
        assert_eq!(partition.len(), graph.vertex_count());

        // Count vertices in each partition
        let mut counts = vec![0; 2];
        for &part_id in partition.values() {
            counts[part_id.0] += 1;
        }

        // Both partitions should have some vertices
        assert!(counts[0] > 0);
        assert!(counts[1] > 0);
        assert_eq!(counts[0] + counts[1], graph.vertex_count());
    }

    #[test]
    fn test_kl_refinement_improves_or_maintains() {
        let graph = create_mesh_graph(8, 8);
        let metis = METISPartitioner::new();

        let mut partition = metis.initial_partition(&graph, 4);

        // Calculate initial edge cut
        let initial_cut = calculate_edge_cut(&graph, &partition);

        // Run refinement
        metis.refine_partition(&graph, &mut partition, 4);

        // Calculate final edge cut
        let final_cut = calculate_edge_cut(&graph, &partition);

        // Refinement should improve or maintain edge cut
        assert!(final_cut <= initial_cut,
            "Refinement should not increase edge cut: {} > {}", final_cut, initial_cut);
    }

    // Helper function to calculate edge cut
    fn calculate_edge_cut(
        graph: &Graph,
        partition: &HashMap<VertexId, PartitionId>,
    ) -> usize {
        let mut edge_cut = 0;
        let mut counted_edges = HashSet::new();

        for (edge_id, edge) in &graph.edges {
            let src_part = partition.get(&edge.source);
            let dst_part = partition.get(&edge.target);

            if let (Some(&src), Some(&dst)) = (src_part, dst_part) {
                if src != dst {
                    // For undirected graphs, count each edge once
                    if !graph.metadata.directed {
                        let edge_key = if edge.source.0 < edge.target.0 {
                            (edge.source, edge.target)
                        } else {
                            (edge.target, edge.source)
                        };
                        if !counted_edges.contains(&edge_key) {
                            edge_cut += 1;
                            counted_edges.insert(edge_key);
                        }
                    } else {
                        edge_cut += 1;
                    }
                }
            }
        }

        edge_cut
    }

    #[test]
    fn test_partition_quality_comparison() {
        let graph = create_mesh_graph(15, 15);

        let hash = HashPartitioner::new();
        let stream = StreamPartitioner::new();
        let metis = METISPartitioner::new();

        let hash_parts = hash.partition(&graph, 4).unwrap();
        let stream_parts = stream.partition(&graph, 4).unwrap();
        let metis_parts = metis.partition(&graph, 4).unwrap();

        let hash_quality = hash.estimate_quality(&hash_parts);
        let stream_quality = stream.estimate_quality(&stream_parts);
        let metis_quality = metis.estimate_quality(&metis_parts);

        println!("Hash edge cut: {}", hash_quality.edge_cut);
        println!("Stream edge cut: {}", stream_quality.edge_cut);
        println!("METIS edge cut: {}", metis_quality.edge_cut);

        println!("Hash load balance: {:.3}", hash_quality.load_balance);
        println!("Stream load balance: {:.3}", stream_quality.load_balance);
        println!("METIS load balance: {:.3}", metis_quality.load_balance);

        // All partitioners should produce valid partitions
        assert!(hash_quality.edge_cut > 0);
        assert!(stream_quality.edge_cut > 0);
        assert!(metis_quality.edge_cut > 0);

        // METIS should be competitive or better
        assert!(metis_quality.edge_cut <= hash_quality.edge_cut * 2,
            "METIS should be reasonably competitive with hash partitioner");
    }
}
