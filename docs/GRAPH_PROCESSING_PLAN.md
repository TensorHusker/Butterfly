# Butterfly Graph Processing Implementation Plan

**Date**: 2025-10-20
**Status**: Design Complete - Ready for Implementation
**Author**: Orchestrator AI Agent

---

## Executive Summary

This document outlines a comprehensive plan to extend Butterfly's distributed LLM inference system to support general **distributed graph processing** using the **Bulk Synchronous Parallel (BSP)** model. The key insight is that **transformers are computation graphs**, and by generalizing Butterfly's architecture, we can support:

1. **LLM inference** (current focus)
2. **PageRank, shortest paths, community detection** (graph analytics)
3. **Graph neural networks** (GNNs)
4. **Custom vertex-centric computations** (extensible framework)

This extension leverages **100% of existing infrastructure**:
- Byzantine coordination protocol → BSP superstep coordination
- Layer partitioning → Graph partitioning (Hash, METIS, Stream)
- Pipeline execution → Vertex-centric message passing
- TensorRef system → Zero-copy graph data structures

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Unified Architecture Design](#2-unified-architecture-design)
3. [Core Data Structures](#3-core-data-structures)
4. [File-by-File Implementation Plan](#4-file-by-file-implementation-plan)
5. [BSP Computation Model](#5-bsp-computation-model)
6. [Integration Strategy](#6-integration-strategy)
7. [Priority Implementation Order](#7-priority-implementation-order)
8. [Testing Strategy](#8-testing-strategy)
9. [Performance Targets](#9-performance-targets)
10. [Future Extensions](#10-future-extensions)

---

## 1. Current State Analysis

### 1.1 Crate Structure Overview

```
butterfly-core/          ✅ Foundation types, state machines, tensor ops
butterfly-partition/     ✅ Layer partitioning (Uniform, LoadBalanced, TopologyAware)
butterfly-coordination/  ✅ Byzantine protocol, state machine, barriers
butterfly-schedule/      ✅ Task scheduling (FIFO, execution coordination)
butterfly-comm/          ✅ Message passing, network layer
butterfly-api/           ⚠️  Placeholder (needs implementation)
butterfly-metrics/       ✅ Observability, metrics collection
butterfly-cli/           ✅ Command-line interface
```

### 1.2 Key Existing Capabilities

#### butterfly-core
- **TensorRef**: Zero-copy tensor operations with Arc-based sharing
- **TensorPool**: Memory pooling for efficient allocation
- **StateMachine**: Type-safe state transitions
- **LayerInfo/ModelPartition**: Computational cost modeling
- **NodeId**: Distributed node identification

**Graph Processing Fit**:
- TensorRef → can represent adjacency lists, edge data
- LayerInfo → maps to Vertex computation cost
- StateMachine → BSP superstep state machine

#### butterfly-partition
- **Partitioning Strategies**: Uniform, LoadBalanced, TopologyAware
- **Cost Estimation**: FLOPs calculation for layers
- **Greedy Bin Packing**: Load balancing algorithm
- **Simulated Annealing**: Communication cost minimization

**Graph Processing Fit**:
- LoadBalanced → can adapt to vertex degree distribution
- TopologyAware → minimizes cross-partition edges
- Need to add: Hash partitioner, METIS wrapper, Stream partitioner

#### butterfly-coordination
- **4-Phase Protocol**: Assignment → Computation → Aggregation → Commitment
- **Byzantine Agreement**: 2f+1 quorum, PBFT-like consensus
- **DependencyGraph**: Already has topological sorting!
- **BarrierCoordinator**: Synchronization primitive
- **Epoch-based Execution**: Versioned computation rounds

**Graph Processing Fit**:
- 4-Phase Protocol → **maps perfectly to BSP supersteps**
  - Assignment = Partition assignment
  - Computation = Vertex compute phase
  - Aggregation = Message exchange phase
  - Commitment = Barrier synchronization
- DependencyGraph → can represent graph topology
- Epochs → BSP superstep numbering

#### butterfly-schedule
- **Scheduler Trait**: Extensible scheduling
- **FifoScheduler**: Simple queue-based scheduling
- **ExecutionCoordinator**: Multi-partition coordination

**Graph Processing Fit**:
- Can schedule vertex computations
- Need to add: Priority-based vertex scheduling, dynamic load balancing

#### butterfly-comm
- **Message Enum**: TensorData, Heartbeat, Ack
- **CommunicationBackend Trait**: Send, Receive, Broadcast
- **LocalBackend**: Testing implementation

**Graph Processing Fit**:
- Add GraphMessage variant for vertex messages
- Message passing naturally fits graph computation

### 1.3 Critical Insight: Natural BSP Mapping

**Butterfly's 4-phase protocol IS a BSP superstep!**

```
BSP Superstep               Butterfly Phase
────────────────────────────────────────────────────
Local Computation     →     Computation Phase (Phase 2)
Message Exchange      →     Aggregation Phase (Phase 3)
Barrier Sync          →     Commitment Phase (Phase 4)
Work Assignment       →     Assignment Phase (Phase 1)
```

This is **not a coincidence** - both are solving the same fundamental problem:
**Coordinating distributed computation with dependencies.**

---

## 2. Unified Architecture Design

### 2.1 The Computation Abstraction

**Key Principle**: Both transformers and graphs are **computation DAGs**.

```rust
// Unified trait for distributed computation
pub trait DistributedComputation {
    type Input;
    type Output;
    type State;

    /// Partition computation across nodes
    fn partition(&self, nodes: &[NodeCapability]) -> Vec<ComputationPartition>;

    /// Execute local computation
    fn compute_local(&mut self, input: Self::Input) -> Self::Output;

    /// Get dependencies for scheduling
    fn dependencies(&self) -> DependencyGraph;
}

// Transformer inference implements this
impl DistributedComputation for TransformerModel { ... }

// Graph algorithms implement this
impl DistributedComputation for PageRankComputation { ... }
impl DistributedComputation for ShortestPathComputation { ... }
impl DistributedComputation for GNNComputation { ... }
```

### 2.2 Type Hierarchy

```
ComputationGraph
├── LayeredGraph (Transformers)
│   ├── TransformerBlock
│   ├── FeedForward
│   └── Attention
└── VertexGraph (General Graphs)
    ├── PropertyGraph
    ├── HypergraphVertex
    └── TemporalVertex
```

### 2.3 Unified Execution Model

```
┌─────────────────────────────────────────────────────────┐
│              Butterfly Execution Engine                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Computation Abstraction Layer            │   │
│  │  DistributedComputation trait                    │   │
│  └──────────────┬───────────────┬──────────────────┘   │
│                 │               │                        │
│     ┌───────────▼─────┐   ┌────▼──────────┐           │
│     │ Transformer     │   │ Graph         │           │
│     │ Computation     │   │ Computation   │           │
│     └───────────┬─────┘   └────┬──────────┘           │
│                 │               │                        │
│  ┌──────────────▼───────────────▼──────────────────┐   │
│  │         Partition Layer                          │   │
│  │  • LayerPartitioner (transformers)               │   │
│  │  • HashPartitioner (graphs)                      │   │
│  │  • METISPartitioner (graphs)                     │   │
│  │  • StreamPartitioner (graphs)                    │   │
│  └──────────────┬───────────────────────────────────┘   │
│                 │                                        │
│  ┌──────────────▼───────────────────────────────────┐   │
│  │         Coordination Layer                        │   │
│  │  BSP Superstep Execution (4-phase protocol)      │   │
│  └──────────────┬───────────────────────────────────┘   │
│                 │                                        │
│  ┌──────────────▼───────────────────────────────────┐   │
│  │         Communication Layer                       │   │
│  │  Message Passing (TensorData | GraphMessage)     │   │
│  └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Core Data Structures

### 3.1 Graph Representation (butterfly-core)

**File**: `/Users/tensorhusker/Git/Butterfly/crates/butterfly-core/src/graph.rs`

```rust
//! Core graph data structures for distributed graph processing

use crate::{NodeId, TensorRef};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for a vertex in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VertexId(pub u64);

/// Unique identifier for an edge in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub u64);

/// A vertex in the computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex<V> {
    pub id: VertexId,
    /// User-defined vertex value/state
    pub value: V,
    /// Outgoing edges (adjacency list)
    pub out_edges: Vec<EdgeId>,
    /// Incoming edges (for some algorithms)
    pub in_edges: Vec<EdgeId>,
    /// Partition this vertex belongs to
    pub partition: Option<u64>,
}

/// An edge connecting two vertices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge<E> {
    pub id: EdgeId,
    pub source: VertexId,
    pub target: VertexId,
    /// User-defined edge weight/data
    pub value: E,
}

/// Computation graph with vertices and edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph<V, E> {
    pub vertices: HashMap<VertexId, Vertex<V>>,
    pub edges: HashMap<EdgeId, Edge<E>>,
    /// Metadata about the graph
    pub metadata: GraphMetadata,
}

/// Metadata about graph structure and properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub num_vertices: usize,
    pub num_edges: usize,
    /// Average vertex degree
    pub avg_degree: f64,
    /// Is the graph directed?
    pub directed: bool,
    /// Does it have self-loops?
    pub has_self_loops: bool,
}

/// A partition of a graph (subgraph assigned to one node)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subgraph<V, E> {
    pub partition_id: u64,
    pub assigned_node: Option<NodeId>,
    /// Vertices owned by this partition
    pub local_vertices: HashMap<VertexId, Vertex<V>>,
    /// Edges where source is local
    pub local_edges: HashMap<EdgeId, Edge<E>>,
    /// Ghost vertices (referenced by local vertices but owned elsewhere)
    pub ghost_vertices: HashMap<VertexId, u64>, // vertex_id -> owning_partition
    /// Estimated computation cost
    pub compute_cost: f64,
    /// Estimated memory requirement
    pub memory_bytes: u64,
}

impl<V, E> Graph<V, E> {
    pub fn new() -> Self {
        Self {
            vertices: HashMap::new(),
            edges: HashMap::new(),
            metadata: GraphMetadata {
                num_vertices: 0,
                num_edges: 0,
                avg_degree: 0.0,
                directed: true,
                has_self_loops: false,
            },
        }
    }

    pub fn add_vertex(&mut self, id: VertexId, value: V) {
        self.vertices.insert(id, Vertex {
            id,
            value,
            out_edges: Vec::new(),
            in_edges: Vec::new(),
            partition: None,
        });
        self.metadata.num_vertices = self.vertices.len();
    }

    pub fn add_edge(&mut self, id: EdgeId, source: VertexId, target: VertexId, value: E) {
        // Update source vertex
        if let Some(src) = self.vertices.get_mut(&source) {
            src.out_edges.push(id);
        }

        // Update target vertex
        if let Some(tgt) = self.vertices.get_mut(&target) {
            tgt.in_edges.push(id);
        }

        self.edges.insert(id, Edge { id, source, target, value });
        self.metadata.num_edges = self.edges.len();
        self.update_avg_degree();
    }

    fn update_avg_degree(&mut self) {
        if self.metadata.num_vertices == 0 {
            self.metadata.avg_degree = 0.0;
        } else {
            self.metadata.avg_degree =
                (self.metadata.num_edges as f64) / (self.metadata.num_vertices as f64);
        }
    }

    /// Get vertex by ID
    pub fn vertex(&self, id: VertexId) -> Option<&Vertex<V>> {
        self.vertices.get(&id)
    }

    /// Get mutable vertex by ID
    pub fn vertex_mut(&mut self, id: VertexId) -> Option<&mut Vertex<V>> {
        self.vertices.get_mut(&id)
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, vertex_id: VertexId) -> Vec<VertexId> {
        self.vertices
            .get(&vertex_id)
            .map(|v| {
                v.out_edges
                    .iter()
                    .filter_map(|&edge_id| {
                        self.edges.get(&edge_id).map(|e| e.target)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl<V, E> Default for Graph<V, E> {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy graph view using TensorRef for adjacency data
pub struct GraphView {
    /// Compressed Sparse Row (CSR) format
    pub row_offsets: TensorRef, // Vertex count + 1
    pub column_indices: TensorRef, // Edge count
    pub edge_data: Option<TensorRef>, // Edge weights
    pub num_vertices: usize,
    pub num_edges: usize,
}

impl GraphView {
    /// Create graph view from CSR arrays
    pub fn from_csr(
        row_offsets: Vec<f32>,
        column_indices: Vec<f32>,
        edge_data: Option<Vec<f32>>,
    ) -> Self {
        let num_vertices = row_offsets.len().saturating_sub(1);
        let num_edges = column_indices.len();

        Self {
            row_offsets: TensorRef::new(row_offsets, vec![num_vertices + 1]),
            column_indices: TensorRef::new(column_indices, vec![num_edges]),
            edge_data: edge_data.map(|d| TensorRef::new(d, vec![num_edges])),
            num_vertices,
            num_edges,
        }
    }

    /// Get neighbors of vertex i
    pub fn neighbors(&self, vertex: usize) -> &[f32] {
        let offsets = self.row_offsets.as_slice();
        let start = offsets[vertex] as usize;
        let end = offsets[vertex + 1] as usize;
        &self.column_indices.as_slice()[start..end]
    }
}
```

### 3.2 BSP Computation Model (butterfly-core)

**File**: `/Users/tensorhusker/Git/Butterfly/crates/butterfly-core/src/bsp.rs`

```rust
//! Bulk Synchronous Parallel (BSP) computation model

use crate::{NodeId, VertexId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Message sent between vertices during BSP computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexMessage<M> {
    pub source: VertexId,
    pub target: VertexId,
    pub payload: M,
    pub superstep: u64,
}

/// Trait for vertex-centric computation in BSP model
pub trait VertexCompute: Send + Sync {
    type VertexValue;
    type EdgeValue;
    type MessageValue: Clone;

    /// Compute function executed at each vertex per superstep
    ///
    /// Returns:
    /// - Updated vertex value
    /// - Messages to send to neighbors
    /// - Whether to halt (vote to halt)
    fn compute(
        &self,
        vertex_id: VertexId,
        vertex_value: &mut Self::VertexValue,
        messages: Vec<VertexMessage<Self::MessageValue>>,
        superstep: u64,
    ) -> ComputeResult<Self::VertexValue, Self::MessageValue>;

    /// Get initial vertex value
    fn initial_value(&self, vertex_id: VertexId) -> Self::VertexValue;

    /// Combine messages for the same target (optional optimization)
    fn combine_messages(
        &self,
        messages: Vec<VertexMessage<Self::MessageValue>>,
    ) -> Vec<VertexMessage<Self::MessageValue>> {
        // Default: no combining
        messages
    }
}

/// Result of vertex computation
#[derive(Debug, Clone)]
pub struct ComputeResult<V, M> {
    /// Updated vertex value (if changed)
    pub updated_value: Option<V>,
    /// Messages to send to neighbors
    pub messages: Vec<VertexMessage<M>>,
    /// Vote to halt (true = no more work for this vertex)
    pub vote_to_halt: bool,
}

/// BSP superstep state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuperstepPhase {
    /// Local computation at vertices
    Compute,
    /// Message exchange between partitions
    MessageExchange,
    /// Barrier synchronization
    Barrier,
    /// Termination check
    TerminationCheck,
}

/// Aggregator for global state (sum, min, max, etc.)
pub trait Aggregator: Send + Sync {
    type Value: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Initial value
    fn initial(&self) -> Self::Value;

    /// Combine two values
    fn combine(&self, a: Self::Value, b: Self::Value) -> Self::Value;

    /// Extract final value
    fn finalize(&self, value: Self::Value) -> Self::Value {
        value
    }
}

/// Common aggregators
pub struct SumAggregator;
pub struct MinAggregator;
pub struct MaxAggregator;
pub struct CountAggregator;

impl Aggregator for SumAggregator {
    type Value = f64;

    fn initial(&self) -> f64 { 0.0 }
    fn combine(&self, a: f64, b: f64) -> f64 { a + b }
}

impl Aggregator for MinAggregator {
    type Value = f64;

    fn initial(&self) -> f64 { f64::INFINITY }
    fn combine(&self, a: f64, b: f64) -> f64 { a.min(b) }
}

impl Aggregator for MaxAggregator {
    type Value = f64;

    fn initial(&self) -> f64 { f64::NEG_INFINITY }
    fn combine(&self, a: f64, b: f64) -> f64 { a.max(b) }
}

impl Aggregator for CountAggregator {
    type Value = u64;

    fn initial(&self) -> u64 { 0 }
    fn combine(&self, a: u64, b: u64) -> u64 { a + b }
}

/// BSP execution context
pub struct BSPContext<V, E, M> {
    pub superstep: u64,
    pub partition_id: u64,
    pub local_vertices: HashMap<VertexId, V>,
    pub outgoing_messages: Vec<VertexMessage<M>>,
    pub aggregators: HashMap<String, Box<dyn Aggregator<Value = f64>>>,
}

impl<V, E, M> BSPContext<V, E, M> {
    pub fn new(partition_id: u64) -> Self {
        Self {
            superstep: 0,
            partition_id,
            local_vertices: HashMap::new(),
            outgoing_messages: Vec::new(),
            aggregators: HashMap::new(),
        }
    }

    /// Send message to target vertex
    pub fn send_message(&mut self, target: VertexId, payload: M)
    where
        M: Clone,
    {
        self.outgoing_messages.push(VertexMessage {
            source: VertexId(0), // Set by caller
            target,
            payload,
            superstep: self.superstep,
        });
    }

    /// Aggregate value to global aggregator
    pub fn aggregate(&mut self, name: &str, value: f64) {
        if let Some(agg) = self.aggregators.get_mut(name) {
            // Combine with current value
            // Note: Full aggregation happens at coordinator
        }
    }

    /// Get aggregated value from previous superstep
    pub fn get_aggregated(&self, name: &str) -> Option<f64> {
        // TODO: Retrieve from coordinator
        None
    }
}
```

### 3.3 Graph Partitioning Strategies (butterfly-partition)

**File**: `/Users/tensorhusker/Git/Butterfly/crates/butterfly-partition/src/graph.rs`

```rust
//! Graph partitioning strategies for distributed graph processing

use butterfly_core::{Graph, Subgraph, VertexId, EdgeId, NodeId};
use super::{NodeCapability, PartitionError};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// Graph partitioning strategy
pub trait GraphPartitioner: Send + Sync {
    /// Partition graph across nodes
    fn partition<V, E>(
        &self,
        graph: &Graph<V, E>,
        nodes: &[NodeCapability],
    ) -> Result<Vec<Subgraph<V, E>>, PartitionError>
    where
        V: Clone + Serialize + for<'de> Deserialize<'de>,
        E: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Estimate quality of partition
    fn estimate_quality<V, E>(
        &self,
        partitions: &[Subgraph<V, E>],
    ) -> GraphPartitionQuality;
}

/// Quality metrics for graph partitions
#[derive(Debug, Clone)]
pub struct GraphPartitionQuality {
    /// Load balance across partitions (0.0 = poor, 1.0 = perfect)
    pub load_balance: f64,
    /// Fraction of edges crossing partitions (lower is better)
    pub edge_cut_ratio: f64,
    /// Communication volume in bytes
    pub communication_volume: u64,
    /// Max vertices per partition
    pub max_vertices_per_partition: usize,
}

/// Simple hash-based partitioning
pub struct HashPartitioner {
    pub num_partitions: usize,
}

impl HashPartitioner {
    pub fn new(num_partitions: usize) -> Self {
        Self { num_partitions }
    }

    fn hash_vertex(&self, vertex_id: VertexId) -> u64 {
        // Simple modulo hash
        vertex_id.0 % (self.num_partitions as u64)
    }
}

impl GraphPartitioner for HashPartitioner {
    fn partition<V, E>(
        &self,
        graph: &Graph<V, E>,
        nodes: &[NodeCapability],
    ) -> Result<Vec<Subgraph<V, E>>, PartitionError>
    where
        V: Clone + Serialize + for<'de> Deserialize<'de>,
        E: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        if nodes.is_empty() {
            return Err(PartitionError::NoNodesAvailable);
        }

        let num_partitions = nodes.len().min(self.num_partitions);
        let mut partitions: Vec<Subgraph<V, E>> = (0..num_partitions)
            .map(|i| Subgraph {
                partition_id: i as u64,
                assigned_node: Some(nodes[i].node_id),
                local_vertices: HashMap::new(),
                local_edges: HashMap::new(),
                ghost_vertices: HashMap::new(),
                compute_cost: 0.0,
                memory_bytes: 0,
            })
            .collect();

        // Assign vertices to partitions
        for (vertex_id, vertex) in &graph.vertices {
            let partition_idx = self.hash_vertex(*vertex_id) as usize % num_partitions;
            partitions[partition_idx].local_vertices.insert(*vertex_id, vertex.clone());
        }

        // Assign edges and identify ghost vertices
        for (edge_id, edge) in &graph.edges {
            let src_partition = self.hash_vertex(edge.source) as usize % num_partitions;

            // Edge belongs to source partition
            partitions[src_partition].local_edges.insert(*edge_id, edge.clone());

            // If target is in different partition, mark as ghost
            let tgt_partition = self.hash_vertex(edge.target) as usize % num_partitions;
            if src_partition != tgt_partition {
                partitions[src_partition]
                    .ghost_vertices
                    .insert(edge.target, tgt_partition as u64);
            }
        }

        Ok(partitions)
    }

    fn estimate_quality<V, E>(&self, partitions: &[Subgraph<V, E>]) -> GraphPartitionQuality {
        let vertex_counts: Vec<usize> = partitions
            .iter()
            .map(|p| p.local_vertices.len())
            .collect();

        let total_vertices: usize = vertex_counts.iter().sum();
        let max_vertices = vertex_counts.iter().max().copied().unwrap_or(0);
        let min_vertices = vertex_counts.iter().min().copied().unwrap_or(0);

        let load_balance = if max_vertices > 0 {
            min_vertices as f64 / max_vertices as f64
        } else {
            1.0
        };

        // Count edge cuts
        let total_edges: usize = partitions.iter().map(|p| p.local_edges.len()).sum();
        let ghost_edges: usize = partitions.iter().map(|p| p.ghost_vertices.len()).sum();

        let edge_cut_ratio = if total_edges > 0 {
            ghost_edges as f64 / total_edges as f64
        } else {
            0.0
        };

        GraphPartitionQuality {
            load_balance,
            edge_cut_ratio,
            communication_volume: ghost_edges as u64 * 64, // Assume 64 bytes per message
            max_vertices_per_partition: max_vertices,
        }
    }
}

/// METIS-based partitioning (k-way multilevel partitioning)
pub struct METISPartitioner {
    pub num_partitions: usize,
    pub balance_constraint: f64, // Imbalance tolerance (e.g., 1.03 = 3%)
}

impl METISPartitioner {
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions,
            balance_constraint: 1.03,
        }
    }

    // TODO: Integrate with METIS library (metis-rs crate)
    // For now, delegate to hash partitioning
}

impl GraphPartitioner for METISPartitioner {
    fn partition<V, E>(
        &self,
        graph: &Graph<V, E>,
        nodes: &[NodeCapability],
    ) -> Result<Vec<Subgraph<V, E>>, PartitionError>
    where
        V: Clone + Serialize + for<'de> Deserialize<'de>,
        E: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        // TODO: Call METIS library
        // For prototype, use hash partitioning
        let hash_partitioner = HashPartitioner::new(self.num_partitions);
        hash_partitioner.partition(graph, nodes)
    }

    fn estimate_quality<V, E>(&self, partitions: &[Subgraph<V, E>]) -> GraphPartitionQuality {
        let hash_partitioner = HashPartitioner::new(self.num_partitions);
        hash_partitioner.estimate_quality(partitions)
    }
}

/// Stream-based partitioning for dynamic graphs
pub struct StreamPartitioner {
    pub capacity_per_partition: usize,
    pub rebalance_threshold: f64, // Trigger rebalance if imbalance > threshold
}

impl StreamPartitioner {
    pub fn new(capacity_per_partition: usize) -> Self {
        Self {
            capacity_per_partition,
            rebalance_threshold: 1.2,
        }
    }

    /// Greedy stream assignment minimizing edge cuts
    fn stream_assign<V, E>(
        &self,
        graph: &Graph<V, E>,
        num_partitions: usize,
    ) -> HashMap<VertexId, u64>
    where
        V: Clone,
        E: Clone,
    {
        let mut assignment: HashMap<VertexId, u64> = HashMap::new();
        let mut partition_sizes = vec![0; num_partitions];

        for vertex_id in graph.vertices.keys() {
            // Find partition that minimizes edge cuts
            let neighbors = graph.neighbors(*vertex_id);

            let mut partition_scores = vec![0; num_partitions];
            for &neighbor_id in &neighbors {
                if let Some(&neighbor_partition) = assignment.get(&neighbor_id) {
                    partition_scores[neighbor_partition as usize] += 1;
                }
            }

            // Choose partition with most neighbors, respecting capacity
            let best_partition = partition_scores
                .iter()
                .enumerate()
                .filter(|(i, _)| partition_sizes[*i] < self.capacity_per_partition)
                .max_by_key(|(_, &score)| score)
                .map(|(i, _)| i)
                .unwrap_or_else(|| {
                    // All partitions full or no neighbors, choose least loaded
                    partition_sizes
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, &size)| size)
                        .map(|(i, _)| i)
                        .unwrap()
                });

            assignment.insert(*vertex_id, best_partition as u64);
            partition_sizes[best_partition] += 1;
        }

        assignment
    }
}

impl GraphPartitioner for StreamPartitioner {
    fn partition<V, E>(
        &self,
        graph: &Graph<V, E>,
        nodes: &[NodeCapability],
    ) -> Result<Vec<Subgraph<V, E>>, PartitionError>
    where
        V: Clone + Serialize + for<'de> Deserialize<'de>,
        E: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        if nodes.is_empty() {
            return Err(PartitionError::NoNodesAvailable);
        }

        let num_partitions = nodes.len();
        let assignment = self.stream_assign(graph, num_partitions);

        // Build subgraphs from assignment
        let mut partitions: Vec<Subgraph<V, E>> = (0..num_partitions)
            .map(|i| Subgraph {
                partition_id: i as u64,
                assigned_node: Some(nodes[i].node_id),
                local_vertices: HashMap::new(),
                local_edges: HashMap::new(),
                ghost_vertices: HashMap::new(),
                compute_cost: 0.0,
                memory_bytes: 0,
            })
            .collect();

        // Assign vertices
        for (vertex_id, vertex) in &graph.vertices {
            if let Some(&partition_id) = assignment.get(vertex_id) {
                partitions[partition_id as usize]
                    .local_vertices
                    .insert(*vertex_id, vertex.clone());
            }
        }

        // Assign edges
        for (edge_id, edge) in &graph.edges {
            if let Some(&src_partition) = assignment.get(&edge.source) {
                partitions[src_partition as usize]
                    .local_edges
                    .insert(*edge_id, edge.clone());

                if let Some(&tgt_partition) = assignment.get(&edge.target) {
                    if src_partition != tgt_partition {
                        partitions[src_partition as usize]
                            .ghost_vertices
                            .insert(edge.target, tgt_partition);
                    }
                }
            }
        }

        Ok(partitions)
    }

    fn estimate_quality<V, E>(&self, partitions: &[Subgraph<V, E>]) -> GraphPartitionQuality {
        let hash_partitioner = HashPartitioner::new(partitions.len());
        hash_partitioner.estimate_quality(partitions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use butterfly_core::{Graph, VertexId, EdgeId};

    #[test]
    fn test_hash_partitioner() {
        let mut graph = Graph::<i32, ()>::new();

        // Create simple graph: 0 -> 1 -> 2
        graph.add_vertex(VertexId(0), 0);
        graph.add_vertex(VertexId(1), 1);
        graph.add_vertex(VertexId(2), 2);
        graph.add_edge(EdgeId(0), VertexId(0), VertexId(1), ());
        graph.add_edge(EdgeId(1), VertexId(1), VertexId(2), ());

        let nodes = vec![
            NodeCapability {
                node_id: NodeId(0),
                compute_power: 1.0,
                memory_gb: 4.0,
                bandwidth_mbps: 1000.0,
            },
            NodeCapability {
                node_id: NodeId(1),
                compute_power: 1.0,
                memory_gb: 4.0,
                bandwidth_mbps: 1000.0,
            },
        ];

        let partitioner = HashPartitioner::new(2);
        let partitions = partitioner.partition(&graph, &nodes).unwrap();

        assert_eq!(partitions.len(), 2);
        assert!(partitions.iter().all(|p| !p.local_vertices.is_empty()));
    }
}
```

---

## 4. File-by-File Implementation Plan

### 4.1 butterfly-core

#### New Files

1. **`src/graph.rs`** (Priority: P0 - Critical)
   - **Lines**: ~400
   - **Content**: Graph, Vertex, Edge, Subgraph, GraphView
   - **Dependencies**: TensorRef, serde
   - **Tests**: Graph construction, neighbor queries, metadata updates

2. **`src/bsp.rs`** (Priority: P0 - Critical)
   - **Lines**: ~350
   - **Content**: VertexCompute trait, BSPContext, Aggregators
   - **Dependencies**: serde
   - **Tests**: Message creation, aggregation, superstep state

3. **`src/computation.rs`** (Priority: P1 - High)
   - **Lines**: ~200
   - **Content**: DistributedComputation trait (unifies transformers and graphs)
   - **Dependencies**: LayerInfo, graph types
   - **Tests**: Trait implementations for dummy computations

#### Modified Files

4. **`src/lib.rs`** (Priority: P0 - Critical)
   - **Changes**: Add `pub mod graph;`, `pub mod bsp;`, `pub mod computation;`
   - **Exports**: Export key types (Graph, VertexId, VertexCompute, etc.)

5. **`src/tensor.rs`** (Priority: P2 - Medium)
   - **Changes**: Add `GraphView::to_tensor()` conversion methods
   - **Lines Added**: ~50

### 4.2 butterfly-partition

#### New Files

1. **`src/graph.rs`** (Priority: P0 - Critical)
   - **Lines**: ~600
   - **Content**: GraphPartitioner trait, Hash/METIS/Stream partitioners
   - **Dependencies**: butterfly-core graph types
   - **Tests**: Partition each strategy, quality metrics

#### Modified Files

2. **`src/lib.rs`** (Priority: P0 - Critical)
   - **Changes**: Add `pub mod graph;`, export graph partitioner types
   - **Re-exports**: `pub use graph::{GraphPartitioner, HashPartitioner, ...};`

### 4.3 butterfly-coordination

#### New Files

1. **`src/bsp_coordinator.rs`** (Priority: P1 - High)
   - **Lines**: ~450
   - **Content**: BSP superstep coordinator, message routing, aggregation
   - **Dependencies**: CoordinationStateMachine, BarrierCoordinator
   - **Tests**: Superstep execution, message aggregation, termination detection

#### Modified Files

2. **`src/types.rs`** (Priority: P0 - Critical)
   - **Changes**: Add `GraphMessage` variant to `CoordinationMessage`
   - **Lines Added**: ~80
   - **New Types**:
     ```rust
     GraphMessage {
         source_partition: u64,
         target_partition: u64,
         messages: Vec<VertexMessage<Vec<u8>>>, // Serialized
         superstep: u64,
     }
     ```

3. **`src/barrier.rs`** (Priority: P1 - High)
   - **Changes**: Extend to support BSP superstep barriers
   - **Lines Added**: ~60
   - **New Methods**: `wait_for_superstep()`, `release_superstep()`

4. **`src/state_machine.rs`** (Priority: P1 - High)
   - **Changes**: Add `SuperstepPhase` state tracking
   - **Lines Added**: ~100

5. **`src/lib.rs`** (Priority: P0 - Critical)
   - **Changes**: Export BSP coordinator types

### 4.4 butterfly-schedule

#### New Files

1. **`src/graph_scheduler.rs`** (Priority: P2 - Medium)
   - **Lines**: ~300
   - **Content**: Vertex-centric scheduler, priority-based vertex scheduling
   - **Dependencies**: Graph types, VertexCompute
   - **Tests**: Vertex scheduling order, load balancing

#### Modified Files

2. **`src/lib.rs`** (Priority: P1 - High)
   - **Changes**: Add graph scheduler types
   - **Lines Added**: ~50

### 4.5 butterfly-comm

#### Modified Files

1. **`src/lib.rs`** (Priority: P0 - Critical)
   - **Changes**: Add `GraphMessage` variant to `Message` enum
   - **Lines Added**: ~40
   - **New Enum Variant**:
     ```rust
     pub enum Message {
         TensorData { ... },
         Heartbeat { ... },
         Ack { ... },
         // NEW
         GraphData {
             from: NodeId,
             to: NodeId,
             vertex_messages: Vec<u8>, // Serialized VertexMessage
             superstep: u64,
         },
     }
     ```

### 4.6 butterfly-api

#### New Files

1. **`src/graph_api.rs`** (Priority: P2 - Medium)
   - **Lines**: ~250
   - **Content**: REST API for graph operations (submit graph, get results)
   - **Endpoints**: `POST /graph/submit`, `GET /graph/result/{id}`

### 4.7 New Crate: butterfly-graph-algorithms

#### Purpose
Pre-built graph algorithms (PageRank, SSSP, community detection, etc.)

#### Structure
```
butterfly-graph-algorithms/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── pagerank.rs         (P1 - High)
│   ├── shortest_path.rs    (P2 - Medium)
│   ├── connected_components.rs (P2 - Medium)
│   ├── triangle_counting.rs (P3 - Low)
│   └── label_propagation.rs (P3 - Low)
└── tests/
    └── integration_test.rs
```

**Example: PageRank** (`src/pagerank.rs`)
```rust
use butterfly_core::{VertexCompute, VertexMessage, ComputeResult, VertexId};

pub struct PageRankCompute {
    pub damping_factor: f64,
    pub num_vertices: usize,
}

impl VertexCompute for PageRankCompute {
    type VertexValue = f64; // PageRank score
    type EdgeValue = (); // Unweighted
    type MessageValue = f64; // Contribution from neighbor

    fn compute(
        &self,
        vertex_id: VertexId,
        vertex_value: &mut Self::VertexValue,
        messages: Vec<VertexMessage<Self::MessageValue>>,
        superstep: u64,
    ) -> ComputeResult<Self::VertexValue, Self::MessageValue> {
        if superstep == 0 {
            // Initial PageRank
            *vertex_value = 1.0 / self.num_vertices as f64;
        } else {
            // Aggregate messages
            let sum: f64 = messages.iter().map(|m| m.payload).sum();
            *vertex_value = (1.0 - self.damping_factor) / self.num_vertices as f64
                + self.damping_factor * sum;
        }

        // Send contribution to neighbors
        // Assume out_degree is stored somewhere
        let contribution = *vertex_value / self.out_degree(vertex_id) as f64;

        let messages = self.neighbors(vertex_id)
            .iter()
            .map(|&target| VertexMessage {
                source: vertex_id,
                target,
                payload: contribution,
                superstep: superstep + 1,
            })
            .collect();

        ComputeResult {
            updated_value: Some(*vertex_value),
            messages,
            vote_to_halt: superstep > 30, // Max iterations
        }
    }

    fn initial_value(&self, _vertex_id: VertexId) -> Self::VertexValue {
        1.0 / self.num_vertices as f64
    }
}
```

---

## 5. BSP Computation Model

### 5.1 Superstep Execution Flow

```
Superstep S:
┌──────────────────────────────────────────────────────────┐
│ 1. Assignment Phase (Butterfly Phase 1)                  │
│    - Coordinator assigns vertices to workers             │
│    - Workers load local graph partition                  │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│ 2. Computation Phase (Butterfly Phase 2)                 │
│    - Each worker executes VertexCompute::compute()       │
│    - Local computation on assigned vertices              │
│    - Generate messages for neighbors                     │
│    - Update vertex values                                │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│ 3. Message Exchange Phase (Butterfly Phase 3)            │
│    - Workers send messages to owning partitions          │
│    - Coordinator aggregates global state (sums, etc.)    │
│    - Messages buffered for next superstep                │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│ 4. Barrier Synchronization (Butterfly Phase 4)           │
│    - All workers report completion                       │
│    - Coordinator checks termination condition            │
│    - If not done, advance to superstep S+1               │
│    - If done, finalize results                           │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Mapping to Existing Coordination Protocol

| BSP Concept | Butterfly Coordination | Implementation |
|-------------|------------------------|----------------|
| **Superstep** | Epoch | `epoch: u64` in CoordinationStateMachine |
| **Local Computation** | Computation Phase | Execute `VertexCompute::compute()` on each local vertex |
| **Message Passing** | Aggregation Phase | Send `GraphMessage` via CommunicationBackend |
| **Barrier Sync** | Commitment Phase | `BarrierCoordinator::wait_for_superstep()` |
| **Termination** | All vertices vote_to_halt | Global AND aggregator |

### 5.3 Termination Detection

**Algorithm**: Global AND of local termination votes

```rust
pub struct TerminationDetector {
    votes_to_halt: HashMap<u64, bool>, // partition_id -> voted_to_halt
    quorum_size: usize,
}

impl TerminationDetector {
    pub fn record_vote(&mut self, partition_id: u64, vote: bool) {
        self.votes_to_halt.insert(partition_id, vote);
    }

    pub fn is_terminated(&self) -> bool {
        self.votes_to_halt.len() >= self.quorum_size
            && self.votes_to_halt.values().all(|&v| v)
    }
}
```

**Integration**: Add to `CoordinationStateMachine`
```rust
pub struct CoordinationStateMachine {
    // ... existing fields
    termination_detector: TerminationDetector,
}
```

---

## 6. Integration Strategy

### 6.1 Phase 1: Core Infrastructure (Week 1-2)

**Goal**: Add graph types and BSP abstractions without breaking existing LLM inference.

**Steps**:
1. Create `butterfly-core/src/graph.rs` with basic Graph/Vertex/Edge types
2. Create `butterfly-core/src/bsp.rs` with VertexCompute trait
3. Add graph module exports to `butterfly-core/src/lib.rs`
4. Write comprehensive unit tests
5. **Validation**: `cargo test --workspace` passes, no regressions

**Deliverables**:
- ✅ Graph data structures
- ✅ BSP computation model
- ✅ Zero-copy GraphView
- ✅ 90%+ test coverage

### 6.2 Phase 2: Graph Partitioning (Week 3)

**Goal**: Implement graph partitioners, validate quality metrics.

**Steps**:
1. Create `butterfly-partition/src/graph.rs`
2. Implement HashPartitioner (simple, fast)
3. Implement StreamPartitioner (edge-cut optimized)
4. Add METIS stub (integrate library later)
5. Benchmark partitioning quality on real graphs

**Deliverables**:
- ✅ 3 partitioning strategies
- ✅ Quality metrics (load balance, edge cut ratio)
- ✅ Benchmarks on graphs: Karate Club, SNAP datasets

### 6.3 Phase 3: BSP Coordination (Week 4)

**Goal**: Extend coordination protocol to support BSP supersteps.

**Steps**:
1. Add `GraphMessage` to coordination types
2. Create `BSPCoordinator` in `butterfly-coordination`
3. Extend `BarrierCoordinator` for superstep sync
4. Implement message routing between partitions
5. Add aggregator support

**Deliverables**:
- ✅ BSP superstep execution
- ✅ Message passing between partitions
- ✅ Global aggregation (sum, min, max)
- ✅ Termination detection

### 6.4 Phase 4: Reference Implementation (Week 5)

**Goal**: Implement PageRank as end-to-end validation.

**Steps**:
1. Create `butterfly-graph-algorithms` crate
2. Implement PageRank using VertexCompute
3. Add integration test: 4-node cluster, Web-Google graph
4. Validate correctness against NetworkX reference
5. Measure performance (throughput, latency)

**Deliverables**:
- ✅ Working PageRank implementation
- ✅ Correctness validation (< 1e-6 error vs NetworkX)
- ✅ Performance baseline (>1M edges/sec)

### 6.5 Phase 5: Additional Algorithms (Week 6+)

**Goal**: Expand algorithm library, optimize performance.

**Steps**:
1. Implement Single-Source Shortest Path (SSSP)
2. Implement Connected Components
3. Implement Triangle Counting
4. Performance optimization (message combining, caching)
5. Add graph loading from common formats (edgelist, GraphML)

---

## 7. Priority Implementation Order

### P0: Critical (Must Have for MVP)

1. **butterfly-core/src/graph.rs**
   - Graph, Vertex, Edge, Subgraph
   - GraphView (zero-copy representation)

2. **butterfly-core/src/bsp.rs**
   - VertexCompute trait
   - VertexMessage
   - ComputeResult
   - Basic aggregators (Sum, Min, Max)

3. **butterfly-partition/src/graph.rs**
   - GraphPartitioner trait
   - HashPartitioner (simple hash-based)

4. **butterfly-coordination/src/types.rs**
   - Add GraphMessage variant

5. **butterfly-coordination/src/bsp_coordinator.rs**
   - Superstep execution
   - Message routing

### P1: High (Essential for Usability)

6. **butterfly-partition/src/graph.rs**
   - StreamPartitioner (edge-cut optimization)

7. **butterfly-coordination/src/barrier.rs**
   - Superstep barrier synchronization

8. **butterfly-graph-algorithms/src/pagerank.rs**
   - Reference implementation for validation

9. **Integration tests**
   - End-to-end PageRank on small graph

### P2: Medium (Important for Production)

10. **butterfly-partition/src/graph.rs**
    - METISPartitioner (library integration)

11. **butterfly-graph-algorithms**
    - SSSP, Connected Components

12. **butterfly-api/src/graph_api.rs**
    - REST API for graph operations

13. **Performance optimization**
    - Message combining
    - Vertex value caching

### P3: Low (Nice to Have)

14. **Advanced algorithms**
    - Triangle Counting
    - Label Propagation
    - Betweenness Centrality

15. **Graph loading utilities**
    - EdgeList parser
    - GraphML parser
    - Integration with graph databases

---

## 8. Testing Strategy

### 8.1 Unit Tests

**butterfly-core/src/graph.rs**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::<i32, ()>::new();
        graph.add_vertex(VertexId(0), 42);
        assert_eq!(graph.vertices.len(), 1);
    }

    #[test]
    fn test_neighbor_query() {
        let mut graph = Graph::<i32, ()>::new();
        graph.add_vertex(VertexId(0), 0);
        graph.add_vertex(VertexId(1), 1);
        graph.add_edge(EdgeId(0), VertexId(0), VertexId(1), ());

        let neighbors = graph.neighbors(VertexId(0));
        assert_eq!(neighbors, vec![VertexId(1)]);
    }
}
```

### 8.2 Integration Tests

**tests/graph_processing_integration.rs**
```rust
use butterfly_core::{Graph, VertexId, EdgeId};
use butterfly_partition::HashPartitioner;
use butterfly_coordination::BSPCoordinator;
use butterfly_graph_algorithms::PageRankCompute;

#[tokio::test]
async fn test_distributed_pagerank() {
    // 1. Create simple graph
    let mut graph = Graph::new();
    // Add vertices and edges

    // 2. Partition across 2 nodes
    let partitioner = HashPartitioner::new(2);
    let partitions = partitioner.partition(&graph, &nodes).unwrap();

    // 3. Execute PageRank
    let pagerank = PageRankCompute::new(0.85, graph.num_vertices());
    let coordinator = BSPCoordinator::new(partitions);
    let results = coordinator.execute(&pagerank).await.unwrap();

    // 4. Validate results
    assert!((results[&VertexId(0)] - expected_pr).abs() < 1e-6);
}
```

### 8.3 Benchmarks

**benches/graph_partitioning.rs**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use butterfly_core::Graph;
use butterfly_partition::HashPartitioner;

fn partition_benchmark(c: &mut Criterion) {
    let graph = load_test_graph("karate_club.txt");
    let nodes = create_test_nodes(4);

    c.bench_function("hash_partition_karate", |b| {
        b.iter(|| {
            let partitioner = HashPartitioner::new(4);
            partitioner.partition(black_box(&graph), black_box(&nodes))
        });
    });
}

criterion_group!(benches, partition_benchmark);
criterion_main!(benches);
```

---

## 9. Performance Targets

### 9.1 Partitioning Performance

| Graph Size | Partitioner | Target Time | Memory |
|------------|-------------|-------------|--------|
| 10K vertices | Hash | < 50ms | < 10MB |
| 100K vertices | Hash | < 500ms | < 100MB |
| 1M vertices | Hash | < 5s | < 1GB |
| 10K vertices | Stream | < 200ms | < 15MB |
| 10K vertices | METIS | < 1s | < 50MB |

### 9.2 BSP Execution Performance

**PageRank on Web-Google (875K vertices, 5.1M edges)**

| Nodes | Partitioner | Supersteps | Time | Throughput |
|-------|-------------|------------|------|------------|
| 4 | Hash | 30 | 45s | 113K edges/s |
| 8 | Hash | 30 | 28s | 182K edges/s |
| 16 | Hash | 30 | 18s | 283K edges/s |
| 4 | Stream | 30 | 35s | 146K edges/s |
| 8 | METIS | 30 | 22s | 232K edges/s |

**Target**: >100K edges/sec per node

### 9.3 Communication Overhead

| Metric | Target |
|--------|--------|
| Message latency (local) | < 1ms |
| Message latency (10Gbps) | < 5ms |
| Serialization overhead | < 10% |
| Barrier synchronization | < 50ms |
| Superstep overhead | < 5% of compute time |

---

## 10. Future Extensions

### 10.1 Advanced Partitioning

1. **Streaming Partitioning**
   - Handle dynamic graphs (edge additions/deletions)
   - Incremental repartitioning

2. **Hierarchical Partitioning**
   - Multi-level partitioning for very large graphs
   - Locality-aware placement

3. **Hybrid Partitioning**
   - Combine vertex-cut and edge-cut
   - Optimize for specific workloads

### 10.2 Computation Models

1. **Asynchronous BSP**
   - Relax barrier synchronization
   - Stale vertex values allowed

2. **GraphLab-style GAS Model**
   - Gather-Apply-Scatter paradigm
   - More flexible than pure BSP

3. **Streaming Computation**
   - Process infinite graph streams
   - Sliding window computations

### 10.3 Integration with ML

1. **Graph Neural Networks**
   - Message passing neural networks
   - Graph convolutions

2. **Embedding Learning**
   - Node2Vec, DeepWalk
   - Knowledge graph embeddings

3. **Hybrid LLM + Graph**
   - Graph-augmented transformers
   - Retrieval-augmented generation on graphs

### 10.4 Optimizations

1. **Message Combining**
   - Aggregate messages before sending
   - Reduce communication volume

2. **Vertex Value Caching**
   - Cache frequently accessed vertices
   - Ghost vertex caching

3. **Computation Pipelining**
   - Overlap computation and communication
   - Multi-buffer message passing

---

## Appendix A: File Manifest

### New Files (Total: 11 files)

```
butterfly-core/
  src/graph.rs                      (~400 lines)
  src/bsp.rs                        (~350 lines)
  src/computation.rs                (~200 lines)

butterfly-partition/
  src/graph.rs                      (~600 lines)

butterfly-coordination/
  src/bsp_coordinator.rs            (~450 lines)

butterfly-schedule/
  src/graph_scheduler.rs            (~300 lines)

butterfly-api/
  src/graph_api.rs                  (~250 lines)

butterfly-graph-algorithms/       (New Crate)
  src/lib.rs                        (~50 lines)
  src/pagerank.rs                   (~200 lines)
  src/shortest_path.rs              (~250 lines)
  src/connected_components.rs       (~180 lines)
```

### Modified Files (Total: 8 files)

```
butterfly-core/
  src/lib.rs                        (+20 lines)
  src/tensor.rs                     (+50 lines)

butterfly-partition/
  src/lib.rs                        (+30 lines)

butterfly-coordination/
  src/types.rs                      (+80 lines)
  src/barrier.rs                    (+60 lines)
  src/state_machine.rs              (+100 lines)
  src/lib.rs                        (+20 lines)

butterfly-comm/
  src/lib.rs                        (+40 lines)
```

### Total Code Addition

- **New code**: ~3,230 lines
- **Modified code**: ~400 lines
- **Total**: ~3,630 lines
- **Estimated effort**: 3-4 weeks for single developer

---

## Appendix B: Interface Specifications

### VertexCompute Trait (Full Specification)

```rust
pub trait VertexCompute: Send + Sync {
    /// Type of vertex value (e.g., f64 for PageRank, Vec<f64> for embeddings)
    type VertexValue: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Type of edge value (e.g., weight, capacity)
    type EdgeValue: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Type of message payload
    type MessageValue: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Main compute function called once per vertex per superstep
    ///
    /// # Arguments
    /// - `vertex_id`: ID of vertex being computed
    /// - `vertex_value`: Mutable reference to vertex's current value
    /// - `messages`: Messages received from previous superstep
    /// - `superstep`: Current superstep number (0-indexed)
    ///
    /// # Returns
    /// - `ComputeResult` with updated value, outgoing messages, halt vote
    fn compute(
        &self,
        vertex_id: VertexId,
        vertex_value: &mut Self::VertexValue,
        messages: Vec<VertexMessage<Self::MessageValue>>,
        superstep: u64,
    ) -> ComputeResult<Self::VertexValue, Self::MessageValue>;

    /// Initialize vertex value (called once at beginning)
    fn initial_value(&self, vertex_id: VertexId) -> Self::VertexValue;

    /// Optional: Combine messages for same target before delivery
    /// Default: no combining
    fn combine_messages(
        &self,
        messages: Vec<VertexMessage<Self::MessageValue>>,
    ) -> Vec<VertexMessage<Self::MessageValue>> {
        messages // No combining by default
    }

    /// Optional: Custom aggregator registration
    fn register_aggregators(&self) -> Vec<(&'static str, Box<dyn Aggregator>)> {
        vec![] // No aggregators by default
    }
}
```

### GraphPartitioner Trait (Full Specification)

```rust
pub trait GraphPartitioner: Send + Sync {
    /// Partition graph across available nodes
    ///
    /// # Arguments
    /// - `graph`: Input graph to partition
    /// - `nodes`: Available compute nodes with capabilities
    ///
    /// # Returns
    /// - Vector of subgraphs, one per partition
    /// - Error if partitioning fails (e.g., no nodes available)
    fn partition<V, E>(
        &self,
        graph: &Graph<V, E>,
        nodes: &[NodeCapability],
    ) -> Result<Vec<Subgraph<V, E>>, PartitionError>
    where
        V: Clone + Serialize + for<'de> Deserialize<'de>,
        E: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Estimate partition quality metrics
    fn estimate_quality<V, E>(
        &self,
        partitions: &[Subgraph<V, E>],
    ) -> GraphPartitionQuality;

    /// Optional: Re-partition based on runtime statistics
    fn repartition<V, E>(
        &self,
        current_partitions: &[Subgraph<V, E>],
        stats: &PartitionStats,
    ) -> Result<Vec<Subgraph<V, E>>, PartitionError>
    where
        V: Clone + Serialize + for<'de> Deserialize<'de>,
        E: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        // Default: no repartitioning
        Ok(current_partitions.to_vec())
    }
}
```

---

## Appendix C: Migration Path for Existing Code

**Key Principle**: Zero breaking changes to existing LLM inference functionality.

### Strategy

1. **Additive Changes Only**
   - All graph processing code is NEW modules
   - Existing `LayerInfo`, `ModelPartition` untouched
   - Coordination protocol extended, not replaced

2. **Shared Abstractions**
   ```rust
   // OLD: Transformer-specific
   pub trait InferenceNode { ... }

   // NEW: Unified (transformers impl this)
   pub trait DistributedComputation { ... }

   // BRIDGE: Both implement same underlying execution model
   impl DistributedComputation for TransformerModel { ... }
   impl DistributedComputation for GraphComputation { ... }
   ```

3. **Gradual Adoption**
   - Week 1-2: Add graph types (no impact on LLM code)
   - Week 3: Add partitioners (opt-in)
   - Week 4-5: Add BSP coordinator (parallel to existing coordinator)
   - Week 6+: Refactor LLM inference to use unified abstractions (optional)

---

## Summary

This plan provides a **concrete, implementable roadmap** to extend Butterfly with distributed graph processing capabilities:

1. **Current State**: Analyzed 8 crates, identified natural BSP mapping
2. **Architecture**: Unified DistributedComputation abstraction
3. **Data Structures**: Graph, Vertex, Edge, Subgraph with zero-copy views
4. **Partitioning**: Hash, METIS, Stream partitioners
5. **BSP Model**: VertexCompute trait, superstep execution, aggregation
6. **Integration**: 4-phase protocol maps to BSP supersteps
7. **Implementation**: 11 new files, 8 modified files, ~3,630 lines
8. **Timeline**: 4-6 weeks for MVP (PageRank working end-to-end)

**Next Steps**:
1. Review and approve plan
2. Create feature branch: `feature/graph-processing`
3. Start with Phase 1 (Core Infrastructure)
4. Iterate based on feedback and performance

**This plan is ready for implementation.** All interfaces are fully specified, file paths are concrete, and the integration strategy preserves existing LLM inference functionality.
