# Butterfly Graph Processing - User Guide

**Version**: 1.0
**Last Updated**: October 20, 2025
**Audience**: Developers using Butterfly for distributed graph algorithms

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Building Graphs](#building-graphs)
5. [Running Algorithms](#running-algorithms)
6. [Understanding Results](#understanding-results)
7. [Performance Tuning](#performance-tuning)
8. [Common Pitfalls](#common-pitfalls)
9. [Example Workflows](#example-workflows)
10. [API Reference](#api-reference)

---

## Introduction

Butterfly's graph processing system enables distributed execution of graph algorithms using the **Bulk Synchronous Parallel (BSP)** model. This system is designed for:

- **Large-scale graphs** that don't fit on a single machine
- **Iterative algorithms** like PageRank, shortest paths, and connected components
- **Fault-tolerant execution** with Byzantine agreement
- **High performance** through intelligent partitioning and message optimization

### Why Use Butterfly for Graph Processing?

- **Unified Architecture**: Same infrastructure powers both LLM inference and graph algorithms
- **Type Safety**: Rust's type system prevents common distributed computing bugs
- **Zero-Copy Operations**: Efficient memory management with `TensorRef`
- **Byzantine Fault Tolerance**: Built-in protection against node failures
- **Flexible Partitioning**: Choose from hash, stream, or METIS strategies

---

## Getting Started

### Prerequisites

```toml
# Add to your Cargo.toml
[dependencies]
butterfly-core = "0.1"
butterfly-partition = "0.1"
butterfly-graph = "0.1"
```

### Your First Graph Algorithm

```rust
use butterfly_core::{GraphBuilder, VertexId};
use butterfly_partition::HashPartitioner;
use butterfly_graph::PageRank;

fn main() {
    // 1. Build a simple graph
    let mut builder = GraphBuilder::new("web-graph");
    builder.directed()
        .add_vertex(VertexId(1))
        .add_vertex(VertexId(2))
        .add_vertex(VertexId(3))
        .add_edge(VertexId(1), VertexId(2))
        .add_edge(VertexId(2), VertexId(3))
        .add_edge(VertexId(3), VertexId(1));
    let graph = builder.build();

    // 2. Partition the graph
    let partitioner = HashPartitioner::new();
    let subgraphs = partitioner.partition(&graph, 2).unwrap();

    println!("Created {} partitions", subgraphs.len());
    println!("Edge cut: {}", subgraphs[0].edge_cut());

    // 3. Run PageRank (requires BSP coordinator - see Integration section)
    let pagerank = PageRank::with_defaults();
    // ... execute on distributed cluster
}
```

---

## Core Concepts

### Graphs

A **Graph** consists of:
- **Vertices**: Nodes with unique IDs and associated data
- **Edges**: Directed or undirected connections between vertices
- **Metadata**: Graph-level properties (directed/undirected, weighted, etc.)

```rust
use butterfly_core::{Graph, Vertex, Edge, VertexId, EdgeId, VertexData};

// Create vertices
let mut v1 = Vertex::new(VertexId(1));
v1.data.set_f64("pagerank", 1.0);

// Create edges
let edge = Edge::with_weight(EdgeId(0), VertexId(1), VertexId(2), 0.5);

// Build graph
let mut graph = Graph::new("my-graph");
graph.add_vertex(v1);
graph.add_edge(edge);
```

### BSP Model

The **Bulk Synchronous Parallel** model divides computation into **supersteps**:

1. **Local Computation**: Each vertex processes incoming messages
2. **Message Passing**: Vertices send messages to neighbors
3. **Barrier Synchronization**: All workers wait for each other
4. **Repeat**: Continue until convergence

```
Superstep 0          Superstep 1          Superstep 2
┌────────────┐      ┌────────────┐      ┌────────────┐
│ Compute    │ ───> │ Compute    │ ───> │ Compute    │
│ + Messages │      │ + Messages │      │ + Messages │
└────────────┘      └────────────┘      └────────────┘
      │                    │                    │
   Barrier             Barrier             Barrier
      │                    │                    │
      ▼                    ▼                    ▼
```

### Partitioning

**Partitioning** divides a graph into subgraphs for distribution:

```rust
use butterfly_partition::{HashPartitioner, StreamPartitioner, GraphPartitioner};

// Hash partitioning (fast, simple)
let hash = HashPartitioner::new();
let subgraphs = hash.partition(&graph, 4)?;

// Stream partitioning (better edge cuts)
let stream = StreamPartitioner::new();
let subgraphs = stream.partition(&graph, 4)?;

// Check quality
let quality = hash.estimate_quality(&subgraphs);
println!("Edge cut: {}", quality.edge_cut);
println!("Load balance: {:.2}", quality.load_balance);
```

### Messages

Vertices communicate via **typed messages**:

```rust
use butterfly_core::{Message, MessagePayload, VertexId};

// Send a numeric value
let msg = Message::new(
    VertexId(1),
    VertexId(2),
    MessagePayload::F64(0.25),
);

// Extract payload
if let Some(value) = msg.payload.as_f64() {
    println!("Received: {}", value);
}
```

---

## Building Graphs

### Using GraphBuilder

The **fluent API** for graph construction:

```rust
use butterfly_core::{GraphBuilder, VertexId, VertexData};

let mut builder = GraphBuilder::new("social-network");

// Configure graph type
builder.undirected()  // or .directed()
    .weighted();      // optional

// Add vertices with data
let mut user1_data = VertexData::new();
user1_data.set_u64("followers", 1000);
builder.add_vertex_with_data(VertexId(1), user1_data);

// Add edges (vertices auto-created if needed)
builder.add_edge(VertexId(1), VertexId(2))
    .add_edge(VertexId(2), VertexId(3))
    .add_weighted_edge(VertexId(1), VertexId(3), 0.8);

let graph = builder.build();
```

### Loading from Files

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};
use butterfly_core::{GraphBuilder, VertexId};

fn load_edge_list(path: &str) -> std::io::Result<butterfly_core::Graph> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut builder = GraphBuilder::new("loaded-graph");
    builder.directed();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') { continue; }  // Skip comments

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let src = VertexId(parts[0].parse().unwrap());
            let dst = VertexId(parts[1].parse().unwrap());
            builder.add_edge(src, dst);
        }
    }

    Ok(builder.build())
}

// Usage
let graph = load_edge_list("data/web-graph.txt")?;
println!("Loaded {} vertices", graph.vertex_count());
```

### Programmatic Generation

```rust
use butterfly_core::{GraphBuilder, VertexId};

// Generate a ring graph: 1 -> 2 -> 3 -> ... -> N -> 1
fn create_ring_graph(n: usize) -> butterfly_core::Graph {
    let mut builder = GraphBuilder::new("ring");
    builder.directed();

    for i in 0..n {
        let src = VertexId(i as u64);
        let dst = VertexId(((i + 1) % n) as u64);
        builder.add_edge(src, dst);
    }

    builder.build()
}

// Generate a complete graph (all vertices connected)
fn create_complete_graph(n: usize) -> butterfly_core::Graph {
    let mut builder = GraphBuilder::new("complete");
    builder.undirected();

    for i in 0..n {
        for j in (i+1)..n {
            builder.add_edge(VertexId(i as u64), VertexId(j as u64));
        }
    }

    builder.build()
}
```

---

## Running Algorithms

### PageRank

Compute vertex importance based on link structure:

```rust
use butterfly_graph::{PageRank, PageRankConfig};
use butterfly_core::VertexCompute;

// Default configuration (damping=0.85, threshold=0.001)
let pagerank = PageRank::with_defaults();

// Custom configuration
let config = PageRankConfig {
    damping_factor: 0.9,
    convergence_threshold: 0.0001,
    max_iterations: 200,
};
let custom_pagerank = PageRank::new(config);

// Initialize vertices
for vertex in graph.vertices.values_mut() {
    custom_pagerank.init(vertex).unwrap();
}

// Compute (in BSP loop - simplified)
let messages = vec![]; // Messages from previous superstep
let result = pagerank.compute(&vertex, &messages, superstep)?;

// Access results
let rank = result.new_value.get_f64("rank").unwrap();
println!("Vertex {} has rank {:.4}", vertex.id.0, rank);
```

### Connected Components

Find clusters in undirected graphs:

```rust
use butterfly_graph::ConnectedComponents;

let cc = ConnectedComponents::new();

// Initialize
for vertex in graph.vertices.values_mut() {
    cc.init(vertex).unwrap();
}

// After convergence, each vertex has a component ID
let component_id = vertex.data.get_u64("component").unwrap();
println!("Vertex {} is in component {}", vertex.id.0, component_id);

// Count components
use std::collections::HashSet;
let components: HashSet<u64> = graph.vertices.values()
    .filter_map(|v| v.data.get_u64("component"))
    .collect();
println!("Found {} components", components.len());
```

### Single-Source Shortest Paths

Find shortest distances from a source vertex:

```rust
use butterfly_graph::SSSP;
use butterfly_core::VertexId;

let sssp = SSSP::new(VertexId(1)); // Source vertex

// After convergence
let distance = vertex.data.get_f64("distance").unwrap();
if distance == f64::INFINITY {
    println!("Vertex {} is unreachable", vertex.id.0);
} else {
    println!("Distance to {}: {:.2}", vertex.id.0, distance);
}
```

### Custom Algorithms

Implement the `VertexCompute` trait:

```rust
use butterfly_core::{
    Vertex, VertexCompute, Message, MessagePayload,
    ComputationResult, ComputeError, VertexData,
};

struct MyAlgorithm;

impl VertexCompute for MyAlgorithm {
    fn compute(
        &self,
        vertex: &Vertex,
        messages: &[Message],
        superstep: u32,
    ) -> Result<ComputationResult, ComputeError> {
        // Your logic here
        let mut data = VertexData::new();
        data.set_f64("value", 42.0);

        // Send messages to neighbors
        let outgoing: Vec<Message> = vertex.neighbors.iter()
            .map(|&neighbor| Message::new(
                vertex.id,
                neighbor,
                MessagePayload::F64(1.0),
            ))
            .collect();

        Ok(ComputationResult::new(data)
            .with_messages(outgoing)
            .active())
    }

    fn init(&mut self, vertex: &mut Vertex) -> Result<(), ComputeError> {
        vertex.data.set_f64("value", 0.0);
        vertex.activate();
        Ok(())
    }

    fn should_terminate(&self, superstep: u32, active_vertices: usize) -> bool {
        active_vertices == 0 || superstep > 100
    }
}
```

---

## Understanding Results

### Vertex Data

Access computed values:

```rust
// PageRank results
let rank = vertex.data.get_f64("rank").unwrap();
let delta = vertex.data.get_f64("delta").unwrap();

// Connected Components results
let component = vertex.data.get_u64("component").unwrap();

// SSSP results
let distance = vertex.data.get_f64("distance").unwrap();
```

### Partition Quality Metrics

Evaluate partition effectiveness:

```rust
use butterfly_partition::GraphPartitionQuality;

let quality = GraphPartitionQuality::from_subgraphs(&subgraphs);

println!("Partition Quality:");
println!("  Edge Cut: {} edges", quality.edge_cut);
println!("  Load Balance: {:.2}%", quality.load_balance * 100.0);
println!("  Communication Volume: {} vertices", quality.communication_volume);
println!("  Replication Factor: {:.2}x", quality.replication_factor);
```

**Interpreting Metrics**:

- **Edge Cut**: Lower is better (fewer cross-partition edges = less communication)
- **Load Balance**: Higher is better (1.0 = perfect balance, 0.5 = one partition has 2x vertices)
- **Communication Volume**: Number of boundary vertices needing message passing
- **Replication Factor**: Ratio of ghost to local vertices (overhead measure)

### Convergence Monitoring

Track algorithm progress:

```rust
use butterfly_core::AggregatedResult;

let aggregation = pagerank.aggregate(&results);

println!("Superstep {} Stats:", superstep);
println!("  Active Vertices: {}", aggregation.active_count);
println!("  Messages Sent: {}", aggregation.messages_sent);
println!("  Avg Delta: {:.6}",
    aggregation.aggregations.get("avg_delta").unwrap_or(&0.0));

// Check convergence
if aggregation.active_count == 0 {
    println!("Algorithm converged!");
}
```

---

## Performance Tuning

### Choose the Right Partitioner

```rust
// Small graphs (<10K vertices): Hash is fastest
let partitioner = HashPartitioner::new();

// Medium graphs (10K-1M vertices): Stream balances quality and speed
let partitioner = StreamPartitioner::new()
    .with_lookahead(200);  // Tune lookahead for quality

// Large graphs (>1M vertices): METIS for best quality (when implemented)
let partitioner = METISPartitioner::new();
```

**Partitioner Comparison**:

| Partitioner | Speed      | Edge Cut Quality | Use Case                |
|-------------|------------|------------------|-------------------------|
| Hash        | ⚡⚡⚡      | ⭐               | Fast prototyping        |
| Stream      | ⚡⚡        | ⭐⭐⭐           | Production workloads    |
| METIS       | ⚡          | ⭐⭐⭐⭐⭐       | Minimize communication  |

### Message Optimization

Use combiners to reduce message volume:

```rust
use butterfly_core::{MessageCombiner, SumCombiner};

// For PageRank: combine multiple rank contributions
let combiner = SumCombiner;
let messages = vec![/* many messages to same target */];
let combined = combiner.combine(messages)?;

// Result: One message instead of N
```

### Memory Management

Control graph representation:

```rust
// Compact representation for large graphs
use butterfly_core::GraphView;

let view = GraphView::from_csr(
    row_offsets,      // CSR row pointer array
    column_indices,   // CSR column array
    Some(edge_weights) // Optional edge data
);

// Zero-copy neighbor access
let neighbors = view.neighbors(vertex_idx);
```

### Partition Size Tuning

```rust
// Rule of thumb: num_partitions = num_worker_nodes
let num_workers = 8;
let subgraphs = partitioner.partition(&graph, num_workers)?;

// For very large graphs: overpartition for better load balancing
let num_partitions = num_workers * 4;
let subgraphs = partitioner.partition(&graph, num_partitions)?;
```

### Convergence Tuning

Adjust algorithm parameters for faster convergence:

```rust
// PageRank: Higher damping = slower convergence but more stable
let config = PageRankConfig {
    damping_factor: 0.85,      // Default
    convergence_threshold: 0.001,  // Looser threshold = faster
    max_iterations: 50,        // Safety limit
};

// Trade-off: Tighter threshold (0.0001) = more accurate, slower
```

---

## Common Pitfalls

### 1. Forgetting to Set Graph Direction

**Problem**: Edges not bidirectional when expected

```rust
// ❌ Wrong: Directed graph for social network
let mut builder = GraphBuilder::new("social");
builder.add_edge(VertexId(1), VertexId(2)); // Only 1->2

// ✅ Correct: Undirected for mutual friendships
builder.undirected()
    .add_edge(VertexId(1), VertexId(2)); // Both 1->2 and 2->1
```

### 2. Partition Count Mismatch

**Problem**: More partitions than worker nodes

```rust
// ❌ Wrong: 16 partitions but only 4 workers
let subgraphs = partitioner.partition(&graph, 16)?;
deploy_to_workers(&subgraphs, 4); // Some workers handle multiple partitions

// ✅ Correct: Match partition count to workers
let subgraphs = partitioner.partition(&graph, 4)?;
```

### 3. Ignoring Vertex Initialization

**Problem**: Algorithm starts with uninitialized data

```rust
// ❌ Wrong: Skip initialization
let result = pagerank.compute(&vertex, &messages, 0)?; // Panic!

// ✅ Correct: Initialize before computing
pagerank.init(&mut vertex)?;
let result = pagerank.compute(&vertex, &messages, 0)?;
```

### 4. Message Type Mismatch

**Problem**: Extracting wrong payload type

```rust
// ❌ Wrong: Algorithm sends F64, code expects U64
let value = message.payload.as_u64(); // Returns None!

// ✅ Correct: Match algorithm's message type
let value = message.payload.as_f64().unwrap();
```

### 5. Not Checking for Dead-End Vertices

**Problem**: Vertices with no outgoing edges cause issues

```rust
// ❌ Wrong: Divide by zero
let contribution = rank / vertex.degree() as f64; // Panic if degree = 0!

// ✅ Correct: Handle dead-ends
if vertex.degree() > 0 {
    let contribution = rank / vertex.degree() as f64;
    // send messages
} else {
    // vertex votes to halt
}
```

### 6. Excessive Message Creation

**Problem**: Creating millions of redundant messages

```rust
// ❌ Wrong: Send message every superstep even if unchanged
for neighbor in &vertex.neighbors {
    messages.push(Message::new(vertex.id, *neighbor, payload));
}

// ✅ Correct: Only send when value changes
if delta > threshold {
    for neighbor in &vertex.neighbors {
        messages.push(Message::new(vertex.id, *neighbor, payload));
    }
}
```

---

## Example Workflows

### Workflow 1: Analyze Web Graph

**Goal**: Find most important pages in a web crawl

```rust
use butterfly_core::{GraphBuilder, VertexId};
use butterfly_partition::StreamPartitioner;
use butterfly_graph::PageRank;

fn analyze_web_graph() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load web graph from crawl data
    let graph = load_edge_list("data/web-crawl.txt")?;
    println!("Loaded {} pages, {} links",
        graph.vertex_count(), graph.edge_count());

    // 2. Partition for distributed processing
    let partitioner = StreamPartitioner::new();
    let subgraphs = partitioner.partition(&graph, 8)?;

    let quality = partitioner.estimate_quality(&subgraphs);
    println!("Partition quality: {:.2}% balanced, {} edge cuts",
        quality.load_balance * 100.0, quality.edge_cut);

    // 3. Initialize PageRank
    let mut pagerank = PageRank::with_defaults();

    // 4. Simulate BSP execution (pseudo-code)
    // In production, this runs on distributed workers
    let mut superstep = 0;
    loop {
        // Execute compute on all vertices
        let mut all_inactive = true;

        for vertex in graph.vertices.values() {
            let messages = vec![]; // Collect from previous superstep
            let result = pagerank.compute(vertex, &messages, superstep)?;

            if result.is_active {
                all_inactive = false;
            }
        }

        superstep += 1;
        if all_inactive || superstep > 100 {
            break;
        }
    }

    // 5. Extract top pages
    let mut pages: Vec<_> = graph.vertices.values()
        .map(|v| (v.id, v.data.get_f64("rank").unwrap_or(0.0)))
        .collect();
    pages.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 10 pages:");
    for (i, (id, rank)) in pages.iter().take(10).enumerate() {
        println!("  {}. Vertex {} - Rank: {:.6}", i+1, id.0, rank);
    }

    Ok(())
}
```

### Workflow 2: Social Network Analysis

**Goal**: Find communities in a friendship network

```rust
use butterfly_graph::ConnectedComponents;

fn find_communities(graph: &butterfly_core::Graph) {
    // Must be undirected for meaningful communities
    assert!(!graph.metadata.directed, "Graph must be undirected");

    let mut cc = ConnectedComponents::new();

    // Initialize all vertices
    for vertex in graph.vertices.values() {
        cc.init(&mut vertex.clone()).unwrap();
    }

    // Run until convergence (pseudo-code)
    // ... BSP execution ...

    // Analyze results
    use std::collections::HashMap;
    let mut components: HashMap<u64, Vec<u64>> = HashMap::new();

    for vertex in graph.vertices.values() {
        let comp_id = vertex.data.get_u64("component").unwrap();
        components.entry(comp_id).or_default().push(vertex.id.0);
    }

    println!("Found {} communities", components.len());

    // Find largest community
    let largest = components.values()
        .max_by_key(|v| v.len())
        .unwrap();
    println!("Largest community has {} members", largest.len());
}
```

### Workflow 3: Road Network Routing

**Goal**: Shortest paths from city center to all locations

```rust
use butterfly_graph::SSSP;
use butterfly_core::VertexId;

fn compute_distances() {
    // Load road network (edges = roads, weights = distances)
    let mut builder = GraphBuilder::new("roads");
    builder.directed().weighted();

    // Add roads with distances in km
    builder.add_weighted_edge(VertexId(1), VertexId(2), 5.2);
    builder.add_weighted_edge(VertexId(2), VertexId(3), 3.7);
    // ...

    let graph = builder.build();

    // Compute shortest paths from city center (vertex 1)
    let sssp = SSSP::new(VertexId(1));

    // ... BSP execution ...

    // Query results
    for vertex in graph.vertices.values() {
        let dist = vertex.data.get_f64("distance").unwrap();

        if dist == f64::INFINITY {
            println!("Location {} is unreachable", vertex.id.0);
        } else {
            println!("Location {}: {:.1} km from center", vertex.id.0, dist);
        }
    }
}
```

---

## API Reference

### Key Types

**`butterfly_core::Graph`**
```rust
pub struct Graph {
    pub vertices: HashMap<VertexId, Vertex>,
    pub edges: HashMap<EdgeId, Edge>,
    pub metadata: GraphMetadata,
}

impl Graph {
    pub fn new(name: impl Into<String>) -> Self;
    pub fn vertex_count(&self) -> usize;
    pub fn edge_count(&self) -> usize;
    pub fn get_vertex(&self, id: VertexId) -> Option<&Vertex>;
}
```

**`butterfly_core::Vertex`**
```rust
pub struct Vertex {
    pub id: VertexId,
    pub data: VertexData,
    pub neighbors: Vec<VertexId>,
    pub active: bool,
}

impl Vertex {
    pub fn new(id: VertexId) -> Self;
    pub fn add_neighbor(&mut self, neighbor: VertexId);
    pub fn degree(&self) -> usize;
    pub fn is_active(&self) -> bool;
}
```

**`butterfly_core::VertexCompute`**
```rust
pub trait VertexCompute: Send + Sync {
    fn compute(
        &self,
        vertex: &Vertex,
        messages: &[Message],
        superstep: u32,
    ) -> Result<ComputationResult, ComputeError>;

    fn init(&mut self, vertex: &mut Vertex) -> Result<(), ComputeError>;
    fn should_terminate(&self, superstep: u32, active_vertices: usize) -> bool;
}
```

**`butterfly_partition::GraphPartitioner`**
```rust
pub trait GraphPartitioner: Send + Sync {
    fn partition(
        &self,
        graph: &Graph,
        num_partitions: usize,
    ) -> Result<Vec<Subgraph>, PartitionError>;

    fn estimate_quality(&self, subgraphs: &[Subgraph]) -> GraphPartitionQuality;
}
```

### See Also

- [Graph Architecture Guide](./GRAPH_ARCHITECTURE.md) - System design and internals
- [Graph Performance Guide](./GRAPH_PERFORMANCE.md) - Benchmarks and optimization
- [Implementation Summary](../IMPLEMENTATION_SUMMARY.md) - Development details
- [Graph Processing Plan](./GRAPH_PROCESSING_PLAN.md) - Complete specification

---

**Questions?** Open an issue at [github.com/butterfly/issues](https://github.com/butterfly/issues)
