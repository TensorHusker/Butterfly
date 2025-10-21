# Graph Processing Improvements Plan

**Date**: 2025-10-20
**Branch**: `feature/distributed-graph-processing`
**Status**: Planning Phase

---

## Executive Summary

The current implementation provides a solid foundation for distributed graph processing with:
- ✅ Core graph data structures (421 lines)
- ✅ BSP computation model (370 lines)
- ✅ Three partitioning algorithms (550 lines)
- ✅ Three reference algorithms (PageRank, CC, SSSP)
- ✅ 21+ unit tests, all passing

**Critical Gap**: The BSP model is defined but **not integrated with distributed execution**. We have the "what" but not the "how" for running graph algorithms across multiple nodes.

This plan addresses the gap by implementing the **missing execution layer**.

---

## Priority 0: Core Execution Infrastructure (CRITICAL)

### 1. BSP Executor (`butterfly-core/src/bsp_executor.rs`)

**Purpose**: Execute BSP supersteps on a local partition/subgraph

**Key Responsibilities**:
- Run `VertexCompute::compute()` on all active vertices in partition
- Manage incoming/outgoing message buffers
- Track active vertex count for local termination
- Aggregate local results for coordinator
- Handle vertex state updates

**Interface**:
```rust
pub struct BSPExecutor<C: VertexCompute> {
    subgraph: Subgraph,
    compute: C,
    message_buffer: MessageBuffer,
    active_vertices: HashSet<VertexId>,
}

impl<C: VertexCompute> BSPExecutor<C> {
    pub fn new(subgraph: Subgraph, compute: C) -> Self;

    /// Execute one superstep on local partition
    pub async fn execute_superstep(
        &mut self,
        superstep: u32,
    ) -> Result<SuperstepResult, ExecutionError>;

    /// Deliver incoming messages from other partitions
    pub fn deliver_messages(&mut self, messages: Vec<Message>);

    /// Extract outgoing messages for routing
    pub fn take_outgoing_messages(&mut self) -> Vec<Message>;

    /// Check if local partition has converged
    pub fn is_locally_converged(&self) -> bool;
}
```

**Estimated Lines**: ~300 lines
**Dependencies**: `Subgraph`, `VertexCompute`, `MessageBuffer`, `SuperstepResult`

---

### 2. BSP Coordinator (`butterfly-coordination/src/bsp_coordinator.rs`)

**Purpose**: Orchestrate BSP execution across distributed partitions

**Key Responsibilities**:
- Coordinate superstep execution across workers
- Route messages between partitions
- Perform global termination detection
- Aggregate results from all partitions
- Integrate with existing Byzantine coordination protocol

**Interface**:
```rust
pub struct BSPCoordinator {
    /// Map of partition ID to worker node
    partition_assignments: HashMap<PartitionId, NodeId>,
    /// Current superstep number
    current_superstep: u32,
    /// Termination detector for global convergence
    termination_detector: TerminationDetector,
    /// Communication backend for message routing
    comm_backend: Arc<dyn CommunicationBackend>,
}

impl BSPCoordinator {
    pub fn new(
        partition_assignments: HashMap<PartitionId, NodeId>,
        comm_backend: Arc<dyn CommunicationBackend>,
    ) -> Self;

    /// Run complete BSP execution until convergence
    pub async fn execute<C: VertexCompute>(
        &mut self,
        compute: C,
        max_supersteps: u32,
    ) -> Result<ExecutionResult, CoordinationError>;

    /// Execute a single superstep across all partitions
    async fn execute_superstep(&mut self) -> Result<SuperstepResult, CoordinationError>;

    /// Route messages between partitions
    async fn route_messages(&mut self, messages: Vec<Message>) -> Result<(), CoordinationError>;

    /// Check global termination condition
    fn check_global_termination(&mut self) -> bool;

    /// Aggregate results from all workers
    async fn aggregate_results(&mut self) -> AggregatedResult;
}
```

**Estimated Lines**: ~450 lines
**Dependencies**: Existing `CoordinationStateMachine`, `BarrierCoordinator`, `CommunicationBackend`

---

### 3. Message Router (extend `butterfly-comm`)

**Purpose**: Efficiently route messages between partitions

**Key Enhancements**:
- Add `GraphMessage` variant to `Message` enum
- Implement partition-aware routing
- Batch messages for efficiency
- Handle ghost vertex updates

**Changes to `butterfly-comm/src/lib.rs`**:
```rust
// Add new message variant
pub enum Message {
    TensorData { /* existing */ },
    Heartbeat { /* existing */ },
    Ack { /* existing */ },

    // NEW: Graph processing messages
    GraphMessage {
        from_partition: PartitionId,
        to_partition: PartitionId,
        superstep: u32,
        messages: Vec<VertexMessage>, // Batched vertex messages
    },
}

// NEW: Message router for graph processing
pub struct GraphMessageRouter {
    comm_backend: Arc<dyn CommunicationBackend>,
    partition_to_node: HashMap<PartitionId, NodeId>,
}

impl GraphMessageRouter {
    /// Route messages to their destination partitions
    pub async fn route_messages(
        &mut self,
        messages: Vec<VertexMessage>,
    ) -> Result<(), CommunicationError>;

    /// Batch messages by destination partition
    fn batch_by_partition(&self, messages: Vec<VertexMessage>) -> HashMap<PartitionId, Vec<VertexMessage>>;
}
```

**Estimated Lines**: ~200 lines
**Dependencies**: Existing `Message`, `CommunicationBackend`

---

### 4. Termination Detector

**Purpose**: Detect when BSP computation has globally converged

**Algorithm**: Global AND of local termination votes

**Implementation** (add to `butterfly-coordination/src/termination.rs`):
```rust
pub struct TerminationDetector {
    /// Local termination votes from each partition
    local_votes: HashMap<PartitionId, bool>,
    /// Number of partitions required for quorum
    quorum_size: usize,
    /// Current superstep
    current_superstep: u32,
}

impl TerminationDetector {
    pub fn new(num_partitions: usize) -> Self;

    /// Record termination vote from a partition
    pub fn record_vote(&mut self, partition: PartitionId, vote_to_halt: bool);

    /// Check if global termination condition is met
    pub fn is_globally_terminated(&self) -> bool;

    /// Reset for next superstep
    pub fn reset(&mut self);
}
```

**Estimated Lines**: ~100 lines

---

## Priority 1: Integration & Testing (HIGH PRIORITY)

### 5. Integration Tests (`tests/graph_integration_test.rs`)

**Purpose**: Validate end-to-end graph algorithm execution

**Test Cases**:
1. **Single-Node PageRank**
   - Small graph (10 vertices)
   - Verify ranks sum to graph size
   - Compare against reference implementation

2. **Multi-Partition PageRank**
   - Partition graph across 2 workers
   - Verify message passing works
   - Check convergence within 30 iterations

3. **Connected Components**
   - Disconnected graph with 3 components
   - Verify correct component labeling

4. **Partition Quality**
   - Compare Hash vs Stream partitioner
   - Measure edge cut ratio
   - Validate load balance

**Estimated Lines**: ~400 lines

---

### 6. Graph I/O (`butterfly-core/src/graph_io.rs`)

**Purpose**: Load graphs from common formats

**Supported Formats**:
1. **Edge List**: `source target [weight]`
2. **Adjacency List**: `vertex: neighbor1 neighbor2 ...`
3. **Matrix Market**: Standard sparse matrix format

**Interface**:
```rust
pub struct GraphLoader;

impl GraphLoader {
    /// Load graph from edge list file
    pub fn from_edge_list(path: &Path, directed: bool) -> Result<Graph, IoError>;

    /// Load graph from adjacency list file
    pub fn from_adjacency_list(path: &Path) -> Result<Graph, IoError>;

    /// Export graph to edge list
    pub fn to_edge_list(graph: &Graph, path: &Path) -> Result<(), IoError>;
}

pub struct ResultExporter;

impl ResultExporter {
    /// Export vertex results to CSV
    pub fn export_vertex_data(
        graph: &Graph,
        key: &str,
        path: &Path,
    ) -> Result<(), IoError>;
}
```

**Estimated Lines**: ~250 lines

---

## Priority 2: Usability & Performance (MEDIUM PRIORITY)

### 7. CLI Commands (extend `butterfly-cli`)

**New Commands**:
```bash
# Run PageRank on a graph
butterfly-cli graph run pagerank --input graph.txt --output ranks.csv --iterations 30

# Partition a graph
butterfly-cli graph partition --input graph.txt --partitions 4 --strategy stream

# Show graph statistics
butterfly-cli graph stats --input graph.txt
```

**Implementation**:
- Add `graph` subcommand to CLI
- Integrate with graph loader
- Support multiple algorithms
- Output results in CSV/JSON

**Estimated Lines**: ~300 lines

---

### 8. Performance Benchmarks (`benches/graph_benchmarks.rs`)

**Benchmark Scenarios**:
1. **Partitioning Speed**
   - Hash vs Stream vs METIS
   - Various graph sizes (1K, 10K, 100K vertices)

2. **PageRank Throughput**
   - Single-node baseline
   - 2/4/8 partitions scaling
   - Measure edges processed per second

3. **Message Routing Overhead**
   - Latency per message
   - Batch size impact
   - Network utilization

**Estimated Lines**: ~350 lines

---

### 9. Enhanced METIS Partitioner

**Current Status**: Placeholder (falls back to Stream partitioner)

**Improvement**:
- Integrate with `metis-rs` crate
- Implement multilevel k-way partitioning
- Add coarsening/refinement phases
- Target: 3-5x better edge cuts than Stream

**Estimated Lines**: ~400 lines

---

## Implementation Strategy

### Phase 1: Core Execution (Week 1)
1. **Day 1-2**: BSP Executor
   - Local superstep execution
   - Message buffer management
   - Unit tests

2. **Day 3-4**: Message Router
   - GraphMessage variant
   - Partition-aware routing
   - Batching logic

3. **Day 5-7**: BSP Coordinator
   - Superstep orchestration
   - Termination detection
   - Integration with existing coordination

**Deliverable**: Working distributed BSP execution framework

---

### Phase 2: Integration & Testing (Week 2)
1. **Day 1-3**: Integration Tests
   - Single-node PageRank test
   - Multi-partition PageRank test
   - Convergence validation

2. **Day 4-5**: Graph I/O
   - Edge list loader
   - Result exporter
   - Test data generation

3. **Day 6-7**: End-to-End Validation
   - Run on real datasets (Web-Google subset)
   - Performance profiling
   - Bug fixes

**Deliverable**: Validated end-to-end graph processing pipeline

---

### Phase 3: Usability (Week 3)
1. **Day 1-3**: CLI Integration
   - Graph subcommands
   - Multiple algorithm support
   - User-friendly output

2. **Day 4-5**: Benchmarks
   - Performance baselines
   - Scalability tests
   - Optimization opportunities

3. **Day 6-7**: Documentation
   - User guide
   - API documentation
   - Performance tuning guide

**Deliverable**: Production-ready graph processing system

---

## Success Criteria

### Functional Requirements
- ✅ PageRank runs on 4-node cluster with correct results
- ✅ Convergence within 30 iterations for typical graphs
- ✅ All integration tests pass
- ✅ CLI can load graphs and run algorithms

### Performance Requirements
- ✅ Superstep latency < 100ms for 10K vertex graph
- ✅ Throughput > 100K edges/sec per node
- ✅ Coordination overhead < 10% of total execution time
- ✅ Linear scaling up to 8 nodes

### Quality Requirements
- ✅ >80% test coverage for new code
- ✅ Zero compiler warnings
- ✅ All clippy lints pass
- ✅ Comprehensive rustdoc documentation

---

## Risk Mitigation

### Risk 1: Coordination Protocol Complexity
**Mitigation**: Reuse existing Byzantine coordination infrastructure, extend incrementally

### Risk 2: Message Routing Performance
**Mitigation**: Implement batching, benchmark early, optimize hot paths

### Risk 3: Termination Detection Reliability
**Mitigation**: Thorough testing with edge cases (disconnected graphs, cyclic dependencies)

### Risk 4: Integration with Existing Code
**Mitigation**: Maintain backward compatibility, use feature flags for experimental features

---

## Code Size Estimate

| Component              | Estimated Lines | Status      |
|------------------------|-----------------|-------------|
| BSP Executor           | 300             | Not Started |
| BSP Coordinator        | 450             | Not Started |
| Message Router         | 200             | Not Started |
| Termination Detector   | 100             | Not Started |
| Integration Tests      | 400             | Not Started |
| Graph I/O              | 250             | Not Started |
| CLI Commands           | 300             | Not Started |
| Benchmarks             | 350             | Not Started |
| METIS Partitioner      | 400             | Not Started |
| **Total New Code**     | **2,750**       |             |

---

## Next Steps

1. ✅ Review and approve this plan
2. ⏳ Use Lyra to optimize implementation prompts
3. ⏳ Use Ultrathink to coordinate implementation
4. ⏳ Implement Phase 1 (Core Execution)
5. ⏳ Implement Phase 2 (Integration & Testing)
6. ⏳ Implement Phase 3 (Usability)

---

**This plan transforms the graph processing foundation into a production-ready distributed system.**
