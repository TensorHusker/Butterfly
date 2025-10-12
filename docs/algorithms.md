# Butterfly Algorithms

## Overview

This document describes the core algorithms that power Butterfly's distributed inference capabilities. The system relies on three fundamental algorithmic domains:

1. **Model Partitioning**: How to split a model across workers
2. **Communication Protocols**: How workers coordinate execution
3. **Scheduling Strategies**: How to optimize resource utilization

## Model Partitioning Algorithms

### Problem Statement

Given:
- A computational graph G = (V, E) representing a neural network
- A set of workers W = {w₁, w₂, ..., wₙ}
- Resource constraints R (memory, compute, bandwidth)

Find:
- A partition P: V → W that assigns each vertex (layer/operation) to a worker
- Minimizing: Total execution time
- Subject to: Resource constraints are satisfied

### Partitioning Strategies

#### 1. Layer-Wise Partitioning (Implemented)

**Algorithm**: Assign consecutive layers to workers sequentially.

```
Input: Model graph G with layers L = [l₁, l₂, ..., lₘ], Workers W
Output: Partition assignment P

function LayerWisePartition(G, W):
    partition = {}
    layers_per_worker = ceiling(len(L) / len(W))

    for i, layer in enumerate(L):
        worker_idx = i // layers_per_worker
        partition[layer] = W[worker_idx]

    return partition
```

**Complexity**: O(m) where m is number of layers

**Advantages**:
- Simple to implement
- Predictable communication patterns
- Works well for sequential models (ResNet, VGG)

**Disadvantages**:
- Doesn't account for layer heterogeneity
- May create imbalanced workloads
- Ignores network topology

**Use Cases**: Initial implementation, simple feedforward networks

#### 2. Load-Balanced Partitioning (Planned)

**Algorithm**: Assign layers to balance computational load across workers.

```
Input: Model graph G, estimated compute costs C, Workers W
Output: Partition assignment P

function LoadBalancedPartition(G, C, W):
    partition = {}
    worker_loads = [0] * len(W)

    # Sort layers by computational cost (descending)
    sorted_layers = sort_by_cost(G.layers, C, descending=True)

    for layer in sorted_layers:
        # Assign to least loaded worker
        min_worker = argmin(worker_loads)
        partition[layer] = W[min_worker]
        worker_loads[min_worker] += C[layer]

    return partition
```

**Complexity**: O(m log m) for sorting + O(m log n) for heap operations

**Advantages**:
- Balances workload across heterogeneous layers
- Adapts to different hardware capabilities
- Reduces stragglers

**Disadvantages**:
- Requires accurate cost estimation
- May increase communication overhead
- Doesn't preserve layer locality

**Use Cases**: Heterogeneous models, varying hardware

#### 3. Topology-Aware Partitioning (Planned)

**Algorithm**: Minimize communication cost by considering network topology.

```
Input: Model graph G, Network topology T, Workers W
Output: Partition assignment P

function TopologyAwarePartition(G, T, W):
    # Build communication cost matrix
    comm_cost = build_comm_cost_matrix(G, T)

    # Use graph partitioning (e.g., METIS)
    partition = metis_partition(G, len(W), comm_cost)

    # Map partitions to workers
    P = assign_partitions_to_workers(partition, W, T)

    return P
```

**Complexity**: O(m + e) where e is number of edges (METIS is near-linear)

**Advantages**:
- Minimizes inter-worker communication
- Respects network bandwidth constraints
- Good for models with complex connectivity

**Disadvantages**:
- Requires network topology knowledge
- More complex implementation
- May create unbalanced loads

**Use Cases**: High-latency networks, complex architectures (Transformers)

#### 4. Pipeline Parallelism Partitioning (Future)

**Algorithm**: Create pipeline stages with balanced throughput.

```
Input: Model graph G, Target throughput T, Workers W
Output: Partition P with stage assignment

function PipelinePartition(G, T, W):
    stages = []
    current_stage = []
    current_cost = 0
    stage_target = estimate_stage_cost(G, T, len(W))

    for layer in G.layers:
        current_stage.append(layer)
        current_cost += compute_cost(layer)

        if current_cost >= stage_target:
            stages.append(current_stage)
            current_stage = []
            current_cost = 0

    if current_stage:
        stages.append(current_stage)

    # Assign stages to workers
    return assign_stages_to_workers(stages, W)
```

**Advantages**:
- Enables multiple requests in flight
- Improves throughput for batch inference
- Natural for streaming applications

**Disadvantages**:
- Requires careful bubble management
- More complex scheduling
- Higher memory usage

### Partition Quality Metrics

```rust
struct PartitionQuality {
    // Load balance: Standard deviation of worker loads
    load_balance: f64,

    // Communication volume: Total bytes transferred
    communication_volume: u64,

    // Critical path length: Maximum sequential dependencies
    critical_path_length: usize,

    // Memory efficiency: Peak memory usage
    peak_memory: u64,
}
```

## Communication Protocols

### Inter-Worker Communication Patterns

#### 1. Point-to-Point Transfer

**Use Case**: Sequential layer execution

```
Protocol: Direct tensor transfer between adjacent layer workers

Sender:
1. Complete local computation
2. Serialize output tensor
3. Send to next layer worker(s)
4. Wait for acknowledgment

Receiver:
1. Receive tensor data
2. Deserialize into local format
3. Begin local computation
4. Send acknowledgment
```

**Message Format**:
```rust
struct TensorMessage {
    request_id: Uuid,
    source_layer: LayerId,
    dest_layer: LayerId,
    tensor_shape: Vec<usize>,
    tensor_data: Vec<u8>,  // Serialized tensor
    checksum: u64,
}
```

#### 2. Broadcast Pattern

**Use Case**: One-to-many communication (e.g., attention mechanisms)

```
Protocol: Efficient tensor broadcasting

Sender:
1. Complete computation
2. Build list of destination workers
3. Serialize tensor once
4. Parallel send to all destinations
5. Collect acknowledgments

Receivers:
1. Receive tensor independently
2. Deserialize locally
3. Acknowledge receipt
```

#### 3. Reduce Pattern

**Use Case**: Many-to-one aggregation (e.g., distributed softmax)

```
Protocol: Tree-based reduction

Workers organized in binary tree:
1. Leaf workers send partial results to parents
2. Parent workers:
   - Receive from both children
   - Aggregate (sum/max/etc.)
   - Send to their parent
3. Root worker completes final aggregation

Complexity: O(log n) communication rounds
```

#### 4. All-Reduce Pattern

**Use Case**: Synchronized gradient aggregation (future training support)

```
Protocol: Ring all-reduce

Workers arranged in logical ring:
1. Phase 1 - Scatter-Reduce:
   - Each worker sends chunk to next worker
   - Reduce received chunk with local chunk
   - Repeat N-1 times

2. Phase 2 - All-Gather:
   - Each worker sends reduced chunk to next
   - Repeat N-1 times

Result: All workers have complete reduced tensor
Bandwidth optimal: 2(N-1)/N ≈ 2 for large N
```

### Error Handling and Reliability

#### Timeout Protocol

```
Default timeouts:
- Tensor transfer: 30 seconds + size/bandwidth estimate
- Worker heartbeat: 5 seconds
- Computation: 60 seconds * estimated_flops

On timeout:
1. Log warning
2. Retry with exponential backoff (3 attempts)
3. If all retries fail, escalate to orchestrator
4. Orchestrator decides: repartition or fail request
```

#### Checksum Verification

```
For each tensor transfer:
1. Sender computes FNV-1a hash of serialized data
2. Hash included in message
3. Receiver computes hash on received data
4. Mismatch triggers retransmission
```

#### Connection Management

```
Worker-to-Worker:
- Persistent TCP connections with keep-alive
- Connection pool with max size
- Automatic reconnection on failure

Worker-to-Orchestrator:
- WebSocket for bidirectional communication
- Heartbeat every 5 seconds
- Orchestrator detects missing heartbeats
```

## Scheduling Strategies

### Request Scheduling

#### 1. First-Come-First-Served (FCFS)

**Algorithm**: Process requests in arrival order

```rust
struct FCFSScheduler {
    queue: VecDeque<InferenceRequest>,
}

impl FCFSScheduler {
    fn schedule(&mut self) -> Option<InferenceRequest> {
        self.queue.pop_front()
    }

    fn submit(&mut self, request: InferenceRequest) {
        self.queue.push_back(request);
    }
}
```

**Advantages**: Simple, fair, predictable latency
**Disadvantages**: Head-of-line blocking, no prioritization

#### 2. Priority Queue Scheduling (Planned)

**Algorithm**: Schedule high-priority requests first

```rust
struct PriorityScheduler {
    queue: BinaryHeap<PrioritizedRequest>,
}

struct PrioritizedRequest {
    request: InferenceRequest,
    priority: u32,
    deadline: Option<Instant>,
}

impl PriorityScheduler {
    fn schedule(&mut self) -> Option<InferenceRequest> {
        // Consider both priority and deadline
        self.queue.pop().map(|pr| pr.request)
    }
}
```

**Use Cases**: Multi-tenant systems, SLA guarantees

#### 3. Batch-Aware Scheduling (Planned)

**Algorithm**: Group compatible requests for efficiency

```rust
struct BatchScheduler {
    pending: HashMap<ModelId, Vec<InferenceRequest>>,
    batch_size: usize,
    max_wait_time: Duration,
}

impl BatchScheduler {
    fn schedule(&mut self) -> Option<Vec<InferenceRequest>> {
        for (model_id, requests) in &mut self.pending {
            if requests.len() >= self.batch_size
                || oldest_age(requests) > self.max_wait_time {
                return Some(drain_batch(requests, self.batch_size));
            }
        }
        None
    }
}
```

**Advantages**: Higher throughput, better GPU utilization
**Disadvantages**: Increased latency for small batches

### Worker Placement

#### Greedy Placement

```
For each partition p:
    candidates = workers with sufficient resources
    selected = argmax(score(worker, p)) for worker in candidates
    assign(p, selected)
    update_resources(selected, p)

score(worker, partition) =
    α * available_memory(worker) +
    β * compute_capacity(worker) +
    γ * (-network_latency(worker, partition_neighbors))
```

#### Constraint Programming (Future)

```
Variables: x[p,w] ∈ {0,1} (partition p assigned to worker w)

Objective: Minimize completion_time

Constraints:
    ∀p: Σw x[p,w] = 1                    # Each partition assigned once
    ∀w: Σp memory[p] * x[p,w] ≤ capacity[w]  # Memory constraints
    ∀w: Σp compute[p] * x[p,w] ≤ max_compute[w]  # Compute constraints

Solve using: CP-SAT, Gurobi, or custom branch-and-bound
```

## Performance Optimization Techniques

### 1. Tensor Compression

```
Options:
- Quantization: FP32 → FP16 or INT8
- Sparsification: Drop near-zero values
- Compression: zstd for large tensors

Decision Logic:
if tensor_size > 10MB:
    if sparsity > 0.8:
        use sparse_encoding
    else:
        use zstd_compression
else:
    send uncompressed
```

### 2. Overlapping Communication and Computation

```
Pipeline stages:
1. Receive input tensor (async)
2. While receiving:
   - Prepare model weights
   - Warm up GPU
3. As soon as receive completes:
   - Begin computation
4. During computation:
   - Pre-allocate output buffer
   - Prepare next layer workers
5. As soon as computation completes:
   - Begin transmission (async)
   - Start cleanup
```

### 3. Memory Management

```rust
struct TensorPool {
    // Pre-allocated buffers for common sizes
    buffers: HashMap<TensorShape, Vec<Buffer>>,

    fn acquire(&mut self, shape: TensorShape) -> Buffer {
        self.buffers.get_mut(&shape)
            .and_then(|v| v.pop())
            .unwrap_or_else(|| allocate_new(shape))
    }

    fn release(&mut self, buffer: Buffer) {
        let shape = buffer.shape();
        self.buffers.entry(shape)
            .or_default()
            .push(buffer);
    }
}
```

## Algorithm Complexity Summary

| Algorithm | Time Complexity | Space Complexity | Network Complexity |
|-----------|----------------|------------------|-------------------|
| Layer-wise partition | O(m) | O(m) | O(m) messages |
| Load-balanced partition | O(m log m) | O(m + n) | O(m) messages |
| Topology-aware partition | O(m + e) | O(m + e) | Minimized |
| Point-to-point transfer | O(1) | O(tensor_size) | 1 message |
| Broadcast | O(n) | O(tensor_size) | n messages |
| Reduce (tree) | O(log n) | O(tensor_size) | 2n-1 messages |
| Ring all-reduce | O(n) | O(tensor_size) | 2(n-1) messages |

Where:
- m = number of layers/operations
- n = number of workers
- e = number of edges in computation graph

## Future Algorithm Research Directions

### 1. Reinforcement Learning for Partitioning

Train an RL agent to make partitioning decisions:
- State: Model graph, resource availability, historical performance
- Action: Partition assignment
- Reward: Negative of inference latency

### 2. AutoML for Partition Search

Use neural architecture search techniques to find optimal partitions:
- Encode partitions as differentiable decisions
- Use gradient-based optimization
- Learn from multiple models and hardware configs

### 3. Dynamic Repartitioning

Adapt partitions during runtime:
- Monitor performance metrics
- Detect bottlenecks (stragglers, network congestion)
- Trigger repartitioning when improvement threshold met
- Perform graceful migration with minimal disruption

### 4. Multi-Objective Optimization

Balance multiple objectives:
- Minimize latency
- Minimize energy consumption
- Maximize throughput
- Respect cost constraints

Use Pareto optimization to explore trade-off frontier.

## References

### Academic Papers
1. "GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism" (Huang et al., 2019)
2. "PipeDream: Fast and Efficient Pipeline Parallel DNN Training" (Narayanan et al., 2019)
3. "Megatron-LM: Training Multi-Billion Parameter Language Models" (Shoeybi et al., 2019)
4. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020)

### Graph Partitioning
1. METIS: http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
2. "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs" (Karypis & Kumar, 1998)

### Communication Collectives
1. "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations" (Pjeivac-Grbovic et al., 2005)
2. NCCL (NVIDIA Collective Communications Library): https://developer.nvidia.com/nccl

## Implementation Notes

### Current Status
- Layer-wise partitioning: Implemented
- Point-to-point communication: Implemented
- FCFS scheduling: Implemented

### In Progress
- Load-balanced partitioning
- Timeout and retry logic
- Compression support

### Planned
- Topology-aware partitioning
- Pipeline parallelism
- Advanced scheduling strategies
- Dynamic repartitioning

See [Architecture Documentation](architecture.md) for how these algorithms integrate into the system.
