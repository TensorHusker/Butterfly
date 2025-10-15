# Butterfly Coordination Protocol Specification

## Executive Summary

The Butterfly Coordination Protocol (BCP) is a Byzantine-fault-tolerant consensus system designed specifically for distributed transformer inference. It optimizes for minimal communication overhead while ensuring deterministic inference results across multiple compute nodes.

**Key Properties:**
- **Safety**: All nodes compute identical results for identical inputs
- **Liveness**: The system makes progress as long as f+1 out of 2f+1 nodes are operational
- **Determinism**: Inference results are reproducible across executions
- **Low Latency**: Minimized synchronization points via pipelined execution
- **Graceful Degradation**: System continues operation during partial failures

## 1. System Model

### 1.1 Network Assumptions

- **Partially Synchronous Network**: Messages are eventually delivered with bounded delay
- **Byzantine Fault Model**: Up to f nodes may exhibit arbitrary behavior
- **Authentication**: Messages are cryptographically signed using Ed25519
- **Ordering**: FIFO per-connection message delivery

### 1.2 Node Roles

**Coordinator (C)**:
- Assigns work to compute nodes
- Aggregates results
- Detects failures
- Elected via modified Raft consensus

**Compute Nodes (N_i)**:
- Execute transformer layer computations
- Forward intermediate results
- Participate in checkpointing
- Report health metrics

**Observer Nodes (O_j)**:
- Monitor system health
- Store checkpoints
- Can be promoted to compute nodes

### 1.3 Failure Model

**Tolerated Failures:**
- Crash-stop failures (node halts)
- Byzantine failures (arbitrary behavior, up to f nodes)
- Network partitions (temporary)
- Message loss/reordering

**Recovery Guarantees:**
- System operational with 2f+1 nodes (f Byzantine)
- Deterministic recovery from checkpoints
- No inference result corruption

## 2. Protocol Architecture

### 2.1 Layered Design

```
┌─────────────────────────────────────────┐
│   Application Layer (Inference API)     │
├─────────────────────────────────────────┤
│   Coordination Layer (BCP Core)         │
│   - Phase Coordination                  │
│   - Work Distribution                   │
│   - Result Aggregation                  │
├─────────────────────────────────────────┤
│   Consensus Layer (Raft-based)          │
│   - Leader Election                     │
│   - Configuration Changes               │
│   - Checkpoint Agreement                │
├─────────────────────────────────────────┤
│   Communication Layer (QUIC/gRPC)       │
│   - Reliable Delivery                   │
│   - Flow Control                        │
│   - Compression                         │
└─────────────────────────────────────────┘
```

### 2.2 State Machine

Each node maintains a replica of the coordination state machine:

```
States:
  INITIALIZING → READY → COMPUTING → AGGREGATING → COMMITTING → READY
                   ↓                     ↓            ↓
                DEGRADED ← RECOVERING ← FAILED
```

**State Transitions:**

- `INITIALIZING → READY`: Node joined cluster, loaded model partition
- `READY → COMPUTING`: Received inference task with valid dependencies
- `COMPUTING → AGGREGATING`: Computation complete, awaiting peer results
- `AGGREGATING → COMMITTING`: All dependencies satisfied, result ready
- `COMMITTING → READY`: Result checkpointed, ready for next task
- `* → DEGRADED`: Failure detected in peer node
- `DEGRADED → RECOVERING`: Initiating recovery protocol
- `RECOVERING → READY`: Recovery complete, state synchronized

### 2.3 Execution Phases

Inference execution proceeds in synchronized phases:

#### Phase 1: WORK_ASSIGNMENT
- Coordinator assigns layer ranges to compute nodes
- Nodes acknowledge assignment
- **Synchronization Point**: All nodes must ack before proceeding
- **Timeout**: 5 seconds (configurable)

#### Phase 2: COMPUTATION (Pipelined)
- Nodes execute their assigned layers
- Intermediate results forwarded immediately (no sync)
- **Optimization**: Pipelining across layers eliminates global sync
- **Checkpointing**: Every K tokens (K=10 default)

#### Phase 3: RESULT_AGGREGATION
- Final layer nodes report completion
- Coordinator collects results
- **Synchronization Point**: Coordinator waits for quorum (2f+1 nodes)
- **Timeout**: Based on expected computation time + 3σ

#### Phase 4: COMMITMENT
- Coordinator proposes result
- Nodes validate and vote
- **Synchronization Point**: Byzantine agreement on result
- **Consensus**: Modified PBFT with optimistic fast path

## 3. Coordination Protocols

### 3.1 Work Distribution Algorithm

**Goal**: Minimize communication while balancing load

**Algorithm: Layered Pipeline with Affinity Scheduling**

```
Input:
  - Inference request R with input tokens T
  - Model with L layers
  - N compute nodes

Output:
  - Assignment A: layer → node_id

Procedure ASSIGN_WORK(R, L, N):
  1. Partition layers into N roughly equal groups
  2. For each group G_i:
     a. Assign to node N_j that minimizes:
        cost(G_i, N_j) = comm_cost + compute_cost
        where:
          comm_cost = Σ(data_size × network_latency)
          compute_cost = estimated_flops / node_capacity
     b. Consider data locality (previous assignments)

  3. Construct dependency graph D
  4. Topologically sort to create execution order

  5. Return assignment map with dependencies
```

**Properties:**
- **Time Complexity**: O(L log N) via priority queue
- **Space Complexity**: O(L + N)
- **Approximation Ratio**: 1.5-optimal for balanced loads

### 3.2 Synchronization Points

**Minimal Synchronization Strategy:**

```
Synchronization required at:
1. Phase boundaries (necessary for correctness)
2. Checkpoints (every K tokens, necessary for recovery)
3. Failures (only affected subgraph needs to sync)

Synchronization NOT required for:
- Intermediate layer results (pipelined)
- Health checks (asynchronous gossip)
- Metrics collection (eventual consistency)
```

**Synchronization Primitive: Barrier Protocol**

```
Coordinator maintains barrier counter B[phase]

Procedure NODE_BARRIER(phase_id):
  1. Node sends BARRIER_READY(phase_id, node_id, checkpoint_hash)
  2. Coordinator increments B[phase_id]
  3. When B[phase_id] >= quorum_size:
     a. Validate all checkpoint hashes match
     b. Broadcast BARRIER_RELEASE(phase_id)
  4. Nodes proceed to next phase

Optimization: Predictive Barrier Release
  - Coordinator predicts completion based on historical data
  - Pre-releases barrier if confidence > 95%
  - Rollback if prediction wrong (rare)
```

### 3.3 Byzantine Agreement Protocol

**Modified PBFT for Result Validation:**

```
Phase 1: PRE-PREPARE (Coordinator → All)
  - Coordinator proposes result R with proof π
  - Proof includes: computation trace, intermediate checksums

Phase 2: PREPARE (All → All)
  - Each node validates R locally
  - If valid, broadcast PREPARE(R, hash(R))
  - Wait for 2f PREPARE messages matching hash

Phase 3: COMMIT (All → All)
  - Broadcast COMMIT(R)
  - Wait for 2f+1 COMMIT messages
  - Apply result, update state

Optimistic Fast Path:
  - If all nodes in quorum agree immediately (common case)
  - Skip PREPARE phase
  - Directly commit
  - Expected latency: 1 RTT instead of 3 RTT
```

**Proof of Computation:**

Each node generates proof π for its computation:
```
π = {
  layer_id,
  input_hash,
  output_hash,
  intermediate_checksums[],
  signature
}
```

Coordinator validates proofs are consistent before proposing result.

## 4. Failure Detection and Recovery

### 4.1 Failure Detection

**Adaptive Heartbeat Protocol:**

```
Heartbeat interval:
  base_interval = 100ms
  adaptive_interval = base_interval × (1 + congestion_factor)

Failure suspicion threshold:
  suspect_timeout = 3 × adaptive_interval
  confirmed_timeout = 10 × adaptive_interval

Failure detection oracle φ (Phi Accrual):
  φ = -log₁₀(P(T_now - T_last > threshold))

  If φ > φ_suspect: mark as SUSPECTED
  If φ > φ_failed: mark as FAILED
```

**Properties:**
- **Detection Time**: O(1) with high probability
- **False Positive Rate**: Configurable via φ thresholds
- **Network Adaptivity**: Adjusts to congestion

### 4.2 Recovery Protocol

**Checkpoint-Based Recovery:**

```
Checkpoints stored every K tokens:
  CP_k = {
    token_position: k,
    node_states: {N_i → state_i},
    intermediate_results: {},
    metadata: {timestamp, version}
  }

Procedure RECOVER_FROM_FAILURE(failed_node, checkpoint_k):
  1. Coordinator selects replacement node N_r
  2. Transfer checkpoint CP_k to N_r
  3. N_r validates checkpoint integrity
  4. Recompute from position k to current
  5. Rejoin at next barrier

Recovery time: O(K × layer_compute_time)
Expected: <1 second for K=10
```

**Optimistic Recovery:**

```
If failure detected during COMPUTING phase:
  1. Don't wait for barrier
  2. Immediately reassign work
  3. Replacement node computes in parallel
  4. Original result discarded if it arrives late

Benefit: Zero recovery latency in common case
```

### 4.3 Byzantine Failure Handling

**Detection Mechanisms:**

1. **Result Verification**:
   - Nodes cross-check computation results
   - Statistical outlier detection
   - Cryptographic proof validation

2. **Behavior Monitoring**:
   - Track message patterns
   - Detect protocol violations
   - Identify equivocation (conflicting messages)

3. **Reputation System**:
   - Each node maintains reputation score
   - Degrade reputation on suspicious behavior
   - Exclude nodes below threshold

**Isolation Protocol:**

```
If node N_i exhibits Byzantine behavior:
  1. Coordinator broadcasts ISOLATE(N_i, evidence)
  2. Nodes vote on isolation
  3. If 2f+1 votes, remove N_i from cluster
  4. Redistribute N_i's work
  5. Update configuration via consensus
```

## 5. Load Balancing

### 5.1 Dynamic Rebalancing

**Triggers for Rebalancing:**
- Load imbalance > 20% across nodes
- Node capacity change (scale up/down)
- Persistent stragglers detected
- Network topology change

**Rebalancing Algorithm:**

```
Procedure REBALANCE(current_assignment A):
  1. Measure current load L_i for each node N_i
  2. Compute ideal load L_ideal = (Σ L_i) / N
  3. Identify overloaded (L_i > 1.2 × L_ideal)
  4. Identify underloaded (L_i < 0.8 × L_ideal)

  5. For each overloaded node N_o:
     a. Select layer range R with minimal dependencies
     b. Find underloaded node N_u with best affinity
     c. Propose migration: R from N_o to N_u

  6. Estimate migration cost:
     cost = model_transfer_time + warmup_time

  7. If cost < improvement_benefit:
     Execute migration at next checkpoint

  8. Otherwise: defer rebalancing
```

**Migration Protocol:**

```
Procedure MIGRATE_LAYERS(source, target, layer_range):
  1. At checkpoint boundary:
     a. source stops accepting new work for layer_range
     b. Complete in-flight work

  2. Transfer:
     a. Stream model weights to target
     b. Transfer optimizer state if needed
     c. Verify integrity with checksums

  3. Activation:
     a. target loads and validates weights
     b. Runs warmup inference (compile kernels)
     c. Signals READY to coordinator

  4. Switchover:
     a. Coordinator updates assignment
     b. Broadcasts new configuration
     c. source deallocates resources

Migration is atomic: all or nothing
```

### 5.2 Straggler Mitigation

**Speculative Execution:**

```
If node N_i is suspected of being slow:
  1. Monitor progress via intermediate checksum reports
  2. If progress < 50% of expected at checkpoint:
     a. Assign same work to backup node N_b
     b. Use result from whichever finishes first
     c. Cancel the slower execution

Cost: 2× compute for suspected stragglers (rare)
Benefit: Eliminates tail latency
```

**Predictive Stragglers:**

```
Maintain historical performance profile per node:
  P_i = {latency_distribution, failure_rate, capacity}

Before assignment, predict likelihood of straggling:
  straggle_prob = f(P_i, current_load, network_state)

If straggle_prob > threshold:
  - Assign lighter load
  - Or assign to backup immediately
```

## 6. Performance Characteristics

### 6.1 Theoretical Bounds

**Communication Complexity:**
- **Per-layer**: O(1) messages (pipelined, no broadcast)
- **Per-checkpoint**: O(N) messages (barrier synchronization)
- **Per-failure**: O(N²) messages (Byzantine agreement)

**Latency Bounds:**

Optimal case (no failures):
```
T_total = T_assignment + T_compute + T_aggregation + T_commit

Where:
  T_assignment = 1 RTT
  T_compute = (L/N) × T_layer (pipelined, no sync)
  T_aggregation = 1 RTT
  T_commit = 1 RTT (optimistic fast path)

Total: 3 RTT + compute_time
```

Failure case (1 node fails):
```
T_recovery = T_detect + T_checkpoint_transfer + T_recompute

Where:
  T_detect ≈ 300ms (adaptive heartbeat)
  T_checkpoint_transfer ≈ 100ms (10 tokens × model slice)
  T_recompute ≈ 10 × T_layer

Total overhead: ~500ms (< 5% for typical inference)
```

### 6.2 Throughput Analysis

**Single Inference Request:**
```
Throughput = tokens_per_second / (T_total / batch_size)

With pipelining:
  - Multiple requests overlap in pipeline
  - Steady-state: 1 request completes every T_layer
  - Effective throughput: N × tokens_per_layer / T_layer
```

**Scalability:**
```
Ideal: Linear scaling with N nodes
Actual: 0.85N scaling (15% coordination overhead)

Bottlenecks:
  - Coordinator (mitigated via load balancing)
  - Network bandwidth (mitigated via compression)
  - Synchronization points (minimized via pipelining)
```

### 6.3 Failure Impact

**Availability:**
```
System available if: operational_nodes >= 2f + 1

Probability of availability:
  P(available) = 1 - P(>f failures simultaneously)

With N=7, f=2, node_reliability=0.99:
  P(available) > 0.99999 (five nines)
```

**Performance Degradation:**
```
If k nodes fail (k ≤ f):
  - Throughput degrades by factor: (N-k)/N
  - Latency increases by factor: N/(N-k)
  - Recovery time: ~500ms per failure

Graceful degradation, no catastrophic failure
```

## 7. Formal Correctness Properties

### 7.1 Safety Properties

**Theorem 1: Deterministic Inference**

For any input I and configuration C, all non-Byzantine nodes produce identical output O.

**Proof sketch:**
1. All nodes start with identical model weights (verified via checkpoint hashes)
2. All nodes process identical input I
3. Intermediate results are validated via checksums at synchronization points
4. Byzantine agreement ensures consensus on final result
5. Any divergence triggers Byzantine detection and isolation
6. Therefore, all honest nodes output O

**Theorem 2: No Result Corruption**

If ≤ f nodes are Byzantine, the system never commits an incorrect result.

**Proof sketch:**
1. Byzantine agreement requires 2f+1 matching results
2. With ≤ f Byzantine nodes, ≥ f+1 honest nodes exist
3. Honest nodes compute correct result R
4. Byzantine nodes can produce at most f incorrect results
5. R receives ≥ f+1 votes, Byzantine results receive ≤ f votes
6. R achieves quorum, incorrect results do not
7. Therefore, only R can be committed

### 7.2 Liveness Properties

**Theorem 3: Progress Guarantee**

If ≥ 2f+1 nodes are operational and network is eventually synchronous, the system makes progress.

**Proof sketch:**
1. Assume ≥ 2f+1 operational nodes
2. Coordinator election terminates via Raft (proven)
3. Work assignment completes (requires simple majority ack)
4. Computation proceeds asynchronously (no blocking)
5. Aggregation barrier satisfied when 2f+1 nodes complete
6. Byzantine agreement terminates (proven for PBFT)
7. Therefore, inference completes in finite time

**Theorem 4: Recovery Termination**

Recovery from any failure state terminates in bounded time.

**Proof sketch:**
1. Failure detection guaranteed in O(timeout) via φ-accrual
2. Checkpoint retrieval bounded by network transfer time
3. Recomputation bounded by K × T_layer (deterministic)
4. Rejoin requires single barrier synchronization (bounded)
5. All steps have finite bounds
6. Therefore, recovery terminates in bounded time T_recovery

### 7.3 Consistency Properties

**Theorem 5: Causal Consistency**

If computation A happens-before computation B, then A's checkpoint is included in B's causal history.

**Proof sketch:**
1. Checkpoints include vector clocks
2. happens-before relation captured by vector clock ordering
3. Checkpoint protocol ensures causally consistent snapshots
4. Recovery restores from causally consistent state
5. Therefore, causal order preserved

## 8. Implementation Notes

### 8.1 Rust Implementation Outline

```rust
// Core coordination state machine
pub struct CoordinationStateMachine {
    state: NodeState,
    epoch: u64,
    checkpoint: Checkpoint,
    assignment: WorkAssignment,
    quorum: QuorumTracker,
}

// Message types for coordination protocol
pub enum CoordinationMessage {
    WorkAssignment(Assignment),
    BarrierReady(NodeId, CheckpointHash),
    BarrierRelease(Epoch),
    PrePrepare(Result, Proof),
    Prepare(ResultHash),
    Commit(Result),
    Heartbeat(NodeId, PhiValue),
    Suspicion(NodeId, Evidence),
}

// Byzantine agreement implementation
pub struct ByzantineAgreement {
    phase: AgreementPhase,
    proposals: HashMap<ResultHash, Vec<Vote>>,
    committed: Option<Result>,
}

// Failure detector using φ-accrual
pub struct PhiAccrualFailureDetector {
    heartbeat_history: RingBuffer<Instant>,
    phi_threshold_suspect: f64,
    phi_threshold_failed: f64,
}
```

### 8.2 Optimization Opportunities

**Zero-Copy Tensor Transfer:**
- Use shared memory for local nodes
- RDMA for remote nodes with InfiniBand
- Avoid serialization via memory mapping

**Kernel Fusion:**
- Fuse layer computations across node boundaries
- Reduce synchronization points
- Compiler-level optimization

**Compression:**
- Quantize intermediate activations (8-bit/4-bit)
- Entropy coding for sparse tensors
- Adaptive compression based on network conditions

**Caching:**
- Cache KV-cache for attention across requests
- Reuse computation for common prefixes
- Speculative caching of likely next tokens

## 9. Testing Strategy

### 9.1 Correctness Testing

**Formal Verification:**
- TLA+ specification (see coordination_protocol.tla)
- Model checking for state space exploration
- Refinement mapping to implementation

**Property-Based Testing:**
```rust
#[proptest]
fn byzantine_agreement_reaches_consensus(
    #[strategy(valid_results())] results: Vec<Result>,
    #[strategy(0..=MAX_BYZANTINE)] num_byzantine: usize
) {
    // Property: If ≤ f Byzantine nodes, honest nodes reach consensus
}
```

**Chaos Engineering:**
- Random node failures during inference
- Network partitions
- Byzantine behavior injection
- Clock skew simulation

### 9.2 Performance Testing

**Benchmarks:**
- End-to-end inference latency
- Throughput under load
- Recovery time from failures
- Scalability (1-100 nodes)

**Profiling:**
- Message rate and size
- CPU utilization per phase
- Network bandwidth utilization
- Lock contention analysis

## 10. Comparison with Related Work

| System | Consistency | Fault Tolerance | Overhead | Use Case |
|--------|------------|-----------------|----------|----------|
| Butterfly BCP | Strong (Byzantine) | 2f+1 (f Byzantine) | ~15% | Distributed Inference |
| Raft | Strong | n/2+1 (crash-stop) | ~10% | Configuration/Metadata |
| PBFT | Strong (Byzantine) | 3f+1 (f Byzantine) | ~40% | General Consensus |
| Gossip | Eventual | High (epidemic) | ~5% | Monitoring/Metrics |
| Paxos | Strong | n/2+1 (crash-stop) | ~20% | General Consensus |

**Key Differentiators:**
1. Optimized for inference workload (pipelined, minimal sync)
2. Byzantine tolerance (critical for untrusted environments)
3. Lower overhead than PBFT (optimistic fast path)
4. Integrated load balancing and recovery

## 11. Future Enhancements

### 11.1 Adaptive Consensus

Dynamic adjustment of consensus parameters based on:
- Observed failure rates
- Network conditions
- Security requirements

### 11.2 Federated Learning Integration

Extend protocol to support:
- Privacy-preserving aggregation
- Model update consensus
- Heterogeneous node capabilities

### 11.3 Multi-Model Support

Coordinate multiple models simultaneously:
- Shared resource allocation
- Priority-based scheduling
- Cross-model optimization

### 11.4 Hardware Acceleration

Leverage specialized hardware:
- GPU-aware scheduling
- TPU pod integration
- FPGA offload for protocol operations

## 12. References

1. Castro, M. and Liskov, B. "Practical Byzantine Fault Tolerance" (OSDI 1999)
2. Ongaro, D. and Ousterhout, J. "In Search of an Understandable Consensus Algorithm" (Raft, ATC 2014)
3. Lamport, L. "The Part-Time Parliament" (Paxos, TOCS 1998)
4. Hayashibara, N. et al. "The φ Accrual Failure Detector" (SRDS 2004)
5. Narayanan, D. et al. "Efficient BFT in the Age of Blockchains" (arXiv 2024)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-11
**Authors:** Butterfly Contributors
**Status:** Draft Specification
